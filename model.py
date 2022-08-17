from transformers import AutoModelForSeq2SeqLM
from decoder import CustomBartDecoder

import torch, copy

import torch.nn.functional as F

##################################################

class TestModel(torch.nn.Module):
	
	def __init__(self, gramm = None, vocab = None, point_size = 256):
		super(TestModel, self).__init__()
		
        #----------------------------------------#
		# Auxiliary Data

		# Grammar Mask
		self.gramm = gramm
		self.smap, self.mask = gramm.build_mask()
		self.mask = torch.tensor(self.mask)

		self.vocab = vocab

		# Head-Dimensions
		self.gramm_size = gramm.size() + 1
		self.vocab_size = len(vocab)
		self.point_size = point_size # TODO: != input length

		# Output-Dimension
		self.output_dim = self.gramm_size + self.vocab_size + 2 * self.point_size

        #----------------------------------------#
		# Model Setup

		self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

		# Custom Decoder
		encoder_config = self.model.model.encoder.config
		decoder_config = copy.deepcopy(encoder_config)
		decoder_config.vocab_size = self.output_dim # TODO: old vocab_size?
		
		del self.model.model.decoder

		self.model.model.decoder = CustomBartDecoder(decoder_config)

        #----------------------------------------#
		# Output Layers

		hidden_dim = self.model.config.d_model
		
		# Grammar-Head
		self.gramm_head = torch.nn.Linear(
			hidden_dim,
			self.gramm_size,
		)

		# Vocabulary-Head
		self.relat_head = torch.nn.Linear(
			hidden_dim,
			self.vocab_size,
		)

		# Pointer-Head
		self.point_head = torch.nn.Linear(
			hidden_dim + self.point_size,
			2 * self.point_size
		)

		#--------------------------------------#

	##################################################

	def get_dimensions(self):
		return {
			"gramm_size": self.gramm_size,
			"vocab_size": self.vocab_size,
			"point_size": self.point_size
		}
		
	##################################################

	def compute_loss(self, logits, labels):
		loss = None
		
		bound_a = self.gramm_size
		bound_b = self.gramm_size + self.vocab_size
		bound_c = self.point_size
		
		#--------------------------------------#

		if labels != None:
			loss_func = torch.nn.CrossEntropyLoss()
			
			g_labels = labels[:,:,0].view(-1).long()
			r_labels = labels[:,:,1].view(-1).long()
			p_labels = labels[:,:,2].view(-1).long()
			q_labels = labels[:,:,3].view(-1).long()
			
			g_logits = logits[:,:,         : bound_a].reshape(-1, self.gramm_size)
			r_logits = logits[:,:, bound_a : bound_b].reshape(-1, self.vocab_size)
			p_logits = logits[:,:, bound_b :-bound_c].reshape(-1, self.point_size)
			q_logits = logits[:,:,-bound_c :        ].reshape(-1, self.point_size)
			
			g_loss = loss_func(g_logits, g_labels)
			r_loss = loss_func(r_logits, r_labels)
			p_loss = loss_func(p_logits, p_labels)
			q_loss = loss_func(q_logits, q_labels)
			
			lam = 0.20
			inv = (1 - lam)/2

			loss = g_loss * inv + r_loss * inv + (p_loss + q_loss) * lam
			
		#--------------------------------------#

		return loss

	##################################################

	def forward(self, input_ids = None, attention_mask = None, encoder_outputs = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):

		#--------------------------------------#
		# 1. MODEL FORWARD

		decoder_outputs = self.model(
			input_ids = input_ids,
			attention_mask = attention_mask,
			decoder_input_ids = decoder_input_ids.float(),
			decoder_attention_mask = decoder_attention_mask,
			encoder_outputs = encoder_outputs,
			output_hidden_states = True,
			output_attentions = True,
			return_dict = True
			#labels = labels
		)

		#--------------------------------------#
		# 2. PREPARE INPUTS

		last_cross_attentions = decoder_outputs.cross_attentions[-1]
		decoder_hidden_states = decoder_outputs.decoder_hidden_states[-1]
		last_cross_attentions = last_cross_attentions.sum(dim = 1)
		
		# zero-padding 'last_cross_attentions'
		shape = last_cross_attentions.shape
		shape = shape[:2] + (self.point_size - shape[-1] ,)
		zeros = torch.zeros(shape, device=input_ids.device)
		
		# zero-padding 'last_cross_attentions'
		last_cross_attentions = torch.cat([last_cross_attentions, zeros], dim=-1)
		last_hidden_attention = torch.cat([decoder_hidden_states, last_cross_attentions], dim=-1)

		#--------------------------------------#
		# 3. FUNCTION HEADS: GRAMM / RELAT / POINT
		
		gramm_logits = self.gramm_head(decoder_hidden_states) # TODO: swap with logits?
		relat_logits = self.relat_head(decoder_hidden_states) # TODO: swap with logits?
		point_logits = self.point_head(last_hidden_attention)
		
		#--------------------------------------#
		# 4. FINAL OUTPUT
		
		logits = torch.cat([gramm_logits, relat_logits, point_logits], dim=-1)
		loss = self.compute_loss(logits, labels)
		
		#--------------------------------------#
		
		return {"logits": logits, "loss": loss}
	
	##################################################

	def generate(
		self,
		input_ids,
		max_length = 256,
		use_cache = True,
		**gen_kwargs
	):

		#----------------------------------------#
		# VARIABLES

		bos_token_id = self.tokenizer.bos_token_id
		eos_token_id = self.tokenizer.eos_token_id
		pad_token_id = self.tokenizer.pad_token_id

		batch_size = input_ids.shape[0]

		#----------------------------------------#
		# ENCODER: ONCE

		attention_mask = input_ids.ne(pad_token_id).long()
		encoder_output = self.model.model.encoder(
			input_ids,
			attention_mask = attention_mask,
			return_dict = True
		)
		gen_kwargs["encoder_outputs"] = encoder_output

		#----------------------------------------#
		# DECODER INPUT

		# TODO:
		decoder_input_ids = torch.full(
			(batch_size, 1),
			bos_token_id,
			dtype = torch.long,
			device = input_ids.device,
		)

		#----------------------------------------#
		# CURRENT "STATE"

		unfinished = input_ids.new(batch_size).fill_(1)
		
		#----------------------------------------#

		length = 1
		past = None

		while length < max_length:

			#------------------------------------#
			# PREPARE

			# TODO:
			model_inputs = self.model.prepare_inputs_for_generation(
				decoder_input_ids,
				past = past,
				attention_mask = attention_mask,
				use_cache = use_cache,
				**gen_kwargs
			)

			# FORWARD
			decoder_outputs = self.model(**model_inputs)

			#------------------------------------#
			# NEXT TOKENS
			
			next_logits = decoder_outputs[0][:, -1, :]
			next_tokens = torch.argmax(next_logits, dim=-1)
			next_tokens = next_tokens * unfinished + (pad_token_id) * (1 - unfinished)
			
			#------------------------------------#			
			# INPUT LENGTH + 1

			decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
			length = length + 1
			
			#------------------------------------#
			# SEQUENCES DONE?

			done = next_tokens == eos_token_id
			unfinished = unfinished.mul((~done).long())
			
			#------------------------------------#
			# MODEL PAST?
			
			if "past_key_values" in decoder_outputs:
				past = decoder_outputs.past_key_values

			# IF ALL_DONE: BREAK
			if unfinished.max() == 0:
				break
			
			#------------------------------------#

		return decoder_input_ids

	def generate_2(self, input_ids = None, max_length = 64, **kwargs):
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		##################################################
		
		# ENCODER: ONLY ONCE
		attention_mask = input_ids.ne(0).long()
		
		encoder_outputs = self.model.model.encoder(
			input_ids = input_ids,
			attention_mask = attention_mask,
			output_hidden_states = True,
			return_dict = True
		)
		
		##################################################
			
		batch_size = input_ids.shape[0]                 # NUMBER OF SENTENCES
		input_size = input_ids.shape[1]                 # LENGTH OF SENTENCES
		
		undone = torch.ones(batch_size, device=device)  # GENERATED SEQUENCES (FLAG)
		
		strings = [ "[BOS] #_RELS_#" ] * batch_size
		self.mask = self.mask.to(device)
		
		decoder_input_ids = torch.zeros(
			batch_size, 1, self.output_dim, device=device
		).float()
		
		decoder_input_ids[:, :, 1] = 1
		decoder_attention_mask = torch.ones(batch_size, 1, device=device)
		
		for length in range(max_length - 1):
			
			if sum(undone) == 0:
				break
				
			##################################################
			
			decoder_outputs = self.forward(
				input_ids = input_ids,
				attention_mask = attention_mask,
				encoder_outputs = encoder_outputs,
				decoder_input_ids = decoder_input_ids,
				#decoder_attention_mask = decoder_attention_mask
			)
			
			##################################################
			
			g_logits = decoder_outputs["logits"][:, -1,                   :   self.gramm_size].detach()
			r_logits = decoder_outputs["logits"][:, -1,   self.gramm_size :-2*self.point_size].detach()
			p_logits = decoder_outputs["logits"][:, -1,-2*self.point_size :                  ].detach()
			
			#------------------------------------------------#
			
			# TODO: REWORK
			for idx, string in enumerate(strings):
				if undone[idx]: 
					l_bound = string.find("#_")
					r_bound = string.find("_#") + 2

					if l_bound == -1: # DONE
						continue

					state = string[l_bound : r_bound] 
					
					g_logits[idx][1:] = g_logits[idx][1:] + self.mask[self.smap[state]]
					g_logits[idx][0 ] = -float("inf")

					production = self.gramm.rule(
						torch.argmax(g_logits[idx][1:])
					)
					
					strings[idx] = strings[idx].replace(state, production[1], 1)
					
			#------------------------------------------------#
			
			g_values = torch.argmax(g_logits, dim =-1)
			r_values = torch.argmax(r_logits, dim =-1)
			p_values = torch.stack(
				[
					torch.argmax(p_logits[:, : self.point_size ], dim=-1),
					torch.argmax(p_logits[:, self.point_size : ], dim=-1) + self.point_size
				],
				dim=-1
			)

			g_logits = torch.zeros(size=g_logits.shape, out=g_logits)
			r_logits = torch.zeros(size=r_logits.shape, out=r_logits)
			p_logits = torch.zeros(size=p_logits.shape, out=p_logits)

			##################################################

			for i, (gv, rv, pv) in enumerate(zip(g_values, r_values, p_values)):
				if undone[i]:
					g_logits[i, gv] = 1     # RULE?
					undone[i] = ("[EOS]" not in strings[i]) # gv != 2

					if "TOKEN" in strings[i]: # gv in [3, 4]: 
						r_logits[i, rv] = 1 # RELATION?
						strings[i] = strings[i].replace("TOKEN", "token", 1)

					if "POINT" in strings[i]: # gv in [5]:
						p_logits[i, pv] = 1 # ENTITY?
						strings[i] = strings[i].replace("POINT", "point", 1)
				else:
					g_logits[i, 0] = 1

			##################################################

			next_tokens = torch.cat([g_logits, r_logits, p_logits], dim = 1)
			decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=1)
			#decoder_attention_mask = torch.cat([decoder_attention_mask, undone[:, None]], dim=1)
		
		##################################################
		
		return decoder_input_ids

	##################################################