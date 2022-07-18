from transformers import BartForConditionalGeneration
from decoder import CustomBartDecoder

import torch, copy

import torch.nn.functional as F

##################################################

class TestModel(torch.nn.Module):
	
	def __init__(self, gramm = None, vocab = None, point_size = 256):
		super(TestModel, self).__init__()
		
		# Building Mask from Grammar
		self.gramm = gramm
		self.smap, self.mask = gramm.build_mask()
		self.mask = torch.tensor(self.mask)
		
		# Output Size
		self.gramm_size = gramm.size() + 1
		self.relat_size = len(vocab)
		self.point_size = point_size
		self.vocab_size = 1 * (self.gramm_size + self.relat_size + 2 * self.point_size)
		
		# Loading Model
		self.b_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
		
		# Configuring Encoder
		self.encoder = self.b_model.get_encoder()
		self.encoder.gradient_checkpointing = True # CHECK?
		
		# Freezing Encoder-Layers
		frozen = 0
		for layer in self.encoder.layers[ : frozen]:
			for param in layer.parameters():
				param.requires_grad = False
		
		# Configuring Decoder
		decoder_config = copy.deepcopy(self.encoder.config)
		decoder_config.vocab_size = self.vocab_size
		decoder_config.gradient_checkpointing = True
		self.decoder = CustomBartDecoder( decoder_config )
		
		# Hidden-State Dimension
		self.d_model = self.decoder.config.d_model
		
		# FUNCTION HEADS
		self.gramm_head = torch.nn.Linear(                       self.d_model, self.gramm_size)
		self.relat_head = torch.nn.Linear(                       self.d_model, self.relat_size)
		self.point_lead = torch.nn.Linear( self.d_model + 1 * self.point_size, self.point_size)
		self.point_read = torch.nn.Linear( self.d_model + 2 * self.point_size, self.point_size)

	##################################################
	
	def compute_loss(self, logits, labels):
		loss = None
		
		bound_a = self.gramm_size
		bound_b = self.gramm_size + self.relat_size
		bound_c = self.point_size
		
		if labels != None:
			loss_func = torch.nn.CrossEntropyLoss()

			shifted_logits = logits[:, :-1].contiguous()
			shifted_labels = labels[:,1:  ].contiguous()
			
			g_labels = shifted_labels[:,:,0].view(-1).long()
			r_labels = shifted_labels[:,:,1].view(-1).long()
			p_labels = shifted_labels[:,:,2].view(-1).long()
			q_labels = shifted_labels[:,:,3].view(-1).long()
			
			g_logits = shifted_logits[:,:,         : bound_a].reshape(-1, self.gramm_size)
			r_logits = shifted_logits[:,:, bound_a : bound_b].reshape(-1, self.relat_size)
			p_logits = shifted_logits[:,:, bound_b :-bound_c].reshape(-1, self.point_size)
			q_logits = shifted_logits[:,:,-bound_c :        ].reshape(-1, self.point_size)
			
			g_loss = loss_func(g_logits, g_labels) * 2/6
			r_loss = loss_func(r_logits, r_labels) * 2/6
			p_loss = loss_func(p_logits, p_labels) * 1/6
			q_loss = loss_func(q_logits, q_labels) * 1/6
			
			loss = g_loss + r_loss + p_loss + q_loss
			
		return loss
		
	##################################################
	
	def forward(self, input_ids = None, attention_mask = None, encoder_outputs = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):
		
		# 1. ENCODER
		if encoder_outputs is None:
			encoder_outputs = self.encoder(
				input_ids = input_ids,
				attention_mask = attention_mask,
				output_hidden_states = True,
				return_dict = True,
			)
		
		encoder_hidden_states = encoder_outputs[0]
		
		# 2. DECODER
		decoder_outputs = self.decoder(
			input_ids = decoder_input_ids,
			attention_mask = decoder_attention_mask,
			encoder_hidden_states = encoder_hidden_states,
			encoder_attention_mask = attention_mask,
			output_hidden_states = True,
			output_attentions = True,
			return_dict = True
		)
		
		# X. batch_size x num_attention_heads x decoder_sequence_length x encoder_seqence_length
		last_cross_attentions = decoder_outputs.cross_attentions[-1]
		decoder_hidden_states = decoder_outputs.hidden_states[-1]
		last_cross_attentions = last_cross_attentions.sum(dim = 1)
		
		# 3. FUNCTION HEAD: GRAMM / RELAT
		gramm_logits = self.gramm_head(decoder_hidden_states)
		relat_logits = self.relat_head(decoder_hidden_states)
		
		# 4. POINTER-NETWORK: LEFT
		point_values_l = torch.cat([last_cross_attentions, decoder_hidden_states], dim = -1)
		point_logits_l = self.point_lead(point_values_l)
		
		# 5. POINTER-NETWORK: RIGHT
		point_values_r = torch.cat([last_cross_attentions, decoder_hidden_states, point_logits_l], dim = -1)
		point_logits_r = self.point_read(point_values_r)
		
		# 6. PREDICTION: LOGITS / LOSS
		logits = torch.cat([gramm_logits, relat_logits, point_logits_l, point_logits_r], dim = -1)
		loss = self.compute_loss(logits, labels)
		
		return {
			"loss": loss,
			"logits": logits
		}
	
	##################################################
	
	def generate(self, input_ids = None, max_length = 64, **kwargs):
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		##################################################
		
		# ENCODER: ONLY ONCE
		attention_mask = input_ids.ne(0).long()
			
		encoder_outputs = self.encoder(
			input_ids = input_ids,
			attention_mask = attention_mask,
			output_hidden_states = True,
			return_dict = True
		)
		
		##################################################
			
		batch_size = input_ids.shape[0]                 # NUMBER OF SENTENCES
		input_size = input_ids.shape[1]                 # LENGTH OF SENTENCES
		
		undone = torch.ones(batch_size, device=device)  # GENERATED SEQUENCES (FLAG)
		
		masking = True
		strings = [ "[BOS] #_RELS_#" ] * batch_size
		self.mask = self.mask.to(device)
		
		decoder_input_ids = torch.zeros(
			batch_size, 1, self.vocab_size, device=device
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
				decoder_attention_mask = decoder_attention_mask
			)
			
			##################################################
			
			g_logits = decoder_outputs["logits"][:, -1,                   :   self.gramm_size].detach()
			r_logits = decoder_outputs["logits"][:, -1,   self.gramm_size :-2*self.point_size].detach()
			p_logits = decoder_outputs["logits"][:, -1,-2*self.point_size :                  ].detach()
			
			#------------------------------------------------#
			
			if masking:
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
			decoder_attention_mask = torch.cat([decoder_attention_mask, undone[:, None]], dim=1)
		
		##################################################
		
		return decoder_input_ids

	##################################################

	def beam_search(self, input_ids = None, num_beams = 1, max_length = 64, **model_kwargs,):

		######################################################

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		######################################################

		# INITIALIZE VALUES
		batch_size = input_ids.shape[0] # ...
		input_size = input_ids.shape[1] # SEQUENCE LENGTH

		# "NEW" BATCH_SIZE
		batch_beam_size = batch_size * num_beams
		
		# SYNCHRONIZED G_LOGITS --> "BATCH_SIZE" 
		undone = torch.ones(batch_beam_size, device=device)  # GENERATED SEQUENCES (FLAG)
		strings = [ "[BOS] #_RELS_#" ] * batch_beam_size # batch_size <--> batch_beam_size
		counter = torch.tensor([ 1 ] * batch_beam_size, device = device)
		
		# MASK FOR G_LOGITS
		self.mask = self.mask.to(device)

		# TABLE FOR BEAM-SCORES
		beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
		beam_scores[:, 1:] = -1e9 # MASKING, 1ST ITERATION
		beam_scores = beam_scores.view((batch_size * num_beams,))
		
		#----------------------------------------------------#
		# ENCODER: ONLY ONCE

		encoder_inputs = input_ids.repeat(1, num_beams).view(batch_size * num_beams, input_size)
		attention_mask = encoder_inputs.ne(0).long()

		encoder_outputs = self.encoder(
			input_ids = encoder_inputs,
			attention_mask = attention_mask,
			output_hidden_states = True,
			return_dict = True
		)

		#----------------------------------------------------#
		# DECODER: PREPARE INPUTS

		decoder_input_ids = torch.zeros(
			batch_beam_size, 1, self.vocab_size, device=device
		).float()

		decoder_input_ids[:, :, 1] = 1
		decoder_attention_mask = torch.ones(batch_beam_size, 1, device=device)

		######################################################

		for length in range(max_length - 1):

			#------------------------------------------------#
			# DECODER FORWARD-PASS

			decoder_outputs = self.forward(
				input_ids = input_ids,
				attention_mask = attention_mask,
				encoder_outputs = encoder_outputs,
				decoder_input_ids = decoder_input_ids,
				decoder_attention_mask = decoder_attention_mask
			)

			#------------------------------------------------#
			# RETRIEVE LOGITS

			g_logits = decoder_outputs["logits"][:, -1,                   :   self.gramm_size].detach()
			r_logits = decoder_outputs["logits"][:, -1,   self.gramm_size :-2*self.point_size].detach()
			p_logits = decoder_outputs["logits"][:, -1,-2*self.point_size :                  ].detach()

			#------------------------------------------------#
			# GRAMMAR MASKING: 'G_LOGITS'

			for idx, sequence in enumerate(strings):
				if undone[idx]: 
					l_bound = sequence.find("#_")
					r_bound = sequence.find("_#") + 2

					state = sequence[l_bound : r_bound]

					g_logits[idx][1:] = g_logits[idx][1:] + self.mask[self.smap[state]]
					g_logits[idx][0 ] = -float("inf")

					production = self.gramm.rule(torch.argmax(g_logits[idx][1:]))
					strings[idx] = strings[idx].replace(state, production[1], 1)

			#------------------------------------------------#
			# CLEANING G_LOGITS
			g_logits = F.one_hot(torch.argmax(g_logits, dim = -1), self.gramm_size)

			# CLEANING P_LOGITS
			l_index, r_index = p_logits.split(self.point_size, dim = -1)
			l_index = F.one_hot(torch.argmax(l_index, dim = -1), self.point_size)
			r_index = F.one_hot(torch.argmax(r_index, dim = -1), self.point_size)
			p_logits = torch.cat([l_index, r_index], dim = -1)

			#------------------------------------------------#
			# NEXT TOKEN SCORES
			next_token_scores = F.log_softmax(r_logits, dim = -1)
			
			# ONLY CONSIDER "TOKEN"-SEQUENCES
			for idx in range(batch_beam_size):
				if "TOKEN" in strings[idx]:
					counter[idx] = counter[idx] + 1
				else:
					next_token_scores[idx] = 0
			
			# ADD & NORMALIZE SCORES
			best_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
			norm_token_scores = best_token_scores / counter[:, None].expand_as(best_token_scores)
			
			# RESHAPE
			best_token_scores = best_token_scores.view(batch_size, num_beams * self.relat_size)
			norm_token_scores = norm_token_scores.view(batch_size, num_beams * self.relat_size)
			
			# BEST NORM_SCORES & INDICES
			best_scores, best_tokens = torch.topk(norm_token_scores, num_beams, dim = -1)
			beam_number = torch.div(best_tokens, self.relat_size, rounding_mode = "floor")
			
			# RETRIEVE UN-NORMALIZES SCORES
			orig_best_scores = torch.clone(best_scores)

			for idx in range(batch_size):
				orig_best_scores[idx] = best_token_scores[idx, best_tokens[idx]]

			best_tokens = best_tokens % self.relat_size
			best_tokens = best_tokens.view(-1)
			
			#------------------------------------------------#
			# RESHAPE FOR SHUFFLING
			g_logits = g_logits.view(batch_size, num_beams, self.gramm_size)
			r_logits = r_logits.view(batch_size, num_beams, self.relat_size)
			p_logits = p_logits.view(batch_size, num_beams, self.point_size * 2)
			
			undone = undone.view(batch_size, num_beams)
			counter = counter.view(batch_size, num_beams)

			beam_scores = beam_scores.view(batch_size, num_beams)

			decoder_input_ids = decoder_input_ids.view(batch_size, num_beams, length + 1, self.vocab_size)
			decoder_attention_mask = decoder_attention_mask.view(batch_size, num_beams, length + 1)
			#------------------------------------------------#
			# SHUFFLE BEAMS
			#---#
			new_strings = [] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			for s_index in range(batch_size):
				for b_index in range(num_beams):
					offset = s_index * num_beams
					item = strings[offset + beam_number[s_index, b_index].item()]
					new_strings.append(item)
			strings = new_strings
			#---# 
			
			for batch_index in range(batch_size):
				shuffles = beam_number[batch_index]

				# SHUFFLE LOGITS
				g_logits[batch_index] = g_logits[batch_index][shuffles]
				r_logits[batch_index] = r_logits[batch_index][shuffles]
				p_logits[batch_index] = p_logits[batch_index][shuffles]
				
				# SHUFFLE BOOKKEEPING
				counter[batch_index] = counter[batch_index][shuffles]
				undone[batch_index] = undone[batch_index][shuffles]

				# SHUFFLE PREV-SCORES
				beam_scores[batch_index] = beam_scores[batch_index][shuffles] + orig_best_scores[batch_index]

				# SHUFFLE DECODER-INPUTS
				decoder_input_ids[batch_index] = decoder_input_ids[batch_index][shuffles]
				decoder_attention_mask[batch_index] = decoder_attention_mask[batch_index][shuffles]

			#------------------------------------------------#
			# RESHAPE FOR COMPUTING
			g_logits = g_logits.view(batch_size * num_beams, self.gramm_size)
			r_logits = r_logits.view(batch_size * num_beams, self.relat_size)
			p_logits = p_logits.view(batch_size * num_beams, self.point_size * 2)
			
			undone = undone.view(batch_size * num_beams)
			counter = counter.view(batch_size * num_beams)

			beam_scores = beam_scores.view(batch_size * num_beams)

			#------------------------------------------------#
			# FOR EACH SAMPLE:		
			for idx in range(batch_beam_size): # batch_size <--> batch_beam_size
				if undone[idx]:
					undone[idx] = "[EOS]" not in strings[idx]

					if "TOKEN" not in strings[idx] and "POINT" not in strings[idx]:
						r_logits[idx] = 0
						p_logits[idx] = 0

					if "TOKEN" in strings[idx]:
						r_logits[idx] = F.one_hot(best_tokens[idx], self.relat_size)
						p_logits[idx] = 0
						strings[idx] = strings[idx].replace("TOKEN", "token")

					if "POINT" in strings[idx]:
						r_logits[idx] = 0
						strings[idx] = strings[idx].replace("POINT", "point")
				else:
					# BATCH_INPUT_IDS: PADDING
					g_logits[idx] = F.one_hot(torch.tensor(0), self.gramm_size)
					r_logits[idx] = 0
					p_logits[idx] = 0

			#------------------------------------------------#
			# DECODER_INPUT_IDS
			next_decoder_input_ids = torch.cat([g_logits, r_logits, p_logits], dim = -1)
			decoder_input_ids = decoder_input_ids.view(batch_beam_size, length + 1, self.vocab_size)
			decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids[:, None]], dim = 1)

			# DECODER_ATTENTION_MASK
			next_decoder_attention = undone.view(batch_size * num_beams, 1)
			decoder_attention_mask = decoder_attention_mask.view(batch_beam_size, length + 1)
			decoder_attention_mask = torch.cat([decoder_attention_mask, next_decoder_attention], dim = -1)
			#------------------------------------------------#
			# EARLY STOPPING?
			if sum(undone) == 0:
				break
			#------------------------------------------------#

		##################################################
		# FINALIZE OUTPUT_SEQUENCE

		length = decoder_input_ids.shape[1]
		decoder_input_ids = decoder_input_ids.view(batch_size, num_beams, length, self.vocab_size)
		decoder_input_ids = decoder_input_ids[:, 0, :, :] # BEST SCORE AT '0'

		##################################################
		
		return decoder_input_ids