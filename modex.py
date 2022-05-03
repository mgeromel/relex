from transformers import *

import os, torch, numpy

from encoder import CustomAlbertModel
from decoder import CustomBertGenerationDecoder

##################################################

class TestModel(torch.nn.Module):
	def __init__(self, gramm = None, vocab = None, point_size = 256):
		super(TestModel, self).__init__()
		
		self.gramm_size = gramm.size() + 1
		self.relat_size = len(vocab)
		self.point_size = point_size
		self.vocab_size = self.gramm_size + self.relat_size + 2 * self.point_size
		
		self.encoder = CustomAlbertModel.from_pretrained(
			"albert-base-v1",
			add_pooling_layer = False,
			bos_token_id = 2, 
			eos_token_id = 3
		)
		
		self.decoder = CustomBertGenerationDecoder(
			BertGenerationConfig(
				vocab_size = self.vocab_size,
				add_cross_attention = True,
				is_decoder = True, 
				use_cache = False,
				pad_token_id = 0, #?
				bos_token_id = 1, #?
				eos_token_id = 2, #?
				hidden_size = 768,
				num_hidden_layers = 8,
				num_attention_heads = 12,
				intermediate_size = 4096
			)
		)
		
		self.decoder.bert.encoder.gradient_checkpointing = True
		
		self.gramm_head = torch.nn.Linear(       self.vocab_size,     self.gramm_size)
		self.relat_head = torch.nn.Linear(       self.vocab_size,     self.relat_size)
		self.point_head = torch.nn.Linear( 768 + self.point_size, 2 * self.point_size)
		
		self.dropout_logits = torch.nn.Dropout(0.0)

	##################################################
	
	def __init__BACKUP(self, gramm = None, vocab = None, point_size = 256):
		super(TestModel, self).__init__()
		
		self.gramm_size = gramm.size() + 1
		self.relat_size = len(vocab)
		self.point_size = point_size
		self.vocab_size = self.gramm_size + self.relat_size + 2 * self.point_size
		
		self.encoder = BertGenerationEncoder.from_pretrained(
			"bert-base-uncased",
			bos_token_id = 101, 
			eos_token_id = 102
		)
		
		self.encoder.encoder.gradient_checkpointing = True
		
		self.decoder = gen_decoder.RelexDecoder(
			BertGenerationConfig(
				vocab_size = self.vocab_size,
				add_cross_attention = True,
				is_decoder = True, 
				use_cache = False,
				pad_token_id = 0, #?
				bos_token_id = 1, #?
				eos_token_id = 2, #?
				hidden_size = 768,
				num_hidden_layers = 8,
				num_attention_heads = 12,
				intermediate_size = 3072,
				#gradient_checkpointing = True
			)
		)
		
		self.decoder.bert.encoder.gradient_checkpointing = True
		
		self.gramm_head = torch.nn.Linear(      self.vocab_size,     self.gramm_size)
		self.relat_head = torch.nn.Linear(      self.vocab_size,     self.relat_size)
		self.point_head = torch.nn.Linear(768 + self.point_size, 2 * self.point_size)
		
		self.dropout_logits = torch.nn.Dropout(0.15)
	
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
			
			g_loss = loss_func(g_logits, g_labels) * 0.25
			r_loss = loss_func(r_logits, r_labels) * 0.25
			p_loss = loss_func(p_logits, p_labels) * 0.25
			q_loss = loss_func(q_logits, q_labels) * 0.25
			
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
		
		logits = self.dropout_logits(decoder_outputs["logits"])
		
		# CROSS_ATTENTIONS: batch_size x num_attention_heads x decoder_sequence_length x encoder_seqence_length
		last_cross_attentions = decoder_outputs["cross_attentions"][-1]
		decoder_hidden_states = decoder_outputs["hidden_states"][-1]
		last_cross_attentions = last_cross_attentions.sum(dim = 1)
		
		# 3. FUNCTION HEAD: GRAMM / RELAT
		gramm_logits = self.gramm_head(logits)
		relat_logits = self.relat_head(logits)
		
		# 4. FUNCTION HEAD: POINTER-NETWORK
		point_values = torch.cat([last_cross_attentions, decoder_hidden_states], dim = -1)
		point_logits = self.point_head(point_values)
		
		# 4. PREDICTION: LOGITS / LOSS
		logits = torch.cat([gramm_logits, relat_logits, point_logits], dim = -1)
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
					undone[i] = gv != 2

					if gv in [3, 4]:
						r_logits[i, rv] = 1 # RELATION?

					if gv in [5]:
						p_logits[i, pv] = 1 # ENTITY?
				else:
					g_logits[i, 0] = 1

			##################################################

			next_tokens = torch.cat([g_logits, r_logits, p_logits], dim = 1)
			decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=1)
			decoder_attention_mask = torch.cat([decoder_attention_mask, undone[:, None]], dim=1)
		
		##################################################
		
		return decoder_input_ids
