from transformers import BertModel

import torch

class MyModel(torch.nn.Model):

	def __init__(self, enc_seq_len = 1024):
		# Bert-Base-Encoder (Pre-Trained) 
		self.encoder = BertModel.from_pretrained(
			"bert-base-uncased",
		)

		# Einfaches Pointer-Network (viele Möglichkeiten, gerne bearbeiten)
		self.pointer_network = torch.nn.Linear(self.encoder.config.hidden_size, 2 * enc_seq_len)

	def compute_loss(logits, labels):

		result = None

		# ...

		return result

	def forward(self,
		input_ids = None,
		attention_mask = None,
		labels = None
	):
		
		# 1. Encoder-Pass
		encoder_outputs = self.encoder(
			input_ids = input_ids,
			attention_mask = attention_mask,
			return_dict = True,
		)

		# 2. Input Pointer-Network
		pointer_input = encoder_outputs.pooler_output # shape = batch_size x encoder_hidden_size

		logits = self.pointer_network(pointer_input)
		loss = compute_loss(logits, labels)

		return {
			"logits" : logits,
			"loss" : loss
		}













		
		# encoder_hidden_states = encoder_outputs[0]

		# 3. SELF ATTENTION & HIDDEN STATES
		# last_cross_attentions = encoder_outputs.cross_attentions[-1] # shappe = (batch_size, num_attn_heads, dec_seq_len, enc_seq_len)
		# last_cross_attentions = last_cross_attentions.sum(dim = 1) # shape = (batch_size, dec_seq_len, enc_seq_len)
		# decoder_hidden_states = decoder_outputs.hidden_states[-1] # shape = (batch_size, dec_seq_len, dec_hidden_size)


		# 4. POINTER-NETWORK (cf. Figure 1, https://aclanthology.org/2020.aacl-srw.13.pdf)
		# values = torch.cat([last_cross_attentions, decoder_hidden_states], dim = -1)
		# logits = self.point_layer(point_values)