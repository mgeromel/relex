from transformers.models.bert_generation.modeling_bert_generation import (
	BertGenerationEmbeddings,
	BertGenerationDecoder,
	BertGenerationEncoder
)

from transformers.modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
	CausalLMOutputWithCrossAttentions
)

import torch
import torch.nn as nn

#-----------------------------------------------------------#

class CustomBertGenerationEmbeddings(BertGenerationEmbeddings):
	def __init__(self, config):
		super().__init__(config)
		self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)

#-----------------------------------------------------------#

class CustomBertGenerationDecoder(BertGenerationDecoder):
	def __init__(self, config):
		super().__init__(config)
		self.bert = CustomGenerationEncoder(config)
	
#-----------------------------------------------------------#

class CustomGenerationEncoder(BertGenerationEncoder):
	def __init__(self, config):
		super().__init__(config)
		self.embeddings = CustomBertGenerationEmbeddings(config)
	
	def forward(
		self,
		input_ids = None,
		attention_mask = None,
		position_ids = None,
		head_mask = None,
		inputs_embeds = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		past_key_values = None,
		use_cache = False,
		output_attentions = True,
		output_hidden_states = True,
		return_dict = True,
	):
		
		#--------------------------------------------------#
		
		input_shape = input_ids.size()[:2]
		
		batch_size, seq_length = input_shape
		
		device = input_ids.device

		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		extended_attention_mask = self.get_extended_attention_mask(
			attention_mask, input_shape, device
		)
		
		#--------------------------------------------------#

		encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
		encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
			
		if encoder_attention_mask is None:
			encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
			
		encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
		
		#--------------------------------------------------#

		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
		
		embedding_output = self.embeddings(
			input_ids=input_ids,
			position_ids=position_ids,
			inputs_embeds=inputs_embeds,
			past_key_values_length=past_key_values_length,
		)

		encoder_outputs = self.encoder(
			embedding_output,
			attention_mask=extended_attention_mask,
			head_mask=head_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_extended_attention_mask,
			past_key_values=past_key_values,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		
		sequence_output = encoder_outputs[0]

		#--------------------------------------------------#
		
		if not return_dict:
			return (sequence_output,) + encoder_outputs[1:]

		return BaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=sequence_output,
			past_key_values=encoder_outputs.past_key_values,
			hidden_states=encoder_outputs.hidden_states,
			attentions=encoder_outputs.attentions,
			cross_attentions=encoder_outputs.cross_attentions,
		)

#-----------------------------------------------------------#