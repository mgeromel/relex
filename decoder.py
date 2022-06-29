from transformers.models.bert_generation.modeling_bert_generation import (
	BertGenerationEmbeddings,
	BertGenerationDecoder,
	BertGenerationEncoder
)

from transformers.models.bart.modeling_bart import (
	BartDecoder,
	_expand_mask
)

from transformers.modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
	CausalLMOutputWithCrossAttentions
)

import torch, random
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
	
#############################################################
#-----------------------------------------------------------#
#############################################################

class CustomBartDecoder(BartDecoder):
	def __init__(self, config, embed_tokens = None, gradient_checkpointing = True):
		super().__init__(config)
		
		self.embed_tokens = nn.Linear(config.vocab_size, config.d_model, self.padding_idx)
		self.gradient_checkpointing = gradient_checkpointing
		
	def forward(
		self,
		input_ids = None,
		attention_mask = None,
		encoder_hidden_states = None,
		encoder_attention_mask = None,
		head_mask = None,
		cross_attn_head_mask = None,
		past_key_values = None,
		inputs_embeds = None,
		use_cache = None,
		output_attentions = None,
		output_hidden_states = None,
		return_dict = None
	):
		
		#----------------------------------------------#
			
		# Retrieve InputIDs
		input_shape = input_ids.size() #input_ids = input_ids.view(-1, input_shape[-1])
		input_shape = input_shape[:2]
		
		# Past Key Values Length
		past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
		
		attention_mask = self._prepare_decoder_attention_mask(
			attention_mask, input_shape[:2], inputs_embeds, past_key_values_length
		)
		
		# Expand Encoder Attention Mask
		if encoder_hidden_states is not None and encoder_attention_mask is not None:
			# [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
			encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

		# Embed Positions
		positions = self.embed_positions(input_shape, past_key_values_length)

		hidden_states = inputs_embeds + positions
		hidden_states = self.layernorm_embedding(hidden_states)
		hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
		next_decoder_cache = () if use_cache else None
		
		#----------------------------------------------#
			
		# check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
		for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
			if attn_mask is not None:
				if attn_mask.size()[0] != (len(self.layers)):
					raise ValueError(
						f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
						f" {head_mask.size()[0]}."
					)

		#----------------------------------------------#
			
		for idx, decoder_layer in enumerate(self.layers):
			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			if output_hidden_states:
				all_hidden_states += (hidden_states,)
			dropout_probability = random.uniform(0, 1)
			if self.training and (dropout_probability < self.layerdrop):
				continue

			past_key_value = past_key_values[idx] if past_key_values is not None else None

			#----------------------------------------------#
			
			# Gradient Checkpointing?
			if self.gradient_checkpointing and self.training:
				
				use_cache = False

				def create_custom_forward(module):
					def custom_forward(*inputs):
						return module(*inputs, output_attentions, use_cache)
					return custom_forward

				layer_outputs = torch.utils.checkpoint.checkpoint(
					create_custom_forward(decoder_layer),
					hidden_states,
					attention_mask,
					encoder_hidden_states,
					encoder_attention_mask,
					head_mask[idx] if head_mask is not None else None,
					cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
					None,
				)
				
			else:
				layer_outputs = decoder_layer(
					hidden_states,
					attention_mask=attention_mask,
					encoder_hidden_states=encoder_hidden_states,
					encoder_attention_mask=encoder_attention_mask,
					layer_head_mask=(head_mask[idx] if head_mask is not None else None),
					cross_attn_layer_head_mask=(
						cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
					),
					past_key_value=past_key_value,
					output_attentions=output_attentions,
					use_cache=use_cache,
				)
			
			#----------------------------------------------#
		
			# All Hidden States
			hidden_states = layer_outputs[0]

			if use_cache:
				next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

				if encoder_hidden_states is not None:
					all_cross_attentions += (layer_outputs[2],)
			
			#----------------------------------------------#
			
		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = next_decoder_cache if use_cache else None
		
		#--------------------------------------------------#
		
		if not return_dict:
			return tuple(
				v
				for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
				if v is not None
			)
		
		#--------------------------------------------------#
		
		return BaseModelOutputWithPastAndCrossAttentions(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
			cross_attentions=all_cross_attentions,
		)
	
		#--------------------------------------------------#