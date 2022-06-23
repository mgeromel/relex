import torch

from transformers.models.bart.modeling_bart import *

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
	"""
	Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
	"""
	bsz, src_len = mask.size()
	tgt_len = tgt_len if tgt_len is not None else src_len

	expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

	inverted_mask = 1.0 - expanded_mask

	return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def custom_encoder_forward(
	self,
	input_ids=None,
	attention_mask=None,
	head_mask=None,
	inputs_embeds=None,
	output_attentions=None,
	output_hidden_states=None,
	return_dict=None,
):
	
	output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
	output_hidden_states = (
		output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
	)
	return_dict = return_dict if return_dict is not None else self.config.use_return_dict

	# retrieve input_ids and inputs_embeds
	if input_ids is not None and inputs_embeds is not None:
		raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
	elif input_ids is not None:
		input_shape = input_ids.size()
		input_ids = input_ids.view(-1, input_shape[-1])
	elif inputs_embeds is not None:
		input_shape = inputs_embeds.size()[:-1]
	else:
		raise ValueError("You have to specify either input_ids or inputs_embeds")

	if inputs_embeds is None:
		inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

	embed_pos = self.embed_positions(input_shape)

	hidden_states = inputs_embeds + embed_pos
	hidden_states = self.layernorm_embedding(hidden_states)
	hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

	# expand attention_mask
	if attention_mask is not None:
		# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
		attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

	encoder_states = () if output_hidden_states else None
	all_attentions = () if output_attentions else None

	# check if head_mask has a correct number of layers specified if desired
	if head_mask is not None:
		assert head_mask.size()[0] == (
			len(self.layers)
		), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
	for idx, encoder_layer in enumerate(self.layers):
		if output_hidden_states:
			encoder_states = encoder_states + (hidden_states,)
		# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
		dropout_probability = random.uniform(0, 1)
		if self.training and (dropout_probability < self.layerdrop):  # skip the layer
			layer_outputs = (None, None)
		else:
			
			if self.training:
				def create_custom_forward(module):
					def custom_forward(*inputs):
						return module(*inputs, output_attentions)

					return custom_forward

				layer_outputs = torch.utils.checkpoint.checkpoint(
					create_custom_forward(encoder_layer),
					hidden_states,
					attention_mask,
					(head_mask[idx] if head_mask is not None else None),
				)
			else:
				layer_outputs = encoder_layer(
					hidden_states,
					attention_mask,
					layer_head_mask=(head_mask[idx] if head_mask is not None else None),
					output_attentions=output_attentions,
				)

			hidden_states = layer_outputs[0]

		if output_attentions:
			all_attentions = all_attentions + (layer_outputs[1],)

	if output_hidden_states:
		encoder_states = encoder_states + (hidden_states,)

	if not return_dict:
		return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
	return BaseModelOutput(
		last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
	)


def custom_decoder_forward(
	self,
	input_ids=None,
	attention_mask=None,
	encoder_hidden_states=None,
	encoder_attention_mask=None,
	head_mask=None,
	cross_attn_head_mask=None,
	past_key_values=None,
	inputs_embeds=None,
	use_cache=None,
	output_attentions=None,
	output_hidden_states=None,
	return_dict=None,
):
	output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
	output_hidden_states = (
		output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
	)
	use_cache = use_cache if use_cache is not None else self.config.use_cache
	return_dict = return_dict if return_dict is not None else self.config.use_return_dict

	# retrieve input_ids and inputs_embeds
	if input_ids is not None and inputs_embeds is not None:
		raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
	elif input_ids is not None:
		input_shape = input_ids.size()
		input_ids = input_ids.view(-1, input_shape[-1])
	elif inputs_embeds is not None:
		input_shape = inputs_embeds.size()[:-1]
	else:
		raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

	# past_key_values_length
	past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

	if inputs_embeds is None:
		inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

	attention_mask = self._prepare_decoder_attention_mask(
		attention_mask, input_shape, inputs_embeds, past_key_values_length
	)

	# expand encoder attention mask
	if encoder_hidden_states is not None and encoder_attention_mask is not None:
		# [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
		encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

	# embed positions
	positions = self.embed_positions(input_shape, past_key_values_length)

	hidden_states = inputs_embeds + positions
	hidden_states = self.layernorm_embedding(hidden_states)

	hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

	# decoder layers
	all_hidden_states = () if output_hidden_states else None
	all_self_attns = () if output_attentions else None
	all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
	next_decoder_cache = () if use_cache else None

	# check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
	for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
		if attn_mask is not None:
			assert attn_mask.size()[0] == (
				len(self.layers)
			), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
	for idx, decoder_layer in enumerate(self.layers):
		# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		dropout_probability = random.uniform(0, 1)
		if self.training and (dropout_probability < self.layerdrop):
			continue

		past_key_value = past_key_values[idx] if past_key_values is not None else None

		if self.training:

			if use_cache:
				#logger.warning(
				#	"`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
				#	"`use_cache=False`..."
				#)
				use_cache = False

			def create_custom_forward(module):
				def custom_forward(*inputs):
					# None for past_key_value
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
		hidden_states = layer_outputs[0]

		if use_cache:
			next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

		if output_attentions:
			all_self_attns += (layer_outputs[1],)

			if encoder_hidden_states is not None:
				all_cross_attentions += (layer_outputs[2],)

	# add hidden states from the last decoder layer
	if output_hidden_states:
		all_hidden_states += (hidden_states,)

	next_cache = next_decoder_cache if use_cache else None
	if not return_dict:
		return tuple(
			v
			for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
			if v is not None
		)
	return BaseModelOutputWithPastAndCrossAttentions(
		last_hidden_state=hidden_states,
		past_key_values=next_cache,
		hidden_states=all_hidden_states,
		attentions=all_self_attns,
		cross_attentions=all_cross_attentions,
	)