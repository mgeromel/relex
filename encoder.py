from transformers import AlbertModel

from transformers.models.albert.modeling_albert import (
	AlbertModel,
	AlbertTransformer,
	AlbertLayerGroup,
	AlbertLayer
)

import torch
import torch.nn as nn

#-----------------------------------------------------------#

class CustomAlbertModel(AlbertModel):
	def __init__(self, config, add_pooling_layer = False):
		super().__init__(config, add_pooling_layer = add_pooling_layer)
		self.encoder = CustomAlbertTransformer(config)
		self.post_init()

#-----------------------------------------------------------#

class CustomAlbertTransformer(AlbertTransformer):
	def __init__(self, config):
		super().__init__(config)
		self.albert_layer_groups = nn.ModuleList(
			[CustomAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
		)
		
#-----------------------------------------------------------#

class CustomAlbertLayerGroup(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.albert_layers = nn.ModuleList(
			[AlbertLayer(config) for _ in range(config.inner_group_num)]
		)
		self.gradient_checkpointing = True
		
	def forward(
		self,
		hidden_states,
		attention_mask = None,
		head_mask = None,
		output_attentions = False,
		output_hidden_states = False
	):
		
		#--------------------------------------------------#
		
		layer_hidden_states = ()
		layer_attentions = ()
		
		#--------------------------------------------------#
		
		for layer_index, albert_layer in enumerate(self.albert_layers):
			
			#----------------------------------------------#
			
			if self.gradient_checkpointing and self.training:
				def create_custom_forward(module):
					
					def custom_forward(*inputs):
						return module(*inputs)
					return custom_forward
				
				layer_output = torch.utils.checkpoint.checkpoint(
					create_custom_forward(albert_layer),
					hidden_states,
					attention_mask,
					head_mask[layer_index],
					output_attentions
				)
				
			else:
				layer_output = albert_layer(
					hidden_states,
					attention_mask,
					head_mask[layer_index],
					output_attentions
				)
				
			#----------------------------------------------#
			
			hidden_states = layer_output[0]

			if output_attentions:
				layer_attentions = layer_attentions + (layer_output[1],)

			if output_hidden_states:
				layer_hidden_states = layer_hidden_states + (hidden_states,)
		
		#--------------------------------------------------#
		
		outputs = (hidden_states,)
		
		if output_hidden_states:
			outputs = outputs + (layer_hidden_states,)
		if output_attentions:
			outputs = outputs + (layer_attentions,)
		
		return outputs
	
#-----------------------------------------------------------#