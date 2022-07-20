from transformers import AlbertModel, BertGenerationDecoder
from transformers import BertGenerationConfig

from transformers import BartForConditionalGeneration
from transformers import BartTokenizer

from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder

import torch

from bart_encoder import *

#------------------------------------------------#

class REBEL(torch.nn.Module):
	
	__load_model = True
	
	def __init__(self, tokenizer):
		super(REBEL, self).__init__()
		
		BartEncoder.forward = custom_encoder_forward
		BartDecoder.forward = custom_decoder_forward
	
		# 0. TOKENIZER
		self.tokenizer = tokenizer
		self.basemodel = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
		self.basemodel.resize_token_embeddings(len(tokenizer))
		
		# 1. ENCODER
		self.encoder = self.basemodel.model.encoder
		self.encoder.gradient_checkpointing = True
		
		# 1.5 FREEZE LAYERS 0..5 
		frozen_layers = 0
		for layer in self.encoder.layers[:frozen_layers]:
			for param in layer.parameters():
				param.requires_grad = False
			
		# 2. DECODER
		self.decoder = self.basemodel.model.decoder
		self.decoder.resize_token_embeddings(len(tokenizer))
		self.decoder.gradient_checkpointing = True
		self.decoder.config.use_cache = False
		
	#--------------------------------------------#
	
	def compute_loss(self, logits, labels):
		loss = None
		
		if labels is not None:
			shifted_logits = logits[:,:-1,:].contiguous()
			shifted_labels = labels[:, 1 : ].contiguous()
			
			loss_function = torch.nn.CrossEntropyLoss()
			
			loss = loss_function(
				shifted_logits.view(-1, self.decoder.config.vocab_size),
				shifted_labels.view(-1)
			)
		
		return loss
	
	#--------------------------------------------#

	def forward(self, input_ids = None, attention_mask = None, encoder_outputs = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):
		
		# 1. ENCODER
		output = self.basemodel(
			input_ids = input_ids,
			attention_mask = attention_mask,
			decoder_input_ids = decoder_input_ids,
			decoder_attention_mask = decoder_attention_mask,
			encoder_outputs = encoder_outputs,
		)
		
		logits = output.logits
		loss = self.compute_loss(logits, labels)
			
		return {
			"loss": loss,
			"logits": logits
		}
	
	#--------------------------------------------#
	
	def generate(self, input_ids = None, max_length = 64, **kwargs):
		
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		bos_token_id = self.tokenizer.bos_token_id
		eos_token_id = self.tokenizer.eos_token_id
		pad_token_id = self.tokenizer.pad_token_id
		
		#----------------------------------------#
		
		# ENCODER: ONLY ONCE
		attention_mask = input_ids.ne(0).long()
			
		encoder_outputs = self.encoder(
			input_ids = input_ids,
			attention_mask = attention_mask,
			output_hidden_states = True,
			return_dict = True
		)
		
		#----------------------------------------#
			
		batch_size = input_ids.shape[0]                 # NUMBER OF SENTENCES
		input_size = input_ids.shape[1]                 # LENGTH OF SENTENCES
		
		undone = torch.ones(batch_size, device=device)  # GENERATED SEQUENCES (FLAG)
		
		decoder_input_ids = torch.zeros(
			batch_size, 1, device=device
		).float()
		
		decoder_input_ids[:, 0] = bos_token_id # "[CLS]"
		decoder_attention_mask = torch.ones(batch_size, 1, device=device)
		
		#----------------------------------------#
		
		for length in range(max_length - 1):
			
			if sum(undone) == 0:
				break
			
			decoder_outputs = self.forward(
				input_ids = input_ids,
				attention_mask = attention_mask,
				encoder_outputs = encoder_outputs,
				decoder_input_ids = decoder_input_ids.int(),
				decoder_attention_mask = decoder_attention_mask
			)
			
			logits = decoder_outputs["logits"].detach()
			tokens = torch.argmax(logits, dim = -1)
			tokens = undone * tokens[:, -1]
			
			for idx, token in enumerate(tokens):
				if token.item() == eos_token_id: # "[SEP]"
					undone[idx] = 0
			
			decoder_input_ids = torch.cat([decoder_input_ids, tokens[:, None]], dim = 1)
			decoder_attention_mask = torch.cat([decoder_attention_mask, undone[:, None]], dim = 1)
		
		#----------------------------------------#
		
		return decoder_input_ids