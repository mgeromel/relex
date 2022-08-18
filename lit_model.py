import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from utils import *

#------------------------------------------------#

class LitModule(pl.LightningModule):

	def __init__(self, config, model, tokenizer):
		super().__init__()

		# 'config' as 'hparams'
		self.save_hyperparameters(config)

		self.model = model
		self.tokenizer = tokenizer

		# for 'valid' / 'tests'
		self.gramm_size = self.model.gramm_size
		self.vocab_size = self.model.vocab_size
		self.point_size = self.model.point_size

		self.gramm = self.model.gramm
		self.vocab = self.model.vocab
		self.bacov = {v:k for k,v in self.vocab.items()}

	#--------------------------------------------#

	def configure_optimizers(self):

		optimizer = torch.optim.Adam(
			self.parameters(),
			lr = 5e-5
		)
		
		return optimizer

	#--------------------------------------------#

	def forward(self, input_ids = None, attention_mask = None, encoder_outputs = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):

		# 1. ENCODER & DECODER
		output = self.model(
			input_ids = input_ids,
			attention_mask = attention_mask,
			decoder_input_ids = decoder_input_ids,
			decoder_attention_mask = decoder_attention_mask,
			encoder_outputs = encoder_outputs,
			labels = labels
		)

		return output
	
	#--------------------------------------------#

	def training_step(self, batch, batch_idx):
		output = self.forward(**batch)
		loss = output["loss"]

		self.log("train_loss", loss)
		
		return loss

	#--------------------------------------------#

	def validation_step(self, batch, batch_idx):

		# Generate-Arguments
		gen_kwargs = {
			"max_length": 128,
			"use_cache": True
		}

		# Generate Sequences
		gen_tokens = self.generate(
			batch["input_ids"],
			**gen_kwargs
		)

		# Decode: 'predic' & 'labels'
		labels = self.translate(batch["input_ids"], batch["decoder_input_ids"])
		predic = self.translate(batch["input_ids"], gen_tokens)

		return { "predic": predic, "labels": labels }

	#--------------------------------------------#

	def validation_epoch_end(self, output):

		result = {
			"predicted": [],
			"label_ids": []
		}

		for batch in output:
			for lab, pre in zip(batch["labels"], batch["predic"]):
				result["predicted"].append(pre)
				result["label_ids"].append(lab)

		metrics = compute_metrics_old(result)

		print(metrics)

		self.log('valid_P', metrics["P"])
		self.log('valid_R', metrics["R"])
		self.log('valid_F', metrics["F"])

	#--------------------------------------------#
	
	def prepare_inputs(
			self,
			decoder_input_ids,
			past = None,
			attention_mask = None,
			head_mask = None,
			decoder_head_mask = None,
			cross_attn_head_mask = None,
			use_cache = None,
			encoder_outputs = None,
			**kwargs
		):

		if past is not None:
			decoder_input_ids = decoder_input_ids[:, -1:]

		return {
			"input_ids": None,
			"encoder_outputs": encoder_outputs,
			"past_key_values": past,
			"decoder_input_ids": decoder_input_ids,
			"attention_mask": attention_mask,
			"head_mask": head_mask,
			"decoder_head_mask": decoder_head_mask,
			"cross_attn_head_mask": cross_attn_head_mask,
			"use_cache": use_cache,
		}
	
	def mask_gramm(self, logits, stacks, length):
		
		padding = torch.Tensor(
			[0] + [-float("inf")] * (self.gramm_size - 1)
		)

		full_mask = padding.repeat(len(stacks), 1)

		#----------------------------------------#
		# Create Gramm-Mask
		
		for idx, stack in enumerate(stacks):
			if len(stack) > 0:
				item = stack.pop(0)

				# Retrieve Item-Mask
				item_mask = self.gramm.get_mask(item)

				# Overwrite 'Padding'
				full_mask[idx] = item_mask
		
		#----------------------------------------#
		# Masking & Arg-Max

		logits = logits + full_mask
		tokens = torch.argmax(logits, dim = -1)

		#----------------------------------------#
		# Update States/Stacks

		for idx, token in enumerate(tokens):
			if token != 0:
				productions = self.gramm.rule(token-1)
				new_symbols = productions[1] 

				stacks[idx] = new_symbols + stacks[idx]

		#----------------------------------------#

		return tokens, stacks

	def mask_token(self,
		gramm_tokens,
		vocab_tokens,
		point_tokens,
		qoint_tokens,
		mask_tensor
	):

		#----------------------------------------#
		# Compute Mask from 'mask_tensor'

		mask = mask_tensor[gramm_tokens]

		# Extract Masks from 'mask'

		vocab_mask = mask[:, 0]
		point_mask = mask[:, 1]

		#----------------------------------------#
		# 'Tokens' --> 'Logits'

		gramm_tokens = F.one_hot(gramm_tokens, self.gramm_size)
		vocab_tokens = F.one_hot(vocab_tokens, self.vocab_size)
		point_tokens = F.one_hot(point_tokens, self.point_size)
		qoint_tokens = F.one_hot(qoint_tokens, self.point_size)
		
		#----------------------------------------#
		# Masking 'Logits'

		vocab_tokens = vocab_tokens * vocab_mask[:, None]
		point_tokens = point_tokens * point_mask[:, None]
		qoint_tokens = qoint_tokens * point_mask[:, None]
		
		#----------------------------------------#
		# Combine 'Logits'
		
		result = torch.cat(
			[gramm_tokens, vocab_tokens, point_tokens, qoint_tokens],
			dim = -1
		)
		
		#----------------------------------------#
		
		return result

	def generate(
		self, input_ids, max_length = 256, use_cache = True, **gen_kwargs
	):
		
		#----------------------------------------#
		# VARIABLES

		bos_token_id = self.tokenizer.bos_token_id # '1' for decoder
		eos_token_id = self.tokenizer.eos_token_id # '2' for decoder
		pad_token_id = self.tokenizer.pad_token_id # '0bat' for decoder

		batch_size = input_ids.shape[0]

		#----------------------------------------#
		# ENCODER: ONCE

		attention_mask = input_ids.ne(pad_token_id).long()
		encoder = self.model.model.get_encoder()
		encoder_output = encoder(
			input_ids,
			attention_mask = attention_mask,
			return_dict = True
		)
		gen_kwargs["encoder_outputs"] = encoder_output

		#----------------------------------------#
		# DECODER INPUT
	
		indices = torch.ones(batch_size, dtype=torch.int64)
		indices = indices.view(-1, 1).to(input_ids.device)
		
		decoder_input_ids = F.one_hot(
			indices,
			self.model.output_dim
		)

		#----------------------------------------#
		# CURRENT "STATE"
		
		unfinished = input_ids.new(batch_size).fill_(1)
		
		# required when masking 'gramm_logits'
		gramm_states = [[ "#_RELS_#" ] for _ in range(batch_size)]

		# For Token-Masking
		tensor_mask = torch.Tensor(
			[
				[0,0,0,1,1,0,0,0],
				[0,0,0,0,0,0,1,0]
			]
		).transpose(0, 1).long()
		
		#----------------------------------------#

		length = 1
		past = None

		while length < max_length:

			#------------------------------------#
			# PREPARE

			# TODO: old approach might also work
			model_inputs = self.prepare_inputs(
				decoder_input_ids,
				past = past,
				attention_mask = attention_mask,
				use_cache = use_cache,
				**gen_kwargs
			)

			# FORWARD
			decoder_outputs = self.model(
				**model_inputs, split_result = True
			)

			#------------------------------------#
			# NEXT TOKENS
			
			gramm_logits = decoder_outputs["gramm_logits"][:, -1, :]
			vocab_logits = decoder_outputs["vocab_logits"][:, -1, :]
			point_logits = decoder_outputs["point_logits"][:, -1, : self.point_size ]
			qoint_logits = decoder_outputs["point_logits"][:, -1, self.point_size : ]

			gramm_tokens, gramm_states = self.mask_gramm(gramm_logits, gramm_states, length)
			
			# Next Tokens (All-In)
			vocab_tokens = torch.argmax(vocab_logits, dim = -1)
			point_tokens = torch.argmax(point_logits, dim = -1)
			qoint_tokens = torch.argmax(qoint_logits, dim = -1)

			# Determine Token-Mask
			final_tokens = self.mask_token(
				gramm_tokens,
				vocab_tokens,
				point_tokens,
				qoint_tokens,
				tensor_mask
			)

			# Finalize Post-Processing
			padding = torch.zeros(batch_size, dtype=torch.int64)
			padding = F.one_hot(padding, self.model.output_dim)

			unfinished = unfinished.view(-1, 1)
			final_tokens = final_tokens * unfinished + padding * (1 - unfinished)
			unfinished = unfinished.view(-1)

			#------------------------------------#			
			# INPUT LENGTH + 1

			# TODO: unsqueeze()?
			decoder_input_ids = torch.cat([decoder_input_ids, final_tokens.unsqueeze(-2)], dim=1)
			length = length + 1
			
			#------------------------------------#
			# SEQUENCES DONE?

			done = torch.argmax(final_tokens, dim=-1) == eos_token_id
			unfinished = unfinished.mul((~done).long())
			
			#------------------------------------#
			# MODEL PAST?
			
			if "past_key_values" in decoder_outputs:
				past = decoder_outputs["past_key_values"]

			# IF ALL_DONE: BREAK
			if unfinished.max() == 0:
				break
			
			#------------------------------------#

		return decoder_input_ids
	
	#--------------------------------------------#
	
	def reduce(self, tokens, logits):
		G = torch.argmax(logits[ : self.gramm_size]).item()
		V = torch.argmax(logits[self.gramm_size : self.gramm_size + self.vocab_size]).item()
		P = (
			torch.argmax(logits[ -2 * self.point_size : -self.point_size ]).item(),
			torch.argmax(logits[    - self.point_size :              ]).item()
		)
		PP = self.tokenizer.decode(tokens[P[0] : P[1]].tolist(), skip_special_tokens = True).strip()

		return (G, V, PP, P)

	#--------------------------------------------#

	def translate(self, input_ids, batch):
		
		result = []
		
		for tokens, output in zip(input_ids, batch):
			table_dicts = []
			
			table_dict = {}
			curr_entry = "DEFAULT"
			
			for logits in output:
				G, V, P, I = self.reduce(tokens, logits)
				
				# STATE 2
				if G == 2:
					if table_dict != {}:
						table_dicts.append(table_dict)
					
					break
					
				# STATE 3
				if G == 3:
					table_dict["TABLE_ID"] = self.bacov[V]
					
				# STATE 4
				if G == 4:
					curr_entry = self.bacov[V]
				
				if G == 5:
					table_dicts.append(table_dict)
					table_dict = {}
					curr_entry = "DEFAULT"
				
				# STATE 5
				if G == 6:
					if curr_entry not in table_dict:
						table_dict[curr_entry] = []
					
					table_dict[curr_entry].append( P )
			
			result.append(table_dicts)
		
		# TRANSFORM TABLES
		final_result = []
		
		for table_dicts in result:
			
			temp_result = []
			for table_dict in table_dicts:
				
				dirty = False
				
				for key in table_dict.keys():
					if key != "TABLE_ID" and "" in table_dict[key]:
						dirty = True
				
				if not dirty and "TABLE_ID" in table_dict:
					temp_result.append(table_dict)

			final_result.append(temp_result)
			
		return final_result

#-----------------------------------------------------------#