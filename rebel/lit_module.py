from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

import torch
import pytorch_lightning as pl

from datas import *
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
		self.seqreader = SEQReader(config.dataset_name)

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
	
	def compute_loss(self, logits, labels):
		loss = None
		
		if labels is not None:
			loss_function = torch.nn.CrossEntropyLoss()
			
			loss = loss_function(
				logits.view(-1, self.model.config.vocab_size),
				labels.view(-1)
			)
		
		return loss
	
	#--------------------------------------------#

	def training_step(self, batch, batch_idx):
		output = self.forward(**batch)
		
		labels = batch["labels"]
		logits = output.logits

		loss = self.compute_loss(logits, labels) # output.loss

		self.log("train_loss", loss)
		
		return loss

	#--------------------------------------------#

	def validation_step(self, batch, batch_idx):

		# Generate-Arguments
		gen_kwargs = {
			"max_length": 128,
			"early_stopping": False,
			"no_repeat_ngram_size": 0,
			"length_penalty": 0,
			"num_beams": 1,
			"use_cache": True
		}

		# Generate Sequences
		gen_tokens = self.generate(
			batch["input_ids"],
			**gen_kwargs
		)

		# Decode: 'predic' & 'labels'
		predic = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens = False)
		labels = self._replace(batch["labels"], -100, self.model.config.pad_token_id)
		labels = self.tokenizer.batch_decode(labels, skip_special_tokens = False)

		# Remove: <BOS>, <EOS>, <PAD>
		labels = [ self._clean_output(text) for text in labels ]
		predic = [ self._clean_output(text) for text in predic ]

		# Parse: String -> Dict
		predic = [ self.seqreader.read(x) for x in predic ]
		labels = [ self.seqreader.read(x) for x in labels ]

		return { "predic": predic, "labels": labels }

	#--------------------------------------------#

	def validation_epoch_end(self, output):

		result = {
			"predic": [],
			"labels": []
		}

		for batch in output:
			for lab, pre in zip(batch["labels"], batch["predic"]):
				result["predic"].append(pre)
				result["labels"].append(lab)

		metrics = compute_metrics(result) 

		print(metrics)

		self.log('valid_P', metrics["P"])
		self.log('valid_R', metrics["R"])
		self.log('valid_F', metrics["F"])

	#--------------------------------------------#

	def _replace(self, tensor, val_1, val_2):
		return torch.where(tensor != val_1, tensor, val_2)

	#--------------------------------------------#

	def _clean_output(self, text):

		remove = [
			self.tokenizer.bos_token,
			self.tokenizer.eos_token,
			self.tokenizer.pad_token
		]

		for item in remove:
			text = text.replace(item, "")

		return text

	#--------------------------------------------#

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