import torch
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
		gen_tokens = self.model.generate(
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