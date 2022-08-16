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
		self.vocab = model.vocab
		self.gramm = model.gramm

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
		loss = output.loss

		self.log("train_loss", loss)
		
		return loss

	#--------------------------------------------#