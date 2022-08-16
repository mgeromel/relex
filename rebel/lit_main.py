from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

import os, yaml, torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from lit_module import *
from datamodule import *
from datas import load_dataset

#------------------------------------------------#

os.environ["TOKENIZERS_PARALLELISM"] = "true"

pl.seed_everything(0)

config = OmegaConf.load("config.yaml")

#------------------------------------------------#

def get_special_tokens():
    return load_dataset(config.dataset_name).tokens()

#------------------------------------------------#

def main():

	#--------------------------------------------#
	# Tokenizer + Special Tokens

	special_tokens = get_special_tokens()

	tokenizer = AutoTokenizer.from_pretrained(
		config.model_name,
		use_fast = True,
		additional_special_tokens = special_tokens
	)
	
	#--------------------------------------------#
	# Model-Config + Model
	
	model_config = AutoConfig.from_pretrained(
		config.model_name,
		decoder_start_token_id = 0,
		early_stopping = False,
		no_repeat_ngram_size = 0,
		forced_bos_token_id = None,
	)
	
	model = AutoModelForSeq2SeqLM.from_pretrained(
		config.model_name, config = model_config
	)
	
	model.resize_token_embeddings(len(tokenizer))
	
	#--------------------------------------------#
	# PL-Modules

	datamodule = LitDataModule(config, model, tokenizer)
	lit_module = LitModule(config, model, tokenizer)

	#--------------------------------------------#
	# PL-Trainier

	trainer = pl.Trainer(
		accelerator = "gpu",
		devices = 1,
		accumulate_grad_batches = 4,
		gradient_clip_val = 10.0,
		max_epochs = 20,
		precision = 16,
		enable_checkpointing = False
	)
	
	trainer.fit(lit_module, datamodule = datamodule)

#------------------------------------------------#

# Invoke main()-Function
if __name__ == "__main__":
	main()