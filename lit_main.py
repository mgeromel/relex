from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

import os, yaml, torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from gramm import *
from model import *
from utils import *

from lit_datamodule import *
from lit_model import *

#------------------------------------------------#

os.environ["TOKENIZERS_PARALLELISM"] = "true"

pl.seed_everything(0)

config = OmegaConf.load("config.yaml")

#------------------------------------------------#

def get_special_tokens(filename):
	vocab = read_file(filename)
	vocab = dict(zip(vocab, range(len(vocab))))

	return vocab

def get_grammar(filename):
	# TODO: rework GRAMMAR() -> Internal Map
	return GRAMMAR(filename)

#------------------------------------------------#

def main():

	# TODO: build transition matrix in grammar
	gramm = get_grammar("gramm_short.txt")
	vocab = get_special_tokens("data/CoNLL04/vocabulary.txt")

	#--------------------------------------------#
	# Tokenizer

	tokenizer = AutoTokenizer.from_pretrained(
		config.model_name,
		use_fast = True,
	)

	#--------------------------------------------#
	# Model
	
	model = TestModel(
		gramm = gramm,
		vocab = vocab,
		point_size = 256,
	)
	
	#--------------------------------------------#
	# PL-Modules

	datamodule = LitDataModule(config, model, vocab, tokenizer)
	lit_module = LitModule(config, model, tokenizer)

	#--------------------------------------------#
	# PL-Trainier

	trainer = pl.Trainer(
		accelerator = "cpu",
		devices = 1,
		accumulate_grad_batches = 4,
		gradient_clip_val = 10.0,
		max_epochs = 30,
		precision = 16,
		enable_checkpointing = False
	)
	
	trainer.fit(lit_module, datamodule = datamodule)

#------------------------------------------------#

# Invoke main()-Function
if __name__ == "__main__":
	main()
