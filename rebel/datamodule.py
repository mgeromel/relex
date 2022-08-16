import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from datas import *
from utils import *

from lit_collator import *

class LitDataModule(pl.LightningDataModule):

    #--------------------------------------------#

    def __init__(self, config, model, tokenizer):
        super().__init__()
        
        #----------------------------------------#

        self.config = config
        self.model = model # ~
        self.tokenizer = tokenizer

        self.data_collator = LitDataCollator(
            self.tokenizer, self.model, pad_token_id = -100
        )

        #----------------------------------------#

        # WHAT DATASET?
        if config.dataset_name == "ADE":
            self.dataloader = ADELoader()

        if config.dataset_name == "CoNLL04":
            self.dataloader = CONLL04Loader()

        if config.dataset_name == "NYT24":
            self.dataloader = NYT24Loader()

        if config.dataset_name == "ATIS":
            self.dataloader = ATISLoader()

        if config.dataset_name == "SNIPS":
            self.dataloader = SNIPSLoader()

        #----------------------------------------#

        # TRAIN-DATASET
        self.train_dataset = build_dataset(
            self.dataloader.load(config.files_path, "train"),
            self.tokenizer,
            encode_length = self.config.encode_length,
            decode_length = self.config.decode_length
        )
        
        # VALID-DATASET
        self.valid_dataset = build_dataset(
            self.dataloader.load(config.files_path, "valid"),
            self.tokenizer,
            encode_length = self.config.encode_length,
            decode_length = self.config.decode_length
        )

        # TESTS-DATASET
        self.tests_dataset = build_dataset(
            self.dataloader.load(config.files_path, "test"),
            self.tokenizer,
            encode_length = self.config.encode_length,
            decode_length = self.config.decode_length
        )

    #--------------------------------------------#

    def train_dataloader(self, *args, **kwargs):

        return DataLoader(
            self.train_dataset,
            batch_size = self.config.train_batch_size,
            num_workers = self.config.dataload_workers,
            collate_fn = self.data_collator,
            shuffle = True
        )

    #--------------------------------------------#

    def val_dataloader(self, *args, **kwargs):

        return DataLoader(
            self.valid_dataset,
            batch_size = self.config.valid_batch_size,
            num_workers = self.config.dataload_workers,
            collate_fn = self.data_collator,
            shuffle = False
        )

    #--------------------------------------------#

    def test_dataloader(self, *args, **kwargs):
        
        return DataLoader(
            self.tests_dataset,
            batch_size = self.config.tests_batch_size,
            num_workers = self.config.dataload_workers,
            collate_fn = self.data_collator,
            shuffle = False
        )

    #--------------------------------------------#