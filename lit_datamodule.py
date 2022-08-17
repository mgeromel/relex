import pytorch_lightning as pl

import pyarrow, datasets

from torch.utils.data import DataLoader

from lit_collator import *
from lit_loadr import *

class LitDataModule(pl.LightningDataModule):

    #--------------------------------------------#

    def __init__(self, config, model, vocab, tokenizer):
        super().__init__()
        
        #----------------------------------------#

        self.config = config
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.data_collator = LitDataCollator(
            self.tokenizer, self.model, pad_token_id = -100
        )

        #----------------------------------------#

        self.dataloader = MyLoader(
            self.config.dataset_name,
            self.vocab,
            self.tokenizer
        )

        #----------------------------------------#

        # TRAIN-DATASET
        self.train_dataset = self.build_dataset(
            self.dataloader.load(config.files_path, "train"),
            encode_length = self.config.encode_length
        )
        
        # VALID-DATASET
        self.valid_dataset = self.build_dataset(
            self.dataloader.load(config.files_path, "valid"),
            encode_length = self.config.encode_length
        )

        # TESTS-DATASET
        self.tests_dataset = self.build_dataset(
            self.dataloader.load(config.files_path, "test"),
            encode_length = self.config.encode_length
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

    def build_dataset(self, data, encode_length = 256):

        encoder_inputs = self.tokenizer(
            data["phrases"],
            truncation = True,
            max_length = encode_length,
        )

        dataset = {
            "input_ids": encoder_inputs.input_ids,
            "attention_mask": encoder_inputs.attention_mask,
            "labels": data["targets"]
        }
        
        dataset = pyarrow.Table.from_pydict(dataset)
        dataset = datasets.Dataset(dataset)

        return dataset