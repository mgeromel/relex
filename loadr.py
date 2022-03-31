import pandas, torch, numpy, datasets

from torch.utils.data import Dataset
from utils import *
from gramm import *

#-----------------------------------------------------------#

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

encoder_max_length = 256
decoder_max_length = 64

def crop_list(vector, size, item):
     return vector[:size] + [item] * (size - len(vector))
    
def build_model_input(batch):
    encoder_inputs = tokenizer(
        batch["phrases"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length
    )
    
    decoder_inputs = [
        crop_list(labels, decoder_max_length, (-100,-100,-100,-100)) for labels in batch["targets"]
    ]
    
    keys = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]
    
    batch[keys[0]] = torch.LongTensor(encoder_inputs.input_ids)
    batch[keys[1]] = torch.LongTensor(encoder_inputs.attention_mask)
    
    #batch[keys[2]] = torch.FloatTensor(
    #    [[ convert(tup) for tup in sample ] for sample in decoder_inputs]
    #)
    
    batch[keys[3]] = torch.LongTensor(
        [[ int( tup != (-100,-100,-100,-100) ) for tup in sample ] for sample in decoder_inputs ]
    )
    batch[keys[4]] = torch.FloatTensor(decoder_inputs)
    
    del batch["phrases"]
    del batch["targets"]
    
    return batch

#-----------------------------------------------------------#

class MyDataset(Dataset):
    def __init__(self, file_name, list_size, gramm, vocab):
        
        data_sent = read_file(file_name + ".sent")[:list_size]
        data_tups = read_file(file_name + ".tup" )[:list_size]
        data_tups = [ extract(rel, sent, gramm, vocab) for sent, rel in zip(data_sent, data_tups) ]
        
        self.data = { "phrases" : data_sent , "targets" : data_tups }
        self.data = build_model_input(self.data)
        self.size = len(data_tups)
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = {}
        
        for key in list(self.data.keys()):
            item[key] = self.data[key][index]
        
        return item

#-----------------------------------------------------------#
