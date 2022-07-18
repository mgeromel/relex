import torch

from utils import *
from gramm import *

#-----------------------------------------------------------#

def crop_list(vector, size, item):
	 return vector[:size] + [item] * (size - len(vector))
	
def build_model_input(batch, tokenizer = None, decode_length = 64, encode_length = 256):
	encoder_inputs = tokenizer(
		batch["phrases"],
		padding = "max_length",
		truncation = True,
		max_length = encode_length
	)
	
	decoder_inputs = [
		crop_list(labels, decode_length, (-100,-100,-100,-100)) for labels in batch["targets"]
	]
	
	keys = ["input_ids", "attention_mask", "decoder_attention_mask", "labels"]
	
	batch[keys[0]] = torch.IntTensor(encoder_inputs.input_ids)
	batch[keys[1]] = torch.IntTensor(encoder_inputs.attention_mask)
	batch[keys[2]] = torch.IntTensor(
		[[ int( tup != (-100,-100,-100,-100) ) for tup in sample ] for sample in decoder_inputs ]
	)
	batch[keys[3]] = torch.IntTensor(decoder_inputs)
	
	del batch["phrases"]
	del batch["targets"]
	
	return batch

#-----------------------------------------------------------#

class MyDataset(torch.utils.data.Dataset):
	def __init__(self, file_name, vocab, tokenizer, decode_length = 64, encode_length = 256, strip = False):
		
		data_sent = read_file(file_name + ".sent")
		data_tups = read_file(file_name + ".tup" )
		
		print("> BUILDING DATA:", file_name)

		data_tups = [ extract(rel, sent, vocab, tokenizer, strip = strip) for sent, rel in zip(data_sent, data_tups) ]
		
		self.data = { "phrases" : data_sent , "targets" : data_tups }
		self.data = build_model_input(
			self.data,
			tokenizer = tokenizer,
			decode_length = decode_length,
			encode_length = encode_length
		)
		self.size = len(data_tups)
		
	def __len__(self):
		return self.size

	def __getitem__(self, index):
		item = {}
		
		for key in list(self.data.keys()):
			item[key] = self.data[key][index]
		
		return item

#-----------------------------------------------------------#
