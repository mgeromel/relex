import torch

from utils import *
from gramm import *

#-----------------------------------------------------------#

def crop_list(vector, size, item):
	 return vector[:size] + [item] * (size - len(vector))
	
def build_model_input(batch, tokenizer = None, decoder_max_length = 64, encoder_max_length = 256):
	encoder_inputs = tokenizer(
		batch["phrases"],
		padding = "max_length",
		truncation = True,
		max_length = encoder_max_length
	)
	
	decoder_inputs = [
		crop_list(labels, decoder_max_length, (-100,-100,-100,-100)) for labels in batch["targets"]
	]
	
	keys = ["input_ids", "attention_mask", "decoder_attention_mask", "labels"]
	
	batch[keys[0]] = torch.LongTensor(encoder_inputs.input_ids)
	batch[keys[1]] = torch.CharTensor(encoder_inputs.attention_mask)
	batch[keys[2]] = torch.ShortTensor(
		[[ int( tup != (-100,-100,-100,-100) ) for tup in sample ] for sample in decoder_inputs ]
	)
	batch[keys[3]] = torch.HalfTensor(decoder_inputs)
	
	del batch["phrases"]
	del batch["targets"]
	
	return batch

#-----------------------------------------------------------#

class MyDataset(torch.utils.data.Dataset):
	def __init__(self, file_name, list_size, vocab, tokenizer, decoder_max_length = 64, encoder_max_length = 256):
		
		data_sent = read_file(file_name + ".sent")[:list_size]
		data_tups = read_file(file_name + ".tup" )[:list_size]	
		
		# REMOVE PUNCTIUATION
		translator = str.maketrans('', '', string.punctuation)
		
		for i, sent in enumerate(data_sent):
			data_sent[i] = sent.translate(translator)
		
		print(file_name)
		
		data_tups = [ extract(rel, sent, vocab, tokenizer) for sent, rel in zip(data_sent, data_tups) ]
		
		self.data = { "phrases" : data_sent , "targets" : data_tups }
		self.data = build_model_input(
			self.data,
			tokenizer = tokenizer,
			decoder_max_length = decoder_max_length,
			encoder_max_length = encoder_max_length
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
