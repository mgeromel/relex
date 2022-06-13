from collections import Counter

import torch

##################################################

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, data, tokenizer, decode_len = 64, encode_len = 256):
		
		phrases = [ x["phrases"] for x in data ]
		targets = [ x["targets"] for x in data ]

		self.data = { "phrases" : phrases , "targets" : targets }
		
		self.data = build_model_input(
			self.data,
			tokenizer,
			decode_len = decode_len,
			encode_len = encode_len
		)
		
		self.size = len(phrases)
		
	def __len__(self):
		return self.size

	def __getitem__(self, index):
		item = {}	
		for key in list(self.data.keys()):
			item[key] = self.data[key][index]
		return item

#------------------------------------------------#

def build_model_input(batch, tokenizer, decode_len = 64, encode_len = 256):
	
	if len(batch) == 0:
		return
		
	encoder_inputs = tokenizer(
		batch["phrases"],
		padding = "max_length",
		truncation = True,
		max_length = encode_len,
		return_tensors = "pt"
	)
	
	decoder_inputs = tokenizer(
		batch["targets"],
		padding = "max_length",
		truncation = True,
		max_length = decode_len,
		return_tensors = "pt"
	)
	
	batch["input_ids"] = encoder_inputs.input_ids
	batch["attention_mask"] = encoder_inputs.attention_mask
	batch["decoder_input_ids"] = decoder_inputs.input_ids
	batch["decoder_attention_mask"] = decoder_inputs.attention_mask
	batch["labels"] = decoder_inputs.input_ids
	
	del batch["phrases"]
	del batch["targets"]
	
	return batch

##################################################

def compute(predic, labels):
	result = list((Counter(predic) & Counter(labels)).elements())
	
	if (len(predic) == 0):
		return 1, 0
	
	precis = len(result) / len(predic)
	recall = len(result) / len(labels)

	return precis, recall

#------------------------------------------------#

def compute_metrics(pred):
	
	labels = pred["labels"]
	predic = pred["predicted"]
	
	#--------------------------------------------#
	
	recall = []
	precis = []
	
	for pre, lab in zip(predic, labels):
		
		pre = [str(x) for x in pre]
		lab = [str(x) for x in lab]
		
		p, r = compute(pre, lab)
		
		recall.append(r)
		precis.append(p)

	recall = sum(recall) / len(recall)
	precis = sum(precis) / len(precis)
	
	if precis + recall > 0:
		fscore = (2 * precis * recall) / (precis + recall)
	else:
		fscore = 0
	
	#--------------------------------------------#
	
	return {
		"R": recall, "P": precis, "F": fscore,
	}

##################################################