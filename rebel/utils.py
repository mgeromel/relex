from collections import Counter

import torch, pyarrow, datasets

##################################################

def build_dataset(data, tokenizer, decode_length = 128, encode_length = 256):

	phrases = [ x["phrases"] for x in data ]
	targets = [ x["targets"] for x in data ]

	encoder_inputs = tokenizer(
		phrases,
		truncation = True,
		max_length = encode_length,
	)
	
	decoder_inputs = tokenizer(
		targets,
		truncation = True,
		max_length = decode_length,
	)

	dataset = {
		"input_ids": encoder_inputs.input_ids,
		"attention_mask": encoder_inputs.attention_mask,
		"labels": decoder_inputs.input_ids
	}
	
	dataset = pyarrow.Table.from_pydict(dataset)
	dataset = datasets.Dataset(dataset)

	return dataset

#------------------------------------------------#

def build_model_input(batch, tokenizer, decode_len, encode_len):

	encoder_inputs = tokenizer(
		batch["phrases"],
		truncation = True,
		max_length = encode_len,
	)
	
	decoder_inputs = tokenizer(
		batch["targets"],
		truncation = True,
		max_length = decode_len,
	)
	
	batch["input_ids"] = encoder_inputs.input_ids
	batch["attention_mask"] = encoder_inputs.attention_mask
	batch["labels"] = decoder_inputs.input_ids

	batch.pop("phrases")
	batch.pop("targets")
	
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
	predic = pred["predic"]
	
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