import torch, numpy, os, random

from transformers import BertTokenizerFast
from collections import Counter
	
##################################################

# Reading Lines of File
def read_file(filename):
	lines = []
	
	with open(filename) as file:
		for line in file.readlines():
			lines.append(line.strip())
			
	return lines

##################################################

def extract(labels, sentence, vocab):
	result = []
	
	tables = labels.split(" || ")
	tables.sort(key = lambda x : x.split(" ;; ")[-1])
	
	for table in tables:
		
		lines = table.split(" ;; ")
		
		result.append( (3, vocab[lines[-1].strip()], -100, -100) )
		
		for line in lines[:-1]:
			
			tokens = line.split(" == ")
			
			if len(tokens) > 1: # DISTINCTION REQUIRED ?
				result.append( (4, vocab[tokens[0].strip()], -100, -100) ) 
			
			tokens = tokens[-1].split(" ^^ ")
			
			for token in tokens:
				result.append( (5, -100) + tuple(find(token, sentence)) )
	
	result = [ (1, -100, -100, -100) ] + result + [ (2, -100, -100, -100) ]
	
	return result

##################################################

def extract_results(results):
	all_entities = []
	all_table_id = []
	
	for labels in results:
		
		slot_labels = []
		tabs_labels = []
		
		for label in labels:
			temp = label.split(" ;; ")
			tabs_labels.append(temp[-1])

			temp = [ X.split(" == ")[1] for X in temp[:-1] ]
			
			slot = []
			for t in temp:
				slot.extend(t.split(" ^^ "))

			slot_labels.extend(slot)
		
		all_entities.append(slot_labels)
		all_table_id.append(tabs_labels)
		
	return all_table_id, all_entities

##################################################

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def find(word, sent, offset = 0):
	w_tokens = tokenizer(word, add_special_tokens = False).input_ids
	s_tokens = tokenizer(sent, add_special_tokens = True).input_ids
	
	# Find W_TOKENS in S_TOKENS 
	for i in range(len(s_tokens) - len(w_tokens)):
		if w_tokens == s_tokens[i : i + len(w_tokens)]:
			return (i + offset, i + offset + len(w_tokens))
	
	import IPython ; IPython.embed() ; exit(1)
	
	return (-1, -1)

##################################################

def compute(predic, labels):
	result = list((Counter(predic) & Counter(labels)).elements())
	
	if (len(predic) == 0):
		return 1, 0
	
	precis = len(result) / len(predic)
	recall = len(result) / len(labels)

	return precis, recall

#-----------------------------------------------------------#
	
def compute_metrics(pred):
	
	labels = pred["label_ids"]
	predic = pred["predicted"]
	
	micro_value = compute_scores(predic, labels)
	
	return {
		"R": micro_value[0],
		"P": micro_value[1],
		"F": micro_value[2],
	}

#-----------------------------------------------------------#

def compute_scores(predic, labels):
	recall = [0] * len(labels)
	precis = [0] * len(labels)
	
	for x in range(len(labels)):
		precis[x], recall[x] = compute(predic[x], labels[x])
		
	recall = sum(recall) / len(labels)
	precis = sum(precis) / len(labels)
	
	if precis + recall > 0:
		fscore = (2 * precis * recall) / (precis + recall)
	else:
		fscore = 0
		
	return (recall, precis, fscore)

##################################################