import torch, numpy, os, random, string

from transformers import BertTokenizerFast, AlbertTokenizerFast
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
	
	translator = str.maketrans('', '', string.punctuation)
	
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
				token = token.translate(translator)
				result.append( (5, -100) + tuple(find(token, sentence)) )
	
	result = [ (1, -100, -100, -100) ] + result + [ (2, -100, -100, -100) ]
	
	return result

##################################################

def extract_results(results):
	all_entities = []
	all_table_id = []
	
	for table_dicts in results:
		
		slot_labels = []
		tabs_labels = []
		
		for table_dict in table_dicts:
			tabs_labels.append(table_dict["TABLE_ID"])
			
			for key in table_dict:
				if key != "TABLE_ID":
					slot_labels.extend(table_dict[key])
		
		all_entities.append(slot_labels)
		all_table_id.append(tabs_labels)
		
	return all_table_id, all_entities

##################################################

#tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v1")

def find(word, sent, offset = 0):
	sent = sent.replace(word, " <pad> " + word + " <pad> ", 1) 
	
	sent = sent.lower()
	word = word.lower()
	
	w_tokens = tokenizer(word, add_special_tokens = False).input_ids
	s_tokens = tokenizer(sent, add_special_tokens = True ).input_ids
	
	word = tokenizer.decode(w_tokens)
	
	#------------------------------#
	
	l_bound = 0
	r_bound = len(s_tokens)
	
	#------------------------------#
	
	# LEFT BOUND 
	for idx in range(r_bound):
		if s_tokens[idx] == 0:
			l_bound = idx + 1
			break
		
	#------------------------------#
	
	# RIGHT BOUND
	for idx in range(l_bound, len(s_tokens)+1):
		if s_tokens[idx] == 0:
			r_bound = idx
			break
	
	if tokenizer.decode(s_tokens[l_bound : r_bound]) != word:
		print("Matching Error.")
		import IPython ; IPython.embed() ; exit(1)
	
	#------------------------------#
	
	return (l_bound - 1, r_bound - 1)

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
	
	#------------------------------#
	
	recall = []
	precis = []
	
	for pre, lab in zip(predic, labels):
		pre = [str(x) for x in pre]
		lab = [str(x) for x in lab]
		
		pre, rec = compute(pre, lab)
		
		recall.append(rec)
		precis.append(pre)

	recall = sum(recall) / len(recall)
	precis = sum(precis) / len(precis)
	
	if precis + recall > 0:
		fscore = (2 * precis * recall) / (precis + recall)
	else:
		fscore = 0
	
	#------------------------------#
	
	return {
		"R": recall, "P": precis, "F": fscore,
	}

##################################################