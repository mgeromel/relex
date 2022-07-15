import string

from transformers import AlbertTokenizerFast
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

def extract(labels, sentence, vocab, tokenizer, strip = False):
	
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
			else:
				result.append( (4, vocab["DEFAULT"], -100, -100) )	
			
			tokens = tokens[-1].split(" ^^ ")
			
			for token in tokens:
				# REMOVE PUNCTUATION
				if strip:
					for sym in string.punctuation:
						token = token.replace(sym, " ")
					token = " ".join(token.split())
				
				result.append( (6, -100) + tuple(find(token, sentence, tokenizer)) )
			
			result.append( (7, -100, -100, -100) )
			
		result.append( (5, -100, -100, -100) )
	
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
words = []


def find(word, sent, tokenizer, offset = 0):
	
	global words
	
	sent = sent.lower()
	word = word.lower()
	
	s_tokens = tokenizer(sent, add_special_tokens = True).input_ids
	w_tokens = tokenizer(word, add_special_tokens = False).input_ids
	
	word = tokenizer.decode(w_tokens).strip(string.punctuation)
	
	#------------------------------#
	length = len(s_tokens)
	
	for w_len in range(1, length):
		for w_pos in range(length - w_len):
			
			span = tokenizer.decode(s_tokens[w_pos : w_pos + w_len])
			span = span.strip(string.punctuation).strip()
			
			if word == span:
				return (w_pos, w_pos + w_len)
	
	print("\nMATCHING ERROR\n")
	
	import IPython ; IPython.embed() ; exit(1)
	
	#------------------------------#
	
	return (0, 1)

##################################################

def compute(predic, labels):
	result = list((Counter(predic) & Counter(labels)).elements())
	
	if (len(predic) == 0):
		return 1, 0

	precis = len(result) / len(predic)
	recall = len(result) / len(labels)

	return precis, recall

#-----------------------------------------------------------#

def linearize(element):
	
	if type(element) is str:
		return element
	
	if type(element) is dict:
		result = ""
	
		for key in sorted(element.keys()):
			#result = result + key + " : "
			result = result + str(element[key]) + " ; "
		
		return result
	
	return str(element)

#-----------------------------------------------------------#

def compute_metrics(pred, debug = False):
	
	labels = pred["label_ids"]
	predic = pred["predicted"]
	
	#------------------------------#
	
	recall = []
	precis = []
	
	for pre, lab in zip(predic, labels):
		
		pre = [linearize(x) for x in pre]
		lab = [linearize(x) for x in lab]
		
		p, r = compute(pre, lab)
		
		recall.append(r)
		precis.append(p)

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