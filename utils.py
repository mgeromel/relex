import string, re

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

def extract(labels, sentence, vocab, tokenizer):
	
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

def clean(text, tokenizer):
	tkns = tokenizer(text, add_special_tokens = False).input_ids
	text = tokenizer.decode( tkns, skip_special_tokens = True)

	return " ".join(text.split())

def find(word, sent, tokenizer):
	word = clean(word, tokenizer)
	
	tokenization = tokenizer(sent, return_offsets_mapping = True, add_special_tokens = True)	
	sents_tokens = tokenization.input_ids

	offset_mapping = tokenization["offset_mapping"]

	#------------------------------#
	# FIND 'WORD' in 'SENT' via REGEX
	bounds = r'(\s|\b|^|$|[!\"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~])'
	search = re.search(bounds + re.escape(word) + bounds, sent)

	matched = search.group()
	l_bound = search.span()[0] + matched.find(word)
	r_bound = l_bound + len(word)
	#------------------------------#
	# FIND 'WORD'-TOKENS
	l_index = len(offset_mapping) - 2
	r_index = 0

	# FIND L_BOUND
	for index, offset in enumerate(offset_mapping):
		if l_bound < offset[0]: 
			l_index = index - 1
			break

	# FIND R_INDEX
	for index, offset in reversed(list(enumerate(offset_mapping[:-1]))):
		if r_bound >= offset[1]:
			r_index = index
			break
	#------------------------------#
	# 'WORD' vs TOKENS 
	
	found = tokenizer.decode(sents_tokens[l_index: r_index + 1], skip_special_tokens = True)

	if found.strip() != word:
		print("\n > MATCHING ERROR < \n")
		import IPython ; IPython.embed() ; exit(1)
	
	return(l_index, r_index + 1)

	#------------------------------#

##################################################

def compute(predic, labels):
	result = list((Counter(predic) & Counter(labels)).elements())
	
	if len(predic) == 0:
		return 1, 0

	if len(labels) == 0:
		print(predic)
	
		import IPython ; IPython.embed() ; exit(1)

		return 0, 1

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
			result = result + key + " : "
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