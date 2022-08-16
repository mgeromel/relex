import torch

from utils import *

#-----------------------------------------------------------#

class MyLoader():

	def __init__(self, name, vocabulary, tokenizer):
		# self._name = name
		# self._vmap = vocab
		# self._tkns = vocab

		self.name = name
		self.vocabulary = vocabulary
		self.tokenizer = tokenizer

	def name(self):
		return self.name

	def vocab(self):
		return self.vocab

	#--------------------------------------------#

	def load(self, path, filename):
		sentences = []
		relations = []
		
		#--------------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())

		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())

		sentences = [ self.clean(sent) for sent in sentences ]
		relations = [ self.parse(sent, rels) for sent, rels in zip(sentences, relations) ]

		#--------------------------------------------#
		
		return { "phrases" : sentences, "targets" : relations}

	#--------------------------------------------#

	def clean(self, text):
		tkns = self.tokenizer(text, add_special_tokens = False).input_ids
		text = self.tokenizer.decode(tkns, skip_special_tokens = True)

		return " ".join(text.split())

	#--------------------------------------------#

	def parse(self, text, labels):
		result = []
		
		#--------------------------------------#
		# EXTRACT ALL RELATIONS

		for table in labels.split(" || "):
			
			#----------------------------------#

			table_rows = table.split(" ;; ")
			table_head = self.vocab[table_rows[-1].strip()]

			result.append( (3, table_head, -100, -100) )
			
			#----------------------------------#
		
			for table_row in table_rows[:-1]:
				
				#------------------------------#
				# EXISTS ARG_CLASS ?
				
				if " == " in table_row:
					arg_class = table_row.split(" == ")[0].strip()
					arg_value = table_row.split(" == ")[1].strip()
					arguments = arg_value.split(" ^^ ")
					result.append( (4, self.vocab[arg_class], -100, -100) )
				else:
					arguments = table_row.split(" ^^ ")				
					result.append( (4, self.vocab["DEFAULT"], -100, -100) )
				
				#------------------------------#
				# ADD COLUMNS

				for argument in arguments:
					l_index, r_index = self.find(argument, sentence)
					result.append( (6, -100, l_index, r_index) )
				
				# ROW DONE
				result.append( (7, -100, -100, -100) )
				
				#------------------------------#

			# TABLE DONE
			result.append( (5, -100, -100, -100) )
			
			#----------------------------------#
		
		result = [ (1, -100, -100, -100) ] + result + [ (2, -100, -100, -100) ]
		
		#--------------------------------------#
		
		return result

	#--------------------------------------------#

	def find_word(self, word, sent):
		bounds = r'(\s|\b|^|$|[!\"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~])'
		search = re.search(bounds + re.escape(word) + bounds, sent)

		matched = search.group()
		
		# CHECK & FIX ERROR
		l_bound = search.span()[0] + matched.find(word)
		r_bound = l_bound + len(word)

		return l_bound, r_bound

	#--------------------------------------------#

	def find_tokens(self, l_bound, r_bound, offset_mapping):
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

		return l_index, r_index

	#--------------------------------------------#

	# TODO: !!!!
	def find(self, word, sent):
		#------------------------------#

		word = self.clean(word)
		
		tokenization = self.tokenizer(
			sent,
			return_offsets_mapping = True,
			add_special_tokens = True
		)
		sents_tokens = tokenization.input_ids

		offset_mapping = tokenization["offset_mapping"]

		#------------------------------#
		# FIND 'WORD' in 'SENT' via REGEX	
		l_bound, r_bound = find_word(word, sent)

		# FIND 'WORD'-TOKENS
		l_index, r_index = find_tokens(l_bound, r_bound, offset_mapping)
		
		# 'WORD' vs TOKENS 
		found = self.tokenizer.decode(
			sents_tokens[l_index: r_index + 1],
			skip_special_tokens = True
		)

		if found.strip() != word:
			print("MATCHING ERROR")
			import IPython ; IPython.embed() ; exit(1)
			pass
			
		return (l_index, r_index + 1)

		#------------------------------#



class MyDataset(torch.utils.data.Dataset):
	def __init__(self, file_name, vocab, tokenizer, decode_length = 64, encode_length = 256):
		
		print("> BUILDING DATA:", file_name)

		data_sent = read_file(file_name + ".sent")
		data_sent = [ clean(sent, tokenizer) for sent in data_sent ]
		data_tups = read_file(file_name + ".tup" )
		data_tups = [ extract(rel, sent, vocab, tokenizer) for sent, rel in zip(data_sent, data_tups) ]
		
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