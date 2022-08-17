import re

class MyLoader():

	#--------------------------------------------#

	def __init__(self, dataset, vocabulary, tokenizer):
		self._name = dataset
		self._vocab = vocabulary
		self.tokenizer = tokenizer

	#--------------------------------------------#

	def load(self, path, filename):
		sentences = []
		relations = []
		
		#--------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())

		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())

		sentences = [ self.clean(sent) for sent in sentences ]
		relations = [ self.parse(sent, rels) for sent, rels in zip(sentences, relations) ]

		#--------------------------------------#

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
			table_head = self._vocab[table_rows[-1].strip()]

			result.append( (3, table_head, -100, -100) )
			
			#----------------------------------#
		
			for table_row in table_rows[:-1]:
				
				#------------------------------#
				# EXISTS ARG_CLASS ?
				
				if " == " in table_row:
					arg_class = table_row.split(" == ")[0].strip()
					arg_value = table_row.split(" == ")[1].strip()
					arguments = arg_value.split(" ^^ ")
					result.append( (4, self._vocab[arg_class], -100, -100) )
				else:
					arguments = table_row.split(" ^^ ")				
					result.append( (4, self._vocab["DEFAULT"], -100, -100) )
				
				#------------------------------#
				# ADD COLUMNS

				for argument in arguments:
					l_index, r_index = self.find(argument, text)
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

	def find(self, word, text):
		
		#------------------------------#

		word = self.clean(word)
		
		#------------------------------#
		
		tokenization = self.tokenizer(
			text,
			return_offsets_mapping = True,
			add_special_tokens = True
		)
		sents_tokens = tokenization.input_ids

		offset_mapping = tokenization["offset_mapping"]

		#------------------------------#

		# FIND 'WORD' in 'SENT' via REGEX	
		l_bound, r_bound = self.find_word(word, text)

		# FIND 'WORD'-TOKENS
		l_index, r_index = self.find_tokens(l_bound, r_bound, offset_mapping)
		
		# 'WORD' vs TOKENS 
		found = self.tokenizer.decode(
			sents_tokens[l_index: r_index + 1],
			skip_special_tokens = True
		)

		#------------------------------#

		if found.strip() != word:
			pass
			#print(f"{found} != {word}")
			
		return (l_index, r_index + 1)

	#--------------------------------------------#
	# CHAR-Bounds: 'Word' in 'Text'

	def find_word(self, word, text):
		bounds = r'(\s|\b|^|$|[!\"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~])'
		search = re.search(bounds + re.escape(word) + bounds, text)

		matched = search.group()
		
		# CHECK & FIX ERROR
		l_bound = search.span()[0] + matched.find(word)
		r_bound = l_bound + len(word)

		return l_bound, r_bound

	#--------------------------------------------#
	# TOKEN-Bounds: 'Word' in 'Text'
	
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