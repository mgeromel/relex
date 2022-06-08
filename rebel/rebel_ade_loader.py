import datasets

class ADELoader():
	
	def __init__(self):
		dataset = datasets.load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")
		dataset = dataset["train"]
		
		self.dataset = dataset
		self._name = "ADE"

	#------------------------------------------------#
	
	def name(self):
		return self._name

	#------------------------------------------------#

	def load(self, sort = False):
		text_dict = {}
	
		#--------------------------------------------#
		
		# 1. Collect Drug-Effect Clusters
		for text, drug, effect in zip(self.dataset["text"], self.dataset["drug"], self.dataset["effect"]):
			if text not in text_dict:
				text_dict[text] = []

			entry = (drug, effect)

			# REMOVE DUPLICATES ~ 2.77% of 6821
			if entry not in text_dict[text]:
				text_dict[text].append( entry )

		result = []
		
		#--------------------------------------------#
		
		# 2. Linearize Drug-Effect Clusters		
		for text, triples in text_dict.items():

			#----------------------------------------#
			
			drug_dict = {}
			for triple in triples:

				if triple[0] not in drug_dict:
					drug_dict[triple[0]] = []

				drug_dict[triple[0]].append(triple[1])

			#----------------------------------------#
			# Sorting by Position in the Soruec Text #
			
			temp_list = []
			
			for drug, effects in drug_dict.items():
				effects.sort(key = lambda x : text.find(x) )
				temp_list.append( (drug, effects) )
			
			temp_list.sort(key = lambda x : text.find(x[0]) )
			
			# ALTERNATIVE
			
			#temp_list = []
			#
			#for drug, effects in drug_dict.items():
			#	effects.sort(key = lambda x : x)
			#	temp_list.append((drug, effects))
			#
			#temp_list.sort(key = lambda x : x[0])
				
			#----------------------------------------#
			
			labels = ""

			for drug, effects in temp_list:
				head = f"<trip> {drug} <head> "
				tail = " <tail> ".join(effects) + " <tail> "
				labels = labels + head + tail

			labels = labels.strip()

			result.append( {"phrases" : text, "targets" : labels} )

			#----------------------------------------#
			
		return result