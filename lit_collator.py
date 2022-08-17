import torch

class LitDataCollator:

	def __init__(self, tokenizer, model, pad_token_id = -100, max_length = None):
		self.tokenizer = tokenizer
		self.model = model
		self.max_length = max_length
		self.pad_token_id = pad_token_id

	#--------------------------------------------#

	def __call__(self, batch):

		padding_element = [self.pad_token_id] * 4

		#--------------------------------------#
		# are 'labels' provided?

		labeled = any("labels" in sample for sample in batch)

		#--------------------------------------#
		# 'labels' --> 'labels' + 'padding'
		
		if labeled:
			total_length = max(len(sample["labels"]) for sample in batch)

			for sample in batch:
				padding = [padding_element] * (total_length - len(sample["labels"]))
				sample["labels"] = sample["labels"] + padding

		#--------------------------------------#
		# 'batch' --> tokenizer('batch')
		
		batch = self.tokenizer.pad(
			batch, padding = True,
			return_tensors = "pt",
		)

		#--------------------------------------#
		# 'labels' --> 'decoder_input_ids'
		
		if (labeled and self.model):
			# create & explode: 'decoder_input_ids'
			batch["decoder_input_ids"] = self.blowup(batch["labels"])

			# shifting 'labels'
			batch["labels"][:,:-1] = batch["labels"][:, 1:]
			batch["labels"][:, -1] = torch.Tensor(padding_element)

		#--------------------------------------#		

		return batch

	#--------------------------------------------#

	def blowup(self, tensor):
		
		#--------------------------------------#
		# Dimensions

		dimensions = self.model.get_dimensions()

		gramm_size = dimensions["gramm_size"]
		vocab_size = dimensions["vocab_size"]
		point_size = dimensions["point_size"]
		
		#--------------------------------------#
		# One-Hot Encoding

		gramm_part = self.get_one_hot(tensor[:,:,0], gramm_size, masking = False)
		vocab_part = self.get_one_hot(tensor[:,:,1], vocab_size, masking = True)
		point_part = self.get_one_hot(tensor[:,:,2], point_size, masking = True)
		qoint_part = self.get_one_hot(tensor[:,:,3], point_size, masking = True)
		
		#--------------------------------------#

		result = torch.cat([gramm_part, vocab_part, point_part, qoint_part], dim = -1)

		return result

	#--------------------------------------------#

	def get_one_hot(self, tensor, size, masking = False):
		mask = tensor.ne(self.pad_token_id)
		idxs = torch.where(tensor != self.pad_token_id, tensor, 0)
		
		result = torch.nn.functional.one_hot(idxs, size)

		if masking:
			result = result * mask[:, :, None]

		return result

	#--------------------------------------------#