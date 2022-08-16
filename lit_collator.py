class LitDataCollator:

	def __init__(self, tokenizer, model, pad_token_id = -100, max_length = None):
		self.tokenizer = tokenizer
		self.model = model
		self.max_length = max_length
		self.pad_token_id = pad_token_id

	def __call__(self, batch):
		
		# are 'labels' provided?
		labeled = any("labels" in sample for sample in batch)
		
		# 'labels' --> 'labels' + 'padding'
		if labeled:
			total_length = max(len(sample["labels"]) for sample in batch)

			for sample in batch:
				padding = [self.pad_token_id] * (total_length - len(sample["labels"]))
				sample["labels"] = sample["labels"][1:] + padding

		# 'batch' --> tokenizer('batch')
		batch = self.tokenizer.pad(
			batch,
			padding = True,
			# max_length = self.max_length,
			return_tensors = "pt",
		)

		# 'labels' --> 'decoder_input_ids'
		if (labeled and self.model):
			decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
			batch["decoder_input_ids"] = decoder_input_ids

		return batch