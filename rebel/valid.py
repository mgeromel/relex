import torch

#-----------------------------------------------------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------#

def validate(model = None, dataloader = None, tqdm_bar = None, tokenizer = None, extractor = None, return_inputs = False):
	
	labels = []
	predic = []
	inputs = []
	
	#------------------------------------------------------#
	
	model.eval()
	for batch in dataloader:

		with torch.no_grad():
			output = model.generate(
				input_ids = batch["input_ids"].to(device),
				max_length = 64
			).to("cpu")
		
		if return_inputs:
			for sequence in batch["input_ids"]:
				text = tokenizer.decode(sequence, skip_special_tokens = True)
				inputs.append(text)
		
		for sequence in batch["labels"]:
			text = tokenizer.decode(sequence, skip_special_tokens = True)
			labels.append(extractor.read(text))
		
		for sequence in output:
			text = tokenizer.decode(sequence, skip_special_tokens = True)
			predic.append(extractor.read(text))
			
		tqdm_bar.update(1)
	
	#------------------------------------------------------#

	result = {"labels": labels, "predicted": predic}
	
	if return_inputs: result["inputs"] = inputs
	
	return result
	
#-----------------------------------------------------------#