import torch

#-----------------------------------------------------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------#

def validate(model = None, dataloader = None, tqdm_bar = None, tokenizer = None, extractor = None):
	
	labels = []
	predic = []
	
	#------------------------------------------------------#
	
	model.eval()
	for batch in dataloader:

		with torch.no_grad():
			output = model.generate(
				input_ids = batch["input_ids"].to(device),
				max_length = 64
			).to("cpu")
		
		for sequence in batch["labels"]:
			text = tokenizer.decode(sequence)
			labels.append(extractor.read(text))
		
		for sequence in output:
			text = tokenizer.decode(sequence)
			predic.append(extractor.read(text))
			
		tqdm_bar.update(1)
	
	#------------------------------------------------------#
	
	return {"labels": labels, "predicted": predic}

#-----------------------------------------------------------#