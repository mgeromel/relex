import torch

#-----------------------------------------------------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------------------#

def extract_ade_tuples(text):
	tuples = []
	
	text = text.strip()
	
	head, tail = "", ""
	mode = "DEFAULT"
	
	for token in text.split():
		
		# STATE 1
		if token == "<trip>":
			mode = "READ_HEAD"
			head = ""
			tail = ""
		
		# STATE 2
		elif token == "<head>":
			mode = "READ_TAIL"
			tail = ""
		
		# STATE 3
		elif token == "<tail>":
			
			if head != "" and tail != "":
				tuples.append({'head': head.strip(), 'tail': tail.strip()})
			
			tail = ""
		# STATE ADD
		else:
			if mode == "READ_HEAD":
				head = head + " " + token
				
			elif mode == "READ_TAIL":
				tail = tail + " " + token
	
	return tuples

#-----------------------------------------------------------#

def validate(model = None, dataloader = None, tqdm_bar = None, tokenizer = None):
	
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
			labels.append(extract_ade_tuples(text))
		
		for sequence in output:
			text = tokenizer.decode(sequence)
			predic.append(extract_ade_tuples(text))
			
		tqdm_bar.update(1)
	
	#------------------------------------------------------#
	
	return {"labels": labels, "predicted": predic}

#-----------------------------------------------------------#