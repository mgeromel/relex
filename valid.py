import torch

#-----------------------------------------------------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gramm_size = None
vocab_size = None
point_size = None

tokenizer = None
bacov = None
vocab = None

#-----------------------------------------------------------#

def _initialize(g_size, v_size, p_size, voc, tknzr):
	global gramm_size, vocab_size, point_size, bacov, tokenizer
	
	if (gramm_size and vocab_size and point_size and bacov and tokenizer) is None:
		gramm_size = g_size
		vocab_size = v_size
		point_size = p_size
		
		tokenizer = tknzr
		vocab = voc
		bacov = { i : s for s, i in vocab.items() }

#-----------------------------------------------------------#

def validate(
	model = None,
	dataloader = None,
	tqdm_bar = None,
	tokenizer = None,
	g_size = None,
	v_size = None,
	p_size = None,
	vocab = None,
	return_inputs = False,
	beam_search = False,
	num_beams = 1,
	debug = False,
):
	
	#------------------------------------------------------#
	
	_initialize(g_size, v_size, p_size, vocab, tokenizer)
	
	labels = []
	predic = []
	inputs = []
	
	#------------------------------------------------------#
	
	model.eval()
	for batch in dataloader:
		
		batch["decoder_input_ids"] = torch.Tensor(
			[[ convert(tup) for tup in sample ] for sample in batch["labels"]]
		).float()

		with torch.no_grad():
			if not beam_search:
				output = model.generate(
					input_ids = batch["input_ids"].to(device),
					max_length = 96
				).to("cpu")
			else:
				output = model.beam_search(
					input_ids = batch["input_ids"].to(device),
					max_length = 96,
					num_beams = num_beams,
				).to("cpu")

		if debug:
			import IPython ; IPython.embed() ; exit(1)

		labels.extend(translate(batch["input_ids"], batch["decoder_input_ids"]))
		predic.extend(translate(batch["input_ids"], output))

		if return_inputs:
			for sequence in batch["input_ids"]:
				text = tokenizer.decode(sequence, skip_special_tokens = True)
				inputs.append(text)
				
		tqdm_bar.update(1)
	
	#------------------------------------------------------#

	result = {"label_ids": labels, "predicted": predic}
	
	if return_inputs:
		result["inputs"] = inputs
	
	return result
	
#-----------------------------------------------------------#

def convert(values):
	
	# FLOAT -> INT
	values = values.int()
	
	# INITIALIZE
	g_list = [0] * gramm_size
	r_list = [0] * vocab_size
	p_list = [0] * point_size
	q_list = [0] * point_size
	
	# PADDING?
	g_list[0] = int(values[0] == -100)
	
	# SETTING VALUES
	if values[0] != -100:
		g_list[values[0]] = 1
	
	if values[1] != -100:
		r_list[values[1]] = 1
	
	if values[2] != -100:
		p_list[values[2]] = 1
	
	if values[3] != -100:
		q_list[values[3]] = 1
		
	return g_list + r_list + p_list + q_list

#-----------------------------------------------------------#

def reduce(tokens, logits):
	G = torch.argmax(logits[:gramm_size]).item()
	V = torch.argmax(logits[gramm_size : gramm_size + vocab_size]).item()
	P = (
		torch.argmax(logits[ -2*point_size : - point_size ]).item(),
		torch.argmax(logits[   -point_size :              ]).item()
	)
	PP = tokenizer.decode(tokens[P[0] : P[1]].tolist(), skip_special_tokens = True).strip()
	
	return (G, V, PP, P)

#-----------------------------------------------------------#

def translate(input_ids, batch, flag = False):
	
	result = []
	
	for tokens, output in zip(input_ids, batch):
		table_dicts = []
		
		table_dict = {}
		curr_entry = "DEFAULT"
		
		for logits in output:
			G, V, P, I = reduce(tokens, logits)
			
			if flag:
				print(f"G: {G}, V: {bacov[V]}, P: {P}, I: {I}")
			
			# STATE 2
			if G == 2:
				if table_dict != {}:
					table_dicts.append(table_dict)
				
				break
				
			# STATE 3
			if G == 3:
				table_dict["TABLE_ID"] = bacov[V]
				
			# STATE 4
			if G == 4:
				if "time_relative" in bacov[V]: 
					temp = bacov[V]
					temp = temp.replace("time_relative", "time")
					curr_entry = temp
				else:
					curr_entry = bacov[V]
			
			if G == 5:
				table_dicts.append(table_dict)
				table_dict = {}
				curr_entry = "DEFAULT"
			
			# STATE 5
			if G == 6:
				if curr_entry not in table_dict:
					table_dict[curr_entry] = []
				
				table_dict[curr_entry].append( P )
		
		result.append(table_dicts)
	
	# TRANSFORM TABLES
	final_result = []
	
	for table_dicts in result:
		
		temp_result = []
		for table_dict in table_dicts:
			
			dirty = False
			
			for key in table_dict.keys():
				if key != "TABLE_ID" and "" in table_dict[key]:
					dirty = True
			
			if not dirty and "TABLE_ID" in table_dict:
				temp_result.append(table_dict)

		final_result.append(temp_result)
		
	return final_result

#-----------------------------------------------------------#