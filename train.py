import torch

#-----------------------------------------------------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
amp_scale = True

gramm_size = None
vocab_size = None
point_size = None
		
#-----------------------------------------------------------#

def _initialize(g_size, v_size, p_size):
	global gramm_size, vocab_size, point_size
	
	if (gramm_size and vocab_size and point_size) is None:
		gramm_size = g_size
		vocab_size = v_size
		point_size = p_size

#-----------------------------------------------------------#

def training(
	model = None,
	dataloader = None,
	tqdm_bar = None,
	optimizer = None,
	scheduler = None,
	ampscaler = None,
	g_size = None,
	v_size = None,
	p_size = None,
):
	
	#------------------------------------------------------#
	
	_initialize(g_size, v_size, p_size)
		
	sum_loss = 0
	
	counting = 0
	l_window = 100
	
	#------------------------------------------------------#
	
	model.train()
	for batch in dataloader:
		batch["decoder_input_ids"] = torch.Tensor(
			[[ convert(tup) for tup in sample ] for sample in batch["labels"]]
	   	).float()
		
		batch = {k: v.to(device) for k, v in batch.items()}
		
		optimizer.zero_grad()
		
		if amp_scale:
			with torch.cuda.amp.autocast():
				output = model.forward(**batch)
				
			ampscaler.scale(output["loss"]).backward()
			ampscaler.step(optimizer)
			ampscaler.update()
		else:
			output = model.forward(**batch)
			output["loss"].backward()
			optimizer.step()
        		
		scheduler.step()
		tqdm_bar.update(1)
		
		sum_loss = sum_loss + output["loss"].item()
		counting = counting + 1
		
		#--------------------------------------------------#
	
		if(counting % l_window == 0):
			loss = sum_loss/l_window
			rate = scheduler.get_last_lr()
			
			tqdm_bar.write(f"> avg_loss: {loss:.4f} ({l_window}) \t rate: {rate}")
			
			sum_loss = 0
			
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