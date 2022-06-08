import torch

#-----------------------------------------------------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

amp_scale = True
#-----------------------------------------------------------#

def training(model = None, dataloader = None, tqdm_bar = None, optimizer = None, scheduler = None, ampscaler = None):
	sum_loss = 0
	
	counting = 0
	l_window = 100
	
	model.train()
	
	for batch in dataloader:
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
		
		if(counting % l_window == 0):
			loss = sum_loss/l_window
			rate = scheduler.get_last_lr()
			
			tqdm_bar.write(f"> avg_loss: {loss:.4f} ({l_window}) \t rate: {rate}")
			
			sum_loss = 0
			
#-----------------------------------------------------------#