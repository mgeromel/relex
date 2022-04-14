import transformers, torch, random, numpy
import gramm, loadr, tqdm
from utils import *
from modex import *

from transformers import AdamW, get_scheduler, BertTokenizerFast
from torch.utils.data import DataLoader

def set_seed(num):
	random.seed(num)
	
	torch.manual_seed(num)
	torch.cuda.manual_seed_all(num)
	torch.backends.cudnn.deterministic = True
	
	numpy.random.seed(num)
	
set_seed(1001)
	
#-----------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

#-----------------------------------------------------------#

def training(model, dataset, tqdm_bar):
	global optimizer, scheduler, ampscaler
	
	sum_loss = 0
	counting = 0
	
	model.train()
	for batch in dataset:
		batch["decoder_input_ids"] = torch.Tensor(
			[[ convert(tup, r_size = len(vocab)) for tup in sample ] for sample in batch["labels"]]
	   	).float()
		
		batch = {k: v.to(device) for k, v in batch.items()}
		
		with torch.cuda.amp.autocast():
			output = model.forward(**batch)
		
		batch = {k: v.to("cpu") for k, v in batch.items()}
		
		optimizer.zero_grad()
		
		ampscaler.scale(output["loss"]).backward()
		ampscaler.step(optimizer)
		ampscaler.update()
		
		scheduler.step()
		
		tqdm_bar.update(1)
		
		sum_loss = sum_loss + output["loss"].item()
		counting = counting + 1
		
		if(counting % 100 == 0):
			loss = round(sum_loss/100, 4)
			rate = scheduler.get_last_lr()
			tqdm_bar.write(f"\tavg_loss: {loss} \t rate: {rate}")
			sum_loss = 0
			
#-----------------------------------------------------------#

def validate(model, dataset, tqdm_bar):
	
	labels = []
	predic = []
	
	model.eval()
	for batch in dataset:
		
		batch["decoder_input_ids"] = torch.Tensor(
			[[ convert(tup, r_size = len(vocab)) for tup in sample ] for sample in batch["labels"]]
		).float()

		with torch.no_grad():
			output = model.generate(
				input_ids = batch["input_ids"].to(device),
				max_length = 64
			).to("cpu")
		
		labels.extend(translate(batch["input_ids"], batch["decoder_input_ids"]))
		predic.extend(translate(batch["input_ids"], output))
		
		tqdm_bar.update(1)
	
	#------------------------------------------------------#
	
	tqdm_bar.write(f"VALID {epoch + 1}/{EPOCHS}: DONE")
	
	# COMPUTE: STRICT F_SCORE 
	total_scores = compute_metrics({"label_ids": labels, "predicted": predic})
	tqdm_bar.write(f"total_score: {str(total_scores)}")
	
	# COMPUTE: F_SCORE (RELATIONS)  
	labels_r = [[ x[0] for x in sample ] for sample in labels]
	predic_r = [[ x[0] for x in sample ] for sample in predic]
	
	relat_scores = compute_metrics({"label_ids": labels_r, "predicted": predic_r})
	tqdm_bar.write(f"RELATION: {str(relat_scores)}")
	
	# COMPUTE: F_SCORE (ENTITIES)
	labels_p = [[ x[1:] for x in sample ] for sample in labels]
	predic_p = [[ x[1:] for x in sample ] for sample in predic]
	
	point_scores = compute_metrics({"label_ids": labels_p, "predicted": predic_p})
	tqdm_bar.write(f"ENTITIES: {str(point_scores)}")
	
	for i in range(15): tqdm_bar.write(f"\tlabels: {labels[i]}\n\tpredic: {predic[i]}\n")

#-----------------------------------------------------------#

def translate(input_ids, batch):
	
	result = []
	
	# batch_size x max_length x vocab_size
	
	for tokens, sample in zip(input_ids, batch):
		
		labels = []
		
		R, A, B = None, None, None
		
		for logits in sample:
			G = torch.argmax(logits[:9])
			
			if G == 4 and R is None:
				R = torch.argmax(logits[9 : 9 + 29]).item()
			
			if G == 5 and A is None:
				A = (
					torch.argmax(logits[-512:-256]).item(),
					torch.argmax(logits[-256:    ]).item()
				)
				A = tokenizer.decode(tokens[A[0] : A[1]].tolist())
			
			if G == 6 and B is None:
				B = (
					torch.argmax(logits[-512:-256]).item(),
					torch.argmax(logits[-256:    ]).item()
				)
				B = tokenizer.decode(tokens[B[0] : B[1]].tolist())
			
			if (G == 2 or G == 3) and not (A in ["", None] or B in ["", None] or R is None):
				labels.append( (R, A, B) )
				R, A, B = None, None, None
			
		result.append(labels)
		
	return result

#-----------------------------------------------------------#

def convert(values, g_size = 9, r_size = 29, p_size = 256):
	
	# FLOAT -> INT
	values = values.int()
	
	# INITIALIZE
	g_list = [0] * g_size
	r_list = [0] * r_size
	p_list = [0] * p_size
	q_list = [0] * p_size
	
	# PADDING
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

gramm = gramm.GRAMMAR("gramm.txt")
vocab = read_file("data/NYT29/relations.txt")
vocab = dict(zip(vocab, range(len(vocab))))

#-----------------------------------------------------------#

EPOCHS = 10

train_size = 99999
valid_size = 99999
tests_size = 99999
batch_size = 64

#-----------------------------------------------------------#

data_train = loadr.MyDataset("data/NYT29/train", train_size, gramm, vocab)
data_valid = loadr.MyDataset("data/NYT29/valid", valid_size, gramm, vocab)
data_tests = loadr.MyDataset( "data/NYT29/test", tests_size, gramm, vocab)

train_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(data_valid, batch_size = batch_size)
tests_loader = DataLoader(data_tests, batch_size = batch_size)

#-----------------------------------------------------------# 

model = TestModel(gramm = gramm, vocab = vocab)
#model.load_state_dict(torch.load("models/relex_base12x12_E5-F0-B52.pt"))

model.encoder.encoder.gradient_checkpointing = True
model.decoder.bert.encoder.gradient_checkpointing = True

for layer in model.encoder.encoder.layer[:4]:
	for param in layer.parameters():
		param.requires_grad = False

model = model.to(device)

#-----------------------------------------------------------#

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)

ampscaler = torch.cuda.amp.GradScaler()

scheduler = transformers.get_cosine_schedule_with_warmup(
	optimizer = optimizer,
	num_warmup_steps = len(train_loader) * 1/5,
	num_training_steps = EPOCHS * len(train_loader)
)

# NYT29, M12x10: VALID: 37.38, TESTS: 31.82

#-----------------------------------------------------------#

train_bar = tqdm.tqdm(total = EPOCHS * len(train_loader), leave = False, position = 0, desc = "TRAIN")
valid_bar = tqdm.tqdm(total = EPOCHS * len(valid_loader), leave = False, position = 1, desc = "VALID")
tests_bar = tqdm.tqdm(total = EPOCHS * len(tests_loader), leave = False, position = 2, desc = "TESTS")

#-----------------------------------------------------------#
	
for epoch in range(EPOCHS):
	
	training(model, train_loader, train_bar)
	train_bar.write(f"EPOCH {epoch + 1}/{EPOCHS}: DONE\n\n")
	
	validate(model, valid_loader, valid_bar)
	valid_bar.write(f"EPOCH {epoch + 1}/{EPOCHS}: DONE\n\n")
	
	validate(model, tests_loader, tests_bar)
	tests_bar.write(f"EPOCH {epoch + 1}/{EPOCHS}: DONE\n\n")
	
	break
#-----------------------------------------------------------#

print("\n" * 3)

import IPython ; IPython.embed() ; exit(1)

torch.save(model.state_dict(), "models/relex_base12x12_E10-F0-B52.pt")
