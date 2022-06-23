import transformers, torch, random, numpy
import gramm, loadr, tqdm
from utils import *
from modex import *

from transformers import AdamW, get_scheduler, BertTokenizerFast, RobertaTokenizerFast
from torch.utils.data import DataLoader

#-----------------------------------------------------------#

def set_seed(num):
	random.seed(num)
	
	torch.manual_seed(num)
	torch.cuda.manual_seed_all(num)
	torch.backends.cudnn.deterministic = True
	
	numpy.random.seed(num)
	
set_seed(1001)
	
#-----------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v1")

#-----------------------------------------------------------#

def write_results(batch, name = "DEFAULT"):
	
	with open(f"{name}_results.txt", "w") as file:
		for inputs, labels, predic in zip(batch["inputs"], batch["label_ids"], batch["predicted"]):
			file.write("INPUTS :: " + inputs + "\n")
			file.write("LABELS :: " + str(labels)[1:-1].replace("[", "").replace("]", "") + "\n")
			file.write("PREDIC :: " + str(predic)[1:-1].replace("[", "").replace("]", "") + "\n")
			file.write("\n")
			
def convert(values):
	
	# FLOAT -> INT
	values = values.int()
	
	# INITIALIZE
	g_list = [0] * gramm_size
	r_list = [0] * vocab_size
	p_list = [0] * point_size
	q_list = [0] * point_size
	
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

def reduce(tokens, logits):
	G = torch.argmax(logits[:gramm_size]).item()
	V = torch.argmax(logits[gramm_size : gramm_size + vocab_size]).item()
	P = (
		torch.argmax(logits[ -2*point_size : - point_size ]).item(),
		torch.argmax(logits[   -point_size :              ]).item()
	)
	P = tokenizer.decode(tokens[P[0] : P[1]].tolist()).strip()
	
	return (G, V, P)

#-----------------------------------------------------------#

def translate(input_ids, batch):
	
	result = []
	
	for tokens, output in zip(input_ids, batch):
		table_dicts = []
		
		table_dict = {}
		curr_entry = "DEFAULT"
		
		for logits in output:
			G, V, P = reduce(tokens, logits)
			
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
	
	return result
	
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

def training(model, dataset, tqdm_bar, epoch = -1):
	sum_loss = 0
	counting = 0
	l_window = 100
	
	model.train()
	
	for batch in dataset:	
		batch["decoder_input_ids"] = torch.Tensor(
			[[ convert(tup) for tup in sample ] for sample in batch["labels"]]
	   	).float()
		
		batch = {k: v.to(device) for k, v in batch.items()}
		
		optimizer.zero_grad()
		
		if use_amp:
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

def validate(model, dataset, tqdm_bar, epoch = -1, return_inputs = False):
	
	inputs = []
	labels = []
	predic = []
	
	#------------------------------------------------------#
	
	model.eval()
	for batch in dataset:
		
		batch["decoder_input_ids"] = torch.Tensor(
			[[ convert(tup) for tup in sample ] for sample in batch["labels"]]
		).float()

		with torch.no_grad():
			output = model.generate(
				input_ids = batch["input_ids"].to(device),
				max_length = 64
			).to("cpu")
		
		if return_inputs:
			for sequence in batch["input_ids"]:
				text = tokenizer.decode(sequence, skip_special_tokens = True)
				inputs.append(text)
				
		labels.extend(translate(batch["input_ids"], batch["decoder_input_ids"]))
		predic.extend(translate(batch["input_ids"], output))
		
		tqdm_bar.update(1)
	
	#------------------------------------------------------#
	
	result = {"label_ids": labels, "predicted": predic}
	
	if return_inputs: result["inputs"] = inputs
	
	return result

#-----------------------------------------------------------#

dataset = "ATIS" # point_size = 55, batch_size = 24
dataset = "NYT24" # point_size = 225, batch_size = 64
dataset = "NYT29" # point_size = 270, batch_size = 64
dataset = "ADE" # point_size = 115, batch_size = 16, lr = 5e-5
dataset = "RAMS/filtered_level_1" # point_size = 512, batch_size = 24
dataset = "MAVEN" # point_size = 340, batch_size = 24
dataset = "CoNLL04" # point_size = 145, batch_size = 4/8
dataset = "ADE/folds" # point_size = 115, batch_size = 16, lr = 5e-5
dataset = "SNIPS" # point_size = 40, batch_size = 24
dataset = "BiQuAD" # point_size = 100, batch_size = 64

# USE THE FOLLOWING:
dataset = "ATIS"

gramm = gramm.GRAMMAR("gramm.txt")
vocab = read_file("data/" + dataset + "/vocabulary.txt")
vocab = dict(zip(vocab, range(len(vocab))))
bacov = { i : s for s, i in vocab.items() }

gramm_size = gramm.size() + 1
vocab_size = len(vocab)
point_size = 55

#-----------------------------------------------------------#

EPOCHS = 50

train_size = 999999
valid_size = 999999
tests_size = 999999
batch_size = 16

use_amp = True
save_model = False

#-----------------------------------------------------------#
		
data_train = loadr.MyDataset("data/" + dataset + "/train", vocab, tokenizer, encode_length = point_size)
data_tests = loadr.MyDataset("data/" + dataset + "/test", vocab, tokenizer, encode_length = point_size)
#data_valid = loadr.MyDataset("data/" + dataset + "/valid", vocab, tokenizer, encode_length = point_size)

train_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
tests_loader = DataLoader(data_tests, batch_size = batch_size, shuffle = True)
#valid_loader = DataLoader(data_valid, batch_size = batch_size, shuffle = True)

#-----------------------------------------------------------# 

model = TestModel(gramm = gramm, vocab = vocab, point_size = point_size)
#model.load_state_dict(torch.load("models/model_snips_e50_b16.pt"))
model = model.to(device)

#-----------------------------------------------------------#

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
ampscaler = torch.cuda.amp.GradScaler()
scheduler = transformers.get_constant_schedule_with_warmup( # cosine <--> constant
	optimizer = optimizer,
	num_warmup_steps = 0.1 * EPOCHS * len(train_loader),
	#num_training_steps = EPOCHS * len(train_loader)
)
#-----------------------------------------------------------#

train_bar = tqdm.tqdm(total = EPOCHS * len(train_loader), leave = False, position = 0, desc = "TRAIN")
tests_bar = tqdm.tqdm(total = EPOCHS * len(tests_loader), leave = False, position = 2, desc = "TESTS")
#valid_bar = tqdm.tqdm(total = EPOCHS * len(valid_loader), leave = False, position = 1, desc = "VALID")

MAX_VALID_TOTAL_F1 = -1
MAX_TESTS_TOTAL_F1 = -1

MAX_VALID_TOTAL_PR = -1
MAX_TESTS_TOTAL_PR = -1

MAX_VALID_TOTAL_RE = -1
MAX_TESTS_TOTAL_RE = -1

MAX_VALID_SLOTS_F1 = -1
MAX_TESTS_SLOTS_F1 = -1

MAX_VALID_TABLE_F1 = -1
MAX_TESTS_TABLE_F1 = -1

################################################################################
	
for epoch in range(EPOCHS):
	
	#-----------------------------------------------------------#
	
	train_bar.write(f"TRAINING {epoch + 1}/{EPOCHS}: {dataset}")
	training(model, train_loader, train_bar)
	tests_bar.write(f"VALIDATE {epoch + 1}/{EPOCHS}:")
	
	#-----------------------------------------------------------#
	
	tests_result = validate(model, tests_loader, tests_bar, return_inputs = True)
	
	tests_tabs_labels, tests_slot_labels = extract_results(tests_result["label_ids"])
	tests_tabs_predic, tests_slot_predic = extract_results(tests_result["predicted"])
	
	tests_over_score = compute_metrics( tests_result )
	tests_tabs_score = compute_metrics({"label_ids": tests_tabs_labels, "predicted": tests_tabs_predic})
	tests_slot_score = compute_metrics({"label_ids": tests_slot_labels, "predicted": tests_slot_predic})
	
	if tests_over_score["F"] > MAX_TESTS_TOTAL_F1:
		MAX_TESTS_TOTAL_PR = tests_over_score["P"]
		MAX_TESTS_TOTAL_RE = tests_over_score["R"]
		MAX_TESTS_TOTAL_F1 = tests_over_score["F"]
		MAX_TESTS_TABLE_F1 = tests_tabs_score["F"]
		MAX_TESTS_SLOTS_F1 = tests_slot_score["F"]
	
		write_results(tests_result, name = dataset)
	
	tests_bar.write(f"MAX_TESTS_TOTAL_PR: {MAX_TESTS_TOTAL_PR}")
	tests_bar.write(f"MAX_TESTS_TOTAL_RE: {MAX_TESTS_TOTAL_RE}")
	tests_bar.write(f"MAX_TESTS_TOTAL_F1: {MAX_TESTS_TOTAL_F1}")
	tests_bar.write(".....")
	tests_bar.write(f"MAX_TESTS_TABLE_F1: {MAX_TESTS_TABLE_F1}")
	tests_bar.write(f"MAX_TESTS_SLOTS_F1: {MAX_TESTS_SLOTS_F1}")
	
	#-----------------------------------------------------------#
	
	if "valid_loader" not in locals():
		continue
		
	valid_result = validate(model, valid_loader, valid_bar)
		
	valid_tabs_labels, valid_slot_labels = extract_results(valid_result["label_ids"])
	valid_tabs_predic, valid_slot_predic = extract_results(valid_result["predicted"])

	valid_over_score = compute_metrics( valid_result )
	valid_tabs_score = compute_metrics({"label_ids": valid_tabs_labels, "predicted": valid_tabs_predic})
	valid_slot_score = compute_metrics({"label_ids": valid_slot_labels, "predicted": valid_slot_predic})

	if valid_over_score["F"] > MAX_VALID_TOTAL_F1:
		MAX_VALID_TOTAL_PR = valid_over_score["P"]
		MAX_VALID_TOTAL_RE = valid_over_score["R"]
		MAX_VALID_TOTAL_F1 = valid_over_score["F"]
		MAX_VALID_TABLE_F1 = valid_tabs_score["F"]
		MAX_VALID_SLOTS_F1 = valid_slot_score["F"]

	tests_bar.write("")

	tests_bar.write(f"MAX_VALID_TOTAL_PR: {MAX_VALID_TOTAL_PR}")
	tests_bar.write(f"MAX_VALID_TOTAL_RE: {MAX_VALID_TOTAL_RE}")
	tests_bar.write(f"MAX_VALID_TOTAL_F1: {MAX_VALID_TOTAL_F1}")
	tests_bar.write(".....")
	tests_bar.write(f"MAX_VALID_TABLE_F1: {MAX_VALID_TABLE_F1}")
	tests_bar.write(f"MAX_VALID_SLOTS_F1: {MAX_VALID_SLOTS_F1}")

	tests_bar.write("o" + "-----" * 6 + "o")

	#-----------------------------------------------------------#

import IPython ; IPython.embed() ; exit(1)
	
################################################################################

if save_model:
	torch.save(model.state_dict(), "models/model_snips_e50_b16.pt")
