import transformers, torch, random, numpy
import gramm, loadr, tqdm

from train import training
from valid import validate
from utils import *
from model import *

from transformers import AdamW, get_scheduler, AlbertTokenizerFast, BartTokenizerFast
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
#tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

#-----------------------------------------------------------#

def write_results(batch, name = "DEFAULT"):
	
	with open(f"{name}_results.txt", "w") as file:
		for inputs, labels, predic in zip(batch["inputs"], batch["label_ids"], batch["predicted"]):
			file.write("INPUTS :: " + inputs + "\n")
			file.write("LABELS :: " + str(labels)[1:-1] + "\n")
			
			if len(predic) > 0:
				file.write("PREDIC :: " + str(predic)[1:-1] + "\n")
			else:
				file.write("PREDIC :: {'TABLE_ID' : 'DEFAULT'}\n")
						   
			file.write("\n")

#-----------------------------------------------------------#

dataset = "ATIS" # point_size = 70, decode_length = 34, batch_size = 24
dataset = "NYT24" # point_size = 225, batch_size = 64
dataset = "NYT29" # point_size = 270, batch_size = 64
dataset = "ADE" # point_size = 115, decode_length = 74, batch_size = 16, lr = 5e-5
dataset = "RAMS/filtered_level_1" # point_size = 512, batch_size = 24
dataset = "MAVEN" # point_size = 340, batch_size = 24
dataset = "CoNLL04" # point_size = 145, batch_size = 4/8
dataset = "ADE/folds" # point_size = 115, batch_size = 16, lr = 5e-5
dataset = "SNIPS" # point_size = 50, decode_length = 25, batch_size = 24
dataset = "BiQuAD" # point_size = 100, batch_size = 64

# USE THE FOLLOWING:
dataset = "SNIPS"

gramm = gramm.GRAMMAR("gramm.txt")
vocab = read_file("data/" + dataset + "/vocabulary.txt")
vocab = dict(zip(vocab, range(len(vocab))))

gramm_size = gramm.size() + 1
vocab_size = len(vocab)

point_size = 50
decode_length = 30

#-----------------------------------------------------------#

EPOCHS = 50

batch_size = 16
save_model = False

#-----------------------------------------------------------#
		
data_train = loadr.MyDataset("data/" + dataset + "/train", vocab, tokenizer, encode_length = point_size, decode_length = decode_length, strip = False)
data_tests = loadr.MyDataset("data/" + dataset + "/test" , vocab, tokenizer, encode_length = point_size, decode_length = decode_length, strip = False)
#data_valid = loadr.MyDataset("data/" + dataset + "/valid", vocab, tokenizer, encode_length = point_size, strip = True)

train_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
tests_loader = DataLoader(data_tests, batch_size = batch_size, shuffle = True)
#valid_loader = DataLoader(data_valid, batch_size = batch_size, shuffle = True)

#-----------------------------------------------------------# 

model = TestModel(gramm = gramm, vocab = vocab, point_size = point_size)
model = model.to(device)
#model.load_state_dict(torch.load("models/model_snips_e50_b16.pt"))

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
	training(
		model = model,
		dataloader = train_loader,
		tqdm_bar = train_bar,
		optimizer = optimizer,
		scheduler = scheduler,
		ampscaler = ampscaler,
		g_size = gramm_size,
		v_size = vocab_size,
		p_size = point_size,
	)
	
	#-----------------------------------------------------------#
	
	tests_bar.write(f"VALIDATE {epoch + 1}/{EPOCHS}:")
	tests_result = validate(
		model = model,
		dataloader = tests_loader,
		tqdm_bar = tests_bar,
		tokenizer = tokenizer,
		g_size = gramm_size,
		v_size = vocab_size,
		p_size = point_size,
		vocab = vocab,
		return_inputs = True
	)
	
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

	valid_result = validate(
		model = model,
		dataloader = valid_loader,
		tqdm_bar = valid_bar,
		tokenizer = tokenizer,
		g_size = gramm_size,
		v_size = vocab_size,
		p_size = point_size,
		vocab = vocab,
	)
	
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
