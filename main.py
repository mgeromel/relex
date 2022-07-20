import transformers, torch, random, numpy
import gramm, loadr, tqdm

from train import training
from valid import validate
from utils import *
from model import *

from transformers import AdamW, get_scheduler, AlbertTokenizerFast
from torch.utils.data import DataLoader

#-----------------------------------------------------------#

def set_seed(num):
	random.seed(num)
	
	torch.manual_seed(num)
	torch.cuda.manual_seed_all(num)
	torch.backends.cudnn.deterministic = True
	
	numpy.random.seed(num)

#-----------------------------------------------------------#

def write_results(batch, name = "DEFAULT"):
	with open(name, "w") as file:
		for inputs, labels, predic in zip(batch["inputs"], batch["label_ids"], batch["predicted"]):
			file.write("INPUTS :: " + inputs + "\n")
			file.write("LABELS :: " + str(labels)[1:-1] + "\n")
			
			if len(predic) > 0:
				file.write("PREDIC :: " + str(predic)[1:-1] + "\n")
			else:
				file.write("PREDIC :: {'TABLE_ID' : 'DEFAULT'}\n")
						   
			file.write("\n")

#-----------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AlbertTokenizerFast.from_pretrained("albert-base-v1")
#tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

#-----------------------------------------------------------#

dataset = "ATIS" # point_size = 70, decode_length = 34, batch_size = 24, lr = 9.5e-5
dataset = "NYT24" # point_size = 225, batch_size = 64
dataset = "NYT29" # point_size = 270, batch_size = 64
dataset = "ADE" # point_size = 115, decode_length = 74, batch_size = 16, lr = 5e-5
dataset = "RAMS/filtered_level_1" # point_size = 512, batch_size = 24
dataset = "MAVEN" # point_size = 340, batch_size = 24
dataset = "CoNLL04" # point_size = 145, batch_size = 4/8
dataset = "ADE/folds" # point_size = 115, batch_size = 16, lr = 5e-5
dataset = "SNIPS" # point_size = 50, decode_length = 25, batch_size = 24, lr = 9.0e-5
dataset = "BiQuAD" # point_size = 100, batch_size = 64

#-----------------------------------------------------------#
# SELECT DATASET
dataset = "CoNLL04"

# READ GRAMMAR / VOCABULARY
gramm = gramm.GRAMMAR("gramm.txt")
vocab = read_file("data/" + dataset + "/vocabulary.txt")
vocab = dict(zip(vocab, range(len(vocab))))

# SET MODEL-PARAMETERS
gramm_size = gramm.size() + 1
vocab_size = len(vocab)

#-----------------------------------------------------------#
# TRAINING PARAMETERS

point_size = 145
decode_length = 35

EPOCHS = 1

batch_size = 32
skip_first = 0 # ignores evaluation of 'skip_first' epochs
save_model = False # saves best model
beam_search = False
num_beams = 1

#-----------------------------------------------------------#
# LOAD: TRAIN- / TESTS- / VALID-DATA

data_train = loadr.MyDataset("data/" + dataset + "/train", vocab, tokenizer, encode_length = point_size, decode_length = decode_length)
data_tests = loadr.MyDataset("data/" + dataset + "/test" , vocab, tokenizer, encode_length = point_size, decode_length = decode_length)
#data_valid = loadr.MyDataset("data/" + dataset + "/valid", vocab, tokenizer, encode_length = point_size, strip = True)

train_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
tests_loader = DataLoader(data_tests, batch_size = batch_size, shuffle = False)
#valid_loader = DataLoader(data_valid, batch_size = batch_size, shuffle = True)

#-----------------------------------------------------------# 
# SWEEP PARAMETERS

minimal_lr = 1e-5
maximal_lr = 1e-4

increments = 0.5 * minimal_lr

grid_steps = int((maximal_lr - minimal_lr) / increments) + 1

best_f1 = 0
best_lr = minimal_lr

for grid_step in range(grid_steps):
	
	# SET SEED & LEARNING RATE
	set_seed(1)

	current_lr = minimal_lr + grid_step * increments

	print("\n\n\n" * int(grid_step > 0))
	print(f"#--------------#--------------#--------------#")
	print(f"> NEW GRID STEP: {grid_step + 1}/{grid_steps}")
	print(f"> LEARNING-RATE: {current_lr} [{minimal_lr}, {maximal_lr}]")
	print(f"> CURRENT SCORE: F1 = {best_f1}, LR = {best_lr}")
	print(f"#--------------#--------------#--------------#\n")

	#-----------------------------------------------------------#
   
	# INITIALIZE MODEL
	model = TestModel(gramm = gramm, vocab = vocab, point_size = point_size)
	model = model.to(device)

	# LOAD SAVED MODEL
	# model.load_state_dict(torch.load("models/model_ATIS_e75_b32.pt"))

	#-----------------------------------------------------------#

	optimizer = torch.optim.AdamW(model.parameters(), lr = current_lr)
	ampscaler = torch.cuda.amp.GradScaler()
	scheduler = transformers.get_constant_schedule(optimizer = optimizer)

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

	#################################################################
		
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
		
		if epoch < skip_first:
			continue

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
			return_inputs = True,
			beam_search = beam_search,
			num_beams = num_beams,
		)
		
		#-----------------------------------------------------------#
		# TESTING 

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

			if save_model:
				torch.save(model.state_dict(), f"models/model_{dataset}_e{EPOCHS}_b{batch_size}.pt")

				with open(f"models/model_{dataset}_e{EPOCHS}_b{batch_size}.log", "w") as file:
					file.write(f"EPOCH: {epoch}/{EPOCHS}\n")
					file.write(f"MAX_TESTS_TOTAL_PR: {MAX_TESTS_TOTAL_PR}\n")
					file.write(f"MAX_TESTS_TOTAL_RE: {MAX_TESTS_TOTAL_RE}\n")
					file.write(f"MAX_TESTS_TOTAL_F1: {MAX_TESTS_TOTAL_F1}\n")
					file.write(f"MAX_TESTS_TABLE_F1: {MAX_TESTS_TABLE_F1}\n")
					file.write(f"MAX_TESTS_SLOTS_F1: {MAX_TESTS_SLOTS_F1}\n")
		
			write_results(tests_result, name = f"logs/TESTS_{dataset}_E{EPOCHS}_B{batch_size}_L{current_lr}.log")
		
		tests_bar.write(f"MAX_TESTS_TOTAL_PR: {MAX_TESTS_TOTAL_PR}")
		tests_bar.write(f"MAX_TESTS_TOTAL_RE: {MAX_TESTS_TOTAL_RE}")
		tests_bar.write(f"MAX_TESTS_TOTAL_F1: {MAX_TESTS_TOTAL_F1}")
		tests_bar.write(".....")
		tests_bar.write(f"MAX_TESTS_TABLE_F1: {MAX_TESTS_TABLE_F1}")
		tests_bar.write(f"MAX_TESTS_SLOTS_F1: {MAX_TESTS_SLOTS_F1}")
		
		#-----------------------------------------------------------#
		# VALIDATION

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
			return_inputs = True,
			beam_search = beam_search,
			num_beams = num_beams,
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

			write_results(tests_result, name = f"logs/VALID_{dataset}_E{EPOCHS}_B{batch_size}_L{current_lr}.log")
		
		tests_bar.write("")

		tests_bar.write(f"MAX_VALID_TOTAL_PR: {MAX_VALID_TOTAL_PR}")
		tests_bar.write(f"MAX_VALID_TOTAL_RE: {MAX_VALID_TOTAL_RE}")
		tests_bar.write(f"MAX_VALID_TOTAL_F1: {MAX_VALID_TOTAL_F1}")
		tests_bar.write(".....")
		tests_bar.write(f"MAX_VALID_TABLE_F1: {MAX_VALID_TABLE_F1}")
		tests_bar.write(f"MAX_VALID_SLOTS_F1: {MAX_VALID_SLOTS_F1}")

		tests_bar.write("o" + "-----" * 6 + "o")

		#-----------------------------------------------------------#

	if MAX_TESTS_TOTAL_F1 > best_f1:
		best_f1 = MAX_TESTS_TOTAL_F1
		best_lr = current_lr
		
	#################################################################

# WRITE FINAL REPORT
with open(f"logs/REPORT_{dataset}_E{EPOCHS}_B{batch_size}.log", "w") as file:
	file.write(f"MINIMAL LR: {minimal_lr}\n")
	file.write(f"MAXIMAL LR: {minimal_lr}\n")
	file.write(f"GRID STEPS: {grid_steps}\n")
	file.write( "--------------------\n" )
	file.write(f"BEST SCORE: {best_f1}\n")
	file.write(f"LEARN RATE: {best_lr}\n")
