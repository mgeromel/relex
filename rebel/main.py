from transformers import AlbertTokenizer
from transformers import BartTokenizer

from model import REBEL
from utils import *
from train import training
from valid import validate
from datas import *

import transformers, torch, random, numpy, tqdm

#------------------------------------------------#

def write_results(batch, name = "DEFAULT"):
	
	with open(f"{name}_results.txt", "w") as file:
		for inputs, labels, predic in zip(batch["inputs"], batch["labels"], batch["predicted"]):
			file.write("INPUTS :: " + inputs + "\n")
			file.write("LABELS :: " + str(labels)[1:-1] + "\n")
			file.write("PREDIC :: " + str(predic)[1:-1] + "\n")
			file.write("\n")

#------------------------------------------------#

def set_seed(num):
	random.seed(num)
	
	torch.manual_seed(num)
	torch.cuda.manual_seed_all(num)
	torch.backends.cudnn.deterministic = True
	
	numpy.random.seed(num)
	
set_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------------------------------------------#

# LOAD / READ DATASET

# dloader = ADELoader() 		# encode_len = 120, decode_len = 96, batch_size = 16, learning_rate = 5e-5
# dloader = NYT24Loader() 		# encode_len = 215, decode_len = 96, batch_size = 16, learning_rate = 5e-5
# dloader = CONLL04Loader() 	# encode_len = 160, decode_len = 96, batch_size = 16, learning_rate = 5e-5
# dloader = ATISLoader() 		# encode_len =  60, decode_len = 32, batch_size = 16, learning_rate = 5e-5
# dloader = SNIPSLoader() 		# encode_len =  45, decode_len = 48, batch_size = 16, learning_rate = 5e-5

dloader = ATISLoader()
dreader = SEQReader(dloader)

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
#tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokenizer.add_tokens(dloader.tokens(), special_tokens = True)

num_epochs = 2
round_skip = 0 # \in [0, ..., num_epochs - 1]
current_lr = 5e-5

batch_size = 1
encode_len = 64
decode_len = 32

num_sample = 10

#------------------------------------------------#

data_train = dloader.load("../data/ATIS/", "train")
data_tests = dloader.load("../data/ATIS/", "test")
#data_valid = dloader.load("../data/ADE/", "valid")

data_train = CustomDataset(data_train, tokenizer, encode_len = encode_len, decode_len = decode_len)
data_tests = CustomDataset(data_tests, tokenizer, encode_len = encode_len, decode_len = decode_len)
#data_valid = CustomDataset(data_valid, tokenizer, encode_len = encode_len, decode_len = decode_len)

train_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, shuffle = True)
tests_loader = torch.utils.data.DataLoader(data_tests, batch_size = batch_size, shuffle = True)
#valid_loader = torch.utils.data.DataLoader(data_valid, batch_size = batch_size, shuffle = True)

#------------------------------------------------#

model = REBEL(tokenizer)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = current_lr)
ampscaler = torch.cuda.amp.GradScaler()
scheduler = transformers.get_constant_schedule(optimizer = optimizer)

train_bar = tqdm.tqdm(total = num_epochs * len(train_loader), leave = False, position = 0, desc = "TRAIN")
tests_bar = tqdm.tqdm(total = num_epochs * len(tests_loader), leave = False, position = 1, desc = "TESTS")
#valid_bar = tqdm.tqdm(total = num_epochs * len(valid_loader), leave = False, position = 2, desc = "VALID")

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

MAX_TESTS_KEYS_F1 = -1
MAX_TESTS_TABS_F1 = -1
MAX_TESTS_SLOT_F1 = -1

################################################################################

for epoch in range(num_epochs):

	#-----------------------------------------------------------#

	train_bar.write(f"TRAINING {epoch + 1}/{num_epochs}: {dloader.name()}")
	
	training(
		model = model,
		dataloader = train_loader,
		tqdm_bar = train_bar,
		optimizer = optimizer,
		scheduler = scheduler,
		ampscaler = ampscaler
	)
	
	#-----------------------------------------------------------#

	if epoch + 1 < round_skip:
		continue

	tests_bar.write(f"VALIDATE {epoch + 1}/{num_epochs}:")

	tests_result = validate(
		model = model,
		dataloader = tests_loader,
		tqdm_bar = tests_bar,
		tokenizer = tokenizer,
		extractor = dreader,
		return_inputs = True
	)

	tests_score = compute_metrics( tests_result )

	if tests_score["F"] > MAX_TESTS_TOTAL_F1:
		MAX_TESTS_TOTAL_PR = tests_score["P"]
		MAX_TESTS_TOTAL_RE = tests_score["R"]
		MAX_TESTS_TOTAL_F1 = tests_score["F"]
		
		write_results(tests_result, name = f"logs/TESTS_{dloader.name()}_E{num_epochs}_B{batch_size}_L{current_lr}.log")
		
	for i in range(num_sample):
		print("labels:", tests_result["labels"][i])
		print("predic:", tests_result["predicted"][i])
		print("")

	tests_bar.write(f"MAX_TESTS_TOTAL_PR: {MAX_TESTS_TOTAL_PR}")
	tests_bar.write(f"MAX_TESTS_TOTAL_RE: {MAX_TESTS_TOTAL_RE}")
	tests_bar.write(f"MAX_TESTS_TOTAL_F1: {MAX_TESTS_TOTAL_F1}")
	
	#-----------------------------------------------------------#

	if "valid_loader" not in locals():
		continue

	valid_result = validate(
		model = model,
		dataloader = valid_loader,
		tqdm_bar = valid_bar,
		tokenizer = tokenizer,
		extractor = dreader,
	)

	valid_score = compute_metrics( valid_result )
	
	if valid_score["F"] > MAX_VALID_TOTAL_F1:
		MAX_VALID_TOTAL_PR = valid_score["P"]
		MAX_VALID_TOTAL_RE = valid_score["R"]
		MAX_VALID_TOTAL_F1 = valid_score["F"]

	tests_bar.write("")

	tests_bar.write(f"MAX_VALID_TOTAL_PR: {MAX_VALID_TOTAL_PR}")
	tests_bar.write(f"MAX_VALID_TOTAL_RE: {MAX_VALID_TOTAL_RE}")
	tests_bar.write(f"MAX_VALID_TOTAL_F1: {MAX_VALID_TOTAL_F1}")

	tests_bar.write("o" + "-----" * 6 + "o")
	
	#-----------------------------------------------------------#