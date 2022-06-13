from transformers import AlbertTokenizer

from model import REBEL
from utils import *
from train import training
from valid import validate
from datas import *

import transformers, torch, random, numpy, tqdm

#------------------------------------------------#

def set_seed(num):
	random.seed(num)
	
	torch.manual_seed(num)
	torch.cuda.manual_seed_all(num)
	torch.backends.cudnn.deterministic = True
	
	numpy.random.seed(num)
	
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_seed(1001)

#------------------------------------------------#

# LOAD / READ DATASET

# dloader = ADELoader() 		# encode_len = 120, decode_len = 96, batch_size = 16, learning_rate = 5e-5
# dloader = NYT24Loader() 		# encode_len = 215, decode_len = 96, batch_size = 16, learning_rate = 5e-5
# dloader = CONLL04Loader() 	# encode_len = 160, decode_len = 96, batch_size = 16, learning_rate = 5e-5
# dloader = ATISLoader() 		# encode_len =  60, decode_len = 32, batch_size = 16, learning_rate = 5e-5
# dloader = SNIPSLoader() 		# encode_len =  45, decode_len = 48, batch_size = 16, learning_rate = 5e-5

dloader = SNIPSLoader()
dreader = SEQReader(dloader)

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
tokenizer.add_tokens(dloader.tokens(), special_tokens = True)

num_epochs = 30
round_skip = 15 # \in [0, ..., num_epochs - 1]

batch_size = 16
encode_len = 45
decode_len = 48

num_sample = 2

#------------------------------------------------#

data_train = dloader.load("../data/SNIPS/", "train")
data_tests = dloader.load("../data/SNIPS/", "test")
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

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
ampscaler = torch.cuda.amp.GradScaler()
scheduler = transformers.get_constant_schedule_with_warmup(
	optimizer = optimizer,
	num_warmup_steps = 0.1 * num_epochs * len(train_loader)
)

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

	if epoch < round_skip:
		continue

	tests_bar.write(f"VALIDATE {epoch + 1}/{num_epochs}:")

	tests_result = validate(
		model = model,
		dataloader = tests_loader,
		tqdm_bar = tests_bar,
		tokenizer = tokenizer,
		extractor = dreader,
	)

	tests_score = compute_metrics( tests_result )

	if tests_score["F"] > MAX_TESTS_TOTAL_F1:
		MAX_TESTS_TOTAL_PR = tests_score["P"]
		MAX_TESTS_TOTAL_RE = tests_score["R"]
		MAX_TESTS_TOTAL_F1 = tests_score["F"]

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

import IPython ; IPython.embed() ; exit(1)