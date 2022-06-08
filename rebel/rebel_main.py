from transformers import AlbertTokenizer

from rebel_model import REBEL
from rebel_utils import *
from rebel_train import training
from rebel_valid import validate
from rebel_ade_loader import *

import transformers, torch, random, numpy, tqdm

#------------------------------------------------#

def set_seed(num):
	random.seed(num)
	
	torch.manual_seed(num)
	torch.cuda.manual_seed_all(num)
	torch.backends.cudnn.deterministic = True
	
	numpy.random.seed(num)
	
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------------------------------------------#

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1")
tokenizer.add_tokens(["<trip>", "<head>", "<tail>"], special_tokens = True)

#------------------------------------------------#
# LOAD / READ DATASET

dloader = ADELoader() # point_size = 120, batch_size = 16, learning_rate = 5e-5

ds_name = dloader.name()
dataset = dloader.load()

#random.shuffle( dataset )

data_size = len(dataset)

for i in range(3):
	
	set_seed(1001)

	lower = int((i + 0)/10 * data_size)
	upper = int((i + 1)/10 * data_size)
	
	print(f"ROUND {i+1}")
	
	data_train = dataset[ : lower ] + dataset[ upper : ]
	data_tests = dataset[ lower : upper ]
	#data_valid = dataset[ int(data_size * 0.1) : int(data_size * 0.2) ]

	point_size = 120

	data_train = CustomDataset(data_train, tokenizer, encoder_max_length = point_size, decoder_max_length = 80)
	data_tests = CustomDataset(data_tests, tokenizer, encoder_max_length = point_size, decoder_max_length = 80)
	#data_valid = CustomDataset(data_valid, tokenizer, encoder_max_length = point_size, decoder_max_length = 80)

	EPOCHS = 50
	batch_size = 16

	train_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, shuffle = True)
	tests_loader = torch.utils.data.DataLoader(data_tests, batch_size = batch_size, shuffle = True)
	#valid_loader = torch.utils.data.DataLoader(data_valid, batch_size = batch_size, shuffle = True)

	#------------------------------------------------#

	model = REBEL(tokenizer)
	model.to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
	ampscaler = torch.cuda.amp.GradScaler()
	scheduler = transformers.get_constant_schedule_with_warmup(
		optimizer = optimizer, num_warmup_steps = 0.1 * EPOCHS * len(train_loader)
	)

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

		train_bar.write(f"TRAINING {epoch + 1}/{EPOCHS}: {ds_name}")

		training(
			model = model,
			dataloader = train_loader,
			tqdm_bar = train_bar,
			optimizer = optimizer,
			scheduler = scheduler,
			ampscaler = ampscaler
		)

		#-----------------------------------------------------------#
		if epoch == EPOCHS - 1:
			continue

		tests_bar.write(f"VALIDATE {epoch + 1}/{EPOCHS}:")

		tests_result = validate(
			model = model,
			dataloader = tests_loader,
			tqdm_bar = tests_bar,
			tokenizer = tokenizer
		)

		tests_score = compute_metrics( tests_result )

		MAX_TESTS_TOTAL_PR = max(tests_score["P"], MAX_TESTS_TOTAL_PR)
		MAX_TESTS_TOTAL_RE = max(tests_score["R"], MAX_TESTS_TOTAL_RE)
		MAX_TESTS_TOTAL_F1 = max(tests_score["F"], MAX_TESTS_TOTAL_F1)

		for i in range(0):
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
			tokenizer = tokenizer
		)

		valid_score = compute_metrics( valid_result )

		MAX_VALID_TOTAL_PR = max(valid_score["P"], MAX_VALID_TOTAL_PR)
		MAX_VALID_TOTAL_RE = max(valid_score["R"], MAX_VALID_TOTAL_RE)
		MAX_VALID_TOTAL_F1 = max(valid_score["F"], MAX_VALID_TOTAL_F1)

		tests_bar.write("")

		tests_bar.write(f"MAX_VALID_TOTAL_PR: {MAX_VALID_TOTAL_PR}")
		tests_bar.write(f"MAX_VALID_TOTAL_RE: {MAX_VALID_TOTAL_RE}")
		tests_bar.write(f"MAX_VALID_TOTAL_F1: {MAX_VALID_TOTAL_F1}")

		tests_bar.write("o" + "-----" * 6 + "o")

		#-----------------------------------------------------------#

import IPython ; IPython.embed() ; exit(1)