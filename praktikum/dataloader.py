import datasets

def load_squad(tokenizer):
	squad_dataset = datasets.load_dataset("squad")

	squad_train = squad_dataset["train"]
	squad_valid = squad_dataset["validation"]

	train_data = build_model_input(squad_train, tokenizer)
	valid_data = build_model_input(squad_valid, tokenizer)

	return train_data, valid_data
	
def build_model_input(dataset, tokenizer):

	context_question = [ [sample["context"], sample["question"]] for sample in dataset ]

	context = [ sample["context"] for sample in dataset ]

	answers = [ sample["answers"]["text"][0] for sample in dataset ]

	tokenized_context_question = tokenizer(
		context_question,
		padding = "max_length",
		truncation = True,
		max_length = 1024 # maximale Länge bei ca. 800
	)

	tokenized_answers = tokenizer(
		answers,
		padding = "max_length",
		truncation = True,
		max_length = 128 # maximale Länge bei ca. 70
	)

	answers_pointer = []

	for text, answ in zip(context_question, answers):
		answers_pointer.append(locate(text[0], answ, tokenizer))

	import IPython ; IPython.embed() ; exit(1)

# Sucht die Antwort im Text (Positionen)
def locate(text, answ, tokenizer):

	pad_token = tokenizer.pad_token
	pad_token_id = tokenizer.pad_token_id

	text = text.lower()
	answ = answ.lower()

	text = text.replace(answ, pad_token + answ + pad_token)
	
	t_tokens = tokenizer(text, add_special_tokens = True).input_ids
	a_tokens = tokenizer(answ, add_special_tokens = False).input_ids
	
	answ = tokenizer.decode(a_tokens)
	
	#------------------------------#
	l_bound = 0
	r_bound = 0

	for idx in range(len(t_tokens)):
		if t_tokens[idx] == pad_token_id:
			l_bound = idx + 1
			break

	for idx in range(l_bound + 1, len(t_tokens)):
		if t_tokens[idx] == pad_token_id:
			r_bound = idx
			break
	#------------------------------#
	
	if t_tokens[l_bound : r_bound] != a_tokens:
		print("MATCHING ERROR")
		import IPython ; IPython.embed() ; exit(1)

	return (l_bound - 1, r_bound - 1)