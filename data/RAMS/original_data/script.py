import json

# -------------------------------------------------- #

def concatenate(super_list):
	result = []
	
	for l in super_list:
		result = result + l
	
	return result

# -------------------------------------------------- #

def find(index, sentences):
	len_sum = 0
	
	for rank, sent in enumerate(sentences):
		next_len_sum = len_sum + len(sent)
		
		if index >= len_sum and index < next_len_sum:
			return rank
		
		len_sum = next_len_sum
		
	return -1
	
# -------------------------------------------------- #

def find_nth(haystack, needle, n):
	start = haystack.find(needle)
	
	while start >= 0 and n > 1:
		start = haystack.find(needle, start+len(needle))
		n -= 1
		
	return start

# -------------------------------------------------- #

argument_names = []
event_names = []

with open("train.jsonlines") as file:

	full_solutions = []
	full_sentences = []

	for line in file:

		line = json.loads(line)
		
		MARKERS = [0] * len(line["sentences"])
		
		all_text = concatenate(line["sentences"])
		solution = ""

		event_span = line["evt_triggers"][0][0:2]
		
		event_trig = " ".join(all_text[event_span[0] : event_span[1] + 1])
		event_name = line["evt_triggers"][0][2][0][0]
		
		# SHORTENING EVENT-NAMES
		temp_range = find_nth(event_name, ".", 1)
		event_name = event_name[:temp_range]
		
		MARKERS[find(event_span[0], line["sentences"])] = 1
		MARKERS[find(event_span[1], line["sentences"])] = 1
		
		# EVENT-TYPE-VOCABULARY
		if event_name not in event_names:
			event_names.append(event_name)

		arguments = {}

		# SLOT FILLING
		for argument in line["gold_evt_links"]:
			argument_span = argument[1]
			
			argument_text = all_text[argument_span[0] : argument_span[1] + 1]
			argument_text = " ".join(argument_text)
			
			MARKERS[find(argument_span[0], line["sentences"])] = 1
			MARKERS[find(argument_span[1], line["sentences"])] = 1
			
			argument_role = argument[2]
			argument_role = argument_role[argument_role.find("arg") + 5:]
			
			# COLLECT ARGUMENT-TYPES
			if argument_role in arguments:
				arguments[argument_role].append(argument_text)
			else:
				arguments[argument_role] = [argument_text]

		# CREATE SERIALIZED TABLE
		for key in sorted(arguments.keys()):
			contents = f"{key} == " + " ^^ ".join(arguments[key])
			solution = solution + contents + " ;; "
		
		solution = solution + f"trigger == {event_trig} ;; {event_name}"
	
		# REQUIRED SENTENCES ?
		sentences = []
		
		for sentence, required in zip(line["sentences"], MARKERS):
			if required:
				sentences.append(sentence)
				
		sentences = concatenate(sentences)
		
		# RECORD SENTENCES / SOLUTIONS
		full_solutions.append(solution)
		full_sentences.append(" ".join(all_text))

# -------------------------------------------------- #

with open("train.sent", "a") as file:
	for line in full_sentences:
		file.write(line + "\n")

with open("train.tup",  "a") as file:
	for line in full_solutions:
		file.write(line + "\n")