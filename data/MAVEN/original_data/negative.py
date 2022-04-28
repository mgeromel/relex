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
		n = n - 1
		
	return start

# -------------------------------------------------- #

sentences = []
solutions = []

event_labels = []

with open("train.jsonl") as file:
	
	for line in file:
		line = json.loads(line)
		
		doc_sentences = []
		doc_events = []
		
		# -------------------------------------------------- #
		
		for idx, content in enumerate(line["content"]):
			tokens = content["tokens"]
			
			doc_sentences.append(tokens)
			tmp_mentions = []
			
			for event in line["events"]:
				for mention in event["mention"]:
					mention["type"] = event["type"]
					
					if mention["type"] not in event_labels:
						event_labels.append(mention["type"])
					
					if mention["sent_id"] == idx:
						tmp_mentions.append(mention)
			
			doc_events.append(tmp_mentions)
		
		# -------------------------------------------------- #
		
		for sentence, mentions in zip(doc_sentences, doc_events):
			if len(mentions) > 0:
				records = []
				
				for mention in mentions:
					record = mention["trigger_word"] + " ;; " + mention["type"]
					records.append(record)
				
				sentences.append(" ".join(sentence))
				solutions.append(" || ".join(records))

# -------------------------------------------------- #

with open("train.sent", "a") as file:
	for line in sentences:
		file.write(line + "\n")

with open("train.tup", "a") as file:
	for line in solutions:
		file.write(line + "\n")

# -------------------------------------------------- #