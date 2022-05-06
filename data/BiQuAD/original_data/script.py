#----------------------------------------#

def read_line(line):
	tmp = line.strip()
	
	key = tmp[tmp.find(":") + 1 : tmp.find(" ")]
	val = tmp[tmp.find(" ") + 1 :              ]
	
	val = val.replace('"', "")
	val = val.replace("}", "")
	
	return key, val

#----------------------------------------#

def read_ids(file_name):
	match_report_ids = []
	
	with open(file_name) as file:
		for line in file.readlines():
			match_report_ids.append(int(line))
	
	return match_report_ids
	
#----------------------------------------#

def read_events(file_name):
	events = {}

	with open(file_name) as file:

		event = {}
		state = "IDLE" # SEEK, TAKE

		for line in file.readlines():
			if state == "SEEK" and "{" in line:
				state = "TAKE"

			if state == "TAKE":
				key, val = read_line(line)
				event[key] = val

				if "}" in line:

					time_key = -1

					if "event/minute" in event:
						time_key = event["event/minute"]
						try:
							time_key = int(time_key)
						except:
							print("time_key error")
							import IPython ; IPython.embed() ; exit(1)
							
					if time_key not in events:
						events[time_key] = []

					events[time_key].append(event)

					event = {}
					state = "SEEK"

			if state == "IDLE" and "[" in line:
				state = "SEEK"
	
	return events

#----------------------------------------#

def read_record(file_name):
	sentences = {}
	
	with open(file_name) as file:
	
		# READ HEAD
		head = next(file).strip()
		sentences[-1] = [head]

		# READ BODY
		for line in file.readlines():
			line = line.strip()
			
			time_key = line[ : line.find(" ") - 1]
			time_key = int(time_key)

			sentence = line[line.find(" ") + 1 : ]

			if time_key not in sentences:
				sentences[time_key] = []

			sentences[time_key].append(sentence)
	
	return sentences

#----------------------------------------#

total_events = []
total_record = []

match_report_ids = read_ids("match_report_ids")

for index in match_report_ids:
	
	# READ FILES
	events = read_events(f"datalog/match_report_{index}.dl")
	record = read_record(f"reports/match_report_{index}.txt")
	
	for minute, events in events.items():
		
		sentences = " ".join(record[minute])
		all_triples = []
		
		for event in events:
			if "event/type" in event:
				event_type = event["event/type"]
				event_subtype = event_type + "." + event_type
				
				if "event/subtype" in event:
					event_subtype = event_type + "." + "_".join(event["event/subtype"].lower().split())
				
				if f"event/{event_type}_type" in event:
					event_subtype = event_subtype + "." + event[f"event/{event_type}_type"]
				
				triple = ""
				
				clean = False
				# TO TRIPLE
				try:
					temp = event["event/player1"]
					
					if temp not in sentences:
						temp = temp.split()[-1]
					if temp not in sentences:
						break
						
					triple = triple + f"event.player1 == {temp} ;; "
				except:
					pass
				
				try:
					temp = event["event/player2"]
					
					if temp not in sentences:
						temp = temp.split()[-1]
					if temp not in sentences:
						break
						
					triple = triple + f"event.player2 == {temp} ;; "
				except:
					pass
					
				triple = triple + event_subtype	
				all_triples.append(triple)
			
			if "match/id" in event:
				triple = ""
				
				for key in event:
					if key != "match/id":
						if event[key] in sentences:
							triple = triple + key.replace("/", ".") + " == " + event[key] + " ;; "
				
				triple = triple + "match.summary"
				
				all_triples.append(triple)
				
		if len(all_triples) > 0:
			total_events.append(" || ".join(all_triples))
			total_record.append(sentences)

import random

liste = list(zip(total_record, total_events))

random.shuffle(liste)

l = len(liste)

train_data = liste[               : int(0.98 * l) ]
valid_data = liste[ int(0.98 * l) : int(0.99 * l) ]
tests_data = liste[ int(0.99 * l) :               ]

with open("train.sent", "a") as file:
	for line in train_data:
		file.write(line[0] + "\n")
		
with open("train.tup", "a") as file:
	for line in train_data:
		file.write(line[1] + "\n")
		
with open("valid.sent", "a") as file:
	for line in valid_data:
		file.write(line[0] + "\n")
		
with open("valid.tup", "a") as file:
	for line in valid_data:
		file.write(line[1] + "\n")
				
with open("test.sent", "a") as file:
	for line in tests_data:
		file.write(line[0] + "\n")
		
with open("test.tup", "a") as file:
	for line in tests_data:
		file.write(line[1] + "\n")
		
#----------------------------------------#