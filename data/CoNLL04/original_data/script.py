import json

# -------------------------------------------------- #

with open("train.json") as file:
	data = json.load(file)
	
	sentences = []
	solutions = []
	
	# [ dict(), ..., dict() ]
	for sample in data:
		
		sentence = " ".join(sample["tokens"])
		entities = sample["entities"]
		
		solution = ""
		
		for relation in sample["relations"]:
			# CREATE RELATION-TABLE
			relation_title = relation["type"]
			
			# HEAD ENTITY
			head_entity_type = entities[relation["head"]]["type"]
			head_entity_text = " ".join(sample["tokens"][
				entities[relation["head"]]["start"] :
				entities[relation["head"]]["end"]
			])
			head_entity = head_entity_type + " == " + head_entity_text
			
			# TAIL ENTITY
			tail_entity_type = entities[relation["tail"]]["type"]
			tail_entity_text = " ".join(sample["tokens"][
				entities[relation["tail"]]["start"] :
				entities[relation["tail"]]["end"]
			])
			tail_entity = tail_entity_type + " == " + tail_entity_text
			
			# FINAL RELATION
			relation = head_entity + " ; " + tail_entity + " ; " + relation_title
			
			# UPDATE TABLE
			solution = (
				solution + int(len(solution) > 0) * " | " + relation
			)
		
		sentences.append(sentence)
		solutions.append(solution)

print("num_sentences:", len(sentences))
print("num_solutions:", len(solutions))

# -------------------------------------------------- #

with open("train.sent", "a") as file:
	for line in sentences:
		file.write(line + "\n")

with open("train.tup", "a") as file:
	for line in solutions:
		file.write(line + "\n")