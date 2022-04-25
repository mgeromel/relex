import json

# -------------------------------------------------- #

def concatenate(super_list):
	result = []
	
	for l in super_list:
		result = result + l
	
	return result

# -------------------------------------------------- #
solutions = []
sentences = []	

read_file_name = "train.json"
write_file_name = "test"

relation_vocabulary = []
entities_vocabulary = []

with open(read_file_name) as file:
	
	for line in file:
		data = json.loads(line)
		
		complete = concatenate(data["sentences"])
		entities = concatenate(data["ner"])
		
		for x in entities:
			if x[2] not in entities_vocabulary:
				entities_vocabulary.append(x[2])
				
		entities = {(l, r) : v for [l, r, v] in entities}
		
		# DIVIDED BY SENTENCES, NOT DOCUMENTS !!
		for sentence, relations in zip(data["sentences"], data["relations"]):
			
			sentence = " ".join(sentence)
			solution = ""
			
			for relation in relations:
				relation_type = relation[-1]
				
				if relation_type not in relation_vocabulary:
					relation_vocabulary.append(relation_type)
				
				head_entity_type = entities[tuple(relation[0 : 2])]
				head_entity_text = " ".join( complete[relation[0] : relation[1] + 1] )
				head_entity = head_entity_type + " == " + head_entity_text
				
				tail_entity_type = entities[tuple(relation[2 : 4])]
				tail_entity_text = " ".join( complete[relation[2] : relation[3] + 1] )
				tail_entity = tail_entity_type + " == " + tail_entity_text
				
				relation_text = head_entity + " ;; " + tail_entity + " ;; " + relation_type
				
				solution = solution + (int(len(solution) > 0) * " || ") + relation_text
				
			sentences.append(sentence)
			solutions.append(solution)

# -------------------------------------------------- #

for token in sorted(relation_vocabulary):
	print(token)

print("---")

for token in sorted(entities_vocabulary):
	print(token)

exit(1)

print("num_sentences:", len(sentences))
print("num_solutions:", len(solutions))

sent_file = open(write_file_name + ".sent", "a")
tups_file = open(write_file_name + ".tup", "a")

for sent, tabs in zip(sentences, solutions):
	if tabs != "":
		sent_file.write( sent + "\n" )
		tups_file.write( tabs + "\n" )