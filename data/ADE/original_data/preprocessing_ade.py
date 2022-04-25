import random

# adverse_effect(adverse_effect, drug)
# dose(dose, drug)
# no_relation

def intersection(list_1, list_2):
	intersect = []
	
	for x in list_1:
		if x in list_2:
			intersect.append(x)
	
	return intersect

#---------------------------------------------------------------------------#

with open("DRUG-AE.rel", "r") as file:
	
	sentences_effect = [  ]
	relations_effect = [""] * 4271
	
	for line in file.readlines():
		columns = line.strip().split("|")
		
		sentence = columns[1].strip()
		relation = columns[2].strip() + " ; " + columns[5].strip() + " ; adverse_effect | "
		
		if sentence not in sentences_effect:
			sentences_effect.append(sentence)
		
		index = len(sentences_effect) - 1
		relations_effect[index] = relations_effect[index] + relation

for i in range(len(relations_effect)):
	relations_effect[i] = relations_effect[i][0:-3]
	
print("adverse drug sentences:", len(sentences_effect))
print("adverse drug relations:", len(relations_effect))

#---------------------------------------------------------------------------#

# DOSE IS A SUBSET OF EFFECT!
with open("DRUG-DOSE.rel", "r") as file:
	
	sentences_dose = [] 
	
	relations_dose = [""] * 213
	
	for line in file.readlines():
		columns = line.strip().split("|")
		
		sentence = columns[1].strip()
		relation = columns[2].strip() + " ; " + columns[5].strip() + " ; dose | "
		
		if sentence not in sentences_dose:
			sentences_dose.append(sentence)
			
		index = len(sentences_dose) - 1
		relations_dose[index] = relations_dose[index] + relation


for i in range(len(relations_dose)):
	relations_dose[i] = relations_dose[i][0:-3]

print("dose sentences:", len(sentences_dose))
print("dose relations:", len(relations_dose))

#---------------------------------------------------------------------------#

with open("ADE-NEG.txt", "r") as file:
	
	sentences_negative = [] 
	
	relations_negative = [""] * 16625
	
	for line in file.readlines():
		columns = line.strip().split("NEG")
		
		sentence = columns[1].strip()
		
		if sentence not in sentences_negative:
			sentences_negative.append(sentence)
			
		index = len(sentences_dose) - 1
		relations_dose[index] = "no_relation"

print("dose sentences:", len(sentences_negative))
print("dose relations:", len(relations_negative))

#---------------------------------------------------------------------------#

sent_rels = list(zip(sentences_effect, relations_effect))

random.shuffle(sent_rels)

train_split = sent_rels[ : - 2 * 213]
test_split = sent_rels[ -2 * 213 : - 213]
dev_split = sent_rels[ - 213 : ]

with open("full.sent", "a") as file:
	for x in sent_rels:
		file.write(x[0].strip() + "\n")
		
with open("full.tup", "a") as file:
	for x in sent_rels:
		file.write(x[1].strip() + "\n")

print(len(train_split))
print(len(test_split))
print(len(dev_split))

with open("train.sent", "a") as file:
	for x in train_split:
		file.write(x[0].strip() + "\n")
		
with open("train.tup", "a") as file:
	for x in train_split:
		file.write(x[1].strip() + "\n")

with open("test.sent", "a") as file:
	for x in test_split:
		file.write(x[0].strip() + "\n")
		
with open("test.tup", "a") as file:
	for x in test_split:
		file.write(x[1].strip() + "\n")

with open("valid.sent", "a") as file:
	for x in dev_split:
		file.write(x[0].strip() + "\n")
		
with open("valid.tup", "a") as file:
	for x in dev_split:
		file.write(x[1].strip() + "\n")