import random

sent_rels = []

def find(text, liste):
	for x in range(len(liste)):
		if text == liste[x]:
			return x
	return -1

with open("corp.tsv", "r") as file:
	chunks = ""
	
	for line in file.read():
		chunks = chunks + line
	
	chunks = chunks.split("\n\n")
	chunks = [chunk.strip() for chunk in chunks]
	
	vocabulary = []
	
	sentences = []
	relations = ["no_relation"] * 5516
	
	for chunk in chunks:
		
		table = chunk.split("\n")
		table = [row.split("\t") for row in table]
		
		if(len(table[0]) == 3):
			
			index = len(sentences) - 1
			
			relation = ""
			
			for row in table:
				arg_a = sentences[index][int(row[0])]
				arg_b = sentences[index][int(row[1])]
				relat = row[2]
				
				if relat not in vocabulary:
					vocabulary.append(relat)
				
				relation = relation + arg_a + " ; " + arg_b + " ; " + relat + " | "
			
			relation = relation[0:-3]
			relations[index] = relation
			
			pass
		else:
			sentence = ()
				
			for row in table:
				sentence = sentence + (row[5].replace("/,", ",").replace("/", " "),)
			
			sentences.append(sentence)
	
	for i in range(len(sentences)):
		sentence = ""
			
		for word in sentences[i]:
			sentence = sentence + " " + word + " "
		
		sentences[i] = sentence
	
	sentences_final = []
	relations_final = []
	for i in range(len(sentences)):
		if sentences[i] not in sentences_final and relations[i] != "no_relation":
			sentences_final.append(sentences[i])
			relations_final.append(relations[i])
		elif relations[i] != "no_relation":
			index = find(sentences[i], sentences_final)
			
			if relations[i] != relations_final[index] and relations[i] not in relations_final[index]:
				if relations_final[index] in relations[i]:
					relations_final[index] = relations[i]
				else: 
					relations_final[index] = relations_final[index] + " | " + relations[i]
					
	print(len(sentences_final))
	print(len(relations_final))
	
	sent_rels = list(zip(sentences_final, relations_final))

	
def clean(text):
	return ''.join(text.lower().split())

copy = sentences_final.copy()

copy = [clean(x) for x in copy]


duplicates = []
for i in range(len(copy)):
	tmp = find(copy[i], copy)
	if tmp != i:
		duplicates.append(tmp)
		

duplicates.sort(reverse=True)

deleted = []

for x in duplicates:
	deleted = sent_rels[x][0]
	del sent_rels[x]

print("Size:", len(sent_rels))
	
exit(1)

content = []

with open("testing_split.txt", "r") as file:
	
	for line in file.readlines():
		line = line[line.find(":") + 1 :].strip()
		content.append(clean(line))
	
	
	
	print("testing data", len(content))
	
	test_split = []
	
	count = 0
	temp = list(zip(sentences, relations))
	for x,y in temp:
		if clean(x) in content:
			count = count + 1
			test_split.append((x,y))
			
	print("count", count)	

	print("sent_rels:", len(temp))
	print("test_split", len(test_split))
		
		
			
		
		
	