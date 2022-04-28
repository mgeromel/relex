sentences = []
solutions = []

with open("full.sent") as file:
	for line in file:
		sentences.append(line)

with open("full.tup") as file:
	for line in file:
		solutions.append(line)
	
size = len(sentences)
FOLDS = 10

for i in range(FOLDS):
	sentences_tests = sentences[ i * (size//FOLDS) : (i + 1) * (size//FOLDS) ]
	solutions_tests = solutions[ i * (size//FOLDS) : (i + 1) * (size//FOLDS) ]
	
	sentences_train = sentences[0 : i * (size//FOLDS)] + sentences[(i + 1) * (size//FOLDS) : ]
	solutions_train = solutions[0 : i * (size//FOLDS)] + solutions[(i + 1) * (size//FOLDS) : ]
	
	with open(f"train_{i+1}.sent", "a") as file:
		for sentence in sentences_train:
			file.write(sentence)
			
	with open(f"train_{i+1}.tup", "a") as file:
		for solution in solutions_train:
			file.write(solution)
	
	with open(f"test_{i+1}.sent", "a") as file:
		for sentence in sentences_tests:
			file.write(sentence)
			
	with open(f"test_{i+1}.tup", "a") as file:
		for solution in solutions_tests:
			file.write(solution)