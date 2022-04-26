
def read_file(filename):
	result = []
	
	with open(filename) as file:
		for line in file.readlines():
			line = " ".join(line.split())
			result.append(line)
	
	return result

data_path = "train/"

seq_in = read_file("train/seq.in")
seq_out = read_file("train/seq.out")
labels = read_file("train/label")

vocabulary = []

for label in labels:
	if label not in vocabulary:
		vocabulary.append(label)
		
#print("vocab_size:", len(vocabulary))
for word in sorted(vocabulary):
	print(word)
	
vocabulary = []

for output in seq_out:
	for tag in output.split():
		tag = tag[tag.find("-")+1:]
		if tag not in vocabulary:
			vocabulary.append(tag)

seq_out = read_file("test/seq.out")

for output in seq_out:
	for tag in output.split():
		tag = tag[tag.find("-")+1:]
		if tag not in vocabulary:
			vocabulary.append(tag)
			
seq_out = read_file("valid/seq.out")

for output in seq_out:
	for tag in output.split():
		tag = tag[tag.find("-")+1:]
		if tag not in vocabulary:
			vocabulary.append(tag)
			
#print("vocab_size:", len(vocabulary))
for word in sorted(vocabulary):
	print(word)