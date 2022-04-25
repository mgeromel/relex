def read_file(filename):
	result = []
	
	with open(filename) as file:
		for line in file.readlines():
			line = " ".join(line.split())
			result.append(line)
	
	return result

file_name = "test"
	
seq_in = read_file(file_name + "/seq.in")
seq_out = read_file(file_name + "/seq.out")
labels = read_file(file_name + "/label")

inp_file = ""
out_file = ""

for label, text, tags in zip( labels, seq_in, seq_out ):
	D = {}
	
	# store argument roles in dictionary D
	for word, tag in zip(text.split(), tags.split()):
		if tag != "O":
			tag = tag[2:] # B-LOCATION -> LOCATION
			
			if tag in D:
				if D[tag] + " " + word in text:
					D[tag] = D[tag] + " " + word
			else:
				D[tag] = word
	
	inp_file = inp_file + text + "\n"
	
	# for each argument role
	for arg_role in sorted(D.keys()):
		argument = arg_role + " == " + D[arg_role] + " ; "
		out_file = out_file + argument
	
	out_file = out_file + label + "\n"


file = open(file_name + ".sent", "a")
file.write(inp_file)
file.close()

file = open(file_name + ".tup", "a")
file.write(out_file)
file.close()