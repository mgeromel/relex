
in_file = "train_valid.tup"
out_file = "train_valid_merged.tup"


new_lines = []

with open(in_file) as file:
	
	for line in file:
		line = line.strip()
		
		triples = line.split(" || ")
		
		if len(triples) == 1:
			triple = line.split(" ;; ")
			
			new_line = f"{triple[1]} ;; {triple[0]} ;; adverse_effect"
			new_lines.append(new_line)
			
		else:
			head_dict = {}
			
			# Read to Dictionary
			for triple in triples:
				triple = triple.split(" ;; ")
				
				if triple[1] not in head_dict:
					head_dict[triple[1]] = []
				
				if triple[0] not in head_dict[triple[1]]:
					head_dict[triple[1]].append(triple[0])
				
			temp_lines = []
			
			for head, tail in head_dict.items():	
				
				# SORT TAIL ?
				
				temp_line = f"{head} ;; " + " ;; ".join(tail) + " ;; adverse_effect"
				temp_lines.append(temp_line)
			
			new_line = " || ".join(temp_lines)
			new_lines.append(new_line)
			
with open(out_file, "a") as file:
	for line in new_lines:
		file.write(line + "\n")
