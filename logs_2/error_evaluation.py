import ast

file_name = "TESTS_SNIPS_E100_B32_L8e-05.log2"

inputs = []
labels = []
predic = []

TABLE_ID_KEY = "TABLE_ID"


with open(file_name) as file:
	
	count = 0
	
	for line in file:
		line = line.strip()
		
		#-------------------------#
		if "INPUTS" in line:
			line = line.replace("INPUTS :: ", "")
			inputs.append(line)
			
		if "LABELS" in line:
			line = line.replace("LABELS :: ", "")
			line = ast.literal_eval(line)
			
			for key in line.keys():
				if type(line[key]) is list:
					line[key] = line[key][-1]
				
			if line == {}:
				line = {TABLE_ID_KEY : "DEFAULT"}
				
			labels.append(line)
			
		if "PREDIC" in line:
			if "{" not in line:
				line = line + " {}"
			
			line = line.replace("PREDIC :: ", "")
			line = ast.literal_eval(line)
			
			for key in line.keys():
				if type(line[key]) is list:
					line[key] = line[key][-1]
					
			if line == {}:
				line = {TABLE_ID_KEY : "DEFAULT"}
				
			predic.append(line)
		#-------------------------#
	
	KEYS_TOTAL = 0
	
	MISSING_KEYS = 0
	MISSING_KEYS_TOTAL = 0
	
	NOEXIST_KEYS = 0
	NOEXIST_KEYS_TOTAL = 0
	
	PARTIAL_KEYS = 0
	PARTIAL_KEYS_TOTAL = 0
	
	PARTIAL_TYPE = 0
	NOEXIST_TYPE = 0
	
	PARTIAL_SLOT_MORE = 0
	PARTIAL_SLOT_MORE_TOTAL = 0
	
	PARTIAL_SLOT_LESS = 0
	PARTIAL_SLOT_LESS_TOTAL = 0
	
	NOEXIST_SLOT = 0
	NOEXIST_SLOT_TOTAL = 0
	
	for text, labl, pred in zip(inputs, labels, predic):
		
		labels_keys = list(labl.keys())
		predic_keys = list(pred.keys())
		
		# MISSING KEYS
		missing = 0
		for key in labels_keys:
			if key not in predic_keys:
				missing = missing + 1
				
				print( labels_keys )
				print( predic_keys )
				print( "\n" )
				
		MISSING_KEYS = MISSING_KEYS + int(missing > 0)
		MISSING_KEYS_TOTAL = MISSING_KEYS_TOTAL + missing
		KEYS_TOTAL = KEYS_TOTAL + len(labels_keys) - 1
		
		# NOEXIST KEYS
		noexist = 0
		for key in predic_keys:
			if key not in labels_keys:
				noexist = noexist + 1
		NOEXIST_KEYS = NOEXIST_KEYS + int(noexist > 0)
		NOEXIST_KEYS_TOTAL = NOEXIST_KEYS_TOTAL + noexist
		
		# PARTIAL_KEYS
		partial = 0
		for key in predic_keys:
			for yek in labels_keys:
				if key != yek and (yek in key or key in yek):
					partial = partial + 1
					#print(f"key: {key}, yek: {yek}")
					
		PARTIAL_KEYS = PARTIAL_KEYS + int(partial > 0)
		PARTIAL_KEYS_TOTAL = PARTIAL_KEYS_TOTAL + partial
		
		# PARTIAL_SLOT
		partial_more = 0
		partial_less = 0
		for key in predic_keys:
			if key not in labels_keys:
				continue
			if pred[key] == labl[key]:
				continue
			if pred[key] in labl[key]:
				partial_less = partial_less + 1
			if labl[key] in pred[key]:
				partial_more = partial_more + 1	
		PARTIAL_SLOT_MORE = PARTIAL_SLOT_MORE + int(partial_more > 0)
		PARTIAL_SLOT_MORE_TOTAL = PARTIAL_SLOT_MORE_TOTAL + partial_more
		
		PARTIAL_SLOT_LESS = PARTIAL_SLOT_LESS + int(partial_less > 0)
		PARTIAL_SLOT_LESS_TOTAL = PARTIAL_SLOT_LESS_TOTAL + partial_less
		
		# NOEXIST SLOT
		noexist = 0
		for key in predic_keys:
			if key != TABLE_ID_KEY and pred[key] not in text:
				noexist = noexist + 1
		
		if noexist > 0:
			print("INPUTS :: " + text)
			print("LABELS :: " + str(labl))
			print("PREDIC :: " + str(pred))
			print()
				
		NOEXIST_SLOT = NOEXIST_SLOT + int(noexist > 0)
		NOEXIST_SLOT_TOTAL = NOEXIST_SLOT_TOTAL + noexist
		
		# NOEXIST TYPE
		if pred[TABLE_ID_KEY] != labl[TABLE_ID_KEY]:
			NOEXIST_TYPE += 1
	
	print("\n\nDONE DONE DONE\n\n")
	
	print(f"MISSING_KEYS: {MISSING_KEYS / len(inputs)} SAMPLES")
	print(f"MISSING_KEYS_TOTAL: {MISSING_KEYS_TOTAL / KEYS_TOTAL} KEYS")
	print()
	
	print(f"NOEXIST_KEYS: {NOEXIST_KEYS / len(inputs)} SAMPLES")
	print(f"NOEXIST_KEYS_TOTAL: {NOEXIST_KEYS_TOTAL / KEYS_TOTAL} KEYS")
	print()
	
	print(f"PARTIAL_KEYS: {PARTIAL_KEYS / len(inputs)} SAMPLES")
	print(f"PARTIAL_KEYS_TOTAL: {PARTIAL_KEYS_TOTAL / KEYS_TOTAL} KEYS")
	print()
	
	print(f"PARTIAL_SLOT_MORE: {PARTIAL_SLOT_MORE / len(inputs)} SAMPLES")
	print(f"PARTIAL_SLOT_MORE_TOTAL: {PARTIAL_SLOT_MORE_TOTAL / KEYS_TOTAL} SLOTS")
	print()
	
	print(f"PARTIAL_SLOT_LESS: {PARTIAL_SLOT_LESS / len(inputs)} SAMPLES")
	print(f"PARTIAL_SLOT_LESS_TOTAL: {PARTIAL_SLOT_LESS_TOTAL / KEYS_TOTAL} SLOTS")
	print()
	
	print(f"NOEXIST_SLOT: {NOEXIST_SLOT / len(inputs)} SAMPLES")
	print(f"NOEXIST_SLOT_TOTAL: {NOEXIST_SLOT_TOTAL / KEYS_TOTAL} SLOTS")
	print()
	
	print(f"NOEXIST_TYPE: {NOEXIST_TYPE / len(inputs)} SAMPLES")