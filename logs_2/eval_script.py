import ast

from collections import Counter

IGNORE_FLAG = True

#------------------------------#

def missing_keys(text, labels_keys, predic):
	overlap = list((Counter(labels_keys) & Counter(labels_keys)).elements())

	return len(labels_keys) - len(overlap)

#------------------------------#

def noexist_keys(text, labels_keys, predic_keys):
	overlap = list((Counter(predic_keys) & Counter(labels_keys)).elements())

	return len(predic_keys) - len(overlap)

#------------------------------#

def partial_keys(text, labels_keys, predic_keys):

	partial = 0
	for key in predic_keys:
		for yek in labels_keys:
			if key != yek and (yek in key or key in yek):
				partial = partial + 1

	return partial

#------------------------------#

def partial_slot_more(text, labels, predic, table_key):

	if IGNORE_FLAG:
		return 0

	# LIST OF KEYS
	labels_keys = []
	for entry in labl:
		for key in entry.keys():
			labels_keys.append(key)
	
	predic_keys = []
	for entry in pred:
		for key in entry.keys():
			predic_keys.append(key)

	# FILTERING
	labels_keys = list(filter(lambda x : x != table_key, labels_keys))
	predic_keys = list(filter(lambda x : x != table_key, predic_keys))

	partial_more = 0
	for key in predic_keys:
		if key not in labels_keys:
			continue
		if predic[key] == labels[key]:
			continue
		if labels[key] in predic[key]:
			partial_more = partial_more + 1	

	return partial_more

#------------------------------#

def partial_slot_less(text, labels, predic, table_key):

	if IGNORE_FLAG:
		return 0

	# LIST OF KEYS
	labels_keys = []
	for entry in labl:
		for key in entry.keys():
			labels_keys.append(key)
	
	predic_keys = []
	for entry in pred:
		for key in entry.keys():
			predic_keys.append(key)

	# FILTERING
	labels_keys = list(filter(lambda x : x != table_key, labels_keys))
	predic_keys = list(filter(lambda x : x != table_key, predic_keys))

	partial_less = 0
	for key in predic_keys:
		if key not in labels_keys:
			continue
		if predic[key] == labels[key]:
			continue
		if predic[key] in labels[key]:
			partial_less = partial_less + 1
		
	return partial_less

#------------------------------#

def noexist_slot(text, labels, predic, table_key):
	noexist = 0

	for entry in predic:
		for key in entry.keys():
			if key != table_key and entry[key] not in text:
				noexist = noexist + 1

	return noexist

#------------------------------#

def noexist_type(text, labels, predic, table_key):
	if IGNORE_FLAG:
		return 0

	labels_type = [ entry[table_key] for entry in labels ]
	predic_type = [ entry[table_key] for entry in predic ]

	overlap = list((Counter(predic_keys) & Counter(labels_keys)).elements())

	return (len(predic_type) - len(overlap)) / len(labels_type) 

#------------------------------#

file_name = "TESTS_ADE_E50_B8_L5e-05.lin"

inputs = []
labels = []
predic = []

TABLE_ID_KEY = "TYPE"

with open(file_name) as file:
	
	count = 0
	
	for line in file:
		line = line.strip()
		
		#-------------------------#
		if "INPUTS" in line:
			line = line.replace("INPUTS :: ", "")
			inputs.append(line)
			
		if "LABELS" in line:
			line = "[" + line.replace("LABELS ::", "") + "]"
			line = ast.literal_eval(line)
			labels.append(line)
			
		if "PREDIC" in line:
			line = "[" + line.replace("PREDIC ::", "") + "]"
			line = ast.literal_eval(line)
			predic.append(line)
		#-------------------------#

	KEYS_TOTAL = 0
	TABS_TOTAL = 0
	
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
		
		# LIST OF KEYS
		labels_keys = []
		for entry in labl:
			for key in entry.keys():
				labels_keys.append(key)
		
		predic_keys = []
		for entry in pred:
			for key in entry.keys():
				predic_keys.append(key)

		# FILTERING
		labels_keys = list(filter(lambda x : x != TABLE_ID_KEY, labels_keys))
		predic_keys = list(filter(lambda x : x != TABLE_ID_KEY, predic_keys))
		
		# TOTAL KEYS
		KEYS_TOTAL = KEYS_TOTAL + len(labels_keys) # 0 if no TYPE
		TABS_TOTAL = TABS_TOTAL + len(labl)

		# MISSING KEYS
		missing = missing_keys(text, labels_keys, predic_keys)
		MISSING_KEYS = MISSING_KEYS + int(missing > 0)
		MISSING_KEYS_TOTAL = MISSING_KEYS_TOTAL + missing
		
		# NOEXIST KEYS
		noexist = noexist_keys(text, labels_keys, predic_keys)
		NOEXIST_KEYS = NOEXIST_KEYS + int(noexist > 0)
		NOEXIST_KEYS_TOTAL = NOEXIST_KEYS_TOTAL + noexist
		
		# PARTIAL_KEYS
		partial = partial_keys(text, labels_keys, predic_keys)
		PARTIAL_KEYS = PARTIAL_KEYS + int(partial > 0)
		PARTIAL_KEYS_TOTAL = PARTIAL_KEYS_TOTAL + partial
		
		# PARTIAL_SLOT MORE
		partial_more = partial_slot_more(text, labl, pred, TABLE_ID_KEY)
		PARTIAL_SLOT_MORE = PARTIAL_SLOT_MORE + int(partial_more > 0)
		PARTIAL_SLOT_MORE_TOTAL = PARTIAL_SLOT_MORE_TOTAL + partial_more

		# PARTIAL SLOT LESS
		partial_less = partial_slot_less(text, labl, pred, TABLE_ID_KEY)
		PARTIAL_SLOT_LESS = PARTIAL_SLOT_LESS + int(partial_less > 0)
		PARTIAL_SLOT_LESS_TOTAL = PARTIAL_SLOT_LESS_TOTAL + partial_less
		
		# NOEXIST SLOT
		noexist = noexist_slot(text, labl, pred, TABLE_ID_KEY)
		NOEXIST_SLOT = NOEXIST_SLOT + int(noexist > 0)
		NOEXIST_SLOT_TOTAL = NOEXIST_SLOT_TOTAL + noexist	
		
		# NOEXIST TYPE
		NOEXIST_TYPE = NOEXIST_TYPE + noexist_type(text, labl, pred, TABLE_ID_KEY)
	
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
	
	print(f"NOEXIST_TYPE: {NOEXIST_TYPE / TABS_TOTAL} SAMPLES")