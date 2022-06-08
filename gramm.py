class GRAMMAR():
	
	def __init__(self, file_name):
		self.G = []
		self.max = 0
		
		with open(file_name, "r") as gram_file:
			for rule in gram_file.readlines():
				i = rule.find(",")     # Split Rule 

				n = rule[ : i].strip() # Non-Terminal
				p = rule[i+1:].strip() # Production
				
				self.G.append((n , p))

	def size(self):
		return len(self.G)
	
	def rule(self, idx):
		return self.G[ idx % len(self.G) ]

	def mask(self, val):
		return [ 0.0 if n == val else -float("inf") for n, _ in self.G ]
	
	def build_mask(self):
		states = []
		
		# NON-TERMINALS
		for rule in self.G:
			if rule[0] not in states:
				states.append(rule[0])
	
		# COLLECT MASKS
		mask = [ self.mask(state) for state in states ]
		smap = dict(enumerate(states))
		smap = { v : k for k , v in smap.items() }
		
		return smap, mask
	
#--------------------------------------------------#