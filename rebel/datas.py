import random

#----------------------------------------------------#

def load_dataset(dataset_name):

	if dataset_name not in ["ADE", "CoNLL04", "NYT24", "ATIS", "SNIPS"]:
		raise ValueError('config.dataset_name must be in ["ADE", "CoNLL04", "NYT24", "ATIS", "SNIPS"]')

	if dataset_name == "ADE":
		dataset = ADELoader()

	if dataset_name == "CoNLL04":
		dataset = CONLL04Loader()

	if dataset_name == "NYT24":
		dataset = NYT24Loader()

	if dataset_name == "ATIS":
		dataset = ATISLoader()

	if dataset_name == "SNIPS":
		dataset = SNIPSLoader()

	return dataset

#----------------------------------------------------#

class SEQReader():
	
	#------------------------------------------------#
	
	def __init__(self, dataloader):
		if type(dataloader) == str:
			dataloader = load_dataset(dataloader)

		self.loader = dataloader
	
	#------------------------------------------------#
	
	def read(self, text):
		name = self.loader.name()
		
		if name in ["ADE"]:
			return self._extract_tuples(text)
		
		if name in ["CoNLL04", "NYT24"]:
			return self._extract_triples(text)
		
		if name in ["ATIS", "SNIPS"]:
			return self._extract_tables(text)
		
	#------------------------------------------------#
	
	def _extract_tuples(self, text):
		tuples = []

		text = text.strip()

		head, tail = "", ""
		mode = "DEFAULT"

		#--------------------------------------------#
		
		for token in text.split():

			# STATE 1
			if token == "<trip>":
				mode = "READ_HEAD"
				head = ""
				tail = ""

			# STATE 2
			elif token == "<head>":
				mode = "READ_TAIL"
				tail = ""

			# STATE 3
			elif token == "<tail>":

				if head != "" and tail != "":
					tuples.append({'head': head.strip(), 'tail': tail.strip()})

				tail = ""
				
			# STATE ADD
			else:
				if mode == "READ_HEAD":
					head = head + " " + token

				elif mode == "READ_TAIL":
					tail = tail + " " + token

		#--------------------------------------------#
		
		return tuples
	
	#------------------------------------------------#
	
	def _extract_triples(self, text):
		triples = []

		text = text.strip()

		tokens = self.loader.tokens()

		head, tail, pred = "", "", ""
		head_type, tail_type = "", ""

		mode = "DEFAULT"

		#--------------------------------------------#
		
		for token in text.split():

			# STATE 1
			if token == "<trip>":
				mode = "READ_HEAD"

				if head != "" and tail != "" and pred != "":
					triples.append({f'h-{head_type}': head.strip(), 'pred': pred.strip() , f't-{tail_type}': tail.strip()})

				head, tail, pred = "", "", ""
				head_type, tail_type = "", ""

			# STATE 2
			elif token in tokens and mode == "READ_HEAD":
				head_type = token[1:-1]
				mode = "READ_TAIL"
				tail = ""

			# STATE 3
			elif token in tokens and mode == "READ_TAIL":
				tail_type = token[1:-1]
				mode = "READ PRED"
				pred = ""

			# STATE READ
			else:
				if mode == "READ_HEAD":
					head = head + " " + token
				elif mode == "READ_TAIL":
					tail = tail + " " + token
				elif mode == "READ PRED":
					pred = pred + " " + token

		#--------------------------------------------#
		
		if head != "" and tail != "" and pred != "":
			triples.append({f'h-{head_type}': head.strip(), 'pred': pred.strip() , f't-{tail_type}': tail.strip()})

		#--------------------------------------------#
		
		return triples	
	
	#------------------------------------------------#
	
	def _extract_tables(self, text):
		text = text.strip()

		tables = []

		table_type = ""
		table = {}

		item_type = ""
		item = ""

		#--------------------------------------------#
		
		for token in text.split():

			# STATE 1
			if token == "<trip>":

				if len(table) > 0:
					table["TYPE"] = item.strip()
					tables.append(table)

				item = ""
				table = {}

			# STATE 2
			elif token in self.loader.tokens():
				item_type = token[1:-1]
				table[item_type] = item.strip()

				item = ""
			# STATE READ
			else:
				item = item + " " + token

		#--------------------------------------------#
		
		if len(table) > 0:
			table["TYPE"] = item
			tables.append(table)

		#--------------------------------------------#
		
		return tables
		
##################################################

class ADELoader():
	
	def __init__(self):
		self._name = "ADE"
		self._vmap = {}
		
		self._tkns = ["<trip>", "<head>", "<tail>"]

	#------------------------------------------------#
	
	def name(self):
		return self._name
	
	def tokens(self):
		return self._tkns

	#------------------------------------------------#

	def load(self, path, filename, sort = False):
		
		sentences = []
		relations = []
		
		#--------------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())
			
		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())
			
		#--------------------------------------------#
		
		result = []
		
		#--------------------------------------------#
		
		# 2. Linearize Drug-Effect Clusters		
		for text, rels in zip(sentences, relations):

			#----------------------------------------#
			
			tuples = {}
			
			# Sort Tails by Position in Source Text
			for triple in rels.split(" || "):
				head, tail, _ = triple.split(" ;; ")
				
				if head not in tuples:
					tuples[head] = []
				
				if tail not in tuples[head]:
					tuples[head].append(tail)
			
			# Sort Heads by Position in Source Text
			tuples = [ (k, v) for k, v in tuples.items() ]
			tuples.sort(key = lambda x : text.find(x[0]) )
			
			random.shuffle( tuples )
				
			#----------------------------------------#
			
			labels = ""

			for drug, effects in tuples:
				head = f"<trip> {drug} <head> "
				tail = " <tail> ".join(effects) + " <tail> "
				labels = labels + head + tail

			labels = labels.strip()

			result.append( {"phrases" : text, "targets" : labels} )

			#----------------------------------------#
		
		return result

##################################################

class CONLL04Loader():
	
	def __init__(self):
		self._name = "CoNLL04"
		self._vmap = {
			"Kill" : "kill", 
			"Live_In" : "lives in", 
			"Located_In" : "located in", 
			"OrgBased_In" : "based in", 
			"Work_For" : "work for", 
		}
		self._tkns = ["<trip>", "<loc>", "<org>", "<peop>", "<other>"]
	
	#--------------------------------------------#
	
	def name(self):
		return self._name

	def tokens(self):
		return self._tkns
	
	#--------------------------------------------#

	def load(self, path, filename, sort = False):
		
		sentences = []
		relations = []
		
		#----------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())
			
		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())
			
		#----------------------------------------#
		
		result = []
		
		for text, rels in zip(sentences, relations):
			
			triples = []
			
			#------------------------------------#
			
			for triple in rels.split(" || "):
				triple = triple.split(" ;; ")
				triple = ( triple[0] , triple[1], triple[2] )
				
				if triple not in triples:
					triples.append(triple)
			
			triples.sort(key = lambda x : x[2]) # text.find(x[0])
			
			#------------------------------------#
			
			labels = ""
			
			for triple in triples:
				head = triple[0].split(" == ")
				tail = triple[1].split(" == ")
				pred = self._vmap[triple[2]]
				
				head = f"{head[1]} <{head[0].lower()}>"
				tail = f"{tail[1]} <{tail[0].lower()}>"
				
				label = f" <trip> {head} {tail} {pred}"
				labels = labels + label
			
			labels = labels.strip()
			
			result.append( { "phrases" : text , "targets" : labels } )
		
		return result

##################################################

class NYT24Loader():
	
	def __init__(self):
		self._name = "NYT24"
		self._vmap = {
			'/business/company/founders' : 'founders',
			'/people/person/place_of_birth' : 'place of birth',
			'/people/deceased_person/place_of_death' : 'place of death',
			'/business/company_shareholder/major_shareholder_of' : 'major shareholder of',
			'/people/ethnicity/people' : 'people',
			'/location/neighborhood/neighborhood_of' : 'neighborhood of',
			'/sports/sports_team/location' : 'location',
			'/business/person/company' : 'company',
			'/business/company/industry' : 'industry',
			'/business/company/place_founded' : 'place founded',
			'/location/administrative_division/country' : 'country',
			'/sports/sports_team_location/teams' : 'teams',
			'/people/person/nationality' : 'nationality',
			'/people/person/religion' : 'religion',
			'/business/company/advisors' : 'advisors',
			'/people/person/ethnicity' : 'ethnicity',
			'/people/ethnicity/geographic_distribution' : 'geographic distribution',
			'/people/person/place_lived' : 'place lived',
			'/business/company/major_shareholders' : 'major shareholders',
			'/people/person/profession' : 'profession',
			'/location/country/capital' : 'capital',
			'/location/location/contains' : 'contains',
			'/location/country/administrative_divisions' : 'administrative divisions',
			'/people/person/children' : 'children'
		}
		self._tkns = ["<trip>", "<head>", "<tail>"]
		
	#--------------------------------------------#
	
	def name(self):
		return self._name

	def tokens(self):
		return self._tkns
	
	#--------------------------------------------#

	def load(self, path, filename, sort = False):
		
		sentences = []
		relations = []
		
		#----------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())
			
		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())
			
		#----------------------------------------#
		
		result = []
		
		for text, rels in zip(sentences, relations):
			
			triples = []
			
			#------------------------------------#
			
			for triple in rels.split(" || "):
				triple = triple.split(" ;; ")
				triple = ( triple[0] , triple[1], self._vmap[triple[2]] )
				
				if triple not in triples:
					triples.append(triple)
			
			triples.sort(key = lambda x : x[2]) # text.find(x[0])
			
			#------------------------------------#
			
			labels = ""
			
			for triple in triples:
				label = f" <trip> {triple[0]} <head> {triple[1]} <tail> {triple[2]}"
				labels = labels + label
			
			labels = labels.strip()
			
			result.append( { "phrases" : text , "targets" : labels } )
		
		return result

##################################################

class SNIPSLoader():
	
	def __init__(self):
		self._name = "SNIPS"
		self._vmap = {
			'AddToPlaylist' : 'add to playlist',
			'BookRestaurant' : 'book restaurant',
			'GetWeather' : 'get weather',
			'PlayMusic' : 'play music',
			'RateBook' : 'rate book',
			'SearchCreativeWork' : 'search creative work',
			'SearchScreeningEvent' : 'search screening event'
		}
		self._tkns = ["<trip>", "<album>", "<artist>", "<best_rating>", "<city>", "<condition_description>", "<condition_temperature>", "<country>", "<cuisine>", "<current_location>", "<entity_name>", "<facility>", "<genre>", "<geographic_poi>", "<location_name>", "<movie_name>", "<movie_type>", "<music_item>", "<object_location_type>", "<object_name>", "<object_part_of_series_type>", "<object_select>", "<object_type>", "<party_size_description>", "<party_size_number>", "<playlist>", "<playlist_owner>", "<poi>", "<rating_unit>", "<rating_value>", "<restaurant_name>", "<restaurant_type>", "<served_dish>", "<service>", "<sort>", "<spatial_relation>", "<state>", "<timerange>", "<track>", "<year>"]
		
	#--------------------------------------------#
	
	def name(self):
		return self._name

	def tokens(self):
		return self._tkns
	
	#--------------------------------------------#

	def load(self, path, filename, sort = False):
		
		sentences = []
		relations = []
		
		#----------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())
			
		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())
			
		#----------------------------------------#
		
		result = []
		
		for text, rels in zip(sentences, relations):
			
			tables = []
			
			#------------------------------------#
			
			labels = ""
			
			for table in rels.split(" || "):
				table = table.split(" ;; ")
				table_type = self._vmap[table[-1]]
				labels = labels + " <trip> "
				
				for entry in table[:-1]:
					entry = entry.split(" == ")
					labels = labels + f"{entry[1]} <{entry[0].lower()}> "
				
				labels = labels + table_type
				
			labels = labels.strip()
			
			#------------------------------------#
			
			result.append( { "phrases" : text , "targets" : labels } )
		
		return result

##################################################
	
class ATISLoader():
	
	def __init__(self):
		self._name = "ATIS"
		self._vmap = {
			'atis_abbreviation': 'abbreviation',
			'atis_aircraft': 'aircraft',
			'atis_aircraft#atis_flight#atis_flight_no': 'aircraft flight no',
			'atis_airfare': 'airfare',
			'atis_airfare#atis_flight': 'airfare flight',
			'atis_airfare#atis_flight_time': 'airfare flight time',
			'atis_airline': 'airline',
			'atis_airline#atis_flight_no': 'airline flight no',
			'atis_airport': 'airport',
			'atis_capacity': 'capacity',
			'atis_cheapest': 'cheapest',
			'atis_city': 'city',
			'atis_day_name': 'day name',
			'atis_distance': 'distance',
			'atis_flight': 'flight',
			'atis_flight#atis_airfare': 'flight airfare',
			'atis_flight#atis_airline': 'flight airline',
			'atis_flight_no': 'flight no',
			'atis_flight_no#atis_airline': 'flight no airline',
			'atis_flight_time': 'flight time',
			'atis_ground_fare': 'ground fare',
			'atis_ground_service': 'ground service',
			'atis_ground_service#atis_ground_fare': 'ground service fare',
			'atis_meal': 'meal',
			'atis_quantity': 'quantity',
			'atis_restriction': 'restriction'
		}
		self._tkns = ["<trip>", "<aircraft_code>", "<airline_code>", "<airline_name>", "<airport_code>", "<airport_name>", "<arrive_date.date_relative>", "<arrive_date.day_name>", "<arrive_date.day_number>", "<arrive_date.month_name>", "<arrive_date.today_relative>", "<arrive_time.end_time>", "<arrive_time.period_mod>", "<arrive_time.period_of_day>", "<arrive_time.start_time>", "<arrive_time.time>", "<arrive_time.time_relative>", "<booking_class>", "<city_name>", "<class_type>", "<compartment>", "<connect>", "<cost_relative>", "<day_name>", "<day_number>", "<days_code>", "<depart_date.date_relative>", "<depart_date.day_name>", "<depart_date.day_number>", "<depart_date.month_name>", "<depart_date.today_relative>", "<depart_date.year>", "<depart_time.end_time>", "<depart_time.period_mod>", "<depart_time.period_of_day>", "<depart_time.start_time>", "<depart_time.time>", "<depart_time.time_relative>", "<economy>", "<fare_amount>", "<fare_basis_code>", "<flight>", "<flight_days>", "<flight_mod>", "<flight_number>", "<flight_stop>", "<flight_time>", "<fromloc.airport_code>", "<fromloc.airport_name>", "<fromloc.city_name>", "<fromloc.state_code>", "<fromloc.state_name>", "<meal>", "<meal_code>", "<meal_description>", "<mod>", "<month_name>", "<or>", "<period_of_day>", "<restriction_code>", "<return_date.date_relative>", "<return_date.day_name>", "<return_date.day_number>", "<return_date.month_name>", "<return_date.today_relative>", "<return_time.period_mod>", "<return_time.period_of_day>", "<round_trip>", "<state_code>", "<state_name>", "<stoploc.airport_code>", "<stoploc.airport_name>", "<stoploc.city_name>", "<stoploc.state_code>", "<time>", "<time_relative>", "<today_relative>", "<toloc.airport_code>", "<toloc.airport_name>", "<toloc.city_name>", "<toloc.country_name>", "<toloc.state_code>", "<toloc.state_name>", "<transport_type>"]
	
	#--------------------------------------------#
	
	def name(self):
		return self._name

	def tokens(self):
		return self._tkns
	
	#--------------------------------------------#

	def load(self, path, filename, sort = False):
		
		sentences = []
		relations = []
		
		#----------------------------------------#
		
		with open(path + filename + ".sent") as file:
			for line in file:
				sentences.append(line.strip())
			
		with open(path + filename + ".tup") as file:
			for line in file:
				relations.append(line.strip())
			
		#----------------------------------------#
		
		result = []
		
		for text, rels in zip(sentences, relations):
			
			tables = []
			
			#------------------------------------#
			
			labels = ""
			
			for table in rels.split(" || "):
				table = table.split(" ;; ")
				table_type = self._vmap[table[-1]]
				
				labels = labels + " <trip> "
				
				for entry in table[:-1]:
					entry = entry.split(" == ")
					labels = labels + f"{entry[1]} <{entry[0]}> "
				
				labels = labels + table_type
				
			labels = labels.strip()
			
			#------------------------------------#
			
			result.append( { "phrases" : text , "targets" : labels } )
		
		return result