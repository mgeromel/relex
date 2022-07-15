import torch.nn.functional as F

	def beam_search(self, input_ids = None, num_beams = 1, max_length = 64, **model_kwargs,):

		######################################################

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		######################################################

		# INITIALIZE VALUES
		batch_size = input_ids.shape[0] # ...
		input_size = input_ids.shape[1] # SEQUENCE LENGTH

		# "NEW" BATCH_SIZE
		batch_beam_size = batch_size * num_beams
		
		# SYNCHRONIZED G_LOGITS --> "BATCH_SIZE" 
		undone = torch.ones(batch_size, device=device)  # GENERATED SEQUENCES (FLAG)
		strings = [ "[BOS] #_RELS_#" ] * batch_size # batch_size <--> batch_beam_size
		
		# MASK FOR G_LOGITS
		self.mask = self.mask.to(device)

		# TABLE FOR BEAM-SCORES
		beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
		beam_scores[:, 1:] = -1e9 # MASKING, 1ST ITERATION
		
		#----------------------------------------------------#
		# ENCODER: ONLY ONCE

		encoder_inputs = input_ids.repeat(1, num_beams).view(batch_size * num_beams, input_size)
		attention_mask = encoder_inputs.ne(0).long()

		encoder_outputs = self.encoder(
			input_ids = encoder_inputs,
			attention_mask = attention_mask,
			output_hidden_states = True,
			return_dict = True
		)

		#----------------------------------------------------#
		# DECODER: PREPARE INPUTS

		decoder_input_ids = torch.zeros(
			batch_beam_size, 1, self.vocab_size, device=device
		).float()

		decoder_input_ids[:, :, 1] = 1
		decoder_attention_mask = torch.ones(batch_beam_size, 1, device=device)

		######################################################

		for length in range(max_length - 1):

			#------------------------------------------------#
			# DECODER FORWARD-PASS

			decoder_outputs = self.forward(
				input_ids = input_ids,
				attention_mask = attention_mask,
				encoder_outputs = encoder_outputs,
				decoder_input_ids = decoder_input_ids,
				decoder_attention_mask = decoder_attention_mask
			)

			#------------------------------------------------#
			# RETRIEVE LOGITS

			g_logits = decoder_outputs["logits"][:, -1,                   :   self.gramm_size].detach()
			r_logits = decoder_outputs["logits"][:, -1,   self.gramm_size :-2*self.point_size].detach()
			p_logits = decoder_outputs["logits"][:, -1,-2*self.point_size :                  ].detach()

			#------------------------------------------------#
			# "AVERAGE VALUES": 'G_LOGITS'
			
			g_logits = torch.sum(g_logits.view(batch_size, num_beams, self.gramm_size), dim = 1) / num_beams

			#------------------------------------------------#
			# GRAMMAR MASKING: 'G_LOGITS'

			for idx, sequence in enumerate(strings):
				if undone[idx]: 
					l_bound = sequence.find("#_")
					r_bound = sequence.find("_#") + 2

					if l_bound == -1: # DONE / ERROR
						continue

					state = sequence[l_bound : r_bound]

					g_logits[idx][1:] = g_logits[idx][1:] + self.mask[self.smap[state]]
					g_logits[idx][0 ] = -float("inf")

					production = self.gramm.rule(torch.argmax(g_logits[idx][1:]))
					strings[idx] = strings[idx].replace(state, production[1], 1)

			#------------------------------------------------#
			# CLEANUP / UPSCALE / CLONING:
			
			g_logits = F.one_hot(torch.argmax(g_logits, dim = -1), self.gramm_size)
			g_logits = g_logits.repeat(1, num_beams)

			#------------------------------------------------#
			# BEAM-SEARCH: 'r_logits'

			# shape = (batch_size, num_beams, vocab_size)
			g_logits = g_logits.view(batch_size, num_beams, self.gramm_size)
			r_logits = r_logits.view(batch_size, num_beams, self.relat_size)
			p_logits = p_logits.view(batch_size, num_beams, self.point_size * 2)

			# decoder_input_ids += (g_logits + r_logits + p_logits)
			decoder_input_ids = decoder_input_ids.view(batch_size, num_beams, length + 1, self.vocab_size)

			# FOR EACH SAMPLE:		
			for idx in range(batch_size): # batch_size <--> batch_beam_size
				if undone[idx]:
					undone[idx] = "[EOS]" not in strings[idx]

					if "TOKEN" not in strings[idx] and "POINT" not in strings[idx]:
						#------------------------------------#
						# STRUCTURE-STEP
						r_logits[idx] = 0 # torch.zeros(size = r_logits[idx].shape, out = r_logits[idx])
						p_logits[idx] = 0 # torch.zeros(size = p_logits[idx].shape, out = p_logits[idx])
						#------------------------------------#

					if "TOKEN" in strings[idx]:
						#------------------------------------#
						# CURRENT SCORES
						next_scores = F.log_softmax(r_logits[idx], dim = -1)

						# TOP-K: SCORES / TOKENS
						best_scores = next_scores + beam_scores[idx, :, None].expand_as(next_scores)
						best_scores, best_tokens = torch.topk(best_scores.view(-1), num_beams, dim = 0)

						# BEAM-INDEX / NEXT-TOKEN
						beam_number = torch.div(best_tokens, self.relat_size, rounding_mode = "floor")
						best_tokens = best_tokens % self.relat_size

						# RE-ORDERING OF BEAMS 
						decoder_input_ids[idx] = decoder_input_ids[idx, beam_number]

						# UPDATE BEAM SCORES
						beam_scores[idx] = best_scores

						# UPDATE DECODER_INPUT_IDS
						r_logits[idx] = F.one_hot(best_tokens, self.relat_size)
						p_logits[idx] = 0 # torch.zeros(size = p_logits[idx].shape, out = p_logits[idx])

						# CLEANUP
						strings[idx] = strings[idx].replace("TOKEN", "token")
						#------------------------------------#

					if "POINT" in strings[idx]:
						#------------------------------------#
						# SLICE P_LOGITS IN HALF
						l_index, r_index = p_logits[idx].split(self.point_size, dim = -1)
						
						# L_INDEX / R_INDEX
						l_index = F.one_hot(torch.argmax(l_index, dim = -1), self.point_size)
						r_index = F.one_hot(torch.argmax(r_index, dim = -1), self.point_size)

						# UPDATE DECODER_INPUT_IDS
						p_logits[idx] = torch.cat([l_index, r_index], dim = -1)
						r_logits[idx] = 0 # torch.zeros(size=r_logits[idx].shape, out=r_logits[idx])

	 					# CLEANUP
						strings[idx] = strings[idx].replace("POINT", "point")
						#------------------------------------#
				else:
					# BATCH_INPUT_IDS: PADDING
					g_logits[idx] = F.one_hot(torch.tensor([0] * num_beams), self.gramm_size)
					r_logits[idx] = 0 # torch.zeros(size = r_logits[idx].shape, out = r_logits[idx])
					p_logits[idx] = 0 # torch.zeros(size = p_logits[idx].shape, out = p_logits[idx])

			#------------------------------------------------#
			# DECODER_INPUT_IDS
			next_decoder_input_ids = torch.cat([g_logits, r_logits, p_logits], dim = -1)
			decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids[:, :, None]], dim = 2)
			decoder_input_ids = decoder_input_ids.view(batch_beam_size, length + 2, self.vocab_size)

			# DECODER_ATTENTION_MASK
			next_decoder_attention = undone.view(batch_size, 1).repeat(1, num_beams)
			next_decoder_attention = next_decoder_attention.view(batch_size * num_beams, 1)
			decoder_attention_mask = torch.cat([decoder_attention_mask, next_decoder_attention], dim = -1)
			#------------------------------------------------#
			# EARLY STOPPING?
			if sum(undone) == 0:
				break
			#------------------------------------------------#

		##################################################
		# FINALIZE OUTPUT_SEQUENCE

		length = decoder_input_ids.shape[1]
		decoder_input_ids = decoder_input_ids.view(batch_size, num_beams, length, self.vocab_size)

		if torch.sum(torch.topk(beam_scores, 1, dim = -1).indices) != 0:
			import IPython ; IPython.embed() ; exit(1)

		decoder_input_ids = decoder_input_ids[:, 0, :, :] # BEST SCORE AT '0'

		##################################################
		
		return decoder_input_ids