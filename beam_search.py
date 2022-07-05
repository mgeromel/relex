from transformers.generation_logits_process import LogitsProcessor
from transformers.gemeration_beam_search import BeamScorer

import torch.nn.functional as F

def beam_search(
	self,
	input_ids = None,
	num_beams = 1,
	max_length = 64,
	output_scores = False,
	output_attentions = False,
	output_hidden_states = False,
	return_dict_in_generate = False,
	**model_kwargs,
):
	##################################################

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	##################################################

	# INITIALIZE VALUES
	pad_token_id = self.config.pad_token_id # ??
	eos_token_id = self.config.eos_token_id # ??

	batch_size = input_ids.shape[0]
	input_size = input_ids.shape[1]

	batch_beam_size = batch_size * num_beams
	
	##################################################
		
	undone = torch.ones(batch_size, device=device)  # GENERATED SEQUENCES (FLAG)
	string = [ "[BOS] #_RELS_#" ] * batch_size
	
	self.mask = self.mask.to(device)

	beam_input_ids = input_ids.repeat(1, 1, num_beams).view(batch_beam_size, input_size)

	##################################################

	beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
	beam_scores[:, 1:] = -1e9
	beam_scores = beam_scores.view((batch_size * num_beams,))

	beam_batch_size

	#------------------------------------------------#

	# ENCODER: ONLY ONCE
	attention_mask = input_ids.ne(0).long()

	encoder_outputs = self.encoder(
		input_ids = input_ids,
		attention_mask = attention_mask,
		output_hidden_states = True,
		return_dict = True
	)

	#------------------------------------------------#

	# DECODER: PREPARE INPUTS
	decoder_input_ids = torch.zeros(
		batch_beam_size, 1, self.vocab_size, device=device
	).float()

	decoder_input_ids[:, :, 1] = 1
	decoder_attention_mask = torch.ones(batch_beam_size, 1, device=device)

	##################################################

	for length in range(max_length - 1):
		
		if sum(undone) == 0:
			break

		#------------------------------------------------#

		decoder_outputs = self.forward(
			input_ids = input_ids,
			attention_mask = attention_mask,
			encoder_outputs = encoder_outputs,
			decoder_input_ids = decoder_input_ids,
			decoder_attention_mask = decoder_attention_mask
		)

		#------------------------------------------------#

		g_logits = decoder_outputs["logits"][:, -1,                   :   self.gramm_size].detach()
		r_logits = decoder_outputs["logits"][:, -1,   self.gramm_size :-2*self.point_size].detach()
		p_logits = decoder_outputs["logits"][:, -1,-2*self.point_size :                  ].detach()

		#------------------------------------------------#

		# "AVERAGE VALUES": 'g_logits'
		g_logits = torch.sum(g_logits.view(batch_size, num_beams, self.gramm_size), dim = 1) / num_beams

		#------------------------------------------------#

		# GRAMMAR MASKING: 'g_logits'

		for idx, string in enumerate(strings):
			if undone[idx]: 
				l_bound = string.find("#_")
				r_bound = string.find("_#") + 2

				if l_bound == -1: # DONE
					continue

				state = string[l_bound : r_bound]

				g_logits[idx][1:] = g_logits[idx][1:] + self.mask[self.smap[state]]
				g_logits[idx][0 ] = -float("inf")

				production = self.gramm.rule(
					torch.argmax(g_logits[idx][1:])
				)

				strings[idx] = strings[idx].replace(state, production[1], 1)

		#------------------------------------------------#

		# CLEANUP / UPSCALE / CLONING:

		g_logits = F.one_hot(torch.argmax(g_logits, dim = -1), self.gramm_size)
		g_logits = g_logits.repeat(1, 1, num_beams).squeeze()
		g_logits = g_logits.view(batch_beam_size, self.gramm_size)

		#------------------------------------------------#

		# BEAM-SEARCH: 'r_logits'

		# shape = (batch_size, num_beams, vocab_size)
		g_logits.view(batch_size, num_beams, self.gramm_size)
		r_logits.view(batch_size, num_beams, self.vocab_size)
		p_logits.view(batch_size, num_beams, self.point_size * 2)

		# shape = (batch_size, num_beams, ?)
		next_scores = F.log_softmax(r_logits, dim = -1)
		beam_scores = beam_scores.view(batch_size, num_beams)

		# FOR EACH BATCH:		
		for idx in range(batch_size):
			if undone[idx]:
				undone[idx] = "[EOS]" not in strings[i]

				if "TOKEN" in string[idx]:
					#------------------------------------#
					# UPDATE BEAM_SCORES
					next_scores[idx] = next_scores[idx] + beam_scores[idx, :, None].expand_as(next_scores)
					
					# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
					
					r_logits[idx] = torch.zeros(size = r_logits[idx].shape, out = r_logits[idx])
					p_logits[idx] = torch.zeros(size = p_logits[idx].shape, out = p_logits[idx])

					string[idx] = string[idx].replace("TOKEN", "token")
					#------------------------------------#

				if "POINT" in string[idx]:
					#------------------------------------#
					l_index, r_index = p_logits[idx].split(self.point_size, dim = -1)
					
					l_index = F.one_hot(torch.argmax(l_index, dim = -1), self.point_size)
					r_index = F.one_hot(torch.argmax(r_index, dim = -1), self.point_size)

					p_logits[idx] = torch.cat([l_index, r_index], dim = -1)
					r_logits[idx] = torch.zeros(size=r_logits[idx].shape, out=r_logits[idx])
 
					string[idx] = string[idx].replace("POINT", "point")
					#------------------------------------#
			else:
				# BATCH_INPUT_IDS: PADDING
				g_logits[idx] = F.one_hot(torch.tensor(0), self.gramm_size).repeat(num_beams, 1)
				r_logits[idx] = torch.zeros(size = r_logits[idx].shape, out = r_logits[idx])
				p_logits[idx] = torch.zeros(size = p_logits[idx].shape, out = p_logits[idx])

		#------------------------------------------------#

		# reshape for beam search
		vocab_size = next_token_scores.shape[-1]
		next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

		next_token_scores, next_tokens = torch.topk(
			next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
		)

		next_indices = torch_int_div(next_tokens, vocab_size)
		next_tokens = next_tokens % vocab_size

		# stateless
		beam_outputs = beam_scorer.process(
			input_ids,
			next_token_scores,
			next_tokens,
			next_indices,
			pad_token_id=pad_token_id,
			eos_token_id=eos_token_id,
			beam_indices=beam_indices,
		)

		beam_scores = beam_outputs["next_beam_scores"]
		beam_next_tokens = beam_outputs["next_beam_tokens"]
		beam_idx = beam_outputs["next_beam_indices"]

		decoder_input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
		decoder_attention_mask


	##################################################
	
	sequence_outputs = beam_scorer.finalize(
		input_ids,
		beam_scores,
		next_tokens,
		next_indices,
		pad_token_id=pad_token_id,
		eos_token_id=eos_token_id,
		max_length=stopping_criteria.max_length,
		beam_indices=beam_indices,
	)

	##################################################
	
	if return_dict_in_generate:
		if not output_scores:
			sequence_outputs["sequence_scores"] = None

		if self.config.is_encoder_decoder:
			return BeamSearchEncoderDecoderOutput(
				sequences=sequence_outputs["sequences"],
				sequences_scores=sequence_outputs["sequence_scores"],
				scores=scores,
				beam_indices=sequence_outputs["beam_indices"],
				encoder_attentions=encoder_attentions,
				encoder_hidden_states=encoder_hidden_states,
				decoder_attentions=decoder_attentions,
				cross_attentions=cross_attentions,
				decoder_hidden_states=decoder_hidden_states,
			)
		else:
			return BeamSearchDecoderOnlyOutput(
				sequences=sequence_outputs["sequences"],
				sequences_scores=sequence_outputs["sequence_scores"],
				scores=scores,
				beam_indices=sequence_outputs["beam_indices"],
				attentions=decoder_attentions,
				hidden_states=decoder_hidden_states,
			)
	else:
		return sequence_outputs["sequences"]