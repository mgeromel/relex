from transformers.models.bert_generation.modeling_bert_generation import BertGenerationEmbeddings, BertGenerationDecoder, BertGenerationEncoder

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

import torch

#-----------------------------------------------------------#

class RelexEmbedding(BertGenerationEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = torch.nn.Linear(config.vocab_size, config.hidden_size)
    
    def forward(
        self, input_ids = None, position_ids = None,
        inputs_embeds=None, past_key_values_length=0           
    ):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        inputs_embeds = self.word_embeddings(input_ids)
        
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

#-----------------------------------------------------------#

class RelexDecoder(BertGenerationDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.bert = RelexEncoder(config)

#-----------------------------------------------------------#

class RelexEncoder(BertGenerationEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = RelexEmbedding(config)
    
    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None,
        inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
        past_key_values=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None,
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()[:2]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        extended_attention_mask = None
        if not use_cache:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

#-----------------------------------------------------------#