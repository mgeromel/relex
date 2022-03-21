from transformers import *
import os, random, torch, numpy
import torch.nn.functional as F
import gen_decoder
from gramm import GRAMMAR

##################################################

class TestModel(torch.nn.Module):
    def __init__(self, gramm = None, vocab = None):
        super(TestModel, self).__init__()
        
        self.gramm_size = gramm.size() + 1
        self.relat_size = len(vocab)
        self.point_size = 512
        
        self.config = BertGenerationConfig(
            vocab_size = self.gramm_size + self.relat_size + self.point_size,
            hidden_size = 768,
            num_hidden_layers = 12,
            add_cross_attention = True,
            is_decoder = True,
        )
        
        self.encoder = BertGenerationEncoder.from_pretrained(
            "bert-base-uncased", bos_token_id = 101, eos_token_id = 102
        )
        
        self.decoder = gen_decoder.RelexDecoder(self.config)
        
        self.gramm_head = torch.nn.Linear(self.config.vocab_size, self.gramm_size)
        self.relat_head = torch.nn.Linear(self.config.vocab_size, self.relat_size)
        self.point_head = torch.nn.Linear(self.config.vocab_size, self.point_size)
        
    ##################################################
    
    def compute_loss(self, logits, labels):
        loss = None
        
        if labels != None:
            loss_func = torch.nn.CrossEntropyLoss()

            logitx = logits[:, :-1].contiguous()
            labelx = labels[:,1:  ].contiguous()
            
            g_labels = labelx[:,:,0].view(-1).long()
            r_labels = labelx[:,:,1].view(-1).long()
            p_labels = labelx[:,:,2].view(-1).long()
            q_labels = labelx[:,:,3].view(-1).long()
            
            g_logits = logitx[:,:,     :    9].reshape(-1, self.gramm_size)
            r_logits = logitx[:,:,    9: -512].reshape(-1, self.relat_size)
            p_logits = logitx[:,:, -512: -256].reshape(-1, self.point_size//2)
            q_logits = logitx[:,:, -256:     ].reshape(-1, self.point_size//2)
            
            g_loss = loss_func(g_logits, g_labels)
            r_loss = loss_func(r_logits, r_labels)
            p_loss = loss_func(p_logits, p_labels)
            q_loss = loss_func(q_logits, q_labels)
            
            loss = g_loss + r_loss + p_loss + q_loss
            
        return loss
    
    def compute_loss_alt(self, logits, labels):
        loss = None
        
        if labels != None:
            loss_func = torch.nn.MultiLabelSoftMarginLoss()

            logitx = logits[:, :-1].contiguous()
            labelx = labels[:,1:  ].contiguous()
            
            loss = loss_func(logitx, labelx)
            
        return loss
        
    ##################################################
    
    def forward(self, input_ids = None, attention_mask = None, encoder_outputs = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):
        
        # 1. ENCODER
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids = input_ids,
                attention_mask = attention_mask,
                output_hidden_states = True,
                return_dict = True,
            )
        
        encoder_hidden_states = encoder_outputs[0]
        
        # 2. DECODER 
        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = attention_mask,
            return_dict = False
        )
        
        logits = decoder_outputs[0]
        logits = torch.nn.functional.relu(logits)
        
        # 3. FUNCTION HEADS 
        gramm_logits = self.gramm_head(logits)
        relat_logits = self.relat_head(logits)
        point_logits = self.point_head(logits)
        
        # 4. LOGITS / LOSS
        logits = torch.cat(
            [gramm_logits, relat_logits, point_logits], dim = 2
        )
        
        #loss = self.compute_loss(logits, labels)
        loss = self.compute_loss_alt(logits, decoder_input_ids)
        
        return {
            "loss": loss,
            "logits": logits
        }
    
    ##################################################
    
    def generate(self, input_ids = None, max_length = 64, **kwargs):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        ##################################################
        
        # ENCODER: ONLY ONCE
        attention_mask = input_ids.ne(0).long()
            
        encoder_outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            return_dict = True
        )
        
        ##################################################
            
        batch_size = input_ids.shape[0]                 # NUMBER OF SENTENCES
        input_size = input_ids.shape[1]                 # LENGTH OF SENTENCES
        
        undone = torch.ones(batch_size, device=device)  # GENERATED SEQUENCES (FLAG)

        decoder_input_ids = torch.zeros(
            batch_size, 1, self.config.vocab_size, device=device
        ).float()
        
        decoder_input_ids[:, :, 1] = 1
        decoder_attention_mask = torch.ones(batch_size, 1, device=device)
        
        for length in range(max_length - 1):
            
            ##################################################
            
            decoder_outputs = self.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                encoder_outputs = encoder_outputs,
                decoder_input_ids = decoder_input_ids,
                decoder_attention_mask = decoder_attention_mask
            )
            
            ##################################################
            
            g_logits = decoder_outputs["logits"][:, -1,  :9].detach()
            r_logits = decoder_outputs["logits"][:, -1, 9:-512].detach()
            p_logits = decoder_outputs["logits"][:, -1, -512:].detach()
            
            g_values = torch.argmax(g_logits, dim =-1)
            r_values = torch.argmax(r_logits, dim =-1)
            
            p_values = torch.stack(
                [
                    torch.argmax(p_logits[:, :256], dim=-1),
                    torch.argmax(p_logits[:, 256:], dim=-1) + 256
                ],
                dim=-1
            )
        
            g_logits = torch.zeros(size=g_logits.shape, out=g_logits)
            r_logits = torch.zeros(size=r_logits.shape, out=r_logits)
            p_logits = torch.zeros(size=p_logits.shape, out=p_logits)
            
            ##################################################
            
            for i, (gv, rv, pv) in enumerate(zip(g_values, r_values, p_values)):
                
                if undone[i]:
                    g_logits[i, gv] = 1     # RULE?
                    undone[i] = gv != 2

                    if gv in [4]:
                        r_logits[i, rv] = 1 # RELATION?

                    if gv in [5, 6]:
                        p_logits[i, pv] = 1 # ENTITY?

                else:
                    g_logits[i, 0] = 1
                    
            ##################################################
            
            next_tokens = torch.cat([g_logits, r_logits, p_logits], dim = 1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=1)
            decoder_attention_mask = torch.cat([decoder_attention_mask, undone[:, None]], dim=1)
        
        ##################################################
        
        return decoder_input_ids
