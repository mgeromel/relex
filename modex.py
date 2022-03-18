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
            num_hidden_layers = 4,
            add_cross_attention = True,
            is_decoder = True,
            
            decoder_start_token_id = 1,
            
            pad_token_id = 0,
            bos_token_id = 1,
            eos_token_id = 2,
        )
        
        self.encoder = BertGenerationEncoder.from_pretrained(
            "bert-base-uncased", bos_token_id = 101, eos_token_id = 102
        )
        
        self.decoder = gen_decoder.RelexDecoder(self.config)
        
        #self.layer_1 = torch.nn.Linear(self.config.vocab_size, 1024)
        #self.layer_2 = torch.nn.Linear(1024, self.config.vocab_size)
        
        self.gramm_head = torch.nn.Linear(self.config.vocab_size, self.gramm_size)
        self.relat_head = torch.nn.Linear(self.config.vocab_size, self.relat_size)
        self.point_head = torch.nn.Linear(self.config.vocab_size, self.point_size)
        
    ##################################################
    
    def compute_loss(self, logits, labels):
        loss = None
        
        # TODO !!
        
        if labels != None:
            loss = 0
            
            batch_size = labels.shape[0]
        
            loss_func = torch.nn.CrossEntropyLoss()

            logits = logits[:,:-1,:].contiguous()
            labels = labels[:, 1 : ].contiguous()
            
            for log, lab in zip(logits, labels):
                
                gramm_labels = lab[:, 0].long()
                relat_labels = lab[:, 1].long()
                
                gramm_logits = log[:,    :9   ]
                relat_logits = log[:,   9:-512]
                
                point_labels_1 = lab[:,2].long()
                point_labels_2 = lab[:,3].long()
                
                point_logits_1 = log[:,-512:-256]
                point_logits_2 = log[:,-256:    ]
                
                # FILTER for R
                relat_logits = relat_logits[gramm_labels == 4]
                relat_labels = relat_labels[gramm_labels == 4]
                
                # FILTER for P
                point_mask = torch.where(gramm_labels == 5, 1, 0) + torch.where(gramm_labels == 6, 1, 0)     
                point_logits_1 = point_logits_1[point_mask == 1]
                point_labels_1 = point_labels_1[point_mask == 1]
                point_logits_2 = point_logits_2[point_mask == 1]
                point_labels_2 = point_labels_2[point_mask == 1]
                
                # FILTER for G
                gramm_logits = gramm_logits[gramm_labels != 0]
                gramm_labels = gramm_labels[gramm_labels != 0]
                
                # COMPUTE the LOSS
                gramm_loss = loss_func(gramm_logits, gramm_labels)
                relat_loss = loss_func(relat_logits, relat_labels)
                point_loss_1 = loss_func(point_logits_1, point_labels_1)
                point_loss_2 = loss_func(point_logits_2, point_labels_2)
                
                loss = loss + (gramm_loss + relat_loss + point_loss_1 + point_loss_2)
            
            loss = loss / batch_size
            
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
        
        logits = decoder_outputs[0] #.logits
        
        # 3. FUNCTION HEADS 
        #gramm_logits = self.gramm_head(logits)
        #relat_logits = self.relat_head(logits)
        #point_logits = self.point_head(logits)
        
        # 4. LOGITS / LOSS
        #logits = torch.cat(
        #    [gramm_logits, relat_logits, point_logits], dim = 2
        #)
        
        loss = self.compute_loss(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "g_logits": logits[:,:,   :9 ], #gramm_logits,
            "r_logits": logits[:,:,9  :29], #relat_logits,
            "p_logits": logits[:,:, 29:  ], #point_logits
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
            
            g_logits = decoder_outputs["g_logits"][:, -1, :].detach()
            r_logits = decoder_outputs["r_logits"][:, -1, :].detach()
            p_logits = decoder_outputs["p_logits"][:, -1, :].detach()
            
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
                
                print(f"({i}, ({gv}, {rv}, {pv}))")
                
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