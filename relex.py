import transformers, datasets, pandas, torch, numpy
from utils import *
from modex import *

import gramm, loadr, tqdm
import torch.nn as nn

from transformers import AdamW, get_scheduler
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer 
from torch.utils.data import DataLoader

#-----------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def translate(batch):
    
    result = []
    
    # batch_size x max_length x vocab_size
    
    for sample in batch:
        
        labels = []
        
        temp_relation = None
        temp_entity_a = None
        temp_entity_b = None
        
        for token in sample:
            rule = torch.argmax(token[:9])
            
            if rule == 4 and not temp_relation:
                temp_relation = torch.argmax(token[9 : 9 + 29]).item()
            
            if rule == 5 and temp_entity_a is None:
                temp_entity_a = (
                    torch.argmax(token[-512:-256]).item(),
                    torch.argmax(token[-256:    ]).item()
                )
            
            if rule == 6 and temp_entity_b is None:
                temp_entity_b = (
                    torch.argmax(token[-512:-256]).item(),
                    torch.argmax(token[-256:    ]).item()
                )
            
            if rule == 2 or rule == 3 and temp_relation is not None:
                
                labels.append( (temp_relation, temp_entity_a, temp_entity_b) )
                
                temp_relation = None
                temp_entity_a = None
                temp_entity_b = None
        
        result.append(labels)
        
    return result

#-----------------------------------------------------------#

gramm = gramm.GRAMMAR("gramm.txt")
vocab = read_file("data/NYT/relations.txt")
vocab = dict(zip(vocab, range(len(vocab))))

#-----------------------------------------------------------#

EPOCHS = 2

train_size = 4000
valid_size = 80
batch_size = 8

#-----------------------------------------------------------#

data_train = loadr.MyDataset("data/NYT/train", train_size, gramm, vocab)
data_valid = loadr.MyDataset("data/NYT/valid", valid_size, gramm, vocab)

train_loader = DataLoader(data_train, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(data_valid, batch_size = batch_size)   

#-----------------------------------------------------------# 

model = TestModel(gramm = gramm, vocab = vocab)

for param in model.encoder.parameters():
    param.requires_grad = False

model.to(device)

optimizer = AdamW(model.parameters(), lr = 0.00005)
scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 0,
    num_training_steps = EPOCHS * len(train_loader),
)

labels = []
predic = []

#-----------------------------------------------------------#

train_bar = tqdm.tqdm(total = EPOCHS * len(train_loader), leave = False, position = 0, desc = "TRAIN")
valid_bar = tqdm.tqdm(total = EPOCHS * len(valid_loader), leave = False, position = 1, desc = "VALID")

for epoch in range(EPOCHS):
    
    #------------------------------------------------------#
    
    model.train()
    for batch in train_loader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        output = model.forward(**batch)
		
        loss = output["loss"]
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        train_bar.update(1)
        
        if(train_bar.n % 10 == 0):
            l = round(loss.item(), 4)
            r = scheduler.get_last_lr()
            train_bar.write(f"loss: {l} \t rate: {r}")
    
    #------------------------------------------------------#
    
    optimizer.zero_grad()
    
    model.eval()
    for batch in valid_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            output = model.generate(
                input_ids = batch["input_ids"],
                max_length = 64
            )
        
        labels.extend(translate(batch["decoder_input_ids"]))
        predic.extend(translate(output))
        
        valid_bar.update(1)
    
    metric_scores = micro_compute({"label_ids": labels, "predicted": predic})
    valid_bar.write(str(metric_scores))
    
    for i in range(15):
        print(f"labels: {labels[i]}\npredic: {predic[i]}\n")
    
    labels = []
    predic = []

#-----------------------------------------------------------#

import IPython ; IPython.embed() ; exit(1)

exit(1)

torch.save(model.state_dict(), "models/relex_greedy_semi.pt")
