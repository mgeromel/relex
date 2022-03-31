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
        
        relation, entity_a, entity_b = None, None, None
        
        for token in sample:
            rule = torch.argmax(token[:9])
            
            if rule == 4 and relation is None:
                relation = torch.argmax(token[9 : 9 + 29]).item()
            
            if rule == 5 and entity_a is None:
                entity_a = (
                    torch.argmax(token[-512:-256]).item(),
                    torch.argmax(token[-256:    ]).item()
                )
            
            if rule == 6 and entity_b is None:
                entity_b = (
                    torch.argmax(token[-512:-256]).item(),
                    torch.argmax(token[-256:    ]).item()
                )
            
            if (rule == 2 or rule == 3) and not (None in [relation, entity_a, entity_b]):
                labels.append( (relation, entity_a, entity_b) )
                relation, entity_a, entity_b = None, None, None
        
        result.append(labels)
        
    return result

def convert(values, g_size = 9, r_size = 29, p_size = 256):
    
    # FLOAT -> INT
    values = values.int()
    
    # INITIALIZE
    g_list = [0] * g_size
    r_list = [0] * r_size
    p_list = [0] * p_size
    q_list = [0] * p_size
    
    # PADDING
    g_list[0] = int(values[0] == -100)
    
    # SETTING VALUES
    if values[0] != -100:
        g_list[values[0]] = 1
    
    if values[1] != -100:
        r_list[values[1]] = 1
    
    if values[2] != -100:
        p_list[values[2]] = 1
    
    if values[3] != -100:
        q_list[values[3]] = 1
        
    return g_list + r_list + p_list + q_list

#-----------------------------------------------------------#

gramm = gramm.GRAMMAR("gramm.txt")
vocab = read_file("data/NYT/relations.txt")
vocab = dict(zip(vocab, range(len(vocab))))

#-----------------------------------------------------------#

EPOCHS = 2

train_size = 800
valid_size = 16
batch_size = 16

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

optimizer = AdamW(model.parameters(), lr = 10e-5)
scheduler = get_scheduler(
    "linear",
    optimizer = optimizer,
    num_warmup_steps = 500,
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
        
        batch["decoder_input_ids"] = torch.Tensor(
            [[ convert(tup) for tup in sample ] for sample in batch["labels"]]
        ).float()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        output = model.forward(**batch)
        loss = output["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_bar.update(1)
        
        if(train_bar.n % 10 == 0):
            l = round(loss.item(), 4)
            r = scheduler.get_last_lr()
            train_bar.write(f"loss: {l} \t rate: {r}")
    
    #------------------------------------------------------#
    
    model.eval()
    for batch in valid_loader:
        
        batch["decoder_input_ids"] = torch.Tensor(
            [[ convert(tup) for tup in sample ] for sample in batch["labels"]]
        ).float()

        with torch.no_grad():
            output = model.generate(
                input_ids = batch["input_ids"].to(device),
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

exit(1)

torch.save(model.state_dict(), "models/relex_greedy_semi.pt")