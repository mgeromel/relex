import torch, numpy, os, random

from transformers import BertTokenizerFast
from collections import Counter
    
##################################################

# Reading Lines of File
def read_file(filename):
    lines = []
    
    with open(filename) as file:
        for line in file.readlines():
            lines.append(line.strip())
            
    return lines

def to_list(nums, size):
    liste = [0] * size
    
    for n in nums:
        if n in range(size):
            liste[n] = 1
        
    return liste


##################################################

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Extracting Relations
def extract(labels, sentence, gramm, vocab):
    
    size = 1 + gramm.size() + len(vocab) + 512
    tuples = []
    
    for label in labels.split("|"):
        words = [word.strip() for word in label.split(";")]
        
        entity_a = find(words[0], sentence, 0)
        entity_b = find(words[1], sentence, 0)
        relation = vocab[words[2]]
        
        tuples.append( (3, 0, 0, 0) )
        tuples.append( (4, relation, 0, 0) )
        tuples.append( (5, 0, entity_a[0], entity_a[1]) )
        tuples.append( (6, 0, entity_b[0], entity_b[1]) )
    
    return [(1, 0, 0, 0)] + tuples + [(2, 0, 0, 0)]

#------------------------------------------------#

def find(word, sent, offset):
    w_tokens = tokenizer(word, add_special_tokens = False).input_ids
    s_tokens = tokenizer(sent, add_special_tokens = True).input_ids
    
    # Find W_TOKENS in S_TOKENS 
    for i in range(len(s_tokens) - len(w_tokens)):
        if w_tokens == s_tokens[i : i + len(w_tokens)]:
            return [i + offset, i + offset + len(w_tokens)]
    
    return (-1, -1)

##################################################

def compute(predic, labels):
    result = list((Counter(predic) & Counter(labels)).elements())
    
    if (len(predic) == 0):
        return 1, 0

    precis = len(result) / len(predic)
    recall = len(result) / len(labels)

    return precis, recall

#------------------------------------------------#

def compute_metrics(pred):
    labels = pred.label_ids
    predic = pred.predictions

    recall = [0] * len(labels)
    precis = [0] * len(labels)

    for x in range(len(labels)):
        precis[x], recall[x] = compute(predic[x], labels[x])

    recall = sum(recall) / len(labels)
    precis = sum(precis) / len(labels)
    fscore = (2 * precis * recall) / (precis + recall)
    
    return {
        "recall": round(recall, 4),
        "precision": round(precis, 4),
        "f_measure": round(fscore, 4),
    }

##################################################
    
def micro_compute(pred):
    
    labels = pred["label_ids"]
    predic = pred["predicted"]
    
    micro_value = micro_score(predic, labels)
    
    return {
        "micro_recall": round(micro_value[0], 4),
        "micro_precis": round(micro_value[1], 4),
        "micro_fscore": round(micro_value[2], 4),
    }

#-----------------------------------------------------------#

def macro_compute(pred):
    labels = pred.label_ids
    predic = pred.predictions
    
    macro_value = macro_score(predic, labels)
    
    return {
        "macro_recall": round(macro_value[0], 4),
        "macro_precis": round(macro_value[1], 4),
        "macro_fscore": round(macro_value[2], 4),
    }

#-----------------------------------------------------------#

def micro_score(predic, labels):
    recall = [0] * len(labels)
    precis = [0] * len(labels)
    
    for x in range(len(labels)):
        precis[x], recall[x] = compute(predic[x], labels[x])
        
    recall = sum(recall) / len(labels)
    precis = sum(precis) / len(labels)
    
    if precis + recall > 0:
        fscore = (2 * precis * recall) / (precis + recall)
    else:
        fscore = 0
        
    return (recall, precis, fscore)

#-----------------------------------------------------------#

def macro_score(predic, labels):
    recall = [0] * len(labels)
    precis = [0] * len(labels)
    
    for x in range(len(labels)):
        p = [l for l in predic[x] if l > 8 and l <= 8 + 29]
        l = [l for l in labels[x] if l > 8 and l <= 8 + 29]
        
        precis[x], recall[x] = compute(p, l)
        
    recall = sum(recall) / len(labels)
    precis = sum(precis) / len(labels)
    fscore = (2 * precis * recall) / (precis + recall)
    
    return (recall, precis, fscore)
