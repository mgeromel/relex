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

##################################################

non = 0

# Extracting Relations
def extract(labels, sentence, gramm, vocab):
    
    size = 1 + gramm.size() + len(vocab) + 512
    tuples = []
    
    sorted_labels = labels.split("|")
    sorted_labels.sort(key = lambda x : x.split(";")[2].strip())
    
    for label in sorted_labels:
        words = [word.strip() for word in label.split(";")]
        
        entity_a = find(words[0], sentence, 0)
        entity_b = find(words[1], sentence, 0)
        relation = vocab[words[2]]
        
        tuples.append( (3, -100, -100, -100) )
        tuples.append( (4, relation, -100, -100) )
        tuples.append( (5, -100, entity_a[0], entity_a[1]) )
        tuples.append( (6, -100, entity_b[0], entity_b[1]) )
    
    return [(1, -100, -100, -100)] + tuples + [(2, -100, -100, -100)]

def extract_2(labels, sentence, gramm, vocab):
    
    size = 1 + gramm.size() + len(vocab) + 512
    tuples = []
    
    for label in labels.split("|"):
        words = [word.strip() for word in label.split(";")]
        
        entity_a = find(words[0], sentence, 0)
        entity_b = find(words[1], sentence, 0)
        relation = vocab[words[2]]
        
        tuples.append( (3, -100, -100, -100) )
        tuples.append( (4, -100, -100, -100) )
        tuples.append( (-100, relation, -100, -100) )
        tuples.append( (5, -100, -100, -100) )
        tuples.append( (-100, -100, entity_a[0], -100) )
        tuples.append( (-100, -100, -100, entity_a[1]) )
        tuples.append( (6, -100, -100, -100) )
        tuples.append( (-100, -100, entity_b[0], -100) )
        tuples.append( (-100, -100, -100, entity_b[1]) )
    
    return [(1, -100, -100, -100)] + tuples + [(2, -100, -100, -100)]

#------------------------------------------------#

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

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
    labels = [l for l in labels if l != 0]
    predic = [p for p in predic if p != 0]
    
    result = list((Counter(predic) & Counter(labels)).elements())
    
    if (len(predic) == 0):
        return 1, 0

    precis = len(result) / len(predic)
    recall = len(result) / len(labels)

    return precis, recall

#-----------------------------------------------------------#
    
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

##################################################