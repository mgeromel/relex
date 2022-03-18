from transformers import BertTokenizerFast, EncoderDecoderModel
import datasets, pandas, torch, numpy

from utils import *
from modex import *
from gramm import *

print("> Done: Importing")

#-----------------------------------------------------------#
#-----------------------------------------------------------#

# Generating: [tok_1, ..., tok_n ] -> [rel_1, ..., rel_k]
def generate(batch):
    tokens = tokenizer(
        batch["phrases"],
        padding = "max_length",
        truncation = True,
        max_length = 256,
        return_tensors = "pt"
    )

    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    
    batch["predicted"] = model.generate(
        input_ids=input_ids,
        #attention_mask=attention_mask
    ).numpy()
    
    return batch

#-----------------------------------------------------------#
#-----------------------------------------------------------#

# Read: Grammar & Vocabulary
gramm = GRAMMAR("gramm.txt")
vocab = read_file("data/relations.txt")
vocab = dict(zip(vocab, range(len(vocab))))

#-----------------------------------------------------------#

# Reading: Sentences & Relations
test_size = 10

test_sent = read_file("data/test.sent")[:test_size]
test_tups = read_file("data/test.tup")[:test_size]
test_tups = [extract(rel, sent, gramm, vocab) for rel, sent in zip(test_tups, test_sent)]

cols_test = list(zip(test_sent, test_tups))
data_test = pandas.DataFrame(cols_test, columns = ["phrases", "targets"])
data_test = datasets.Dataset.from_pandas(data_test)

print("> Done: Reading & Converting")

#-----------------------------------------------------------#
#-----------------------------------------------------------#

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = load_model("relex_greedy_semi", gramm, vocab)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

model.config.do_sample = False
model.config.num_beams = 1
model.config.num_beam_groups = 1

model.config.min_length = 0
model.config.max_length = 64

# SETUP: CONFIG
model.config.pad_token = 0
model.config.bos_token = 1
model.config.eos_token = 2
model.config.rel_token = 7
model.config.ent_token = 8

print("> Done: Loading Model")

#-----------------------------------------------------------#
#-----------------------------------------------------------#

print("> Begin Testing")

batch_size = 4

outputs = data_test.map(
    generate,
    batched = True,
    batch_size = batch_size,
    remove_columns=["phrases"]
)

micro_score = micro_compute(outputs)
macro_score = macro_compute(outputs)

import IPython ; IPython.embed() ; exit(0)

with open("test_log.txt", "a") as log:
    log.write(f'precis: {precis} \n')
    log.write(f'recall: {recall} \n')
    log.write(f'fscore: {fscore} \n')

#-----------------------------------------------------------#
#-----------------------------------------------------------#
