from transformers import BertTokenizerFast
from dataloader import * 

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

load_squad(tokenizer)

print("DONE")
exit(1)

example = ["My name is Bert Bertleton.", "What is my name?"]

model_input = tokenizer(*example, return_tensors = "pt")

input_ids = model_input.input_ids
attention_mask = model_input.attention_mask

output = model.forward(input_ids = input_ids, attention_mask = attention_mask, output_attentions = True)

