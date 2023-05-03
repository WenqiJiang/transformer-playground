from transformers import BertTokenizer, BertModel
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.to(device)

# text = "Replace me by any text you'd like."
batch_size = 32
text = ["Hello "* (512 - 2)] * batch_size
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input_warmup = tokenizer("warm up 1 2 3", return_tensors='pt')
encoded_input.to(device)
encoded_input_warmup.to(device)

output = model(**encoded_input_warmup)

start = time.time()
output = model(**encoded_input)
end = time.time()

print('time consumption: {} ms'.format((end - start) * 1000))