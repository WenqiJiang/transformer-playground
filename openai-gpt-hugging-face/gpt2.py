from transformers import GPT2Tokenizer, GPT2Model
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')

model.to(device)

out_length = 1024 - 1
batch_size = 1
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
inputs = tokenizer(["Hey " * out_length] * batch_size, return_tensors="pt")
inputs_dummy = tokenizer("Hello " * 512, return_tensors="pt")
inputs.to(device)
inputs_dummy.to(device)
outputs = model(**inputs_dummy) # warm up

start = time.time()
outputs = model(**inputs)
end = time.time()

last_hidden_states = outputs.last_hidden_state
print('shape last_hidden_states', last_hidden_states.shape)
print('time consumption: {} ms ({} us per step)'.format((end - start) * 1000, (end - start) * 1e6 / out_length))

