from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt")
model.to(device)

out_length = 512
batch_size = 64
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
inputs = tokenizer(["Hello " * out_length] * batch_size, return_tensors="pt")
inputs_dummy = tokenizer("Hello " * 512, return_tensors="pt")
inputs.to(device)
inputs_dummy.to(device)
outputs = model(**inputs_dummy) # warm up

start = time.time()
outputs = model(**inputs)
end = time.time()

last_hidden_states = outputs.last_hidden_state
print('shape last_hidden_states', last_hidden_states.shape)
print('time consumption: {} sec'.format(end - start))

