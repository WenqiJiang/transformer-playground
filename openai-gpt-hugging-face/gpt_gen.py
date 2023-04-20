from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='openai-gpt')
set_seed(42)
results = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

print(results)
