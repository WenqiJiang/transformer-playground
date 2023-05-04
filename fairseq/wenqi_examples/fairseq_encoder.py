import torch 
import time

from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.data import Dictionary
from fairseq.models.transformer import TransformerConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.encoder = { \
    "embed_dim" : 1024, 
    "ffn_embed_dim": 4096, 
    "layers" : 12,
    "attention_heads" : 16}

cfg = TransformerConfig.from_namespace(args)
print(f"The config created from args: {cfg}")

# vocab to vocab ID
dictionary = Dictionary()

# Input embeddings
vocab_size = 10
enc_embs = torch.nn.Embedding(vocab_size, args.encoder["embed_dim"], dictionary.pad())

model = TransformerEncoder(args, dictionary, enc_embs)
print(model)

batch_size = 2
seq_len = 512
input_tokens = torch.tensor([[1] * seq_len] * batch_size)

# enc_embs.to(device)
# model.to(device)
# input_tokens.to(device)

"""
encoder output dict:
    - **encoder_out** (Tensor): the last encoder layer's output of
        shape `(src_len, batch, embed_dim)`
    - **encoder_padding_mask** (ByteTensor): the positions of
        padding elements of shape `(batch, src_len)`
    - **encoder_embedding** (Tensor): the (scaled) embedding lookup
        of shape `(batch, src_len, embed_dim)`
    - **encoder_states** (List[Tensor]): all intermediate
        hidden states of shape `(src_len, batch, embed_dim)`.
        Only populated if *return_all_hiddens* is True.
"""

out_dict = model(input_tokens)

with torch.no_grad():
	start = time.time()
	out_dict = model(input_tokens)
	end = time.time()
                  
print('\noutput:\n', out_dict)
print('\nencoder_out:\n', out_dict['encoder_out'], out_dict['encoder_out'][0].shape)
print('\nencoder_padding_mask:\n', out_dict['encoder_padding_mask'], out_dict['encoder_padding_mask'][0].shape)
print('\nencoder_embedding:\n', out_dict['encoder_embedding'], out_dict['encoder_embedding'][0].shape)
print('\nencoder_states:\n', out_dict['encoder_states'])
print('time consumption: {} ms'.format((end - start) * 1000))

