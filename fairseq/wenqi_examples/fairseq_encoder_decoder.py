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

args.decoder = { \
    "embed_dim" : 1024, 
    "ffn_embed_dim": 4096, 
    "layers" : 12,
    "attention_heads" : 16}

cfg = TransformerConfig.from_namespace(args)
print(f"The config created from args: {args}")

# vocab to vocab ID
dictionary = Dictionary()

# Input embeddings
vocab_size = 10
enc_embs = torch.nn.Embedding(vocab_size, args.encoder["embed_dim"], dictionary.pad())
dec_embs = torch.nn.Embedding(vocab_size, args.encoder["embed_dim"], dictionary.pad())

model_encoder = TransformerEncoder(args, dictionary, enc_embs)
model_decoder = TransformerDecoder(args, dictionary, enc_embs, no_encoder_attn=False)
print(f"Model Encoder: {model_encoder}")
print(f"Model Decoder: {model_decoder}")

batch_size = 1
seq_len = 512
input_tokens = torch.tensor([[1] * seq_len] * batch_size)

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

with torch.no_grad():
    start = time.time()
    encoder_out_dict = model_encoder(input_tokens)
    end = time.time()
    print('Encoder time consumption: {} ms'.format((end - start) * 1000))

    out_seq_len = 64
    incremental_state = dict()
    for seq_len in range(1, out_seq_len + 1):
        output_tokens = torch.tensor([[1] * seq_len] * batch_size)
        start = time.time()
        decoder_out_tensor, decoder_out_dict = model_decoder(
            output_tokens, encoder_out=encoder_out_dict, incremental_state=incremental_state)
        end = time.time()
        print('Decoder step: {}\ttime consumption: {} ms'.format(seq_len, (end - start) * 1000))
        for k in incremental_state:
            print("Shape of one layer in incremental_state: ", incremental_state[k]['prev_key'].shape)
            break