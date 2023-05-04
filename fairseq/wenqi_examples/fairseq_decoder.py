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
args.decoder = { \
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
dec_embs = torch.nn.Embedding(vocab_size, args.decoder["embed_dim"], dictionary.pad())

model = TransformerDecoder(args, dictionary, dec_embs, no_encoder_attn=True)
print(model)

batch_size = 1
seq_len = 512
input_tokens = torch.tensor([[1] * seq_len] * batch_size)

# dec_embs.to(device)
# model.to(device)
# input_tokens.to(device)

# instantiate a new dict for the first time
# out_tensor, out_dict = model(input_tokens)
# incremental_state = dict()
# out_tensor, out_dict = model(input_tokens, incremental_state=incremental_state)

def test_incremental_inference(model, prefix_len=500, final_len=512, batch_size=1, incremental=True):
    """
    Test incremental inference performance. 
    """
    print("Enable incremental inference: ",  incremental)
    with torch.no_grad():

        incremental_state = dict() # initiate
        time_array = []

        for seq_len in range(prefix_len, final_len + 1):
            input_tokens = torch.tensor([[0] * seq_len] * batch_size)
            start = time.time()
            """
            The incremental_state will only add the current state into the dictionary, 
                i.e., its size grows linearly with its step steps. 
            Thus, it incremental inference must start from the beginning, i.e.,
                not with a prefix of 500

            Example:
                step 510: 32.01556205749512 ms
                Shape of one layer in incremental_state:  torch.Size([1, 16, 510, 64])
                step 511: 31.69989585876465 ms
                Shape of one layer in incremental_state:  torch.Size([1, 16, 511, 64])
                step 512: 30.326128005981445 ms
                Shape of one layer in incremental_state:  torch.Size([1, 16, 512, 64])
            """
            if incremental:
                out_tensor, out_dict = model(input_tokens, incremental_state=incremental_state)
            else:
                out_tensor, out_dict = model(input_tokens)
            end = time.time()
            time_array.append(end - start)
            print('step {}: {} ms'.format(seq_len, (end - start) * 1000))
            if incremental:
                for k in incremental_state:
                    print("Shape of one layer in incremental_state: ", incremental_state[k]['prev_key'].shape)
                    break
                  
test_incremental_inference(model, prefix_len=500, final_len=512, batch_size=1, incremental=False)
test_incremental_inference(model, prefix_len=1, final_len=512, batch_size=1, incremental=True)

# print('output', out_tensor, out_dict)
# print('time consumption: {} ms ({} us per step)'.format((end - start) * 1000, (end - start) * 1e6 / seq_len))


"""
Default:
  (dropout_module): FairseqDropout()
  (embed_tokens): Embedding(10, 128, padding_idx=1)
  (project_in_dim): Linear(in_features=128, out_features=512, bias=False)
  (embed_positions): SinusoidalPositionalEmbedding()
  (layers): ModuleList(
    (0-5): 6 x TransformerDecoderLayerBase(
      (dropout_module): FairseqDropout()
      (self_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (activation_dropout_module): FairseqDropout()
      (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (encoder_attn): MultiheadAttention(
        (dropout_module): FairseqDropout()
        (k_proj): Linear(in_features=512, out_features=512, bias=True)
        (v_proj): Linear(in_features=512, out_features=512, bias=True)
        (q_proj): Linear(in_features=512, out_features=512, bias=True)
        (out_proj): Linear(in_features=512, out_features=512, bias=True)
      )
      (encoder_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (fc1): Linear(in_features=512, out_features=2048, bias=True)
      (fc2): Linear(in_features=2048, out_features=512, bias=True)
      (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (output_projection): Linear(in_features=512, out_features=4, bias=False)
)
"""

        # decoder = TransformerDecoder(
        #     RobertaEncDecModel.read_args_from_roberta(roberta_enc.args),
        #     dictionary,
        #     dec_embs,
        #     no_encoder_attn=False,
        #     output_projection=lm_head,
        # )