import torch 
import time

from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.data import Dictionary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {}
dictionary = Dictionary()
vocab_size = 10
embedding_dim = 128
dec_embs = torch.nn.Embedding(vocab_size, embedding_dim, dictionary.pad())
# dec_embs.to(device)
model = TransformerDecoder(args, dictionary, dec_embs)
print(model)

batch_size = 1
seq_len = 512
input_tokens = torch.tensor([[0] * seq_len] * batch_size)

# model.to(device)
# input_tokens.to(device)

model(input_tokens)

with torch.no_grad():
    start = time.time()
    output = model(input_tokens)
    end = time.time()
  
print('output', output)
print('time consumption: {} ms ({} us per step)'.format((end - start) * 1000, (end - start) * 1e6 / seq_len))


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