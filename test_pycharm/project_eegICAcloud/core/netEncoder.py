import torch.nn as nn
from core.TransformerLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, n_layer, d_model, n_head, d_k, d_v, d_inner, dropout):
        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_k, d_v, d_inner, dropout=dropout)
            for _ in range(n_layer)])

    def forward(self, enc_output, d_outpt, enc_mask, return_attns=False):
        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = enc_output.view(-1, d_outpt)

        return enc_output, enc_slf_attn_list
