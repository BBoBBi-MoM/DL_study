import torch
import torch.nn

from layers import *

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, drop_out=0.1):
        super().__init__()

        self.self_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.fc_layer = PositionWiseFeedForward(d_model, d_ff, drop_out)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, encoding_mask=None):
        encoding_attention = self.self_attention_layer(x, mask=encoding_mask)
        encoding_attention = self.layer_norm1(x + encoding_attention)

        encoding_out = self.fc_layer(encoding_attention)
        encoding_out = self.layer_norm2(encoding_attention + encoding_out)
        return encoding_out
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, drop_out=0.1):
        super().__init__()

        self.self_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.cross_attention_layer = MultiHeadAttention(d_model, num_head)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.fc_layer = PositionWiseFeedForward(d_model, d_ff, drop_out)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoding_output, decoding_mask=None, cross_mask=None):
        decoding_attention = self.self_attention_layer(x, mask=decoding_mask)
        decoding_attention = self.layer_norm1(x + decoding_attention)

        cross_attention = self.cross_attention_layer(x=decoding_attention, x2=encoding_output, mask=cross_mask)
        cross_attention = self.layer_norm2(decoding_attention + cross_attention)

        cross_out = self.fc_layer(cross_attention)
        cross_out = self.layer_norm3(cross_attention + cross_out)
        return cross_out