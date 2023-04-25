import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotAttention(nn.Module):
    def __init__(self, d_k=64):
        super().__init__()

        self.scailing_factor = d_k ** 0.5

    def forward(self, query, key, value, mask=None):
        attention_score = torch.matmul(query,key.transpose(-2,-1))
        attention_score /= self.scailing_factor
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_proba = F.softmax(attention_score, dim=-1)
        attention = torch.matmul(attention_proba, value)
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8):
        super().__init__()
        d_k = int(d_model / num_head)

        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_k

        self.query_fc_layer = nn.Linear(d_model, d_k*num_head)
        self.key_fc_layer = nn.Linear(d_model, d_k*num_head)
        self.value_fc_layer = nn.Linear(d_model, d_k*num_head)

        self.attention_layers = ScaledDotAttention(d_k)
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x, x2=None, mask=None):
        batch_size, seq_len, dimension = x.shape

        if x2 is None:
            query = x   
            key = x
            value = x
        else:
            query = x
            key = x2
            value = x2

        query = self.query_fc_layer(query)
        key = self.key_fc_layer(key)
        value = self.value_fc_layer(value)

        query = self._split_dimension(query)
        key = self._split_dimension(key)
        value = self._split_dimension(value)

        attention = self.attention_layers(query, key, value, mask)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, dimension)
        out = self.output_layer(attention)
        return out

    def _split_dimension(self, x):
        batch_size, seq_len, dimension = x.shape
        x = x.view(batch_size, seq_len, self.num_head, dimension//self.num_head)
        x = x.transpose(1,2)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_out=0.1):
        super().__init__()

        self.fc_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=drop_out),
            nn.Linear(d_ff,d_model),
            nn.Dropout(p=drop_out)
        )        

    def forward(self, x):
        return self.fc_layer(x)
    
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
    
class Encoder(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1):
        super().__init__()

        self.layers = [EncoderBlock(d_model, num_head, d_ff, drop_out) for _ in range(num_repeats)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, encoding_mask=None):
        for layer in self.layers:
            x = layer(x, encoding_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1):
        super().__init__()

        self.layers = [DecoderBlock(d_model, num_head, d_ff, drop_out) for _ in range(num_repeats)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, encoding_out, decoding_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, encoding_out, decoding_mask, cross_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1):
        super().__init__()

        self.token_embedding = nn.Linear(1,d_model)
        self.positional_encoding = nn.Linear(d_model, d_model)

        self.encoder = Encoder(d_model, num_head, d_ff, num_repeats, drop_out)
        self.decoder = Decoder(d_model, num_head, d_ff, num_repeats, drop_out)

    def forward(self, encoding_x, decoding_x):
        encoding_mask = self._make_pad_mask(encoding_x, encoding_x)
        cross_mask = self._make_pad_mask(decoding_x, encoding_x)
        decoding_mask = self._make_pad_mask(decoding_x, decoding_x) & self._make_subsequent_mask(decoding_x, decoding_x)

        encoding_x = self.token_embedding(encoding_x.unsqueeze(-1))  # (batch, seq_len, d_model)
        decoding_x = self.token_embedding(decoding_x.unsqueeze(-1))  # (batch, seq_len, d_model)

        encoding_x = self.positional_encoding(encoding_x)  # (batch, seq_len, d_model)
        decoding_x = self.positional_encoding(decoding_x)  # (batch, seq_len, d_model)

        enc_output = self.encoder(encoding_x, encoding_mask)
        dec_output = self.decoder(decoding_x, enc_output, decoding_mask, cross_mask)
        return dec_output


    def _make_pad_mask(self, query, key, query_pad_idx=1, key_pad_idx=1):
        batch, len_q = query.shape
        batch, len_k = key.shape

        query_mask = (query != query_pad_idx).view(batch, 1, len_q, 1)
        query_mask = query_mask.repeat(1, 1, 1, len_k)

        key_mask = (key != key_pad_idx).view(batch, 1, 1, len_k)
        key_mask = key_mask.repeat(1, 1, len_q, 1) 

        pad_mask = query_mask & key_mask
        return pad_mask

    def _make_subsequent_mask(self, query, key):
        len_q = query.shape[1]
        len_k = key.shape[1]

        matrix = torch.ones(len_q, len_k)
        subsequent_mask = torch.tril(matrix).bool()  # lower triangular matrix
        return subsequent_mask

if __name__ == '__main__':
    # TODO 1: token embedding
    # TODO 2: positional encoding

    model = Transformer(d_model=512, num_head=8, d_ff=2048, num_repeats=6, drop_out=0.1)

    x = torch.randn(1, 64)  # (batch, seq_len, dimension)
    y = model(x, x)
    print(y.shape)
