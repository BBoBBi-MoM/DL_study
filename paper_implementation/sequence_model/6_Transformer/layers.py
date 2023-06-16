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