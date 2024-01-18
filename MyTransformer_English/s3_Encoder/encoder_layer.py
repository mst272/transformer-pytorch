# step 3.1、The modules have been written before, and now they need to be assembled to form the encoder layer.
import torch.nn as nn
from MyTransformer_English.s2_MultiHeadAttention.muti_head_attention import MultiHeadAttention
from MyTransformer_English.s2_MultiHeadAttention.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """
        Encoder Layer

        input size:[batch_size, seq_length, dim_vector]
        return size: [batch_size, seq_length, dim_vector]

        Args:
            dim_vector: the dimension of embedding vector for each input word.
            n_head: Number of heads
            dim_hidden: The parameter in the feedforward layer
            dropout: probability of an element to be zeroed.
    """
    def __init__(self, dim_vector, n_head, dim_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(dim_vector, n_head)
        self.norm1 = nn.LayerNorm(dim_vector)
        self.dropout1 = nn.Dropout(dropout)

        self.feedforward = FeedForward(dim_vector, dim_hidden, dropout)
        self.norm2 = nn.LayerNorm(dim_vector)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 1、 calculate multi-head-attention
        out = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2、 Add and Norm
        out = self.dropout1(out)
        out = self.norm1(x + out)

        # 3、 FeedForward
        _x = out
        out = self.feedforward(out)

        # 4、Add and Norm
        out = self.dropout2(out)
        out = self.norm2(out + _x)
        return out


