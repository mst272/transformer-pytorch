# step 7、前面已经把各个模块写好了，现在就需要组装起来，组成encoder layer
import torch.nn as nn
from MyTransformer.AllLayers.muti_head_attention import MultiHeadAttention
from MyTransformer.AllLayers.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, dim_vector, n_head, dim_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(dim_vector, n_head)
        self.norm1 = nn.LayerNorm(dim_vector)
        self.dropout1 = nn.Dropout(dropout)

        self.feedforward = FeedForward(dim_vector, dim_hidden, dropout)
        self.norm2 = nn.LayerNorm(dim_vector)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 1、 注意力计算
        out = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2、 Add 和 Norm
        out = self.dropout1(out)
        out = self.norm1(x + out)

        # 3、 FeedForward
        _x = out       # add要加的残差，即原始输入
        out = self.feedforward(out)

        # 4、再一层Add 和 Norm
        out = self.dropout2(out)
        out = self.norm2(out + _x)
        return out


