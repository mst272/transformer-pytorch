# step 2.1、 Build multi-head attention layer

import torch.nn as nn
from MyTransformer_English.s1_Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer_English.s2_MultiHeadAttention.calculate_attention import calculate_attention
import torch


# The input now accepted is token embedding + position embedding.
# Size: [batch_size, seq_len, dim_vector]

class MultiHeadAttention(nn.Module):
    # n_head即多头，其原理是将dim_vector分割为n_head个。
    """
        Build multi-head attention layer

        input q, k, v size:[batch_size, seq_length, dim_vector]
        return size: [batch_size, seq_length, dim_vector]

        Args:
            dim_vector: the dimension of embedding vector for each input word.
            n_head: Number of heads
    """
    def __init__(self, dim_vector, n_head):
        super().__init__()
        self.n_head = n_head

        # The linear layer in torch is a linear transformation of the input, which will have a Weights parameter
        # matrix, which is learnable. It is also because of this matrix that the Linear layer can change the input
        # in_features into out_features.
        self.w_q = nn.Linear(dim_vector, dim_vector)   # w_q size: [dim_vector, dim_vector]
        self.w_k = nn.Linear(dim_vector, dim_vector)
        self.w_v = nn.Linear(dim_vector, dim_vector)
        self.linear = nn.Linear(dim_vector, dim_vector)

    def forward(self, q, k, v, mask=None):
        # 1、Input q k v(token embedding + position embedding),and then get the true q k v through the linear layer
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2、dim split
        q, k, v = self.dim_head_split(q), self.dim_head_split(k), self.dim_head_split(v)

        # 3、calculate attention
        out, attention = calculate_attention(q, k, v, mask)  # out size: [batch_size, n_head, seq_len, dim_split]

        # 4、Concat the results
        batch_size, n_head, seq_len, dim_split = out.size()
        dim_vector = n_head * dim_split

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim_vector) # out size: [batch_size, seq_len, dim_vector]

        # 5、Finally, multiply by a linear layer
        out = self.linear(out)

        # 6、 Attention matrix can be used to visualize attention if needed
        # attention.....

        return out

    def dim_head_split(self, tensor):
        """
        Divide q k v into specified n_head heads in dim_vector dimension

        tensor size: [batch_size, seq_len, dim_vector]
        return size:  [batch_size, n_head, seq_len, dim_split]
        """
        batch_size, seq_len, dim_vector = tensor.size()
        dim_split = dim_vector // self.n_head

        # After splitting, our goal is to get the dimensions in the above return. If we use view directly,
        # although the dimensions are the same, the numbers in the operation are changed.
        tensor = tensor.view(batch_size, seq_len, self.n_head, dim_split)
        # After dimensional segmentation, transpose to match our data composition format
        tensor = tensor.transpose(1, 2)

        return tensor

