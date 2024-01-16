# Build a sub-part of the multi-head attention layer, i.e., an implementation module for attention calculation

import math
import torch
import torch.nn.functional as F


def calculate_attention(q, k, v, mask=None, drop_out=None):
    """
        calculate self attention

        input q, k, v size:[batch_size, n_head, seq_length, dim_split]
        return size: [batch_size, n_head, seq_length, dim_split]

        Args:
            q: query
            k: key
            v: value
            drop_out: probability of an element to be zeroed.
            mask: mask
    """
    # The input is a four-dimensional matrix
    batch_size, n_head, seq_len, dim_split = q.size()

    # 1、calculate
    k_t = k.transpose(2, 3)
    attention = torch.matmul(q, k_t) / math.sqrt(dim_split)

    # 2、whether to mask
    if mask is not None:
        attention = attention.masked_fill(mask == 0, -10000)

    # 3、 Do softmax ,so parameters are in the range 0-1
    attention = F.softmax(attention, dim=-1)
    if drop_out is not None:
        attention = F.dropout(attention, drop_out)

    # 4、Multiply the result by V
    out = torch.matmul(attention, v)
    return out, attention
