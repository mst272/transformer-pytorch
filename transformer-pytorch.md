# Transformer code: Step-by-step Understanding



  阅读文章 Solving Transformer by Hand: A Step-by-Step Math Example后对 transformer 有了进一步的认识，遗憾的是文章中并没有给出每一步的具体代码示例，故本文基于上述提到的文章，对每个部分都进行了代码示例及讲解。强烈建议与Solving Transformer by Hand: A Step-by-Step Math Example 一起阅读本文。

  我计划用简洁的语言和详细的代码进行解释，提供一个完整的代码指南（**for both coders and non-coders**）with a step-by-step approach to understanding how they work.。

  下面是代码中的总结构图，包括每一个步骤及其中包含的字模块

## Table of Contents



## Step 1. Embedding



### Step 1.1 Token Embedding

第一步就是Token Embedding了，即将每一个词用向量表示。代码还是比较简单的，用了torch中现成的Embedding模块。

``` python
# step 1、The first step is Token Embedding, which is word vector encoding.

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token Embedding represents each word as a vector
    
    input size: [batch_size, seq_length]
    return size: [batch_size, seq_length, dim_vector]

    Args:
        vocab_size: size of vocabulary,the vocabulary size determines the total number of unique words in our dataset.
        dim_vector: the dimension of embedding vector for each input word.
    """

    def __init__(self, vocab_size, dim_vector):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim_vector)

    def forward(self, x):
        return self.token_embed(x)
```



  一般我们的起始输入矩阵size为[batch_size, seq_length]，token embedding 后每个单词以向量表示，输出size就变为了[batch_size, seq_length, dim_vector]



### Step 1.2  Positional Embedding

现在我们进行第二步，对token embedding后的输入进行位置编码。

用论文中的两个公式进行计算，这里的输入与上一步的输入相同，size均为[batch_size, seq_length]，其中的PE矩阵代表着全部的位置信息。最后的输出size为[seq_length, dim_vector]详细代码如下;

```python
# step 2、 Positional embeddings for our input
import torch
from torch import nn
import math


class PositionalEmbedding(nn.Module):
    """
        Use the two formulas in the paper to calculate PositionalEmbedding

        input size: [batch_size, seq_length]
        return size: [batch_size, seq_length, dim_vector]

        Args:
            max_len: Maximum length of input sentence
            dim_vector: the dimension of embedding vector for each input word.
    """

    def __init__(self, dim_vector, max_len):
        super().__init__()

        self.dim_vector = dim_vector
        self.max_len = max_len

        pe = torch.zeros(max_len, dim_vector)
        for pos in range(max_len):
            for i in range(0, dim_vector, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim_vector)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / dim_vector)))

        # The size of the pe matrix is [max_len, dim_vector].
        print(f"pe size：{pe.size()}")

        # Register buffer, indicating that this parameter is not updated. Tips: Registering the buffer is equivalent
        # to defining self.pe in__init__, so self.pe can be called in the forward function below, but the parameters
        # will not be updated.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # The input x to the position code is [batch_size, seq_len], where seq_len is the length of the sentence
        batch_size, seq_len = x.size()
        # Returns location information for the number of previous seq_len
        return self.pe[:seq_len, :]
```



### Step 1.3  Concatenating Positional and Word Embeddings

第三步很简单，就是add word embeddings and positional embeddings，并且在其中加入dropout层，避免过拟合。

```python
# step 3、  Concatenating Positional and Word Embeddings

import torch
import torch.nn as nn
from MyTransformer_English.s1_Embedding.PositionalEmbedding import PositionalEmbedding
from MyTransformer_English.s1_Embedding.TokenEmbedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
            Concatenating Positional and Word Embeddings

            input size: [batch_size, seq_length]
            return size: [batch_size, seq_length, dim_vector]

            Args:
                max_len: Maximum length of input sentence
                dim_vector: the dimension of embedding vector for each input word.
                vocab_size: size of vocabulary,the vocabulary size determines the total number of unique words in our dataset.
                drop_out: probability of an element to be zeroed.
    """
    def __init__(self, vocab_size, dim_vector, max_len, drop_out):
        super().__init__()
        self.token_embd = TokenEmbedding(vocab_size, dim_vector)
        self.position_embd = PositionalEmbedding(dim_vector, max_len)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # input
        batch_size, seq_len = x.size()
        token_embd = self.token_embd(x)  # output size: [batch_size, seq_len, dim_vector]
        position_embd = self.position_embd(x)  # output size: [seq_len, dim_vector]
        return self.dropout(token_embd + position_embd)
```

## Step 2. MultiHead Attention

### Step 2.1 Building Multi Head Attention

  我们进行构建Multi Head Attention，输入是从上一步相加的token embedding and position embedding 矩阵，q， k， v均是此矩阵，size为[batch_size, seq_length, dim_vector]。

 这一步需要传入dim_vector维度以及head数量，因为多头注意力的代码实现是将dim_vector分成head个dim，这样就相当于多头注意力，每个头的维度都是dim_split。For example,我们的维度是512，分成了8个头，那么每个dim_split就是512/8=64。

具体代码如下：

```python
# step 4、 Build multi-head attention layer

import torch.nn as nn
from MyTransformer_English.s1_Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer.AllLayers.calculate_attention import calculate_attention
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

```



其中 calculate attention 实现如下，作为我们的 Step 2.1.1：

#### Step 2.1.1 calculate self-attention

 calculate attention , 代码如下所示：

```python
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
```





### Step 2.2 Add&Norm

 这一层中的Add就是residual block，所以需要在后面的模型中实现，本次的代码只实现了Norm。

Norm层pytorch已经有了实现，我们只需要调用nn.LayerNorm即可，但在这一层中我们还是自己写一下以加深印象，后续模型中的Norm不会使用本层的MyAddNorm，而是直接使用nn.LayerNorm。

```python
# step 2.2、Add&Norm. In fact,add here is actually residual block.
# This part actually only implements Norm, the residual block needs to be implemented later.
import torch
import torch.nn as nn


# The following is the implementation of the Norm layer, in fact, there is already an implementation in pytorch,
# we just need to call nn. LayerNorm. So I won't be using MyAddNorm later
class MyAddNorm(nn.Module):
    def __init__(self, dim_vector, error=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim_vector))
        self.beta = nn.Parameter(torch.zeros(dim_vector))
        self.error = error

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
```



### Step 2.3 FeedForward

A very simple structure, a combination of Linear and Relu。代码如下：

```python
# step 2.3 FeedForward，A very simple structure, a combination of Linear and Relu
# FFN(x) = max(0, xW1 + b1)W2 + b2
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim_vector, dim_hidden, dropout=0.1):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim_vector, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_vector)
        )

    def forward(self, x):
        out = self.feedforward(x)
        return out
```



## Step 3. Encoder

待更新。。。。













