# Transformer code: Step-by-step Understanding



  阅读文章 Solving Transformer by Hand: A Step-by-Step Math Example后对 transformer 有了进一步的认识，遗憾的是文章中并没有给出每一步的具体代码示例，故本文基于上述提到的文章，对每个部分都进行了代码示例及讲解。强烈建议与Solving Transformer by Hand: A Step-by-Step Math Example 一起阅读本文。

  我计划用简洁的语言和详细的代码进行解释，提供一个完整的代码指南（**for both coders and non-coders**）with a step-by-step approach to understanding how they work.。

  下面是代码中的总结构图，包括每一个步骤及其中包含的字模块

## Table of Contents



## Step 1. Token Embedding

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



## Step 2.  Positional Embedding

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



## Step 3.  Concatenating Positional and Word Embeddings

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

## Step 4. Building Multi Head Attention
  待更新
