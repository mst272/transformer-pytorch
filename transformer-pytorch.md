# Transformer code: Step-by-step Understanding



  阅读文章 Solving Transformer by Hand: A Step-by-Step Math Example后对 transformer 有了进一步的认识，遗憾的是文章中并没有给出每一步的具体代码示例，故本文基于上述提到的文章，对每个部分都进行了代码示例及讲解。强烈建议与Solving Transformer by Hand: A Step-by-Step Math Example 一起阅读本文。

  我计划用简洁的语言和详细的代码进行解释，提供一个完整的代码指南（**for both coders and non-coders**）with a step-by-step approach to understanding how they work.。

# Table of Contents



# Step 1 Token Embedding

第一步就是Token Embedding了，即将每一个词用向量表示。代码还是比较简单的，用了torch中现成的Embedding模块。

``` python
# step 1、The first step is Token Embedding, which is word vector encoding.

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Token Embedding represents each word as a vector

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



一般我们的起始输入矩阵size为[batch_size, seq_length]，

