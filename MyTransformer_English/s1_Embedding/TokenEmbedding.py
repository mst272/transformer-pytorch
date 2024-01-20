# step 1.1、The first step is Token Embedding, which is word vector encoding.

import torch
import torch.nn as nn


# 1.1 TokenEmbedding
class TokenEmbedding(nn.Module):
    """
        Token Embedding represents each word as a vector

        input x size: [batch_size, seq_length]
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

