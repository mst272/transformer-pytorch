# step 1.3、  Concatenating Positional and Word Embeddings

import torch
import torch.nn as nn
from MyTransformer_English.s1_Embedding.PositionalEmbedding import PositionalEmbedding
from MyTransformer_English.s1_Embedding.TokenEmbedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
        Concatenating Positional and Word Embeddings

        input x size: [batch_size, seq_length]
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

