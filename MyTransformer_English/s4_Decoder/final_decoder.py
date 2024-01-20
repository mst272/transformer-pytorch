# step 4.2 The final Decoder part, including Embedding and decoder_layer
import torch.nn as nn
from MyTransformer_English.s1_Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer_English.s4_Decoder.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
        Final Decoder Layer

        input size:[batch_size, seq_length, dim_vector]
        return size: [batch_size, seq_length, vocab_size]

        Args:
            vocab_size: size of vocabulary,the vocabulary size determines the total number of unique words in our dataset.
            dim_vector: the dimension of embedding vector for each input word.
            n_head: Number of heads
            max_len: Maximum length of input sentence
            dim_hidden: The parameter in the feedforward layer
            dropout: probability of an element to be zeroed.
            num_layer: The number of encoders
    """
    def __init__(self, vocab_size, dim_vector, max_len, dropout, num_layer, n_head, dim_hidden):
        super().__init__()
        self.embed = TransformerEmbedding(vocab_size, dim_vector, max_len, dropout)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dim_vector, n_head, dropout, dim_hidden) for _ in range(num_layer)])
        self.linear = nn.Linear(dim_vector, vocab_size)

    def forward(self, decoder_input, encoder_output, trg_mask, src_mask):
        # 1、 embedding
        out = self.embed(decoder_input)

        # 2、attention
        for layer in self.decoder_layers:
            out = layer(encoder_output, out, trg_mask, src_mask)

        # 3、 linear
        out = self.linear(out)
        return out


