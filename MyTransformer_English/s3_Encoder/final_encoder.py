# step 3.2 The final Encoder part, including Embedding and encoder_layer
import torch.nn as nn
import torch
from MyTransformer_English.s1_Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer_English.s3_Encoder.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
        Final Encoder Layer

        input size:[batch_size, seq_length, dim_vector]
        return size: [batch_size, seq_length, dim_vector]

        Args:
            vocab_size: size of vocabulary,the vocabulary size determines the total number of unique words in our dataset.
            dim_vector: the dimension of embedding vector for each input word.
            n_head: Number of heads
            max_len: Maximum length of input sentence
            dim_hidden: The parameter in the feedforward layer
            drop_out: probability of an element to be zeroed.
            num_layer: The number of encoders
    """
    def __init__(self, vocab_size, dim_vector, max_len, drop_out, num_layer, n_head, dim_hidden):
        super().__init__()
        self.embed = TransformerEmbedding(vocab_size, dim_vector, max_len, drop_out)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(dim_vector, n_head, dim_hidden, drop_out) for _ in range(num_layer)])

    def forward(self, x, src_mask):
        x = self.embed(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x


# --------------------------------测试-----------------------------------------
if __name__ == '__main__':
    # 1、Encoder部分测试
    input_word = torch.LongTensor([[1, 0, 4, 5], [4, 3, 2, 9]])
    print(input_word.size())
    encoder = Encoder(10, 6, 10, 0.1, 3, 2, 4)
    out = encoder(input_word, None)
    print(out.size())
    print(out)


