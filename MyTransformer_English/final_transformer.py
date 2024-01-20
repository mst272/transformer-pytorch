# step 5 Final Transformer
# The implementation of src_mask and trg_mask is also covered here.
import torch.nn as nn
import torch
from MyTransformer_English.s3_Encoder.final_encoder import Encoder
from MyTransformer_English.s4_Decoder.final_decoder import Decoder


class Transformer(nn.Module):
    """
        Final Transformer

        Args:
            encoder_voc_size: size of encoder vocabulary,the vocabulary size determines the total number of unique words in our dataset.
            decoder_voc_size: size of decoder vocabulary,the vocabulary size determines the total number of unique words in our dataset.
            dim_vector: the dimension of embedding vector for each input word.
            n_head: Number of heads
            max_len: Maximum length of input sentence
            dim_hidden: The parameter in the feedforward layer
            dropout: probability of an element to be zeroed.
            num_layer: The number of encoders
            src_pad_idx: mask idx
            trg_pad_idx: mask idx
    """

    def __init__(self, src_pad_idx, trg_pad_idx, encoder_voc_size, decoder_voc_size, dim_vector, n_head,
                 max_len, dim_hidden, num_layer, dropout):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(
            encoder_voc_size, dim_vector, max_len, dropout, num_layer, n_head, dim_hidden
        )
        self.decoder = Decoder(
            decoder_voc_size, dim_vector, max_len, dropout, num_layer, n_head, dim_hidden
        )

    def forward(self, src_input, trg_input):
        src_mask = get_src_mask(src_input, self.src_pad_idx)
        trg_mask = get_trg_mask(trg_input, self.trg_pad_idx)

        encoder_out = self.encoder(src_input, src_mask)
        final_out = self.decoder(trg_input, encoder_out, trg_mask, src_mask)
        return final_out


# This function echoes masked_fill in calculate_attention
def get_src_mask(seq, src_pad_idx):
    return (seq != src_pad_idx).unsqueeze(1).unsqueeze(2)


def get_trg_mask(seq, trg_pad_idx):
    batch_size, seq_len = seq.size()
    trg_pad_mask = (seq != trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_sub_mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor)
    trg_mask = trg_sub_mask & trg_pad_mask
    return trg_mask


# final test
if __name__ == '__main__':
    src_input = torch.LongTensor([[1, 1, 4, 0], [4, 3, 2, 9]])
    trg_input = torch.LongTensor([[5, 2, 5, 0], [6, 7, 9, 8]])
    src_pad_idx = 0
    trg_pad_idx = 0
    encoder_voc_size = 10
    decoder_voc_size = 10
    dim_vector = 6
    n_head = 2
    max_len = 10
    dim_hidden = 3
    num_layer = 9
    dropout = 0.1

    model = Transformer(src_pad_idx, trg_pad_idx, encoder_voc_size, decoder_voc_size, dim_vector, n_head, max_len,
                        dim_hidden,
                        num_layer, dropout)
    out = model(src_input, trg_input)
    print(out)
