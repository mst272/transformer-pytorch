# step 11、开始构建最终的Transformer
# 这里还会涉及到mask相关实现，分别是src_mask 和 trg_mask
import torch.nn as nn
import torch
from MyTransformer_Chinese.Encoder.final_encoder import Encoder
from MyTransformer_Chinese.Decoder.final_decoder import Decoder


class Transformer(nn.Module):
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


# 这个函数与calculate_attention中的masked_fill呼应,
def get_src_mask(seq, src_pad_idx):
    """
        是在encoder阶段的padding mask，为了保证输入的句子长度都一样
        输出的维度是[batch_size,1,1,seq_len]，这个输出的src_mask最开始的使用位置就是calculate_attention中的第二小步

        其中的 attention = attention.masked_fill(mask == 0, -10000):
            attention是q*k之后还没乘 v的矩阵，故维度是[batch_size,n_head,seq_len,seq_len]
            故通过 unsqueeze(1) unsqueeze(2)两个变换让其可广播，可以使用masked_fill

        示例可见test2.py
    Args:
        seq: 输入的tensor
        src_pad_idx: 如果tensor中的数值不等于pad_idx(即不需要mask),则对应掩码值为True，否则为False
    """
    return (seq != src_pad_idx).unsqueeze(1).unsqueeze(2)


def get_trg_mask(seq, trg_pad_idx):
    batch_size, seq_len = seq.size()
    # [batch_size, 1, seq_len, 1]，变成这个维度是为了与sub_mask做  & 操作
    trg_pad_mask = (seq != trg_pad_idx).unsqueeze(1).unsqueeze(3)

    trg_sub_mask = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor)

    trg_mask = trg_sub_mask & trg_pad_mask
    return trg_mask
