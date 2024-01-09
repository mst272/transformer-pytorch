# step 11、开始构建最终的Transformer
# 这里还会涉及到mask相关实现，分别是src_mask 和 trg_mask
import torch.nn as nn
from MyTransformer.Encoder.final_encoder import Encoder
from MyTransformer.Decoder.final_decoder import Decoder


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


# 这个函数与calculate_attention中的masked_fill呼应,
def get_src_mask(seq, pad_idx):
    """
        是在encoder阶段的padding mask，为了保证输入的句子长度都一样

    Args:
        seq: 输入的tensor
        pad_idx: 如果tensor中的数值不等于pad_idx(即不需要mask),则对应掩码值为True，否则为False
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def get_trg_mask(seq):
    batch_size, seq_length = seq.size()

