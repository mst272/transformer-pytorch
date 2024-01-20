# step 8、最终的Encoder部分，包括Embedding 和 encoder_layer,且我们的encoder_layer实现了一层，实际论文中是叠加了32层
import torch.nn as nn
import torch
from MyTransformer_Chinese.Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer_Chinese.Encoder.encoder_layer import EncoderLayer


class Encoder(nn.Module):
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


