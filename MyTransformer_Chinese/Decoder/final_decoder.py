# step 10、构建好decoder layer后开始构建最终的 decoder
import torch.nn as nn
from MyTransformer_Chinese.s1_Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer_Chinese.Decoder.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, vocab_size, dim_vector, max_len, dropout, num_layer, n_head, dim_hidden):
        super().__init__()
        self.embed = TransformerEmbedding(vocab_size, dim_vector, max_len, dropout)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(dim_vector, n_head, dropout, dim_hidden) for _ in range(num_layer)])

        self.linear = nn.Linear(dim_vector, vocab_size)  # 即输出词表数量的神经元，以进行最终的logist预测

    def forward(self, decoder_input, encoder_output, trg_mask, src_mask):
        # 1、 对decoder原始输入部分进行embed
        out = self.embed(decoder_input)

        # 2、开始注意力部分
        for layer in self.decoder_layers:
            out = layer(encoder_output, out, trg_mask, src_mask)

        # 3、 linear输出变换
        out = self.linear(out)
        return out


