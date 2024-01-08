# step 8、最终的Encoder部分，包括Embedding 和 encoder_layer,且我们的encoder_layer实现了一层，实际论文中是叠加了32层
import torch.nn as nn
from MyTransformer.Embedding.TransformerEmbedding import TransformerEmbedding
from MyTransformer.Encoder.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = TransformerEmbedding()

    def forward(self, x, src_mask):
