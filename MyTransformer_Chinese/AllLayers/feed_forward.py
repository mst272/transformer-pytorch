# step 6、FeedForward 前馈层，非常简单的一个结构木九十Linear和Relu的组合
# FFN(x) = max(0, xW1 + b1)W2 + b2
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim_vector, dim_hidden, dropout=0.1):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim_vector, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_vector)
        )

    def forward(self, x):
        out = self.feedforward(x)
        return out


