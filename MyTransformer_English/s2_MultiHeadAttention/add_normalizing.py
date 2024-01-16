# step 2.2、Add&Norm. In fact,add here is actually residual block.
# This part actually only implements Norm, the residual block needs to be implemented later.
import torch
import torch.nn as nn


# The following is the implementation of the Norm layer, in fact, there is already an implementation in pytorch,
# we just need to call nn. LayerNorm. So I won't be using MyAddNorm later
class MyAddNorm(nn.Module):
    def __init__(self, dim_vector, error=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim_vector))
        self.beta = nn.Parameter(torch.zeros(dim_vector))
        self.error = error

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
