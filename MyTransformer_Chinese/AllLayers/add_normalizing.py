# step 5、Add&Norm，其实这里的Add就是进行残差啦,故在后面模型结构时实现，Norm是对其进行标准化。
# 也就是说本部分实际上只实现了Norm，Add也就是残差需要在后面实现(实现比较简单，相加即可)。
import torch
import torch.nn as nn


# 下面是Norm层的实现，其实pytorch中已经有了实现，我们只需要调用nn.LayerNorm即可，不用自己写了。
# 这个就没必要测试了，在后续的架构中直接使用nn.LayerNorm即可。
# 写了MyAddNorm主要是帮助实现原理。
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


