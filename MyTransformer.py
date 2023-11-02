import torch
import torch.nn as nn

class MyMultiheadAttention(nn.Module):



class MyTransformerEncoderLayer(nn.Module):
    def __init__(self,d_model,nhead,dropout=0.1):
        super(MyTransformerEncoderLayer,self).__init__()

        self.self_attention =