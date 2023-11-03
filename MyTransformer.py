import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_head_attention_forward(
        query,  # [tgt_len,batch_size, d_model]
        key,
        value,
        n_head,
        dropout,
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=None,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        att_mask=None
):
    # Step 1：计算得到Q、K、V
    q = F.linear(query, q_proj_weight)
    k = F.linear(key, k_proj_weight)
    v = F.linear(value, v_proj_weight)

    # Step2：缩放以及attn_mask维度的判断
    tar_len, batch_size, d_model = query.size()
    src_len = key.size(0)
    head_dim = d_model // n_head
    scaling = float(head_dim) ** -0.5
    q = q * scaling

    if att_mask is not None:
        if att_mask.dim() == 2:
            att_mask = att_mask.unsqueeze(0)  # 扩充维度
            if list(att_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif att_mask.dim() == 3:
            if list(att_mask.size()) != [batch_size * n_head, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
    # 现在attn_mask的维度变成了3D

    # Step 3：计算得到注意力权重矩阵
    q = q.contiguous().reshape(tar_len, batch_size * n_head, head_dim).transpose(0, 1)
    k = k.contiguous().reshape(-1, batch_size * n_head, head_dim).transpose(0, 1)
    v = v.contiguous().reshape(-1, batch_size * n_head, head_dim).transpose(0, 1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))

    # Step 4：进行相关mask操作
    if att_mask is not None:
        attn_output_weights += att_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.reshape(batch_size, n_head, tar_len, src_len)
        attn_output_weights = attn_output_weights.masked_file(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        # 扩展维度，从[batch_size,src_len]变成[batch_size,1,1,src_len]
        attn_output_weights = attn_output_weights.reshape(batch_size*n_head, tar_len, src_len)

    # Step 5：进行归一化操作
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    """
    Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    =  [batch_size * num_heads,tgt_len,vdim]
    这就num_heads个Attention(Q,K,V)结果
    """

    Z = F.linear(attn_output, out_proj_weight, out_proj_bias)


class MyMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, bias=True):
        super(MyMultiheadAttention, self).__init__()
        """
        :param d_model:   词嵌入的维度，论文中的默认值为512
        :param nhead:   多头注意力机制中多头的数量，论文默认值为 8
        :param bias:        最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        self.d_model = d_model
        self.single_head_dim = d_model // nhead  # 即单个头的dim维度
        self.kdim = self.single_head_dim
        self.vdim = self.single_head_dim
        self.nhead = nhead
        self.dropout = dropout
        assert self.single_head_dim * self.nhead == self.d_model, "d_model/nhead 必须为整数"
        # 下面q，k，v第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.q_proj_weight = nn.Parameter(torch.tensor(d_model, d_model))
        self.k_proj_weight = nn.Parameter(torch.tensor(d_model, d_model))
        self.v_proj_weight = nn.Parameter(torch.tensor(d_model, d_model))
        # 初始化一个线性层来对这一结果进行一个线性变换。
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    # 开始定义前向传播
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        pass


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()

        self.self_attention = MyMultiheadAttention(d_model, nhead, dropout=dropout)
