import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

is_print_shape = True


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
        att_mask=None):
    # Step 1：计算得到Q、K、V
    q = q_proj_weight(query)
    k = k_proj_weight(key)
    v = v_proj_weight(value)

    if is_print_shape:
        print("" + "=" * 80)
        print("进入多头注意力计算:")
        print(
            f"\t 多头num_heads = {n_head}, d_model={query.size(-1)}, d_k = d_v = d_model/num_heads={query.size(-1) // n_head}")
        print(f"\t query的shape([tgt_len, batch_size, embed_dim]):{query.shape}")
        print(f"\t  W_q 的shape([embed_dim,kdim * num_heads]):{q_proj_weight.weight.shape}")
        print(f"\t   Q  的shape([tgt_len, batch_size,kdim * num_heads]):{q.shape}")
        print("\t" + "-" * 70)

        print(f"\t  key 的shape([src_len,batch_size, embed_dim]):{key.shape}")
        print(f"\t  W_k 的shape([embed_dim,kdim * num_heads]):{k_proj_weight.weight.shape}")
        print(f"\t   K  的shape([src_len,batch_size,kdim * num_heads]):{k.shape}")
        print("\t" + "-" * 70)

        print(f"\t value的shape([src_len,batch_size, embed_dim]):{value.shape}")
        print(f"\t  W_v 的shape([embed_dim,vdim * num_heads]):{v_proj_weight.weight.shape}")
        print(f"\t   V  的shape([src_len,batch_size,vdim * num_heads]):{v.shape}")
        print("\t" + "-" * 70)
        print(
            "\t ***** 注意，这里的W_q, W_k, W_v是多个head同时进行计算的. 因此，Q,K,V分别也是包含了多个head的q,k,v堆叠起来的结果 *****")

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
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        # 扩展维度，从[batch_size,src_len]变成[batch_size,1,1,src_len]
        attn_output_weights = attn_output_weights.reshape(batch_size * n_head, tar_len, src_len)

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

    if is_print_shape:
        print(f"\t 多头注意力中,多头计算结束后的形状（堆叠）为([tgt_len,batch_size,num_heads*kdim]){attn_output.shape}")
        print(
            f"\t 多头计算结束后，再进行线性变换时的权重W_o的形状为([num_heads*vdim, num_heads*vdim  ]){out_proj_weight.shape}")
        print(f"\t 多头线性变化后的形状为([tgt_len,batch_size,embed_dim]) {Z.shape}")

    return Z, attn_output_weights.sum(dim=1) / n_head  # 将num_heads个注意力权重矩阵按对应维度取平均


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
        self.q_proj_weight = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj_weight = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj_weight = nn.Linear(d_model, d_model, bias=bias)
        # 初始化一个线性层来对这一结果进行一个线性变换。
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    # 开始定义前向传播
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        return multi_head_attention_forward(
            query, key, value, self.nhead, self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            att_mask=attn_mask
        )


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        '''
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1  
        '''

        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈层实现
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param src_mask:  None
        :param src_key_padding_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """

        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差链接
        src = self.norm1(src) # [src_len,batch_size,num_heads*kdim]

        src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()

        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forwar(self, src, mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:# [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        output =src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask) # 多个encoder layers层堆叠后的前向传播过程

        if self.norm is not  None:
            output = self.norm(output)

        return output  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]

class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()

        self.self_attn = MyMultiheadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.multi_attn = MyMultiheadAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask, ):
        """
        :param tgt:  解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param memory: 编码部分的输出（memory）, [src_len,batch_size,embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        pass



if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    dmodel = 32
    num_head = 1
    src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [src_len, src_len]
    my_mh = MyMultiheadAttention(dmodel, num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
