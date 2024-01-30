# 注意：为了容易理解，我对代码进行了简化，同时不考虑batch size，实际使用时还是要用官方代码
import torch
import torch.nn as nn
import torch.functional as F


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(self.hidden_dim, 8)
        self.experts = nn.ModuleList([MLP(config) for _ in range(8)])

    def forward(self, x):
        # 对每个token计算8个expert的权重，并将权重归一化
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1)
        # 每个token选择top-2 experts的权重、索引， 并将索引转为size=(len(tokens), 8)的独热编码
        routing_weights, selected_experts = torch.top2(routing_weights, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=8)
        # 重新将top-2 expert的权重归一化（因为删掉其余6个expert，权重的和不等于1了）
        routing_weights /= routing_weights.sum(dim=-1)
        # 创建形状和x一致，初始值为0的矩阵，用来存储每个expert的输出
        final_hidden_states = torch.zeros_like(x)

        for expert_idx in range(8):
            # 选择当前使用的expert
            expert_layer = self.experts[expert_idx]
            # 选择当前expert对应的index
            idx_list, top_x_list = torch.where(expert_mask[expert_idx])
            # 选择需要计算的状态
            current_state = x[top_x_list]
            # 选择当前expert对每个token的权重
            current_routing_weights = routing_weights.t()[top_x_list, idx_list]
            # 将选择的状态输入给专家模型计算，并乘上权重
            current_hidden_states = expert_layer(current_state) * current_routing_weights
            # 将每个expert的输出按照索引加到最终结果里
            final_hidden_states.index_add_(0, top_x_list, current_hidden_states)
        return final_hidden_states
