import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)


class Expert(nn.Module):
    def __init__(self, n_embed, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TopKRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        # x is the output tensor from multi-head self attention block
        logits = self.linear(x)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))


class NoisyTopKRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkrouter_linear = nn.Linear(n_embed, num_experts)
        self.noisy_linear = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        # x is the output tensor from multi-head self attention block
        logits = self.topkrouter_linear(x)

        # Noise logits
        noise_logits = self.noisy_linear(x)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.rand_like(logits) * F.softplus(noise_logits)
        noise_added_logits = logits + noise

        top_k_logits, indices = noise_added_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noise_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMOE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.router = NoisyTopKRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        print(f"gating output:{gating_output}", '\n' f"indices:{indices}")
        final_output = torch.zeros_like(x)
        print(f"final_output:\n {final_output}")

        flat_x = x.view(-1, x.size(-1))
        print(f"flat_x:{flat_x} \n flat_x size:{flat_x.size()}")


if __name__ == '__main__':
    num_experts = 8
    top_k = 2
    n_embd = 6
    dropout = 0.1
    mh_output = torch.randn(3, 4, 6)
    smoe = SparseMOE(n_embd, num_experts, top_k)
    smoe(mh_output)
