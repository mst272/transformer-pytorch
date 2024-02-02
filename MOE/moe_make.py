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

        # 展平，其实就是相当于把，每个batch拼接
        flat_x = x.view(-1, x.size(-1))
        print(f"flat_x:{flat_x} \n flat_x size:{flat_x.size()}")
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        print(f"flat_gating_output:{flat_gating_output} \n flat_gating_output size:{flat_gating_output.size()}")

        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            print(f"expert_mask:{expert_mask}", '\n', expert_mask.size())
            flat_mask = expert_mask.view(-1)
            print(f"flat_mask:{flat_mask}", '\n', flat_mask.size())

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                print(f"expert_input:{expert_input}", expert_input.size())
                expert_output = expert(expert_input)
                print(f"expert_output:{expert_output}", expert_output.size())

                # Extract and apply gating scores
                gating_score = flat_gating_output[flat_mask, i].unsqueeze(1)
                print(f"gating_score:{gating_score}", gating_score.size())
                weight_output = expert_output * gating_score
                print(f"weight_output:{weight_output}", weight_output.size())

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weight_output.squeeze(1)
                print(f"weight_output:{weight_output}")
                print(f"weight_output.squeeze(-1):{weight_output.squeeze(1)}")
                print(f"final_output:{final_output}", final_output.size())

        return final_output


if __name__ == '__main__':
    num_experts = 8
    top_k = 2
    n_embd = 4
    dropout = 0.1
    mh_output = torch.randn(2, 3, n_embd)
    smoe = SparseMOE(n_embd, num_experts, top_k)
    print(f"smoe(mh_output):{smoe(mh_output)}")
