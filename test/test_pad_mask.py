import torch

# 假设 attention 和 mask 是你的张量
attention = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                           [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]],
                          [[[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]],
                           [[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60], [61, 62, 63, 64]]]])

mask = torch.tensor([[[[1, 0, 1, 0]]],
                    [[[0, 1, 0, 1]]]])
print(mask.size())

# 使用 masked_fill 进行填充操作
result = attention.masked_fill(mask == 0, -10000)

print(result)