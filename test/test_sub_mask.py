import torch

# 假设 trg 是一个形状为 (batch_size, trg_len) 的 tensor
trg = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])

# 假设 trg_pad_idx 为用于填充的索引
trg_pad_idx = 0

# 生成 trg_pad_mask
trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)

print("Input trg tensor:")
print(trg.size(), '\n', trg)
print("\nTrg Pad Mask:")
print(trg_pad_mask.size(), '\n', trg_pad_mask)

# ---------------2-----------------------
print('-----------------------------------2-------------------------------')

# 假设 trg_len 为 trg 的长度
trg_len = 4

# 生成 trg_sub_mask
trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)

print("Trg Sub Mask:")
print(trg_sub_mask.size(), '\n', trg_sub_mask)

# ---------------3-----------------------
print('-----------------------------------3-------------------------------')

# 计算 trg_mask
trg_mask = trg_pad_mask & trg_sub_mask

print("Trg Mask:")
print(trg_mask.size(), '\n', trg_mask)

# ---------------4-----------------------
print('-----------------------------------4---------------------------------')
# 假设 attention 是你的注意力分数矩阵
attention = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                           [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]]],
                          [[[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]],
                           [[49, 50, 51, 52], [53, 54, 55, 56], [57, 58, 59, 60], [61, 62, 63, 64]]]])

# 使用 masked_fill 将不满足条件的位置替换为 -10000
result = attention.masked_fill(trg_mask == 0, -10000)

print("Attention Matrix:")
print(attention.size(), '\n', attention)
print("\nResult After Masking:")
print(result.size(), '\n', result)
