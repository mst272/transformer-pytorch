# from collections import Counter
#
# # 示例1：统计列表中元素的出现次数
# my_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# my_counter = Counter(my_list)
# print(my_counter[2])  # 输出: Counter({4: 4, 3: 3, 2: 2, 1: 1})
#


import torch

# 假设有一个批量大小为2的输入序列
input_sequence = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]])

# 创建填充掩码
padding_mask = (input_sequence != 0).unsqueeze(1).unsqueeze(2)

# 创建未来信息掩码
future_mask = torch.triu(torch.ones_like(input_sequence).unsqueeze(1).expand(-1, input_sequence.size(1), -1), diagonal=1).bool()


# 应用掩码
masked_input = input_sequence.masked_fill(padding_mask == 0, -10000)
# future_masked_input = masked_input.masked_fill(future_mask, -10000)

print("原始输入序列:")
print(input_sequence)

print("\n填充掩码:")
print(padding_mask.size(), padding_mask)

# print("\n未来信息掩码:")
# print(future_mask)

print("\n应用填充掩码后的输入:")
print(masked_input.size(),masked_input)

# print("\n应用未来信息掩码后的输入:")
# print(future_masked_input)
