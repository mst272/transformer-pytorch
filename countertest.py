from collections import Counter

# 示例1：统计列表中元素的出现次数
my_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
my_counter = Counter(my_list)
print(my_counter[2])  # 输出: Counter({4: 4, 3: 3, 2: 2, 1: 1})



