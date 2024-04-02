import torch
import math

emb_size = 11
maxlen = 20
# 0 - emb_size，逐项+2
_a_range = torch.arange(0, emb_size, 2)
print(_a_range)
# print(_a_range.reshape(6, 1))
# print(math.log(math.e))
pos = torch.arange(0, maxlen).reshape(maxlen, 1)
print(pos)
print(pos * _a_range)

# 输入矩阵保留主对角线与主对角线以上的元素，其余元素为0
sz = 10
square = torch.triu(torch.ones((sz, sz)))
print("square: ", square)
square = square.transpose(0, 1)
print("transpose: ", square)
mask = square.float().masked_fill(square == 0, float('-inf')).masked_fill(square == 1, float(0.0))
print(mask)


tensor_test = torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]])
print(torch.max(tensor_test, dim=0))