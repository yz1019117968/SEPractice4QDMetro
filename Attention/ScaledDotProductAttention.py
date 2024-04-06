import torch
from torch import nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        print(f"q: {q.size()}, k: {k.size()}, v: {v.size()}")
        # q: [5, 4, 128] [batch_size, len_q, dim_q]
        # k: [5, 4, 128] [batch_size, len_k, dim_k]
        # k: [5, 4, 128] [batch_size, len_v, dim_v]
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        # u: [5, 4, 4] [batch_size, len_q, len_k]
        u = u / self.scale # 2.Scale
        print(f"u: {u.size()}\n{u}")
        print(f"mask: {mask}")
        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask
        print(f"masked_u: {u}")
        attn = self.softmax(u) # 4.Softmax
        print(f"attn: {attn.size()}, {attn}")
        # [batch_size, len_q, dim_v] 因为是用v表示q，最终q的维度=v的维度
        output = torch.bmm(attn, v) # 5.Output
        print(f"output: {output.size()}, {output}")
        return attn, output

if __name__ == "__main__":
    # SHA
    batch = 1
    # Q查询每个token在K中的权重，形成注意力矩阵，随后通过V的加权求和得到新的Q，所以K.size()=V.size()
    n_q, n_k, n_v = 4, 6, 6
    d_q, d_k, d_v = 10, 10, 8

    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()
    for i in range(mask.size(0)):
        if np.random.random() > 0.5:
            for j in range(mask.size(1)):
                if np.random.random() > 0.5:
                    mask[i][j][-2:] = True
    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())