from ScaledDotProductAttention import ScaledDotProductAttention
import numpy as np
from torch import nn
import torch

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()
        print(f"before q: {q.size()}, k: {k.size()}, v: {v.size()}")
        q = self.fc_q(q) # 1.为单头变多头做准备，线性变换后维度为n_head的整数倍
        k = self.fc_k(k)
        v = self.fc_v(v)
        print(f"after q: {q.size()}, k: {k.size()}, v: {v.size()}")
        # 在使用permutate,transpose等操作后要求执行contiguous，与底层tensor的存储有关
        # q.view(batch, n_q, n_head, d_q): [5, 2, 3, 5] [batch_size, len_q, n_head, dim]
        # q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3): [3, 5, 2, 5] [n_head, batch_size, len_q, dim]
        # q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q): [15, 2, 5] [n_head*batch_size, len_q, dim]
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
          # 第一个维度重复n_head倍
          mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出
        # output.view(n_head, batch, n_q, d_v): [3, 5, 2, 5] [n_head, batch_size, len_q, dim] 按照q的形状输出
        # output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3): [batch_size, len_q, n_head, dim]
        # output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1): [batch_size, len_q, n_head*dim] 各head的dim合并
        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output

if __name__ == "__main__":
    # MHA
    batch = 5
    n_q, n_k, n_v = 2, 4, 4
    d_q_, d_k_, d_v_ = 10, 10, 10

    q = torch.randn(batch, n_q, d_q_)
    k = torch.randn(batch, n_k, d_k_)
    v = torch.randn(batch, n_v, d_v_)
    mask = torch.zeros(batch, n_q, n_k).bool()

    mha = MultiHeadAttention(n_head=3, d_k_=10, d_v_=10, d_k=5, d_v=5, d_o=10)
    attn, output = mha(q, k, v, mask=mask)

    print(attn.size())
    print(output.size())