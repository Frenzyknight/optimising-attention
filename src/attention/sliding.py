import torch
import torch.nn as nn
import math

class SlidingWindowAttention(nn.Module):
    def __init__(self, num_heads, d_model, window_size, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model//num_heads
        self.num_queries_per_head = num_heads//num_kv_heads
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=self.num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(in_features=d_model, out_features=self.num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x: torch.Tensor):
        batch_len, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_len, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(x).view(batch_len, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
        V = self.W_v(x).view(batch_len, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
        Q = Q.view(batch_len, self.num_kv_heads, self.num_queries_per_head, seq_len, self.head_dim)
        K = K.unsqueeze(2)
        A = Q @ K.transpose(-2,-1)
        mask_lower = torch.tril(torch.ones(seq_len, seq_len, device="cuda"))
        mask = torch.triu(mask_lower, diagonal=-self.window_size+1)
        A.masked_fill_(mask == 0, float("-inf"))
        A = torch.softmax(A/math.sqrt(self.head_dim), dim=-1)
        V = V.unsqueeze(2)
        O = (A @ V).view(batch_len, self.num_heads, seq_len, self.head_dim)
        O = O.transpose(1,2).contiguous().view(batch_len, seq_len, self.d_model)
        return self.W_o(O)

# model = SlidingWindowAttention(d_model=512, num_heads=8, num_kv_heads=2, window_size=4)
# x = torch.randn(4, 1024, 512)
# print(model(x).shape)