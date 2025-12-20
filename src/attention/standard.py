import math
import torch
import torch.nn as nn

class StandardMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model) #only the last dim matters rest all handled via broadcasting
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads #gives int division
        assert d_model % num_heads == 0, "model dimension must be divisible by num_heads" #to make sure dimesionality errors don't occur


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape #extract batch size and sequence length from tensor shape as shape is (B, L, E)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #shape becomes (B, L, H, D) the resahped to (B, H, L, D) to make matmul easier
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        A = Q @ torch.transpose(K, dim0=-1, dim1=-2)  #after (B, H, L, D) @ (B, H, D, L) shape here becomes (B, H, L, L)
        mask = torch.tril(torch.ones(size=(seq_len, seq_len), device="cuda"))
        A.masked_fill_(mask == 0, float("-inf"))
        print(A[0, 0, :5, :5])  # Should see -inf in upper triangle
        A = torch.softmax(A/math.sqrt(self.head_dim), dim=-1)
        O = (A @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(O)
    
# attn = StandardMultiHeadAttention(d_model=512, num_heads=8)
# x = torch.randn(4, 1024, 512)
# out = attn(x)
# print(out.shape)  # prints out (4, 1024, 512)
