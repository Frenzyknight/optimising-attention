import math
import torch 
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=self.head_dim) #divide by heads becasue we only have 1 key and value head
        self.W_v = nn.Linear(in_features=d_model, out_features=self.head_dim)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)
        
        assert d_model % num_heads == 0, "Bro embedding dim is not completely divisible by num_heads"
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape #shape is (B, L, E)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) #shape becomes (B,L,H,D) from (B, H, L, D)
        K = self.W_k(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1,2) #shape becomes (B, 1, L, D)
        V = self.W_v(x).view(batch_size, seq_len, 1, self.head_dim).transpose(1,2)

        A = Q @ K.transpose(-2, -1) #(B, H, L, D) @ (B, 1, D, L) becomes (B, 1, L, L)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        A.masked_fill_(mask == 0, float("-inf"))
        A = torch.softmax(A/math.sqrt(self.head_dim), dim=-1)
        O = (A @ V).transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(O)

model = MultiQueryAttention(d_model=512, num_heads=8)
x = torch.rand(4, 1024, 512)
y = model(x)
print(y.shape)