import torch 
import torch.nn as nn 
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, d_model, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model//num_heads
        self.num_queries_per_kv = num_heads//num_kv_heads #num_queries_per_kv is the no of query heads shared by each KV head 
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=self.num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(in_features=d_model, out_features=self.num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model)
        assert d_model % num_heads == 0, "embedding dim is not divisible by num heads completely"
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #(B, H, L, D)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) #(B, G, L, D)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        #3 ways to make Q and K compatible for matmul as broadcasting won't work
        #repeat the heads dim num_queries_per_kv times. Not efficient, makes a new tensor in memory, copies values, slower, KV cache increases
        # K = torch.repeat_interleave(input=K, repeats=self.num_queries_per_kv, dim=1) 
        #done using stride tricks, no copying, but kv cache still big
        # K = K.unsqeeze(2) 
        # K = K.expand(-1, -1, self.num_queries_per_kv, -1, -1)
        # K = K.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        #gropued matmul, no kv duplication, no cache increase
        Q = Q.view(batch_size, self.num_kv_heads, self.num_queries_per_kv, seq_len, self.head_dim)
        K = K.unsqueeze(2)
        A = Q @ K.transpose(-2,-1)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        A.masked_fill_(mask == 0, float("-inf"))
        A = torch.softmax(A/math.sqrt(self.head_dim), dim=-1)
        V = V.unsqueeze(2)
        O = (A @ V).view(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(O)

model = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=2)
x = torch.rand(4, 1024, 512)
y = model(x)
print(y.shape)

