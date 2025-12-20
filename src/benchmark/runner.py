import torch
import time

from src.attention.standard import StandardMultiHeadAttention
from src.attention.mqa import MultiQueryAttention
from src.attention.gqa import GroupedQueryAttention
from src.attention.sliding import SlidingWindowAttention

def benchmark_attention( model, batch_size, seq_len, num_runs=100, warmup=10):
    num_heads = 8
    d_model = 512
    attention = model(num_heads, d_model).to("cuda")
    x = torch.randn(batch_size, seq_len, d_model)
    for i in range(warmup):
        print(f"warmup iteration: {i}")
        attention(x)
    for j in range(num_runs):
        print(f"iteration: {j}")
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = attention(x)
        torch.cuda.synchronize()
        end = time.perf_counter()


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16
    benchmark_attention(model=StandardMultiHeadAttention, batch_size=batch_size, seq_len=seq_len)
    