import torch
import time
import os
import json
from src.attention.standard import StandardMultiHeadAttention
from src.attention.mqa import MultiQueryAttention
from src.attention.gqa import GroupedQueryAttention
from src.attention.sliding import SlidingWindowAttention

def benchmark_attention( model_class, model_kwargs,  batch_size, seq_len, num_runs=100, warmup=10):
    d_model = 512
    attention = model_class(**model_kwargs).to("cuda")
    total_time = 0
    peak_mem = 0
    x = torch.randn(batch_size, seq_len, d_model).to("cuda")
    for i in range(warmup):
        print(f"warmup iteration: {i}")
        attention(x)
    for j in range(num_runs):
        if j % 10 == 0:
            print(f"iteration: {j}")
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = attention(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_time += (end - start)
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > peak_mem else peak_mem
        torch.cuda.reset_peak_memory_stats()
    
    avg_time = (total_time/num_runs)
    return {
        "total_time": total_time,
        "avg_latency": avg_time,
        "throughput" : (batch_size * seq_len) / avg_time,
        "memory_mb" : (peak_mem/1024**2)
    }

if __name__ == "__main__":
    batch_size = 2
    seq_len = [128, 512, 1024, 2048]
    models = {
        "Standard": {
            "class": StandardMultiHeadAttention,
            "kwargs": {"d_model": 512, "num_heads": 8}
        },
        "MQA": {
            "class": MultiQueryAttention,
            "kwargs": {"d_model": 512, "num_heads": 8}
        },
        "GQA": {
            "class": GroupedQueryAttention,
            "kwargs": {"num_heads": 8, "d_model": 512, "num_kv_heads": 2}
        },
        "Sliding": {
            "class": SlidingWindowAttention,
            "kwargs": {"num_heads": 8, "d_model": 512, "window_size": 128, "num_kv_heads": 2}
        }
    }
    results = {}
    for name, config in models.items():
        results[name] = {}
        for seq in seq_len:
            print(f"\nBenchmarking {name} at seq_len={seq}...")
            results[name][seq] = benchmark_attention(
                model_class=config["class"],
                model_kwargs=config["kwargs"],
                batch_size=batch_size,
                seq_len=seq
            )
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE - Results saved to results/benchmark_results.json")
    print("="*70)
    