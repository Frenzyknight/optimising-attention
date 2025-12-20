import json 
import matplotlib.pyplot as plt

def load_results(filepath):
    pass

def plot_latency_vs_seq_len(resutls):
    pass

def plot_throughput_comparision(results, seq_len):
    pass

def plot_memory_comparison(results, seq_len):
    pass

if __name__ == "main":
    results = load_results("results/benchmark_results.json")
    plot_latency_vs_seq_len(results)
    plot_throughput_comparision(results, seq_len=1024)
    plot_memory_comparison(results, seq_len=1024)
    plt.show()