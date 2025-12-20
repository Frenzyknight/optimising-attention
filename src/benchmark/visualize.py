import json 
import matplotlib.pyplot as plt

def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)

def plot_latency_vs_seq_len(results):
    seq_len = [128, 512, 1024, 2048]
    plt.figure(figsize=(10, 6))
    for model_name, seq_data in results.items():
        latencies = [seq_data[str(seq)]["avg_latency"] * 1000 for seq in seq_len]
        plt.plot(seq_len, latencies, marker='o', label= model_name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Latency(ms)")
    plt.title("Attention latency vs Sequence Length")
    plt.legend()
    plt.show()
def plot_throughput_comparision(results):
    seqs = [128, 512, 1024, 2048]
    plt.figure(figsize=(10, 6))
    for model_name, seq_data in results.items():
        throughput = [seq_data[str(seq)]["throughput"] for seq in seqs]
        plt.plot(seqs, throughput, marker='o', label= model_name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput")
    plt.title("Throughput vs Sequence Length")
    plt.legend()
    plt.show()

def plot_memory_comparison(results):
    seqs = [128, 512, 1024, 2048]
    plt.figure(figsize=(10, 6))
    for model_name, seq_data in results.items():
        memory = [seq_data[str(seq)]["memory_mb"] for seq in seqs]
        plt.plot(seqs, memory, marker='o', label= model_name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory")
    plt.title("Memory usage vs Sequence Length")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    results = load_results("benchmark_results.json")
    plot_latency_vs_seq_len(results)
    plot_throughput_comparision(results)
    plot_memory_comparison(results)
    plt.show()