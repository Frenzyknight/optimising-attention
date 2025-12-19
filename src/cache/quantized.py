import torch
'''
some questions to keeep in mind:
granularity, how many different scales do we store ?
one scale per tensor (batch_len, seq_len, num_heads, head_dim)
'''
class QuantizedKVCache():
    def __init__(self, batch_len, seq_len, num_heads, head_dim):
        self.cached_k = torch.ones((batch_len, num_heads, seq_len, head_dim), dtype=torch.int8)
        self.cached_v = torch.ones((batch_len, num_heads, seq_len, head_dim), dtype=torch.int8)
        self.scales_k = torch.zeros(batch_len, num_heads, seq_len, 1)
        self.scales_v = torch.zeros(batch_len, num_heads, seq_len, 1) #the 1 here is a broadcasting placeholder
        self.current_pos = 0


    def update(self, new_k, new_v):
        self.scales_k[:, :, self.current_pos, :] = new_k.abs().max(dim=-1, keepdim=True).values.squeeze(2)/127  #we use 127 here cause of int8 range
        self.scales_v[:, :, self.current_pos, :] = new_v.abs().max(dim=-1, keepdim=True).values.squeeze(2)/127
        self.cached_k[:, :, self.current_pos, :] = (new_k/self.scales_k[:, :, self.current_pos, :].unsqueeze(2)).round().clamp(-127, 127).to(torch.int8).squeeze(2)
        self.cached_v[:, :, self.current_pos, :] = (new_v/self.scales_v[:, :, self.current_pos, :].unsqueeze(2)).round().clamp(-127, 127).to(torch.int8).squeeze(2)
        self.current_pos += 1
        
    def get(self):
        return (
            self.cached_k[:, :, :self.current_pos, :] * self.scales_k[:, :, :self.current_pos, :],
            self.cached_v[:, :, :self.current_pos, :] * self.scales_v[:, :, :self.current_pos, :]
        )

cache = QuantizedKVCache(batch_len=2, seq_len=1024, num_heads=8, head_dim=64)
original_k = torch.randn(2, 8, 1, 64)  #seq_len is 1 becasue it will be kv is calculated per token
original_v = torch.randn(2, 8, 1, 64)
cache.update(original_k, original_v)
recovered_k, recovered_v = cache.get()
print(f"Max error: {(original_k.squeeze(2) - recovered_k.squeeze(2)).abs().max()}")