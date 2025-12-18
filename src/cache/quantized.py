import torch
'''
some questions to keeep in mind:
granularity, how many different scales do we store ?
one scale per tensor (batch_len, seq_len, num_heads, head_dim)
'''

class QuantizedKVCache():
    def __init__(self, batch_len, seq_len, num_heads, head_dim):
        self.cached_k = torch.ones(batch_len, num_heads, seq_len, head_dim, dtype=torch.int8)
        self.cached_v = torch.ones(batch_len, num_heads, seq_len, head_dim, dtype=torch.int8)
        self.current_pos = 0
        self.scale = torch.randn(num_heads)

    def update(self, new_k, new_v):
        self.cached_k[:, :, self.current_pos, :] = round(new_k/self.scale)
        self.cached_v[:, :, self.current_pos, :] = round(new_v/self.scale)
        self.current_pos += 1
        
    def get(self):
        return (
            self.cached_k[:, :, :self.current_pos, :],
            self.cached_v[:, :, :self.current_pos, :]
        )