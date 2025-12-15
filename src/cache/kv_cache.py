import torch

class KVCache():
    def __init__(self, batch_size, num_heads, max_seq_len, head_dim):
        self.cached_k = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)
        self.cached_v = torch.zeros(batch_size, num_heads, max_seq_len, head_dim)
        self.current_pos = 0

    def update(self, new_k, new_v):
        # self.cached_k = torch.cat([self.cached_k, new_k], dim=2) #concat is slower because new malloc, copy old cache always, frag, mem cpys grow, sync point problems O(n^2)
        self.cached_k [:, :, self.current_pos, :] = new_k.squeeze(2) #contigous mem, allocates once, can resuse mem, rotate pointers
        self.cached_v [:, :, self.current_pos, :] = new_v.squeeze(2)
        self.current_pos += 1

    def get(self):
        return (
            self.cached_k [:, :, :self.current_pos, :],
            self.cached_v [:, :, :self.current_pos, :],
        )
 

cache = KVCache(batch_size=2, num_heads=8, max_seq_len=1024, head_dim=64)

for i in range(3):
    new_k = torch.randn(2, 8, 1, 64)
    new_v = torch.randn(2, 8, 1, 64)
    cache.update(new_k=new_k, new_v=new_v)
k, v = cache.get()
print(f"After 3 updates this is the shape: {k.shape}")
