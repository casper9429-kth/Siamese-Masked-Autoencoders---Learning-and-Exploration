# class Attention(nn.Module):
#     """
#         (Multi-head) self-attention layer.
#     """
#     embed_dim : int
#     num_heads : int
#     def setup(self):
#         self.qkv = nn.Dense(self.embed_dim*3)

#     def __call__(self, x):
#         B, N, D = x.shape # B: #batches, N: #patches + 1 (cls token), D: embed_dim
#         proj_dim = D // self.num_heads
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2] # shape: B x num_heads x N x proj_dim

#         att_scores = q @ k.transpose(2,3) # B x num_heads x N x N
#         att_scores_sm = torch.tensor(nn.functional.softmax(att_scores, -1))
#         weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x N x N x proj_dim
#         sum = weighted_vals.sum(dim=2) # B x num_heads x N x proj_dim
        
#         out  = sum.reshape(B, N, D)
#         return out

# class CrossAttention(nn.Module):
#     """
#         (Multi-head) cross attention layer.
#     """
#     def __init__(self, dim, num_heads) -> None:
#         super().__init__(dim, num_heads)

#         self.num_heads = num_heads
#         self.q = nn.Linear(dim, dim)
#         self.kv = nn.Linear(dim, dim*2)

#     def forward(self, x1, x2):
#         B, N, D = x1.shape # B: #batches, N: #patches + 1 (cls token), D: embed_dim
#         proj_dim = D // self.num_heads
#         # get queries from x2 (embedded, encoded and masked future frame)
#         q = self.q(x2).reshape(B, N, self.num_heads, proj_dim).permute(0, 2, 1, 3)
#         # get keys and values from unmasked frame
#         kv = self.kv(x1).reshape(B, N, 2, self.num_heads, proj_dim).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         # from here it's the same as the self-attention
#         att_scores = q @ k.transpose(2,3) # B x num_heads x N x N
#         att_scores_sm = torch.tensor(nn.functional.softmax(att_scores, -1))
#         weighted_vals = v[:,:,:,None,:] * att_scores_sm.transpose(-2,-1)[:,:,:,:,None] # B x num_heads x N x N x proj_dim
#         sum = weighted_vals.sum(dim=2) # B x num_heads x N x proj_dim
        
#         out  = sum.reshape(B, N, D)
#         return out