import numpy as np
import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class AddPads(nn.Module):
    "add pads to max_seq_len"
    def __init__(self, max_seq_len=512):
        super(AddPads, self).__init__()
        self.max_seq_len = max_seq_len
    
    def forward(self, x, device):
        # x can be [batch, seq_len, d_input]
        # e.g. "say hello to world" 1x4
        
        if x.size()[1] < self.max_seq_len:
            B, S = x.size()                             # 1x4
            pads = torch.zeros([B, self.max_seq_len-S], dtype=torch.long).to(device) # 1x(512-4) let a pad be [0]
            x = torch.cat([x, pads], 1)                 # 1x512
        return x

class EmbedEncode(nn.Module):
    "also multiply the weights by sqrt(d_model) after embedding according to the paper"
    def __init__(self, d_model=512, d_input=1024, max_seq_len=512, dropout=None):
        super(EmbedEncode, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(d_input, d_model)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        # 1. Embedding
        x = self.embedding(x)            # 1x512x512
        x = x * math.sqrt(self.d_model)  # 1x512x512

        # 2. Position Encoding
        x = self.pos_encoding(x)         # 1x512x512
        if self.dropout is not None:
            x = self.dropout(x)

        return x
    
    def pos_encoding(self, x):
        "PE (pos, 2i)   = sin(pos / 10000^(2i/d_model))" # position is word pos in sequence
        "PE (pos, 2i+1) = cos(pos / 10000^(2i/d_model))" # i is index in d_model
        B, _, _ = x.size() # 1x512x512
        device = next(self.parameters()).device
        even_i = torch.arange(0, self.d_model, 2).to(device).float()           # 256 (d_model / 2)
        denominator = torch.pow(even_i, (even_i / self.d_model))               # 256 (d_model / 2)
        position = torch.arange(self.max_seq_len).to(device).reshape(self.max_seq_len, 1) # 512x1 (seq_len x 1)

        even_PE = torch.sin(position / denominator)                            # 512x256 (seq_len x (d_model/2))
        odd_PE  = torch.cos(position / denominator)                            # 512x256 (seq_len x (d_model/2))

        stacked = torch.stack([even_PE, odd_PE], dim=-1)                       # 512x256x2 (seq_len x (d_model/2) x 2)
        pe = torch.flatten(stacked, start_dim=-2, end_dim=-1)                  # 512x512 (seq_len x d_model) [[even_0 odd_0 even_1 odd_1...]...]
        batch_pe = pe.unsqueeze(0).repeat(B, 1, 1)                             # 1x512x512
        x = x + batch_pe                                                       # 1x512x512
        return x

class LayerNorm(nn.Module):
    def __init__(self, d_model=512, max_seq_len=512, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(max_seq_len, d_model))
        self.beta =  nn.Parameter(torch.zeros(max_seq_len, d_model))

    def forward(self, x):
        # x: 1x512x512 (d_batch, seq_len, d_model)
        mean = x.mean(-1).mean(-1)              # 1 (d_batch)
        device = next(self.parameters()).device

        diff = torch.empty(x.size()).to(device) # 1x512x512
        for i in range(x.size()[0]):
            diff[i] = (x[i] - mean[i]) ** 2     # find better solution
        
        var = diff.mean(-1).mean(-1)            # 1 (d_batch)
        std = (var + self.eps).sqrt()           # 1 (d_batch)

        y = torch.empty(x.size()).to(device)    # 1x512x512
        for i in range(x.size()[0]):
            y[i] = (x[i] - mean[i]) / std[i]    # find better solution

        out = y * self.gamma + self.beta        # 1x512x512 (d_batch, seq_len, d_model)
        return out

class MultiHeadAttention(nn.Module):
    "take x but linear maps to q k v"
    "also linear maps the output at the end"
    def __init__(self, d_model=512, max_seq_len=512, num_heads=8, dropout=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.linearQ  = nn.Linear(d_model, 1*d_model) # 1x512x512 -> 1x512x512
        self.linearKV = nn.Linear(d_model, 2*d_model) # 1x512x512 -> 1x512x1024
        self.linearOut = nn.Linear(d_model, d_model) # do linear mapping at the end
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x, m=None, decoderMask=False):
        B, _, _ =  x.size()                                                     # 1x512x512 (d_batch, seq_len, d_model)
        # m has the same size as x if not None
        q_ = self.linearQ(x)                                                    # 1x512x512  (d_batch, seq_len, 1 x d_model)
        kv_ = self.linearKV(x) if m is None else self.linearKV(m)               # 1x512x1024 (d_batch, seq_len, 2 x d_model)

        qkv = torch.cat([q_, kv_], -1)                                          # 1x512x1536 (d_batch, seq_len, 3 x d_model)
        
        qkv = qkv.reshape(B, self.max_seq_len, self.num_heads, 3 * self.d_head) # 1x512x8x192 (d_batch, seq_len, num_heads, 3 x d_head) d_head is 64
        qkv = qkv.permute(0, 2, 1, 3)                                           # 1x8x512x192 (d_batch, num_heads, seq_len, 3 x d_head)

        q, k, v = qkv.chunk(3, dim=-1)                                       # Q: 1x8x512x64 (d_batch, num_heads, seq_len, d_head)
        
        next_v, _ = self.scaled_dot_product(q, k, v, decoderMask)               # 1x8x512x64 (d_batch, num_heads, seq_len, d_head)

        next_v = next_v.permute(0, 2, 1, 3)                                     # 1x512x8x64 (d_batch, seq_len, num_heads, d_head)
        next_v = next_v.reshape(B, self.max_seq_len, self.d_model)              # 1x512x512 (d_batch, seq_len, d_model)

        out = self.linearOut(next_v)                                            # 1x512x512 (d_batch, seq_len, d_model)
        return out

    def scaled_dot_product(self, q, k, v, decoderMask=False):
        d_q = q.size()[-1]                                             # 64 (d_head)
        scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_q) # 1x8x512x512 (d_batch, num_heads, seq_len, seq_len)
        # 1x8x512x64 dot (1x8x512x64)^T = 1x8x512x64 dot 1x8x64x512 = 1x8x512x512

        if decoderMask:
            scaled = scaled + self.make_mask(scaled.size())            # 1x8x512x512
        
        attention = F.softmax(scaled, dim=-1)                          # 1x8x512x512

        if self.dropout is not None:
            attention = self.dropout(attention)

        values = torch.matmul(attention, v)                            # 1x8x512x64 ( 1x8x512x512 dot 1x8x512x64 = 1x8x512x64)
        return values, attention
    
    def make_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.full(size, float('-inf')).to(device)              # 1x8x512x512
        mask = torch.triu(mask, diagonal=1)                            # 1x8x512x512
        return mask

class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_hidden=1024, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        #                     1x512x512  (d_batch, seq_len, d_model)
        x = self.linear1(x) # 1x512x1024 (d_batch, seq_len, d_hidden)
        x = F.relu(x)       # 1x512x1024 (d_batch, seq_len, d_hidden)
        x = self.dropout(x) # 1x512x1024 (d_batch, seq_len, d_hidden)
        x = self.linear2(x) # 1x512x512  (d_batch, seq_len, d_model)
        return x

class GPTdecodeLayer(nn.Module):
    def __init__(self, d_model=512, max_seq_len=512, num_heads=8, eps=1e-5, dropout=0.1):
        super(GPTdecodeLayer, self).__init__()
        self.multiHeadAttn = MultiHeadAttention(d_model, max_seq_len, num_heads, dropout) # 1x512x512 -> 1x512x512
        self.norm1 = LayerNorm(d_model, max_seq_len, eps)                                 # 1x512x512 -> 1x512x512
        self.feedForward = FeedForward(d_model, 2*d_model, dropout)          # 1x512x512 -> 1x512x512 
        self.norm2 = LayerNorm(d_model, max_seq_len, eps)                                 # 1x512x512 -> 1x512x512

    def forward(self, x):
        #                                                x: 1x512x512
        y = self.multiHeadAttn(x, decoderMask=True)       # 1x512x512
        y = x + y                                # Residual 1x512x512
        y = self.norm1(y)                                 # 1x512x512
        z = self.feedForward(y)                           # 1x512x512
        z = y + z                                # Residual 1x512x512
        z = self.norm2(z)                                 # 1x512x512
        return z
    
class GPT(nn.Module):
    def __init__(self, d_model=512, d_input=1024, max_seq_len=512, N=6, num_heads=8, eps=1e-5, dropout=0.1):
        super(GPT, self).__init__()
        self.d_input = d_input
        self.max_seq_len = max_seq_len
        self.add_pads = AddPads(max_seq_len)                                                          # 1x4       -> 1x512
        self.embed_encode = EmbedEncode(d_model, d_input, max_seq_len, dropout)                       # 1x512     -> 1x512x512
        self.norm = LayerNorm(d_model, max_seq_len, eps)                                              # 1x512x512 -> 1x512x512
        self.decoders = self.clones(GPTdecodeLayer(d_model, max_seq_len, num_heads, eps, dropout), N) # 1x512x512 -> 1x512x512
        self.linear = nn.Linear(d_model, d_input)                                                     # 1x512x512 -> 1x512x1024 (map to output prediction)

    def clones(self, module, N):
        "Produce N identical layers."
        modules = [copy.deepcopy(module) for _ in range(N)]
        return nn.ModuleList(modules)

    def forward(self, x, y=None):
        # x: 1x4, y: 1x3
        n_batch, _ = x.size()

        device = next(self.parameters()).device
        x = self.add_pads(x, device) # 1x512
        x = self.embed_encode(x)     # 1x512x512
        x = self.norm(x)             # 1x512x512
        for decoder in self.decoders:
            x = decoder(x)           # 1x512x512

        x = self.linear(x)           # 1x512x1024

        if y is None:
            loss = None
        else:
            y = self.add_pads(y, device) # 1x512
            x = x.reshape(n_batch * self.max_seq_len, self.d_input)
            y = y.reshape(n_batch * self.max_seq_len)
            loss = F.cross_entropy(x, y)

        return x, loss               # 1x512x512

    def generate(self, x, new_seq_len):

        for _ in range(new_seq_len):
            z = x
            if z.size()[1] > self.max_seq_len:
                z = z[:,-self.max_seq_len:]
            z, _ = self(z)
            z = z[:, -1, :]
            probs = F.softmax(z, dim=-1)
            char_vecs = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, char_vecs], dim=-1)
        return x