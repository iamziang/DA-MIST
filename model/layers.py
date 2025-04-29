import torch
from torch import nn
from einops import rearrange
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class State_Aware_Dual_Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head 
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkvg = nn.Linear(dim, inner_dim * 4, bias=False)
        self.to_out = nn.Sequential(nn.Linear(2 * inner_dim, dim), nn.Dropout(dropout))
        self.register_buffer('pos_template', None)
        # self.register_buffer("position_enc", self._generate_position_encoding(5000))
        self.sigma = nn.Parameter(torch.full((1, heads, 1, 1), 1.92))
       

    def prepare_positional_differences(self, n):
        if self.pos_template is None or self.pos_template.size(0) != n:
            pos = torch.arange(1, n + 1).float().cuda()
            self.pos_template = (pos.view(-1, 1) - pos.view(1, -1)).abs()

    # def _generate_position_encoding(self, max_len):
    #     position = torch.arange(max_len).float().unsqueeze(1)
    #     div_term = torch.exp(
    #         torch.arange(0, self.dim_head, 2).float() * 
    #         (-math.log(10000.0) / self.dim_head))
        
    #     enc = torch.zeros(max_len, self.dim_head)
    #     enc[:, 0::2] = torch.sin(position * div_term)
    #     enc[:, 1::2] = torch.cos(position * div_term)
    #     return enc  # [max_len, dim_head]

    def forward(self, x):
        b, n, d = x.size()
        self.prepare_positional_differences(n)
        # pos_enc = self.position_enc[:n].unsqueeze(0).unsqueeze(0) 
        q, k, vg, vsigma = self.to_qkvg(x).chunk(4, dim=-1)
        q, k, vg, vsigma = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, vg, vsigma))
        
        # Corr^G = softmax(QK^T/√D)
        # q = q + pos_enc.to(q.device)
        # k = k + pos_enc.to(k.device)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        Corr_G  = self.attend(dots)

        # Corr^sigma
        sigma = self.sigma.expand(b, self.heads, n, 1)  
        diffs = self.pos_template[:n, :n].unsqueeze(0).unsqueeze(0).repeat(b, self.heads, 1, 1)  # torch.Size([128, 4, 200, 200])
        gaussian_mask = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-(diffs ** 2) / (2 * sigma ** 2)) # torch.Size([128, 4, 200, 200])
        Corr_sigma = gaussian_mask / gaussian_mask.sum(-1, keepdim=True)

        out = torch.cat([torch.matmul(Corr_G, vg), torch.matmul(Corr_sigma, vsigma)], dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Global_Attention(nn.Module):
    '''Ablation Study (Only Global Branch)'''
    def __init__(self, dim, heads=4, dim_head=128, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head 
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  
            nn.Dropout(dropout)
        )
        self.register_buffer("position_enc", self._generate_position_encoding(5000))
    
    def _generate_position_encoding(self, max_len):
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_head, 2).float() * 
            (-math.log(10000.0) / self.dim_head))
        
        enc = torch.zeros(max_len, self.dim_head)
        enc[:, 0::2] = torch.sin(position * div_term)
        enc[:, 1::2] = torch.cos(position * div_term)
        return enc  # [max_len, dim_head]

    def forward(self, x):
        b, n, d = x.size()
        pos_enc = self.position_enc[:n].unsqueeze(0).unsqueeze(0) 
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))     
        q = q + pos_enc.to(q.device)
        k = k + pos_enc.to(k.device)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn1 = torch.matmul(attn, v)  # [b, h, n, d_head]       
        out = rearrange(attn1, 'b h n d -> b n (h d)')  # [b, n, inner_dim]
        return self.to_out(out)  # [b, n, dim]

class Local_Attention(nn.Module):
    '''Ablation Study (Only Local Branch)'''
    def __init__(self, dim, heads=4, dim_head=128, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_g = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.register_buffer('pos_template', None)
        self.sigma = nn.Parameter(torch.full((1, heads, 1, 1), 1.92))

    def prepare_positional_differences(self, n):
        if self.pos_template is None or self.pos_template.size(0) != n:
            pos = torch.arange(1, n + 1).float().cuda()
            self.pos_template = (pos.view(-1, 1) - pos.view(1, -1)).abs()

    def forward(self, x):
        b, n, d = x.size()
        self.prepare_positional_differences(n)
        vsigma = rearrange(self.to_g(x), 'b n (h d) -> b h n d', h=self.heads)
        # Corr^sigma
        sigma = self.sigma.expand(b, self.heads, n, 1)  
        diffs = self.pos_template[:n, :n].unsqueeze(0).unsqueeze(0).repeat(b, self.heads, 1, 1)  # torch.Size([128, 4, 200, 200])
        gaussian_mask = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-(diffs ** 2) / (2 * sigma ** 2)) # torch.Size([128, 4, 200, 200])
        Corr_sigma = gaussian_mask / gaussian_mask.sum(-1, keepdim=True)
        attn2 = torch.matmul(Corr_sigma, vsigma)  # [b, h, n, d_head]       
        out = rearrange(attn2, 'b h n d -> b n (h d)')  # [b, n, inner_dim]
        return self.to_out(out)  # [b, n, dim]

class Memory_Bank(nn.Module):
    def __init__(self, nums, dim):
        super().__init__()
        self.dim = dim  
        self.nums = nums  
        self.memory_block = nn.Parameter(torch.empty(nums, dim))  
        self.sig = nn.Sigmoid() 
        self.reset_parameters()  

    def reset_parameters(self): 
        stdv = 1. / math.sqrt(self.memory_block.size(1))
        self.memory_block.data.uniform_(-stdv, stdv)
        if self.memory_block is not None:
            self.memory_block.data.uniform_(-stdv, stdv)
      
    def forward(self, data):  
        attention = self.sig(torch.einsum('btd,kd->btk', data, self.memory_block) / (self.dim**0.5))   
        temporal_att = torch.topk(attention, self.nums//16+1, dim = -1)[0].mean(-1)  
        augment = torch.einsum('btk,kd->btd', attention, self.memory_block) 
        return temporal_att, augment
        


