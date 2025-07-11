import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from ConExBiMamba.ExBiMamba import ExBimamba,Mamba0
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

# helper classes

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


#-----------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

        # 对自注意力和前馈网络使用 LayerNorm 进行预归一化
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = PreNorm(dim, self.ff1)

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask) + x  # 自注意力 + 残差连接
        x = self.ff1(x) + x  # 前馈网络 + 残差连接
        x = self.post_norm(x)  # 最后进行 LayerNorm 归一化
        return x


import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(TransformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            ))

    def forward(self, x, mask=None):
        for block in self.layers:
            x = block(x, mask=mask)
        return x


    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
from mamba_ssm.modules.mamba2 import Mamba2


class BiMamba2Encoder(nn.Module):
    def __init__(self, d_model, n_state, headdim):
        super(BiMamba2Encoder, self).__init__()
        self.d_model = d_model
        
        self.mamba2 = Mamba2(d_model, n_state, headdim=32)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba2(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba2(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual
        return output
    
class BiMambas2_FFN(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        headdim=32,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BiMamba2Encoder(
                d_model = dim, 
                n_state = 16,
                headdim=32
            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
from mamba_ssm.modules.mamba_simple import Mamba

# from mamba.mamba_ssm.modules.mamba_simple import Mamba#exp75，76

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # self.concat_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),#concat的时候加上2 * d_model
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # self.feed_forward2 = nn.Sequential(
        #     nn.Linear(d_model, d_model * 4),
        #     Swish(),
        #     nn.Dropout(0),
        #     nn.Linear(d_model * 4, d_model),
        #     nn.Dropout(0)
        # )

    def forward(self, x):
        # #------------去掉双向，只剩单向----------------------
        # # Residual connection of the original input
        # residual = x

        # # #FNN
        # # mamba_in = self.norm1(x)
        # # ff_in = self.feed_forward(mamba_in)
        # # input = ff_in + residual
        
        # # Forward Mamba
        # # x_norm = self.norm1(input)#2FNN
        # x_norm = self.norm1(x)
        # mamba_out_forward = self.mamba(x_norm)

        # # # Backward Mamba
        # # x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        # # mamba_out_backward = self.mamba(x_flip)
        # # mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
        # # mamba_out = mamba_out_forward + mamba_out_backward

        # mamba_out = mamba_out_forward
        
        # mamba_out = self.norm2(mamba_out)
        # ff_out = self.feed_forward(mamba_out)

        # output = ff_out + residual


        #-=-----去掉FFN-----
        # Residual connection of the original input
        # residual = x

        
        # # Forward Mamba
        # x_norm = self.norm1(x)
        # mamba_out_forward = self.mamba(x_norm)

        # # Backward Mamba
        # x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        # mamba_out_backward = self.mamba(x_flip)
        # mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
        # mamba_out = mamba_out_forward + mamba_out_backward

        
        # mamba_out = self.norm2(mamba_out)
        # ff_out = self.feed_forward(mamba_out)
        # output = mamba_out + residual

        # #-----------去掉3个LayerNorm--------------
        # # Residual connection of the original input
        # residual = x
        
        # # Forward Mamba
        # # x_norm = self.norm1(x)
        # mamba_out_forward = self.mamba(x)

        # # # Backward Mamba
        # x_flip = torch.flip(x, dims=[1])  # Flip Sequence
        # mamba_out_backward = self.mamba(x_flip)
        # mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
        # mamba_out = mamba_out_forward + mamba_out_backward

        
        # # mamba_out = self.norm2(mamba_out)
        # ff_out = self.feed_forward(mamba_out)

        # output = ff_out + residual


        # # #----------原始 -------
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence

        # x_flip = self.norm1(x_flip)#2.14加
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # print("mamba_out_forward",mamba_out_forward.shape)#torch.Size([20, 208, 144])
        # print("mamba_out_backward",mamba_out_backward.shape)#torch.Size([20, 208, 144])


        # ADD forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        # print("add_mamba_out",mamba_out.shape) #([20, 208, 144])
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)
        # ff_out = self.feed_forward2(mamba_out)


        output = ff_out + residual
        return output

        
    
class BiMambas_FFN(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BiMambaEncoder(
                d_model = dim, 
                n_state = 16
            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x


#-----------------------------------------------------------------Conformer Mamba-------------------------------------------------------------------------------
# Conformer Mamba Block

class BiMamba_Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        # self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ExBimamba=ExBimamba(d_model=dim)
        # self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # self.ExBimamba = PreNorm(dim, self.ExBimamba)
        # self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # # print("x1",x.shape)#torch.Size([32, 209, 144])
        # x = self.ff1(x) + x
        # # print("x2",x.shape)#torch.Size([32, 209, 144])
        # x = self.ExBimamba(x) + x
        # # print("x3",x.shape)#torch.Size([32, 209, 144])
        # x = self.conv(x) + x
        # # print("x4",x.shape)#torch.Size([32, 209, 144])
        # x = self.ff2(x) + x
        # # print("x5",x.shape)#torch.Size([32, 209, 144])

        x = self.ExBimamba(x)
        x = self.post_norm(x)
        # print("x6",x.shape)#torch.Size([32, 209, 144])
        return x

# Conformer_BiMamba

class BiMambas(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BiMamba_Block(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal

            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x
    
#---------------------------------------------------------------------Mamba-------------------------------------------------------------------------------
# Mamba Block

class Mamba_Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        # self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.Mamba=Mamba0(d_model=dim)
        # self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # self.ExBimamba = PreNorm(dim, self.ExBimamba)
        # self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # # print("x1",x.shape)#torch.Size([32, 209, 144])
        # x = self.ff1(x) + x
        # # print("x2",x.shape)#torch.Size([32, 209, 144])
        # x = self.ExBimamba(x) + x
        # # print("x3",x.shape)#torch.Size([32, 209, 144])
        # x = self.conv(x) + x
        # # print("x4",x.shape)#torch.Size([32, 209, 144])
        # x = self.ff2(x) + x
        # # print("x5",x.shape)#torch.Size([32, 209, 144])

        x = self.Mamba(x)
        x = self.post_norm(x)
        # print("x6",x.shape)#torch.Size([32, 209, 144])
        return x

# Mamba

class Mambas(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(Mamba_Block(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal

            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x