import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import math

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: tuple):
    """
    将输入的张量按三维窗口大小划分为多个窗口
    
    Args:
        x (Tensor): 输入张量，形状为 (B, T, H, W, C)
        window_size (tuple): 三维窗口的大小，表示在时间、高度、宽度三个维度上的窗口大小
    
    Returns:
        windows (Tensor): 划分后的窗口，形状为 (num_windows*B, T_w, H_w, W_w, C)
    """
    B, T, H, W, C = x.shape
    T_w, H_w, W_w = window_size  # 解包窗口大小，分别为 T_w, H_w, W_w
    
    # 将输入数据重塑为三维窗口形状
    x = x.view(B, T // T_w, T_w, H // H_w, H_w, W // W_w, W_w, C)
    
    # 调整维度顺序，确保窗口的排列
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    
    # 重新排列形状为 (num_windows*B, T_w, H_w, W_w, C)
    windows = windows.view(-1, T_w, H_w, W_w, C)
    
    return windows

def window_reverse(windows, window_size: tuple, T: int, H: int, W: int):
    """
    将划分的窗口恢复为原始的特征图
    
    Args:
        windows (Tensor): 划分后的窗口，形状为 (num_windows*B, T_w, H_w, W_w, C)
        window_size (tuple): 三维窗口大小 (T_w, H_w, W_w)
        T (int): 时间维度的大小
        H (int): 图像的高度
        W (int): 图像的宽度
    
    Returns:
        x (Tensor): 还原后的特征图，形状为 (B, T, H, W, C)
    """
    T_w, H_w, W_w = window_size  # 解包窗口大小

    # 计算每个窗口的数量
    num_windows = (T * H * W) // (H_w * W_w * T_w)
    
    # 计算批次大小 B
    B = windows.shape[0] // num_windows
    
    # 获取窗口的通道数 C
    C = windows.shape[-1]

    # 将窗口重塑为 (B, num_windows, T_w, H_w, W_w, C)
    x = windows.view(B, num_windows, T_w, H_w, W_w, C)
    
    
    # 重塑为 (B, T, H//H_w, W//W_w, T_w, H_w, W_w, C)
    x = x.view(B, T // T_w, H // H_w, W // W_w, T_w, H_w, W_w, C)
    
    # 调整维度顺序，得到 (B, T, H, W, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6, 7).contiguous()
    
    # 重塑为原始形状
    x = x.view(B, T, H, W, C)
    
    return x


class PatchEmbed(nn.Module):
    """
    3D Image to Patch Embedding
    """
    def __init__(self, patch_size=(16, 3, 3), in_c=1, embed_dim=96, norm_layer=None): # window_size 通常指的是每个窗口内包含的补丁（patch）数量
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim

        # 使用Conv3d代替Conv2d，适应三维数据
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 如果传入了norm_layer，则使用，否则使用Identity层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 假设输入维度为 (B, T, H, W, C)
        B, T, H, W, C = x.shape

        # 将C维度移到前面，调整为 (B, C, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # 转为 (B, C, T, H, W)

        # 进行补丁嵌入
        x = self.proj(x)  # 进行卷积操作，生成补丁
        _, embed_dim, T_p, H_p, W_p = x.shape  # 获取卷积后数据的尺寸

        # 展平并调整维度 (B, T_p * H_p * W_p, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # 归一化
        return x, T_p, H_p, W_p


class PatchMerging(nn.Module):
    r""" Patch Merging Layer for 3D data.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 4 * dim, bias=False)  # From 8*C to 4*C
        self.norm = norm_layer(8 * dim)

    def forward(self, x, T, H, W):
        """
        x: B, T*H*W, C
        """
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"

        # Reshape to (B, T, H, W, C)
        x = x.view(B, T, H, W, C)
        # Padding if needed (if H, W, or T is not divisible by 2)
        pad_input = (T % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))  # Pad T, H, W dimensions if odd
        # Create 8 patches by taking 2x2x2 windows from T, H, W dimensions
        x0 = x[:, 0::2, 0::2, 0::2, :]  # [B, T/2, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, 0::2, :]  # [B, T/2, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, 0::2, :]  # [B, T/2, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, 0::2, :]  # [B, T/2, H/2, W/2, C]
        x4 = x[:, 0::2, 0::2, 1::2, :]  # [B, T/2, H/2, W/2, C]
        x5 = x[:, 1::2, 0::2, 1::2, :]  # [B, T/2, H/2, W/2, C]
        x6 = x[:, 0::2, 1::2, 1::2, :]  # [B, T/2, H/2, W/2, C]
        x7 = x[:, 1::2, 1::2, 1::2, :]  # [B, T/2, H/2, W/2, C]

        # Concatenate the 8 patches along the channel dimension (C)
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # [B, T/2, H/2, W/2, 8*C]
        # Reshape to (B, T/2*H/2*W/2, 8*C)
        x = x.view(B, -1, 8 * C)

        # Apply normalization and reduction
        x = self.norm(x)
        x = self.reduction(x)  # [B, T/2*H/2*W/2, 4*C]
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module for 3D data.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height, width, and time of the window (Th, Tw, Tw).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Th, Tw, Tw)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子

        # Define a parameter table for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # [2*Th-1 * 2*Tw-1 * 2*Tw-1, nH]

        # Get pairwise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size[0])  # Time coordinates, 0 to Th-1
        coords_h = torch.arange(self.window_size[1])  # Height coordinates, 0 to Tw-1
        coords_w = torch.arange(self.window_size[2])  # Width coordinates, 0 to Tw-1
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w]))  # [3, Th, Tw, Tw]
        coords_flatten = torch.flatten(coords, 1)  # [3, Th*Tw*Tw]
        # Compute relative coordinates
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, Th*Tw*Tw, Th*Tw*Tw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Th*Tw*Tw, Th*Tw*Tw, 3]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # [Th*Tw*Tw, Th*Tw*Tw]
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection layer
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input features with shape of (num_windows*B, Th*Tw*Tw, C)
            mask: (0/-inf) mask with shape of (num_windows, Th*Tw*Tw, Th*Tw*Tw) or None
        """
        B_, N, C = x.shape
        # qkv projection: -> [batch_size*num_windows, Th*Tw*Tw, 3 * total_embed_dim]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Separate q, k, v

        # Scaling
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Apply relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Th*Tw*Tw, Th*Tw*Tw]

        # Expand relative_position_bias to match attn shape
        relative_position_bias = relative_position_bias.unsqueeze(0)  # [1, nH, Th*Tw*Tw, Th*Tw*Tw]
        # Add bias to attention
        attn = attn + relative_position_bias

        # Masking (optional)
        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size tuple(int): Window size.
        shift_size tuple(int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(4,2,2), shift_size=(2,1,1),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert all(0 <= shift < window for shift, window in zip(self.shift_size, self.window_size)), "shift_size must be less than window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        T, H, W = self.T, self.H, self.W  # 在class BasicLayer里有进行赋值 blk.T, blk.H, blk.W
        B, L, C = x.shape
        assert L == T * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # x = x.view(B, T, H, W, C)
        c = x.shape[-1]
        x = x.reshape(B, C, H, W, T) # c,t换位置为了padding
        
        # 把feature map给pad到window size的整数倍
        w_t, w_h, w_w = self.window_size
        # 假设 window_size 是一个三维元组 (window_size_T, window_size_H, window_size_W)
        pad_t = (w_t - T % w_t) % w_t  # 对时间维度T进行padding
        pad_h = (w_h - H % w_h) % w_h  # 对高度H进行padding
        pad_w = (w_w - W % w_w) % w_w  # 对宽度W进行padding 2-3%2  % 2

        # 对x进行padding，padding的顺序是 (left, right, top, bottom, front, back)
        # padding的顺序是 (width_padding_left, width_padding_right, height_padding_top, height_padding_bottom, time_padding_front, time_padding_back)
        x = F.pad(x, (0, pad_t, 0, pad_w, 0, pad_h))
        x = x.permute(0, 4, 2, 3, 1)
        
        # 获取新的尺寸 Padding后尺寸改变
        _, Tp, Hp, Wp, _ = x.shape
        
        # cyclic shift
        if any(shift > 0 for shift in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, w_t * w_h * w_w, C)  
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, w_t, w_h, w_w, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Tp, Hp, Wp) 

        # reverse cyclic shift
        if any(shift > 0 for shift in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :T, :H, :W, :].contiguous()
        
        x = x.view(B,T * H * W, C)
        # Feed Forward Network
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size tuple(int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.wt, self.wh, self.ww = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = (self.wt//2, self.wh//2, self.ww//2)

        # build blocks
        self.blocks = nn.ModuleList([  # 通过 nn.ModuleList，将多个 SwinTransformerBlock 层按顺序构建成一个模块列表
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, T, H, W):
        # calculate attention mask for SW-MSA
        # 保证Tp, Hp和Wp是window_size的整数倍
        Tp = int(np.ceil(T / self.wt)) * self.wt
        Hp = int(np.ceil(H / self.wh)) * self.wh
        Wp = int(np.ceil(W / self.ww)) * self.ww
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1,Tp, Hp, Wp, 1), device=x.device)  # [1, Tp, Hp, Wp, 1]
        t_slices = (slice(0, -self.wt),
                    slice(-self.wt, -self.wt),
                    slice(-self.wt, None))
        h_slices = (slice(0, -self.wh),
                    slice(-self.wh, -self.wh),
                    slice(-self.wh, None))
        w_slices = (slice(0, -self.ww),
                    slice(-self.ww, -self.ww),
                    slice(-self.ww, None))
        cnt = 0
        for t in t_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, t, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.wt * self.wh * self.ww)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, T, H, W):
        if isinstance(x, int):
            x = torch.tensor(x).float().to(device) 
        attn_mask = self.create_mask(x, T, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.T, blk.H, blk.W = T, H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x,T, H, W)
            T, H, W =(T + 1) // 2, (H + 1) // 2, (W + 1) // 2

        return x, T, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (tuple(int)): Patch size
        in_chans (int): Number of input image channels.
        num_classes (int): Number of classes for classification head.
        embed_dim (int): Patch embedding dimension.
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple(int)): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate. 
        norm_layer (nn.Module): Normalization layer.
        patch_norm (bool): If True, add normalization after patch embedding.
        use_checkpoint (bool): Whether to use checkpointing to save memory.
    """

    def __init__(self, patch_size=(16,3,3), in_chans=1, num_classes=2,
                 embed_dim=96, depths=(2, 2, 4, 2), num_heads=(2, 2, 4, 6),
                 window_size=(4,2,2), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 4 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 4 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x, T, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, T, H, W = layer(x, T, H, W)

        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        y = torch.flatten(x, 1)
        x = self.head(y)
        return x