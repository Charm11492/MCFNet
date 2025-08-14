#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn
import math
import torch
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
# from .mamba_cross import CrossMamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
   
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

        

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
        
class LinearTransformLayer(nn.Module):
    def __init__(self, num_channels):
        super(LinearTransformLayer, self).__init__()
        # γ_h 和 β_h 作为可学习参数，形状为 (C, 1, 1)
        self.gamma_h = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.beta_h = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, E_0_h):
        # Z_0^h = E_0^h ⊙ γ_h + β_h
        Z_0_h = E_0_h * self.gamma_h + self.beta_h
        return Z_0_h

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super(Upsample, self).__init__()
        # 通道变换层，将输入通道数变为原来的4倍
        self.channel_transform = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1, stride=1, padding=0)
        # PixelShuffle层，空间上上采样因子为2
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        # 先进行通道变换
        x = self.channel_transform(x)
        # 使用PixelShuffle进行空间上采样
        x = self.pixel_shuffle(x)
        return x

# class DeformableTransConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
#         super(DeformableTransConv2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.stride = stride
        
#         # 偏移量卷积层，输出通道为 2 * kernel_size * kernel_size
#         self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        
#         # 转置卷积层，步幅为 2，输出的高度和宽度是输入的两倍
#         self.trans_conv = nn.ConvTranspose2d(in_channels*9, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1)

#     def forward(self, x):
#         b, c, h, w = x.shape
#         k = self.kernel_size
        
#         # 生成偏移量 -> (b, 2*k*k, h, w)
#         offsets = self.offset_conv(x)  # (b, 2*k*k, h, w)
        
#         # 生成标准网格 (b, h, w, 2)
#         grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device))
#         grid = torch.stack((grid_x, grid_y), dim=-1)  # (h, w, 2)
#         grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (b, h, w, 2)
        
#         # 将offsets reshape为每个采样点的偏移 -> (b, h, w, k*k, 2)
#         offsets = offsets.permute(0, 2, 3, 1).reshape(b, h, w, k * k, 2)  # (b, h, w, k*k, 2)

#         # 生成卷积核采样点的标准相对位置 (k*k, 2)
#         kernel_grid_y, kernel_grid_x = torch.meshgrid(torch.linspace(-1, 1, k, device=x.device), torch.linspace(-1, 1, k, device=x.device))
#         kernel_grid = torch.stack((kernel_grid_x, kernel_grid_y), dim=-1)  # (k, k, 2)
#         kernel_grid = kernel_grid.view(-1, 2)  # (k*k, 2)

#         # 计算每个位置的偏移量，得到实际的采样网格坐标
#         kernel_grid = kernel_grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, k*k, 2)
#         kernel_grid = kernel_grid.expand(b, h, w, -1, -1)  # (b, h, w, k*k, 2)
#         sampling_grid = grid.unsqueeze(-2) + kernel_grid + offsets  # (b, h, w, k*k, 2)
        
#         # 将采样网格reshape为(b, h, w, k*k, 2)，并按每个点单独采样
#         sampled_features = []
#         for i in range(k * k):
#             single_sampled = F.grid_sample(x, sampling_grid[:, :, :, i, :], align_corners=True)  # (b, c, h, w)
#             sampled_features.append(single_sampled)

#         # 将所有采样结果拼接 -> (b, c*k*k, h, w)
#         x = torch.cat(sampled_features, dim=1)

#         # 使用转置卷积进行上采样
#         x = self.trans_conv(x)
        
#         return x

# class AdaUp(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AdaUp, self).__init__()
#         self.adaptive_pooling = nn.AdaptiveAvgPool2d((1,1))  # 自适应池化到1x1
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积
#         self.trans_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 转置卷积

#     def forward(self, x):
#         # 自适应池化
#         L = self.adaptive_pooling(x)
        
#         # 1x1卷积进行通道上的特征分布序列交互
#         L1 = self.conv1x1(L)

#         L = L.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)
#         L1 = L1.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)
#         # 卷积核的逐元素乘法（广播）
#         W_f = self.trans_conv.weight *L * L1
        
#         # 转置卷积进行上采样
#         out = F.conv_transpose2d(x, W_f, stride=self.trans_conv.stride, padding=self.trans_conv.padding, output_padding=self.trans_conv.output_padding)

#         # 手动加上 bias
#         if self.trans_conv.bias is not None:
#             out += self.trans_conv.bias.view(1, -1, 1, 1)
        
#         return out

# class AdaUp(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(AdaUp, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = 2
#         self.padding = 1
#         self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        
#         # 自适应池化和1x1卷积
#         # self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         # 创建5个不同的转置卷积层，分别用于五种偏移
#         self.trans_convs = nn.ModuleList([
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=self.padding, output_padding=1)
#             for _ in range(5)
#         ])

#         # 1x1卷积，用于最后将通道数调整回原来的值
#         self.final_conv = nn.Conv2d(5 * out_channels, out_channels, kernel_size=1)

#     def forward(self, x ):
#         # 步骤1：自适应池化
    
#         L = self.adaptive_pooling(x)
#         # 步骤2：1x1卷积进行通道上的特征分布序列交互
#         L = self.conv1x1(L)
#         L = L.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)

#         # 步骤3：卷积核的逐元素乘法（广播）

#         W_fs = [self.trans_convs[i].weight * L  for i in range(5)]

#         # 不同方向的偏移操作
#         x_left = F.pad(x, [1, 0, 0, 0])[:, :, :, :-1]  # 向左偏移
#         x_right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]  # 向右偏移
#         x_up = F.pad(x, [0, 0, 1, 0])[:, :, :-1, :]    # 向上偏移
#         x_down = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]   # 向下偏移
#         x_center = x                                   # 不偏移（中心）

#         # 对每个偏移后的特征图执行动态卷积操作
#         out_left = F.conv_transpose2d(x_left, W_fs[0], stride=self.stride, padding=self.padding, output_padding=1)
#         out_right = F.conv_transpose2d(x_right, W_fs[1], stride=self.stride, padding=self.padding, output_padding=1)
#         out_up = F.conv_transpose2d(x_up, W_fs[2], stride=self.stride, padding=self.padding, output_padding=1)
#         out_down = F.conv_transpose2d(x_down, W_fs[3], stride=self.stride, padding=self.padding, output_padding=1)
#         out_center = F.conv_transpose2d(x_center, W_fs[4], stride=self.stride, padding=self.padding, output_padding=1)

#         # 在通道维度拼接卷积结果
#         out = torch.cat([out_left, out_right, out_up, out_down, out_center], dim=1)

#         # 通过1x1卷积将通道数恢复

#         # 手动加上 bias

#         out = self.final_conv(out)

#         bias_mean = 0
#         num_biases = 0
#         for i in range(5):
#             if self.trans_convs[i].bias is not None:
#                 bias_mean += self.trans_convs[i].bias.view(1, -1, 1, 1)
#                 num_biases += 1

#         if num_biases > 0:
#             bias_mean /= num_biases  # 计算 bias 的平均值
#             out += bias_mean          # 将 bias 平均值加到输出中

#         return out


class Spatial_Attention(nn.Module):
    def __init__(self, reduction=16, kernel_size=7):
        super(Spatial_Attention, self).__init__()
       
        self.attention = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)  # 空间注意力部分

    def forward(self, x):
     
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        x_attention = torch.cat([max_pool, avg_pool], dim=1)
        x = torch.sigmoid(self.attention(x_attention))
       
        return x

class AdaptivePixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor = 2):
        super(AdaptivePixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        # 用于处理 a 的卷积
        self.conv_a = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        # 用于融合 b 特征的卷积
        self.sa = Spatial_Attention()
        # Pixel Shuffle 操作
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # 调整输出通道

    def forward(self, a, b):
        # 对 a 进行卷积并 Pixel Shuffle
        a_up = self.conv_a(a)
        a_up = self.pixel_shuffle(a_up)  # 输出形状 (b, inter_channels, 2*h, 2*w)
        # 将 a 和 b 进行拼接融合
        rgb_mask = self.sa(b)
        output = a_up + a_up * rgb_mask
        # 调整输出通道

        return output


# class AdaptivePixelShuffle(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor = 2):
#         super(AdaptivePixelShuffle, self).__init__()
#         self.upscale_factor = upscale_factor
#         # 用于处理 a 的卷积
#         self.conv_a = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
#         # 用于融合 b 特征的卷积
#         self.fusion_conv = nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=1)
#         # Pixel Shuffle 操作
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         # 调整输出通道

#     def forward(self, a, b):
#         # 对 a 进行卷积并 Pixel Shuffle
#         a_up = self.conv_a(a)
#         a_up = self.pixel_shuffle(a_up)  # 输出形状 (b, inter_channels, 2*h, 2*w)
#         # 将 a 和 b 进行拼接融合
#         fusion = torch.cat([a_up, b], dim=1)  # 通道拼接
#         output = self.fusion_conv(fusion)  # 调整融合后的通道数
#         # 调整输出通道

#         return output

# class AdaUp(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(AdaUp, self).__init__()
#         self.adaptive_pooling1 = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
#         self.adaptive_pooling2 = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
#         self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积
#         self.trans_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 转置卷积

#     def forward(self, x ,rgb):
#         # 步骤1：自适应池化
#         L_R = self.adaptive_pooling1(rgb)
#         L_E = self.adaptive_pooling2(x)
#         # 步骤2：1x1卷积进行通道上的特征分布序列交互
#         L = L_R + L_E
#         L1 = self.conv1x1(L)
#         L = L.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)
#         L1 = L1.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)
#         # 步骤3：卷积核的逐元素乘法（广播）
#         W_f = self.trans_conv.weight *L * L1
        
#         # 步骤4：转置卷积进行上采样
#         out = F.conv_transpose2d(x, W_f, stride=self.trans_conv.stride, padding=self.trans_conv.padding, output_padding=self.trans_conv.output_padding)

#         # 步骤5：手动加上 bias
#         if self.trans_conv.bias is not None:
#             out += self.trans_conv.bias.view(1, -1, 1, 1)
        
#         return out

class AdaUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaUp, self).__init__()
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积
        self.trans_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 转置卷积

    def forward(self, x ):
        # 步骤1：自适应池化
        L = self.adaptive_pooling(x)

        # 步骤2：1x1卷积进行通道上的特征分布序列交互
        L = self.conv1x1(L)
        L = L.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)
        # 步骤3：卷积核的逐元素乘法（广播）
        W_f = self.trans_conv.weight *L 
        
        # 步骤4：转置卷积进行上采样
        out = F.conv_transpose2d(x, W_f, stride=self.trans_conv.stride, padding=self.trans_conv.padding, output_padding=self.trans_conv.output_padding)

        # 步骤5：手动加上 bias
        if self.trans_conv.bias is not None:
            out += self.trans_conv.bias.view(1, -1, 1, 1)
        
        return out

class AdaUp_SA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaUp_SA, self).__init__()
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        self.sa = Spatial_Attention()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积
        self.trans_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 转置卷积

    def forward(self, x ,rgb ):
        # 步骤1：自适应池化
        L = self.adaptive_pooling(x)

        # 步骤2：1x1卷积进行通道上的特征分布序列交互
        L = self.conv1x1(L)
        L = L.mean(dim=0, keepdim=True)  # (1, out_channels, 1, 1)
        # 步骤3：卷积核的逐元素乘法（广播）
        W_f = self.trans_conv.weight *L 
        
        # 步骤4：转置卷积进行上采样
        out = F.conv_transpose2d(x, W_f, stride=self.trans_conv.stride, padding=self.trans_conv.padding, output_padding=self.trans_conv.output_padding)

        # 步骤5：手动加上 bias
        if self.trans_conv.bias is not None:
            out += self.trans_conv.bias.view(1, -1, 1, 1)

        rgb_mask = self.sa(rgb)
        out = out + out * rgb_mask
        
        return out

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem_event = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.dark2_event = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        # self.dark3_conv = Conv(base_channels * 2, base_channels * 4, 3, 2, act=act)
        # self.dark3_CSP = CSPLayer(
        #         base_channels * 4,
        #         base_channels * 4,
        #         n=base_depth * 3,
        #         depthwise=depthwise,
        #         act=act,
        #     )
        self.dark3_event = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.dark4_event = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.dark5_event = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )


        hidden_dim1 = 48
        hidden_dim2 = 96
        hidden_dim3 = 192
        hidden_dim4 = 384
        hidden_dim5 = 768
    
        self.linear_transform_layer3 = LinearTransformLayer(num_channels=hidden_dim3)
        self.linear_transform_layer3_event = LinearTransformLayer(num_channels=hidden_dim3)
        self.linear_transform_layer4 = LinearTransformLayer(num_channels=hidden_dim4)
        self.linear_transform_layer4_event = LinearTransformLayer(num_channels=hidden_dim4)
        self.linear_transform_layer5 = LinearTransformLayer(num_channels=hidden_dim5)
        self.linear_transform_layer5_event = LinearTransformLayer(num_channels=hidden_dim5)
       
        self.mamba3 = SS2D(d_model=hidden_dim3)
        self.mamba4 = SS2D(d_model=hidden_dim4)
        self.mamba5 = SS2D(d_model=hidden_dim5)

        # self.upsample = Upsample(in_channels=hidden_dim3)
        self.upsample = AdaUp_SA(hidden_dim3, hidden_dim3)
        # self.upsample = AdaptivePixelShuffle(hidden_dim3 , hidden_dim3)
        # self.upsample = DeformableTransConv2d(hidden_dim3, hidden_dim3)


    def forward(self, x, sif):
        outputs = {}
        outputs_event = {}
        # print(x.shape)
        x = self.stem(x)
        # sa = self.sa(x)
        sif = self.stem_event(sif)

        outputs_event["stem"] = sif

        outputs["stem"] = x
        
        x = self.dark2(x)

        sif = self.dark2_event(sif)

        outputs_event["dark2"] = sif
        # x = self.nl2(x, sif)
        outputs["dark2"] = x

        x = self.dark3(x) # bug
        sif = self.dark3_event(sif)

        sif = self.upsample(sif,x)
        # print(sif.shape)
        # sif = self.upsample(sif)

        x = self.linear_transform_layer3(x)
        sif = self.linear_transform_layer3_event(sif)
        b, c, h, w = x.shape
        sif_split = sif.unsqueeze(4)  # (B, C, H, W, 1)
        x_split = x.unsqueeze(4)      # (B, C, H, W, 1)
      
        # print(sif_split.shape)
        merged = torch.cat([sif_split, x_split], dim=4)  # (B, C, H, W, 2)

        output = merged.contiguous().view(b, c, h, w * 2).permute(0,2,3,1)
        # print(output.shape)
        fusion = self.mamba3(output)
        output_reshaped = fusion.permute(0,3,1,2)
        output_reshaped = output_reshaped.view(b, c, h, w, 2)  # (B, C, H, W, 2)

        event_f = output_reshaped[..., 0]  # 提取出 event 特征 (B, C, H, W)
        rgb_f = output_reshaped[..., 1]    # 提取出 rgb 特征 (B, C, H, W)

        x = rgb_f  + x
        sif = event_f + sif

        outputs["dark3"] = x
        outputs_event["dark3"] = sif
        
        #########################################
        
        x = self.dark4(x)
        sif = self.dark4_event(sif)

        x = self.linear_transform_layer4(x)
        sif = self.linear_transform_layer4_event(sif)
        b, c, h, w = x.shape
        # print(b,c,h,w)
        sif_split = sif.unsqueeze(4)  # (B, C, H, W, 1)
        x_split = x.unsqueeze(4)      # (B, C, H, W, 1)
        merged = torch.cat([sif_split, x_split], dim=4)  # (B, C, H, W, 2)

        output = merged.contiguous().view(b, c, h, w * 2).permute(0,2,3,1)
        # print(output.shape)
        fusion = self.mamba4(output)
        output_reshaped = fusion.permute(0,3,1,2)
        output_reshaped = output_reshaped.view(b, c, h, w, 2)  # (B, C, H, W, 2)

        event_f = output_reshaped[..., 0]  # 提取出 event 特征 (B, C, H, W)
        rgb_f = output_reshaped[..., 1]    # 提取出 rgb 特征 (B, C, H, W)

        x = x + rgb_f
        sif =sif + event_f 

        outputs_event["dark4"] = sif
        outputs["dark4"] = x

        #########################################
        x = self.dark5(x)
        sif = self.dark5_event(sif)
        # outputs_event["dark5"] = sif
        x = self.linear_transform_layer5(x)
        sif = self.linear_transform_layer5_event(sif)
        b, c, h, w = x.shape
        # print(b,c,h,w)
        sif_split = sif.unsqueeze(4)  # (B, C, H, W, 1)
        x_split = x.unsqueeze(4)      # (B, C, H, W, 1)

        merged = torch.cat([sif_split, x_split], dim=4)  # (B, C, H, W, 2)
        output = merged.contiguous().view(b, c, h, w * 2).permute(0,2,3,1)
        # print(output.shape)
        fusion = self.mamba5(output)
        output_reshaped = fusion.permute(0,3,1,2)
        output_reshaped = output_reshaped.view(b, c, h, w, 2)  # (B, C, H, W, 2)

        event_f = output_reshaped[..., 0]  # 提取出 event 特征 (B, C, H, W)
        rgb_f = output_reshaped[..., 1]    # 提取出 rgb 特征 (B, C, H, W)

        x = rgb_f + x
        sif = event_f + sif
     
        outputs["dark5"] = x
        outputs_event["dark5"] = sif
        # print("outputs:", outputs)
        return {k: v for k, v in outputs.items() if k in self.out_features}, {k: v for k, v in outputs_event.items() if k in self.out_features}
