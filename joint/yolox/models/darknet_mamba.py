#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn
import math
import torch
import torch.nn.functional as F
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from .mamba_cross import CrossMamba
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

        self.ln = nn.LayerNorm(self.d_model)
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
        x = self.ln(x)
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


class AdaUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaUp, self).__init__()
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到1x1
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1卷积
        self.trans_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 转置卷积

    def forward(self, x):
        # 步骤1：自适应池化
        L = self.adaptive_pooling(x)
        
        # 步骤2：1x1卷积进行通道上的特征分布序列交互
        L1 = self.conv1x1(L)
        
        # 步骤3：卷积核的逐元素乘法（广播）
        W_f = self.trans_conv.weight *L * L1
        print(W_f.shape)
        # 步骤4：转置卷积进行上采样
        out = F.conv_transpose2d(x, W_f, stride=self.trans_conv.stride, padding=self.trans_conv.padding, output_padding=self.trans_conv.output_padding)

        # 步骤5：手动加上 bias
        if self.trans_conv.bias is not None:
            out += self.trans_conv.bias.view(1, -1, 1, 1)
        
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

        # kernel_size = 7
        # self.sa = SpatialAttention(kernel_size)
        # self.sa1 = SpatialAttention(kernel_size)
        # ###m
        # self.nl1 = _NonLocalBlockND_pool(48)
        # self.nl2 = _NonLocalBlockND_pool(96)

        hidden_dim1 = 48
        hidden_dim2 = 96
        hidden_dim3 = 192
        hidden_dim4 = 384
        hidden_dim5 = 768
        # self.cross_mamba1= CrossMamba(hidden_dim1)
        # self.cross_mamba2= CrossMamba(hidden_dim2)
        # self.cross_mamba3 = CrossMamba(hidden_dim3)
        # self.cross_mamba4 = CrossMamba(hidden_dim4)
        # self.cross_mamba5 = CrossMamba(hidden_dim5)

        self.linear_transform_layer3 = LinearTransformLayer(num_channels=hidden_dim3)
        self.linear_transform_layer3_event = LinearTransformLayer(num_channels=hidden_dim3)
        self.linear_transform_layer4 = LinearTransformLayer(num_channels=hidden_dim4)
        self.linear_transform_layer4_event = LinearTransformLayer(num_channels=hidden_dim4)
        self.linear_transform_layer5 = LinearTransformLayer(num_channels=hidden_dim5)
        self.linear_transform_layer5_event = LinearTransformLayer(num_channels=hidden_dim5)
       
        self.mamba3 = SS2D(d_model=hidden_dim3)
        self.mamba4 = SS2D(d_model=hidden_dim4)
        self.mamba5 = SS2D(d_model=hidden_dim5)
        ###l
        # self.nl1 = _NonLocalBlockND_pool(64)
        # self.nl2 = _NonLocalBlockND_pool(128)

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

        x = self.linear_transform_layer3(x)
        sif = self.linear_transform_layer3_event(sif)
        b, c, h, w = x.shape
        sif_split = sif.unsqueeze(4)  # (B, C, H, W, 1)
        x_split = x.unsqueeze(4)      # (B, C, H, W, 1)

        merged = torch.cat([sif_split, x_split], dim=4)  # (B, C, H, W, 2)

        output = merged.contiguous().view(b, c, h, w * 2).permute(0,2,3,1)
        # print(output.shape)
        fusion = self.mamba3(output)
        output_reshaped = fusion.permute(0,3,1,2)
        output_reshaped = output_reshaped.view(b, c, h, w, 2)  # (B, C, H, W, 2)

        event_f = output_reshaped[..., 0]  # 提取出 event 特征 (B, C, H, W)
        rgb_f = output_reshaped[..., 1]    # 提取出 rgb 特征 (B, C, H, W)

        x = rgb_f + x
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

        x = rgb_f + x
        sif = event_f + sif

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
