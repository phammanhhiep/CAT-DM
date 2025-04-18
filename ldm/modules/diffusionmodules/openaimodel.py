from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    # My Note: Original code
    # def _forward(self, x, emb):
    #     if self.updown:
    #         in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
    #         h = in_rest(x)
    #         h = self.h_upd(h)
    #         x = self.x_upd(x)
    #         h = in_conv(h)
    #     else:
    #         h = self.in_layers(x)
    #         emb_out = self.emb_layers(emb).type(h.dtype)
    #     while len(emb_out.shape) < len(h.shape):
    #         emb_out = emb_out[..., None]
    #     if self.use_scale_shift_norm:
    #         out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
    #         scale, shift = th.chunk(emb_out, 2, dim=1)
    #         h = out_norm(h) * (1 + scale) + shift
    #         h = out_rest(h)
    #     else:
    #         h = h + emb_out
    #         h = self.out_layers(h)
    #     return self.skip_connection(x) + h

    # My Note: fix to use 16-mixed precision
    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            with th.cuda.amp.autocast(enabled=False):
                h = self.in_layers(x.float())
                emb_out_fp32 = self.emb_layers(emb.float())
            emb_out = emb_out_fp32.to(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        with th.cuda.amp.autocast(enabled=False):
            skip_result_fp32 = self.skip_connection(x.float())
        final_skip_result = skip_result_fp32.to(h.dtype)
        return final_skip_result + h    


class My_ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, 4, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, 4, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    def __init__(self,
                image_size,                         # 32
                in_channels,                        # 9
                out_channels,                       # 4
                model_channels,                     # 320
                attention_resolutions,              # [ 4, 2, 1 ]
                num_res_blocks,                     # 2
                channel_mult=(1, 2, 4, 8),          # [ 1, 2, 4, 4 ]
                num_heads=-1,                       # 8
                use_spatial_transformer=False,      # True
                transformer_depth=1,                # 1
                context_dim=None,                   # 768
                use_checkpoint=False,               # True
                legacy=True,                        # False   
                add_conv_in_front_of_unet=False,    # False
                dropout=0,
                conv_resample=True,
                dims=2, 
                num_classes=None,  
                num_head_channels=-1,   
                num_heads_upsample=-1,    
                use_scale_shift_norm=False): 

        super().__init__()
        
        self.image_size = image_size                                # 32
        self.in_channels = in_channels                              # 9
        self.out_channels = out_channels                            # 4
        self.model_channels = model_channels                        # 320
        self.attention_resolutions = attention_resolutions          # [4,2,1]
        self.num_res_blocks = num_res_blocks                        # 2
        self.channel_mult = channel_mult                            # [1,2,4,4]
        num_heads_upsample = num_heads                              # 8
        self.use_checkpoint = use_checkpoint                        # True
        self.add_conv_in_front_of_unet=add_conv_in_front_of_unet    # False
        self.dropout = dropout                                      # 0
        self.conv_resample = conv_resample                          # True
        self.num_classes = num_classes                              # None
        self.num_heads = num_heads                                  # 8
        self.num_head_channels = num_head_channels                  # -1
        self.num_heads_upsample = num_heads_upsample                # -1
        self.dtype = th.float32


        # 时间编码器 320 -> 320*4 -> 320*4
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )


        # 一阶段 self.input_blocks
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        dims,           # 2
                        in_channels,    # 9
                        model_channels, # 320
                        kernel_size=3,  # 3
                        padding=1       # 1
                    )
                )
            ]
        )

        input_block_chans = [model_channels] # [320]
        ch = model_channels # 320
        ds = 1 # 1

        for level, mult in enumerate(channel_mult): # [0,1,2,3], [1,2,4,4]
            for _ in range(num_res_blocks): # 2
                layers = [
                    ResBlock(
                        ch,                                         # [1,1,1,2,2,4,4,4]*320
                        time_embed_dim,                             # 320*4
                        dropout,                                    # 0
                        out_channels=mult * model_channels,         # [1,1,2,2,4,4,4,4]*320
                        dims=dims,                                  # 2
                        use_checkpoint=use_checkpoint,              # True
                        use_scale_shift_norm=use_scale_shift_norm,  # False
                    )
                ]
                ch = mult * model_channels # [1,1,2,2,4,4,4,4]*320
                if ds in attention_resolutions: # 前6次小循环
                    dim_head = ch // num_heads  # [1,1,2,2,4,4]*40
                    layers.append(
                        SpatialTransformer(
                            ch,                         # [1,1,2,2,4,4]*320
                            num_heads,                  # 8
                            dim_head,                   # [1,1,2,2,4,4]*40
                            depth=transformer_depth,    # 1
                            context_dim=context_dim     # 768
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch) # [1,1,2,2,4,4,4,4]*320
            if level != len(channel_mult) - 1: # 前3次大循环
                out_ch = ch # [1,2,4]*320
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch,                     # [1,2,4]*320 
                            conv_resample,          # True
                            dims=dims,              # 2
                            out_channels=out_ch     # [1,2,4]*320
                        )
                    )
                )
                ch = out_ch # [1,2,4]*320
                input_block_chans.append(ch) # [1,2,4]*320
                ds *= 2 # 1 -> 2 -> 4 -> 8


        # 二阶段 self.middle_block
        dim_head = ch // num_heads # 1280 // 8 

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,                                         # 4*320
                time_embed_dim,                             # 320
                dropout,                                    # 0
                dims=dims,                                  # 2
                use_checkpoint=use_checkpoint,              # True
                use_scale_shift_norm=use_scale_shift_norm,  # False
            ),
            SpatialTransformer(
                ch,                         # 4*320
                num_heads,                  # 8
                dim_head,                   # 160
                depth=transformer_depth,    # 1
                context_dim=context_dim     # 768
            ),
            ResBlock(
                ch,                                         # 4*320
                time_embed_dim,                             # 320
                dropout,                                    # 0
                dims=dims,                                  # 2
                use_checkpoint=use_checkpoint,              # True
                use_scale_shift_norm=use_scale_shift_norm,  # False
            ),
        )


        # 三阶段 self.output_blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: # [3,2,1,0], [4,4,2,1]
            for i in range(num_res_blocks + 1): # 3
                ich = input_block_chans.pop() # [4,4, 4,4,4, 2,2,2, 1,1,1,  1]*320
                layers = [
                    ResBlock(
                        ch + ich,                                   # [4,4,4,4,4,4,4,2,2,2,1,1]*320+ich
                        time_embed_dim,                             # 320
                        dropout,                                    # 0
                        out_channels=model_channels*mult,           # [4,4,4,4,4,4,2,2,2,1,1,1]*320
                        dims=dims,                                  # 2
                        use_checkpoint=use_checkpoint,              # True
                        use_scale_shift_norm=use_scale_shift_norm,  # False
                    )
                ]
                ch = model_channels * mult # [4,4,4,4,4,4,2,2,2,1,1,1]*320
                if ds in attention_resolutions:  # 后三次大循环
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks: # 前三次大循环中每次最后的小循环
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # 四阶段 self.out
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, context=None):
        hs = []

        t_emb = timestep_embedding(timesteps, self.model_channels) # [N, 320]
        emb = self.time_embed(t_emb) # [N, 320*4]
        h = x.type(self.dtype) # 将 x 转换为 torch.float32

        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)

        return self.out(h) # [N,4,H,W]

