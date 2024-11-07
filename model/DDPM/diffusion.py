"""
Reference from https://github.com/ermongroup/ddim
"""
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint

from model.adapter_utils import Adapter


class Model(nn.Module):
    def __init__(
            self,
            ch,
            in_channels,
            out_ch,
            ch_mult,
            num_res_blocks,
            attn_resolutions,
            dropout,
            resamp_with_conv,
            model_type,
            img_size,
            num_timesteps,
            with_adapter=False,
            adapter_dim=16,
            adapter_patch_size=4,
            adapter_num_heads=4,
            adapter_qkv_bias=True,
            adapter_drop=0.1
    ):
        super().__init__()

        if model_type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = img_size
        self.in_channels = in_channels
        self.with_adapter = with_adapter

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = img_size
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         with_adapter=self.with_adapter,
                                         adapter_dim=adapter_dim,
                                         adapter_patch_size=adapter_patch_size,
                                         adapter_num_heads=adapter_num_heads,
                                         adapter_qkv_bias=adapter_qkv_bias,
                                         adapter_drop=adapter_drop))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in,
                                          with_adapter=self.with_adapter,
                                          adapter_dim=adapter_dim,
                                          adapter_patch_size=adapter_patch_size,
                                          adapter_num_heads=adapter_num_heads,
                                          adapter_qkv_bias=adapter_qkv_bias,
                                          adapter_drop=adapter_drop))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       with_adapter=self.with_adapter,
                                       adapter_dim=adapter_dim,
                                       adapter_patch_size=adapter_patch_size,
                                       adapter_num_heads=adapter_num_heads,
                                       adapter_qkv_bias=adapter_qkv_bias,
                                       adapter_drop=adapter_drop)
        self.mid.attn_1 = AttnBlock(block_in,
                                    with_adapter=self.with_adapter,
                                    adapter_dim=adapter_dim,
                                    adapter_patch_size=adapter_patch_size,
                                    adapter_num_heads=adapter_num_heads,
                                    adapter_qkv_bias=adapter_qkv_bias,
                                    adapter_drop=adapter_drop)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       with_adapter=self.with_adapter,
                                       adapter_dim=adapter_dim,
                                       adapter_patch_size=adapter_patch_size,
                                       adapter_num_heads=adapter_num_heads,
                                       adapter_qkv_bias=adapter_qkv_bias,
                                       adapter_drop=adapter_drop)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         with_adapter=self.with_adapter,
                                         adapter_dim=adapter_dim,
                                         adapter_patch_size=adapter_patch_size,
                                         adapter_num_heads=adapter_num_heads,
                                         adapter_qkv_bias=adapter_qkv_bias,
                                         adapter_drop=adapter_drop))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in,
                                          with_adapter=self.with_adapter,
                                          adapter_dim=adapter_dim,
                                          adapter_patch_size=adapter_patch_size,
                                          adapter_num_heads=adapter_num_heads,
                                          adapter_qkv_bias=adapter_qkv_bias,
                                          adapter_drop=adapter_drop))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        if self.with_adapter:
            self.end_adapter = Adapter(in_channels=block_in,
                                       out_channels=out_ch,
                                       temb_channels=None,
                                       embed_dim=adapter_dim,
                                       patch_size=adapter_patch_size,
                                       num_heads=adapter_num_heads,
                                       qkv_bias=adapter_qkv_bias,
                                       drop=adapter_drop)
        else:
            self.end_adapter = nn.Identity()

        if with_adapter:
            # freeze the base DDPM model when model with adapter
            for name, param in self.named_parameters():
                if "adapter" not in name.lower():
                    param.requires_grad = False

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.with_adapter:
            adapter_out = self.end_adapter(h, None)
        else:
            adapter_out = 0.0
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h + adapter_out


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout,
            temb_channels=512,
            with_adapter=False,
            adapter_dim=16,
            adapter_patch_size=4,
            adapter_num_heads=4,
            adapter_qkv_bias=True,
            adapter_drop=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.with_adapter = with_adapter

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

        self.init_weights()
        if self.with_adapter:
            self.adapter = Adapter(in_channels=in_channels,
                                   out_channels=out_channels,
                                   temb_channels=temb_channels,
                                   embed_dim=adapter_dim,
                                   patch_size=adapter_patch_size,
                                   num_heads=adapter_num_heads,
                                   qkv_bias=adapter_qkv_bias,
                                   drop=adapter_drop)
        else:
            self.adapter = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight, .0, .02)
                if m.bias is not None:
                    init.constant_(m.bias, .0)
            elif isinstance(m, nn.GroupNorm):
                init.constant_(m.bias, .0)
                init.constant_(m.weight, 1.0)

    def forward(self, x, temb):
        return checkpoint(self._forward, *(x, temb))

    def _forward(self, x, temb):
        h = x
        if self.with_adapter:
            adapter_out = self.adapter(x, temb)
        else:
            adapter_out = 0.0
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h + adapter_out


class AttnBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            with_adapter=False,
            adapter_dim=16,
            adapter_patch_size=4,
            adapter_num_heads=4,
            adapter_qkv_bias=True,
            adapter_drop=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.with_adapter = with_adapter

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        if self.with_adapter:
            self.adapter = Adapter(in_channels=in_channels,
                                   out_channels=in_channels,
                                   embed_dim=adapter_dim,
                                   patch_size=adapter_patch_size,
                                   num_heads=adapter_num_heads,
                                   qkv_bias=adapter_qkv_bias,
                                   drop=adapter_drop)
        else:
            self.adapter = nn.Identity()

    def forward(self, x):
        return checkpoint(self._forward, x)

    def _forward(self, x):
        h_ = x
        if self.with_adapter:
            adapter_out = self.adapter(x)
        else:
            adapter_out = 0.0
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        # q = q.reshape(b, c, h * w)
        # q = q.permute(0, 2, 1)  # b,hw,c
        # k = k.reshape(b, c, h * w)  # b,c,hw
        # w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c) ** (-0.5))
        # w_ = torch.nn.functional.softmax(w_, dim=2)
        #
        # # attend to values
        # v = v.reshape(b, c, h * w)
        # w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = torch.bmm(v, w_)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w).permute(0, 2, 1)  # b,hw,c
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # b,hw,c
        h_ = scaled_dot_product_attention(q, k, v)

        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_ + adapter_out


if __name__ == '__main__':
    num_timesteps = 1000
    cfg = {
        "ch": 128,
        "in_channels": 3,
        "out_ch": 3,
        "ch_mult": (1, 1, 2, 2,),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.1,
        "resamp_with_conv": True,
        "model_type": "fixedlarge",
        "img_size": 256,
        "num_timesteps": num_timesteps,
        "with_adapter": True}

    model = Model(**cfg).cuda()
    B = 2
    x = torch.rand(B, 3, 256, 256).cuda()

    t = torch.randint(low=0, high=num_timesteps, size=(B // 2 + 1,)).cuda()
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:B]

    with torch.no_grad():
        output = model(x, t)

    print(f"output: {output.shape}")
    params_requires_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_not_requires_grad = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"parameters requires grad size: {params_requires_grad}")
    print(f"parameters not requires grad size: {params_not_requires_grad}")
    print(f"train parameters rate {params_requires_grad / (params_requires_grad + params_not_requires_grad)}")
