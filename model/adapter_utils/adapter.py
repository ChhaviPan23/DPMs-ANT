import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn import init
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_output = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # attn_output = scaled_dot_product_attention(q, k, v, dropout_p=self.drop).transpose(1, 2).reshape(B, N, C)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Adapter(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 temb_channels=None,
                 embed_dim=16,
                 patch_size=4,
                 num_heads=4,
                 qkv_bias=True,
                 drop=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.temb_channels = temb_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_pooling = nn.AvgPool2d(patch_size, stride=patch_size)
        self.down_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        # self.down_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        if in_channels % 2 == 0:
            self.norm1 = nn.GroupNorm(num_groups=int(in_channels / 2), num_channels=in_channels, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.BatchNorm2d(in_channels, eps=1e-6)
        self.up_conv = nn.Conv2d(embed_dim, out_channels or in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=int(embed_dim / 4), num_channels=embed_dim, eps=1e-6, affine=True)

        if temb_channels != None:
            self.temb_proj = nn.Linear(temb_channels, embed_dim)

        self.attention_block = AttentionBlock(dim=embed_dim,
                                              num_heads=num_heads,
                                              qkv_bias=qkv_bias,
                                              drop=drop,
                                              attn_drop=drop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight, .0, .0002)
                if m.bias is not None:
                    init.constant_(m.bias, .0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, .0)
                init.constant_(m.weight, 1.0)

    def nonlinearity(self, x):
        # swish
        return x * torch.sigmoid(x)

    def forward(self, x, temb=None):
        # return checkpoint(self._forward, *(x, temb))
        return self._forward(x, temb)

    def _forward(self, x, temb=None):
        B, C, H, W = x.shape
        ph = int(H / self.patch_size)
        pw = int(W / self.patch_size)

        assert H // self.patch_size != 0 or W // self.patch_size != 0, f"H:{H}, patch_size:{self.patch_size} Do not support that"

        # Down sampling
        x = self.down_pooling(x)  # B C Ph Pw
        x = self.norm1(x)
        x = self.down_conv(x).flatten(2).transpose(1, 2)  # B Ph*Pw EC

        if temb != None:
            temb = self.temb_proj(self.nonlinearity(temb)).unsqueeze(dim=1)
            x = x + temb

        # Self-attention + MLP layers
        x = self.attention_block(x)

        # Up sampling
        x = x.reshape(B, int(ph * pw), self.embed_dim).transpose(1, 2)
        x = x.reshape(B, self.embed_dim, ph, pw)  # B EC Ph Pw
        x = torch.nn.functional.interpolate(x, scale_factor=self.patch_size, mode="nearest")  # B EC H W
        x = self.norm2(x)
        x = self.up_conv(x)  # B C H W
        return x
