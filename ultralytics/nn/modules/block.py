

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from ultralytics.nn.modules.transformer import TransformerBlock
from einops import rearrange
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",

)

class DFL(nn.Module):

    def __init__(self, c1: int = 16):

        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class Proto(nn.Module):

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):

        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class HGStem(nn.Module):

    def __init__(self, c1: int, cm: int, c2: int):

        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class HGBlock(nn.Module):

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):

        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class SPP(nn.Module):

    def __init__(self, c1: int, c2: int, k: Tuple[int, ...] = (5, 9, 13)):

        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class MixPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        if self.training:
            lambda_val = torch.sigmoid(self.weight)

            max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
            avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

            return lambda_val * max_pool + (1 - lambda_val) * avg_pool
        else:
            lambda_val = torch.sigmoid(self.weight.detach())
            max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
            avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
            return lambda_val * max_pool + (1 - lambda_val) * avg_pool

class LKAAttention(nn.Module):

    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, padding=dilation, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.act(attn)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class AtrousBlock(nn.Module):

    def __init__(self, c_in, c_out, rates=[6, 12]):
        super().__init__()
        num_branches = 1 + len(rates)

        branch_channels = c_out // num_branches
        remainder = c_out % num_branches

        self.conv_1x1 = Conv(c_in, branch_channels + remainder, k=1)
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                Conv(c_in, branch_channels, k=3, p=rate, d=rate)
            )

    def forward(self, x):
        branch_outputs = [self.conv_1x1(x)]
        for conv in self.atrous_convs:
            branch_outputs.append(conv(x))

        return torch.cat(branch_outputs, dim=1)

class LMSA_SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

        self.m = nn.Sequential(

            nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        )

    def forward(self, x):

        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class SPPF(nn.Module):

    def __init__(self, c1: int, c2: int, k: int = 5):

        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class C1(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1):

        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.cv1(x)
        return self.m(y) + y

class C2(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)

        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

def conv_relu_bn(in_channel, out_channel, dirate):

    return nn.Sequential(

        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),

        nn.BatchNorm2d(out_channel),

        nn.ReLU(inplace=True)
    )

class BAM(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):

        super(BAM, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)

        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)

        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)

        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')

        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')

        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')

        att = self.softmax(self.s_conv(att))
        return att

class Conv(nn.Module):
    def __init__(self, in_dim):

        super(Conv, self).__init__()

        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):

        for conv in self.convs:
            x = conv(x)
        return x

class DConv(nn.Module):
    def __init__(self, in_dim):

        super(DConv, self).__init__()

        dilation = [2, 4, 2]

        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):

        for dconv in self.dconvs:
            x = dconv(x)
        return x

class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):

        super(ConvAttention, self).__init__()

        self.conv = Conv(in_dim)

        self.dconv = DConv(in_dim)

        self.att = BAM(in_dim, in_feature, out_feature)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        q = self.conv(x)

        k = self.dconv(x)

        v = q + k

        att = self.att(x)

        out = torch.matmul(att, v)

        return self.gamma * out + v + x

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):

        super(FeedForward, self).__init__()

        self.conv = conv_relu_bn(in_dim, out_dim, 1)

        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.conv(x)

        x = self.x_conv(x)

        return x + out

class CLFT(nn.Module):
    def __init__(self, in_dim, out_dim, in_feature, out_feature):

        super(CLFT, self).__init__()

        self.attention = ConvAttention(in_dim, in_feature, out_feature)

        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):

        x = self.attention(x)

        out = self.feedforward(x)
        return out

class BottConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):

        super(BottConv, self).__init__()

        mid_channels = max(mid_channels, 2)

        assert in_channels == out_channels, "in_channels and out_channels must be equal for Sandglass block"
        assert stride == 1, "stride must be 1 for Sandglass block"

        self.depthwise_1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                     bias=False)

        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)

        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        self.depthwise_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=out_channels,
                                     bias=False)

    def forward(self, x):

        residual = x

        x = self.depthwise_1(x)

        x = self.pointwise_1(x)

        x = self.pointwise_2(x)

        x = self.depthwise_2(x)

        return x + residual

class Gated_Bottleneck_Convolution(nn.Module):

    def __init__(self, in_channels, norm_type='GN') -> None:

        super().__init__()

        self.proj = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)

        self.norm = nn.InstanceNorm3d(in_channels)

        if norm_type == 'GN':

            num_groups = max(in_channels // 16, 1)

            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups)

        self.nonliner = nn.SiLU()

        self.proj2 = BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1)

        self.norm2 = nn.InstanceNorm3d(in_channels)

        if norm_type == 'GN':

            num_groups2 = max(in_channels // 16, 1)

            self.norm2 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups2)

        self.nonliner2 = nn.SiLU()

        self.proj3 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)

        self.norm3 = nn.InstanceNorm3d(in_channels)

        if norm_type == 'GN':

            num_groups3 = max(in_channels // 16, 1)

            self.norm3 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups3)

        self.nonliner3 = nn.SiLU()

        self.proj4 = BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0)

        self.norm4 = nn.InstanceNorm3d(in_channels)

        if norm_type == 'GN':

            num_groups4 = max(in_channels // 16, 1)

            self.norm4 = nn.GroupNorm(num_channels=in_channels, num_groups=num_groups4)

        self.nonliner4 = nn.SiLU()

    def forward(self, x):

        x_residual = x

        x1_1 = self.proj(x)

        x1_1 = self.norm(x1_1)

        x1_1 = self.nonliner(x1_1)

        x1 = self.proj2(x1_1)

        x1 = self.norm2(x1)

        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)

        x2 = self.norm3(x2)

        x2 = self.nonliner3(x2)

        x = x1 * x2

        x = self.proj4(x)

        x = self.norm4(x)

        x = self.nonliner4(x)

        return x + x_residual

class C2f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):

        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(Gated_Bottleneck_Convolution(self.c)  for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:

        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
class DWConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()

        self.dw = Conv(c1, c1, k, s, g=c1, act=act)

        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))

class C3(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3x(C3):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))

class RepC3(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

class C3TR(C3):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

class C3Ghost(C3):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class GhostBottleneck(nn.Module):

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):

        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.conv(x) + self.shortcut(x)

import torch
import torch.nn as nn
import warnings

class Bottleneck(nn.Module):

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class ResNetBlock(nn.Module):

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):

        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))

class ResNetLayer(nn.Module):

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):

        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layer(x)

class MaxSigmoidAttnBlock(nn.Module):

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):

        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:

        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)

class C2fAttn(nn.Module):

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):

        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:

        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

class ImagePoolingAttn(nn.Module):

    def __init__(
        self, ec: int = 256, ch: Tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):

        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: List[torch.Tensor], text: torch.Tensor) -> torch.Tensor:

        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text

class ContrastiveHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

class BNContrastiveHead(nn.Module):

    def __init__(self, embed_dims: int):

        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)

        self.bias = nn.Parameter(torch.tensor([-10.0]))

        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):

        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias

class RepBottleneck(Bottleneck):

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):

        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, k[0], 1)

class RepCSP(C3):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):

        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

class RepNCSPELAN4(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):

        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

class ELAN1(RepNCSPELAN4):

    def __init__(self, c1: int, c2: int, c3: int, c4: int):

        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

class AConv(nn.Module):

    def __init__(self, c1: int, c2: int):

        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)

class HWD(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_bn_relu = nn.Sequential(

            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, ::]

        y_LH = yH[0][:, :, 1, ::]

        y_HH = yH[0][:, :, 2, ::]

        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

        x = self.conv_bn_relu(x)

        return x

class ADown(nn.Module):

    def __init__(self, c1: int, c2: int,*args, **kwargs):
        super().__init__()
        self.c = c2 // 2

        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class SPPELAN(nn.Module):

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):

        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))

class CBLinear(nn.Module):

    def __init__(self, c1: int, c2s: List[int], k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1):

        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        return self.conv(x).split(self.c2s, dim=1)

class _ConvBlock(nn.Sequential):

    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):

        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.LayerNorm([out_planes, h, w]),
            nn.ReLU(inplace=True)
        )

class TVConv(nn.Module):

    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=64,
                 TVConv_inter_layers=3,
                 TVConv_Bias=False,
                 h=3,
                 w=3,
                 **kwargs):
        super(TVConv, self).__init__()

        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k ** 2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))

        self.bias_layers = None

        out_chans = self.TVConv_k_square * self.channels

        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)

        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h,
                                               w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers,
                                                 h, w)

        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k - 1) // 2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):

        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):

        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w)

        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w)

        out = (weight * out).sum(dim=2)

        if self.bias_layers is not None:

            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out

class Gated_Bottleneck_Convolution_TV(nn.Module):

    def __init__(self, in_channels, norm_type='GN') -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm_type = norm_type

        self.proj = None
        self.proj2 = None
        self.proj3 = None
        self.proj4 = None

        self.norm = None
        self.norm2 = None
        self.norm3 = None
        self.norm4 = None

        self.nonliner = nn.ReLU()
        self.nonliner2 = nn.ReLU()
        self.nonliner3 = nn.ReLU()
        self.nonliner4 = nn.ReLU()

    def _initialize_layers(self, x):

        _, _, h, w = x.shape

        device = x.device

        self.proj = TVConv(self.in_channels, TVConv_k=3, h=h, w=w).to(device)
        self.proj2 = TVConv(self.in_channels, TVConv_k=3, h=h, w=w).to(device)
        self.proj3 = TVConv(self.in_channels, TVConv_k=1, h=h, w=w).to(device)
        self.proj4 = TVConv(self.in_channels, TVConv_k=1, h=h, w=w).to(device)

        if self.norm_type == 'GN':
            num_groups = max(self.in_channels // 16, 1)
            self.norm = nn.GroupNorm(num_channels=self.in_channels, num_groups=num_groups).to(device)
            self.norm2 = nn.GroupNorm(num_channels=self.in_channels, num_groups=num_groups).to(device)
            self.norm3 = nn.GroupNorm(num_channels=self.in_channels, num_groups=num_groups).to(device)
            self.norm4 = nn.GroupNorm(num_channels=self.in_channels, num_groups=num_groups).to(device)
        else:
            self.norm = nn.LayerNorm([self.in_channels, h, w]).to(device)
            self.norm2 = nn.LayerNorm([self.in_channels, h, w]).to(device)
            self.norm3 = nn.LayerNorm([self.in_channels, h, w]).to(device)
            self.norm4 = nn.LayerNorm([self.in_channels, h, w]).to(device)

    def forward(self, x):

        if self.proj is None:
            self._initialize_layers(x)

        x_residual = x

        x1_1 = self.proj(x)
        x1_1 = self.norm(x1_1)
        x1_1 = self.nonliner(x1_1)

        x1 = self.proj2(x1_1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 * x2

        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual

class CBFuse(nn.Module):

    def __init__(self, idx: List[int]):

        super().__init__()
        self.idx = idx

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:

        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)

class C3f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))

class C3k2(C2f):

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):

        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(

            C3k(self.c, self.c, 2, shortcut, g) if c3k else Gated_Bottleneck_Convolution(self.c) for _ in range(n)

        )

class EdgeAwareFeatureEnhancer(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAwareFeatureEnhancer, self).__init__()

        self.edge_extractor = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):

        edge_features = x - self.edge_extractor(x)

        edge_weights = self.weight_generator(edge_features)

        enhanced_features = edge_weights * x
        return enhanced_features

class C3k(C3):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):

        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)

        self.m=nn.Sequential(*(Gated_Bottleneck_Convolution(c_) for _ in range(n)))

class RepVGGDW(torch.nn.Module):

    def __init__(self, ed: int) -> None:

        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:

        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):

        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1

class CIB(nn.Module):

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(C2f):

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):

        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5, eps: float = 1e-4):

        super().__init__()

        self.eps = eps

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        qkv_out_dim = dim + (self.key_dim * num_heads) * 2
        self.qkv = Conv(dim, qkv_out_dim, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        if H > 1 and W > 1:

            spatial_var = torch.var(x, dim=(-2, -1), keepdim=True)

            attention_weight = torch.softmax(spatial_var, dim=1)

            x_att = x * attention_weight
        else:

            x_att = x

        N = H * W
        qkv = self.qkv(x_att)

        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        v_pe = self.pe(v.reshape(B, C, H, W))
        attended_v = (v @ attn.transpose(-2, -1)).view(B, C, H, W)

        output = attended_v + v_pe
        output = self.proj(output)

        return output

class ImprovedAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5, k_ratio: float = 0.8):

        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.k_ratio = k_ratio

        self.suppression_factor = nn.Parameter(torch.tensor(0.0))

        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        nh_kd = self.key_dim * num_heads
        qkv_dim = dim + nh_kd * 2

        self.qkv = Conv(dim, qkv_dim, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)

        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature

        num_tokens_to_keep = int(N * self.k_ratio)

        topk_indices = torch.topk(attn, k=num_tokens_to_keep, dim=-1, largest=True)[1]

        topk_mask = torch.zeros_like(attn, requires_grad=False)
        topk_mask.scatter_(-1, topk_indices, 1.)

        suppression_scale = torch.sigmoid(self.suppression_factor)

        weights = torch.where(topk_mask > 0, 1.0, suppression_scale)

        weighted_attn = attn * weights

        attn_weights = weighted_attn.softmax(dim=-1)

        output = (v @ attn_weights.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))

        output = self.proj(output)

        return output

import torch
import torch.nn as nn
import warnings
from .conv import Conv

def autopad(k, p=None):

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = avg_out + max_out
        return x * scale

class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, 7, padding=3, groups=1),
            nn.SiLU(),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        sum_out = torch.sum(x, dim=1, keepdim=True)
        pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        attention = self.conv(pool)
        return x * attention

class QGAF(nn.Module):

    def __init__(self, c1, c2, *args):
        super(QGAF, self).__init__()

        self.expected_c1 = c1

        self.ca = None
        self.sa = None
        self.q1_conv = None
        self.q2_conv = None
        self.q3_conv = None
        self.q4_conv = None
        self.gate = None

    def _initialize_modules(self, x):

        actual_c1 = x.shape[1]
        if actual_c1 != self.expected_c1:
            warnings.warn(
                f"QGAF.py 警告: 检测到通道不匹配。 "
                f"YAML配置的期望通道数为 {self.expected_c1}，但在运行时接收到的实际通道数为 {actual_c1}。 "
                f"正在动态调整模块以适应 {actual_c1} 通道。"
            )

        self.ca = ChannelAttention(actual_c1).to(x.device)
        self.sa = SpatialAttention().to(x.device)

        quadrant_channels = actual_c1
        self.q1_conv = Conv(quadrant_channels, quadrant_channels, k=1, g=quadrant_channels).to(x.device)
        self.q2_conv = Conv(quadrant_channels, quadrant_channels, k=3, g=quadrant_channels).to(x.device)
        self.q3_conv = Conv(quadrant_channels, quadrant_channels, k=5, g=quadrant_channels).to(x.device)
        self.q4_conv = Conv(quadrant_channels, quadrant_channels, k=7, g=quadrant_channels).to(x.device)

        self.gate = nn.Sequential(
            nn.Conv2d(actual_c1 * 2, 1, 1, bias=True),
            nn.Sigmoid()
        ).to(x.device)

    def forward(self, x):
        if self.ca is None:
            self._initialize_modules(x)

        x_global = self.sa(self.ca(x))

        x_h_split1, x_h_split2 = x.chunk(2, dim=2)
        x_q1, x_q2 = x_h_split1.chunk(2, dim=3)
        x_q3, x_q4 = x_h_split2.chunk(2, dim=3)

        q1_proc = self.q1_conv(x_q1)
        q2_proc = self.q2_conv(x_q2)
        q3_proc = self.q3_conv(x_q3)
        q4_proc = self.q4_conv(x_q4)

        top_half = torch.cat((q1_proc, q2_proc), dim=3)
        bottom_half = torch.cat((q3_proc, q4_proc), dim=3)
        x_local = torch.cat((top_half, bottom_half), dim=2)

        gate_weights = self.gate(torch.cat((x_global, x_local), dim=1))

        return x_global * gate_weights + x_local * (1 - gate_weights)

class PSABlock(nn.Module):

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:

        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class PSA(nn.Module):

    def __init__(self, c1: int, c2: int, e: float = 0.5):

        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

class ECA(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        return x * self.activaton(y)

class C2PSA(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):

        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        layers = [
            PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) if i % 2 == 0 else QGAF(c1, c1)

            for i in range(n)

        ]
        self.m = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class C2fPSA(C2f):

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):

        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)

        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))

class SCDown(nn.Module):

    def __init__(self, c1: int, c2: int, k: int, s: int):

        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.cv2(self.cv1(x))

class TorchVision(nn.Module):

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):

        import torchvision

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

class AAttn(nn.Module):

    def __init__(self, dim: int, num_heads: int, area: int = 1):

        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)

class ABlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):

        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):

        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.attn(x)
        return x + self.mlp(x)

class A2C2f(nn.Module):

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):

        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

class SwiGLUFFN(nn.Module):

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:

        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class Residual(nn.Module):

    def __init__(self, m: nn.Module) -> None:

        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)

        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.m(x)

class SAVPE(nn.Module):

    def __init__(self, ch: List[int], c3: int, embed: int):

        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: List[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:

        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
