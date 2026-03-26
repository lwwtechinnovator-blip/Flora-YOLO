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

        self.shared_mlp = nn.Sequential(

            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),

            nn.GELU(),

            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.shared_mlp(self.avg_pool(x))

        max_out = self.shared_mlp(self.max_pool(x))

        scale = self.sigmoid(avg_out + max_out)

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
                f"QGAF.py Warning: Channel mismatch detected. "
                f"YAML configured expected channels: {self.expected_c1}, but actual runtime channels: {actual_c1}. "
                f"Dynamically adjusting modules to fit {actual_c1} channels."
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
