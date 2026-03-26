import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.GroupNorm(1, out_planes),
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
                 TVConv_posi_map_size=8,
                 **kwargs):
        super(TVConv, self).__init__()
        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k ** 2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.bias_layers = None
        out_chans = self.TVConv_k_square * self.channels
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, TVConv_posi_map_size, TVConv_posi_map_size))
        nn.init.ones_(self.posi_map)
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers)
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k - 1) // 2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers):
        layers = [_ConvBlock(in_chans, inter_chans, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, bias=False))
        layers.append(nn.Conv2d(inter_chans, out_chans, kernel_size=3, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        posi_map = F.interpolate(self.posi_map, size=(h, w), mode='bilinear', align_corners=False)
        weight = self.weight_layers(posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, h, w)
        out = self.unfold(x).view(b, self.channels, self.TVConv_k_square, h, w)
        out = (weight * out).sum(dim=2)
        if self.bias_layers is not None:
            bias = self.bias_layers(posi_map)
            out = out + bias
        return out

class TVConv_Downsample_Wrapper(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, **kwargs):

        super().__init__()

        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=(k - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

        self.tv_conv = TVConv(channels=c2, TVConv_k=k, **kwargs)

    def forward(self, x):

        x_downsampled = self.act(self.bn(self.conv(x)))

        return self.tv_conv(x_downsampled)

if __name__ == "__main__":

    batch_size = 2
    input_channels = 64
    output_channels = 128
    input_h, input_w = 32, 32
    stride = 2

    model = TVConv_Downsample_Wrapper(
        c1=input_channels,
        c2=output_channels,
        k=3,
        s=stride,

        TVConv_inter_chans=32
    )
    model.eval()

    print(f"--- Test downsampling wrapper module ---")
    input_tensor = torch.rand(batch_size, input_channels, input_h, input_w)
    output = model(input_tensor)

    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')

    expected_h = (input_h + stride - 1) // stride
    expected_w = (input_w + stride - 1) // stride
    print(f'Expected output size: ({batch_size}, {output_channels}, {expected_h}, {expected_w})')
    assert output.shape == (batch_size, output_channels, expected_h, expected_w)
    print("Size assertion succeeded!")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total module parameters: {total_params / 1e6:.2f}M')
