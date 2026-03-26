import torch
import torch.nn as nn

from .conv import Conv
class CARAFE(nn.Module):
    def __init__(self, in_channels, k_enc=3, k_up=5, c_mid=64, scale=2):

        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = Conv(in_channels, c_mid)
        self.enc = Conv(c_mid, (scale * k_up) ** 2, k=k_enc, act=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)
        W = self.enc(W)
        W = self.pix_shf(W)
        W = torch.softmax(W, dim=1)

        X = self.upsmp(X)
        X = self.unfold(X)
        X = X.view(b, c, -1, h_, w_)

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])
        return X
if __name__ == '__main__':
    model = CARAFE(3, k_enc=5, k_up=5, c_mid=64)
    x=torch.randn(1, 3, 640, 640)
    y=model(x)
    print(x.shape)
    print(y.shape)