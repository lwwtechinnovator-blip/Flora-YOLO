import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

from .conv import Conv

class HeightWidthDiagonalFeatureProcessor(nn.Module):
    def __init__(self, c1, c2):

        super().__init__()

        assert c1 == c2, f"Input channels ({c1}) and output channels ({c2}) must be equal."

        self.dwt = DWTForward(J=1, mode='zero', wave='haar')

        self.feature_fuser = Conv(c1 * 4, c1, 1)

        self.upsample = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)

    def forward(self, x):

        residual = x

        low, high_list = self.dwt(x)

        h_detail = high_list[0][:, :, 0, :, :]
        v_detail = high_list[0][:, :, 1, :, :]
        d_detail = high_list[0][:, :, 2, :, :]

        merged_features = torch.cat([low, h_detail, v_detail, d_detail], dim=1)

        fused = self.feature_fuser(merged_features)

        reconstructed_enhancement = self.upsample(fused)

        output = residual + reconstructed_enhancement

        return output

if __name__ == '__main__':

    enhancer_module = HeightWidthDiagonalFeatureProcessor(c1=64, c2=64)

    input_tensor = torch.rand(2, 64, 32, 32)

    output_tensor = enhancer_module(input_tensor)

    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output_tensor.size()}')

    assert input_tensor.size() == output_tensor.size(), "Module input and output sizes do not match!"

    print("\nModule test passed, input and output sizes remain consistent.")
