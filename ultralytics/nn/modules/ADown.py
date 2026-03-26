import torch
import torch.nn as nn
import warnings
from pytorch_wavelets import DWTForward
from conv import  Conv
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

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = HWD(c1 // 2,self.c)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

if __name__ == '__main__':

    batch_size = 4
    in_channels = 64
    out_channels = 128
    height = 32
    width = 32

    input_tensor = torch.randn(batch_size, in_channels, height, width)
    print(f"Original input shape: {input_tensor.shape}\n")

    adown_layer = ADown(c1=in_channels, c2=out_channels)

    output_tensor = adown_layer(input_tensor)

    print(f"Module final output shape: {output_tensor.shape}")

    print("\n--- Detailed step shape changes ---")
    x = input_tensor
    print(f"1. Input x: {x.shape}")

    x_pooled = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
    print(f"2. After avg pool: {x_pooled.shape}")

    x1_chunk, x2_chunk = x_pooled.chunk(2, 1)
    print(f"3. After chunk x1: {x1_chunk.shape}, x2: {x2_chunk.shape}")

    x1_conv = adown_layer.cv1(x1_chunk)
    print(f"4. Branch1 (after conv) x1: {x1_conv.shape}")

    x2_pooled = torch.nn.functional.max_pool2d(x2_chunk, 3, 2, 1)
    print(f"5. Branch2 (after max pool) x2: {x2_pooled.shape}")

    x2_conv = adown_layer.cv2(x2_pooled)
    print(f"6. Branch2 (after conv) x2: {x2_conv.shape}")

    final_output = torch.cat((x1_conv, x2_conv), 1)
    print(f"7. After concat final output: {final_output.shape}")