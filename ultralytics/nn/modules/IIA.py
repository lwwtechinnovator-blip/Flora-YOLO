import torch
import torch.nn as nn

class IIA_YOLO(nn.Module):
    def __init__(self, c1, c2, kernel_size=7):

        super(IIA_YOLO, self).__init__()
        assert c1 == c2, f"IIA_YOLO is an enhancement module, input channels ({c1}) must equal output channels ({c2})."

        self.channel = c1
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.conv2 = nn.Conv1d(self.channel, self.channel, kernel_size=kernel_size, padding=padding,
                               groups=self.channel, bias=False)
        self.bn = nn.BatchNorm1d(self.channel)
        self.sigmoid = nn.Sigmoid()

    def _generate_attention_weight(self, x_permuted):

        B, D1, C, D2 = x_permuted.shape

        pooled_features = torch.cat(
            (torch.max(x_permuted, 1, keepdim=True)[0], torch.mean(x_permuted, 1, keepdim=True)),
            dim=1
        )

        pooled_features = pooled_features.view(B, 2, C, D2)

        weight = self.conv1(pooled_features).view(B, C, D2)
        weight = self.sigmoid(self.bn(self.conv2(weight)))

        weight = weight.view(B, 1, C, D2)

        return weight

    def forward(self, x):

        x_h_permuted = x.permute(0, 3, 1, 2).contiguous()
        weight_h = self._generate_attention_weight(x_h_permuted)
        x_h = (x_h_permuted * weight_h).permute(0, 2, 3, 1)

        x_w_permuted = x.permute(0, 2, 1, 3).contiguous()
        weight_w = self._generate_attention_weight(x_w_permuted)
        x_w = (x_w_permuted * weight_w).permute(0, 2, 1, 3)

        return x + x_h + x_w

if __name__ == '__main__':

    iia_module = IIA_YOLO(c1=64, c2=64)

    input_tensor = torch.rand(2, 64, 32, 32)

    output_tensor = iia_module(input_tensor)

    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output_tensor.size()}')

    assert input_tensor.size() == output_tensor.size(), "Module input and output sizes do not match!"

    print("\nModule test passed, input and output sizes remain consistent.")
