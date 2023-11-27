python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RCS(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RCS, self).__init__()

        assert in_channels % 2 == 0, "Input channels must be divisible by 2"
        half_channels = in_channels // 2

        self.conv1x1 = nn.Conv2d(half_channels, half_channels, kernel_size=1, stride=stride, padding=0, groups=2)
        self.bn1 = nn.BatchNorm2d(half_channels)

        self.conv3x3 = nn.Conv2d(half_channels, half_channels, kernel_size=3, stride=stride, padding=1, groups=2)
        self.bn2 = nn.BatchNorm2d(half_channels)

    def forward(self, x):
        c = x.shape[1] // 2
        x1 = x[:, :c, :, :]
        x2 = x[:, c:, :, :]

        out1 = x1

        out2 = F.relu(self.bn1(self.conv1x1(x2)))

        out3 = F.relu(self.bn2(self.conv3x3(x2)))

        out = torch.cat([out1, out2 + out3], dim=1)

        out = out[:, torch.randperm(out.shape[1]), :, :]

        return out
