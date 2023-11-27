python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RCS(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RCS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = F.relu(x + out)
        return out

class RCS_OSA(nn.Module):
    def __init__(self, in_channels, out_channels, num_modules, stride=1):
        super(RCS_OSA, self).__init__()

        assert out_channels % num_modules == 0, "Output channels must be divisible by the number of RCS modules"
        rcs_out_channels = out_channels // num_modules

        self.rcs_modules = nn.ModuleList()
        for _ in range(num_modules):
            self.rcs_modules.append(RCS(in_channels, rcs_out_channels, stride))
            in_channels = rcs_out_channels  # Output of one RCS module is the input to the next

        self.conv = nn.Conv2d(num_modules * rcs_out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        outputs = []
        for rcs_module in self.rcs_modules:
            x = rcs_module(x)
            outputs.append(x)

        # Feature aggregation
        agg = torch.cat(outputs, dim=1)
        out = self.conv(agg)

        return out
