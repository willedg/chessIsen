# wp2/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)

class AlphaChessNet(nn.Module):
    def __init__(self, in_planes=19, channels=128, n_blocks=12, action_size=4672):
        super().__init__()
        self.conv_in = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False) 
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = self.relu(self.bn_in(self.conv_in(x)))
        out = self.blocks(out)

        # Policy
        p = self.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        logits = self.policy_fc(p)

        # Value
        v = self.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return logits, v.squeeze(1)
