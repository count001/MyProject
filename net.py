import torch
import torch.nn as nn
import torch.nn.functional as F
from go import *


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual_connection = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_connection:
            residual = self.residual_connection(residual)

        out += residual
        out = self.relu(out)
        return out

class PolicyNetwork(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(PolicyNetwork, self).__init__()
        self.initial_conv = nn.Conv2d(15, 32, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Additional convolutional layers before the residual blocks
        self.extra_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.extra_bn1 = nn.BatchNorm2d(64)
        self.extra_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.extra_bn2 = nn.BatchNorm2d(32)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(32, 32) for _ in range(num_residual_blocks)]
        )

        # Additional convolutional layers after the residual blocks
        self.post_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.post_bn1 = nn.BatchNorm2d(64)
        self.post_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.post_bn2 = nn.BatchNorm2d(32)

        self.conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        blank = x[:, 0]
        x = x.float()
        x = self.relu(self.initial_bn(self.initial_conv(x)))

        # Pass through additional pre-residual block conv layers
        x = self.relu(self.extra_bn1(self.extra_conv1(x)))
        x = self.relu(self.extra_bn2(self.extra_conv2(x)))

        x = self.residual_blocks(x)
        # Pass through additional post-residual block conv layers
        x = self.relu(self.post_bn1(self.post_conv1(x)))
        x = self.relu(self.post_bn2(self.post_conv2(x)))

        x = self.conv(x)
        x = self.flatten(x)
        x = torch.cat((x * blank.view(-1, 19 * 19), torch.ones((len(x), 1)).to(x.device) * 1e-50), dim=1)
        return x

class PlayoutNetwork(nn.Module):
    def __init__(self):
        super(PlayoutNetwork, self).__init__()
        self.conv1 = nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(19 * 19, 19 * 19 + 1)

    def forward(self, x):
        blank = x[:, 0]
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1, 19 * 19)
        x = self.linear(x)
        x = torch.cat((x[:, :-1] * blank.view(-1, 19 * 19), x[:, -1:]), dim=1)
        x = F.log_softmax(x, dim=1)
        return x


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(2 * 19 * 19, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # 是否需要 relu？
        x = self.conv5(x)
        x = x.view(-1, 2 * 19 * 19)
        x = self.linear(x)
        x = x.view(-1)
        x = torch.sigmoid(x)
        return x
