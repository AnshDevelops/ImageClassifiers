import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, interm_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=interm_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=interm_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=interm_channels, out_channels=interm_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=interm_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, interm_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=interm_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=interm_channels)

        # 3x3 conv
        self.conv2 = nn.Conv2d(in_channels=interm_channels, out_channels=interm_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=interm_channels)

        # 1x1 conv
        self.conv3 = nn.Conv2d(in_channels=interm_channels, out_channels=interm_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(interm_channels * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
