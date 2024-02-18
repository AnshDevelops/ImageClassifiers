import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, interm_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=interm_channels, kernel_size=3, stride=stride,
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
        self.conv1 = nn.Conv2d(in_channels, out_channels=interm_channels, kernel_size=1, bias=False)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual: bool = False, grayscale: bool = False):
        super(ResNet, self).__init__()
        # handles both color and grayscale images
        input_channels = 1 if grayscale else 3
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, interm_channels=64, blocks=layers[0])
        self.layer2 = self._make_layer(block, interm_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, interm_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, interm_channels=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        # kaiming initialization for conv, constant initialization for bn
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch so that each residual block behaves like an identity.
        # Only use when extremely large batches / mini-batches
        # Credits: https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, interm_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != interm_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=interm_channels * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(num_features=interm_channels * block.expansion),
            )

        layers = [block(self.in_channels, interm_channels, stride, downsample)]
        self.in_channels = interm_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, interm_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x
