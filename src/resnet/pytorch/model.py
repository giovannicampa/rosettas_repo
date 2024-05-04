import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Source: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
"""


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super(ResidualBlock, self).__init__()

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2), nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + shortcut
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(
        self, in_channels, repeat, residual_block: nn.Module, use_bottleneck: bool = None, num_classes: int = 10
    ):
        super(ResNet, self).__init__()

        self.in_channels = in_channels

        if use_bottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv2_1", residual_block(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module("conv2_%d" % (i + 1,), residual_block(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv3_1", residual_block(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module("conv3_%d" % (i + 1,), residual_block(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv4_1", residual_block(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module("conv2_%d" % (i + 1,), residual_block(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module("conv5_1", residual_block(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module("conv3_%d" % (i + 1,), residual_block(filters[4], filters[4], downsample=False))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(filters[4], num_classes)

    def forward(self, x):

        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


def build_model_pt(num_classes: int = 10):

    return ResNet(
        in_channels=1, residual_block=ResidualBlock, repeat=[2, 2, 2, 2], use_bottleneck=False, num_classes=num_classes
    )


if __name__ == "__main__":
    build_model_pt(num_classes=10)
