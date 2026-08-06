"""DLA-34 from the official UC Berkeley implementation.

Source: https://github.com/ucbdrive/dla
Copyright (c) 2018, Fisher Yu. BSD-3-Clause; see LICENSE.
"""

import torch
from torch import nn


WEIGHTS_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/dla-34/dla34-ba72cf86.pth"


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride,
            padding=dilation, bias=False, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1,
            padding=dilation, bias=False, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, residual=None):
        residual = x if residual is None else residual
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1,
            bias=False, padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *children):
        out = self.bn(self.conv(torch.cat(children, dim=1)))
        if self.residual:
            out = out + children[0]
        return self.relu(out)


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super().__init__()
        root_dim = 2 * out_channels if root_dim == 0 else root_dim
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        else:
            self.tree1 = Tree(
                levels - 1, block, in_channels, out_channels, stride,
                root_dim=0, root_kernel_size=root_kernel_size,
                dilation=dilation, root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1, block, out_channels, out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size, dilation=dilation,
                root_residual=root_residual,
            )
        self.levels = levels
        self.level_root = level_root
        self.downsample = nn.MaxPool2d(stride, stride=stride) if stride > 1 else None
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        children.append(x1)
        return self.tree2(x1, children=children)


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000):
        super().__init__()
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], BasicBlock, channels[1], channels[2], stride=2)
        self.level3 = Tree(levels[3], BasicBlock, channels[2], channels[3], stride=2, level_root=True)
        self.level4 = Tree(levels[4], BasicBlock, channels[3], channels[4], stride=2, level_root=True)
        self.level5 = Tree(levels[5], BasicBlock, channels[4], channels[5], stride=2, level_root=True)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Conv2d(channels[-1], num_classes, 1, bias=True)
        self._initialize_weights()

    @staticmethod
    def _make_conv_level(in_channels, out_channels, convs, stride=1):
        modules = []
        for index in range(convs):
            modules.extend([
                nn.Conv2d(
                    in_channels, out_channels, 3,
                    stride=stride if index == 0 else 1,
                    padding=1, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        return nn.Sequential(*modules)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.base_layer(x)
        for index in range(6):
            x = getattr(self, f"level{index}")(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x.flatten(1)


def create_model(pretrained=False, num_classes=1000):
    model = DLA(
        [1, 1, 1, 2, 2, 1],
        [16, 32, 64, 128, 256, 512],
        num_classes=1000 if pretrained else num_classes,
    )
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, map_location="cpu", check_hash=True,
        )
        model.load_state_dict(state_dict)
        if num_classes != 1000:
            model.fc = nn.Conv2d(512, num_classes, 1)
    return model
