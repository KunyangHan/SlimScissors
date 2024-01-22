from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(ResNet):
    """
    ResNetEncoder inherits from torchvision's official ResNet. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    global average pooling layer and the fully connected layer that was originally
    used for classification. The forward method  additionally returns the feature
    maps at all resolutions for decoder's use.
    """
    
    layers = {
        'resnet18': [2, 2, 2, 2],
        'resnet50':  [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
    }
    
    def __init__(self, in_channels, variant='resnet101', norm_layer=None):
        rswd = [False, False, True]
        # rswd = [False, False, False] if variant in ['resnet18'] else [False, False, True]
        block = BasicBlock if variant in ['resnet18'] else Bottleneck
        super().__init__(
            block=block,
            layers=self.layers[variant],
            replace_stride_with_dilation=rswd,
            norm_layer=norm_layer)
        
        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            
        # Delete fully-connected layer
        del self.avgpool
        del self.fc
    
    def forward(self, x):
        x0 = x  # 1/1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x  # 1/4
        x = self.layer2(x)
        x3 = x  # 1/8
        # GCA
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x  # 1/16
        # import pdb;pdb.set_trace()
        return x4, x3, x2, x1, x0
