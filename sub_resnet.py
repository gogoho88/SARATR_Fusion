"""
# In[0]
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py 참조
https://github.com/heartcored98/Standalone-DeepLearning/blob/master/Lec6/Lab8_CIFAR_100_with_ResNet.ipynb 참조
"""
import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

"""
# In[0] Block Module 구현
"""
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding (stride 1에서는 크기 유지)"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3_1d(in_planes, out_planes, stride=1):
    """3x3 convolution with padding (stride 1에서는 크기 유지)"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1_1d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, base_width=64, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class Bottleneck(nn.Module):
    """
    - 50 layer 이상의 network에서는 2층의 basic block대신 3층의 bottleneck block 사용
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, base_width=64, downsample=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

"""
# In[0] Resnet network 구현
"""
class ResNet(nn.Module):

    def __init__(self, block, layers, flag_abs=False, num_classes=11, width_per_group=64, zero_init_residual=False):
        """
        - block: [BasicBlock, Bottleneck] 중 무엇 선택할지
        - layers (List[int]): 각 layer의 block 개수
        - width_per_group (int): 각 block의 채널 수 정수배 할 때 사용 예를들어 원래 block 내에서 64-64 연결을 64-64*with로 바꿈
        즉, 원래 inplanes 64 자체를 바꾸는 것은 아님
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.base_width = width_per_group

        if flag_abs==True:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)   # //2
        else:
            self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)   # //2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                 # //2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)                 # //2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)                 # //2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)                 # //2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                                     # (W,H)->1로 만듬
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 모듈들 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:         # identity mapping이 아닌 경우
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),    # identity mapping 대신 downsampling 및 channel 조정
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.base_width, downsample))    # layer 내 맨 처음 block만 downsample
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

"""
# In[0] Resnet architecture 정의
"""
def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model

# ResNet 계열
def resnet18(**kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs: Any) -> ResNet:
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs: Any) -> ResNet:
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)

# Wide ResNet 게열
def wide_resnet50_2(**kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], **kwargs)