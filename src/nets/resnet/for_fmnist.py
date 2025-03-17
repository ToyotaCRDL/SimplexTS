'''
We refered to the open source code https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py to implement the ResNet model.
We updated this code for Simplex Temperature Scaling.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nets import Branch
from src.nets import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, # for Fashion-MNIST
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) # for Fashion-MNIST
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.branch = Branch(512*block.expansion)
        self.calibration = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out) # for Fashion-MNIST
        out = out.view(out.size(0), -1)
        logit = self.linear(out)
        if not self.calibration:
            return logit
        
        temp = self.branch(out)
        return logit, temp



def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def test(num_classes):
    net = ResNet18(num_classes)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
