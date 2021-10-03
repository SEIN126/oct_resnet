'''
OctResnet(cvpr 19')
CIFAR implementation

implementation detail

첫 block : 
    - 첫 conv에서 h2h, h2l만 실행 (alpha_in = 0, alpha_out = alpha)
    - 이후 h2h, h2l, l2h, l2l 모두 실행

마지막 block :
    - 첫 conv에서 h2h, l2h만 실행 (alpha_in = alpha, alpha_out = 0)
    - 이후 h2h만 실행
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .octconv import *

__all__ = ['oct_resnet10', 'oct_resnet18', 
           'oct_resnet26', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152', 'oct_resnet200']

class BasicBlock(nn.Module):
    '''
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
        in_channel    : input channels
        out_channel   : output channels
        downsample    : if input's channel # != output's channel #, you need downsample
        alpha_in      : alpha value for input's channel
        alpha_out     : alpha value for output's channel
        first(bool)   : First Block or Not 
                        -> if first : a_in : 0, a_out = alpha
        output(bool)  : Last Block or Not
                        -> if last : a_in : alpha, a_out = 0
    '''
    expansion = 1
    def __init__(self, 
                 ops, 
                 in_channel, 
                 out_channel, 
                 stride=1, 
                 downsample=None,
                 alpha_in=0.5, 
                 alpha_out=0.5, 
                 norm_layer=None, 
                 first=False, 
                 output=False):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.downsample = downsample

        self.conv1 = Conv_BN_ACT(in_channel, 
                                 out_channel,
                                 kernel_size=3, 
                                 stride=stride, 
                                 padding=1,
                                 alpha_in=alpha_in if not first else 0,
                                 alpha_out=alpha_out,
                                 norm_layer=norm_layer)
        self.conv2 = Conv_BN(out_channel, 
                             out_channel,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             norm_layer=norm_layer,
                             alpha_in=0 if output else alpha_in,
                             alpha_out=0 if output else alpha_out)
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        '''
        x : 1) (x_h, x_l) /or/ 2) x
        '''
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        if self.downsample is not None:
            # transition phase
            identity_h, identity_l = self.downsample(x)

        x_h = x_h + identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l

class BasicBlockLast(nn.Module):
    expansion = 1

    def __init__(self, ops, in_channel, out_channel, stride=1, downsample=None,
                alpha_in=0.5, alpha_out=0.5, norm_layer=None, output=False):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.downsample = downsample

        self.conv1 = Conv_BN_ACT(in_channel, 
                                 out_channel,
                                 kernel_size=3, 
                                 stride=stride, 
                                 padding=1,
                                 alpha_in=alpha_in,
                                 alpha_out=alpha_out,
                                 norm_layer=norm_layer)
        self.conv2 = Conv_BN(out_channel, 
                             out_channel,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             norm_layer=norm_layer,
                             alpha_in=0 if output else alpha_in,
                             alpha_out=0 if output else alpha_out)
        self.relu = nn.ReLU()
        self.stride = stride

    def forward(self, x):
        '''
        x : 1) (x_h, x_l) /or/ 2) x
        '''
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        # print(x_h.shape, x_l.shape)
        x_h, x_l = self.conv2((x_h, x_l))

        if self.downsample is not None:
            # transition phase
            identity_h, identity_l = self.downsample(x)
        
        x_h = x_h + identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l

class Bottleneck(nn.Module):
    '''
    Args:
        ops(torch.nn) : quantization method. if nn : no quantization
        in_channel    : input channels
        out_channel   : output channels
        downsample    : if input's channel # != output's channel #, you need downsample
        alpha_in      : alpha value for input's channel
        alpha_out     : alpha value for output's channel
        first(bool)   : First Block or Not 
                        -> if first : a_in : 0, a_out = alpha
        output(bool)  : Last Block or Not
                        -> if last : a_in : alpha, a_out = 0
    '''
    expansion = 4
    def __init__(self,
                 ops,
                 in_channel,
                 out_channel,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 alpha_in=0.5,
                 alpha_out=0.5,
                 norm_layer=None,
                 first=False,
                 output=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channel * (base_width / 64.)) * groups

        self.conv1 = Conv_BN_ACT(in_channel, 
                                 out_channel,
                                 kernel_size=1,
                                 alpha_in=alpha_in if not first else 0,
                                 alpha_out=alpha_out,
                                 norm_layer=norm_layer)
        self.conv2 = Conv_BN_ACT(width,
                                 width,
                                 kernel_size=3,
                                 stride=stride,
                                 padding=1,
                                 groups=groups,
                                 norm_layer=norm_layer,
                                 alpha_in=0 if output else alpha_in,
                                 alpha_out=0 if output else alpha_out)
        self.conv3 = Conv_BN(width, 
                             out_channel * self.expansion,
                             kernel_size=1,
                             norm_layer=norm_layer,
                             alpha_in=0 if output else alpha_in,
                             alpha_out=0 if output else alpha_out)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride=stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        x_h, x_l = self.conv3((x_h, x_l))

        if self.downsample is not None:
            identity_h, identity_l = self.downsample(x)
        
        x_h = x_h + identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l


class OctResNet(nn.Module):
    def __init__(self, ops, block, num_blocks, num_classes=10, norm_layer=None, alpha=0.5):
        super(OctResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()

        self.layer1 = self._make_layer(ops, block, 64, num_blocks[0], stride=1, alpha_in=alpha, alpha_out=alpha, first=True) # num_blocks = 2
        self.layer2 = self._make_layer(ops, block, 128, num_blocks[1], stride=2, alpha_in=alpha, alpha_out=alpha) # num_blocks = 2
        self.layer3 = self._make_layer(ops, block, 256, num_blocks[2], stride=2, alpha_in=alpha, alpha_out=alpha) # num_blocks = 2
        self.layer4 = self._make_layer(ops, block, 512, num_blocks[3], stride=2, alpha_in=alpha, alpha_out=0, output=True) # num_blocks = 2
        
        self.linear = ops.Linear(512*block.expansion, num_classes)

    def _make_layer(self, 
                    ops,
                    block, 
                    out_channel, 
                    num_blocks, 
                    stride, 
                    alpha_in, 
                    alpha_out, 
                    norm_layer=None, 
                    first=False,
                    output=False):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        downsample = None

        if first or stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                Conv_BN(self.in_channel, out_channel * block.expansion, 
                        kernel_size=1, stride=stride, padding=0, alpha_in=alpha_in if not first else 0, alpha_out=alpha_out)
            ) 
        
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        
        # block 시작부분 -> transition phase에만 stride가 2인 downsample layer(shortcut) 필요
        # blcok의 시작부분에만 downsample 필요
        layers.append(block(ops, self.in_channel, out_channel, stride, downsample,
                            alpha_in=alpha_in, alpha_out=alpha_out, norm_layer=norm_layer, first=first, output=output))
        self.in_channel = out_channel * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(ops, self.in_channel, out_channel, norm_layer=norm_layer,
                                alpha_in=0 if output else alpha_in, alpha_out=0 if output else alpha_out, output=output))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))
        x_h, x_l = self.layer4((x_h, x_l))

        x = F.avg_pool2d(x_h, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def oct_resnet10(ops, **kwargs):
    """Constructs a Octave ResNet-10 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def oct_resnet18(ops, **kwargs):
    """Constructs a Octave ResNet-10 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def oct_resnet26(ops, **kwargs):
    """Constructs a Octave ResNet-26 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, Bottleneck, [2, 2, 2, 2], **kwargs)
    return model

def oct_resnet50(ops, **kwargs):
    """Constructs a Octave ResNet-50 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def oct_resnet101(ops, **kwargs):
    """Constructs a Octave ResNet-101 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def oct_resnet152(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-152 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def oct_resnet200(pretrained=False, **kwargs):
    """Constructs a Octave ResNet-200 model.
    Args:
        ops(torch.nn) : quantization method. if torch.nn : no quantization
    """
    model = OctResNet(ops, Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    from torchinfo import summary
    model = oct_resnet18(nn, num_classes=100, alpha=0.25)
    out = model(torch.randn(64,3,32,32))
    print(out.shape)
    # summary(model, input_size=(64, 224, 224), device='cpu')
    for n,m in model.named_modules():
        if isinstance(m, OctaveConv):
            print(n, m.alpha_in, m.alpha_out)