import torch
import torch.nn as nn
import math

__all__ = ['OctaveConv', 'Conv_BN', 'Conv_BN_ACT']

class OctaveConv(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                alpha_in=0.5,
                alpha_out=0.5,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False):
        super(OctaveConv, self).__init__()
        '''
        H2H, H2L, L2H, L2L octave convolution 연산 수행하는 커널
        args
            in_channels : input channels
            out_channels: output channels
            ...
            alpha_in :    input alpha
            alpha_out:    output alpha
            ...
            groups   :    group conv할때 쓰는거인듯
            ...
        '''
        self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        
        # assert (stride == 1 or stride == 2), "stride should be 1 or 2"
        self.stride     = stride
        # groups == in_channels : depthwise convolution
        self.is_dw      = groups == in_channels 
        self.alpha_in   = alpha_in
        self.alpha_out  = alpha_out
        
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)
        
    def forward(self, x):
        '''
        input : 1) (x_h, x_l) /or/ 2) (x_h, None)
        -> 1)번 case의 경우는 그대로 진행하면 됨.
        -> 2)번 case의 경우는 x_l을 None으로 따로 지정해 줘야함

        downsampling : stride가 2이면 size 줄어야 하므로 self.downsample 해줌. 
        아니면 안해줌 ; first block case
        -> misalignment 때문에 stride 2 conv 보다 downsample + stride 1 conv 해줌
        '''
        x_h, x_l = x if type(x) is tuple else (x, None)

        # stride == 2 : transition phase -> 이미지 크기를 줄여줘야 함
        # downsample(avgpool) 한 다음에 conv(stride==1) 함
        # 근데 H path에서 transition phase에서도 이렇게 해줘야 하나?
        # avgpool+conv는 애초에 upsample에서의 misalign을 방지하기 위함이 아니었나? 잘 모르겠네
        x_h   = self.downsample(x_h) if self.stride == 2 else x_h 
        # print(x_h.shape)
        # x_h2h
        x_h2h = self.conv_h2h(x_h)
        # x_h2l : transition phase : downsample이 두번 됨.
        # center of patch는 계속 유지됨.
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None 

        if x_l is not None:
            # transition phase : stride 2 conv 대신 downsample + stride 1 conv
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            # x_l2l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None

            if self.is_dw:
                return x_h2h, x_l2l
            else :
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h   = x_l2h + x_h2h
                # x_h2l, x_l2l 모두 존재할 때 => x_l 존재
                x_l   = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else :
            # x_l is None : network 시작부분
            return x_h2h, x_h2l

class Conv_BN(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                alpha_in,
                alpha_out,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        '''
        conv-bn module
        conv의 output이 두개일 경우가 대부분이라 다루기 힘들기 때문에, bn까지 같이 붙여준다.
        '''
        self.conv = OctaveConv(in_channels, 
                            out_channels, 
                            kernel_size, 
                            alpha_in, 
                            alpha_out, 
                            stride, 
                            padding, 
                            dilation,
                            groups,
                            bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h      = self.bn_h(x_h)
        x_l      = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l

class Conv_BN_ACT(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                alpha_in,
                alpha_out,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        '''
        conv-bn-relu
        '''
        self.conv = OctaveConv(in_channels, 
                            out_channels, 
                            kernel_size, 
                            alpha_in, 
                            alpha_out, 
                            stride, 
                            padding, 
                            dilation,
                            groups,
                            bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act  = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h      = self.act(self.bn_h(x_h))
        x_l      = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l