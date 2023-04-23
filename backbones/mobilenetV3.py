from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

# to make sure the output v can be divided by v,and the new v will be round by the half plus of the divisor
# if the divisor % 2 == 0,if the v % divisor >= divisor // 2,then equal ceil function,else equal floor function
# if the divisor % 2 == 1,if the v % divisor > divisor // 2, then equal ceil function,else equal floor function


def make_divisible(v: int, divisor: int = 8, min_value: int = 8):
    new_v = max(min_value, int((v + divisor // 2)) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List, Tuple],
                 stride: Union[int, List, Tuple],
                 padding: int = 0,
                 groups: int = 1,
                 use_activation: bool = True,
                 activation: str = None):
        super(ConvBNLayer, self).__init__()
        self.use_activation = use_activation
        self.activation_name = activation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            # 一定要paddle框架的代码对应起来
            bias=False
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_activation:
            if self.activation_name == "relu":
                x = F.relu(x)
            elif self.activation_name == "hardswish":
                x = F.hardswish(x)
            else:
                raise ValueError("we only support the activation with ['relu','hardswish'],but get {}".format(self.activation_name))
        return x


class ResidualUnit(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List, Tuple],
                 stride: Union[int, List, Tuple],
                 use_se: bool = False,
                 activation: str = None):
        super(ResidualUnit, self).__init__()
        self.use_se = use_se
        # 只有输入和输出的shape完全相同,才能使用shortcut
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)

        # 逐点卷积
        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            use_activation=True,
            activation=activation
        )

        # 非逐点卷积,并且降采样
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            # 所以这里最好选用奇数的卷积核
            padding=(kernel_size - 1) // 2,
            groups=mid_channels,
            use_activation=True,
            activation=activation
        )

        # 注意力模块，就是在每个channel上乘一个data
        if self.use_se:
            self.mid_se = SEModule(in_channels=mid_channels)
        # 恢复成原来的channel
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.expand_conv(x)
        output = self.bottleneck_conv(output)
        if self.use_se:
            output = self.mid_se(output)
        output = self.linear_conv(output)
        if self.use_shortcut:
            output = torch.add(x, output)
        return output


class SEModule(nn.Module):
    def __init__(self, in_channels: int = None, reduction: int = 4):
        super(SEModule, self).__init__()
        # a shape transform n x c  x h x w -> n x c x 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        if in_channels < reduction:
            raise ValueError("the in_channels must be greater than reduction,but get {} and {}!".format(
                in_channels, reduction
            ))
        # 因为上一步中已经搞了一个池化操作了
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.avg_pool(x)
        output = self.conv1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = F.hardsigmoid(output)
        return x * output


class MobileNetV3(nn.Module):
    def __init__(self, in_channels: int = 3, model_name="large", scale: float = 0.5, disable_se: bool = False):
        super(MobileNetV3, self).__init__()
        self.disable_se = disable_se
        if model_name == "large":
            cfg = [
                # k, exp, out,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 2],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', 2],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', 2],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', 2],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise ValueError("the model_name only support 'large' and 'small',but get {}".format(model_name))

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        if scale not in supported_scale:
            raise ValueError("the scale only support '{}',but get {}".format(supported_scale, scale))

        inplanes = 16
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(scale * inplanes),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            use_activation=True,
            activation="hardswish"
        )

        self.stages = []
        self.out_channels = []
        block_list = []
        # update the inplanes
        inplanes = make_divisible(inplanes * scale)
        start_indices = 2 if model_name == "large" else 0
        for i, (kernel_size, expand_channels, out_channels, use_se, activation, stride) in enumerate(cfg):
            use_se = use_se and (not self.disable_se)
            # 因为small的网络结构一来就开始降采样
            if stride == 2 and i > start_indices:
                self.out_channels.append(inplanes)
                self.stages.append(
                    nn.Sequential(*block_list)
                )
                block_list = []
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * expand_channels),
                    out_channels=make_divisible(out_channels * scale),
                    kernel_size=kernel_size,
                    stride=stride,
                    use_se=use_se,
                    activation=activation
                )
            )
            inplanes = make_divisible(scale * out_channels)

        # the next stage
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                use_activation=True,
                activation="hardswish"
            )
        )
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

        for i, stage in enumerate(self.stages):
            self.add_module("stage{}".format(i), module=stage)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list
    
    def get_feature_map_channels_list(self):
        return self.out_channels
