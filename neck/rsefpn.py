from typing import List

import torch
from torch import nn
from torch.nn import functional as F

# 构建FPN层
"""
构建过程,取上面的4个stage的输出,
1.将每个stage输出的channel通过 1x1卷积变换到指定的out_channels
2.从倒数第二个stage开始,将当前的stage上采样2倍，然后和前一个stage相加得到out，依次进行下去，所以会得到三个out
3.将最后一个stage和上一个步骤得到的三个fpn层的channel数减为1/4，利用一个3x3的卷积核
4.将上述的stage都上采样到1/4,放大倍数为1,2,4,8
5.将上采样之后的几个stage沿着channel维度拼接，作为最后的输出
"""


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


class RSELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, shortcut: bool = True):
        super(RSELayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )

        self.se_block = SEModule(in_channels=out_channels, reduction=4)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        if self.shortcut:
            output = output + self.se_block(output)
        else:
            output = self.se_block(output)
        return output


class RSEFPN(nn.Module):
    def __init__(self, feature_map_channels_list: List[int], out_channels: int, shortcut: bool = True):
        """
        Args:
            feature_map_channels_list:List[int],this list store the each feature map out channel
            out_channels:int,the finnal out channels
            shortcut:bool,if True,we will plus the input and output...
        """
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        # 和sequential的区别就是不一定是严格按照顺序执行的，所以他们的输入输出维度不一定对的上
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        for i in range(len(feature_map_channels_list)):
            self.ins_conv.append(
                RSELayer(
                    in_channels=feature_map_channels_list[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    shortcut=True
                )
            )

            self.inp_conv.append(
                RSELayer(
                    in_channels=out_channels,
                    out_channels=out_channels // 4,
                    kernel_size=3,
                    shortcut=1
                )
            )

    def forward(self, x):
        # 每个stage的feature_map
        c2, c3, c4, c5 = x
        # 首先将channel变换成out_channel
        in2 = self.ins_conv[0](c2)
        in3 = self.ins_conv[1](c3)
        in4 = self.ins_conv[2](c4)
        in5 = self.ins_conv[3](c5)

        # 然后进行2倍上采样并且相加
        out4 = in4 + F.upsample(in5, scale_factor=2, mode="nearest")
        out3 = in3 + F.upsample(out4, scale_factor=2, mode="nearest")
        out2 = in2 + F.upsample(out3, scale_factor=2, mode="nearest")

        p2 = self.inp_conv[0](out2)
        p3 = self.inp_conv[1](out3)
        p4 = self.inp_conv[2](out4)
        p5 = self.inp_conv[3](in5)

        # upsample
        p3 = F.upsample(p3, scale_factor=2, mode="nearest")
        p4 = F.upsample(p4, scale_factor=4, mode="nearest")
        p5 = F.upsample(p5, scale_factor=8, mode="nearest")
        # 沿着channel维度拼接
        return torch.concat([p2, p3, p4, p5], dim=1)
