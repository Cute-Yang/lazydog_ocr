from typing import List

import torch
from torch import nn
from torch.nn import functional as F

# 将我们的 1/4 的feature_map恢复到和输入一样的大小


class HeadNet(nn.Module):
    def __init__(self, in_channels: int, kernel_list: List[int] = [3, 2, 2], **kwargs):
        """
        Args:
            in_channels:int,the input channles of tensor
            kernel_list:List[int],we will construct a module list,
                and will expand the size of the input -> 4
        """
        super(HeadNet, self).__init__()
        assert (len(kernel_list) == 3), "the kernel_list must be 3!but get {}".format(len(kernel_list))
        assert in_channels % 4 == 0 and in_channels > 4, "get invalid in_channels {}".format(in_channels)
        # 输出同等大小的卷积
        if kernel_list[0] % 2 == 0:
            raise ValueError("the identity conv transform in pytorch must be has odd number,but get {}".format(kernel_list[0]))
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=kernel_list[0] // 2
        )
        self.conv_bn1 = nn.BatchNorm2d(
            num_features=in_channels // 4
        )
        # pytorch的batch norm层没有提供激活函数的参数
        self.bn_act1 = nn.ReLU()

        # 放大2倍
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2
        )
        self.conv_bn2 = nn.BatchNorm2d(num_features=in_channels // 4)
        self.bn_act2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.bn_act1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.bn_act2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


class DBHead(nn.Module):
    def __init__(self, in_channels: int, k: int = 50, **kwargs):
        """
        k:int,expand factor,the exprience value is 50
        """
        super(DBHead, self).__init__()
        self.k = k
        # 二值化
        self.binarize = HeadNet(in_channels=in_channels, **kwargs)
        # 阈值图
        self.thresh = HeadNet(in_channels=in_channels, **kwargs)

    def step_function(self, x, y):
        """
        对应公式 1 / (exp(-k * (x - y)))
        """
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x: torch.Tensor):
        #收缩标签需要用到的map
        #我们在后处理中使用收缩标签法来计算框
        shrink_maps = self.binarize(x)  # 让他自动学习二值化的参数
        if not self.training:
            return shrink_maps
        #阈值图
        threshold_maps = self.thresh(x)
        # db方法获取概率图 可微二值化
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        # 训练阶段输出三个预测图
        return torch.concat([shrink_maps, threshold_maps, binary_maps], dim=1)
