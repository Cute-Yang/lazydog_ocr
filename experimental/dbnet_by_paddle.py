# 使用paddle加载模型，并且转换成对应的权重

"""
firstly,build the model from paddle
"""
import math

import paddle
import paddle.nn.functional as F
from paddle import ParamAttr, nn


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    # 计算divisor的整数倍
    """
    他这个计算过程是这样的
    给定expression = int(v + divisor / 2) // divisor * divisor
    如果v % divisor的结果 小于 divisor余数的一半，那么 expression得到的结果就和 v // divisor * divisor 是一样的
    如果v % divisor的结果 大于 divisor余数的一般，那么expression得到的结果 就是 (v // divisor + 1) * divisor
    手动的四舍五入
    """
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super(MobileNetV3, self).__init__()

        self.disable_se = disable_se

        # 中间部分配置参数
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
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        # 为什么要用这些scale呢，是为了可以被8整除,就是为了增加或者减少我们输出的feature map的数量
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),  # 输出8个通道
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish')
        # 这里输出通道是8...

        # 第一个通道输出的channle是8 conv + bn -> 8 channels

        self.stages = []
        self.out_channels = []
        block_list = []
        i = 0
        # 这个make_divisible就是一个四舍五入的工具 scale是减少的大小
        # 他在这里做了一个通道适应的修改，昨天没有看见，ca
        # 这个make_divisible函数的作用就是将输入的数字，四舍五入成8的倍数，并且如果计算后小于原来的 0.9倍,就在加以哈
        # 将输入的通道数改为8
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == 'large' else 0
            # 如果stride == 2,就开始添加层,此时作为一个stage
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(
                    # 输入的通道数
                    in_channels=inplanes,
                    # 中间通道数
                    mid_channels=make_divisible(scale * exp),
                    # 输出的通道数
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            # 同样这里改变inplanes的值，使得下一层的输入channels和上一层的输出channel对应的上
            inplanes = make_divisible(scale * c)
            i += 1

        # 逐点卷积
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                # 输出60个通道
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                # 步长为1，自然就不需要补0了
                padding=0,
                groups=1,
                if_act=True,
                act='hardswish'))
        # 最后一个stage
        self.stages.append(nn.Sequential(*block_list))
        # 收集每个stage输出的channel的数值
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        # 将所有的stage串联起来
        for i, stage in enumerate(self.stages):
            self.add_sublayer(sublayer=stage, name="stage{}".format(i))
            # 只是为了能偶通过名字来访问
            # 原理类似于构建一个map std::map<string,Module>,如果名字重复了,可能会报警告,然后直接覆盖

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


# 组合了一个conv层和一个bn层
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        super(ConvBNLayer, self).__init__()
        #
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm(num_channels=out_channels, act=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


# 这个
class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        super(ResidualUnit, self).__init__()
        # 这个条件就是判断输入和输出的shape是否一样，如果一样，就做一个shortcut
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            # 补0,使得输出shape一样
            padding=int((kernel_size - 1) // 2),
            # 这个groups参数决定了把channels分成几个组
            # 这里他把每个channel当成了一组,所以conv的weight的形状是 (mid_channels,1,kernel_size,kernel_size)
            # 每一个参与实际运算的kernel的shape是(1,kernel_size,kernl_size) -> (1,kernel_size,kernel_size) 每一个channel产生也给数据
            # 所以每个channel数据只用到了一次
            # 指定group可以明显减少卷积的weight参数,bias参数则不受影响
            groups=mid_channels,
            if_act=True,
            act=act)
        # 注意力
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        return x


# 这个模块的作用就是给每个channel计算一个scale，然后让输入乘以这个scale
class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        # if the input_tensor is n c h w,the output tensor is n c 1 1
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        # 这两个卷积的作用是变换channles的数量,起到一个特征变换的作用

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        # 这个的输出维度也是 n c 1 1
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


# 这个层的作用是先做一个卷积,在做一个注意力
class RSELayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        # 使用这个的原因是为了保证所有的参数能够被正确的注册,使用python列表就会出现问题
        # 并且这个的使用方法和python的list完全相同，对外接口一样
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()

        for i in range(len(in_channels)):
            # 这个层的作用是改变channle,并且是逐点卷积
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            # 这个层的作用是把channel 变为 1 / 4,并且使用3x3的卷积核
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))

    # 我终于知道了这里为什么要求是32的倍数，因为这里要做特征融合
    def forward(self, x):
        print(len(x))
        # 每个stage不同的输出 MobilenetV3的有4个stage,，所以我们这里的输入是4个feature_map
        c2, c3, c4, c5 = x

        # 因为这里c5的chnnales对应最后一个层,这里将所有的in 的channle变成output channel
        in5 = self.ins_conv[3](c5)  # 1 / 32
        in4 = self.ins_conv[2](c4)  # 1 / 16
        in3 = self.ins_conv[1](c3)  # 1/ 8
        in2 = self.ins_conv[0](c2)  # 1 / 4

        # 上采样2倍,在这前的基础上 x 2,对in5上采样3次,shape 扩大 1 / 32 * 8,out 1 / 4
        # 1 / 32 * 2,out 1 / 16
        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        # 1 / 16 * 2,out 1 / 8
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        # 1 / 8 * 2,out 1 / 4
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        # 然后再把通道缩小为 原来的 1/4,因为后面要沿着通道叠加四次,所以这里先把每个通道缩小为1/4
        p5 = self.inp_conv[3](in5)  # 1 / 32
        p4 = self.inp_conv[2](out4)  # 1 / 16
        p3 = self.inp_conv[1](out3)  # 1 / 8
        p2 = self.inp_conv[0](out2)  # 1 / 4

        # 再将上述得到的几个中间结果上采样到 1/4,就可以将所有的特征拼接在一起了
        # 拼接的时候沿着第一个维度,也就是channle维度，产生多个channel
        # 1 / 32 * 8,out 1/4,
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        # 1/16 * 4,out 1/ 4
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        # 1/8 * 2,out 1/4
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)
        # 拼接起来,作为最后网络的输出特征向量
        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        print("paddle neck output:", fuse)

        return fuse


def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr

# 之前我们送进来的feature_map是原图的 1/4,经过两次反卷积操作，即可恢复到原图大小


class Head(nn.Layer):
    # 这里选用了kernel_size 是为了输出的feature的shape刚好扩充2倍,由此可见
    # 我们在用c++做推理时，输出的cpu的内存可以复用输入的，而输入的可以申请足够大
    # 的内存，从而实现只需要申请一次的效果
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()

        # 维持输入形状不变
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        # 扩大两倍,类似于一个中间层
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        # 再扩大两倍，输出层
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {'maps': y}


class PaddleDBNet(nn.Layer):
    def __init__(self):
        super(PaddleDBNet, self).__init__()
        in_channels = 3
        self.backbone = MobileNetV3(
            in_channels=in_channels,
            disable_se=True
        )
        feats_out_channels_list = self.backbone.out_channels
        self.neck = RSEFPN(
            in_channels=feats_out_channels_list,
            out_channels=96
        )
        neck_out_channels = self.neck.out_channels
        self.head = DBHead(in_channels=neck_out_channels)

    def forward(self, x):
        output = self.backbone(x)
        output = self.neck(output)
        output = self.head(output)
        return output
