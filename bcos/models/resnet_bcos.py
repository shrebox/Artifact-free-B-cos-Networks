"""
B-cos ResNet/ResNeXt/Wide-ResNet models

Modified from https://github.com/pytorch/vision/blob/0504df5ddf9431909130e7788faf05446bb8a2/torchvision/models/resnet.py

CIFAR10 modifications from
https://github.com/chenyaofo/pytorch-cifar-models/blob/e9482ebc665084761ad9c84d36c83cbb82/pytorch_cifar_models/resnet.py
"""

import math
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

from bcos.common import BcosUtilMixin

# from torchvision.ops import StochasticDepth
from bcos.modules import BcosConv2d, BcosLinear, LogitLayer, StochasticDepth, norms
from bcos.modules.common import BcosSequential

"""
__all__ = [
    "BcosResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    # cifar
    "cifar10_resnet20",
    "cifar10_resnet32",
    "cifar10_resnet44",
    "cifar10_resnet56",
    "cifar10_resnet110",
    "cifar10_resnext29_8x64d",
    "cifar10_resnext29_16x64d",
    "cifar10_resnext29_32x4d",
    "cifar10_resnext29_16x8d",
    "cifar10_resnext47_16x8d",
    "cifar10_resnext47_32x4d",
    "cifar10_resnext65_16x8d",
    "cifar10_resnext65_32x4d",
    "cifar10_resnext101_16x8d",
    "cifar10_resnext101_32x4d",
]
"""


# DEFAULT_NORM_LAYER = norms.NoBias(norms.DetachablePositionNorm2d)
DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)  # bnu-linear
DEFAULT_CONV_LAYER = BcosConv2d

## New ResNet Model Wrapper with mlp conv and linear, avgpool, logit layer and BN
# class ResnetModelWrapper(nn.Module):
#     def __init__(
#         self,
#         backbone: nn.Module,
#         in_channels: int,
#         num_classes: int,
#         *,
#         use_mlp: bool = False,
#         mlp_hidden_dim: Optional[int] = None,
#         mlp_num_layers: int = 1,
#         use_conv_head: bool = True,
#         with_avg_pool: bool = True,
#         with_last_bn: bool = True,
#         with_last_bn_affine: bool = True,
#         logit_layer: bool = True,
#         logit_bias: Optional[float] = None,
#         logit_temperature: Optional[float] = None,
#     ):
#         super().__init__()
#         self.backbone = backbone
#         self.with_avg_pool = with_avg_pool
#         self.use_mlp = use_mlp
#         self.use_conv_head = use_conv_head

#         if use_mlp:
#             assert mlp_hidden_dim is not None, "MLP hidden dim required if using MLP"

#             if use_conv_head:
#                 self.fc0 = BcosConv2d(in_channels, mlp_hidden_dim, kernel_size=1)
#                 self.bn0 = norms.NoBias(norms.BatchNormUncentered2d)(mlp_hidden_dim)
#             else:
#                 self.fc0 = BcosLinear(in_channels, mlp_hidden_dim)
#                 self.bn0 = norms.NoBias(norms.BatchNormUncentered1d)(mlp_hidden_dim)

#             self.fc_layers = nn.ModuleList()
#             self.bn_layers = nn.ModuleList()

#             for i in range(1, mlp_num_layers):
#                 out_ch = num_classes if i == mlp_num_layers - 1 else mlp_hidden_dim
#                 if use_conv_head:
#                     self.fc_layers.append(BcosConv2d(mlp_hidden_dim, out_ch, kernel_size=1))
#                     if i != mlp_num_layers - 1 or with_last_bn:
#                         self.bn_layers.append(
#                             norms.NoBias(norms.BatchNormUncentered2d)(
#                                 out_ch, affine=with_last_bn_affine
#                             )
#                         )
#                     else:
#                         self.bn_layers.append(nn.Identity())
#                 else:
#                     self.fc_layers.append(BcosLinear(mlp_hidden_dim, out_ch))
#                     if i != mlp_num_layers - 1 or with_last_bn:
#                         self.bn_layers.append(
#                             norms.NoBias(norms.BatchNormUncentered1d)(
#                                 out_ch, affine=with_last_bn_affine
#                             )
#                         )
#                     else:
#                         self.bn_layers.append(nn.Identity())

#         else:
#             if use_conv_head:
#                 self.fc = BcosConv2d(in_channels, num_classes, kernel_size=1, bias=False)
#             else:
#                 self.fc = BcosLinear(in_channels, num_classes)

#         if with_avg_pool:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         self.logit_layer = (
#             LogitLayer(
#                 logit_temperature=logit_temperature,
#                 logit_bias=logit_bias or -math.log(num_classes - 1),
#             )
#             if logit_layer
#             else nn.Identity()
#         )

#     def forward(self, x):
#         x = self.backbone(x)

#         if self.use_mlp:
#             x = self.fc0(x)
#             x = self.bn0(x)
#             for fc, bn in zip(self.fc_layers, self.bn_layers):
#                 x = fc(x)
#                 x = bn(x)
#         else:
#             x = self.fc(x)

#         if self.with_avg_pool:
#             x = self.avgpool(x)

#         x = x.flatten(1)
#         x = self.logit_layer(x)
#         return x


## Updated version with MLP support
class ResnetModelWrapper(BcosSequential):
    def __init__(
        self,
        backbone,
        in_channels,
        num_classes,
        use_mlp=False,
        mlp_hidden_dim=None,
        mlp_num_layers=2,
        logit_bias=None,
        logit_temperature=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_mlp = use_mlp

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if use_mlp:
            assert (
                mlp_hidden_dim is not None
            ), "Need `mlp_hidden_dim` if use_mlp is True"
            self.fc = BcosLinearHead(
                in_channels, mlp_hidden_dim, num_classes, mlp_num_layers
            )
        else:
            self.fc = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

        self.logit_layer = LogitLayer(
            logit_temperature=logit_temperature,
            logit_bias=logit_bias or -math.log(num_classes - 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.logit_layer(x)
        return x


# class ResnetModelWrapper(nn.Module):
#     def __init__(
#         self,
#         backbone,
#         in_channels,
#         num_classes,
#         logit_bias=None,
#         logit_temperature=None,
#     ):
#         super(ResnetModelWrapper, self).__init__()
#         # for ResNet50: in_channels = 2048, num_classes = 1000
#         self.backbone = backbone
#         self.in_channels = in_channels
#         self.num_classes = num_classes

#         # set adaptive pool layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Conv2d(
#             self.in_channels, self.num_classes, kernel_size=1, bias=False
#         )

#         self.logit_layer = LogitLayer(
#             logit_temperature=logit_temperature,
#             logit_bias=logit_bias or -math.log(num_classes - 1),
#         )

#     def forward(self, x):

#         x = self.backbone(x)
#         x = self.fc(x)
#         x = self.avgpool(x)
#         x = x.flatten(1)
#         x = self.logit_layer(x)

#         return x


class ResnetWithBcosLinearModelWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        in_channels,
        hid_channels,
        num_classes,
        num_layers,
        logit_bias=None,
        logit_temperature=None,
    ):
        super(ResnetWithBcosLinearModelWrapper, self).__init__()
        # for ResNet50: in_channels = 2048, num_classes = 1000
        self.backbone = backbone
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        # set adaptive pool layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # logit layer
        self.logit_layer = LogitLayer(
            logit_temperature=logit_temperature,
            logit_bias=logit_bias or -math.log(num_classes - 1),
        )
        self.head = BcosLinearHead(in_channels, hid_channels, num_classes, num_layers)

    def forward(self, x):

        x = self.backbone(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.logit_layer(x)

        return x


# this is a BcosLinear MLP head
# note the MLP head uses BN layers
class BcosLinearHead(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        num_classes,
        num_layers,
        with_last_bn=False,
        with_last_bn_affine=False,
    ):
        super(BcosLinearHead, self).__init__()

        if num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        if num_layers == 1:
            assert (
                hid_channels == num_classes
            ), "Since num_layers == 1, hid_channels should be \
                    equal to num_classes"
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.fc0 = BcosConv2d(in_channels, hid_channels, kernel_size=1)
        self.bn0 = norms.NoBias(norms.BatchNormUncentered2d)(hid_channels)

        # for further layers
        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = num_classes if i == num_layers - 1 else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f"fc{i}", BcosConv2d(hid_channels, this_channels, kernel_size=1)
                )
                self.add_module(
                    f"bn{i}", norms.NoBias(norms.BatchNormUncentered2d)(this_channels)
                )
                self.bn_names.append(f"bn{i}")
            else:
                self.add_module(
                    f"fc{i}", BcosConv2d(hid_channels, this_channels, kernel_size=1)
                )
                if with_last_bn:
                    self.add_module(
                        f"bn{i}",
                        norms.NoBias(norms.BatchNormUncentered2d)(
                            this_channels, affine=with_last_bn_affine
                        ),
                    )
                    self.bn_names.append(f"bn{i}")
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f"fc{i}")

        if num_layers == 1:
            self.bn0 = nn.Identity()

    def forward(self, pre_logits):
        """The forward process."""
        # The final classification head.
        pre_logits = self.fc0(pre_logits)
        pre_logits = self.bn0(pre_logits)

        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            pre_logits = fc(pre_logits)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                pre_logits = bn(pre_logits)

        return pre_logits


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
):
    """3x3 convolution with padding"""
    return conv_layer(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
):
    """1x1 convolution"""
    return conv_layer(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        # act_layer: Callable[..., nn.Module] = None,
        stochastic_depth_prob: float = 0.0,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.Identity
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(
            inplanes,
            planes,
            stride,
            conv_layer=conv_layer,
        )
        self.bn1 = norm_layer(planes)
        # self.act = act_layer(inplace=True)
        self.conv2 = conv3x3(
            planes,
            planes,
            conv_layer=conv_layer,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob, "row")
            if stochastic_depth_prob
            else None
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.act(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        # act_layer: Callable[..., nn.Module] = None,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(
            inplanes,
            width,
            conv_layer=conv_layer,
        )
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width,
            width,
            stride,
            groups,
            dilation,
            conv_layer=conv_layer,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            width,
            planes * self.expansion,
            conv_layer=conv_layer,
        )
        self.bn3 = norm_layer(planes * self.expansion)
        # self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.stochastic_depth = (
            StochasticDepth(stochastic_depth_prob, "row")
            if stochastic_depth_prob
            else None
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stochastic_depth is not None:
            out = self.stochastic_depth(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # out = self.act(out)

        return out


class BcosResNet(BcosUtilMixin, nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
        layers: List[int] = None,
        # num_classes: int = 1000,
        in_chans: int = 6,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = DEFAULT_NORM_LAYER,
        conv_layer: Callable[..., nn.Module] = DEFAULT_CONV_LAYER,
        # act_layer: Callable[..., nn.Module] = None,
        inplanes: int = 64,
        small_inputs: bool = False,
        stochastic_depth_prob: float = 0.0,
        logit_bias: Optional[float] = None,
        logit_temperature: Optional[float] = None,
        frozen_stages=-1,
        **kwargs: Any,  # ignore rest
    ):
        super().__init__()

        if kwargs:
            print("The following args passed to model will be ignored", kwargs)

        if norm_layer is None:
            norm_layer = nn.Identity
        self._norm_layer = norm_layer
        self._conv_layer = conv_layer
        # self._act_layer = act_layer

        # frozen_stages
        self.frozen_stages = frozen_stages

        self.inplanes = inplanes
        self.dilation = 1
        n = len(layers)  # number of stages
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False] * (n - 1)
        if len(replace_stride_with_dilation) != n - 1:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a {n - 1}-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        if small_inputs:
            self.conv1 = conv3x3(
                in_chans,
                self.inplanes,
                conv_layer=conv_layer,
            )
            self.pool = None
        else:
            self.conv1 = conv_layer(
                in_chans,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
            )
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = norm_layer(self.inplanes)
        # self.act = act_layer(inplace=True)

        self.__total_num_blocks = sum(layers)
        self.__num_blocks = 0
        self.layer1 = self._make_layer(
            block,
            inplanes,
            layers[0],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.layer2 = self._make_layer(
            block,
            inplanes * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.layer3 = self._make_layer(
            block,
            inplanes * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        try:
            self.layer4 = self._make_layer(
                block,
                inplanes * 8,
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
                stochastic_depth_prob=stochastic_depth_prob,
            )
            last_ch = inplanes * 8
        except IndexError:
            self.layer4 = None
            last_ch = inplanes * 4

        self.num_features = last_ch * block.expansion

        """
        # removing this as this is a Backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes
        self.fc = conv_layer(
            self.num_features,
            self.num_classes,
            kernel_size=1,
        )
        self.logit_layer = LogitLayer(
            logit_temperature=logit_temperature,
            logit_bias=logit_bias or -math.log(num_classes - 1),
        )
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        self._freeze_stages()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        stochastic_depth_prob: float = 0.0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        conv_layer = self._conv_layer
        # act_layer = self._act_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride,
                    conv_layer=conv_layer,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer=norm_layer,
                conv_layer=conv_layer,
                # act_layer=act_layer,
                stochastic_depth_prob=stochastic_depth_prob
                * self.__num_blocks
                / (self.__total_num_blocks - 1),
            )
        )
        self.__num_blocks += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    conv_layer=conv_layer,
                    # act_layer=act_layer,
                    stochastic_depth_prob=stochastic_depth_prob
                    * self.__num_blocks
                    / (self.__total_num_blocks - 1),
                )
            )
            self.__num_blocks += 1

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        # redundant
        return

    def forward_features(self, x):
        B, C, H, W = x.shape
        if C != 6:
            # since bcos requires [r,g,b,1-r,1-g,1-b]
            x = torch.cat([x, 1 - x], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.act(x)
        if self.pool is not None:
            x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        """
        x = self.fc(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.logit_layer(x)
        """

        return x

    def train(self, mode=True):
        super(BcosResNet, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ""):
        """
        see: https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/resnet.py
        """
        return

    '''
    def get_classifier(self) -> nn.Module:
        """Returns the classifier part of the model. Note this comes before global pooling."""
        return self.fc

    def get_feature_extractor(self) -> nn.Module:
        """Returns the feature extractor part of the model. Without global pooling."""
        modules = [
            self.conv1,
            self.bn1,
            # self.act,
        ]
        if self.pool is not None:
            modules += [self.pool]
        modules += [
            self.layer1,
            self.layer2,
            self.layer3,
        ]
        if self.layer4 is not None:
            modules += [self.layer4]
        return nn.Sequential(*modules)
    '''


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool = False,
    progress: bool = True,
    inplanes: int = 64,
    **kwargs: Any,
) -> BcosResNet:
    model = BcosResNet(block, layers, inplanes=inplanes, **kwargs)
    if pretrained:
        raise ValueError(
            "If you want to load pretrained weights, then please use the entrypoints in "
            "bcos.pretrained or bcos.model.pretrained instead."
        )
    return model


# ---------------------
# ResNets for ImageNet
# ---------------------
def bcosresnet18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def bcosresnet34(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def bcosresnet50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def bcosresnet101(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def bcosresnet152(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def bcosresnext50_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def bcosresnext101_32x8d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )


def wide_bcosresnet50_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_bcosresnet101_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> BcosResNet:
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )


"""

# ---------------------
# ResNets for CIFAR-10
# ---------------------
def _update_if_not_present(key, value, d):
    if key not in d:
        d[key] = value


def _update_default_cifar(kwargs) -> None:
    _update_if_not_present("num_classes", 10, kwargs)
    _update_if_not_present("small_inputs", True, kwargs)


def cifar10_resnet20(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet20",
        BasicBlock,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet32(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet32",
        BasicBlock,
        [5] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet44(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet44",
        BasicBlock,
        [7] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet56(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet56",
        BasicBlock,
        [9] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnet110(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnet110",
        BasicBlock,
        [18] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


# --------------------
# ResNeXts
# --------------------
# These are model configs specified in the ResNeXt Paper https://arxiv.org/pdf/1611.05431.pdf


def cifar10_resnext29_8x64d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 8
    kwargs["width_per_group"] = 64
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_8x64d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=64,
        **kwargs,
    )


def cifar10_resnext29_16x64d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 64
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_16x64d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=64,
        **kwargs,
    )


# The model configs specified in the ResNeXt Paper
# are very large. We use smaller ones here.
# So instead of the 8x64d or 16x64d settings (with [64, 128, 256] widths)
# we have 32x4d and 16x8d settings ([16, 32, 64]).
# first 32x4d


def cifar10_resnext29_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_32x4d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext47_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext47_32x4d",
        Bottleneck,
        [5] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext65_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext65_32x4d",
        Bottleneck,
        [7] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext101_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext101_32x4d",
        Bottleneck,
        [11] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


# next with 16x8d


def cifar10_resnext29_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext29_32x4d",
        Bottleneck,
        [3] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext47_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext47_32x4d",
        Bottleneck,
        [5] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext65_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext65_32x4d",
        Bottleneck,
        [7] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )


def cifar10_resnext101_16x8d(
    pretrained: bool = False, progress: bool = True, **kwargs
) -> BcosResNet:
    kwargs["groups"] = 16
    kwargs["width_per_group"] = 8
    _update_default_cifar(kwargs)
    return _resnet(
        "cifar10_resnext101_32x4d",
        Bottleneck,
        [11] * 3,
        pretrained=pretrained,
        progress=progress,
        inplanes=16,
        **kwargs,
    )
"""