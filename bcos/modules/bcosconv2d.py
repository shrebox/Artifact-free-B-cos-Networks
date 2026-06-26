import warnings
from typing import Optional, Tuple, Union

import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair

from .blurpool import BlurPool
from .common import DetachableModule
from .flc_pooling import FLC_Pooling

__all__ = [
    "NormedConv2d",
    "BcosConv2d",
    "FLCThenConv1",
    "BlurThenConv1",
]


class NormedConv2d(nn.Conv2d):
    """
    Standard 2D convolution, but with unit norm weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = None
        self.use_weight_norm = True

    def forward(self, in_tensor: Tensor) -> Tensor:
        # For toggling weight normalisation
        if self.use_weight_norm:
            w = self.weight / LA.vector_norm(
                self.weight, dim=(1, 2, 3), keepdim=True
            )  # [out_channels, in_channels, height, width]
        else:
            w = self.weight
        # For scaling the weights with initial pre-trained weights
        if self.scale is not None and self.use_weight_norm:
            w = self.scale * w
        return self._conv_forward(
            input=in_tensor, weight=w, bias=self.bias
        )  # Here bias = False not possible as _conv_forward expects bias tensor

    def set_scale(self, weight: Tensor, trainable=False):
        self.scale = nn.Parameter(
            weight.norm(p=2, dim=(1, 2, 3), keepdim=True), requires_grad=trainable
        )

    def toggle_weight_norm(self, use_weight_norm):
        self.use_weight_norm = use_weight_norm


class BcosConv2d(DetachableModule):
    """
    BcosConv2d is a 2D convolution with unit norm weights and a cosine similarity
    activation function. The cosine similarity is calculated between the
    convolutional patch and the weight vector. The output is then scaled by the
    cosine similarity.

    See the paper for more details: https://arxiv.org/abs/2205.10268

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image
    out_channels : int
        Number of channels produced by the convolution
    kernel_size : int | tuple[int, ...]
        Size of the convolving kernel
    stride : int | tuple[int, ...]
        Stride of the convolution. Default: 1
    padding : int | tuple[int, ...]
        Zero-padding added to both sides of the input. Default: 0
    dilation : int | tuple[int, ...]
        Spacing between kernel elements. Default: 1
    groups : int
        Number of blocked connections from input channels to output channels.
        Default: 1
    padding_mode : str
        Padding mode. One of ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        Default: ``'zeros'``
    device : Optional[torch.device]
        The device of the weights.
    dtype : Optional[torch.dtype]
        The dtype of the weights.
    b : int | float
        The base of the exponential used to scale the cosine similarity.
    max_out : int
        Number of MaxOut units to use. If 1, no MaxOut is used.
    **kwargs : Any
        Ignored.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        bias: bool = False,
        # special (note no scale here! See BcosConv2dWithScale below)
        b: Union[int, float] = 2,
        max_out: int = 1,
        **kwargs,  # bias is always False
    ):
        assert max_out > 0, f"max_out should be greater than 0, was {max_out}"
        super().__init__()

        # save everything
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.bias = None

        self.b = b
        self.max_out = max_out

        # check dilation
        # if dilation > 1:
        if any(d > 1 for d in self.dilation):
            warnings.warn("dilation > 1 is much slower!")
            self.calc_patch_norms = self._calc_patch_norms_slow

        self.linear = NormedConv2d(
            in_channels=in_channels,
            out_channels=out_channels * max_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass implementation.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        return self.forward_impl(in_tensor)

    def forward_impl(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (B, C, H, W)

        Returns:
            BcosConv2d output on the input tensor.
        """
        # Simple linear layer
        out = self.linear(in_tensor)

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            O = self.out_channels  # noqa: E741
            out = out.unflatten(dim=1, sizes=(O, M))
            out = out.max(dim=2, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1:
            return out

        # Calculating the norm of input patches: ||x||
        norm = self.calc_patch_norms(in_tensor)

        # Calculate the dynamic scale (|cos|^(B-1))
        # Note that cos = (x·ŵ)/||x||
        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / norm
        else:
            abs_cos = (maybe_detached_out / norm).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        # put everything together
        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)
        return out

    def calc_patch_norms(self, in_tensor: Tensor) -> Tensor:
        """
        Calculates the norms of the patches.
        """
        squares = in_tensor**2
        if self.groups == 1:
            # normal conv
            squares = squares.sum(1, keepdim=True)
        else:
            G = self.groups
            C = self.in_channels
            # group channels together and sum reduce over them
            # ie [N,C,H,W] -> [N,G,C//G,H,W] -> [N,G,H,W]
            # note groups MUST come first
            squares = squares.unflatten(1, (G, C // G)).sum(2)

        norms = (
            F.avg_pool2d(
                squares,
                self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                divisor_override=1,
            )
            + 1e-6  # stabilizing term
        ).sqrt_()

        if self.groups > 1:
            # norms.shape will be [N,G,H,W] (here H,W are spatial dims of output)
            # we need to convert this into [N,O,H,W] so that we can divide by this norm
            # (because we can't easily do broadcasting)
            N, G, H, W = norms.shape
            O = self.out_channels  # noqa: E741
            norms = torch.repeat_interleave(norms, repeats=O // G, dim=1)

        return norms

    def _calc_patch_norms_slow(self, in_tensor: Tensor) -> Tensor:
        # this is much slower but definitely correct
        # use for testing or something difficult to implement
        # like dilation
        ones_kernel = torch.ones_like(self.linear.weight)

        return (
            F.conv2d(
                in_tensor**2,
                ones_kernel,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
            + 1e-6
        ).sqrt_()

    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        # final comma as self.linear is shown in next line
        s += ","

        return s.format(**self.__dict__)


class BcosConv2dWithScale(BcosConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 1,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # special
        b: Union[int, float] = 2,
        max_out: int = 1,
        # the following come from the original v1 https://github.com/moboehle/B-cos
        scale: Optional[float] = None,  # use default init
        scale_factor: Union[int, float] = 100.0,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            device,
            dtype,
            b,
            max_out,
            **kwargs,
        )

        if scale is None:
            ks_scale = (
                kernel_size
                if not isinstance(kernel_size, tuple)
                else np.sqrt(np.prod(kernel_size))
            )
            self.scale = (ks_scale * np.sqrt(self.in_channels)) / scale_factor
        else:
            assert scale != 1.0, "For scale=1.0, use the normal BcosConv2d instead!"
            self.scale = scale

        warnings.warn(
            "BcosConv2dWithScale is deprecated and will be removed in a future version. "
            "Use BcosConv2d with scale=1.0 instead.",
            DeprecationWarning,
        )

    def forward(self, in_tensor: Tensor) -> Tensor:
        out = self.forward_impl(in_tensor)
        return out / self.scale

    def extra_repr(self) -> str:
        result = super().extra_repr()
        result = f"scale={self.scale:.3f}, " + result
        return result


class ModifiedBlurBcosConv2dPoolThenConv(nn.Module):
    """
    BcosConv2d + BlurPool where downsampling happens BEFORE the conv (reversed forward),
    analogous to your FLC/ASAP PoolThenConv wrappers.

    - pooling does the stride/downsample
    - conv is always stride=1
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=(0, 0),
        b=2,
        max_out=1,
    ):
        super().__init__()
        current_stride = stride[0] if isinstance(stride, tuple) else stride

        # BlurPool operates on *input* channels
        self.pool = (
            BlurPool(in_channels, stride=current_stride)
            if current_stride > 1
            else nn.Identity()
        )

        # Convolution always stride=1; pooling handles downsampling
        self.conv = BcosConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            b=b,
            max_out=max_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


# --- New: BlurThenConv1 ---
class BlurThenConv1(nn.Module):
    """Replaces a stride-2 conv1 with: BlurPool downsample -> same conv1 but stride=1.

    Copies weights/scales from the existing BcosConv2d conv1.

    Notes:
      - BlurPool operates on the *input* channels (e.g., 6 for B-cosification AddInverse).
      - Downsampling factor is 2 (stride=2).
    """

    def __init__(self, old_conv1: BcosConv2d):
        super().__init__()
        assert isinstance(old_conv1, BcosConv2d)

        old_lin = old_conv1.linear  # NormedConv2d

        # BlurPool downsampling on input channels
        self.pool = BlurPool(old_lin.in_channels, stride=2)

        # Recreate conv1 as stride=1 but keep everything else identical
        self.conv = BcosConv2d(
            in_channels=old_lin.in_channels,
            out_channels=old_lin.out_channels,
            kernel_size=old_lin.kernel_size,
            stride=1,
            padding=old_lin.padding,
            dilation=old_lin.dilation,
            groups=old_lin.groups,
            padding_mode=old_lin.padding_mode,
            b=old_conv1.b,
            max_out=old_conv1.max_out,
        )

        # Copy weights (and optional scale) so you keep pretrained init
        with torch.no_grad():
            self.conv.linear.weight.copy_(old_lin.weight)
            if (
                getattr(old_lin, "scale", None) is not None
                and old_lin.scale is not None
            ):
                self.conv.linear.scale = nn.Parameter(
                    old_lin.scale.detach().clone(),
                    requires_grad=old_lin.scale.requires_grad,
                )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class ModifiedFLCBcosConv2dPoolThenConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=(0, 0),
        b=2,
        max_out=1,
        transpose=False,
        odd=False,
    ):
        super().__init__()
        current_stride = stride[0] if isinstance(stride, tuple) else stride

        self.pool = (
            FLC_Pooling(transpose=transpose, odd=odd)
            if current_stride > 1
            else nn.Identity()
        )

        self.conv = BcosConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,  # keep stride=1
            padding=padding,
            b=b,
            max_out=max_out,
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class FLCThenConv1(nn.Module):
    """
    Replaces a stride-2 conv1 with: FLC downsample -> same conv1 but stride=1.
    Copies weights/scales from the existing BcosConv2d conv1.
    """

    def __init__(
        self, old_conv1: BcosConv2d, transpose: bool = False, odd: bool = False
    ):
        super().__init__()
        assert isinstance(old_conv1, BcosConv2d)

        # FLC does the downsampling (factor 2)
        self.pool = FLC_Pooling(transpose=transpose, odd=odd)

        # Recreate conv1 as stride=1 but keep everything else identical
        old_lin = old_conv1.linear  # NormedConv2d
        self.conv = BcosConv2d(
            in_channels=old_lin.in_channels,
            out_channels=old_lin.out_channels,  # should be 64
            kernel_size=old_lin.kernel_size,  # (7,7)
            stride=1,  # IMPORTANT: stride 1 now
            padding=old_lin.padding,  # (3,3)
            dilation=old_lin.dilation,
            groups=old_lin.groups,
            padding_mode=old_lin.padding_mode,
            b=old_conv1.b,
            max_out=old_conv1.max_out,
        )

        # Copy weights (and optional scale) so you keep pretrained init
        with torch.no_grad():
            self.conv.linear.weight.copy_(old_lin.weight)
            if (
                getattr(old_lin, "scale", None) is not None
                and old_lin.scale is not None
            ):
                self.conv.linear.scale = nn.Parameter(
                    old_lin.scale.detach().clone(),
                    requires_grad=old_lin.scale.requires_grad,
                )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
