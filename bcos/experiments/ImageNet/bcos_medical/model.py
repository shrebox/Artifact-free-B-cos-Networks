"""
Model factory for bcos_medical base_network.

Builds B-cos ResNet-50 / DenseNet-121 with anti-aliased pooling modifications
(BlurPool or ASAP) originally developed for VinBigXray, targeting ImageNet
(1000 classes).

Per-layer pretrained-weight control
------------------------------------
For BlurPool and ASAP, each stride-2 layer that gets replaced can independently
keep its pretrained weights or be randomly re-initialised.  The choice is
encoded in ``model_config["pretrained_layers"]`` -- a list of layer-group names.

ResNet-50 layer groups (Bottleneck):
    conv1  - stem 7x7 conv (stride 2)
    l2     - layer2[0].conv2 + layer2[0].downsample[0]  (stride 2)
    l3     - layer3[0].conv2 + layer3[0].downsample[0]  (stride 2)
    l4     - layer4[0].conv2 + layer4[0].downsample[0]  (stride 2)

DenseNet-121 layer groups:
    conv0  - features.conv0 stem 7x7 conv (stride 2)

Pooling types
-------------
    Baseline  - standard torchvision model (no B-cos)
    Bcos      - original pretrained B-cos model, classifier only replaced
    BlurPool  - blur-based anti-aliased downsampling  (BlurThenConv1 + ModifiedBlurBcosConv2dPoolThenConv)
    ASAP      - FLC-based anti-aliased downsampling   (FLCThenConv1 + ModifiedFLCBcosConv2dPoolThenConv)
"""

import math

import torch
import torchvision
from torch import nn

from bcos.modules import LogitLayer
from bcos.modules.bcosconv2d import (
    BcosConv2d,
    BlurThenConv1,
    FLCThenConv1,
    ModifiedBlurBcosConv2dPoolThenConv,
    ModifiedFLCBcosConv2dPoolThenConv,
    NormedConv2d,
)

__all__ = ["get_model"]

NUM_CLASSES = 1_000


# ── helpers ─────────────────────────────────────────────────────────────


def _assert_same_weight(old: BcosConv2d, new_wrapper, name: str):
    """Verify that pretrained weights were correctly preserved in a wrapper."""
    new_conv = new_wrapper.conv if hasattr(new_wrapper, "conv") else new_wrapper
    max_diff = (old.linear.weight - new_conv.linear.weight).abs().max().item()
    print(f"[check] {name} max|Δw| = {max_diff:.6g}")


def _copy_layer_weights(old_layer, new_bcos_conv, name: str = ""):
    """
    Copy pretrained weights from *old_layer* into *new_bcos_conv*.
    Handles both ``BcosConv2d`` and ``nn.Conv2d`` sources.
    (Pattern taken from ConvNeXt ``_copy_downsample_weights``.)
    """
    with torch.no_grad():
        if isinstance(old_layer, BcosConv2d):
            src_lin = old_layer.linear
            dst_lin = new_bcos_conv.linear
            dst_lin.weight.copy_(src_lin.weight)
            if getattr(src_lin, "scale", None) is not None:
                dst_lin.scale = nn.Parameter(
                    src_lin.scale.detach().clone(),
                    requires_grad=src_lin.scale.requires_grad,
                )
            print(f"[copy] {name}: copied BcosConv2d weights ({src_lin.weight.shape})")
        elif isinstance(old_layer, nn.Conv2d):
            new_bcos_conv.linear.weight.copy_(old_layer.weight)
            print(f"[copy] {name}: copied Conv2d weights ({old_layer.weight.shape})")
        else:
            print(
                f"[warn] {name}: cannot copy from {type(old_layer).__name__}; "
                f"using random init"
            )


def _reinit_conv(wrapper, name: str = ""):
    """
    Re-initialise conv weights inside a ``BlurThenConv1`` / ``FLCThenConv1``
    wrapper.  These wrappers auto-copy pretrained weights on construction,
    so we call this when pretrained weights are **not** desired for that layer.
    """
    conv = wrapper.conv if hasattr(wrapper, "conv") else wrapper
    if hasattr(conv, "linear") and hasattr(conv.linear, "weight"):
        nn.init.kaiming_normal_(conv.linear.weight, mode="fan_out")
    elif hasattr(conv, "weight"):
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    print(f"[reinit] {name}: re-initialised weights to random")


# ── ResNet-50 builders ──────────────────────────────────────────────────


def _build_resnet50_blurpool(model, pretrained_layers, num_classes):
    """Apply BlurPool modifications to a pretrained B-cos ResNet-50."""

    # conv1 ── stem 7x7 (stride 2 -> blur + stride 1)
    model.conv1 = BlurThenConv1(model.conv1)
    if "conv1" not in pretrained_layers:
        _reinit_conv(model.conv1, "conv1")

    # layer2 ── save old -> replace -> optionally copy
    old_l2_conv2 = model.layer2[0].conv2
    old_l2_ds = model.layer2[0].downsample[0]
    model.layer2[0].conv2 = ModifiedBlurBcosConv2dPoolThenConv(
        128,
        128,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        b=2,
    )
    model.layer2[0].downsample[0] = ModifiedBlurBcosConv2dPoolThenConv(
        256,
        512,
        kernel_size=(1, 1),
        stride=(2, 2),
        b=2,
    )
    if "l2" in pretrained_layers:
        _copy_layer_weights(old_l2_conv2, model.layer2[0].conv2.conv, "layer2[0].conv2")
        _copy_layer_weights(
            old_l2_ds, model.layer2[0].downsample[0].conv, "layer2[0].downsample[0]"
        )

    # layer3
    old_l3_conv2 = model.layer3[0].conv2
    old_l3_ds = model.layer3[0].downsample[0]
    model.layer3[0].conv2 = ModifiedBlurBcosConv2dPoolThenConv(
        256,
        256,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        b=2,
    )
    model.layer3[0].downsample[0] = ModifiedBlurBcosConv2dPoolThenConv(
        512,
        1024,
        kernel_size=(1, 1),
        stride=(2, 2),
        b=2,
    )
    if "l3" in pretrained_layers:
        _copy_layer_weights(old_l3_conv2, model.layer3[0].conv2.conv, "layer3[0].conv2")
        _copy_layer_weights(
            old_l3_ds, model.layer3[0].downsample[0].conv, "layer3[0].downsample[0]"
        )

    # layer4
    old_l4_conv2 = model.layer4[0].conv2
    old_l4_ds = model.layer4[0].downsample[0]
    model.layer4[0].conv2 = ModifiedBlurBcosConv2dPoolThenConv(
        512,
        512,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        b=2,
    )
    model.layer4[0].downsample[0] = ModifiedBlurBcosConv2dPoolThenConv(
        1024,
        2048,
        kernel_size=(1, 1),
        stride=(2, 2),
        b=2,
    )
    if "l4" in pretrained_layers:
        _copy_layer_weights(old_l4_conv2, model.layer4[0].conv2.conv, "layer4[0].conv2")
        _copy_layer_weights(
            old_l4_ds, model.layer4[0].downsample[0].conv, "layer4[0].downsample[0]"
        )

    # classifier + logit layer
    model.fc.linear = NormedConv2d(
        2048,
        num_classes,
        kernel_size=(1, 1),
        stride=(1, 1),
        bias=False,
    )
    model.logit_layer = LogitLayer(
        logit_temperature=None,
        logit_bias=-math.log(num_classes - 1),
    )
    return model


def _build_resnet50_asap(model, pretrained_layers, num_classes):
    """Apply ASAP (FLC-based) modifications to a pretrained B-cos ResNet-50."""

    # conv1
    old_conv1 = model.conv1
    model.conv1 = FLCThenConv1(model.conv1, transpose=False, odd=False)
    if "conv1" not in pretrained_layers:
        _reinit_conv(model.conv1, "conv1")
    else:
        _assert_same_weight(old_conv1, model.conv1, "conv1")

    # layer2
    old_l2_conv2 = model.layer2[0].conv2
    old_l2_ds = model.layer2[0].downsample[0]
    model.layer2[0].conv2 = ModifiedFLCBcosConv2dPoolThenConv(
        128,
        128,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        b=2,
        transpose=False,
        odd=False,
    )
    model.layer2[0].downsample[0] = ModifiedFLCBcosConv2dPoolThenConv(
        256,
        512,
        kernel_size=(1, 1),
        stride=(2, 2),
        b=2,
        transpose=False,
        odd=False,
    )
    if "l2" in pretrained_layers:
        _copy_layer_weights(old_l2_conv2, model.layer2[0].conv2.conv, "layer2[0].conv2")
        _copy_layer_weights(
            old_l2_ds, model.layer2[0].downsample[0].conv, "layer2[0].downsample[0]"
        )

    # layer3
    old_l3_conv2 = model.layer3[0].conv2
    old_l3_ds = model.layer3[0].downsample[0]
    model.layer3[0].conv2 = ModifiedFLCBcosConv2dPoolThenConv(
        256,
        256,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        b=2,
        transpose=True,
        odd=True,
    )
    model.layer3[0].downsample[0] = ModifiedFLCBcosConv2dPoolThenConv(
        512,
        1024,
        kernel_size=(1, 1),
        stride=(2, 2),
        b=2,
        transpose=True,
        odd=True,
    )
    if "l3" in pretrained_layers:
        _copy_layer_weights(old_l3_conv2, model.layer3[0].conv2.conv, "layer3[0].conv2")
        _copy_layer_weights(
            old_l3_ds, model.layer3[0].downsample[0].conv, "layer3[0].downsample[0]"
        )

    # layer4
    old_l4_conv2 = model.layer4[0].conv2
    old_l4_ds = model.layer4[0].downsample[0]
    model.layer4[0].conv2 = ModifiedFLCBcosConv2dPoolThenConv(
        512,
        512,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        b=2,
        transpose=False,
        odd=False,
    )
    model.layer4[0].downsample[0] = ModifiedFLCBcosConv2dPoolThenConv(
        1024,
        2048,
        kernel_size=(1, 1),
        stride=(2, 2),
        b=2,
        transpose=False,
        odd=False,
    )
    if "l4" in pretrained_layers:
        _copy_layer_weights(old_l4_conv2, model.layer4[0].conv2.conv, "layer4[0].conv2")
        _copy_layer_weights(
            old_l4_ds, model.layer4[0].downsample[0].conv, "layer4[0].downsample[0]"
        )

    # classifier + logit layer
    model.fc.linear = NormedConv2d(
        2048,
        num_classes,
        kernel_size=(1, 1),
        stride=(1, 1),
        bias=False,
    )
    model.logit_layer = LogitLayer(
        logit_temperature=None,
        logit_bias=-math.log(num_classes - 1),
    )
    return model


# ── DenseNet-121 builders ───────────────────────────────────────────────


def _build_densenet121_blurpool(model, pretrained_layers, num_classes):
    """Apply BlurPool modifications to a pretrained B-cos DenseNet-121."""

    # conv0 ── stem 7x7 (stride 2 -> blur + stride 1)
    model.features.conv0 = BlurThenConv1(model.features.conv0)
    if "conv0" not in pretrained_layers:
        _reinit_conv(model.features.conv0, "features.conv0")

    # classifier + logit layer
    model.classifier.linear = NormedConv2d(
        1024,
        num_classes,
        kernel_size=(1, 1),
        stride=(1, 1),
        bias=False,
    )
    model.logit_layer = LogitLayer(
        logit_temperature=None,
        logit_bias=-math.log(num_classes - 1),
    )
    return model


def _build_densenet121_asap(model, pretrained_layers, num_classes):
    """Apply ASAP (FLC-based) modifications to a pretrained B-cos DenseNet-121."""

    old = model.features.conv0
    model.features.conv0 = FLCThenConv1(
        model.features.conv0, transpose=False, odd=False
    )
    if "conv0" not in pretrained_layers:
        _reinit_conv(model.features.conv0, "features.conv0")
    else:
        _assert_same_weight(old, model.features.conv0, "features.conv0")

    # classifier + logit layer
    model.classifier.linear = NormedConv2d(
        1024,
        num_classes,
        kernel_size=(1, 1),
        stride=(1, 1),
        bias=False,
    )
    model.logit_layer = LogitLayer(
        logit_temperature=None,
        logit_bias=-math.log(num_classes - 1),
    )
    return model


# ── Main entry point ────────────────────────────────────────────────────


def get_model(model_config) -> nn.Module:
    """
    Build a model for ImageNet (1000 classes) with medical-domain pooling
    modifications from VinBigXray.

    Expected ``model_config`` keys:
        name              : ``"resnet50"`` | ``"densenet121"``
        pooling_type      : ``"BlurPool"`` | ``"ASAP"`` | ``"Bcos"`` | ``"Baseline"``
        pretrained_layers : list of layer-group names whose modified replacements
                            should keep pretrained weights (BlurPool/ASAP only).
                            ResNet-50:    subset of ``["conv1","l2","l3","l4"]``
                            DenseNet-121: subset of ``["conv0"]``
        weights           : truthy for pretrained, falsy for random init (Bcos only)
        args.num_classes  : default 1000
    """
    arch_name = model_config["name"]
    pooling_type = model_config.get("pooling_type", None)
    pretrained_layers = model_config.get("pretrained_layers", [])
    num_classes = model_config.get("args", {}).get("num_classes", NUM_CLASSES)

    # ── Baseline: standard torchvision model (no B-cos) ──────────────
    if pooling_type == "Baseline":
        baseline_weights = model_config.get("weights", "DEFAULT")
        tv_weights = "DEFAULT" if baseline_weights else None
        if "resnet50" in arch_name:
            model = torchvision.models.resnet50(weights=tv_weights)
            model.fc = nn.Linear(2048, num_classes)
        elif "densenet121" in arch_name:
            model = torchvision.models.densenet121(weights=tv_weights)
            model.classifier = nn.Linear(1024, num_classes)
        else:
            raise ValueError(f"Unsupported architecture for Baseline: {arch_name}")
        print(
            f"Loaded {'pretrained' if tv_weights else 'random init'} "
            f"torchvision {arch_name}"
        )
        return model

    # ── Bcos: original pretrained B-cos model (no pooling mods) ──────
    if pooling_type == "Bcos":
        pretrained = bool(model_config.get("weights", None))
        hub_name = "resnet50" if "resnet50" in arch_name else "densenet121"
        print(
            f"Loading {'pretrained' if pretrained else 'random init'} B-cos {hub_name}..."
        )
        model = torch.hub.load("B-cos/B-cos-v2", hub_name, pretrained=pretrained)
        # Pretrained B-cos already has 1000-class head; no replacement needed.
        return model

    # ── BlurPool / ASAP: load B-cos base, then modify layers ─────────
    if pooling_type in ("BlurPool", "ASAP"):
        pretrained_base = bool(model_config.get("weights", "pretrained"))
        if "resnet50" in arch_name:
            print(
                f"Loading {'pretrained' if pretrained_base else 'random init'} "
                f"B-cos resnet50 for {pooling_type} modification..."
            )
            model = torch.hub.load(
                "B-cos/B-cos-v2", "resnet50", pretrained=pretrained_base
            )
            if pooling_type == "BlurPool":
                return _build_resnet50_blurpool(model, pretrained_layers, num_classes)
            else:  # ASAP
                return _build_resnet50_asap(model, pretrained_layers, num_classes)

        elif "densenet121" in arch_name:
            print(
                f"Loading {'pretrained' if pretrained_base else 'random init'} "
                f"B-cos densenet121 for {pooling_type} modification..."
            )
            model = torch.hub.load(
                "B-cos/B-cos-v2", "densenet121", pretrained=pretrained_base
            )
            if pooling_type == "BlurPool":
                return _build_densenet121_blurpool(
                    model, pretrained_layers, num_classes
                )
            else:  # ASAP
                return _build_densenet121_asap(model, pretrained_layers, num_classes)

        else:
            raise ValueError(
                f"Unsupported architecture for {pooling_type}: {arch_name}"
            )

    raise ValueError(
        f"Unknown pooling_type: {pooling_type!r}. "
        f"Expected one of: Baseline, BlurPool, ASAP, Bcos"
    )
