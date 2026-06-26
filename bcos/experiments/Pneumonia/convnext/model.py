import math

import torch
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


def _assert_same_weight(old: BcosConv2d, new_wrapper, name: str):
    new_conv = new_wrapper.conv if hasattr(new_wrapper, "conv") else new_wrapper
    max_diff = (old.linear.weight - new_conv.linear.weight).abs().max().item()
    print(f"[check] {name} max|Δw| = {max_diff:.6g}")


def _convnext_stage_dims(arch_name: str):
    name = arch_name.lower()
    if "tiny" in name or "small" in name:
        return (96, 192, 384, 768)
    if "base" in name:
        return (128, 256, 512, 1024)
    if "large" in name:
        return (192, 384, 768, 1536)
    if "xlarge" in name:
        return (256, 512, 1024, 2048)
    raise ValueError(f"Unknown ConvNeXt size in arch_name: {arch_name!r}")


def get_model(model_config) -> nn.Module:
    # extract args
    arch_name = model_config["name"]

    # Anti-aliasing B-cos
    pooling_type = model_config.get("pooling_type", None)
    if pooling_type is not None:
        is_convnext = "convnext" in arch_name
        pretrained = bool(model_config.get("weights", None))

        # arch_type: pn vs bnu (you encoded this via args.norm_layer)
        # pn => args.norm_layer is None
        args_cfg = model_config.get("args", {}) or {}
        keep_pos_norm = args_cfg.get("norm_layer", None) is None

        if is_convnext:
            if not keep_pos_norm:
                model_to_load = arch_name + "_bnu"
            else:
                model_to_load = arch_name

            stage_c1, stage_c2, stage_c3, stage_c4 = _convnext_stage_dims(arch_name)

            if pretrained:
                print(f"Loading pretrained weights for {model_to_load}...")
                model = torch.hub.load("B-cos/B-cos-v2", model_to_load, pretrained=True)
            else:
                print(f"Loading random init for {model_to_load}...")
                model = torch.hub.load(
                    "B-cos/B-cos-v2", model_to_load, pretrained=False
                )

            if pooling_type in ["BcosPretrained", "Bcos"]:
                print("Only replacing the classifier layer and logit layer...")
                # Only replace the classifier layer and logit layer, keeping all the pretrained weights (including conv1 and pool)
                model.classifier[1] = NormedConv2d(
                    stage_c4, 2, kernel_size=(1, 1), stride=(1, 1), bias=False
                )

                model.logit_layer = LogitLayer(
                    logit_temperature=None,
                    logit_bias=-math.log(1),
                )
                return model

            elif pooling_type == "BlurPool":
                model.features[0] = BlurThenConv1(model.features[0])
                model.features[3][1] = ModifiedBlurBcosConv2dPoolThenConv(
                    stage_c1,
                    stage_c2,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    b=2,
                )
                model.features[5][1] = ModifiedBlurBcosConv2dPoolThenConv(
                    stage_c2,
                    stage_c3,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    b=2,
                )
                model.features[7][1] = ModifiedBlurBcosConv2dPoolThenConv(
                    stage_c3,
                    stage_c4,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    b=2,
                )

                model.classifier[1] = NormedConv2d(
                    stage_c4, 2, kernel_size=(1, 1), stride=(1, 1), bias=False
                )

                model.logit_layer = LogitLayer(
                    logit_temperature=None,
                    logit_bias=-math.log(1),
                )

            elif pooling_type == "FLCPool":
                old = model.features[0]
                model.features[0] = FLCThenConv1(
                    model.features[0], transpose=False, odd=False
                )
                _assert_same_weight(old, model.features[0], "features[0]")

                model.features[3][1] = ModifiedFLCBcosConv2dPoolThenConv(
                    stage_c1,
                    stage_c2,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    b=2,
                    transpose=True,
                    odd=True,
                )
                model.features[5][1] = ModifiedFLCBcosConv2dPoolThenConv(
                    stage_c2,
                    stage_c3,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    b=2,
                    transpose=False,
                    odd=False,
                )
                model.features[7][1] = ModifiedFLCBcosConv2dPoolThenConv(
                    stage_c3,
                    stage_c4,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    b=2,
                    transpose=True,
                    odd=True,
                )

                model.classifier[1] = NormedConv2d(
                    stage_c4, 2, kernel_size=(1, 1), stride=(1, 1), bias=False
                )
                model.logit_layer = LogitLayer(
                    logit_temperature=None, logit_bias=-math.log(1)
                )
            else:
                # For non-convnext or no pooling change, just load the standard model (pretrained or random init)
                print(
                    f"Loading standard torchvision model for {arch_name} with pretrained={pretrained}..."
                )
                from torchvision.models import (
                    convnext_base,
                    convnext_tiny,
                )

                if pretrained:
                    print(f"Loading pretrained weights for {arch_name}...")
                    if "tiny" in arch_name.lower():
                        model = convnext_tiny(weights="DEFAULT")
                    else:
                        model = convnext_base(weights="DEFAULT")
                else:
                    print(f"Loading random init for {arch_name}...")
                    if "tiny" in arch_name.lower():
                        model = convnext_tiny()
                    else:
                        model = convnext_base()
                model.classifier[2] = nn.Linear(
                    in_features=stage_c4, out_features=2, bias=True
                )
        return model
    else:
        raise NotImplementedError(
            f"Pooling type {pooling_type} not implemented for arch {arch_name}"
        )
