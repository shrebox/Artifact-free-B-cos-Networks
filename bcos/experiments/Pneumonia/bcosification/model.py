import torch
import torchvision
from torch import nn
from torchvision.models.densenet import DenseNet121_Weights, _load_state_dict
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet18_Weights,
    ResNet50_Weights,
)

from bcos.models.standard_models import DenseNetBcos, ResNetBcos
from bcos.modules import LogitLayer
from bcos.modules.bcosconv2d import (
    BcosConv2d,
    BlurThenConv1,
    FLCThenConv1,
    ModifiedBlurBcosConv2dPoolThenConv,
    ModifiedFLCBcosConv2dPoolThenConv,
    NormedConv2d,
)
from bcosify import BcosifyNetwork

__all__ = ["get_model"]


def get_torch_model_modified(arch_name: str, model_config):
    if arch_name == "resnet18":
        tv_model = ResNetBcos(BasicBlock, [2, 2, 2, 2])
        weight_type = model_config["weights"]
        if weight_type:
            weights = ResNet18_Weights.verify(model_config["weights"])
            tv_model.load_state_dict(weights.get_state_dict(progress=False))
        return tv_model
    if arch_name == "resnet50":
        tv_model = ResNetBcos(Bottleneck, [3, 4, 6, 3])
        weight_type = model_config["weights"]
        if weight_type:
            weights = ResNet50_Weights.verify(model_config["weights"])
            tv_model.load_state_dict(weights.get_state_dict(progress=False))
        return tv_model
    if arch_name == "densenet121":
        tv_model = DenseNetBcos(32, (6, 12, 24, 16), 64)
        weight_type = model_config["weights"]
        if weight_type:
            weights = DenseNet121_Weights.verify(model_config["weights"])
            _load_state_dict(model=tv_model, weights=weights, progress=False)
        return tv_model


def _assert_same_weight(old: BcosConv2d, new_wrapper, name: str):
    new_conv = new_wrapper.conv if hasattr(new_wrapper, "conv") else new_wrapper
    max_diff = (old.linear.weight - new_conv.linear.weight).abs().max().item()
    print(f"[check] {name} max|Δw| = {max_diff:.6g}")


def get_model(model_config) -> nn.Module:
    # extract args
    arch_name = model_config["name"]
    pretrained = bool(model_config.get("weights", None))

    # Anti-aliasing B-cos
    pooling_type = model_config.get("pooling_type", None)
    if pooling_type is not None:
        model = torch.hub.load("B-cos/B-cos-v2", "resnet50", pretrained=True)
        if pooling_type == "BlurPool":
            # # Make conv1 stride=1 and downsample via BlurPool (pre-conv), while keeping pretrained weights
            model.conv1 = BlurThenConv1(model.conv1)
            model.layer2[0].conv2 = ModifiedBlurBcosConv2dPoolThenConv(
                128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), b=2
            )
            model.layer2[0].downsample[0] = ModifiedBlurBcosConv2dPoolThenConv(
                256, 512, kernel_size=(1, 1), stride=(2, 2), b=2
            )
            model.layer3[0].conv2 = ModifiedBlurBcosConv2dPoolThenConv(
                256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), b=2
            )
            model.layer3[0].downsample[0] = ModifiedBlurBcosConv2dPoolThenConv(
                512, 1024, kernel_size=(1, 1), stride=(2, 2), b=2
            )
            model.layer4[0].conv2 = ModifiedBlurBcosConv2dPoolThenConv(
                512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), b=2
            )
            model.layer4[0].downsample[0] = ModifiedBlurBcosConv2dPoolThenConv(
                1024, 2048, kernel_size=(1, 1), stride=(2, 2), b=2
            )
            model.fc.linear = NormedConv2d(
                2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False
            )  # code from B-cos paper reused to adjust network

            model.logit_layer = LogitLayer(
                logit_temperature=None,
                logit_bias=0.0,
            )

        elif pooling_type == "FLCPool":
            # --- NEW: make conv1 stride=1 and downsample via FLC ---
            # model has attribute "conv1" and "pool" (not maxpool)
            # Test for the weights copy
            old = model.conv1
            model.conv1 = FLCThenConv1(model.conv1, transpose=False, odd=False)
            _assert_same_weight(old, model.conv1, "conv1")
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
            model.fc.linear = NormedConv2d(
                2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False
            )  # code from B-cos paper reused to adjust network

            model.logit_layer = LogitLayer(
                logit_temperature=None,
                logit_bias=0.0,
            )
        elif pooling_type == "Bcos":
            if pretrained:
                print("Loading pretrained B-cos ResNet50...")
            else:
                print("Loading random init B-cos ResNet50...")
            model = torch.hub.load("B-cos/B-cos-v2", "resnet50", pretrained=pretrained)
            model.fc.linear = NormedConv2d(
                2048, 2, kernel_size=(1, 1), stride=(1, 1), bias=False
            )
            model.logit_layer = LogitLayer(logit_temperature=None, logit_bias=0.0)
            return model
        elif pooling_type == "Baseline":
            model = torchvision.models.resnet50(weights="DEFAULT")
            model.fc = nn.Linear(2048, 2)
            return model
    else:
        # B-cosification pipeline
        model = BcosifyNetwork(
            get_torch_model_modified(arch_name, model_config),
            model_config,
            add_channels=True,
            logit_layer=True,
        )
        # For standard changes
        standard_changes = model_config.get("standard_changes", None)
        for k, v in standard_changes.items():
            print("Changing maxpool to avgpool")
            exec(f"model.model.{k} = v")
        # Making all the bias parameters None
        print("Removing bias parameters (making None)")
        for mod in model.modules():
            if hasattr(mod, "bias") and mod.bias is not None:
                mod.bias = None
    return model
