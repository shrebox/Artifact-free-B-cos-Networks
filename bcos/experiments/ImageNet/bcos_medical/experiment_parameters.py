"""
Experiment configs for ``bcos_medical`` base_network on ImageNet.

Training hyper-parameters are taken from ``ImageNet/bcosification``
(batch 64, 90 epochs, Adam lr=1e-4, cosine LR, AGC, etc.).

Model modifications (anti-aliased pooling) come from the VinBigXray
experiments, re-targeting 1000 ImageNet classes instead of 14.

Naming convention
-----------------
Each experiment name encodes the architecture, pooling type, and which
modified stride-2 layers keep their pretrained weights::

    {arch}_{depth}_{pooling}_{pretrained_preset}

Pretrained presets for **ResNet-50** (layers with stride > 1):

    ============== =======================================
    Preset         Layers with pretrained weights
    ============== =======================================
    preNone        (none -- all modified layers random)
    preConv1       conv1
    preConv1L2     conv1, layer2
    preConv1L2L3   conv1, layer2, layer3
    preAll         conv1, layer2, layer3, layer4
    ============== =======================================

Pretrained presets for **DenseNet-121** (only conv0 is modified):

    ============== =======================================
    Preset         Layers with pretrained weights
    ============== =======================================
    preNone        (none -- conv0 random)
    preAll         conv0
    ============== =======================================

``Bcos`` and ``Baseline`` have no layer modifications, so no preset suffix.

Full list of generated experiment names (26 base, 78 with seeds)::

    # ResNet-50 (pretrained backbone)
    resnet_50_BlurPool_preNone
    resnet_50_BlurPool_preConv1
    resnet_50_BlurPool_preConv1L2
    resnet_50_BlurPool_preConv1L2L3
    resnet_50_BlurPool_preAll
    resnet_50_ASAP_preNone
    resnet_50_ASAP_preConv1
    resnet_50_ASAP_preConv1L2
    resnet_50_ASAP_preConv1L2L3
    resnet_50_ASAP_preAll
    resnet_50_Bcos
    resnet_50_Baseline

    # DenseNet-121 (pretrained backbone)
    densenet_121_BlurPool_preNone
    densenet_121_BlurPool_preAll
    densenet_121_ASAP_preNone
    densenet_121_ASAP_preAll
    densenet_121_Bcos
    densenet_121_Baseline

    # Non-pretrained variants (entire model from random init)
    resnet_50_BlurPool_noPretrained
    resnet_50_ASAP_noPretrained
    resnet_50_Bcos_noPretrained
    resnet_50_Baseline_noPretrained
    densenet_121_BlurPool_noPretrained
    densenet_121_ASAP_noPretrained
    densenet_121_Bcos_noPretrained
    densenet_121_Baseline_noPretrained

    # + seed variants  (append _seed5 / _seed420 / _seed1337)
"""

import math  # noqa
from collections import OrderedDict

from torch import nn

from bcos.data.presets import (
    ImageNetClassificationPresetEval,
    ImageNetClassificationPresetTrain,
)
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.modules import norms
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)
from bcos.optim import LRSchedulerFactory, OptimizerFactory

__all__ = ["CONFIGS"]

# ═════════════════════════════════════════════════════════════════════════
# Constants  (from ImageNet/bcosification)
# ═════════════════════════════════════════════════════════════════════════
NUM_CLASSES = 1_000
NUM_TRAIN_EXAMPLES: int = 1_281_167
NUM_EVAL_EXAMPLES: int = 50_000

DEFAULT_BATCH_SIZE = 64  # per GPU; x4 GPUs = 256 effective
DEFAULT_NUM_EPOCHS = 90
DEFAULT_LR = 1e-4
DEFAULT_CROP_SIZE = 224

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)  # bnu-linear

# ── optimizers ──────────────────────────────────────────────────────────
DEFAULT_OPTIMIZER = OptimizerFactory(
    name="Adam",
    lr=DEFAULT_LR,
    bcosify=True,
    b_opt=False,
)
BASELINE_OPTIMIZER = OptimizerFactory(
    name="Adam",
    lr=DEFAULT_LR,
    bcosify=False,
    b_opt=False,
)

# ── LR schedule ────────────────────────────────────────────────────────
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="cosineannealinglr",
    epochs=DEFAULT_NUM_EPOCHS,
)

# ── transforms ──────────────────────────────────────────────────────────
BCOS_TRAIN_TRANSFORM = ImageNetClassificationPresetTrain(
    crop_size=DEFAULT_CROP_SIZE,
    is_bcos=True,
)
BCOS_TEST_TRANSFORM = ImageNetClassificationPresetEval(
    crop_size=DEFAULT_CROP_SIZE,
    is_bcos=True,
)
BASELINE_TRAIN_TRANSFORM = ImageNetClassificationPresetTrain(
    crop_size=DEFAULT_CROP_SIZE,
    is_bcos=False,
)
BASELINE_TEST_TRANSFORM = ImageNetClassificationPresetEval(
    crop_size=DEFAULT_CROP_SIZE,
    is_bcos=False,
)

# ═════════════════════════════════════════════════════════════════════════
# Default config dict  (B-cos variant -- used by BlurPool, ASAP, Bcos)
# ═════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    data=dict(
        train_transform=BCOS_TRAIN_TRANSFORM,
        test_transform=BCOS_TEST_TRANSFORM,
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=16,
        num_classes=NUM_CLASSES,
        # ImageNet is single-label multiclass, NOT multilabel.
        # Explicit flag prevents the trainer from auto-detecting multilabel
        # due to the BCE-like loss used by B-cos models.
        multilabel=False,
    ),
    model=dict(
        is_bcos=True,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=DEFAULT_NORM_LAYER,
            logit_bias=-math.log(NUM_CLASSES - 1),
        ),
        bcos_args=dict(b=2, max_out=1),
    ),
    criterion=UniformOffLabelsBCEWithLogitsLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(
        max_epochs=DEFAULT_NUM_EPOCHS,
    ),
    use_agc=True,
)


def update_default(new_config):
    return update_config(DEFAULTS, new_config)


# ═════════════════════════════════════════════════════════════════════════
# Pretrained-layer presets
# ═════════════════════════════════════════════════════════════════════════
# For BlurPool / ASAP: which modified stride-2 layers keep their
# pretrained weights.  Layers NOT listed are randomly re-initialised.
#
# ResNet-50 (Bottleneck) modified layer groups:
#   conv1  - stem 7x7 conv (stride 2)
#   l2     - layer2[0].conv2 + layer2[0].downsample[0]
#   l3     - layer3[0].conv2 + layer3[0].downsample[0]
#   l4     - layer4[0].conv2 + layer4[0].downsample[0]
#
# DenseNet-121 modified layer groups:
#   conv0  - features.conv0 stem 7x7 conv (stride 2)
# ═════════════════════════════════════════════════════════════════════════

RESNET_PRETRAINED_PRESETS = OrderedDict(
    [
        ("preNone", []),
        ("preConv1", ["conv1"]),
        ("preConv1L2", ["conv1", "l2"]),
        ("preConv1L2L3", ["conv1", "l2", "l3"]),
        ("preAll", ["conv1", "l2", "l3", "l4"]),
    ]
)

DENSENET_PRETRAINED_PRESETS = OrderedDict(
    [
        ("preNone", []),
        ("preAll", ["conv0"]),
    ]
)

# ═════════════════════════════════════════════════════════════════════════
# Pooling types
# ═════════════════════════════════════════════════════════════════════════
POOLINGS = ["BlurPool", "ASAP", "Bcos", "Baseline"]

# ═════════════════════════════════════════════════════════════════════════
# Config generation
# ═════════════════════════════════════════════════════════════════════════
CONFIGS = {}

# ── ResNet-50 ───────────────────────────────────────────────────────────
for pooling in POOLINGS:
    if pooling in ("BlurPool", "ASAP"):
        for preset_name, preset_layers in RESNET_PRETRAINED_PRESETS.items():
            name = f"resnet_50_{pooling}_{preset_name}"
            CONFIGS[name] = update_default(
                dict(
                    model=dict(
                        name="resnet50",
                        pooling_type=pooling,
                        pretrained_layers=list(preset_layers),
                    ),
                )
            )

    elif pooling == "Bcos":
        CONFIGS["resnet_50_Bcos"] = update_default(
            dict(
                model=dict(
                    name="resnet50",
                    pooling_type="Bcos",
                    weights="pretrained",
                ),
            )
        )

    elif pooling == "Baseline":
        CONFIGS["resnet_50_Baseline"] = update_default(
            dict(
                data=dict(
                    train_transform=BASELINE_TRAIN_TRANSFORM,
                    test_transform=BASELINE_TEST_TRANSFORM,
                ),
                model=dict(
                    is_bcos=False,
                    name="resnet50",
                    pooling_type="Baseline",
                ),
                criterion=nn.CrossEntropyLoss(),
                test_criterion=nn.CrossEntropyLoss(),
                optimizer=BASELINE_OPTIMIZER,
            )
        )

# ── DenseNet-121 ────────────────────────────────────────────────────────
for pooling in POOLINGS:
    if pooling in ("BlurPool", "ASAP"):
        for preset_name, preset_layers in DENSENET_PRETRAINED_PRESETS.items():
            name = f"densenet_121_{pooling}_{preset_name}"
            CONFIGS[name] = update_default(
                dict(
                    model=dict(
                        name="densenet121",
                        pooling_type=pooling,
                        pretrained_layers=list(preset_layers),
                    ),
                )
            )

    elif pooling == "Bcos":
        CONFIGS["densenet_121_Bcos"] = update_default(
            dict(
                model=dict(
                    name="densenet121",
                    pooling_type="Bcos",
                    weights="pretrained",
                ),
            )
        )

    elif pooling == "Baseline":
        CONFIGS["densenet_121_Baseline"] = update_default(
            dict(
                data=dict(
                    train_transform=BASELINE_TRAIN_TRANSFORM,
                    test_transform=BASELINE_TEST_TRANSFORM,
                ),
                model=dict(
                    is_bcos=False,
                    name="densenet121",
                    pooling_type="Baseline",
                ),
                criterion=nn.CrossEntropyLoss(),
                test_criterion=nn.CrossEntropyLoss(),
                optimizer=BASELINE_OPTIMIZER,
            )
        )

# ── Non-pretrained variants (entire backbone from random init) ──────────
# These differ from preNone: preNone still loads a pretrained backbone
# and only re-initialises the *modified* layers; noPretrained starts
# the entire model from random initialisation.
ARCHITECTURES_NP = {
    "resnet_50": "resnet50",
    "densenet_121": "densenet121",
}

for _arch_display, _arch_name in ARCHITECTURES_NP.items():
    for _pooling in POOLINGS:
        _name = f"{_arch_display}_{_pooling}_noPretrained"

        if _pooling in ("BlurPool", "ASAP"):
            CONFIGS[_name] = update_default(
                dict(
                    model=dict(
                        name=_arch_name,
                        pooling_type=_pooling,
                        pretrained_layers=[],
                        weights=None,
                    ),
                )
            )

        elif _pooling == "Bcos":
            CONFIGS[_name] = update_default(
                dict(
                    model=dict(
                        name=_arch_name,
                        pooling_type="Bcos",
                        # weights absent -> defaults to None -> random init
                    ),
                )
            )

        elif _pooling == "Baseline":
            CONFIGS[_name] = update_default(
                dict(
                    data=dict(
                        train_transform=BASELINE_TRAIN_TRANSFORM,
                        test_transform=BASELINE_TEST_TRANSFORM,
                    ),
                    model=dict(
                        is_bcos=False,
                        name=_arch_name,
                        pooling_type="Baseline",
                        weights=None,
                    ),
                    criterion=nn.CrossEntropyLoss(),
                    test_criterion=nn.CrossEntropyLoss(),
                    optimizer=BASELINE_OPTIMIZER,
                )
            )

# ── Seed expansion ──────────────────────────────────────────────────────
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5, 420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
