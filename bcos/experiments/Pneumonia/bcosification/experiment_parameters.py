import os
import math  # noqa

import torchvision.transforms as T
from torch import nn

from bcos.data.transforms import AddInverse
from bcos.experiments.utils import (
    configs_cli,
    create_configs_with_different_seeds,
    update_config,
)
from bcos.modules import norms
from bcos.optim import LRSchedulerFactory, OptimizerFactory
from bcos.modules.losses import (
    BinaryCrossEntropyLoss,
    UniformOffLabelsBCEWithLogitsLoss,
)

__all__ = ["CONFIGS"]

# ----------------------------
# Pneumonia task (binary)
# ----------------------------
NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 30
DEFAULT_LR = 1e-4 #1e-5
DEFAULT_CROP_SIZE = 224

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)

DEFAULT_OPTIMIZER = OptimizerFactory(name="Adam", lr=DEFAULT_LR, bcosify=True, b_opt=False)

# ----------------------------
# LR scheduler presets
# ----------------------------
SCHEDULES = {
    "cosineLR": LRSchedulerFactory(
        name="cosineannealinglr",
        epochs=DEFAULT_NUM_EPOCHS,
        warmup_method="linear",
        warmup_epochs=5, #3
        warmup_decay=0.01,
    ),
    "plateau": LRSchedulerFactory(
        name="reduceonplateau",
        monitor="val_loss",
        mode="min",
        factor=0.1,
        patience=3,
    ),
}

DEFAULT_SCHEDULER_KEY = "cosineLR"  # choose: "cosineLR" or "plateau"
DEFAULT_LR_SCHEDULE = SCHEDULES[DEFAULT_SCHEDULER_KEY]

# --- Paths copied from your old ServerScript_*.py ---
DEFAULT_CSV_PATH = os.getenv("PNEUMONIA_CSV_PATH")
DEFAULT_IMAGE_FOLDER = os.getenv("PNEUMONIA_IMAGE_FOLDER")
DEFAULT_SPLITS_PATH = os.getenv("PNEUMONIA_SPLITS_PATH")

def _ensure_addinverse(x):
    # B-cos expects 6ch: [rgb, 1-rgb]
    if isinstance(x, T.Compose):
        ts = list(x.transforms)
    else:
        ts = [x]

    if not any(isinstance(t, AddInverse) for t in ts):
        ts.append(AddInverse())
    return T.Compose(ts)

def _wrap_transforms(x, add_inverse: bool):
    """Optionally append AddInverse so B-cos models receive 6-channel inputs."""
    if isinstance(x, T.Compose):
        ts = list(x.transforms)
    else:
        ts = [x]

    if add_inverse and not any(isinstance(t, AddInverse) for t in ts):
        ts.append(AddInverse())

    return T.Compose(ts)

def get_pneumonia_transforms(
    augmentation: str,
    crop_size: int = DEFAULT_CROP_SIZE,
    *,
    add_inverse: bool = True,
):
    """
    Uses the exact same augmentations as the old server script, but we
    resize to crop_size so batching is guaranteed.
    """
    augmentation = str(augmentation).lower()

    # Import from your old medical-paper repo
    from bcos.data import augmentations as aug  # adjust import if needed

    if augmentation == "no":
        train_base = aug.get_no_augmentations_resize()
    elif augmentation == "light":
        train_base = aug.get_light_augmentations_resize()
    elif augmentation == "heavy":
        train_base = aug.get_heavy_augmentations_no_rotation_resize()
    else:
        raise ValueError(f"augmentation must be one of: no/light/heavy (got {augmentation!r})")

    # validation always no-aug
    test_base = aug.get_no_augmentations_resize()

    if add_inverse:
        train_t = _ensure_addinverse(train_base)
        test_t  = _ensure_addinverse(test_base)
    else:
        train_t = _wrap_transforms(train_base, add_inverse=add_inverse)
        test_t = _wrap_transforms(test_base, add_inverse=add_inverse)
    return train_t, test_t


DEFAULT_AUG = "no"
_default_train_t, _default_test_t = get_pneumonia_transforms(DEFAULT_AUG)
_default_train_t_baseline, _default_test_t_baseline = get_pneumonia_transforms(DEFAULT_AUG, add_inverse=False)

DEFAULTS = dict(
    data=dict(
        # PneumoniaDataModule-required keys:
        csv_path=DEFAULT_CSV_PATH,
        image_folder=DEFAULT_IMAGE_FOLDER,
        splits_path=DEFAULT_SPLITS_PATH,
        fold_index=0,
        sampling=False,  # oversampling toggle

        # Standard dataloader knobs:
        batch_size=DEFAULT_BATCH_SIZE,
        num_workers=16,
        num_classes=NUM_CLASSES,

        # Transforms:
        train_transform=_default_train_t,
        test_transform=_default_test_t,

        # bookkeeping (optional, but nice for logging):
        augmentation=DEFAULT_AUG,
        crop_size=DEFAULT_CROP_SIZE,
    ),
    model=dict(
        is_bcos=True,
        args=dict(
            num_classes=NUM_CLASSES,
            norm_layer=DEFAULT_NORM_LAYER,
            logit_bias=0.0,  # for 2 classes, -log(1)=0
        ),
        bcos_args=dict(b=2, max_out=1),
    ),
    criterion=UniformOffLabelsBCEWithLogitsLoss(),
    # criterion = BinaryCrossEntropyLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(max_epochs=DEFAULT_NUM_EPOCHS),
    use_agc=True,
)


def update_default(new_config):
    return update_config(DEFAULTS, new_config)


# ----------------------------
# Config grid you can tune
# ----------------------------
RESNET_DEPTHS = [50]
# POOLINGS = ["BlurPool", "FLCPool", "FLCASAPPool", "Bcos", "Baseline"]
POOLINGS = ["BlurPool", "FLCPool", "Bcos", "Baseline"]
WEIGHTS = ["pretrained", "randomInit"]
AUGS = ["no", "light", "heavy"]
SAMPLINGS = [False, True]
FOLDS = [0, 1, 2, 3, 4]  # expand if you want fold sweep

# Which LR scheduler preset(s) to run. Must be keys of SCHEDULES.
SCHEDULER_KEYS = [DEFAULT_SCHEDULER_KEY]
# Example to sweep both:
# SCHEDULER_KEYS = ["cosineLR", "plateau"]
SCHEDULER_KEYS = ["cosineLR"]

# ----------------------------
# LR + batch-size sweep presets
# ----------------------------
LRS = [1e-4]
BATCH_SIZES = [16]


CONFIGS = {}

for depth in RESNET_DEPTHS:
    for pooling in POOLINGS:
        for aug in AUGS:
            train_t, test_t = get_pneumonia_transforms(aug, DEFAULT_CROP_SIZE)
            for sampling in SAMPLINGS:
                for fold_idx in FOLDS:
                    for sched_key in SCHEDULER_KEYS:
                        for weight in WEIGHTS:
                            for lr in LRS:
                                for bs in BATCH_SIZES:
                                    samp_text = "oversamp" if sampling else "nosamp"
                                    w_text = "pretrained" if weight == "pretrained" else "randomInit"
                                    lr_text = (
                                        f"lr{lr:.0e}"
                                        .replace("e-0", "e-")
                                        .replace("e+0", "e+")
                                    )
                                    bs_text = f"bs{bs}"

                                    name = (
                                        f"resnet_{depth}_{pooling}_{w_text}_{lr_text}_{bs_text}_"
                                        f"{aug}_{samp_text}_{sched_key}_fold{fold_idx}"
                                    )

                                    CONFIGS[name] = update_default(
                                        dict(
                                            data=dict(
                                                batch_size=bs,
                                                train_transform=train_t if pooling != "Baseline" else _default_train_t_baseline,
                                                test_transform=test_t if pooling != "Baseline" else _default_test_t_baseline,
                                                augmentation=aug,
                                                sampling=sampling,
                                                fold_index=fold_idx,
                                                csv_path=DEFAULT_CSV_PATH,
                                                image_folder=DEFAULT_IMAGE_FOLDER,
                                                splits_path=DEFAULT_SPLITS_PATH,
                                            ),
                                            model=dict(
                                                name=f"resnet{depth}",
                                                last_layer_name="fc",
                                                weights=f"ResNet{depth}_Weights.DEFAULT" if weight == "pretrained" else None,
                                                bcosify_args=dict(
                                                    fix_b=True,
                                                    use_bias=False,
                                                    norm_layer="BnUncV2",
                                                    manual_optim=False,
                                                    gap=True,
                                                    act_layer=True,
                                                ),
                                                standard_changes={"maxpool": nn.AvgPool2d(kernel_size=3, stride=2, padding=1)},
                                                pooling_type=pooling,
                                            ),
                                            lr_scheduler=SCHEDULES[sched_key],
                                        )
                                    )

CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5, 420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)