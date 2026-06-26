import os

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

__all__ = ["CONFIGS"]

# ----------------------------
# Pneumonia task (binary)
# ----------------------------
NUM_CLASSES = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 30
DEFAULT_LR = 1e-4
DEFAULT_CROP_SIZE = 224

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)

DEFAULT_OPTIMIZER = OptimizerFactory(
    name="Adam",
    lr=DEFAULT_LR,
    bcosify=True,
    b_opt=False,
)

# Baseline optimizer defaults (weight_decay is swept below)
DEFAULT_BASELINE_OPT_NAME = "Adamw"
DEFAULT_BASELINE_WEIGHT_DECAY = 1e-3
DEFAULT_OPTIMIZER_BASELINE = OptimizerFactory(
    name=DEFAULT_BASELINE_OPT_NAME,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_BASELINE_WEIGHT_DECAY,
    bcosify=False,
    b_opt=False,
)
DEFAULT_LR_SCHEDULE = LRSchedulerFactory(
    name="reduceonplateau",
    monitor="val_loss",
    mode="min",
    factor=0.1,
    patience=3,
)

# --- Paths copied from your old ServerScript_*.py ---
DEFAULT_CSV_PATH = os.getenv("VINBIG_CSV_PATH")
DEFAULT_IMAGE_FOLDER = os.getenv("VINBIG_IMAGE_FOLDER")
DEFAULT_SPLITS_PATH = os.getenv("VINBIG_SPLITS_PATH")


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
        raise ValueError(
            f"augmentation must be one of: no/light/heavy (got {augmentation!r})"
        )

    # validation always no-aug
    test_base = aug.get_no_augmentations_resize()

    train_t = _wrap_transforms(train_base, add_inverse=add_inverse)
    test_t = _wrap_transforms(test_base, add_inverse=add_inverse)
    return train_t, test_t


DEFAULT_AUG = "no"
_default_train_t, _default_test_t = get_pneumonia_transforms(
    DEFAULT_AUG, add_inverse=True
)
_default_train_t_baseline, _default_test_t_baseline = get_pneumonia_transforms(
    DEFAULT_AUG, add_inverse=False
)

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
    criterion=nn.CrossEntropyLoss(),
    test_criterion=nn.CrossEntropyLoss(),
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
AUGS = ["no", "light", "heavy"]
SAMPLINGS = [False, True]
FOLDS = [0, 1, 2, 3, 4]  # expand if you want fold sweep

# Sweep baseline weight decay values here (set to [DEFAULT_BASELINE_WEIGHT_DECAY] to disable sweep)
WEIGHT_DECAYS = [0.0, 1e-3]

CONFIGS = {}

# ----------------------------
# Baseline (non-B-cos) configs
# ----------------------------
for depth in RESNET_DEPTHS:
    for aug in AUGS:
        train_t, test_t = get_pneumonia_transforms(
            aug, DEFAULT_CROP_SIZE, add_inverse=False
        )
        for sampling in SAMPLINGS:
            for fold_idx in FOLDS:
                for wd in WEIGHT_DECAYS:
                    samp_text = "oversamp" if sampling else "nosamp"
                    wd_text = f"wd{wd:g}".replace("-", "m").replace(".", "p")
                    name = f"resnet_{depth}_baseline_{aug}_{samp_text}_{wd_text}_fold{fold_idx}"

                    baseline_optimizer = OptimizerFactory(
                        name=DEFAULT_BASELINE_OPT_NAME,
                        lr=DEFAULT_LR,
                        weight_decay=wd,
                        bcosify=False,
                        b_opt=False,
                    )

                    CONFIGS[name] = update_default(
                        dict(
                            data=dict(
                                train_transform=train_t,
                                test_transform=test_t,
                                augmentation=aug,
                                sampling=sampling,
                                fold_index=fold_idx,
                                csv_path=DEFAULT_CSV_PATH,
                                image_folder=DEFAULT_IMAGE_FOLDER,
                                splits_path=DEFAULT_SPLITS_PATH,
                            ),
                            model=dict(
                                # Baseline models use 3-channel input and no AddInverse
                                is_bcos=False,
                                name=f"resnet{depth}",
                                num_classes=NUM_CLASSES,
                                weights=f"ResNet{depth}_Weights.DEFAULT",
                            ),
                            optimizer=baseline_optimizer,
                        )
                    )

CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5, 420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
