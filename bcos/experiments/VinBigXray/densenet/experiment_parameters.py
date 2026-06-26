import math  # noqa
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

# Optional: custom BCE-style losses used in the B-cos medical codebase.
# We keep these imports guarded so this file still runs even if the classes
# are moved/renamed in different branches.
try:
    from bcos.modules.losses import (  # type: ignore
        BinaryCrossEntropyLoss,
        UniformOffLabelsBCEWithLogitsLoss,
    )
except Exception:  # pragma: no cover
    UniformOffLabelsBCEWithLogitsLoss = None  # type: ignore
    BinaryCrossEntropyLoss = None  # type: ignore

__all__ = ["CONFIGS"]

# ----------------------------
# VinBigXray task (multi-label)
# ----------------------------
NUM_CLASSES = 14
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
    # weight_decay=1e-3,
)
LRS = [1e-4]
# BATCH_SIZES = [16, 32, 64]
BATCH_SIZES = [64]


def make_optimizer(lr: float):
    return OptimizerFactory(name="Adam", lr=lr, bcosify=True, b_opt=False)


DEFAULT_CSV_PATH = os.getenv("VINBIG_CSV_PATH")
DEFAULT_IMAGE_FOLDER = os.getenv("VINBIG_IMAGE_FOLDER")
DEFAULT_SPLITS_PATH = os.getenv("VINBIG_SPLITS_PATH")

# ----------------------------
# LR scheduler presets
# ----------------------------
SCHEDULES = {
    "cosineLR": LRSchedulerFactory(
        name="cosineannealinglr",
        epochs=DEFAULT_NUM_EPOCHS,
        warmup_method="linear",
        warmup_epochs=3,
        warmup_decay=0.01,
    ),
    "plateau": LRSchedulerFactory(
        name="reduceonplateau",
        monitor="val_mAP",
        mode="min",
        factor=0.1,
        patience=3,
    ),
}

DEFAULT_SCHEDULER_KEY = "cosineLR"  # choose: "cosineLR" or "plateau"
DEFAULT_LR_SCHEDULE = SCHEDULES[DEFAULT_SCHEDULER_KEY]


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


def get_vinbig_transforms(
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

    # Reuse the same resize-aug interface as your old scripts.
    # If you keep your augmentations elsewhere, adjust this import accordingly.
    from bcos.data import augmentations as aug

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

    if add_inverse:
        train_t = _ensure_addinverse(train_base)
        test_t = _ensure_addinverse(test_base)
    else:
        train_t = _wrap_transforms(train_base, add_inverse=add_inverse)
        test_t = _wrap_transforms(test_base, add_inverse=add_inverse)
    return train_t, test_t


DEFAULT_AUG = "no"
_default_train_t, _default_test_t = get_vinbig_transforms(DEFAULT_AUG)
_default_train_t_baseline, _default_test_t_baseline = get_vinbig_transforms(
    DEFAULT_AUG, add_inverse=False
)

DEFAULTS = dict(
    data=dict(
        # Medical multi-label DataModule-required keys:
        csv_path=DEFAULT_CSV_PATH,
        image_folder=DEFAULT_IMAGE_FOLDER,
        splits_path=DEFAULT_SPLITS_PATH,
        fold_index=0,
        sampling=False,  # oversampling toggle
        multilabel=True,  # important for downstream logic (metrics, loss, etc.)
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
            # Multi-label training uses independent sigmoid heads; keep logits unbiased by default.
            logit_bias=-math.log(NUM_CLASSES - 1),
        ),
        bcos_args=dict(b=2, max_out=1),
    ),
    criterion=nn.BCEWithLogitsLoss(),
    test_criterion=nn.BCEWithLogitsLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(max_epochs=DEFAULT_NUM_EPOCHS),
    use_agc=True,
)


# ----------------------------
# Loss sweep
# ----------------------------
# NOTE: VinBigXray is multi-label. BCE-style losses are appropriate.
# CrossEntropyLoss assumes a single class index per sample.
# LOSS_KEYS = ["bce", "offlogitbce"]
LOSS_KEYS = ["offlogitbce"]


def make_losses(loss_key: str):
    """Return (criterion, test_criterion) for the given key."""
    lk = str(loss_key).lower()
    if lk == "bce":
        return nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()

    # if lk == "ce":
    #     # Only valid for single-label setups (targets are class indices).
    #     return nn.CrossEntropyLoss(), nn.CrossEntropyLoss()

    if lk == "offlogitbce":
        if UniformOffLabelsBCEWithLogitsLoss is None:
            raise ImportError(
                "UniformOffLabelsBCEWithLogitsLoss not found. "
                "Update the import path in experiment_parameters.py."
            )
        crit = UniformOffLabelsBCEWithLogitsLoss()
        if BinaryCrossEntropyLoss is not None:
            test_crit = BinaryCrossEntropyLoss()
        else:
            test_crit = nn.BCEWithLogitsLoss()
        return crit, test_crit

    raise ValueError(f"Unknown loss_key={loss_key!r}. Expected one of {LOSS_KEYS}.")


def update_default(new_config):
    return update_config(DEFAULTS, new_config)


# ----------------------------
# Config grid you can tune
# ----------------------------
DENSENET_DEPTHS = [121]
POOLINGS = ["BlurPool", "FLCPool", "Bcos", "Baseline"]
# WEIGHTS = ["pretrained", "randomInit"]
WEIGHTS = ["pretrained"]
AUGS = ["no", "light", "heavy"]
SAMPLINGS = [False, True]
FOLDS = [0, 1, 2, 3, 4]  # expand if you want fold sweep

# Which LR scheduler preset(s) to run. Must be keys of SCHEDULES.
SCHEDULER_KEYS = [DEFAULT_SCHEDULER_KEY]
# Example to sweep both:
SCHEDULER_KEYS = ["cosineLR"]

CONFIGS = {}

for depth in DENSENET_DEPTHS:
    for pooling in POOLINGS:
        for aug in AUGS:
            train_t, test_t = get_vinbig_transforms(aug, DEFAULT_CROP_SIZE)
            for sampling in SAMPLINGS:
                for fold_idx in FOLDS:
                    for sched_key in SCHEDULER_KEYS:
                        for weight in WEIGHTS:
                            for loss_key in LOSS_KEYS:
                                for lr in LRS:
                                    for bs in BATCH_SIZES:
                                        samp_text = "oversamp" if sampling else "nosamp"
                                        w_text = (
                                            "pretrained"
                                            if weight == "pretrained"
                                            else "randomInit"
                                        )
                                        lr_text = f"lr{lr:.0e}".replace(
                                            "e-0", "e-"
                                        ).replace("e+0", "e+")
                                        bs_text = f"bs{bs}"

                                        criterion, test_criterion = make_losses(
                                            loss_key
                                        )

                                        name = (
                                            f"densenet_{depth}_{pooling}_{w_text}_{lr_text}_{bs_text}_"
                                            f"{aug}_{samp_text}_{sched_key}_{loss_key}_fold{fold_idx}"
                                        )

                                        CONFIGS[name] = update_default(
                                            dict(
                                                data=dict(
                                                    batch_size=bs,
                                                    train_transform=train_t
                                                    if pooling != "Baseline"
                                                    else _default_train_t_baseline,
                                                    test_transform=test_t
                                                    if pooling != "Baseline"
                                                    else _default_test_t_baseline,
                                                    augmentation=aug,
                                                    sampling=sampling,
                                                    fold_index=fold_idx,
                                                    csv_path=DEFAULT_CSV_PATH,
                                                    image_folder=DEFAULT_IMAGE_FOLDER,
                                                    splits_path=DEFAULT_SPLITS_PATH,
                                                    multilabel=(loss_key != "ce"),
                                                ),
                                                model=dict(
                                                    is_bcos=(pooling != "Baseline"),
                                                    name=f"densenet{depth}",
                                                    last_layer_name="fc",
                                                    weights=f"DenseNet{depth}_Weights.DEFAULT",
                                                    bcosify_args=dict(
                                                        fix_b=True,
                                                        use_bias=False,
                                                        norm_layer="BnUncV2",
                                                        manual_optim=False,
                                                        gap=True,
                                                        act_layer=True,
                                                    ),
                                                    pooling_type=pooling,
                                                ),
                                                optimizer=make_optimizer(lr),
                                                criterion=criterion,
                                                test_criterion=test_criterion,
                                                lr_scheduler=SCHEDULES[sched_key],
                                            )
                                        )

CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5, 420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
