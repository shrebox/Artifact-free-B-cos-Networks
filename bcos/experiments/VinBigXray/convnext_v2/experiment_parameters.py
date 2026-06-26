import math  # noqa
import os

import torchvision.transforms as T

from bcos.data.transforms import AddInverse
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

# ----------------------------
# Pneumonia task (binary)
# ----------------------------
NUM_CLASSES = 14
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_CROP_SIZE = 224

DEFAULT_NORM_LAYER = norms.NoBias(norms.BatchNormUncentered2d)

# ----------------------------
# Optimizer + LR sweep presets
# ----------------------------
# OptimizerFactory expects the torch optimizer class name (e.g. "Adam", "AdamW").
# AdamW with weight_decay helps close the gap for modified pooling models,
# matching the regularization used in the ImageNet pretraining recipe.
# OPTIMIZER_NAMES = ["Adam", "AdamW"]

DEFAULT_ADAMW_WEIGHT_DECAY = 0.01


def make_optimizer(name: str, lr: float):
    name_l = str(name).lower()
    if name_l == "adam":
        return OptimizerFactory(name="Adam", lr=lr, bcosify=True, b_opt=False)
    if name_l == "adamw":
        return OptimizerFactory(
            name="AdamW",
            lr=lr,
            bcosify=True,
            b_opt=False,
            weight_decay=DEFAULT_ADAMW_WEIGHT_DECAY,
        )
    raise ValueError(f"Unsupported optimizer name: {name!r}")


DEFAULT_OPTIMIZER = make_optimizer("Adam", DEFAULT_LR)

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
    # Longer warmup (5 epochs) helps when pooling layers start from copied
    # (but distribution-shifted) weights -- gives more time to calibrate.
    "cosineLR_warmup5": LRSchedulerFactory(
        name="cosineannealinglr",
        epochs=DEFAULT_NUM_EPOCHS,
        warmup_method="linear",
        warmup_epochs=5,
        warmup_decay=0.01,
    ),
    # Extended training (50 epochs) for modified pooling models that need
    # more time for the new pooling operators to fully co-adapt with the
    # pretrained backbone features.
    "cosineLR_50ep": LRSchedulerFactory(
        name="cosineannealinglr",
        epochs=50,
        warmup_method="linear",
        warmup_epochs=5,
        warmup_decay=0.01,
    ),
}

DEFAULT_SCHEDULER_KEY = "cosineLR"  # choose: "cosineLR" or "plateau"
DEFAULT_LR_SCHEDULE = SCHEDULES[DEFAULT_SCHEDULER_KEY]

DEFAULT_CSV_PATH = os.getenv("VINBIG_CSV_PATH")
DEFAULT_IMAGE_FOLDER = os.getenv("VINBIG_IMAGE_FOLDER")
DEFAULT_SPLITS_PATH = os.getenv("VINBIG_SPLITS_PATH")


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
    criterion=UniformOffLabelsBCEWithLogitsLoss(),
    test_criterion=BinaryCrossEntropyLoss(),
    optimizer=DEFAULT_OPTIMIZER,
    lr_scheduler=DEFAULT_LR_SCHEDULE,
    trainer=dict(max_epochs=DEFAULT_NUM_EPOCHS),
    # EMA stabilizes training and is used in the ImageNet pretraining recipe.
    # Particularly helpful for modified pooling models where the new pooling
    # operators need time to co-adapt with pretrained features.
    ema=dict(
        steps=32,
        decay=0.99998,
    ),
    use_agc=True,
)


def update_default(new_config):
    return update_config(DEFAULTS, new_config)


# ----------------------------
# Config grid you can tune (ConvNeXt + pooling sweep)
# ----------------------------
# CONVNEXT_SIZES = ["base", "tiny"]
# POOLINGS = ["BlurPool", "FLCPool", "Bcos", "Baseline"]
# Optionally: sweep pretrained vs random init
# WEIGHTS = ["pretrained", "randomInit"]
# AUGS = ["no", "light", "heavy"]
# SAMPLINGS = [False, True]
# FOLDS = [0, 1, 2, 3, 4]  # expand if you want fold sweep

# Arch type sweep: "pn" = model default pos-norm, "bnu" = our BatchNormUncentered
# ARCH_TYPES = ["pn", "bnu"]

# Which LR scheduler preset(s) to run. Must be keys of SCHEDULES.
# Example to sweep both:
# SCHEDULER_KEYS = ["cosineLR", "cosineLR_warmup5", "cosineLR_50ep"]

# Map scheduler keys to their max_epochs (for trainer config).
# If not listed, DEFAULT_NUM_EPOCHS is used.
# SCHED_EPOCHS = {
#     "cosineLR_50ep": 50,
# }

# LR sweep
# LRS = [1e-2, 1e-3, 1e-4]

# Batch-size sweep
# BATCH_SIZES = [16, 32]

# for size in CONVNEXT_SIZES:
#     for pooling in POOLINGS:
#         for arch_type in ARCH_TYPES:
#             for aug in AUGS:
#                 train_t, test_t = get_vinbig_transforms(
#                     aug, DEFAULT_CROP_SIZE, add_inverse=True
#                 )
#                 for sampling in SAMPLINGS:
#                     for fold_idx in FOLDS:
#                         for sched_key in SCHEDULER_KEYS:
#                             for weight in WEIGHTS:
#                                 for opt_name in OPTIMIZER_NAMES:
#                                     for lr in LRS:
#                                         for bs in BATCH_SIZES:
#                                             samp_text = (
#                                                 "oversamp" if sampling else "nosamp"
#                                             )
#                                             w_text = (
#                                                 "pretrained"
#                                                 if weight == "pretrained"
#                                                 else "randomInit"
#                                             )

#                                             opt_text = opt_name.lower()
#                                             lr_text = f"lr{lr:.0e}".replace(
#                                                 "e-0", "e-"
#                                             ).replace("e+0", "e+")
#                                             bs_text = f"bs{bs}"

#                                             if pooling == "Baseline":
#                                                 name = (
#                                                     f"convnext_{size}_{pooling}_{w_text}_{opt_text}_{lr_text}_"
#                                                     f"{bs_text}_{aug}_{samp_text}_{sched_key}_fold{fold_idx}"
#                                                 )
#                                             else:
#                                                 tokens = [
#                                                     f"convnext_{size}",
#                                                     arch_type,
#                                                     pooling,
#                                                     w_text,
#                                                     opt_text,
#                                                     lr_text,
#                                                     bs_text,
#                                                     aug,
#                                                     samp_text,
#                                                     sched_key,
#                                                     f"fold{fold_idx}",
#                                                 ]
#                                                 name = "_".join(tokens)

# Use scheduler-specific epochs if defined
#                                             max_ep = SCHED_EPOCHS.get(
#                                                 sched_key, DEFAULT_NUM_EPOCHS
#                                             )

#                                             CONFIGS[name] = update_default(
#                                                 dict(
#                                                     data=dict(
#                                                         batch_size=bs,
#                                                         train_transform=train_t
#                                                         if pooling != "Baseline"
#                                                         else _default_train_t_baseline,
#                                                         test_transform=test_t
#                                                         if pooling != "Baseline"
#                                                         else _default_test_t_baseline,
#                                                         augmentation=aug,
#                                                         sampling=sampling,
#                                                         fold_index=fold_idx,
#                                                         csv_path=DEFAULT_CSV_PATH,
#                                                         image_folder=DEFAULT_IMAGE_FOLDER,
#                                                         splits_path=DEFAULT_SPLITS_PATH,
#                                                     ),
#                                                     model=dict(
#                                                         is_bcos=False
#                                                         if pooling == "Baseline"
#                                                         else True,
#                                                         name=f"convnext_{size}",
#                                                         last_layer_name="fc",
#                                                         weights=(
#                                                             "DEFAULT"
#                                                             if weight == "pretrained"
#                                                             else None
#                                                         ),
#                                                         args=dict(
#                                                             norm_layer=(
#                                                                 DEFAULT_NORM_LAYER
#                                                                 if pooling == "Baseline"
#                                                                 else (
#                                                                     None
#                                                                     if arch_type == "pn"
#                                                                     else DEFAULT_NORM_LAYER
#                                                                 )
#                                                             ),
#                                                         ),
#                                                         pooling_type=pooling,
#                                                         bcosify_args=dict(
#                                                             fix_b=True,
#                                                             use_bias=False,
#                                                             norm_layer="BnUncV2",
#                                                             manual_optim=False,
#                                                             gap=True,
#                                                             act_layer=True,
#                                                         ),
#                                                     ),
#                                                     optimizer=make_optimizer(
#                                                         opt_name, lr
#                                                     ),
#                                                     lr_scheduler=SCHEDULES[sched_key],
#                                                     trainer=dict(max_epochs=max_ep),
#                                                 )
#                                             )

# ============================================================
# Focused ablation recipes for modified-pooling improvement
# ============================================================
# Each "step" adds one change on top of the previous, so results
# are directly comparable.
#
# All recipes fix:
#   pretrained, bnu, no oversampling, lr=1e-3, bs=16,
#   UniformOffLabelsBCE
#
# Augmentation is swept: light and heavy.
#
# Weight copying (model.py fix) is active for every BlurPool /
# FLCPool config.  "Bcos" is included as the reference ceiling.
#
# Experiment-name pattern:
#   convnext_{size}_bnu_{pooling}_pretrained_{recipe}_{aug}_nosamp_fold{f}
# ============================================================

RECIPE_LR = 1e-3
RECIPE_BS = 16
# RECIPE_AUGS = ["light", "heavy"]
RECIPE_AUGS = ["light"]
RECIPE_SAMPLING = False

RECIPES = {
    # Step 1: weight-copy fix + Adam (your current best, now with correct init)
    "step1_adam_30ep": dict(
        optimizer=make_optimizer("Adam", RECIPE_LR),
        lr_scheduler=SCHEDULES["cosineLR"],
        trainer=dict(max_epochs=30),
    ),
    # Step 2: switch to AdamW with weight decay (matches ImageNet recipe)
    #     "step2_adamw_30ep": dict(
    #         optimizer=make_optimizer("AdamW", RECIPE_LR),
    #         lr_scheduler=SCHEDULES["cosineLR"],
    #         trainer=dict(max_epochs=30),
    #     ),
    # Step 3: Adam + longer warmup (5 epochs instead of 3)
    #     "step3_adam_warmup5_30ep": dict(
    #         optimizer=make_optimizer("Adam", RECIPE_LR),
    #         lr_scheduler=SCHEDULES["cosineLR_warmup5"],
    #         trainer=dict(max_epochs=30),
    #     ),
    # Step 4: AdamW + longer warmup
    "step4_adamw_warmup5_30ep": dict(
        optimizer=make_optimizer("AdamW", RECIPE_LR),
        lr_scheduler=SCHEDULES["cosineLR_warmup5"],
        trainer=dict(max_epochs=30),
    ),
    # Step 5: AdamW + extended training (50 epochs, warmup 5)
    #     "step5_adamw_warmup5_50ep": dict(
    #         optimizer=make_optimizer("AdamW", RECIPE_LR),
    #         lr_scheduler=SCHEDULES["cosineLR_50ep"],
    #         trainer=dict(max_epochs=50),
    #     ),
}

RECIPE_POOLINGS = ["BlurPool", "FLCPool", "Bcos", "Baseline"]
RECIPE_SIZES = ["base", "tiny"]
RECIPE_FOLDS = [0, 1, 2, 3, 4]

CONFIGS = {}
for recipe_name, recipe_overrides in RECIPES.items():
    for size in RECIPE_SIZES:
        # Paper used only base->step1_adam_30ep and tiny->step4_adamw_warmup5_30ep
        if (size, recipe_name) not in {
            ("base", "step1_adam_30ep"),
            ("tiny", "step4_adamw_warmup5_30ep"),
        }:
            continue
        for pooling in RECIPE_POOLINGS:
            is_baseline = pooling == "Baseline"
            for aug in RECIPE_AUGS:
                _r_train_t, _r_test_t = get_vinbig_transforms(
                    aug, DEFAULT_CROP_SIZE, add_inverse=not is_baseline
                )
                for fold_idx in RECIPE_FOLDS:
                    # Baseline has no bnu token (standard torchvision model)
                    if is_baseline:
                        exp_name = (
                            f"convnext_{size}_{pooling}_pretrained_"
                            f"{recipe_name}_{aug}_nosamp_fold{fold_idx}"
                        )
                    else:
                        exp_name = (
                            f"convnext_{size}_bnu_{pooling}_pretrained_"
                            f"{recipe_name}_{aug}_nosamp_fold{fold_idx}"
                        )

                    CONFIGS[exp_name] = update_default(
                        dict(
                            data=dict(
                                batch_size=RECIPE_BS,
                                train_transform=_r_train_t,
                                test_transform=_r_test_t,
                                augmentation=aug,
                                sampling=RECIPE_SAMPLING,
                                fold_index=fold_idx,
                                csv_path=DEFAULT_CSV_PATH,
                                image_folder=DEFAULT_IMAGE_FOLDER,
                                splits_path=DEFAULT_SPLITS_PATH,
                            ),
                            model=dict(
                                is_bcos=not is_baseline,
                                name=f"convnext_{size}",
                                last_layer_name="fc",
                                weights="DEFAULT",
                                args=dict(norm_layer=DEFAULT_NORM_LAYER),
                                pooling_type=pooling,
                                bcosify_args=dict(
                                    fix_b=True,
                                    use_bias=False,
                                    norm_layer="BnUncV2",
                                    manual_optim=False,
                                    gap=True,
                                    act_layer=True,
                                ),
                            ),
                            **recipe_overrides,
                        )
                    )

# ============================================================
# Seed expansion (applies to ALL configs above: grid + recipes)
# ============================================================
CONFIGS.update(create_configs_with_different_seeds(CONFIGS, seeds=[5, 420, 1337]))

if __name__ == "__main__":
    configs_cli(CONFIGS)
