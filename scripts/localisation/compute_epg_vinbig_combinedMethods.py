#!/usr/bin/env python3
"""
EPG (Energy-based Pointing Game) — VinBigXray multilabel variant.

Evaluates ALL explanation methods for a single experiment in ONE pass,
computing **per-class** EPG metrics for each of the 14 VinBigXray classes.

Key differences from the Pneumonia pipeline:
  - Multilabel (14 independent classes), not binary
  - Per-class contribution maps: one gradient/CAM per class
  - Per-class bbox filtering: ONLY bboxes of the target class are used
  - Multiple annotators: union of all annotators' bboxes per class (default)
  - Sigmoid predictions with 0.5 threshold for TP/FN classification
  - Per-label metrics + macro-averaged aggregate in output JSON
  - Bbox coordinates are already at 224×224 in train224.csv (no rescaling)

Output: per-method JSON files in ``<experiment>/epg_results/`` with the
same directory convention as the Pneumonia pipeline.

Usage:
  # Single experiment, all auto-detected methods:
  python compute_epg_vinbig_combinedMethods.py --experiment_paths /path/to/exp1

  # Multiple experiments, specific methods:
  python compute_epg_vinbig_combinedMethods.py \\
      --experiment_paths /path/to/exp1 /path/to/exp2 \\
      --methods bcos gradcam layercam

  # With threshold sweep and sample limit:
  python compute_epg_vinbig_combinedMethods.py \\
      --experiment_paths /path/to/exp1 \\
      --thresholds 0.0 0.2 0.4 0.6 0.8 \\
      --max_samples 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from bcos.experiments.utils import Experiment  # noqa: E402

# CAM imports (optional — gracefully degrade if not installed)
try:
    from pytorch_grad_cam import (
        AblationCAM,
        FinerCAM,
        GradCAM,
        GradCAMPlusPlus,
        HiResCAM,
        LayerCAM,
    )
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 14
SIGMOID_THRESHOLD = 0.5  # threshold for "predicted positive"

VINBIG_CLASSES = [
    "Aortic Enlargement",  # 0
    "Atelectasis",  # 1
    "Calcification",  # 2
    "Cardiomegaly",  # 3
    "Consolidation",  # 4
    "ILD",  # 5
    "Infiltration",  # 6
    "Lung Opacity",  # 7
    "Nodule/Mass",  # 8
    "Other lesion",  # 9
    "Pleural Effusion",  # 10
    "Pleural Thickening",  # 11
    "Pneumothorax",  # 12
    "Pulmonary Fibrosis",  # 13
]

# Map of supported CAM methods (name -> class)
CAM_REGISTRY = {}
if HAS_GRAD_CAM:
    CAM_REGISTRY = {
        "gradcam": GradCAM,
        "layercam": LayerCAM,
        "hirescam": HiResCAM,
        "gradcampp": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "finercam": FinerCAM,
    }

ALL_METHODS = ["bcos"] + list(CAM_REGISTRY.keys()) + ["gradxinput"]

_UNDEFINED = float("nan")


# ============================================================================
# EPG core functions (same as Pneumonia pipeline — faithful to paper Eqs. 2-3)
# ============================================================================
def epg_precision_single_box(
    contribution_map: torch.Tensor,
    bbox_xyxy: Tuple[int, int, int, int],
    threshold: float = 0.0,
) -> float:
    """EPG Precision for a single bounding box (Eq. 2)."""
    cm = contribution_map.clone()
    cm[cm < 0] = 0
    if threshold > 0.0:
        max_val = cm.max()
        if max_val > 0:
            cm[cm < threshold * max_val] = 0
    total_energy = cm.sum().item()
    if total_energy < 1e-7:
        return _UNDEFINED
    x1, y1, x2, y2 = bbox_xyxy
    bbox_energy = cm[y1:y2, x1:x2].sum().item()
    return bbox_energy / total_energy


def epg_recall_single_box(
    contribution_map: torch.Tensor,
    bbox_xyxy: Tuple[int, int, int, int],
    threshold: float = 0.0,
) -> float:
    """EPG Recall for a single bounding box (Eq. 3)."""
    cm = contribution_map.clone()
    x1, y1, x2, y2 = bbox_xyxy
    bbox_region = cm[y1:y2, x1:x2]
    positive_in_bbox = bbox_region.clone()
    positive_in_bbox[positive_in_bbox < 0] = 0
    negative_in_bbox = bbox_region.clone()
    negative_in_bbox[negative_in_bbox > 0] = 0
    if threshold > 0.0:
        cm_pos = cm.clone()
        cm_pos[cm_pos < 0] = 0
        global_max = cm_pos.max()
        if global_max > 0:
            positive_in_bbox[positive_in_bbox < threshold * global_max] = 0
    sum_pos = positive_in_bbox.sum().item()
    sum_abs_neg = negative_in_bbox.abs().sum().item()
    denom = sum_pos + sum_abs_neg
    if denom < 1e-7:
        return _UNDEFINED
    return sum_pos / denom


def epg_precision_union(
    contribution_map: torch.Tensor,
    bboxes: List[Tuple[int, int, int, int]],
    threshold: float = 0.0,
) -> float:
    """EPG Precision over the union of all bounding boxes."""
    cm = contribution_map.clone()
    cm[cm < 0] = 0
    if threshold > 0.0:
        max_val = cm.max()
        if max_val > 0:
            cm[cm < threshold * max_val] = 0
    total_energy = cm.sum().item()
    if total_energy < 1e-7:
        return _UNDEFINED
    union_mask = torch.zeros_like(cm, dtype=torch.bool)
    for x1, y1, x2, y2 in bboxes:
        union_mask[y1:y2, x1:x2] = True
    union_energy = cm[union_mask].sum().item()
    return union_energy / total_energy


def epg_recall_union(
    contribution_map: torch.Tensor,
    bboxes: List[Tuple[int, int, int, int]],
    threshold: float = 0.0,
) -> float:
    """EPG Recall over the union of all bounding boxes."""
    cm = contribution_map.clone()
    union_mask = torch.zeros(cm.shape, dtype=torch.bool, device=cm.device)
    for x1, y1, x2, y2 in bboxes:
        union_mask[y1:y2, x1:x2] = True
    union_region = cm[union_mask]
    positive = union_region.clone()
    positive[positive < 0] = 0
    negative = union_region.clone()
    negative[negative > 0] = 0
    if threshold > 0.0:
        cm_pos = cm.clone()
        cm_pos[cm_pos < 0] = 0
        global_max = cm_pos.max()
        if global_max > 0:
            positive[positive < threshold * global_max] = 0
    sum_pos = positive.sum().item()
    sum_abs_neg = negative.abs().sum().item()
    denom = sum_pos + sum_abs_neg
    if denom < 1e-7:
        return _UNDEFINED
    return sum_pos / denom


def bbox_iou(
    contribution_map: torch.Tensor,
    bboxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> float:
    """BBox IoU between binarised attribution map and GT boxes."""
    cm = contribution_map.clone()
    cm[cm < 0] = 0
    cm_min = cm.min()
    cm_max = cm.max()
    if (cm_max - cm_min).abs() < 1e-7:
        if cm_max.item() == 0:
            return _UNDEFINED
        cm = cm / cm_max
    else:
        cm = (cm - cm_min) / (cm_max - cm_min)
    attr_bin = cm > iou_threshold
    bb_mask = torch.zeros_like(cm, dtype=torch.bool)
    for x1, y1, x2, y2 in bboxes:
        bb_mask[y1:y2, x1:x2] = True
    intersection = (attr_bin & bb_mask).sum().item()
    union = (attr_bin | bb_mask).sum().item()
    if union == 0:
        return _UNDEFINED
    return intersection / union


def _nanmean(xs: List[float]) -> float:
    defined = [v for v in xs if not np.isnan(v)]
    return float(np.mean(defined)) if defined else _UNDEFINED


def compute_epg_for_image(
    contribution_map: torch.Tensor,
    bboxes: List[Tuple[int, int, int, int]],
    threshold: float = 0.0,
    iou_threshold: float = 0.5,
) -> Dict:
    """Compute all localisation metrics for one image + one class."""
    nan = _UNDEFINED
    empty = {
        "perbox_precision": nan,
        "perbox_recall": nan,
        "union_precision": nan,
        "union_recall": nan,
        "bbox_iou": nan,
    }
    if len(bboxes) == 0:
        return empty

    precisions = []
    recalls = []
    for bbox in bboxes:
        precisions.append(
            epg_precision_single_box(contribution_map, bbox, threshold=threshold)
        )
        recalls.append(
            epg_recall_single_box(contribution_map, bbox, threshold=threshold)
        )

    return {
        "perbox_precision": _nanmean(precisions),
        "perbox_recall": _nanmean(recalls),
        "union_precision": epg_precision_union(
            contribution_map, bboxes, threshold=threshold
        ),
        "union_recall": epg_recall_union(contribution_map, bboxes, threshold=threshold),
        "bbox_iou": bbox_iou(contribution_map, bboxes, iou_threshold=iou_threshold),
    }


# ============================================================================
# Model helpers
# ============================================================================
def is_bcos_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "explanation_mode")


def get_cam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Auto-detect the CAM target layer for ResNet / ConvNeXt / DenseNet.

    DenseNet's ``model.features`` ends with a norm layer (``norm5``), so
    ``features[-1]`` is a ``BatchNorm2d`` which is NOT subscriptable.
    The correct target is ``features.denseblock4`` (= ``features[-2]``).

    ConvNeXt's ``model.features[-1]`` is a ``Sequential`` of ``CNBlock``
    modules, so ``features[-1][-1]`` works.
    """
    # ResNet family: model.layer4[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    elif hasattr(model, "features"):
        # DenseNet: features has named children; look for the last denseblock
        named = dict(model.features.named_children())
        if "denseblock4" in named:
            target = named["denseblock4"]
            print(
                f"  [{type(model).__name__}] CAM target layer: "
                f"features.denseblock4 ({type(target).__name__})"
            )
            return target
        # ConvNeXt: features[-1] is a Sequential of CNBlocks
        last = model.features[-1]
        if hasattr(last, "__getitem__"):
            target = last[-1]
            print(
                f"  [{type(model).__name__}] CAM target layer: "
                f"features[-1][-1] ({type(target).__name__})"
            )
            return target
        # features[-1] is a single module (unknown arch) — use it directly
        print(
            f"  [{type(model).__name__}] CAM target layer: "
            f"features[-1] ({type(last).__name__})"
        )
        return last
    else:
        # Fallback: last Conv2d
        last = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("Cannot find a Conv2d target layer for CAM.")
        return last


# ============================================================================
# Per-class contribution map computation
# ============================================================================
def get_contribution_maps_multilabel(
    model: torch.nn.Module,
    method: str,
    img: torch.Tensor,  # [C, H, W] single image on device
    class_indices: List[int],  # which class logits to compute maps for
    cam_obj=None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[int, torch.Tensor], List[float], List[float]]:
    """
    Compute 2D contribution maps for MULTIPLE classes (multilabel).

    Returns:
      maps:     {class_idx: [H, W] tensor}
      logits:   [NUM_CLASSES] raw logit values
      sigmoids: [NUM_CLASSES] sigmoid probabilities
    """
    maps = {}

    if method == "bcos":
        with torch.enable_grad(), model.explanation_mode():
            x = img.unsqueeze(0).requires_grad_(True)
            out = model(x)  # [1, NUM_CLASSES]
            logits = out.detach().cpu()[0].tolist()
            sigmoids = torch.sigmoid(out.detach())[0].cpu().tolist()

            for i, cls in enumerate(class_indices):
                grad = torch.autograd.grad(
                    out[0, cls],
                    x,
                    retain_graph=(i < len(class_indices) - 1),
                    create_graph=False,
                )[0]
                # Contribution map = (input * gradient).sum(channel_dim)
                cm = (img * grad.squeeze(0)).sum(dim=0).detach()
                maps[cls] = cm

    elif method == "gradxinput":
        x = img.unsqueeze(0).requires_grad_(True)
        out = model(x)
        logits = out.detach().cpu()[0].tolist()
        sigmoids = torch.sigmoid(out.detach())[0].cpu().tolist()

        for i, cls in enumerate(class_indices):
            model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad.zero_()
            out[0, cls].backward(
                inputs=[x],
                retain_graph=(i < len(class_indices) - 1),
            )
            grad = x.grad.detach().squeeze(0)
            cm = (img.detach() * grad).sum(dim=0)
            maps[cls] = cm

    elif method in CAM_REGISTRY:
        if cam_obj is None:
            raise RuntimeError(f"CAM object not initialized for method '{method}'")
        with torch.no_grad():
            out = model(img.unsqueeze(0))
        logits = out.detach().cpu()[0].tolist()
        sigmoids = torch.sigmoid(out.detach())[0].cpu().tolist()

        for cls in class_indices:
            x = img.unsqueeze(0)  # fresh tensor each time for CAM
            if method == "finercam":
                # FinerCAM requires targets=None to activate its own
                # FinerWeightedTarget (contrastive loss).  Passing explicit
                # ClassifierOutputTarget makes it fall back to vanilla GradCAM.
                cam_np = cam_obj(input_tensor=x, targets=None)
            else:
                targets = [ClassifierOutputTarget(cls)]
                cam_np = cam_obj(input_tensor=x, targets=targets)  # [B, H_cam, W_cam]
            cm = torch.from_numpy(cam_np[0]).to(device).float()
            # Resize to input res if needed
            if cm.shape != img.shape[1:]:
                cm = (
                    torch.nn.functional.interpolate(
                        cm.unsqueeze(0).unsqueeze(0),
                        size=img.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            maps[cls] = cm

    else:
        raise ValueError(f"Unknown method: {method}")

    return maps, logits, sigmoids


# ============================================================================
# Bounding box helpers
# ============================================================================
def clamp_bbox(
    x1: int, y1: int, x2: int, y2: int, H: int, W: int
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp bbox to image bounds and ensure valid (non-empty)."""
    x1 = max(0, min(int(x1), W))
    x2 = max(0, min(int(x2), W))
    y1 = max(0, min(int(y1), H))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_bboxes_for_image_class(
    df_boxes: pd.DataFrame,
    image_id: str,
    class_id: int,
    target_h: int = 224,
    target_w: int = 224,
) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes for a specific image AND class, with union across
    all annotators.

    The train224.csv bbox coordinates are already at 224×224, so no rescaling
    is needed when target_h/w == 224. If the contribution map has a different
    resolution (e.g. CAM output), we rescale proportionally.

    CRITICAL: This filters by class_id to avoid the bug where all bboxes
    for an image (across all classes) are used indiscriminately.

    Returns list of (x1, y1, x2, y2) tuples.
    """
    # Filter by image_id AND class_id (not class_id 14 = "No finding")
    mask = (df_boxes["image_id"].astype(str) == str(image_id)) & (
        df_boxes["class_id"] == class_id
    )
    rows = df_boxes[mask]

    if rows.empty:
        return []

    # Bbox CSV is at 224×224; compute scale factors if target differs
    CSV_SIZE = 224
    scale_x = target_w / CSV_SIZE
    scale_y = target_h / CSV_SIZE

    boxes = []
    for _, r in rows.iterrows():
        if pd.isna(r["x_min"]):
            continue
        x1 = int(round(float(r["x_min"]) * scale_x))
        y1 = int(round(float(r["y_min"]) * scale_y))
        x2 = int(round(float(r["x_max"]) * scale_x))
        y2 = int(round(float(r["y_max"]) * scale_y))
        bbox = clamp_bbox(x1, y1, x2, y2, target_h, target_w)
        if bbox is not None:
            boxes.append(bbox)

    return boxes


def get_classes_with_bboxes(df_boxes: pd.DataFrame, image_id: str) -> List[int]:
    """
    Return sorted list of class_ids that have at least one bbox annotation
    for this image (excluding class_id 14 = "No finding").
    """
    mask = (df_boxes["image_id"].astype(str) == str(image_id)) & (
        df_boxes["class_id"] < 14
    )
    rows = df_boxes[mask]
    # Only keep rows with non-NaN coordinates
    rows = rows.dropna(subset=["x_min"])
    return sorted(rows["class_id"].unique().tolist())


# ============================================================================
# Method resolution helpers
# ============================================================================
def resolve_methods(
    args_methods: Optional[List[str]], model: torch.nn.Module
) -> List[str]:
    """Determine which methods to run, given user args and model type."""
    if args_methods is not None:
        return args_methods
    methods = []
    if is_bcos_model(model):
        methods.append("bcos")
    if HAS_GRAD_CAM:
        methods.extend(["gradcam", "layercam"])
    methods.append("gradxinput")
    return methods


# ============================================================================
# Bbox CSV resolution
# ============================================================================
def resolve_bbox_csv(args_csv: Optional[str]) -> str:
    """Find the VinBigXray bbox CSV, trying user arg, then known paths."""
    candidates = [
        args_csv,
        os.path.join(SCRIPT_DIR, "train224.csv"),
        os.path.join(
            REPO_ROOT,
            "datasets",
            "vinbigdata-chest-xray-abnormalities-detection",
            "train224.csv",
        ),
        os.getenv("VINBIG_CSV_PATH"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Could not find train224.csv (VinBigXray bbox annotations). "
        "Provide it via --bbox_csv or place it in scripts/explanation/."
    )


# ============================================================================
# Metric keys
# ============================================================================
_METRIC_KEYS = [
    "all_precision",
    "all_recall",
    "all_union_precision",
    "all_union_recall",
    "tp_precision",
    "tp_recall",
    "tp_union_precision",
    "tp_union_recall",
    "fn_precision",
    "fn_recall",
    "fn_union_precision",
    "fn_union_recall",
]


# ============================================================================
# Summary CSV writer (VinBigXray version with per-label columns)
# ============================================================================
def write_summary_csv(all_results: List[Dict], out_path: str):
    """Write a flat CSV summarizing all experiments × methods × thresholds."""
    rows = []
    for res in all_results:
        if res.get("skipped"):
            continue

        # IoU fields (threshold-independent)
        iou_fields = {}
        for k in res:
            if k.startswith("bbox_iou"):
                iou_fields[k] = res[k]

        for tk, metrics in res.get("thresholds", {}).items():
            row = {
                "experiment": res.get("experiment_tag", ""),
                "method": res.get("method", ""),
                "model_type": res.get("model_type", ""),
                "reload": res.get("reload", ""),
                "loaded_epoch": res.get("loaded_epoch", ""),
                "threshold": float(tk),
                "n_images": res.get("n_images", ""),
                "n_classes_evaluated": res.get("n_classes_evaluated", ""),
            }
            # Aggregate metrics
            row.update(metrics.get("aggregate", {}))
            row.update(iou_fields)

            # Per-label metrics (flattened with class name prefix)
            per_label = metrics.get("per_label", {})
            for cls_name, cls_metrics in per_label.items():
                safe_name = cls_name.replace(" ", "_").replace("/", "-")
                for mk, mv in cls_metrics.items():
                    row[f"{safe_name}__{mk}"] = mv

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nSummary CSV saved: {out_path}")


# ============================================================================
# Combined evaluation: all methods in one data pass
# ============================================================================
def evaluate_experiment_combined(
    exp_path: str,
    methods: List[str],
    df_boxes: pd.DataFrame,
    thresholds: List[float],
    max_samples: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    reload: str = "last",
    iou_threshold: float = 0.5,
    sigmoid_threshold: float = SIGMOID_THRESHOLD,
) -> List[Dict]:
    """
    Evaluate multiple explanation methods for one VinBigXray experiment.

    Model is loaded once, data is iterated once. For each image, we:
      1. Identify GT-positive classes with bboxes
      2. For each method, compute per-class contribution maps
      3. For each class, compute EPG metrics against class-filtered bboxes

    Returns a list of per-method result dicts.
    """
    t0 = time.time()
    exp_tag = os.path.basename(exp_path.rstrip("/"))

    print(f"\n{'=' * 70}")
    print(f"Experiment: {exp_tag}")
    print(f"Methods:    {methods}")
    print(f"Checkpoint: {reload}")
    print(f"Thresholds: {thresholds}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Load model ONCE
    # ------------------------------------------------------------------
    exp = Experiment(exp_path)
    loaded = exp.load_trained_model(
        reload=reload, return_training_ckpt_if_possible=True
    )
    if isinstance(loaded, dict):
        model = loaded["model"].to(device)
        training_ckpt = loaded.get("ckpt")
        loaded_epoch = (
            training_ckpt["epoch"]
            if training_ckpt and "epoch" in training_ckpt
            else None
        )
    else:
        model = loaded.to(device)
        loaded_epoch = None
    del loaded
    model.eval()
    print(f"  Loaded epoch: {loaded_epoch}")

    bcos = is_bcos_model(model)
    model_type = "bcos" if bcos else "baseline"

    # ------------------------------------------------------------------
    # Filter methods by compatibility
    # ------------------------------------------------------------------
    valid_methods = []
    skipped_results = []

    for method in methods:
        if method == "bcos" and not bcos:
            print(
                f"  SKIP method '{method}': model is not B-cos ({type(model).__name__})"
            )
            skipped_results.append(
                {
                    "skipped": True,
                    "reason": "model is not B-cos",
                    "experiment_path": exp_path,
                    "experiment_tag": exp_tag,
                    "method": method,
                    "model_type": model_type,
                    "reload": reload,
                    "loaded_epoch": loaded_epoch,
                }
            )
        elif method in CAM_REGISTRY and not HAS_GRAD_CAM:
            print(f"  SKIP method '{method}': pytorch-grad-cam not installed")
            skipped_results.append(
                {
                    "skipped": True,
                    "reason": "pytorch-grad-cam not installed",
                    "experiment_path": exp_path,
                    "experiment_tag": exp_tag,
                    "method": method,
                    "model_type": model_type,
                    "reload": reload,
                    "loaded_epoch": loaded_epoch,
                }
            )
        else:
            valid_methods.append(method)

    if not valid_methods:
        print("  No valid methods — returning skipped results only.")
        return skipped_results

    print(f"  Valid methods: {valid_methods}")

    # ------------------------------------------------------------------
    # Setup dataloader ONCE
    # ------------------------------------------------------------------
    dm = exp.get_datamodule()
    dm.setup("val")
    loader = dm.val_dataloader()
    dataset = loader.dataset

    # When running on CPU (e.g. CUDA init failed on the node), the default
    # DataLoader with pin_memory=True and num_workers>0 will crash because
    # PyTorch's _MultiProcessingDataLoaderIter calls torch.cuda.current_device()
    # internally.  Recreate with CPU-safe settings.
    if device.type == "cpu" and (
        getattr(loader, "pin_memory", False) or getattr(loader, "num_workers", 0) > 0
    ):
        print(
            "  WARNING: Rebuilding DataLoader with pin_memory=False, "
            "num_workers=0 for CPU fallback (will be slower)"
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=getattr(loader, "drop_last", False),
        )

    # VinBigXray does NOT have return_id, so we access image IDs via the
    # dataset's dataframe. We need to track the global sample index.

    # ------------------------------------------------------------------
    # Init CAM objects for all CAM methods at once
    # ------------------------------------------------------------------
    cam_objects = {}
    cam_methods_needed = [m for m in valid_methods if m in CAM_REGISTRY]
    if cam_methods_needed:
        target_layer = get_cam_target_layer(model)
        for m in cam_methods_needed:
            cam_objects[m] = CAM_REGISTRY[m](model=model, target_layers=[target_layer])
        print(
            f"  CAM target layer: {type(target_layer).__name__}  "
            f"methods: {cam_methods_needed}"
        )

    # ------------------------------------------------------------------
    # Per-method, per-class storage
    # ------------------------------------------------------------------
    # Structure: per_method[method][class_id] = {
    #   "results_by_threshold": {t: {metric_key: [...]}},
    #   "iou_results": {"all": [], "tp": [], "fn": []},
    #   "n_images": 0, "n_tp": 0, "n_fn": 0,
    # }
    per_method = {}
    for method in valid_methods:
        per_method[method] = {
            "per_class": {},
            "n_images_total": 0,
            "n_skipped_no_box": 0,
            "n_method_errors": 0,
            "n_classes_evaluated": set(),
        }
        for cls_id in range(NUM_CLASSES):
            per_method[method]["per_class"][cls_id] = {
                "results_by_threshold": {
                    t: {k: [] for k in _METRIC_KEYS} for t in thresholds
                },
                "iou_results": {"all": [], "tp": [], "fn": []},
                "n_images": 0,
                "n_tp": 0,
                "n_fn": 0,
            }

    n_images_global = 0  # images that have at least one class with bboxes
    global_sample_idx = 0  # tracks position within the dataset

    # ------------------------------------------------------------------
    # Main loop — iterate data ONCE
    # ------------------------------------------------------------------
    for batch_idx, batch in enumerate(loader):
        # VinBigXray dataset returns (images, labels) — no patient_ids
        images, labels = batch[0], batch[1]
        images = images.to(device)
        batch_size = len(images)

        for i in range(batch_size):
            sample_idx = global_sample_idx + i

            if max_samples is not None and n_images_global >= max_samples:
                break

            img = images[i]
            gt_vec = labels[i].cpu().numpy()  # [14] multi-hot

            # Get image_id from the dataset's dataframe
            if sample_idx >= len(dataset.dataframe):
                print(
                    f"  WARNING: sample_idx {sample_idx} exceeds dataframe "
                    f"length {len(dataset.dataframe)}, skipping rest of batch"
                )
                break
            image_id = str(dataset.dataframe.iloc[sample_idx][dataset.image_col])

            # Find which classes have GT bboxes for this image
            classes_with_bboxes = get_classes_with_bboxes(df_boxes, image_id)
            if not classes_with_bboxes:
                for method in valid_methods:
                    per_method[method]["n_skipped_no_box"] += 1
                continue

            n_images_global += 1

            # --- For each method, compute per-class contribution maps ---
            for method in valid_methods:
                store = per_method[method]

                try:
                    maps, logits_list, sigmoids_list = get_contribution_maps_multilabel(
                        model,
                        method,
                        img,
                        classes_with_bboxes,
                        cam_obj=cam_objects.get(method),
                        device=device,
                    )
                except Exception as e:
                    print(
                        f"  WARNING: {method} failed on {image_id}: "
                        f"{type(e).__name__}: {e}"
                    )
                    store["n_method_errors"] += 1
                    continue

                store["n_images_total"] += 1

                # --- For each class with bboxes, compute EPG ---
                for cls_id in classes_with_bboxes:
                    if cls_id not in maps:
                        continue

                    cm = maps[cls_id]
                    cm_h, cm_w = cm.shape

                    # CRITICAL: get bboxes ONLY for this class
                    bboxes = get_bboxes_for_image_class(
                        df_boxes, image_id, cls_id, cm_h, cm_w
                    )
                    if len(bboxes) == 0:
                        continue

                    store["n_classes_evaluated"].add(cls_id)
                    cls_store = store["per_class"][cls_id]

                    # TP/FN: is this class correctly predicted?
                    gt_positive = gt_vec[cls_id] > 0.5
                    pred_positive = sigmoids_list[cls_id] >= sigmoid_threshold
                    is_correct = gt_positive and pred_positive

                    if is_correct:
                        cls_store["n_tp"] += 1
                    else:
                        cls_store["n_fn"] += 1
                    cls_store["n_images"] += 1

                    # Compute metrics at each threshold
                    for ti, t in enumerate(thresholds):
                        m = compute_epg_for_image(
                            cm, bboxes, threshold=t, iou_threshold=iou_threshold
                        )

                        prefix = "tp_" if is_correct else "fn_"
                        for key, val in [
                            ("precision", m["perbox_precision"]),
                            ("recall", m["perbox_recall"]),
                            ("union_precision", m["union_precision"]),
                            ("union_recall", m["union_recall"]),
                        ]:
                            if not np.isnan(val):
                                cls_store["results_by_threshold"][t][
                                    "all_" + key
                                ].append(val)
                                cls_store["results_by_threshold"][t][
                                    prefix + key
                                ].append(val)

                        # IoU is threshold-independent; collect on first threshold
                        if ti == 0:
                            iou_val = m["bbox_iou"]
                            if not np.isnan(iou_val):
                                cls_store["iou_results"]["all"].append(iou_val)
                                cls_store["iou_results"][
                                    "tp" if is_correct else "fn"
                                ].append(iou_val)

        global_sample_idx += batch_size

        if max_samples is not None and n_images_global >= max_samples:
            break

        # Progress
        if (batch_idx + 1) % 20 == 0:
            print(
                f"  Processed {batch_idx + 1} batches, "
                f"{n_images_global} images with boxes..."
            )

    # Clean up CAM objects
    for cam_obj in cam_objects.values():
        del cam_obj
    cam_objects.clear()

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Aggregate per-method results
    # ------------------------------------------------------------------
    def safe_mean(xs):
        return float(np.mean(xs)) if len(xs) > 0 else float("nan")

    def safe_std(xs):
        return float(np.std(xs)) if len(xs) > 0 else float("nan")

    def safe_count(xs):
        return len(xs)

    all_results = list(skipped_results)

    for method in valid_methods:
        store = per_method[method]
        n_classes_eval = len(store["n_classes_evaluated"])

        output = {
            "experiment_path": exp_path,
            "experiment_tag": exp_tag,
            "method": method,
            "model_type": model_type,
            "reload": reload,
            "loaded_epoch": loaded_epoch,
            "n_images": store["n_images_total"],
            "n_classes_evaluated": n_classes_eval,
            "n_skipped_no_box": store["n_skipped_no_box"],
            "n_method_errors": store["n_method_errors"],
            "elapsed_seconds": round(elapsed, 2),
            "thresholds": {},
        }

        # --- Aggregate IoU across ALL classes ---
        # Micro: pool all individual (image, class) values into one list
        micro_iou_all = []
        micro_iou_tp = []
        micro_iou_fn = []
        # Macro: collect per-class means, then average
        macro_iou_means = []
        macro_iou_tp_means = []
        macro_iou_fn_means = []
        for cls_id in range(NUM_CLASSES):
            cls_s = store["per_class"][cls_id]
            micro_iou_all.extend(cls_s["iou_results"]["all"])
            micro_iou_tp.extend(cls_s["iou_results"]["tp"])
            micro_iou_fn.extend(cls_s["iou_results"]["fn"])
            # Per-class mean for macro
            cls_iou_mean = safe_mean(cls_s["iou_results"]["all"])
            if not np.isnan(cls_iou_mean):
                macro_iou_means.append(cls_iou_mean)
            cls_iou_tp_mean = safe_mean(cls_s["iou_results"]["tp"])
            if not np.isnan(cls_iou_tp_mean):
                macro_iou_tp_means.append(cls_iou_tp_mean)
            cls_iou_fn_mean = safe_mean(cls_s["iou_results"]["fn"])
            if not np.isnan(cls_iou_fn_mean):
                macro_iou_fn_means.append(cls_iou_fn_mean)

        # Micro IoU (pooled across all classes — frequent classes dominate)
        output["bbox_iou_micro_mean"] = safe_mean(micro_iou_all)
        output["bbox_iou_micro_std"] = safe_std(micro_iou_all)
        output["bbox_iou_micro_n"] = safe_count(micro_iou_all)
        output["bbox_iou_micro_tp_mean"] = safe_mean(micro_iou_tp)
        output["bbox_iou_micro_tp_std"] = safe_std(micro_iou_tp)
        output["bbox_iou_micro_fn_mean"] = safe_mean(micro_iou_fn)
        output["bbox_iou_micro_fn_std"] = safe_std(micro_iou_fn)
        # Macro IoU (mean of per-class means — each class equal weight)
        output["bbox_iou_macro_mean"] = safe_mean(macro_iou_means)
        output["bbox_iou_macro_std"] = safe_std(macro_iou_means)
        output["bbox_iou_macro_n_classes"] = safe_count(macro_iou_means)
        output["bbox_iou_macro_tp_mean"] = safe_mean(macro_iou_tp_means)
        output["bbox_iou_macro_fn_mean"] = safe_mean(macro_iou_fn_means)

        # --- Per-threshold: aggregate + per-label ---
        _AGG_KEYS = [
            "precision",
            "recall",
            "union_precision",
            "union_recall",
            "precision_tp",
            "recall_tp",
            "union_precision_tp",
            "union_recall_tp",
            "precision_fn",
            "recall_fn",
            "union_precision_fn",
            "union_recall_fn",
        ]

        for t in thresholds:
            tk = f"{t:.2f}"

            # Collect per-label metrics
            per_label = {}
            # Macro: collect per-class means, then average across classes
            class_means = {k: [] for k in _AGG_KEYS}
            # Micro: pool all individual (image, class) values across classes
            micro_pools = {k: [] for k in _AGG_KEYS}

            for cls_id in range(NUM_CLASSES):
                cls_name = VINBIG_CLASSES[cls_id]
                cls_s = store["per_class"][cls_id]
                r = cls_s["results_by_threshold"][t]

                label_metrics = {
                    "n_images": cls_s["n_images"],
                    "n_tp": cls_s["n_tp"],
                    "n_fn": cls_s["n_fn"],
                    "epg_precision_mean": safe_mean(r["all_precision"]),
                    "epg_precision_std": safe_std(r["all_precision"]),
                    "epg_precision_n_defined": safe_count(r["all_precision"]),
                    "epg_recall_mean": safe_mean(r["all_recall"]),
                    "epg_recall_std": safe_std(r["all_recall"]),
                    "epg_recall_n_defined": safe_count(r["all_recall"]),
                    "union_precision_mean": safe_mean(r["all_union_precision"]),
                    "union_precision_std": safe_std(r["all_union_precision"]),
                    "union_precision_n_defined": safe_count(r["all_union_precision"]),
                    "union_recall_mean": safe_mean(r["all_union_recall"]),
                    "union_recall_std": safe_std(r["all_union_recall"]),
                    "union_recall_n_defined": safe_count(r["all_union_recall"]),
                    # TP
                    "epg_precision_tp_mean": safe_mean(r["tp_precision"]),
                    "epg_recall_tp_mean": safe_mean(r["tp_recall"]),
                    "union_precision_tp_mean": safe_mean(r["tp_union_precision"]),
                    "union_recall_tp_mean": safe_mean(r["tp_union_recall"]),
                    # FN
                    "epg_precision_fn_mean": safe_mean(r["fn_precision"]),
                    "epg_recall_fn_mean": safe_mean(r["fn_recall"]),
                    "union_precision_fn_mean": safe_mean(r["fn_union_precision"]),
                    "union_recall_fn_mean": safe_mean(r["fn_union_recall"]),
                    # IoU for this class
                    "bbox_iou_mean": safe_mean(cls_s["iou_results"]["all"]),
                    "bbox_iou_n_defined": safe_count(cls_s["iou_results"]["all"]),
                }

                per_label[cls_name] = label_metrics

                # --- Collect for macro-average (per-class mean, only if defined) ---
                _m = label_metrics
                _MACRO_MAP = [
                    ("precision", "epg_precision_mean"),
                    ("recall", "epg_recall_mean"),
                    ("union_precision", "union_precision_mean"),
                    ("union_recall", "union_recall_mean"),
                    ("precision_tp", "epg_precision_tp_mean"),
                    ("recall_tp", "epg_recall_tp_mean"),
                    ("union_precision_tp", "union_precision_tp_mean"),
                    ("union_recall_tp", "union_recall_tp_mean"),
                    ("precision_fn", "epg_precision_fn_mean"),
                    ("recall_fn", "epg_recall_fn_mean"),
                    ("union_precision_fn", "union_precision_fn_mean"),
                    ("union_recall_fn", "union_recall_fn_mean"),
                ]
                for agg_key, lbl_key in _MACRO_MAP:
                    v = _m[lbl_key]
                    if not np.isnan(v):
                        class_means[agg_key].append(v)

                # --- Collect for micro-average (all individual values) ---
                _MICRO_MAP = [
                    ("precision", "all_precision"),
                    ("recall", "all_recall"),
                    ("union_precision", "all_union_precision"),
                    ("union_recall", "all_union_recall"),
                    ("precision_tp", "tp_precision"),
                    ("recall_tp", "tp_recall"),
                    ("union_precision_tp", "tp_union_precision"),
                    ("union_recall_tp", "tp_union_recall"),
                    ("precision_fn", "fn_precision"),
                    ("recall_fn", "fn_recall"),
                    ("union_precision_fn", "fn_union_precision"),
                    ("union_recall_fn", "fn_union_recall"),
                ]
                for agg_key, store_key in _MICRO_MAP:
                    micro_pools[agg_key].extend(r[store_key])

            # Macro-averaged aggregate (mean of per-class means — equal weight per class)
            aggregate = {
                # --- MACRO: mean of per-class means ---
                "epg_precision_macro_mean": safe_mean(class_means["precision"]),
                "epg_precision_macro_n_classes": safe_count(class_means["precision"]),
                "epg_recall_macro_mean": safe_mean(class_means["recall"]),
                "epg_recall_macro_n_classes": safe_count(class_means["recall"]),
                "union_precision_macro_mean": safe_mean(class_means["union_precision"]),
                "union_precision_macro_n_classes": safe_count(
                    class_means["union_precision"]
                ),
                "union_recall_macro_mean": safe_mean(class_means["union_recall"]),
                "union_recall_macro_n_classes": safe_count(class_means["union_recall"]),
                # Macro TP
                "epg_precision_macro_tp_mean": safe_mean(class_means["precision_tp"]),
                "epg_recall_macro_tp_mean": safe_mean(class_means["recall_tp"]),
                "union_precision_macro_tp_mean": safe_mean(
                    class_means["union_precision_tp"]
                ),
                "union_recall_macro_tp_mean": safe_mean(class_means["union_recall_tp"]),
                # Macro FN
                "epg_precision_macro_fn_mean": safe_mean(class_means["precision_fn"]),
                "epg_recall_macro_fn_mean": safe_mean(class_means["recall_fn"]),
                "union_precision_macro_fn_mean": safe_mean(
                    class_means["union_precision_fn"]
                ),
                "union_recall_macro_fn_mean": safe_mean(class_means["union_recall_fn"]),
                # --- MICRO: mean of all individual (image, class) values ---
                "epg_precision_micro_mean": safe_mean(micro_pools["precision"]),
                "epg_precision_micro_std": safe_std(micro_pools["precision"]),
                "epg_precision_micro_n": safe_count(micro_pools["precision"]),
                "epg_recall_micro_mean": safe_mean(micro_pools["recall"]),
                "epg_recall_micro_std": safe_std(micro_pools["recall"]),
                "epg_recall_micro_n": safe_count(micro_pools["recall"]),
                "union_precision_micro_mean": safe_mean(micro_pools["union_precision"]),
                "union_precision_micro_std": safe_std(micro_pools["union_precision"]),
                "union_precision_micro_n": safe_count(micro_pools["union_precision"]),
                "union_recall_micro_mean": safe_mean(micro_pools["union_recall"]),
                "union_recall_micro_std": safe_std(micro_pools["union_recall"]),
                "union_recall_micro_n": safe_count(micro_pools["union_recall"]),
                # Micro TP
                "epg_precision_micro_tp_mean": safe_mean(micro_pools["precision_tp"]),
                "epg_recall_micro_tp_mean": safe_mean(micro_pools["recall_tp"]),
                "union_precision_micro_tp_mean": safe_mean(
                    micro_pools["union_precision_tp"]
                ),
                "union_recall_micro_tp_mean": safe_mean(micro_pools["union_recall_tp"]),
                # Micro FN
                "epg_precision_micro_fn_mean": safe_mean(micro_pools["precision_fn"]),
                "epg_recall_micro_fn_mean": safe_mean(micro_pools["recall_fn"]),
                "union_precision_micro_fn_mean": safe_mean(
                    micro_pools["union_precision_fn"]
                ),
                "union_recall_micro_fn_mean": safe_mean(micro_pools["union_recall_fn"]),
            }

            output["thresholds"][tk] = {
                "aggregate": aggregate,
                "per_label": per_label,
            }

        # Print per-method summary table
        print(f"\n  --- {method} ---")
        print(
            f"  {store['n_images_total']} images, "
            f"{n_classes_eval} classes evaluated, "
            f"{store['n_skipped_no_box']} no-box, "
            f"{store['n_method_errors']} errors"
        )
        # Print aggregate metrics for the first threshold
        if thresholds:
            tk0 = f"{thresholds[0]:.2f}"
            agg0 = output["thresholds"][tk0]["aggregate"]
            print(
                f"  MACRO (t={thresholds[0]:.2f}): "
                f"Prec={agg0['epg_precision_macro_mean']:.4f}  "
                f"Rec={agg0['epg_recall_macro_mean']:.4f}  "
                f"U-Prec={agg0['union_precision_macro_mean']:.4f}  "
                f"U-Rec={agg0['union_recall_macro_mean']:.4f}"
            )
            print(
                f"  MICRO (t={thresholds[0]:.2f}): "
                f"Prec={agg0['epg_precision_micro_mean']:.4f}  "
                f"Rec={agg0['epg_recall_micro_mean']:.4f}  "
                f"U-Prec={agg0['union_precision_micro_mean']:.4f}  "
                f"U-Rec={agg0['union_recall_micro_mean']:.4f}  "
                f"(n={agg0['epg_precision_micro_n']})"
            )
            # Per-class summary at first threshold
            pl0 = output["thresholds"][tk0]["per_label"]
            print(
                f"  {'Class':<24} {'N':>5} {'TP':>4} {'FN':>4} "
                f"{'Prec':>8} {'Rec':>8} {'U-Prec':>8} {'U-Rec':>8} {'IoU':>8}"
            )
            print(
                f"  {'-' * 24} {'-' * 5} {'-' * 4} {'-' * 4} "
                f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}"
            )
            for cls_id in range(NUM_CLASSES):
                cls_name = VINBIG_CLASSES[cls_id]
                lm = pl0[cls_name]
                print(
                    f"  {cls_name:<24} {lm['n_images']:>5} "
                    f"{lm['n_tp']:>4} {lm['n_fn']:>4} "
                    f"{lm['epg_precision_mean']:>8.4f} "
                    f"{lm['epg_recall_mean']:>8.4f} "
                    f"{lm['union_precision_mean']:>8.4f} "
                    f"{lm['union_recall_mean']:>8.4f} "
                    f"{lm['bbox_iou_mean']:>8.4f}"
                )

        print(
            f"  BBox IoU macro: {output['bbox_iou_macro_mean']:.4f}  "
            f"(TP: {output['bbox_iou_macro_tp_mean']:.4f}, "
            f"FN: {output['bbox_iou_macro_fn_mean']:.4f}, "
            f"n_classes: {output['bbox_iou_macro_n_classes']})"
        )
        print(
            f"  BBox IoU micro: {output['bbox_iou_micro_mean']:.4f} +/- "
            f"{output['bbox_iou_micro_std']:.4f}  "
            f"(TP: {output['bbox_iou_micro_tp_mean']:.4f}, "
            f"FN: {output['bbox_iou_micro_fn_mean']:.4f}, "
            f"n={output['bbox_iou_micro_n']})"
        )

        all_results.append(output)

    print(
        f"\n  Total elapsed: {elapsed:.1f}s for {len(valid_methods)} methods, "
        f"{n_images_global} images with boxes"
    )

    return all_results


# ============================================================================
# Argument parser
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute EPG Precision & Recall for VinBigXray (multilabel) — "
            "all methods in one pass per experiment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment_paths",
        nargs="+",
        required=True,
        help="Path(s) to experiment directories.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            f"Explanation methods. Options: {ALL_METHODS}. "
            "Default: auto-detect based on model type."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0],
        help="Threshold values. Default: [0.0]",
    )
    parser.add_argument(
        "--bbox_csv",
        type=str,
        default=None,
        help="Path to VinBigXray train224.csv. Default: auto-detect.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max images (with boxes) to evaluate. Default: all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Summary output directory. Default: common parent/epg_results/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--reload",
        type=str,
        default="last",
        help="Which checkpoint to load. Default: 'last'.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="Binarisation threshold for BBox IoU metric. Default: 0.5.",
    )
    parser.add_argument(
        "--sigmoid_threshold",
        type=float,
        default=SIGMOID_THRESHOLD,
        help=f"Sigmoid threshold for TP/FN classification. Default: {SIGMOID_THRESHOLD}.",
    )
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Bbox CSV
    bbox_csv_path = resolve_bbox_csv(args.bbox_csv)
    print(f"Bbox CSV: {bbox_csv_path}")
    df_boxes = pd.read_csv(bbox_csv_path)
    # Exclude "No finding" (class_id 14)
    df_boxes_valid = df_boxes[df_boxes["class_id"] < 14].dropna(subset=["x_min"])
    n_images_with_boxes = df_boxes_valid["image_id"].nunique()
    print(
        f"  {len(df_boxes)} total rows, {len(df_boxes_valid)} with bbox coords, "
        f"{n_images_with_boxes} unique images with boxes"
    )

    # Output directory
    if args.output_dir:
        summary_dir = args.output_dir
    else:
        abs_paths = [os.path.abspath(p.rstrip("/")) for p in args.experiment_paths]
        summary_dir = os.path.join(os.path.commonpath(abs_paths), "epg_results")
    os.makedirs(summary_dir, exist_ok=True)
    print(f"Summary output dir: {summary_dir}")

    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_results = []

    for exp_path in args.experiment_paths:
        exp_path = exp_path.rstrip("/")
        if not os.path.isdir(exp_path):
            print(f"\nWARNING: experiment path does not exist: {exp_path}")
            continue

        # Per-experiment output directory
        exp_out_dir = os.path.join(exp_path, "epg_results")
        os.makedirs(exp_out_dir, exist_ok=True)

        # Determine methods (auto-detect requires peeking at the model)
        if args.methods is None:
            try:
                exp = Experiment(exp_path)
                model_peek = exp.load_trained_model(reload=args.reload)
                if isinstance(model_peek, dict):
                    model_peek = model_peek["model"]
                model_peek.eval()
                methods = resolve_methods(None, model_peek)
                del model_peek
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"\nERROR loading {exp_path} for method detection: {e}")
                continue
        else:
            methods = args.methods

        print(f"\nMethods for {os.path.basename(exp_path)}: {methods}")

        # Evaluate all methods in one pass
        try:
            results = evaluate_experiment_combined(
                exp_path=exp_path,
                methods=methods,
                df_boxes=df_boxes,
                thresholds=args.thresholds,
                max_samples=args.max_samples,
                device=device,
                reload=args.reload,
                iou_threshold=args.iou_threshold,
                sigmoid_threshold=args.sigmoid_threshold,
            )
        except Exception as e:
            print(f"\nERROR evaluating {exp_path}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Save per-method JSONs
        for result in results:
            all_results.append(result)
            if not result.get("skipped"):
                exp_tag = result["experiment_tag"]
                method = result["method"]
                json_path = os.path.join(
                    exp_out_dir,
                    f"epg_{exp_tag}_{method}_{run_tag}.json",
                )
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  JSON saved: {json_path}")

        # Free GPU memory between experiments
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save combined summary CSV + JSON
    csv_path = os.path.join(summary_dir, f"epg_summary_vinbig_{run_tag}.csv")
    write_summary_csv(all_results, csv_path)

    combined_json = os.path.join(summary_dir, f"epg_all_vinbig_{run_tag}.json")
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined JSON: {combined_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
