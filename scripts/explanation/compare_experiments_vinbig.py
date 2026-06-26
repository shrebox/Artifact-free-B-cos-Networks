#!/usr/bin/env python3
"""
Cross-experiment comparison for VinBigData (14-class multilabel classification).

Layout:
  Rows    = test images (selected by EXPLAIN_CLASS GT-positive status)
  Col 0   = original X-ray with GT bounding boxes for EXPLAIN_CLASS
  Col 1+  = one column per (experiment, explanation_method) pair

Each entry in EXPERIMENT_COLUMNS independently pairs an experiment path with
an explanation method. Supports mixing B-cos and baseline experiments.
All columns explain the SAME class (EXPLAIN_CLASS) for direct comparison.

Explanation methods (for the 'method' field in EXPERIMENT_COLUMNS):
  B-cos inherent (shows N/A on baseline models):
    "bcos_contribs"  — contribution heatmap (bwr colormap)
    "bcos_rgba"      — B-cos colored explanation (RGBA)
    "bcos_overlay"   — contribution overlay on X-ray
  Post-hoc CAM (works on any model):
    "GradCAM", "LayerCAM", "HiResCAM", "GradCAM++", "AblationCAM"

Multilabel specifics:
  - Sigmoid (not softmax) for probabilities
  - Each class logit is backpropagated independently
  - Confidence filtering uses the first experiment's model
"""

import os
import sys
from collections import defaultdict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_grad_cam import (
    AblationCAM,
    FinerCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from bcos.experiments.utils import Experiment

# ===================================================================
# CONFIGURATION
# ===================================================================

# Each entry defines one column in the comparison grid.
# Fields:
#   exp_path : path to experiment directory
#   method   : explanation method (see docstring above)
#   smooth   : B-cos smoothing kernel (only for bcos_* methods; ignored for CAMs)
#   label    : column header label (auto-generated from basename + method if None)
EXPERIMENT_COLUMNS = [
    # Example:
    # {
    #     "exp_path": "experiments/VinBigXray/bcosification/resnet_50_...",
    #     "method": "GradCAM",
    #     "smooth": 15,
    #     "label": None,
    # },
]

# Which class to explain (name or integer index). All columns explain this class.
# Examples: "Lung Opacity", "Cardiomegaly", 7, 3
EXPLAIN_CLASS = "Lung Opacity"

# Only show images where EXPLAIN_CLASS is GT-positive.
# Set False to also include GT-negative images (useful for false-positive analysis).
REQUIRE_GT_POSITIVE = True

NUM_IMAGES = 5  # number of image rows per PDF
bg_opacity = 0.3  # background X-ray opacity in overlay cells (0=white, 1=full)
sigmoid_threshold = 0.5  # threshold for "predicted positive" display

# --- Confidence filter (same as per-experiment script) ---
# Only show images where the first experiment's model sigmoid >= MIN_CONFIDENCE
# for EXPLAIN_CLASS. Set None to disable.
MIN_CONFIDENCE = None
MAX_SEARCH = 500  # max forward passes for confidence filtering

# --- Bbox settings ---
BBOX_CSV_PATH = os.getenv("VINBIG_CSV_PATH")
# Bbox display mode:
# "first"   : first annotator bbox per class (clean)
# "all"     : all annotator bboxes per class
# "merge"   : average all annotator bboxes per class
# "random"  : random annotator bbox per class
# "largest" : largest-area annotator bbox per class
BBOX_MODE = "first"

# --- Overlay style (same as per-experiment script) ---
# "jet"     : B-cos overlay uses bwr+magnitude, CAM rows use JET+uniform
# "bcos"    : everything B-cos style — CAM overlays also use bwr+magnitude
# "all_jet" : everything JET style — B-cos overlay also uses JET+magnitude
CAM_OVERLAY_STYLE = "jet"

# Output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparisons_vinbig")

# ===================================================================
# Constants
# ===================================================================
NUM_CLASSES = 14
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

CAM_CLASS_MAP = {
    "GradCAM": GradCAM,
    "LayerCAM": LayerCAM,
    "HiResCAM": HiResCAM,
    "GradCAM++": GradCAMPlusPlus,
    "AblationCAM": AblationCAM,
    "FinerCAM": FinerCAM,
}
BCOS_METHODS = {"bcos_contribs", "bcos_rgba", "bcos_overlay"}
ALL_METHODS = BCOS_METHODS | set(CAM_CLASS_MAP.keys())


# ===================================================================
# Utility functions (shared with per-experiment scripts)
# ===================================================================
def idx_to_class(idx: int) -> str:
    if 0 <= idx < len(VINBIG_CLASSES):
        return VINBIG_CLASSES[idx]
    return str(idx)


def class_name_to_idx(name: str) -> int:
    """Resolve a class name to its index. Case-insensitive partial match."""
    name_lower = name.strip().lower()
    for i, c in enumerate(VINBIG_CLASSES):
        if c.lower() == name_lower:
            return i
    for i, c in enumerate(VINBIG_CLASSES):
        if name_lower in c.lower():
            return i
    raise ValueError(f"Unknown class name '{name}'. Available: {VINBIG_CLASSES}")


def resolve_explain_class(explain_class):
    """Convert EXPLAIN_CLASS (str or int) to an integer index."""
    if isinstance(explain_class, int):
        if not (0 <= explain_class < NUM_CLASSES):
            raise ValueError(
                f"EXPLAIN_CLASS index {explain_class} out of range [0, {NUM_CLASSES})"
            )
        return explain_class
    return class_name_to_idx(explain_class)


def to_numpy(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


def _is_bcos_model(model):
    return hasattr(model, "explanation_mode")


def _get_cam_target_layer(model):
    model_name = type(model).__name__
    if hasattr(model, "layer4"):
        target = model.layer4[-1]
        print(
            f"  [{model_name}] CAM target: model.layer4[-1] ({type(target).__name__})"
        )
        return target
    elif hasattr(model, "features"):
        target = model.features[-1][-1]
        print(
            f"  [{model_name}] CAM target: model.features[-1][-1] ({type(target).__name__})"
        )
        return target
    else:
        raise ValueError(
            f"Cannot auto-detect CAM target layer for '{model_name}'. "
            f"Add support in _get_cam_target_layer()."
        )


def gradient_to_image(
    image, linear_mapping, smooth=15, alpha_percentile=99.5, return_contribs=False
):
    """
    B-cos gradient-to-image with 6-channel normalization.
    Returns RGBA (H,W,4) and optionally raw contributions (1,H,W).
    """
    contribs = (image * linear_mapping).sum(0, keepdim=True)
    rgb_grad = linear_mapping / (
        linear_mapping.abs().max(0, keepdim=True).values + 1e-12
    )
    rgb_grad = rgb_grad.clamp(min=0)
    rgb_grad = rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12)

    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    alpha = (alpha / torch.quantile(alpha, q=alpha_percentile / 100)).clip(0, 1)

    rgb_grad = torch.cat([rgb_grad, alpha], dim=0)
    grad_image = rgb_grad.permute(1, 2, 0)

    if return_contribs:
        return grad_image.detach().cpu().numpy(), contribs.detach().cpu().numpy()
    return grad_image.detach().cpu().numpy()


def _overlay_contribs_on_image(
    contribs_2d,
    orig_image_np,
    cmap_name="bwr",
    percentile=99.5,
    overlay_strength=1,
    bg_alpha=1.0,
):
    """Magnitude-based alpha overlay with signed colormap (bwr)."""
    cutoff = np.percentile(np.abs(contribs_2d), percentile) + 1e-8
    clipped = np.clip(contribs_2d, -cutoff, cutoff)
    vrange = np.max(np.abs(clipped)) + 1e-8
    normed = (clipped / (2 * vrange)) + 0.5
    cmap = plt.cm.get_cmap(cmap_name)
    heatmap = cmap(normed)[..., :3]
    magnitude = np.abs(clipped) / vrange
    alpha = (magnitude * overlay_strength)[..., None]
    faded_bg = bg_alpha * orig_image_np + (1 - bg_alpha) * 1.0
    overlay = alpha * heatmap + (1 - alpha) * faded_bg
    return np.clip(overlay, 0, 1)


def _jet_magnitude_overlay(contribs_2d, orig_image_np, percentile=99.5, bg_alpha=1.0):
    """JET colormap with magnitude-based alpha for B-cos contributions."""
    magnitude = np.abs(contribs_2d)
    cutoff = np.percentile(magnitude, percentile) + 1e-8
    normed = np.clip(magnitude / cutoff, 0, 1)
    cam_uint8 = np.uint8(255 * normed)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = (
        cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    )
    alpha = normed[..., None]
    faded_bg = bg_alpha * orig_image_np + (1 - bg_alpha) * 1.0
    overlay = alpha * heatmap_rgb + (1 - alpha) * faded_bg
    return np.clip(overlay, 0, 1)


def _cam_overlay_on_image(cam_map, orig_image_np, cam_alpha=0.4, bg_alpha=1.0):
    """Standard JET overlay with uniform alpha for CAM maps."""
    h, w = orig_image_np.shape[:2]
    cam_resized = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = (
        cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    )
    faded_bg = bg_alpha * orig_image_np + (1 - bg_alpha) * 1.0
    overlay = cam_alpha * heatmap_rgb + (1 - cam_alpha) * faded_bg
    return np.clip(overlay, 0, 1)


def _set_box(ax, *, color="black", lw=2.0, ls="solid"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_linestyle(ls)


def _draw_na_panel(ax, label="N/A\n(baseline model)"):
    ax.set_facecolor("#f0f0f0")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#888888",
        fontstyle="italic",
    )
    _set_box(ax, color="#cccccc", lw=1.0)


# ===================================================================
# Bounding box helpers (VinBigData — coords already at 224x224)
# ===================================================================
def load_bbox_df(csv_path):
    if not os.path.exists(csv_path):
        print(f"WARNING: bbox CSV not found at {csv_path}, skipping bbox overlays.")
        return None
    df = pd.read_csv(csv_path)
    required = ["image_id", "class_id", "x_min", "y_min", "x_max", "y_max"]
    for col in required:
        if col not in df.columns:
            print(f"WARNING: bbox CSV missing column '{col}', skipping bbox overlays.")
            return None
    print(f"Loaded bbox CSV: {len(df)} rows from {csv_path}")
    return df


def _get_all_bboxes(bbox_df, image_id, class_id=None):
    """Get ALL annotator bboxes for an image (and optionally a specific class)."""
    if bbox_df is None:
        return []
    mask = bbox_df["image_id"].astype(str) == str(image_id)
    if class_id is not None:
        mask = mask & (bbox_df["class_id"] == class_id)
    rows = bbox_df[mask]
    bboxes = []
    for _, row in rows.iterrows():
        if pd.notna(row["x_min"]):
            bboxes.append(
                {
                    "class_id": int(row["class_id"]),
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                }
            )
    return bboxes


def get_bboxes_for_image(bbox_df, image_id, class_id=None, mode="first"):
    """Get bboxes with the selected BBOX_MODE applied."""
    all_bboxes = _get_all_bboxes(bbox_df, image_id, class_id)
    if not all_bboxes:
        return []

    if mode == "first":
        # Keep only first bbox per class_id
        seen = set()
        result = []
        for bb in all_bboxes:
            if bb["class_id"] not in seen:
                seen.add(bb["class_id"])
                result.append(bb)
        return result
    elif mode == "merge":
        # Average coordinates per class_id
        grouped = defaultdict(list)
        for bb in all_bboxes:
            grouped[bb["class_id"]].append(bb)
        result = []
        for cls_id, group in grouped.items():
            n = len(group)
            result.append(
                {
                    "class_id": cls_id,
                    "x_min": sum(b["x_min"] for b in group) / n,
                    "y_min": sum(b["y_min"] for b in group) / n,
                    "x_max": sum(b["x_max"] for b in group) / n,
                    "y_max": sum(b["y_max"] for b in group) / n,
                }
            )
        return result
    elif mode == "random":
        import random

        grouped = defaultdict(list)
        for bb in all_bboxes:
            grouped[bb["class_id"]].append(bb)
        return [random.choice(group) for group in grouped.values()]
    elif mode == "largest":
        grouped = defaultdict(list)
        for bb in all_bboxes:
            grouped[bb["class_id"]].append(bb)
        result = []
        for group in grouped.values():
            best = max(
                group,
                key=lambda b: (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]),
            )
            result.append(best)
        return result
    else:  # "all"
        return all_bboxes


def draw_bboxes_on_ax(ax, bboxes, color="lime", lw=2):
    for bb in bboxes:
        rect = patches.Rectangle(
            (bb["x_min"], bb["y_min"]),
            bb["x_max"] - bb["x_min"],
            bb["y_max"] - bb["y_min"],
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)


# ===================================================================
# Core computation
# ===================================================================
def get_predictions_multilabel(model, img_cpu, device, is_bcos):
    """
    Get multilabel predictions using sigmoid.
    B-cos: inside explanation_mode (matches explanation inference).
    Baseline: standard no_grad.
    Returns (logits_list, sigmoid_probs_list).
    """
    model.eval()
    if is_bcos:
        with torch.enable_grad(), model.explanation_mode():
            img_t = img_cpu[None].to(device).requires_grad_(True)
            out = model(img_t)
            logits = out.detach().cpu()[0].tolist()
            sigmoid_probs = torch.sigmoid(out.detach())[0].cpu().tolist()
    else:
        with torch.no_grad():
            img_t = img_cpu[None].to(device)
            out = model(img_t)
            logits = out.detach().cpu()[0].tolist()
            sigmoid_probs = torch.sigmoid(out.detach())[0].cpu().tolist()
    return logits, sigmoid_probs


def compute_explanation(
    model, img_cpu, target_cls, method, device, smooth, orig_np, is_bcos
):
    """
    Compute a single explanation for one image, one class.

    Returns one of:
      ("contribs", contribs_2d, vrange)  — for bcos_contribs
      ("rgba", rgba_array)               — for bcos_rgba
      ("rgb", overlay_array)             — for bcos_overlay and CAM methods
      None                               — N/A (bcos method on baseline model)
    """
    model.eval()

    if method in BCOS_METHODS:
        if not is_bcos:
            return None

        with torch.enable_grad(), model.explanation_mode():
            img_t = img_cpu[None].to(device).requires_grad_(True)
            out = model(img_t)
            # Single class backprop — no retain_graph needed
            grad = torch.autograd.grad(
                out[0, target_cls], img_t, retain_graph=False, create_graph=False
            )[0]
            grad = grad.detach().cpu()[0]

        rgba_expl, contribs = gradient_to_image(
            img_cpu, grad, smooth=smooth, return_contribs=True
        )
        contribs = contribs.squeeze()

        if contribs.size:
            cutoff = np.percentile(np.abs(contribs), 99.5)
            contribs = np.clip(contribs, -cutoff, cutoff)
            vrange = float(np.max(np.abs(contribs)))
        else:
            vrange = 0.0

        if method == "bcos_contribs":
            return ("contribs", contribs, vrange)
        elif method == "bcos_rgba":
            return ("rgba", rgba_expl)
        elif method == "bcos_overlay":
            if CAM_OVERLAY_STYLE == "all_jet":
                overlay = _jet_magnitude_overlay(contribs, orig_np, bg_alpha=bg_opacity)
            else:
                overlay = _overlay_contribs_on_image(
                    contribs, orig_np, cmap_name="bwr", bg_alpha=bg_opacity
                )
            return ("rgb", overlay)

    elif method in CAM_CLASS_MAP:
        # CAM runs OUTSIDE explanation_mode
        target_layer = _get_cam_target_layer(model)
        cam_cls = CAM_CLASS_MAP[method]
        cam = cam_cls(model=model, target_layers=[target_layer])

        img_t = img_cpu[None].to(device)
        if method == "FinerCAM":
            # FinerCAM requires targets=None to activate its own
            # FinerWeightedTarget (contrastive loss).  Passing explicit
            # ClassifierOutputTarget makes it fall back to vanilla GradCAM.
            grayscale_cam = cam(input_tensor=img_t, targets=None)
        else:
            targets = [ClassifierOutputTarget(target_cls)]
            grayscale_cam = cam(input_tensor=img_t, targets=targets)
        cam_map = grayscale_cam[0]
        del cam

        if CAM_OVERLAY_STYLE == "bcos":
            overlay = _overlay_contribs_on_image(cam_map, orig_np, bg_alpha=bg_opacity)
        else:
            overlay = _cam_overlay_on_image(cam_map, orig_np, bg_alpha=bg_opacity)
        return ("rgb", overlay)

    else:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(ALL_METHODS)}")


# ===================================================================
# Image selection with confidence filtering
# ===================================================================
def select_images(ref_dataset, first_exp_path, explain_cls_idx, device):
    """
    Select images for comparison, optionally filtering by confidence.
    Uses the first experiment's model for confidence checks (when MIN_CONFIDENCE is set).

    Returns list of dicts: {img_cpu, gt_np, image_id, idx}
    """
    n_total = len(ref_dataset)
    rng = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=rng).tolist()

    # --- Load first model for confidence filtering (if needed) ---
    conf_model = None
    conf_is_bcos = False
    if MIN_CONFIDENCE is not None:
        exp = Experiment(first_exp_path)
        # Load from this experiment's own dataset for proper preprocessing
        conf_dm = exp.get_datamodule()
        conf_dm.setup("val")
        conf_dataset = conf_dm.val_dataloader().dataset

        conf_model = exp.load_trained_model(reload="last")
        conf_model.to(device)
        conf_model.eval()
        conf_is_bcos = _is_bcos_model(conf_model)
        print(
            f"  Confidence filter: using {type(conf_model).__name__} "
            f"({'B-cos' if conf_is_bcos else 'baseline'})"
        )

    selected = []
    n_skipped_gt = 0
    n_skipped_conf = 0
    n_checked_conf = 0
    best_min_score = -1.0
    best_min_score_id = None

    for idx in perm:
        if len(selected) >= NUM_IMAGES:
            break
        if MIN_CONFIDENCE is not None and n_checked_conf >= MAX_SEARCH:
            break

        img, gt = ref_dataset[idx]
        gt_np = to_numpy(gt)

        # GT filter
        if REQUIRE_GT_POSITIVE and gt_np[explain_cls_idx] < 0.5:
            n_skipped_gt += 1
            continue

        # Confidence filter
        if MIN_CONFIDENCE is not None and conf_model is not None:
            n_checked_conf += 1

            # Use the confidence model's own dataset for proper preprocessing
            conf_img = conf_dataset[idx][0]

            if conf_is_bcos:
                with torch.enable_grad(), conf_model.explanation_mode():
                    img_t = conf_img[None].to(device).requires_grad_(True)
                    out = conf_model(img_t)
                    sig = torch.sigmoid(out.detach())[0].cpu().numpy()
            else:
                with torch.no_grad():
                    img_t = conf_img[None].to(device)
                    out = conf_model(img_t)
                    sig = torch.sigmoid(out)[0].cpu().numpy()

            score = float(sig[explain_cls_idx])

            # Debug: print first 20 candidates
            if n_checked_conf <= 20:
                img_id_dbg = str(ref_dataset.dataframe.iloc[idx][ref_dataset.image_col])
                print(
                    f"    [DEBUG #{n_checked_conf}] idx={idx} id={img_id_dbg}"
                    f" | GT={int(gt_np[explain_cls_idx])}"
                    f" | sig({idx_to_class(explain_cls_idx)})={score:.4f}"
                    f" | {'PASS' if score >= MIN_CONFIDENCE else 'REJECT'}"
                )

            if score < MIN_CONFIDENCE:
                if score > best_min_score:
                    best_min_score = score
                    best_min_score_id = str(
                        ref_dataset.dataframe.iloc[idx][ref_dataset.image_col]
                    )
                n_skipped_conf += 1
                continue

        # Get image_id for bbox lookup
        image_id = str(ref_dataset.dataframe.iloc[idx][ref_dataset.image_col])

        selected.append(
            {
                "img_cpu": img.detach().cpu(),
                "gt_np": gt_np,
                "image_id": image_id,
                "idx": idx,
            }
        )

    # Clean up confidence model
    if conf_model is not None:
        del conf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Diagnostic output
    print(
        f"  Selected {len(selected)} images"
        f" (skipped {n_skipped_gt} by GT filter, {n_skipped_conf} by confidence filter)"
    )
    if MIN_CONFIDENCE is not None:
        print(f"  Checked {n_checked_conf}/{MAX_SEARCH} candidates with forward pass")

    if len(selected) < NUM_IMAGES and MIN_CONFIDENCE is not None:
        print()
        print("  " + "!" * 60)
        if len(selected) == 0:
            print(f"  WARNING: No images found with MIN_CONFIDENCE >= {MIN_CONFIDENCE}")
        else:
            print(
                f"  WARNING: Only found {len(selected)}/{NUM_IMAGES} images"
                f" with MIN_CONFIDENCE >= {MIN_CONFIDENCE}"
            )
        if best_min_score >= 0:
            print(
                f"  Best rejected candidate: {best_min_score_id}"
                f" (sigmoid = {best_min_score:.4f})"
            )
            print(
                f"  -> Try lowering MIN_CONFIDENCE to {best_min_score:.2f}"
                f" or below to include it"
            )
        if n_checked_conf >= MAX_SEARCH:
            print(
                f"  -> Search stopped at MAX_SEARCH={MAX_SEARCH}."
                f" Try increasing MAX_SEARCH."
            )
        print("  " + "!" * 60)
        print()

    return selected


# ===================================================================
# Rendering helpers
# ===================================================================
def _render_cell(ax, result, bboxes=None):
    """Render a computed explanation onto a matplotlib axis."""
    if result is None:
        _draw_na_panel(ax)
        return

    kind = result[0]
    if kind == "contribs":
        _, contribs, vrange = result
        ax.imshow(contribs, cmap="bwr", vmin=-vrange, vmax=vrange)
    elif kind == "rgba":
        _, rgba = result
        ax.imshow(rgba)
    elif kind == "rgb":
        _, overlay = result
        ax.imshow(overlay)

    if bboxes:
        draw_bboxes_on_ax(ax, bboxes, color="lime", lw=1.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _get_column_label(col_cfg):
    """Generate display label for an experiment column.
    Set label to a short string like "BcosResNet GradCAM", or None to auto-generate."""
    if col_cfg.get("label") is not None:
        return str(col_cfg["label"])
    basename = os.path.basename(os.path.normpath(col_cfg["exp_path"]))
    if len(basename) > 40:
        basename = basename[:37] + "..."
    return f"{basename}\n{col_cfg['method']}"


# ===================================================================
# Plotting
# ===================================================================
def plot_comparison_grid(
    ref_images, column_results, bbox_df, explain_cls_idx, title=None
):
    """
    Create the comparison figure.

    ref_images:       list of dicts {img_cpu, gt_np, image_id, idx}
    column_results:   list of dicts {config, data: [{result, logits, sigmoid, ...}], is_bcos}
    bbox_df:          DataFrame or None
    explain_cls_idx:  integer class index being explained
    """
    nrows = len(ref_images)
    ncols = 1 + len(column_results)  # original column + experiment columns

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5), dpi=200, squeeze=False
    )

    explain_cls_name = idx_to_class(explain_cls_idx)

    for row, ref in enumerate(ref_images):
        img_cpu = ref["img_cpu"]
        gt_np = ref["gt_np"]
        image_id = ref["image_id"]
        gt_val = int(gt_np[explain_cls_idx])

        orig_np = to_numpy(img_cpu[:3].permute(1, 2, 0))
        if orig_np.max() > 1:
            orig_np = orig_np / (orig_np.max() + 1e-8)

        # Bboxes for EXPLAIN_CLASS (shared across all columns)
        bboxes = get_bboxes_for_image(
            bbox_df, image_id, class_id=explain_cls_idx, mode=BBOX_MODE
        )

        # --- Col 0: Original X-ray ---
        ax = axes[row, 0]
        ax.imshow(orig_np)
        if bboxes:
            draw_bboxes_on_ax(ax, bboxes, color="lime", lw=2)
        ax.set_xticks([])
        ax.set_yticks([])

        # Show all GT-positive classes in the title
        gt_pos_names = [idx_to_class(c) for c in range(NUM_CLASSES) if gt_np[c] > 0.5]
        gt_str = ", ".join(gt_pos_names) if gt_pos_names else "none"
        id_short = image_id[:12] if len(image_id) > 12 else image_id

        if row == 0:
            ax.set_title(
                f"Original\nGT+: {gt_str}\n{id_short}",
                fontsize=8,
                fontweight="bold",
            )
        else:
            ax.set_title(f"GT+: {gt_str}\n{id_short}", fontsize=8)

        # Border: green if explain class is GT+, grey otherwise
        _set_box(ax, color="green" if gt_val else "#aaaaaa", lw=2.0)

        # --- Col 1+: Experiment columns ---
        for col_idx, col_result in enumerate(column_results):
            col_data = col_result["data"][row]
            col_cfg = col_result["config"]

            ax = axes[row, 1 + col_idx]
            _render_cell(ax, col_data["result"], bboxes=bboxes)

            sig_val = col_data["sigmoid"][explain_cls_idx]
            pred_pos = sig_val >= sigmoid_threshold

            # Title color coding: TP/FP/FN/TN
            if gt_val == 1 and pred_pos:
                title_color = "green"  # true positive
            elif gt_val == 0 and pred_pos:
                title_color = "red"  # false positive
            elif gt_val == 1 and not pred_pos:
                title_color = "orange"  # false negative
            else:
                title_color = "black"  # true negative

            if row == 0:
                col_label = _get_column_label(col_cfg)
                ax.set_title(
                    f"{col_label}\nsig={sig_val:.3f}",
                    fontsize=8,
                    color=title_color,
                    fontweight="bold" if pred_pos else "normal",
                )
            else:
                ax.set_title(
                    f"sig={sig_val:.3f}",
                    fontsize=9,
                    color=title_color,
                    fontweight="bold" if pred_pos else "normal",
                )

            # Border: green if TP, red if FP, orange if FN, grey if TN
            border_colors = {
                (1, True): "green",
                (0, True): "red",
                (1, False): "orange",
                (0, False): "#aaaaaa",
            }
            _set_box(ax, color=border_colors[(gt_val, pred_pos)], lw=2.0)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig


# ===================================================================
# Main
# ===================================================================
def run_comparison():
    if not EXPERIMENT_COLUMNS:
        print("ERROR: EXPERIMENT_COLUMNS is empty. Add experiment-method pairs.")
        return

    # Validate methods
    for col_cfg in EXPERIMENT_COLUMNS:
        method = col_cfg["method"]
        if method not in ALL_METHODS:
            print(f"ERROR: Unknown method '{method}'. Available: {sorted(ALL_METHODS)}")
            return

    # Resolve EXPLAIN_CLASS
    explain_cls_idx = resolve_explain_class(EXPLAIN_CLASS)
    explain_cls_name = idx_to_class(explain_cls_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bbox_df = load_bbox_df(BBOX_CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Print settings
    print("=" * 60)
    print("Comparison settings (VinBigData):")
    print(
        f"  EXPLAIN_CLASS      = {EXPLAIN_CLASS!r} -> idx {explain_cls_idx} ({explain_cls_name})"
    )
    print(f"  REQUIRE_GT_POSITIVE= {REQUIRE_GT_POSITIVE}")
    print(f"  MIN_CONFIDENCE     = {MIN_CONFIDENCE!r}")
    print(f"  MAX_SEARCH         = {MAX_SEARCH}")
    print(f"  NUM_IMAGES         = {NUM_IMAGES}")
    print(f"  sigmoid_threshold  = {sigmoid_threshold}")
    print(f"  bg_opacity         = {bg_opacity}")
    print(f"  BBOX_MODE          = {BBOX_MODE!r}")
    print(f"  CAM_OVERLAY_STYLE  = {CAM_OVERLAY_STYLE!r}")
    print(f"  OUT_DIR            = {OUT_DIR}")
    print(f"  Columns ({len(EXPERIMENT_COLUMNS)}):")
    for i, c in enumerate(EXPERIMENT_COLUMNS):
        label = _get_column_label(c).replace("\n", " | ")
        print(f"    [{i + 1}] {label}")
    print("=" * 60)

    # Load reference dataset from first experiment
    print("Loading reference dataset from first experiment...")
    ref_exp = Experiment(EXPERIMENT_COLUMNS[0]["exp_path"])
    ref_dm = ref_exp.get_datamodule()
    ref_dm.setup("val")
    ref_dataset = ref_dm.val_dataloader().dataset
    print(f"Reference dataset: {len(ref_dataset)} samples")

    # Select images
    print(f"\nSelecting images for {explain_cls_name} (class {explain_cls_idx})...")
    ref_images = select_images(
        ref_dataset, EXPERIMENT_COLUMNS[0]["exp_path"], explain_cls_idx, device
    )

    if not ref_images:
        print("No images selected. Exiting.")
        return

    # For each experiment column: load model + dataset, compute explanations
    column_results = []
    for col_cfg in EXPERIMENT_COLUMNS:
        exp_path = col_cfg["exp_path"]
        method = col_cfg["method"]
        smooth = col_cfg.get("smooth", 15)

        exp = Experiment(exp_path)
        model = exp.load_trained_model(reload="last")
        model.to(device)
        model.eval()

        is_bcos = _is_bcos_model(model)
        model_name = type(model).__name__
        bcos_str = "B-cos" if is_bcos else "baseline"
        print(
            f"  Column: {os.path.basename(exp_path)}"
            f" + {method} ({model_name}, {bcos_str})"
        )

        # Load this experiment's own dataset (handles B-cos 6ch vs baseline 3ch)
        dm = exp.get_datamodule()
        dm.setup("val")
        dataset = dm.val_dataloader().dataset

        col_data = []
        for ref_img in ref_images:
            idx = ref_img["idx"]
            item = dataset[idx]
            img_cpu = item[0].detach().cpu()

            # Get predictions (multilabel: sigmoid)
            logits, sigmoid_probs = get_predictions_multilabel(
                model, img_cpu, device, is_bcos
            )

            orig_np = to_numpy(img_cpu[:3].permute(1, 2, 0))
            if orig_np.max() > 1:
                orig_np = orig_np / (orig_np.max() + 1e-8)

            # Compute explanation for EXPLAIN_CLASS
            result = compute_explanation(
                model,
                img_cpu,
                explain_cls_idx,
                method,
                device,
                smooth,
                orig_np,
                is_bcos,
            )

            col_data.append(
                {
                    "result": result,
                    "logits": logits,
                    "sigmoid": sigmoid_probs,
                }
            )

        column_results.append(
            {
                "config": col_cfg,
                "data": col_data,
                "is_bcos": is_bcos,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot
    gt_req_str = "GT+" if REQUIRE_GT_POSITIVE else "any"
    title = (
        f"Comparison — {explain_cls_name} (class {explain_cls_idx})"
        f" — {gt_req_str} — {len(ref_images)} images"
    )

    fig = plot_comparison_grid(
        ref_images, column_results, bbox_df, explain_cls_idx, title=title
    )

    # Generate filename
    cls_safe = explain_cls_name.replace(" ", "").replace("/", "-")
    methods_str = "_".join(c["method"] for c in EXPERIMENT_COLUMNS)
    fname = f"comparison_vinbig_{cls_safe}_{len(ref_images)}imgs_{methods_str}.pdf"
    savepath = os.path.join(OUT_DIR, fname)

    try:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {savepath}")
    except Exception as e:
        print(f"\nFailed: {savepath}: {e}")
    finally:
        plt.close(fig)

    print("\nAll done.")


run_comparison()
