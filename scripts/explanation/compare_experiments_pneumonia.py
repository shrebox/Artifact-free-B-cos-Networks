#!/usr/bin/env python3
"""
Cross-experiment comparison for Pneumonia (binary classification).

Layout:
  Rows    = test images (selected by GT class)
  Col 0   = original X-ray with GT bounding boxes and labels
  Col 1+  = one column per (experiment, explanation_method) pair

Each entry in EXPERIMENT_COLUMNS independently pairs an experiment path with
an explanation method. Supports mixing B-cos and baseline experiments.

Explanation methods (for the 'method' field in EXPERIMENT_COLUMNS):
  B-cos inherent (shows N/A on baseline models):
    "bcos_contribs"  — contribution heatmap (bwr colormap)
    "bcos_rgba"      — B-cos colored explanation (RGBA)
    "bcos_overlay"   — contribution overlay on X-ray
  Post-hoc CAM (works on any model):
    "GradCAM", "LayerCAM", "HiResCAM", "GradCAM++", "AblationCAM"
"""

import argparse
import os
import sys

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
    #     "exp_path": "experiments/Pneumonia/bcosification/resnet_50_...",
    #     "method": "GradCAM",
    #     "smooth": 15,
    #     "label": None,
    # },
    {
        "exp_path": "experiments/Pneumonia/bcosification/resnet_50_Baseline_pretrained_lr1e-4_bs16_heavy_oversamp_cosineLR_fold4-seed=1337",
        "method": "GradCAM",
        "smooth": 15,
        "label": "Baseline GradCAM",
    },
    {
        "exp_path": "experiments/Pneumonia/bcosification/resnet_50_Baseline_pretrained_lr1e-4_bs16_heavy_oversamp_cosineLR_fold4-seed=1337",
        "method": "LayerCAM",
        "smooth": 15,
        "label": "Baseline LayerCAM",
    },
    {
        "exp_path": "experiments/Pneumonia/bcosification/resnet_50_Baseline_pretrained_lr1e-4_bs16_heavy_oversamp_cosineLR_fold4-seed=1337",
        "method": "FinerCAM",
        "smooth": 15,
        "label": "Baseline FinerCAM",
    },
    {
        "exp_path": "experiments/Pneumonia/bcosification/resnet_50_Bcos_pretrained_lr1e-4_bs16_heavy_oversamp_cosineLR_fold4-seed=1337",
        "method": "bcos_overlay",
        "smooth": 15,
        "label": "B-cos Inherent",
    },
    {
        "exp_path": "experiments/Pneumonia/bcosification/resnet_50_BlurPool_pretrained_lr1e-4_bs16_heavy_oversamp_cosineLR_fold4-seed=1337",
        "method": "bcos_overlay",
        "smooth": 15,
        "label": "B-cos BlurPool",
    },
    {
        "exp_path": "experiments/Pneumonia/bcosification/resnet_50_FLCPool_pretrained_lr1e-4_bs16_heavy_oversamp_cosineLR_fold4-seed=1337",
        "method": "bcos_overlay",
        "smooth": 15,
        "label": "B-cos ASAP Pool",
    },
]

# GT class(es) to generate comparison PDFs for (one PDF per class).
CLASSES_TO_COMPARE = [1]  # 0 = No Pneumonia, 1 = Pneumonia

# Which class logit to explain:
#   "gt"   — explain the ground-truth class (same target across all models, best for
#            fair comparison since all columns explain the same thing)
#   "pred" — explain each model's own prediction (shows what each model attends to;
#            target may differ across columns if models disagree)
#   int    — explain a fixed class index regardless of GT or prediction
EXPLAIN_TARGET = "gt"

NUM_IMAGES = 3  # number of image rows per PDF
bg_opacity = 1  # background X-ray opacity in overlay cells (0=white, 1=full)

# --- Bbox settings (same as per-experiment script) ---
BBOX_CSV_PATH = os.getenv("PNEUMONIA_CSV_PATH")
ORIGINAL_SIZE = 1024  # RSNA DICOM images are 1024x1024
DISPLAY_SIZE = 224  # model input / explanation size
BBOX_MODE = "all"  # "first", "all", "merge", "largest"
DRAW_BBOXES = True  # set False to hide GT bounding-box overlays in PDFs and PNGs

# --- Overlay style (same as per-experiment script) ---
# "jet"     : B-cos overlay uses bwr+magnitude, CAM rows use JET+uniform
# "bcos"    : everything B-cos style — CAM overlays also use bwr+magnitude
# "all_jet" : everything JET style — B-cos overlay also uses JET+magnitude
CAM_OVERLAY_STYLE = "all_jet"

# Output directory
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "comparisons_pneumonia"
)

# ===================================================================
# Constants
# ===================================================================
PNEUMONIA_CLASSES = {0: "No Pneumonia", 1: "Pneumonia"}

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
    return PNEUMONIA_CLASSES.get(int(idx), str(int(idx)))


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
# Bounding box helpers (RSNA Pneumonia)
# ===================================================================
def load_bbox_df(csv_path):
    if not os.path.exists(csv_path):
        print(f"WARNING: bbox CSV not found at {csv_path}, skipping bbox overlays.")
        return None
    df = pd.read_csv(csv_path)
    required = ["patientId", "Target", "x", "y", "width", "height"]
    for col in required:
        if col not in df.columns:
            print(f"WARNING: bbox CSV missing column '{col}', skipping bbox overlays.")
            return None
    print(f"Loaded bbox CSV: {len(df)} rows from {csv_path}")
    return df


def _get_all_bboxes_for_patient(bbox_df, patient_id):
    if bbox_df is None:
        return []
    scale = DISPLAY_SIZE / ORIGINAL_SIZE
    rows = bbox_df[
        (bbox_df["patientId"].astype(str) == str(patient_id)) & (bbox_df["Target"] == 1)
    ]
    bboxes = []
    for _, row in rows.iterrows():
        if pd.notna(row["x"]):
            x = float(row["x"]) * scale
            y = float(row["y"]) * scale
            w = float(row["width"]) * scale
            h = float(row["height"]) * scale
            bboxes.append({"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h})
    return bboxes


def get_bboxes_for_patient(bbox_df, patient_id, mode="largest"):
    all_bboxes = _get_all_bboxes_for_patient(bbox_df, patient_id)
    if not all_bboxes:
        return []
    if mode == "first":
        return all_bboxes[:1]
    elif mode == "merge":
        n = len(all_bboxes)
        return [
            {
                "x_min": sum(b["x_min"] for b in all_bboxes) / n,
                "y_min": sum(b["y_min"] for b in all_bboxes) / n,
                "x_max": sum(b["x_max"] for b in all_bboxes) / n,
                "y_max": sum(b["y_max"] for b in all_bboxes) / n,
            }
        ]
    elif mode == "largest":
        best = max(
            all_bboxes,
            key=lambda b: (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"]),
        )
        return [best]
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
def get_predictions_binary(model, img_cpu, device, is_bcos):
    """
    Get binary classification predictions.
    B-cos: inside explanation_mode (matches explanation inference).
    Baseline: standard no_grad.
    Returns (pred_class, probs_list).
    """
    model.eval()
    if is_bcos:
        with torch.enable_grad(), model.explanation_mode():
            img_t = img_cpu[None].to(device).requires_grad_(True)
            out = model(img_t)
            probs = F.softmax(out.detach(), dim=1)[0].cpu().tolist()
            pred = int(out.argmax(dim=1).item())
    else:
        with torch.no_grad():
            img_t = img_cpu[None].to(device)
            out = model(img_t)
            probs = F.softmax(out.detach(), dim=1)[0].cpu().tolist()
            pred = int(out.argmax(dim=1).item())
    return pred, probs


def compute_explanation(
    model, img_cpu, target_cls, method, device, smooth, orig_np, is_bcos
):
    """
    Compute a single explanation for one image.

    Returns one of:
      ("contribs", contribs_2d, vrange)  — for bcos_contribs (needs cmap rendering)
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
            grad = torch.autograd.grad(out[0, target_cls], img_t)[0]
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
            # comparison_categories=[1] because Pneumonia is binary (2 classes);
            # the default [1,2,3] causes IndexError for <4 classes.
            grayscale_cam = cam(
                input_tensor=img_t, targets=None, comparison_categories=[1]
            )
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
# Individual image saving (standalone PNGs, no axes/titles)
# ===================================================================
def save_individual_images(
    ref_images, column_results, bbox_df, out_folder, save_dpi=200
):
    """
    Save each subplot cell as a standalone high-quality PNG (no axes/titles).

    Files are named  row{ri}_original.png  and  row{ri}_col{ci}_{tag}.png
    where tag is derived from the column label.
    """
    os.makedirs(out_folder, exist_ok=True)

    for ri, ref in enumerate(ref_images):
        img_cpu = ref["img_cpu"]
        patient_id = ref["patient_id"]

        orig_np = to_numpy(img_cpu[:3].permute(1, 2, 0))
        if orig_np.max() > 1:
            orig_np = orig_np / (orig_np.max() + 1e-8)

        bboxes = []
        if DRAW_BBOXES and patient_id is not None and bbox_df is not None:
            bboxes = get_bboxes_for_patient(bbox_df, patient_id, mode=BBOX_MODE)

        h, w = orig_np.shape[:2]
        scale = 3.5  # same as per-cell size in plot_comparison_grid
        fig_w = scale
        fig_h = scale * (h / w)

        # --- Original column ---
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=save_dpi)
        ax.imshow(orig_np)
        if bboxes:
            draw_bboxes_on_ax(ax, bboxes, color="lime", lw=2)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            os.path.join(out_folder, f"row{ri}_original.png"),
            dpi=save_dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        # --- Experiment columns ---
        for ci, col_result in enumerate(column_results):
            col_data = col_result["data"][ri]
            col_cfg = col_result["config"]
            result = col_data["result"]

            col_tag = (
                _get_column_label(col_cfg)
                .lower()
                .replace("\n", "_")
                .replace(" ", "_")
                .replace("-", "_")
            )
            fname = f"row{ri}_col{ci}_{col_tag}.png"

            if result is None:
                # N/A panel (bcos method on baseline model) — skip
                continue

            kind = result[0]
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=save_dpi)

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
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(
                os.path.join(out_folder, fname),
                dpi=save_dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)

    n_saved = len(ref_images) * (1 + len(column_results))
    print(f"  Saved {n_saved} individual images -> {out_folder}")


# ===================================================================
# Plotting
# ===================================================================
def plot_comparison_grid(ref_images, column_results, bbox_df, cls, title=None):
    """
    Create the comparison figure.

    ref_images:     list of dicts {img_cpu, gt, patient_id, idx}
    column_results: list of dicts {config, data: [{result, pred, probs, target_cls}], is_bcos}
    bbox_df:        DataFrame or None
    cls:            the GT class being compared
    """
    nrows = len(ref_images)
    ncols = 1 + len(column_results)  # original column + experiment columns

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5), dpi=200, squeeze=False
    )

    for row, ref in enumerate(ref_images):
        img_cpu = ref["img_cpu"]
        gt = ref["gt"]
        patient_id = ref["patient_id"]

        orig_np = to_numpy(img_cpu[:3].permute(1, 2, 0))
        if orig_np.max() > 1:
            orig_np = orig_np / (orig_np.max() + 1e-8)

        # Look up bboxes once per image (shared across all columns)
        bboxes = []
        if DRAW_BBOXES and patient_id is not None and bbox_df is not None:
            bboxes = get_bboxes_for_patient(bbox_df, patient_id, mode=BBOX_MODE)

        # --- Col 0: Original X-ray ---
        ax = axes[row, 0]
        ax.imshow(orig_np)
        if bboxes:
            draw_bboxes_on_ax(ax, bboxes, color="lime", lw=2)
        ax.set_xticks([])
        ax.set_yticks([])

        gt_name = idx_to_class(gt)
        id_str = f"\n{patient_id}" if patient_id else ""
        if row == 0:
            ax.set_title(
                f"Original\nGT: {gt_name}{id_str}", fontsize=9, fontweight="bold"
            )
        else:
            ax.set_title(f"GT: {gt_name}{id_str}", fontsize=9)
        _set_box(ax, color="green" if gt == cls else "#aaaaaa", lw=2.0)

        # --- Col 1+: Experiment columns ---
        for col_idx, col_result in enumerate(column_results):
            col_data = col_result["data"][row]
            col_cfg = col_result["config"]

            ax = axes[row, 1 + col_idx]
            _render_cell(ax, col_data["result"], bboxes=bboxes)

            pred = col_data["pred"]
            probs = col_data["probs"]
            correct = pred == gt

            pred_name = idx_to_class(pred)
            pred_prob = probs[pred]

            # Column header on row 0, prediction info on all rows
            if row == 0:
                col_label = _get_column_label(col_cfg)
                ax.set_title(
                    f"{col_label}\nPred: {pred_name} ({pred_prob:.2f})",
                    fontsize=8,
                )
            else:
                ax.set_title(f"Pred: {pred_name} ({pred_prob:.2f})", fontsize=9)

            # Border: green if correct, red if wrong
            if correct:
                _set_box(ax, color="green", lw=2.0)
            else:
                _set_box(ax, color="red", lw=2.5)
                # Show GT annotation on misclassifications
                ax.text(
                    0.02,
                    0.02,
                    f"GT: {gt_name}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color="green",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="green", lw=1),
                )

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


# ===================================================================
# Main
# ===================================================================
def run_comparison(tag=""):
    if not EXPERIMENT_COLUMNS:
        print("ERROR: EXPERIMENT_COLUMNS is empty. Add experiment-method pairs.")
        return

    # Validate methods
    for col_cfg in EXPERIMENT_COLUMNS:
        method = col_cfg["method"]
        if method not in ALL_METHODS:
            print(f"ERROR: Unknown method '{method}'. Available: {sorted(ALL_METHODS)}")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bbox_df = load_bbox_df(BBOX_CSV_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Print settings
    print("=" * 60)
    print("Comparison settings (Pneumonia):")
    print(f"  CLASSES_TO_COMPARE = {CLASSES_TO_COMPARE}")
    print(f"  EXPLAIN_TARGET     = {EXPLAIN_TARGET!r}")
    print(f"  NUM_IMAGES         = {NUM_IMAGES}")
    print(f"  bg_opacity         = {bg_opacity}")
    print(f"  BBOX_MODE          = {BBOX_MODE!r}")
    print(f"  DRAW_BBOXES        = {DRAW_BBOXES}")
    print(f"  CAM_OVERLAY_STYLE  = {CAM_OVERLAY_STYLE!r}")
    print(f"  OUT_DIR            = {OUT_DIR}")
    print(f"  tag                = {tag!r}")
    print(f"  Columns ({len(EXPERIMENT_COLUMNS)}):")
    for i, c in enumerate(EXPERIMENT_COLUMNS):
        label = _get_column_label(c).replace("\n", " | ")
        print(f"    [{i + 1}] {label}")
    print("=" * 60)

    # Load reference dataset from first experiment (for image selection + display)
    print("Loading reference dataset from first experiment...")
    ref_exp = Experiment(EXPERIMENT_COLUMNS[0]["exp_path"])
    ref_dm = ref_exp.get_datamodule()
    ref_dm.config["return_id"] = True
    ref_dm.setup("val")
    ref_dataset = ref_dm.val_dataloader().dataset
    print(f"Reference dataset: {len(ref_dataset)} samples")

    for cls in CLASSES_TO_COMPARE:
        cls_name = idx_to_class(cls)
        print(f"\n{'=' * 40}")
        print(f"Class {cls}: {cls_name}")
        print(f"{'=' * 40}")

        # Select images with GT = cls
        class_indices = []
        for i in range(len(ref_dataset)):
            item = ref_dataset[i]
            if int(item[1]) == cls:
                class_indices.append(i)

        if not class_indices:
            print(f"  No images found for class {cls}, skipping.")
            continue

        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(class_indices), generator=rng).tolist()
        selected = [class_indices[i] for i in perm[:NUM_IMAGES]]

        # Load reference images
        ref_images = []
        for idx in selected:
            item = ref_dataset[idx]
            img, gt = item[0], item[1]
            patient_id = str(item[2]) if len(item) > 2 else None
            ref_images.append(
                {
                    "img_cpu": img.detach().cpu(),
                    "gt": int(gt),
                    "patient_id": patient_id,
                    "idx": idx,
                }
            )

        print(f"  Selected {len(ref_images)} images")

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
            dm.config["return_id"] = True
            dm.setup("val")
            dataset = dm.val_dataloader().dataset

            col_data = []
            for ref_img in ref_images:
                idx = ref_img["idx"]
                item = dataset[idx]
                img_cpu = item[0].detach().cpu()

                # Get predictions
                pred, probs = get_predictions_binary(model, img_cpu, device, is_bcos)

                # Determine explanation target
                if EXPLAIN_TARGET == "pred":
                    target_cls = pred
                elif EXPLAIN_TARGET == "gt":
                    target_cls = ref_img["gt"]
                else:
                    target_cls = int(EXPLAIN_TARGET)

                orig_np = to_numpy(img_cpu[:3].permute(1, 2, 0))
                if orig_np.max() > 1:
                    orig_np = orig_np / (orig_np.max() + 1e-8)

                result = compute_explanation(
                    model, img_cpu, target_cls, method, device, smooth, orig_np, is_bcos
                )

                col_data.append(
                    {
                        "result": result,
                        "pred": pred,
                        "probs": probs,
                        "target_cls": target_cls,
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
        explain_str = (
            f"target={EXPLAIN_TARGET}" if EXPLAIN_TARGET != "gt" else "target=GT class"
        )
        title = (
            f"Comparison — GT class {cls} ({cls_name})"
            f" — {explain_str} — {len(ref_images)} images"
        )

        fig = plot_comparison_grid(
            ref_images, column_results, bbox_df, cls, title=title
        )

        methods_str = "_".join(c["method"] for c in EXPERIMENT_COLUMNS)
        base_name = (
            f"comparison_pneumonia_GTclass{cls}_{len(ref_images)}imgs_{methods_str}"
        )
        if tag:
            base_name = f"{base_name}_{tag}"

        fname = f"{base_name}.pdf"
        savepath = os.path.join(OUT_DIR, fname)

        try:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")
            print(f"  Saved: {savepath}")
        except Exception as e:
            print(f"  Failed: {savepath}: {e}")
        finally:
            plt.close(fig)

        # Save each subplot cell as a standalone high-res PNG
        png_folder = os.path.join(OUT_DIR, base_name)
        save_individual_images(
            ref_images, column_results, bbox_df, png_folder, save_dpi=200
        )

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-experiment comparison for Pneumonia."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix appended to output PDF and PNG folder names "
        "(e.g. --tag ablation_v2).",
    )
    args = parser.parse_args()
    run_comparison(tag=args.tag)
