"""
Implementation of Energy-based Pointing Game proposed in Score-CAM.
Adjusted by Marcel Kleinmann
Device-safe fixes + bounds clamping.
"""

import torch


def _clamp_bbox_xyxy(bbox, H: int, W: int):
    # bbox = (x1, y1, x2, y2) in pixel coords
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # clamp to image bounds
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))

    # ensure proper ordering and non-empty box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # avoid empty region (optional: keep empty => return None)
    if x2 == x1 or y2 == y1:
        return None

    return x1, y1, x2, y2


def energy_point_game(bbox, saliency_map, threshold=None):
    """
    bbox: (x1,y1,x2,y2)
    saliency_map: [H,W] tensor (CPU or CUDA)
    """
    if not isinstance(saliency_map, torch.Tensor):
        saliency_map = torch.as_tensor(saliency_map)

    # Ensure 2D
    if saliency_map.ndim != 2:
        raise ValueError(f"saliency_map must be 2D [H,W], got shape {tuple(saliency_map.shape)}")

    H, W = saliency_map.shape
    bbox = _clamp_bbox_xyxy(bbox, H=H, W=W)
    if bbox is None:
        return torch.tensor(0.0, device=saliency_map.device, dtype=saliency_map.dtype)

    x1, y1, x2, y2 = bbox

    if threshold is not None:
        max_val = saliency_map.max()
        thresh_val = threshold * max_val
        saliency_map = torch.where(
            saliency_map >= thresh_val,
            saliency_map,
            torch.zeros_like(saliency_map),
        )

    # IMPORTANT: create mask on same device + dtype
    mask = torch.zeros((H, W), device=saliency_map.device, dtype=saliency_map.dtype)
    mask[y1:y2, x1:x2] = 1

    energy_bbox = (saliency_map * mask).sum()
    energy_whole = saliency_map.sum()

    # avoid division by zero if map is all zeros
    if energy_whole.abs() < 1e-12:
        return torch.tensor(0.0, device=saliency_map.device, dtype=saliency_map.dtype)

    return energy_bbox / energy_whole


def energy_point_game_recall(bbox, saliency_map, threshold=0.0):
    """
    "Recall" variant as in your code.
    bbox: (x1,y1,x2,y2)
    saliency_map: [H,W] tensor
    """
    if not isinstance(saliency_map, torch.Tensor):
        saliency_map = torch.as_tensor(saliency_map)

    if saliency_map.ndim != 2:
        raise ValueError(f"saliency_map must be 2D [H,W], got shape {tuple(saliency_map.shape)}")

    H, W = saliency_map.shape
    bbox = _clamp_bbox_xyxy(bbox, H=H, W=W)
    if bbox is None:
        return torch.tensor(0.0, device=saliency_map.device, dtype=saliency_map.dtype)

    x1, y1, x2, y2 = bbox

    bounding_box_map = saliency_map[y1:y2, x1:x2]

    if threshold is not None:
        max_val = saliency_map.max()
        thresh_val = threshold * max_val
        bounding_box_map = torch.where(
            bounding_box_map >= thresh_val,
            bounding_box_map,
            torch.zeros_like(bounding_box_map),
        )

    full_bbox_mask = torch.zeros((H, W), device=saliency_map.device, dtype=saliency_map.dtype)
    full_bbox_mask[y1:y2, x1:x2] = 1
    energy_bbox = (saliency_map * full_bbox_mask).abs().sum()

    if energy_bbox.abs() < 1e-12:
        return torch.tensor(0.0, device=saliency_map.device, dtype=saliency_map.dtype)

    return bounding_box_map.sum() / energy_bbox


def energy_point_game_mask(mask, saliency_map, threshold=0):
    """
    mask: Precomputed mask with 1s in ALL target regions, shape [H,W]
    saliency_map: Explanation heatmap, shape [H,W]
    """
    if not isinstance(saliency_map, torch.Tensor):
        saliency_map = torch.as_tensor(saliency_map)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    if mask.shape != saliency_map.shape:
        raise AssertionError(f"Mask/saliency shape mismatch: {mask.shape} vs {saliency_map.shape}")

    # Move mask to same device/dtype
    mask = mask.to(device=saliency_map.device, dtype=saliency_map.dtype)

    if threshold is not None:
        max_val = saliency_map.max()
        thresh_val = threshold * max_val
        saliency_map = torch.where(
            saliency_map >= thresh_val,
            saliency_map,
            torch.zeros_like(saliency_map),
        )

    total_energy = saliency_map.sum()
    if total_energy.abs() < 1e-12:
        return 0.0

    masked_energy = (saliency_map * mask).sum()
    return (masked_energy / total_energy).item()
