"""
enhancement.py
===============
Test-time image enhancement utility shared by the evaluation scripts in
this module.

Rationale
---------
In addition to reporting standard validation metrics on the raw dataset
(`compute_metrics.py`), we separately evaluate the trained detector under
a lightweight, non-learned enhancement transform (CLAHE contrast
normalization + unsharp masking). This is a deliberately different
mechanism from the learned CycleGAN enhancement used at *training* time
(`../cyclegan/`): applying it only at evaluation time isolates how much of
the reported gain in detection confidence is attributable to the trained
detector's learned representations vs. to simple test-time contrast/detail
enhancement, which is a standard ablation-style control in image-quality
literature.
"""

from __future__ import annotations

import cv2
import numpy as np


def clahe_unsharp_enhance(
    image_rgb: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 1.5,
    unsharp_weight: float = -0.5,
) -> np.ndarray:
    """Apply CLAHE contrast enhancement (on the L channel of LAB color
    space) followed by unsharp masking.

    Args:
        image_rgb: HxWx3 RGB image (uint8).
        clahe_clip_limit: CLAHE contrast-clipping threshold.
        clahe_tile_grid_size: CLAHE local tile grid size.
        unsharp_sigma: Gaussian blur sigma used to build the unsharp mask.
        unsharp_amount: weight applied to the enhanced image in the
            weighted sum (sharpening strength).
        unsharp_weight: weight applied to the blurred image (negative,
            standard unsharp-mask formulation).

    Returns:
        HxWx3 RGB image (uint8) after enhancement.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    blurred = cv2.GaussianBlur(enhanced_rgb, (0, 0), unsharp_sigma)
    sharpened = cv2.addWeighted(enhanced_rgb, unsharp_amount, blurred, unsharp_weight, 0)
    return sharpened
