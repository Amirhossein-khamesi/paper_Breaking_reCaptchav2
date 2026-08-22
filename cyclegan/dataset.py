"""
dataset.py
==========
Minimal image dataset used to feed raw reCAPTCHA tiles into the CycleGAN
training loop.

The current experimental setup performs single-domain image enhancement
(domain A = raw reCAPTCHA tiles, domain B = enhanced tiles), so no paired
domain-B images are required at training time; B-domain targets are
implicitly defined through the adversarial + perceptual objective. This
class therefore only needs to serve domain-A images; see train_cyclegan.py
for how real_B is constructed during training.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def list_images(root_dir: str) -> List[str]:
    """Recursively collect all image paths under root_dir."""
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(paths)


class ReCaptchaImageDataset(Dataset):
    """Loads raw images for CycleGAN training.

    Args:
        image_paths: list of image file paths.
        image_size: target square resolution (default 256, matching the
            generator's expected input size).
        max_samples: optional cap on dataset size, used for fast
            experimentation on limited compute (e.g. Colab T4 GPUs).
    """

    def __init__(
        self,
        image_paths: List[str],
        image_size: int = 256,
        max_samples: Optional[int] = None,
    ):
        self.image_paths = image_paths[:max_samples] if max_samples else image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        return self.transform(pil_img)
