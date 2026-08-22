"""
generate_synthetic.py
======================
Loads a trained G_A2B generator checkpoint and produces enhanced synthetic
images from raw reCAPTCHA tiles. The output of this script is the input to
the pseudo-label generation stage (see ../pseudo_labeling/) and to the
augmented YOLOv8 training set (see ../yolov8/).

Usage:
    python generate_synthetic.py \
        --checkpoint ./checkpoints/cyclegan_epoch10.pt \
        --data_dir /content/yolo_dataset/images/train \
        --output_dir /content/synthetic_enhanced \
        --num_images 200
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from dataset import list_images
from models import CycleGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CycleGAN-enhanced synthetic images.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True,
                    help="Directory of raw source images to translate.")
    p.add_argument("--output_dir", type=str, default="./synthetic_enhanced")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_images", type=int, default=200)
    p.add_argument("--n_res_blocks", type=int, default=9)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = CycleGenerator(n_res_blocks=args.n_res_blocks).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["G_A2B"])
    generator.eval()
    print(f"Loaded generator from {args.checkpoint} (trained for {ckpt.get('epoch', '?')} epochs).")

    image_paths = list_images(args.data_dir)[: args.num_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    resize = transforms.Resize((args.image_size, args.image_size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    to_pil = transforms.ToPILImage()

    print(f"Generating {len(image_paths)} enhanced images -> {args.output_dir}")
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            img = Image.open(path).convert("RGB")
            img = resize(img)
            tensor = normalize(to_tensor(img)).unsqueeze(0).to(device)

            fake = generator(tensor)                      # output range [-1, 1]
            fake = fake.squeeze(0).cpu()
            fake = (fake + 1) / 2                          # rescale to [0, 1]

            out_img = to_pil(fake.clamp(0, 1))
            out_img.save(os.path.join(args.output_dir, f"enhanced_{i}.png"))

    print(f"Done. {len(image_paths)} enhanced images saved.")


if __name__ == "__main__":
    main()
