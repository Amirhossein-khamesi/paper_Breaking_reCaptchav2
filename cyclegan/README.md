# CycleGAN-based Image Enhancement Module

This directory contains the image-enhancement component of the *Breaking
reCAPTCHAv2* pipeline. It implements an attention-augmented, spectrally
normalized CycleGAN that translates raw reCAPTCHA image tiles into a visually
enhanced domain, used both (a) as a data-augmentation / domain-adaptation
step prior to detector training, and (b) as a source of additional training
images for the pseudo-labeling stage.

## 1. Motivation

Publicly available reCAPTCHA-style image corpora are small, visually noisy,
and exhibit inconsistent illumination, compression artifacts, and low
effective resolution for small object classes (e.g. traffic lights, fire
hydrants, crosswalks). Rather than relying solely on classical image
enhancement (e.g. histogram equalization), we learn a data-driven
enhancement mapping with a generative adversarial network, so that the
transformation is optimized jointly for (i) adversarial realism, (ii)
content preservation via cycle-consistency, and (iii) perceptual similarity
in a pretrained feature space. The resulting enhanced images are then used
to generate additional pseudo-labeled training examples for the YOLOv8
detector (see `../pseudo_labeling/` and `../yolov8/`).

## 2. Architecture

| Component | Design | Reference |
|---|---|---|
| Generator (`CycleGenerator`) | ResNet-style encoder–bottleneck–decoder, 9 residual blocks, reflection padding, instance normalization | Johnson et al. (2016); Zhu et al. (2017) |
| Attention | Convolutional Block Attention Module (channel + spatial attention) inside every residual block | Woo et al. (2018) |
| Discriminator (`PatchDiscriminator`) | 70×70 PatchGAN, 5 convolutional layers | Isola et al. (2017); Zhu et al. (2017) |
| Stabilization | Spectral normalization on every convolution in the generator's residual blocks and in the discriminator | Miyato et al. (2018) |
| Perceptual loss | Multi-layer VGG16 (ImageNet-pretrained, frozen) feature L1 distance | Johnson et al. (2016) |
| Adversarial loss | Least-squares GAN (LSGAN) objective | Mao et al. (2017) |

All images are processed at 256×256 resolution with pixel values normalized
to `[-1, 1]`; the generator's final `Tanh` activation matches this range.

### Why CBAM + spectral normalization

The training corpus available for this task is orders of magnitude smaller
than the datasets typically used to train CycleGAN (e.g. Cityscapes, Horse2Zebra,
tens of thousands of images). Under such low-data regimes, vanilla CycleGAN
training is prone to discriminator over-confidence and generator mode
collapse. Spectral normalization constrains the Lipschitz constant of each
discriminator (and generator residual) layer, which empirically stabilizes
the minimax game under small-batch, small-dataset conditions. CBAM is added
so the generator's capacity is spent selectively on informative spatial
regions and channels rather than uniformly re-texturing the whole tile,
which is particularly relevant given the small-object nature of several
target classes.

## 3. Objective function

For a raw input image `x` (domain A) and its CycleGAN-transformed
counterpart `G_A2B(x)` (domain B):

```
L_G  = L_adv(D_B, G_A2B(x))
     + lambda_cycle * ||G_B2A(G_A2B(x)) - x||_1
     + lambda_perc  * L_perceptual(G_A2B(x), x)

L_D  = 0.5 * [ (D_B(x) - 1)^2 + (D_B(G_A2B(x).detach()))^2 ]
```

Default hyperparameters (see `train_cyclegan.py --help`):

| Hyperparameter | Default | Notes |
|---|---|---|
| `lambda_cycle` | 10.0 | Weight of the cycle-consistency term, following the original CycleGAN paper |
| `lambda_perc` | 1.0 | Weight of the VGG16 perceptual term |
| Learning rate | 2e-4 | Adam, β1=0.5, β2=0.999 (standard GAN configuration) |
| Epochs | 10 | For the reported experiments on the reduced training subset |
| Batch size | 1 | Per-image instance normalization statistics, consistent with CycleGAN's original design |
| Residual blocks | 9 | Standard for 256×256 inputs |

## 4. Repository contents

```
cyclegan/
├── README.md                  # this file
├── models.py                  # CBAM, ResidualBlock, CycleGenerator,
│                               # PatchDiscriminator, PerceptualLoss
├── dataset.py                 # lightweight image dataset / loader utilities
├── train_cyclegan.py          # training entry point (CLI, checkpointing, CSV logging)
├── generate_synthetic.py      # inference: produce enhanced images from a trained checkpoint
└── cyclegan_recaptcha.ipynb   # original exploratory Colab notebook (full pipeline)
```

The `.ipynb` notebook is kept for provenance and end-to-end reproducibility
(it also contains the YOLOv8 training/evaluation cells, which are being
factored out into `../yolov8/` and `../pseudo_labeling/`). The `.py` files
in this folder are the cleaned, modular, and CLI-driven versions of the
CycleGAN-specific cells (model definitions, training loop, and image
generation), intended for direct reuse and citation in the paper's code
availability statement.

## 5. Usage

### 5.1 Environment

```bash
pip install torch torchvision timm opencv-python pillow tqdm numpy
```

### 5.2 Training

```bash
python train_cyclegan.py \
    --data_dir /path/to/raw_recaptcha_images \
    --output_dir ./checkpoints \
    --epochs 10 \
    --batch_size 1 \
    --max_samples 500
```

Outputs:
- `checkpoints/cyclegan_epoch<N>.pt` — full training state (`G_A2B`, `G_B2A`,
  `D_B`, both optimizers, and the run configuration) for reproducibility and
  resuming.
- `checkpoints/loss_log.csv` — per-epoch generator/discriminator loss, used
  to produce the training-curve figure in the paper (see `../scripts/`).

### 5.3 Generating enhanced images for downstream stages

```bash
python generate_synthetic.py \
    --checkpoint ./checkpoints/cyclegan_epoch10.pt \
    --data_dir /path/to/raw_recaptcha_images \
    --output_dir ./synthetic_enhanced \
    --num_images 200
```

The resulting images in `synthetic_enhanced/` are the direct input to the
pseudo-label generation stage in `../pseudo_labeling/`, which assigns
YOLO-format bounding-box labels using a pretrained detector before the
images are merged back into the YOLOv8 training set.

## 6. Reproducibility notes

- A fixed random seed (`--seed`, default 42) is applied to `random`,
  `numpy`, and `torch` at the start of training.
- All checkpoints store the full argument namespace used to launch the run,
  so any reported result can be traced back to its exact configuration.
- The current experimental configuration performs *single-domain*
  enhancement (domain B targets are constructed from domain A at training
  time); this is documented explicitly in `train_cyclegan.py` and should be
  described in the paper's methodology section to avoid confusion with the
  classic two-domain CycleGAN (unpaired image-to-image translation) setting.

## 7. References

- Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired
  Image-to-Image Translation using Cycle-Consistent Adversarial Networks*.
  ICCV.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for
  Real-Time Style Transfer and Super-Resolution*. ECCV.
- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image
  Translation with Conditional Adversarial Networks*. CVPR.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM:
  Convolutional Block Attention Module*. ECCV.
- Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). *Spectral
  Normalization for Generative Adversarial Networks*. ICLR.
- Mao, X., Li, Q., Xie, H., Lau, R. Y. K., Wang, Z., & Smolley, S. P.
  (2017). *Least Squares Generative Adversarial Networks*. ICCV.
