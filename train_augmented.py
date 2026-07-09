"""
train_augmented.py — YOLO11 Nano training with a robust augmentation pipeline

Embedded AI course project — Deggendorf Institute of Technology (Prof. Dr. Thomas Ewender)

Designed for the 6-class hand-gesture dataset (~270 images) used by the Air-Gesture
Touchless Controller.  Combines:

  1. Ultralytics built-in augmentations (HSV, geometric, mosaic, mixup, RandAugment, …)
  2. Custom Albumentations transforms (blur, noise, CLAHE, occlusion, compression)

Augmentation is intentionally aggressive because the dataset is small and must
generalize across laptop webcams and the Raspberry Pi 5 / Sony IMX500 deployment.

Usage:
    python train_augmented.py

Optional:
    pip install albumentations   # required for the custom transform stack
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths & hardware
# ---------------------------------------------------------------------------

# Dataset YAML (6 classes: fist_close, index_finger_up, open_palm, thumbs_down, thumbs_up, two_fingers)
DATA_YAML = Path(__file__).resolve().parent / "dataset.yaml"

# Pretrained backbone to fine-tune
BASE_MODEL = "yolo11n.pt"

# Compute device: 0 = first CUDA GPU, "cpu" = force CPU.
# When CUDA is unavailable (e.g. CPU-only torch wheel), training auto-falls back to CPU.
DEVICE: int | str = 0

# Data-loader workers.  Keep at 0 on Windows to avoid WinError 1455 paging-file issues.
WORKERS = 0

# Where Ultralytics stores run outputs (weights, plots, args.yaml, …)
PROJECT_DIR = "runs/detect"
RUN_NAME = "train-augmented"

# Set to a checkpoint path (e.g. runs/detect/train-augmented/weights/last.pt) to resume.
RESUME_CHECKPOINT: str | None = None

# ---------------------------------------------------------------------------
# Core training hyperparameters
# ---------------------------------------------------------------------------

EPOCHS = 150
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 50          # early-stopping patience (epochs without mAP improvement)
SEED = 42              # reproducibility

# Learning-rate schedule
COSINE_LR = True
LR0 = 0.01             # initial learning rate
LRF = 0.01             # final LR = LR0 * LRF
WARMUP_EPOCHS = 5.0
WEIGHT_DECAY = 0.0005

# Randomly vary input resolution each batch (fraction of IMAGE_SIZE, e.g. 0.25 → ±25 %)
MULTI_SCALE = 0.25

# Disable mosaic in the last N epochs so the model converges on realistic single images.
CLOSE_MOSAIC = 15

# Cache images in RAM for faster epoch times (enable if you have enough memory).
CACHE_IMAGES = False

# ---------------------------------------------------------------------------
# Layer 1 — Ultralytics built-in augmentation
# ---------------------------------------------------------------------------
# Probabilities are in [0.0, 1.0] unless noted otherwise.
# Tuned for hand gestures: strong colour / scale / lighting variation, no vertical flip.

ULTRALYTICS_AUGMENTATION: dict = {
    # --- Colour (simulate different webcams, room lighting, Pi camera exposure) ---
    "hsv_h": 0.025,       # hue jitter (fraction of colour wheel)
    "hsv_s": 0.80,        # saturation swing
    "hsv_v": 0.55,        # brightness swing
    "bgr": 0.10,          # occasional BGR channel swap (different camera drivers)

    # --- Geometric (hand distance, wrist angle, off-centre framing) ---
    "degrees": 20.0,      # rotation ± degrees
    "translate": 0.20,    # shift ± fraction of image
    "scale": 0.70,        # zoom ± fraction (0.7 → 0.3× … 1.7× apparent size)
    "shear": 6.0,         # shear ± degrees
    "perspective": 0.0008,  # mild perspective warp (webcam angle changes)

    # --- Flips ---
    "flipud": 0.0,        # disabled — upside-down gestures are not valid classes
    "fliplr": 0.5,        # horizontal mirror (matches mirrored inference in local_tester.py)

    # --- Composite augmentations (critical for small datasets) ---
    "mosaic": 1.0,        # stitch 4 images → richer backgrounds & scale diversity
    "mixup": 0.20,        # blend two images + labels
    "cutmix": 0.15,       # cut-and-paste rectangular regions between images
    "copy_paste": 0.10,   # paste object instances (helps single-hand detection)

    # --- Policy-based & occlusion ---
    "auto_augment": "randaugment",  # RandAugment policy (alternatives: "autoaugment", "augmix")
    "erasing": 0.45,      # random erasing — simulates partial hand occlusion
}

# ---------------------------------------------------------------------------
# Layer 2 — Custom Albumentations (optional but recommended)
# ---------------------------------------------------------------------------
# These REPLACE Ultralytics' default Albumentations block while Layer 1 stays active.
# Install: pip install albumentations

USE_ALBUMENTATIONS = True


def build_albumentations_pipeline():
    """
    Build a hand-gesture-oriented Albumentations transform list.

  Returns None when albumentations is not installed or USE_ALBUMENTATIONS is False.
    """
    if not USE_ALBUMENTATIONS:
        return None

    try:
        import albumentations as A
    except ImportError:
        print(
            "[WARN] albumentations not installed — skipping custom transform layer.\n"
            "       Install with: pip install albumentations"
        )
        return None

    # Compose several OneOf groups so only one variant fires per group per image.
    # Parameter names target albumentations 2.x (installed: 2.0.8).
    return [
        # Motion / defocus — swipe gestures and cheap webcam optics
        A.OneOf(
            [
                A.MotionBlur(blur_limit=9, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0),
            ],
            p=0.35,
        ),
        # Sensor noise — laptop webcam & Pi camera ISO grain
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ],
            p=0.30,
        ),
        # Exposure / contrast — uneven room lighting, auto-exposure pumping
        A.OneOf(
            [
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.35, contrast_limit=0.35, p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=18,
                    sat_shift_limit=35,
                    val_shift_limit=30,
                    p=1.0,
                ),
            ],
            p=0.50,
        ),
        # Compression artefacts — video-stream / JPEG re-encoding from capture pipeline
        A.ImageCompression(quality_range=(45, 90), p=0.25),
        # Partial occlusion — fingers behind mug, desk edge, another hand, etc.
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(0.04, 0.15),
            hole_width_range=(0.04, 0.15),
            fill=0,
            p=0.30,
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: int | str = DEVICE) -> tuple[int | str, int]:
    """
    Pick a valid Ultralytics device and batch size.

    Falls back to CPU when CUDA is requested but unavailable (common with
    CPU-only PyTorch wheels installed via uv/pip).
    """
    batch = BATCH_SIZE

    if requested == "cpu":
        return "cpu", min(batch, 8)

    wants_cuda = requested == "cuda" or (
        isinstance(requested, int) or (isinstance(requested, str) and str(requested).isdigit())
    )

    if wants_cuda and torch.cuda.is_available():
        return requested, batch

    if wants_cuda:
        print(
            "[WARN] CUDA device requested but not available.\n"
            "       torch.cuda.is_available(): False\n"
            "       Your venv has a CPU-only PyTorch build (torch-*+cpu).\n"
            "       Falling back to device='cpu'. For GPU training, reinstall PyTorch with CUDA:\n"
            "       https://pytorch.org/get-started/locally/\n"
        )
        return "cpu", min(batch, 8)

    return requested, batch


def validate_dataset(data_yaml: Path) -> None:
    """Fail fast when the dataset YAML or image folders are missing."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    import yaml

    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    train_images = root / cfg["train"]
    val_images = root / cfg["val"]

    if not train_images.is_dir():
        raise FileNotFoundError(f"Training images folder missing: {train_images}")
    if not val_images.is_dir():
        raise FileNotFoundError(f"Validation images folder missing: {val_images}")

    train_count = len(list(train_images.glob("*")))
    val_count = len(list(val_images.glob("*")))
    print(f"Dataset OK — train: {train_count} images, val: {val_count} images, classes: {cfg['nc']}")


def build_train_kwargs(albumentations_pipeline, device: int | str, batch: int) -> dict:
    """Assemble the full keyword-argument dict passed to model.train()."""
    kwargs = {
        # --- data & model ---
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": batch,
        "device": device,
        "workers": WORKERS,
        # --- run bookkeeping ---
        "project": PROJECT_DIR,
        "name": RUN_NAME,
        "seed": SEED,
        "patience": PATIENCE,
        "plots": True,
        "verbose": True,
        # --- optimisation ---
        "cos_lr": COSINE_LR,
        "lr0": LR0,
        "lrf": LRF,
        "warmup_epochs": WARMUP_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "multi_scale": MULTI_SCALE,
        "close_mosaic": CLOSE_MOSAIC,
        "cache": CACHE_IMAGES,
        "amp": device != "cpu",
        # --- augmentation layer 1 ---
        **ULTRALYTICS_AUGMENTATION,
    }

    if albumentations_pipeline is not None:
        kwargs["augmentations"] = albumentations_pipeline

    if RESUME_CHECKPOINT:
        kwargs["resume"] = True

    return kwargs


def print_augmentation_summary(
    albumentations_pipeline,
    device: int | str,
    batch: int,
) -> None:
    """Print a human-readable summary before training starts."""
    print("\n" + "=" * 72)
    print("  ROBUST AUGMENTATION TRAINING — configuration summary")
    print("=" * 72)
    print(f"  Base model     : {BASE_MODEL}")
    print(f"  Data           : {DATA_YAML}")
    print(f"  Epochs         : {EPOCHS}  (patience={PATIENCE}, close_mosaic={CLOSE_MOSAIC})")
    print(f"  Image size     : {IMAGE_SIZE}  (multi_scale={MULTI_SCALE})")
    print(f"  Batch / device : {batch} / {device}")
    print("-" * 72)
    print("  Ultralytics augmentations:")
    for key, value in ULTRALYTICS_AUGMENTATION.items():
        print(f"    {key:18s} = {value}")
    print("-" * 72)
    if albumentations_pipeline is not None:
        print(f"  Albumentations : {len(albumentations_pipeline)} transform groups ENABLED")
    else:
        print("  Albumentations : DISABLED (install albumentations or set USE_ALBUMENTATIONS=True)")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    validate_dataset(DATA_YAML)

    device, batch = resolve_device(DEVICE)
    albumentations_pipeline = build_albumentations_pipeline()
    print_augmentation_summary(albumentations_pipeline, device, batch)

    if RESUME_CHECKPOINT:
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        model = YOLO(RESUME_CHECKPOINT)
    else:
        model = YOLO(BASE_MODEL)

    train_kwargs = build_train_kwargs(albumentations_pipeline, device, batch)

    print("Starting training…\n")
    results = model.train(**train_kwargs)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"  Best weights : {best_weights}")
    print(f"  Results dir  : {results.save_dir}")
    print("\nNext steps:")
    print("  1. Test locally  → python local_tester.py  (update MODEL_PATH if needed)")
    print("  2. Export for Pi   → python export.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.", file=sys.stderr)
        sys.exit(130)
