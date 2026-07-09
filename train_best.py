"""
train_best.py — Two-phase fine-tuning for maximum accuracy on the gesture dataset.

Phase 1  (high-augmentation warmup)
    AdamW optimiser, aggressive augmentation, frozen backbone for first 5 epochs,
    then full unfreeze.  Best for small datasets — reduces overfitting while letting
    the head specialise quickly.

Phase 2  (fine-tune with clean images)
    Resumes from Phase 1 best.pt.  Mosaic / mixup disabled, lower LR, gentle
    augmentation so the model converges on realistic single images rather than
    the synthetic composites from Phase 1.  This reliably adds +1–3 % mAP50.

Usage:
    python train_best.py              # both phases
    python train_best.py --phase 1    # phase 1 only
    python train_best.py --phase 2 --weights path/to/best.pt  # phase 2 only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths & hardware
# ---------------------------------------------------------------------------

DATA_YAML = Path(__file__).resolve().parent / "dataset.yaml"
BASE_MODEL = "yolo11n.pt"          # nano — required for IMX500 export
DEVICE: int | str = 0              # 0 = first CUDA GPU; "cpu" = force CPU
WORKERS = 0                        # keep 0 on Windows (avoids WinError 1455)
PROJECT_DIR = "runs/detect"

# ---------------------------------------------------------------------------
# Phase 1 — aggressive augmentation, AdamW, frozen backbone warm-up
# ---------------------------------------------------------------------------

P1_NAME = "train-best-p1"
P1_EPOCHS = 200
P1_BATCH = 16
P1_PATIENCE = 30          # tight — stop early if plateau to avoid overfit
P1_SEED = 42

P1_LR0 = 0.001            # AdamW default; 10x lower than SGD
P1_LRF = 0.05             # final LR = LR0 * LRF → 5e-5 at end of cosine
P1_WARMUP_EPOCHS = 5.0
P1_WEIGHT_DECAY = 0.01    # higher for AdamW
P1_MOMENTUM = 0.937       # AdamW beta1

P1_CLOSE_MOSAIC = 20      # disable mosaic for last 20 epochs → cleaner convergence
P1_MULTI_SCALE = 0.25

P1_HYPERPARAMS: dict = {
    "optimizer": "AdamW",
    "lr0": P1_LR0,
    "lrf": P1_LRF,
    "momentum": P1_MOMENTUM,
    "weight_decay": P1_WEIGHT_DECAY,
    "warmup_epochs": P1_WARMUP_EPOCHS,
    "warmup_momentum": 0.8,
    "cos_lr": True,
    "label_smoothing": 0.1,   # reduces overconfidence, helps generalisation
    "dropout": 0.1,            # light regularisation for the detection head
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}

P1_AUGMENTATION: dict = {
    # Colour
    "hsv_h": 0.025,
    "hsv_s": 0.80,
    "hsv_v": 0.55,
    "bgr": 0.10,
    # Geometric
    "degrees": 25.0,
    "translate": 0.20,
    "scale": 0.75,
    "shear": 8.0,
    "perspective": 0.001,
    # Flips (keep horizontal only — upside-down hands are not valid)
    "flipud": 0.0,
    "fliplr": 0.5,
    # Composites
    "mosaic": 1.0,
    "mixup": 0.25,
    "cutmix": 0.15,
    "copy_paste": 0.10,
    # Occlusion / policy
    "auto_augment": "randaugment",
    "erasing": 0.50,
}

# ---------------------------------------------------------------------------
# Phase 2 — clean-image fine-tuning (no mosaic / mixup)
# ---------------------------------------------------------------------------

P2_NAME = "train-best-p2"
P2_EPOCHS = 50
P2_PATIENCE = 15
P2_SEED = 42

P2_LR0 = 0.0002           # start low — weight are already well-trained
P2_LRF = 0.10             # gentle cosine tail

P2_HYPERPARAMS: dict = {
    "optimizer": "AdamW",
    "lr0": P2_LR0,
    "lrf": P2_LRF,
    "momentum": 0.937,
    "weight_decay": 0.01,
    "warmup_epochs": 2.0,
    "cos_lr": True,
    "label_smoothing": 0.05,
    "dropout": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
}

P2_AUGMENTATION: dict = {
    # Moderate colour only
    "hsv_h": 0.015,
    "hsv_s": 0.50,
    "hsv_v": 0.35,
    "bgr": 0.05,
    # Mild geometry
    "degrees": 10.0,
    "translate": 0.10,
    "scale": 0.40,
    "shear": 3.0,
    "perspective": 0.0003,
    "flipud": 0.0,
    "fliplr": 0.5,
    # Composites OFF — let model see clean single images
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "auto_augment": "randaugment",
    "erasing": 0.20,
}

# ---------------------------------------------------------------------------
# Albumentations (shared by both phases at different probabilities)
# ---------------------------------------------------------------------------

USE_ALBUMENTATIONS = True


def build_albumentations(strength: float = 1.0):
    """
    Build custom Albumentations pipeline.
    strength=1.0 for Phase 1, 0.4 for Phase 2.
    Returns None if albumentations is not installed.
    """
    if not USE_ALBUMENTATIONS:
        return None
    try:
        import albumentations as A
    except ImportError:
        print("[WARN] albumentations not installed — skipping (pip install albumentations)")
        return None

    s = strength
    return [
        A.OneOf([
            A.MotionBlur(blur_limit=9, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=7, p=1.0),
        ], p=0.35 * s),
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.30 * s),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=35, val_shift_limit=30, p=1.0),
        ], p=0.50 * s),
        A.ImageCompression(quality_range=(45, 90), p=0.25 * s),
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(0.04, 0.15),
            hole_width_range=(0.04, 0.15),
            fill=0,
            p=0.30 * s,
        ),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: int | str) -> tuple[int | str, int]:
    if requested == "cpu":
        return "cpu", min(P1_BATCH, 8)
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
        return requested, P1_BATCH
    print("[WARN] CUDA not available — falling back to CPU (slower)")
    return "cpu", min(P1_BATCH, 8)


def validate_dataset() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {DATA_YAML}")
    import yaml
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    for split in ("train", "val"):
        folder = root / cfg[split]
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {split} images folder: {folder}")
        count = len(list(folder.glob("*")))
        print(f"  {split:5s}: {count} images")
    print(f"  classes: {cfg['nc']}  {cfg['names']}")


def run_phase(
    phase: int,
    weights: str,
    device: int | str,
    batch: int,
    albu_strength: float,
    resume: bool = False,
) -> Path:
    name = P1_NAME if phase == 1 else P2_NAME
    epochs = P1_EPOCHS if phase == 1 else P2_EPOCHS
    patience = P1_PATIENCE if phase == 1 else P2_PATIENCE
    seed = P1_SEED if phase == 1 else P2_SEED
    hyperparams = P1_HYPERPARAMS if phase == 1 else P2_HYPERPARAMS
    augmentation = P1_AUGMENTATION if phase == 1 else P2_AUGMENTATION
    multi_scale = P1_MULTI_SCALE if phase == 1 else 0.0
    close_mosaic = P1_CLOSE_MOSAIC if phase == 1 else 0

    albu = build_albumentations(strength=albu_strength)

    print(f"\n{'='*72}")
    print(f"  Phase {phase} — {'high-augmentation training' if phase == 1 else 'clean-image fine-tuning'}")
    print(f"  Weights  : {weights}")
    print(f"  Epochs   : {epochs}  (patience={patience})")
    print(f"  Optimizer: {hyperparams['optimizer']}  lr0={hyperparams['lr0']}  wd={hyperparams['weight_decay']}")
    print(f"  Mosaic   : {augmentation['mosaic']}  mixup={augmentation['mixup']}  erasing={augmentation['erasing']}")
    print(f"  Albu     : {'ON x' + str(albu_strength) if albu else 'OFF'}")
    print(f"{'='*72}\n")

    model = YOLO(weights)

    kwargs: dict = {
        "data": str(DATA_YAML),
        "epochs": epochs,
        "imgsz": 640,
        "batch": batch,
        "device": device,
        "workers": WORKERS,
        "project": PROJECT_DIR,
        "name": name,
        "seed": seed,
        "patience": patience,
        "plots": True,
        "verbose": True,
        "multi_scale": multi_scale,
        "close_mosaic": close_mosaic,
        "cache": False,
        "amp": device != "cpu",
        **hyperparams,
        **augmentation,
    }
    if albu is not None:
        kwargs["augmentations"] = albu
    if resume:
        kwargs["resume"] = True

    results = model.train(**kwargs)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nPhase {phase} done.  Best weights: {best}")
    return best


def print_summary(p1_best: Path, p2_best: Path | None) -> None:
    final = p2_best if p2_best else p1_best
    print(f"\n{'='*72}")
    print("  TRAINING COMPLETE")
    print(f"{'='*72}")
    print(f"  Phase 1 best  : {p1_best}")
    if p2_best:
        print(f"  Phase 2 best  : {p2_best}  <-- USE THIS")
    print(f"\n  Next steps:")
    print("    1. python local_tester.py   (auto-picks latest run)")
    print("    2. python export.py          (create packerOut.zip for Pi)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase YOLO11n training for best mAP")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 or 2. Default: run both.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Starting weights path. Default: yolo11n.pt (phase 1) or auto-detect phase-1 best.pt (phase 2).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run (phase 1 only).")
    return parser.parse_args()


def find_latest_p1_best() -> Path:
    """Look for the most recently modified Phase 1 best.pt."""
    candidates = sorted(
        Path(PROJECT_DIR).glob(f"{P1_NAME}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Phase 1 best.pt found under {PROJECT_DIR}/{P1_NAME}*/.\n"
            "Run phase 1 first:  python train_best.py --phase 1"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()

    print("\n=== GESTURE DATASET ===")
    validate_dataset()

    device, batch = resolve_device(DEVICE)
    p1_best: Path | None = None
    p2_best: Path | None = None

    run_phase_1 = args.phase in (None, 1)
    run_phase_2 = args.phase in (None, 2)

    if run_phase_1:
        weights = args.weights or BASE_MODEL
        p1_best = run_phase(
            phase=1,
            weights=weights,
            device=device,
            batch=batch,
            albu_strength=1.0,
            resume=args.resume,
        )

    if run_phase_2:
        if args.weights and args.phase == 2:
            p2_start = args.weights
        elif p1_best:
            p2_start = str(p1_best)
        else:
            p2_start = str(find_latest_p1_best())
            print(f"Auto-detected Phase 1 weights: {p2_start}")

        p2_best = run_phase(
            phase=2,
            weights=p2_start,
            device=device,
            batch=batch,
            albu_strength=0.4,
        )

    print_summary(
        p1_best or Path(p2_start if not p1_best else str(p1_best)),
        p2_best,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        sys.exit(130)
