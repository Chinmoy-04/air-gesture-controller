"""
Export YOLO11 Nano hand-gesture weights to Sony IMX500 format (packerOut.zip).

Run on Windows (training machine):

    python export.py

Output folder (next to best.pt):

    runs/detect/runs/detect/train-augmented-3/weights/best_imx_model/
        packerOut.zip
        labels.txt
        ...

Transfer best_imx_model/ to the Pi, then package:

    imx500-package -i packerOut.zip -o out
"""

from __future__ import annotations

from pathlib import Path

import ultralytics.engine.exporter

# Patch LINUX flag so imxconv can run on Windows during export.
ultralytics.engine.exporter.LINUX = True

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
_RUN_ROOT = ROOT / "runs" / "detect"


def resolve_model_path() -> Path:
    """Return the most recently modified best.pt under runs/detect/."""
    candidates = sorted(
        _RUN_ROOT.glob("**/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No best.pt found under {_RUN_ROOT}. Train a model first (e.g. python train_best.py)."
    )


def main() -> None:
    model_path = resolve_model_path()
    data_yaml = ROOT / "dataset.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    # INT8 post-training quantisation; imxconv-pt calibrates on dataset.yaml val split.
    # Requires Java 17+, imx500-converter, and the LINUX flag patch above on Windows.
    print("Starting INT8 IMX export (calibration uses dataset.yaml val split)...")
    model.export(format="imx", int8=True, data=str(data_yaml), imgsz=640)

    imx_dir = model_path.parent / "best_imx_model"
    packer = imx_dir / "packerOut.zip"
    labels = imx_dir / "labels.txt"

    print("\nExport complete.")
    print(f"  Folder:      {imx_dir}")
    print(f"  packerOut:   {packer} {'OK' if packer.is_file() else 'MISSING — check imxconv install'}")
    print(f"  labels.txt:  {labels} {'OK' if labels.is_file() else 'MISSING'}")
    print("\nNext: copy best_imx_model/ to the Pi and run:")
    print("  imx500-package -i packerOut.zip -o out")


if __name__ == "__main__":
    main()
