"""
capture_training_data.py — Webcam capture for YOLO training data (auto-labeled).


This script uses your existing YOLO model to:
  1. Detect a target gesture in the webcam feed
  2. Save the frame + YOLO-format label (class cx cy w h) automatically

Output goes into `updated dataset/images/` and `updated dataset/labels/`,
then re-run prepare_training_data.py before training.

Usage:
    python capture_training_data.py --class thumbs_down
    python capture_training_data.py --class thumbs_down --manual-only
    python capture_training_data.py --class thumbs_down --conf 0.70 --interval 1.0

Controls:
    SPACE  — save current frame if target gesture is detected (manual)
    A      — toggle auto-capture on/off
    Q      — quit
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "updated dataset"
CLASSES_FILE = DATASET_DIR / "classes.txt"

MODEL_CANDIDATES = (
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "train-augmented-3" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "train-augmented-2" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "train-9" / "weights" / "best.pt",
)

WEBCAM_INDEX = 0
INFERENCE_IMGSZ = 640
MIRROR = True
AUTO_VOTE_FRAMES = 5  # consecutive frames before auto-save

# Preview window size (display only — saved images use full camera resolution).
PREVIEW_WINDOW_WIDTH = 1280
PREVIEW_WINDOW_HEIGHT = 720

# Requested webcam capture resolution (camera may round to nearest supported mode).
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


def load_class_names() -> list[str]:
    if not CLASSES_FILE.is_file():
        raise FileNotFoundError(f"Missing {CLASSES_FILE}")
    return [line.strip() for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No YOLO weights found. Train a model first or set --model.")


def xyxy_to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str:
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{class_id} {cx:.10f} {cy:.10f} {bw:.10f} {bh:.10f}"


def configure_webcam(cap: cv2.VideoCapture, width: int, height: int) -> tuple[int, int]:
    """Request a capture resolution; returns the actual width/height the camera reports."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def scale_preview_for_display(frame, max_w: int, max_h: int):
    """Scale frame to fit the preview window while keeping aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if scale >= 0.999:
        return frame
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def pick_target_box(result, target_class_id: int, conf_threshold: float):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_conf = -1.0
    for box in boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if cls_id != target_class_id or conf < conf_threshold:
            continue
        if conf > best_conf:
            best_conf = conf
            best = box.xyxy[0].tolist()
    return best


def save_sample(frame, xyxy: list[float], class_id: int, images_dir: Path, labels_dir: Path) -> str:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = f"cap_{uuid.uuid4().hex[:8]}"
    img_path = images_dir / f"{stem}.jpg"
    lbl_path = labels_dir / f"{stem}.txt"

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    label_line = xyxy_to_yolo_line(class_id, x1, y1, x2, y2, w, h)

    cv2.imwrite(str(img_path), frame)
    lbl_path.write_text(label_line + "\n", encoding="utf-8")
    return stem


def parse_args() -> argparse.Namespace:
    classes = load_class_names()
    parser = argparse.ArgumentParser(description="Capture auto-labeled YOLO training samples from webcam")
    parser.add_argument(
        "--class",
        dest="target_class",
        required=True,
        choices=classes,
        help="Gesture class to capture (e.g. thumbs_down)",
    )
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.65, help="Min detection confidence to save")
    parser.add_argument("--interval", type=float, default=1.5, help="Min seconds between auto-saves")
    parser.add_argument("--vote-frames", type=int, default=AUTO_VOTE_FRAMES, help="Frames in a row before auto-save")
    parser.add_argument("--manual-only", action="store_true", help="Disable auto-capture; use SPACE only")
    parser.add_argument("--camera", type=int, default=WEBCAM_INDEX, help="Webcam index")
    parser.add_argument("--output", type=Path, default=DATASET_DIR, help="Dataset root (images/ + labels/)")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH, help="Webcam capture width")
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT, help="Webcam capture height")
    parser.add_argument("--preview-width", type=int, default=PREVIEW_WINDOW_WIDTH, help="Preview window width")
    parser.add_argument("--preview-height", type=int, default=PREVIEW_WINDOW_HEIGHT, help="Preview window height")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_class_names()
    target_class_id = class_names.index(args.target_class)

    model_path = args.model if args.model else resolve_model_path()
    images_dir = args.output / "images"
    labels_dir = args.output / "labels"

    print(f"Model:  {model_path}")
    print(f"Target: {args.target_class} (class id {target_class_id})")
    print(f"Output: {images_dir}")
    print(f"Conf:   {args.conf:.0%}  |  Auto interval: {args.interval:.1f}s")
    print()
    print("Controls:  SPACE = save now  |  A = toggle auto  |  Q = quit")
    print()

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    actual_w, actual_h = configure_webcam(cap, args.width, args.height)
    print(f"Camera: {actual_w}x{actual_h}  |  Preview window: {args.preview_width}x{args.preview_height}")

    auto_enabled = not args.manual_only
    vote_streak = 0
    saved_count = 0
    last_save_time = 0.0
    last_xyxy: list[float] | None = None

    window = f"Capture: {args.target_class}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.preview_width, args.preview_height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if MIRROR:
                frame = cv2.flip(frame, 1)

            results = model.predict(
                source=frame,
                imgsz=INFERENCE_IMGSZ,
                conf=args.conf * 0.85,
                verbose=False,
            )
            result = results[0]
            preview = result.plot()

            xyxy = pick_target_box(result, target_class_id, args.conf)
            if xyxy is not None:
                last_xyxy = xyxy
                vote_streak += 1
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
            else:
                vote_streak = 0

            mode = "AUTO ON" if auto_enabled else "AUTO OFF"
            status = f"{mode} | saved: {saved_count} | streak: {vote_streak}/{args.vote_frames}"
            cv2.putText(preview, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                preview,
                f"Target: {args.target_class}  conf>={args.conf:.0%}",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            # Auto-save when gesture is stable for vote_streak frames and interval elapsed
            if (
                auto_enabled
                and xyxy is not None
                and vote_streak >= args.vote_frames
                and (time.monotonic() - last_save_time) >= args.interval
            ):
                stem = save_sample(frame, xyxy, target_class_id, images_dir, labels_dir)
                saved_count += 1
                last_save_time = time.monotonic()
                vote_streak = 0
                print(f"[AUTO] Saved {stem}.jpg + .txt  (total {saved_count})")

            cv2.imshow(window, scale_preview_for_display(preview, args.preview_width, args.preview_height))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("a"):
                auto_enabled = not auto_enabled
                vote_streak = 0
                print(f"Auto-capture: {'ON' if auto_enabled else 'OFF'}")
            if key == ord(" "):
                box = xyxy if xyxy is not None else last_xyxy
                if box is None:
                    print("[MANUAL] No detection — hold the gesture in frame first.")
                else:
                    stem = save_sample(frame, box, target_class_id, images_dir, labels_dir)
                    saved_count += 1
                    last_save_time = time.monotonic()
                    vote_streak = 0
                    print(f"[MANUAL] Saved {stem}.jpg + .txt  (total {saved_count})")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved_count} new samples in {args.output}")
        print("Next: python prepare_training_data.py  then  python train_augmented.py")


if __name__ == "__main__":
    main()
