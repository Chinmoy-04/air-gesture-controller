"""
local_tester.py — Windows laptop test harness (webcam + YOLO).

Runs YOLO11n inference on the laptop GPU/CPU, then feeds detections into the
shared gesture_engine pipeline and OpenCV UI.

Keys:  P = cycle profile  |  M = fullscreen mode  |  Q = quit

Model weights: auto-picks the newest runs/detect/**/best.pt
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pyautogui
from ultralytics import YOLO

from gesture_engine import (
    CENTROID_EMA_ALPHA,
    CONFIDENCE_THRESHOLD,
    DRAW_CENTROID_TRAIL,
    INFERENCE_IMGSZ,
    INFERENCE_IOU,
    MAX_DETECTIONS,
    MIRROR_PREVIEW,
    MOUSE_POINTER_CLASS,
    PYAUTOGUI_FAILSAFE,
    STATIC_GESTURE_CLASSES,
    SWIPE_THRESHOLD_X_FRAC,
    SWIPE_THRESHOLD_Y_FRAC,
    SWIPE_TRACKING_CLASS,
    STATIC_VOTE_FRAMES,
    GesturePipeline,
    cycle_control_profile,
    cycle_fullscreen_mode,
    pick_best_from_yolo,
)
from gesture_ui import (
    WINDOW_TITLE,
    apply_profile_window_mode,
    compose_display,
    draw_centroid_trail,
    setup_display_window,
    wire_os_action_hooks,
)

_RUN_ROOT = Path(__file__).resolve().parent / "runs" / "detect"
# Prefer newest best.pt: scan all run folders sorted by mtime, newest first.
# Falls back to the hard-coded list if the glob finds nothing.
_FALLBACK_CANDIDATES = (
    _RUN_ROOT / "runs" / "detect" / "train-augmented-3" / "weights" / "best.pt",
    _RUN_ROOT / "runs" / "detect" / "train-augmented-2" / "weights" / "best.pt",
    _RUN_ROOT / "train-augmented-2" / "weights" / "best.pt",
    _RUN_ROOT / "train-9" / "weights" / "best.pt",
)

WEBCAM_INDEX = 0


def resolve_model_path() -> Path:
    """Return the most recently modified best.pt under runs/detect/."""
    candidates = sorted(
        _RUN_ROOT.glob("**/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    for path in _FALLBACK_CANDIDATES:
        if path.is_file():
            return path
    return _FALLBACK_CANDIDATES[0]


def main() -> None:
    pyautogui.FAILSAFE = PYAUTOGUI_FAILSAFE
    pyautogui.PAUSE = 0

    model_path = resolve_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            "Model weights not found. Tried:\n" + "\n".join(f"  - {p}" for p in _FALLBACK_CANDIDATES)
        )

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))
    print(f"Classes: {list(model.names.values())}")
    print(
        f"Pipeline: conf>={CONFIDENCE_THRESHOLD}, static_vote={STATIC_VOTE_FRAMES} frames, "
        f"EMA alpha={CENTROID_EMA_ALPHA}, swipe scale={SWIPE_THRESHOLD_X_FRAC:.0%}/{SWIPE_THRESHOLD_Y_FRAC:.0%}"
    )
    from gesture_engine import control_profile_summary
    print(f"Control profile: {control_profile_summary()}  (press P to cycle)")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {WEBCAM_INDEX}. "
            "Check that no other application is using the camera."
        )

    pipeline = GesturePipeline()
    setup_display_window(WINDOW_TITLE)
    wire_os_action_hooks(WINDOW_TITLE)

    print("Starting webcam loop.")
    print("  Index finger up -> swipes  |  Two fingers -> mouse  |  q to quit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam — exiting.")
                break

            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]

            # Host-side YOLO inference (contrast with Pi where IMX500 NPU infers on-sensor)
            results = model.predict(
                source=frame,
                imgsz=INFERENCE_IMGSZ,
                conf=CONFIDENCE_THRESHOLD,
                iou=INFERENCE_IOU,
                max_det=MAX_DETECTIONS,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()

            # Route best detections into static / swipe / mouse lanes
            static_detection = pick_best_from_yolo(result, STATIC_GESTURE_CLASSES)
            index_detection = pick_best_from_yolo(result, frozenset({SWIPE_TRACKING_CLASS}))
            mouse_detection = pick_best_from_yolo(result, frozenset({MOUSE_POINTER_CLASS}))

            state = pipeline.process_frame(
                static_detection, index_detection, mouse_detection, frame_w, frame_h,
            )

            if DRAW_CENTROID_TRAIL and state.is_tracking:
                draw_centroid_trail(annotated, pipeline.centroid_queue)

            display = compose_display(
                annotated,
                state.active_gesture,
                state.last_action_label,
                state.last_action_time,
                state.is_tracking,
                state.mouse_active,
                state.mouse_sequence_armed,
                state.queue_len,
                state.static_vote_candidate,
                state.static_vote_streak,
            )

            cv2.imshow(WINDOW_TITLE, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break
            if key == ord("m"):
                mode = cycle_fullscreen_mode()
                print(f"Fullscreen mode -> {mode}")
            if key == ord("p"):
                profile = cycle_control_profile()
                apply_profile_window_mode(WINDOW_TITLE)
                print(f"Control profile -> {profile}")

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Webcam released. Goodbye.")


if __name__ == "__main__":
    main()
