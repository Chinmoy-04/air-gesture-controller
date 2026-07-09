"""
pi_tester.py — Raspberry Pi 5 + AI Camera (IMX500) gesture controller.

Inference runs on the IMX500 sensor NPU (not the Pi CPU). The Pi receives
bounding boxes via Picamera2 and runs the shared gesture_engine pipeline.

Prerequisites on Pi:
    sudo apt install -y python3-picamera2 imx500-all imx500-tools
    sudo apt install -y xdotool wmctrl python3-xlib playerctl
    # X11 session recommended for reliable OS keyboard/mouse control

Run on the Pi desktop (HDMI monitor recommended for full resolution):

    python3 pi_tester.py \\
        --model best_imx_model/out/network.rpk \\
        --labels best_imx_model/labels.txt

Keys:  P = cycle profile  |  M = fullscreen mode  |  Q = quit
Lite UI (default): 1280x720.  Pass --full-ui for 1440x900 rendering.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyautogui

import gesture_engine as ge
from gesture_engine import (
    DRAW_CENTROID_TRAIL,
    INFERENCE_IOU,
    MAX_DETECTIONS,
    MIRROR_PREVIEW,
    MOUSE_POINTER_CLASS,
    PYAUTOGUI_FAILSAFE,
    STATIC_GESTURE_CLASSES,
    SWIPE_TRACKING_CLASS,
    Detection,
    GesturePipeline,
    cycle_control_profile,
    cycle_fullscreen_mode,
    pick_best_from_list,
)
from gesture_ui import (
    WINDOW_TITLE,
    UI_LITE,
    apply_profile_window_mode,
    compose_display,
    configure_ui_lite,
    draw_centroid_trail,
    setup_display_window,
    wire_os_action_hooks,
)
from os_actions import print_linux_capabilities, start_window_tracker, stop_window_tracker

try:
    from picamera2 import MappedArray, Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
except ImportError as exc:
    raise SystemExit(
        "picamera2 is required on the Raspberry Pi. Install with:\n"
        "  sudo apt install -y python3-picamera2 imx500-all imx500-tools\n"
    ) from exc


@dataclass(frozen=True)
class ParsedImxDetection:
    """One IMX500 detection after coordinate scaling to the display frame."""
    class_name: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]


def load_labels(path: Path) -> list[str]:
    """Load class names from labels.txt (one name per line, IMX export format)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


_bbox_debug_done = False  # Print raw bbox values once for coordinate debugging


def parse_imx_outputs(
    imx500: IMX500,
    metadata: dict,
    intrinsics: NetworkIntrinsics,
    labels: list[str],
    threshold: float,
    iou: float,
    max_detections: int,
    last_detections: list[ParsedImxDetection],
    frame_w: int,
    frame_h: int,
) -> list[ParsedImxDetection]:
    """Convert IMX500 tensor outputs to display-frame bounding boxes.

    Uses manual scaling instead of imx500.convert_inference_coords() because
    custom Ultralytics YOLO exports don't always provide compatible intrinsics.
    """
    global _bbox_debug_done
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=threshold,
            iou_thres=iou,
            max_out_dets=max_detections,
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        parsed: list[ParsedImxDetection] = []
        for box, score, category in zip(boxes, scores, classes):
            if float(score) <= threshold:
                continue
            class_id = int(category)
            if class_id < 0 or class_id >= len(labels):
                continue
            # nanodet scale_boxes returns x0,y0,x1,y1 in [0,1]
            x0, y0, x1, y1 = [float(v) for v in box]
            px = max(0, int(x0 * frame_w))
            py = max(0, int(y0 * frame_h))
            pw = max(1, int((x1 - x0) * frame_w))
            ph = max(1, int((y1 - y0) * frame_h))
            parsed.append(ParsedImxDetection(labels[class_id], float(score), (px, py, pw, ph)))
        return parsed

    # Standard path ─ boxes shape (N, 4)
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

    # Compute scale factors: normalise tensor coords to [0,1] then scale to frame
    if intrinsics.bbox_normalization:
        # boxes already divided by input_h elsewhere — treat as [0,1]
        boxes = boxes / input_h
        sx, sy = float(frame_w), float(frame_h)
    else:
        # raw pixel coords in tensor space
        sx = float(frame_w) / float(input_w)
        sy = float(frame_h) / float(input_h)

    parsed = []
    for box, score, category in zip(boxes, scores, classes):
        if float(score) <= threshold:
            continue
        class_id = int(category)
        if class_id < 0 or class_id >= len(labels):
            continue

        if not _bbox_debug_done:
            print(f"[BBOX] raw={[round(float(v),4) for v in box]}  "
                  f"norm={intrinsics.bbox_normalization}  order={intrinsics.bbox_order}  "
                  f"input={input_w}x{input_h}  frame={frame_w}x{frame_h}", flush=True)
            _bbox_debug_done = True

        # Unpack based on coordinate order
        if intrinsics.bbox_order == "xy":
            a0, b0, a1, b1 = [float(v) for v in box]  # x0,y0,x1,y1
            px = max(0, int(a0 * sx))
            py = max(0, int(b0 * sy))
            pw = max(1, int((a1 - a0) * sx))
            ph = max(1, int((b1 - b0) * sy))
        else:  # yx
            a0, b0, a1, b1 = [float(v) for v in box]  # y0,x0,y1,x1
            px = max(0, int(b0 * sx))
            py = max(0, int(a0 * sy))
            pw = max(1, int((b1 - b0) * sx))
            ph = max(1, int((a1 - a0) * sy))

        parsed.append(ParsedImxDetection(labels[class_id], float(score), (px, py, pw, ph)))

    return parsed


def mirror_parsed_detections(detections: list[ParsedImxDetection], frame_w: int) -> list[ParsedImxDetection]:
    """Flip bbox x-coords to match the horizontally mirrored preview frame."""
    mirrored: list[ParsedImxDetection] = []
    for det in detections:
        x, y, w, h = det.bbox_xywh
        mx = frame_w - x - w
        mirrored.append(ParsedImxDetection(det.class_name, det.confidence, (mx, y, w, h)))
    return mirrored


def to_engine_detections(parsed: list[ParsedImxDetection]) -> list[Detection]:
    """Adapt Pi detections into gesture_engine.Detection objects (centroid + bbox size)."""
    engine_dets: list[Detection] = []
    for det in parsed:
        x, y, w, h = det.bbox_xywh
        cx, cy = x + w // 2, y + h // 2
        engine_dets.append(Detection(det.class_name, det.confidence, (cx, cy), float(w), float(h)))
    return engine_dets


def to_bgr(frame: np.ndarray) -> np.ndarray:
    """Picamera2 RGB888 is already BGR in memory. Just drop alpha if present."""
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3].copy()
    return frame  # RGB888 = BGR in memory, no conversion needed


def annotate_frame(
    frame: np.ndarray,
    detections: list[ParsedImxDetection],
    *,
    in_place: bool = False,
) -> np.ndarray:
    annotated = frame if in_place else frame.copy()
    line_type = cv2.LINE_8
    for det in detections:
        x, y, w, h = det.bbox_xywh
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x + 4, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            line_type,
        )
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Air-gesture controller for Raspberry Pi AI Camera")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("best_imx_model/out/network.rpk"),
        help="Path to network.rpk (from imx500-package)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("best_imx_model/labels.txt"),
        help="Path to labels.txt (one class name per line)",
    )
    parser.add_argument("--fps", type=int, default=15, help="Camera frame rate")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Detection confidence threshold (INT8 models often need 0.55-0.65)",
    )
    parser.add_argument("--iou", type=float, default=INFERENCE_IOU, help="NMS IoU threshold")
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS, help="Max detections per frame")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, default=None,
                        help="Override bbox normalization from model intrinsics")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default=None,
                        help="Override bbox coordinate order from model intrinsics")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 or 1)")
    parser.add_argument(
        "--full-ui",
        action="store_true",
        help="Use full-quality UI (default on Pi is lightweight 1280x720)",
    )
    parser.add_argument("--print-intrinsics", action="store_true", help="Print network intrinsics and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} labels: {labels}")

    ge.CONFIDENCE_THRESHOLD = args.threshold

    configure_ui_lite(not args.full_ui)
    if UI_LITE:
        ge.DRAW_CENTROID_TRAIL = False

    pyautogui.FAILSAFE = PYAUTOGUI_FAILSAFE
    pyautogui.PAUSE = 0

    imx500 = IMX500(str(args.model))
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    if intrinsics.task and intrinsics.task != "object detection":
        print(f"Warning: network task is '{intrinsics.task}', expected object detection", file=sys.stderr)

    intrinsics.task = "object detection"
    intrinsics.labels = labels
    intrinsics.inference_rate = args.fps
    # Only override bbox settings if user explicitly passed the flag;
    # otherwise let the model's baked-in intrinsics control them.
    if args.bbox_normalization is not None:
        intrinsics.bbox_normalization = args.bbox_normalization
    if args.bbox_order is not None:
        intrinsics.bbox_order = args.bbox_order
    intrinsics.update_with_defaults()
    # Ultralytics YOLO11 IMX export defaults (used when model has no intrinsics)
    if intrinsics.bbox_normalization is None:
        intrinsics.bbox_normalization = True
    if intrinsics.bbox_order is None:
        intrinsics.bbox_order = "xy"
    print(f"bbox_normalization={intrinsics.bbox_normalization}  bbox_order={intrinsics.bbox_order}")

    if args.print_intrinsics:
        print(intrinsics)
        return

    picam2 = Picamera2(imx500.camera_num)
    # Picamera2 "RGB888" stores bytes as B,G,R in memory — exactly what OpenCV expects.
    # "BGR888" is the opposite (R,G,B) and will appear blue in OpenCV. Don't change this.
    config = picam2.create_preview_configuration(
        main={"format": "RGB888"},
        controls={"FrameRate": args.fps},
        buffer_count=12,
    )

    print(f"Loading IMX500 model: {args.model}")
    print(f"Pipeline: conf>={args.threshold}, static_vote={ge.STATIC_VOTE_FRAMES} frames")
    print(f"UI: {'full' if args.full_ui else 'lite (1280x720)'}")
    print(f"Control profile: {ge.control_profile_summary()}  (press P to cycle)")
    print("Starting AI camera loop. Press Q to quit.\n")

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    import sys as _sys
    if _sys.platform.startswith("linux"):
        print_linux_capabilities()

    # Verify pyautogui cursor control works on this display
    try:
        pyautogui.position()
        print("pyautogui cursor control: OK")
    except Exception as e:
        print(f"WARNING: pyautogui cursor control unavailable ({e})")
        print("  Install python3-xlib:  sudo apt install python3-xlib")

    pipeline = GesturePipeline()
    setup_display_window(WINDOW_TITLE)
    wire_os_action_hooks(WINDOW_TITLE)
    # Background thread tracks last-focused non-gesture window every 500 ms.
    # This is used by OS actions (close window, seek) to refocus the target app.
    start_window_tracker(WINDOW_TITLE)
    last_parsed: list[ParsedImxDetection] = []

    try:
        while True:
            # --- Capture frame; inference already ran on the IMX500 NPU ---
            request = picam2.capture_request()
            metadata = request.get_metadata()

            with MappedArray(request, "main") as m:
                frame = to_bgr(m.array.copy())

            request.release()

            if MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]

            # Read bounding boxes from IMX500 metadata (not from host-side YOLO)
            last_parsed = parse_imx_outputs(
                imx500,
                metadata,
                intrinsics,
                labels,
                args.threshold,
                args.iou,
                args.max_detections,
                last_parsed,
                frame_w,
                frame_h,
            )

            if MIRROR_PREVIEW:
                last_parsed = mirror_parsed_detections(last_parsed, frame_w)
            engine_dets = to_engine_detections(last_parsed)

            # Split detections into the three gesture pipeline lanes
            static_detection = pick_best_from_list(engine_dets, STATIC_GESTURE_CLASSES)
            index_detection = pick_best_from_list(engine_dets, frozenset({SWIPE_TRACKING_CLASS}))
            mouse_detection = pick_best_from_list(engine_dets, frozenset({MOUSE_POINTER_CLASS}))

            state = pipeline.process_frame(
                static_detection, index_detection, mouse_detection, frame_w, frame_h,
            )

            annotated = annotate_frame(frame, last_parsed, in_place=False)
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
        stop_window_tracker()
        picam2.stop()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("Camera released. Goodbye.")


if __name__ == "__main__":
    main()
