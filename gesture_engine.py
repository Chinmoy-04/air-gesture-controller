"""
gesture_engine.py — Shared gesture detection pipeline and OS action mapping.

Used by local_tester.py (Windows + YOLO webcam) and pi_tester.py (Pi + IMX500).

Architecture
------------
Each camera frame produces up to three detection streams (static / swipe / mouse).
GesturePipeline.process_frame() runs them through parallel logic lanes:

  1. Static gestures  — 7-frame voting streak + 1.5 s cooldown → keyboard/OS action
  2. Swipe tracking   — index_finger_up centroid queue → seek / volume
  3. Mouse control    — two_fingers → relative cursor + click sequences

Control profiles (Presentation vs Desktop/OS) remap static gestures at runtime.
User preferences (profile, fullscreen mode) persist in .gesture_config.json.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyautogui

CONFIG_PATH = Path(__file__).resolve().parent / ".gesture_config.json"

# ---------------------------------------------------------------------------
# Pipeline configuration — tune these to balance responsiveness vs false triggers
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75   # Min detection score passed to the gesture pipeline
INFERENCE_IOU = 0.45          # NMS IoU when running YOLO on Windows (local_tester)
MAX_DETECTIONS = 1            # Only the single best box per frame is processed
COOLDOWN_SECONDS = 1.5        # Min gap between any two fired actions
VOLUME_STEP_COUNT = 10        # Windows only: repeated volume key presses per swipe

# Swipe thresholds scale with bounding-box size so distance is relative to hand size
SWIPE_THRESHOLD_X_FRAC = 0.28
SWIPE_THRESHOLD_Y_FRAC = 0.30
SWIPE_MIN_X = 55              # Absolute minimum horizontal displacement (pixels)
SWIPE_MIN_Y = 70
SWIPE_MIN_VELOCITY = 2.5      # Avg px/frame; filters slow drift from intentional swipes

# Index finger swipes control seek and volume (play/pause is closed fist).
CENTROID_QUEUE_LENGTH = 15    # Frames of centroid history for swipe detection
MIN_QUEUE_SAMPLES_FOR_SWIPE = 8
CENTROID_EMA_ALPHA = 0.35     # EMA smoothing for swipe centroid trail
STATIC_VOTE_FRAMES = 7        # Consecutive frames before a static gesture fires
TRACKING_GRACE_FRAMES = 5     # Frames to keep swipe state after brief detection loss
STATIC_BLOCKS_SWIPE = True    # Don't swipe while a static gesture is held

SWIPE_TRACKING_CLASS = "index_finger_up"
MOUSE_POINTER_CLASS = "two_fingers"
MOUSE_SEQUENCE_WINDOW_SECONDS = 1.2   # Time window for two_fingers → fist/palm click
MOUSE_SEQUENCE_VOTE_FRAMES = 4        # Frames to confirm click sequence gesture
MOUSE_EMA_ALPHA = 0.22                # Heavier smoothing than swipe (cursor stability)
CURSOR_SCREEN_LERP = 0.28             # Absolute-mode only: lerp toward target screen pos
MOUSE_GRACE_FRAMES = 12               # Hold cursor when two_fingers briefly lost
# Relative (trackpad) cursor mode: hand movement is treated as a delta from
# the cursor's current position rather than mapping hand position to screen
# position absolutely.  CURSOR_RELATIVE_SENSITIVITY controls how many screen
# pixels move per camera pixel of hand movement.  Higher = faster, more reach.
CURSOR_RELATIVE_MODE = True
CURSOR_RELATIVE_SENSITIVITY = 4.0

INFERENCE_IMGSZ = 640
DRAW_CENTROID_TRAIL = True

# Open Palm fullscreen modes — cycle at runtime with M in the app window.
FULLSCREEN_MODES: list[tuple[str, str, str]] = [
    ("f", "Video", "YouTube, VLC, HTML5"),
    ("f11", "PDF / Browser", "SumatraPDF, Edge, Chrome"),
    ("ctrl+l", "Acrobat", "Adobe Reader"),
]

_fullscreen_mode_index = 0

# Control profile — changes what static thumb/palm/fist gestures do.
# Cycle at runtime with P in the app window.
CONTROL_PROFILES: list[tuple[str, str]] = [
    ("presentation", "Presentation"),
    ("desktop", "Desktop / OS"),
]

_control_profile_index = 0


def _load_user_config() -> None:
    global _fullscreen_mode_index, _control_profile_index
    if not CONFIG_PATH.is_file():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        idx = int(data.get("fullscreen_mode_index", 0))
        if 0 <= idx < len(FULLSCREEN_MODES):
            _fullscreen_mode_index = idx
        # control_profile_index is intentionally NOT restored — always start
        # in Presentation mode so Desktop destructive actions never fire unexpectedly.
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass


def _save_user_config() -> None:
    try:
        # Only persist fullscreen mode — NOT the control profile.
        # Desktop/OS profile opens terminals and closes windows, so inheriting it
        # silently across sessions causes unintended destructive actions.
        # The profile always resets to Presentation on startup and must be set
        # explicitly each session with the P key.
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "fullscreen_mode_index": _fullscreen_mode_index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def get_fullscreen_mode() -> tuple[str, str, str]:
    """Return (key, short_label, hint) for the active fullscreen mode."""
    return FULLSCREEN_MODES[_fullscreen_mode_index]


def get_fullscreen_key() -> str:
    return get_fullscreen_mode()[0]


def fullscreen_binding_display() -> str:
    """Short ASCII badge for the legend (OpenCV-safe)."""
    key = get_fullscreen_key()
    if key == "ctrl+l":
        return "Ctrl+L"
    return key.upper()


def fullscreen_mode_summary() -> str:
    key, label, _ = get_fullscreen_mode()
    return f"{label} ({fullscreen_binding_display()})"


def cycle_fullscreen_mode() -> str:
    """Advance to the next fullscreen mode, save, and return a summary string."""
    global _fullscreen_mode_index
    _fullscreen_mode_index = (_fullscreen_mode_index + 1) % len(FULLSCREEN_MODES)
    _save_user_config()
    return fullscreen_mode_summary()


def get_control_profile() -> tuple[str, str]:
    """Return (profile_id, display_label)."""
    return CONTROL_PROFILES[_control_profile_index]


def control_profile_summary() -> str:
    return get_control_profile()[1]


def cycle_control_profile() -> str:
    """Cycle presentation <-> desktop/OS profile."""
    global _control_profile_index
    _control_profile_index = (_control_profile_index + 1) % len(CONTROL_PROFILES)
    _save_user_config()
    return control_profile_summary()


def action_label(action_key: str) -> str:
    if action_key == "open_palm" and get_control_profile()[0] == "presentation":
        return f"Toggle Fullscreen ({fullscreen_binding_display()})"
    if get_control_profile()[0] == "desktop" and action_key in DESKTOP_ACTIONS:
        return DESKTOP_ACTIONS[action_key]
    return GESTURE_ACTIONS.get(action_key, action_key)


_load_user_config()

MIRROR_PREVIEW = True
PYAUTOGUI_FAILSAFE = False

STATIC_GESTURE_CLASSES = frozenset({"thumbs_up", "thumbs_down", "open_palm", "fist_close"})
STATIC_ONLY_CLASSES = frozenset({"thumbs_up", "thumbs_down"})
MOUSE_SEQUENCE_TRIGGER_CLASSES = frozenset({"fist_close", "open_palm"})

GESTURE_ACTIONS: dict[str, str] = {
    "thumbs_up": "Next Page (Page Down)",
    "thumbs_down": "Previous Page (Page Up)",
    "open_palm": "Toggle Fullscreen",
    "fist_close": "Play / Pause",
    "mouse_move": "Mouse Move (Two Fingers)",
    "click_left": "Left Click (Two Fingers -> Fist)",
    "click_right": "Right Click (Two Fingers -> Palm)",
    "swipe_left": "Seek Back 5s",
    "swipe_right": "Seek Forward 5s",
    "swipe_up": "Volume Up",
    "swipe_down": "Volume Down",
    "play_pause": "Play / Pause",
}

LEGEND_STATIC: list[tuple[str | None, str, str, str]] = [
    (None, "STATIC GESTURES", "", ""),
    ("thumbs_up", "Thumbs Up", "Next slide", "PgDn"),
    ("thumbs_down", "Thumbs Down", "Previous slide", "PgUp"),
    ("open_palm", "Open Palm", "Fullscreen toggle", "F"),
    ("fist_close", "Closed Fist", "Play / pause media", ">/||"),
]

LEGEND_DESKTOP: list[tuple[str | None, str, str, str]] = [
    (None, "DESKTOP / OS", "", ""),
    ("thumbs_up", "Thumbs Up", "Switch application", "Alt+Tab"),
    ("thumbs_down", "Thumbs Down", "Open terminal", "Term"),
    ("open_palm", "Open Palm", "App launcher / menu", "Win"),
    ("fist_close", "Closed Fist", "Close window", "Alt+F4"),
]

DESKTOP_ACTIONS: dict[str, str] = {
    "thumbs_up": "Switch App (Alt+Tab)",
    "thumbs_down": "Open Terminal",
    "open_palm": "App Launcher",
    "fist_close": "Close Window (Alt+F4)",
}


def get_legend_static() -> list[tuple[str | None, str, str, str]]:
    if get_control_profile()[0] == "desktop":
        return LEGEND_DESKTOP
    rows = [row for row in LEGEND_STATIC]
    # Patch open-palm binding for presentation fullscreen sub-mode.
    patched: list[tuple[str | None, str, str, str]] = []
    for gesture_key, title, hint, binding in rows:
        if gesture_key == "open_palm":
            binding = fullscreen_binding_display()
        patched.append((gesture_key, title, hint, binding))
    return patched


LEGEND_MOUSE: list[tuple[str | None, str, str, str]] = [
    (None, "MOUSE MODE", "", ""),
    ("two_fingers", "Two Fingers", "Move cursor", "Mouse"),
    ("click_left", "Two Fingers -> Fist", "Left click", "LMB"),
    ("click_right", "Two Fingers -> Palm", "Right click", "RMB"),
]

LEGEND_DYNAMIC: list[tuple[str | None, str, str, str]] = [
    (None, "MEDIA SWIPES", "", ""),
    ("swipe_right", "Swipe Right", "Seek forward ~5s", "->"),
    ("swipe_left", "Swipe Left", "Seek back ~5s", "<-"),
    ("swipe_up", "Swipe Up", "Volume up", "Vol+"),
    ("swipe_down", "Swipe Down", "Volume down", "Vol-"),
]


def get_legend_rows() -> list[tuple[str | None, str, str, str]]:
    """Legend entries for the active control profile only."""
    if get_control_profile()[0] == "desktop":
        return list(LEGEND_DESKTOP) + list(LEGEND_MOUSE)
    return get_legend_static() + list(LEGEND_DYNAMIC)


MotionSample = tuple[int, int, float, float]


@dataclass(frozen=True)
class Detection:
    """Single detection used by the gesture pipeline."""

    class_name: str
    confidence: float
    centroid: tuple[int, int]
    bbox_width: float
    bbox_height: float


@dataclass
class FrameState:
    """Per-frame UI and pipeline state returned after processing detections."""

    active_gesture: str | None
    last_action_label: str | None
    last_action_time: float
    is_tracking: bool
    mouse_active: bool
    mouse_sequence_armed: bool
    queue_len: int
    static_vote_candidate: str | None
    static_vote_streak: int


def _cooldown_elapsed(last_action_time: float) -> bool:
    return (time.monotonic() - last_action_time) >= COOLDOWN_SECONDS


def remaining_cooldown(last_action_time: float) -> float:
    remaining = COOLDOWN_SECONDS - (time.monotonic() - last_action_time)
    return max(0.0, remaining)


def execute_static_gesture(gesture: str) -> None:
    """Dispatch a confirmed static gesture to the correct OS action for the active profile."""
    if get_control_profile()[0] == "desktop":
        from os_actions import execute_desktop_gesture

        execute_desktop_gesture(gesture)
        return

    import sys

    if sys.platform.startswith("linux"):
        # Send keys straight to the last focused (non-gesture) window — pyautogui's
        # X11 key events go wherever OUR window has focus, which is usually wrong.
        from os_actions import linux_send_key_to_focused

        if gesture == "thumbs_up":
            linux_send_key_to_focused("pagedown")
        elif gesture == "thumbs_down":
            linux_send_key_to_focused("pageup")
        elif gesture == "open_palm":
            linux_send_key_to_focused(get_fullscreen_key())
        elif gesture == "fist_close":
            execute_play_pause()
        else:
            raise ValueError(f"Unknown static gesture: {gesture}")
        return

    if gesture == "thumbs_up":
        pyautogui.press("pagedown")
    elif gesture == "thumbs_down":
        pyautogui.press("pageup")
    elif gesture == "open_palm":
        key = get_fullscreen_key()
        if "+" in key:
            mods, single = key.rsplit("+", 1)
            pyautogui.hotkey(*mods.split("+"), single)
        else:
            pyautogui.press(key)
    elif gesture == "fist_close":
        execute_play_pause()
    else:
        raise ValueError(f"Unknown static gesture: {gesture}")


def execute_mouse_click(button: str) -> None:
    if button == "left":
        pyautogui.click(button="left")
    elif button == "right":
        pyautogui.click(button="right")
    else:
        raise ValueError(f"Unknown mouse button: {button}")


def execute_swipe(swipe: str) -> None:
    """Media controls — left/right seek; up/down volume."""
    import sys

    if sys.platform.startswith("linux"):
        from os_actions import linux_volume, linux_seek
        if swipe == "swipe_left":
            linux_seek("back")
        elif swipe == "swipe_right":
            linux_seek("forward")
        elif swipe == "swipe_up":
            linux_volume("up")
        elif swipe == "swipe_down":
            linux_volume("down")
        else:
            raise ValueError(f"Unknown swipe: {swipe}")
        return

    # Windows
    if swipe == "swipe_left":
        pyautogui.press("left")
    elif swipe == "swipe_right":
        pyautogui.press("right")
    elif swipe == "swipe_up":
        pyautogui.press("volumeup", presses=VOLUME_STEP_COUNT, interval=0.02)
    elif swipe == "swipe_down":
        pyautogui.press("volumedown", presses=VOLUME_STEP_COUNT, interval=0.02)
    else:
        raise ValueError(f"Unknown swipe: {swipe}")


def execute_play_pause() -> None:
    """Toggle media play/pause; uses platform-native method on Linux."""
    import sys

    if sys.platform.startswith("linux"):
        from os_actions import linux_play_pause
        linux_play_pause()
        return
    pyautogui.press("playpause")


def map_centroid_to_screen(cx: int, cy: int, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Map a camera-frame centroid to absolute screen coordinates (absolute cursor mode)."""
    screen_w, screen_h = pyautogui.size()
    x = int(cx / max(1, frame_width) * screen_w)
    y = int(cy / max(1, frame_height) * screen_h)
    return max(0, min(screen_w - 1, x)), max(0, min(screen_h - 1, y))


def fire_action(action_key: str, last_action_time: float, centroid_queue: deque[MotionSample]) -> float:
    """Execute one action if cooldown has elapsed; return updated last_action_time."""
    if not _cooldown_elapsed(last_action_time):
        return last_action_time

    if action_key in STATIC_GESTURE_CLASSES:
        execute_static_gesture(action_key)
    elif action_key == "click_left":
        execute_mouse_click("left")
    elif action_key == "click_right":
        execute_mouse_click("right")
    elif action_key.startswith("swipe_"):
        execute_swipe(action_key)
    elif action_key == "play_pause":
        execute_play_pause()
    else:
        return last_action_time

    print(f"[ACTION] {action_label(action_key)}")
    centroid_queue.clear()
    return time.monotonic()


def _sequence_click_key(gesture: str) -> str | None:
    if gesture == "fist_close":
        return "click_left"
    if gesture == "open_palm":
        return "click_right"
    return None


def compute_centroid(xyxy: list[float] | tuple[float, ...]) -> tuple[int, int]:
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def pick_best_from_list(detections: list[Detection], allowed_classes: frozenset[str]) -> Detection | None:
    best: Detection | None = None
    for det in detections:
        if det.class_name not in allowed_classes:
            continue
        if det.confidence < CONFIDENCE_THRESHOLD:
            continue
        if best is None or det.confidence > best.confidence:
            best = det
    return best


def pick_best_from_yolo(result: Any, allowed_classes: frozenset[str]) -> Detection | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    names = result.names
    best: Detection | None = None

    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        class_name = names[class_id]
        if class_name not in allowed_classes:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        centroid = compute_centroid((x1, y1, x2, y2))
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        candidate = Detection(class_name, confidence, centroid, bbox_w, bbox_h)
        if best is None or confidence > best.confidence:
            best = candidate

    return best


class StaticGestureVoter:
    """Require the same gesture to win N consecutive frames before confirming it."""
    def __init__(self, required_frames: int) -> None:
        self.required_frames = required_frames
        self._candidate: str | None = None
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def candidate(self) -> str | None:
        return self._candidate

    def update(self, gesture: str | None) -> str | None:
        if gesture is None:
            self._candidate = None
            self._streak = 0
            return None

        if gesture == self._candidate:
            self._streak += 1
        else:
            self._candidate = gesture
            self._streak = 1

        if self._streak >= self.required_frames:
            self._streak = 0
            self._candidate = None
            return gesture
        return None

    def reset(self) -> None:
        self._candidate = None
        self._streak = 0


class CentroidSmoother:
    """Exponential moving average filter for noisy bounding-box centroids."""
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None
        self._bbox_w: float | None = None
        self._bbox_h: float | None = None

    @property
    def active(self) -> bool:
        return self._x is not None

    def update(self, x: int, y: int, bbox_w: float, bbox_h: float) -> tuple[int, int, float, float]:
        if self._x is None:
            self._x, self._y = float(x), float(y)
            self._bbox_w, self._bbox_h = bbox_w, bbox_h
        else:
            a = self.alpha
            self._x = (1.0 - a) * self._x + a * x
            self._y = (1.0 - a) * self._y + a * y
            self._bbox_w = (1.0 - a) * self._bbox_w + a * bbox_w
            self._bbox_h = (1.0 - a) * self._bbox_h + a * bbox_h
        assert self._bbox_w is not None and self._bbox_h is not None
        return int(self._x), int(self._y), self._bbox_w, self._bbox_h

    def reset(self) -> None:
        self._x = self._y = self._bbox_w = self._bbox_h = None


class ScreenCursorController:
    """Controls the mouse cursor from hand centroid detections.

    Two modes (selected by the module-level CURSOR_RELATIVE_MODE flag):

    Absolute mode (CURSOR_RELATIVE_MODE = False):
        Hand position within the camera frame maps directly to a proportional
        position on the screen.  Moving your hand to the left edge of the frame
        moves the cursor to the left edge of the screen regardless of where the
        cursor was before.  Uses lerp smoothing toward the target position.

    Relative mode (CURSOR_RELATIVE_MODE = True, default):
        Acts like a trackpad.  Only the *change* in hand position between frames
        is used — the cursor starts wherever it currently is and moves by
        (delta * CURSOR_RELATIVE_SENSITIVITY) each frame.  Re-entering the
        two-finger gesture always picks up from the cursor's current location
        with no jump.
    """

    def __init__(self, ema_alpha: float, screen_lerp: float) -> None:
        self._centroid = CentroidSmoother(ema_alpha)
        self._screen_lerp = screen_lerp
        self._screen_x: float | None = None
        self._screen_y: float | None = None
        # Previous smoothed hand position — used only in relative mode
        self._prev_hand_x: float | None = None
        self._prev_hand_y: float | None = None

    @property
    def active(self) -> bool:
        return self._screen_x is not None

    def update(
        self,
        x: int,
        y: int,
        bbox_w: float,
        bbox_h: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        sx, sy, _, _ = self._centroid.update(x, y, bbox_w, bbox_h)
        screen_w, screen_h = pyautogui.size()

        if CURSOR_RELATIVE_MODE:
            # --- Relative / trackpad mode ---
            if self._screen_x is None:
                # First frame of a tracking session: anchor to real cursor pos.
                cur = pyautogui.position()
                self._screen_x, self._screen_y = float(cur.x), float(cur.y)
                self._prev_hand_x, self._prev_hand_y = float(sx), float(sy)
            else:
                assert self._prev_hand_x is not None and self._prev_hand_y is not None
                dx = (sx - self._prev_hand_x) * CURSOR_RELATIVE_SENSITIVITY
                dy = (sy - self._prev_hand_y) * CURSOR_RELATIVE_SENSITIVITY
                self._screen_x = max(0.0, min(float(screen_w - 1), self._screen_x + dx))
                self._screen_y = max(0.0, min(float(screen_h - 1), self._screen_y + dy))
                self._prev_hand_x, self._prev_hand_y = float(sx), float(sy)
        else:
            # --- Absolute mode: hand position → proportional screen position ---
            target_x, target_y = map_centroid_to_screen(sx, sy, frame_width, frame_height)
            if self._screen_x is None:
                cur = pyautogui.position()
                self._screen_x, self._screen_y = float(cur.x), float(cur.y)
            else:
                t = self._screen_lerp
                self._screen_x += (target_x - self._screen_x) * t
                self._screen_y += (target_y - self._screen_y) * t

        return int(self._screen_x), int(self._screen_y)

    def current_position(self) -> tuple[int, int] | None:
        if self._screen_x is None or self._screen_y is None:
            return None
        return int(self._screen_x), int(self._screen_y)

    def reset(self) -> None:
        self._centroid.reset()
        self._screen_x = self._screen_y = None
        self._prev_hand_x = self._prev_hand_y = None


def detect_swipe(centroid_queue: deque[MotionSample]) -> str | None:
    """Classify swipe direction from smoothed centroid motion over the queue window."""
    if len(centroid_queue) < MIN_QUEUE_SAMPLES_FOR_SWIPE:
        return None

    x_start, y_start, _, _ = centroid_queue[0]
    x_end, y_end, _, _ = centroid_queue[-1]
    avg_w = sum(s[2] for s in centroid_queue) / len(centroid_queue)
    avg_h = sum(s[3] for s in centroid_queue) / len(centroid_queue)

    thresh_x = max(SWIPE_MIN_X, int(avg_w * SWIPE_THRESHOLD_X_FRAC))
    thresh_y = max(SWIPE_MIN_Y, int(avg_h * SWIPE_THRESHOLD_Y_FRAC))

    dx = x_end - x_start
    dy = y_end - y_start
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    frames = len(centroid_queue) - 1

    if frames > 0:
        vel_x = abs_dx / frames
        vel_y = abs_dy / frames
    else:
        vel_x = vel_y = 0.0

    if abs_dx >= thresh_x and abs_dx >= abs_dy * 0.85 and vel_x >= SWIPE_MIN_VELOCITY:
        return "swipe_left" if dx < 0 else "swipe_right"

    if abs_dy >= thresh_y and abs_dy > abs_dx and vel_y >= SWIPE_MIN_VELOCITY:
        return "swipe_up" if dy < 0 else "swipe_down"

    return None


class GesturePipeline:
    """Stateful gesture processor — call process_frame() once per camera frame."""

    def __init__(self) -> None:
        self.centroid_queue: deque[MotionSample] = deque(maxlen=CENTROID_QUEUE_LENGTH)
        self.swipe_smoother = CentroidSmoother(CENTROID_EMA_ALPHA)
        self.cursor_controller = ScreenCursorController(MOUSE_EMA_ALPHA, CURSOR_SCREEN_LERP)
        self.static_voter = StaticGestureVoter(STATIC_VOTE_FRAMES)
        self.sequence_voter = StaticGestureVoter(MOUSE_SEQUENCE_VOTE_FRAMES)
        self.last_action_time = 0.0
        self.last_action_label: str | None = None
        self.active_gesture: str | None = None
        self.swipe_miss_frames = 0
        self.mouse_miss_frames = 0
        self.last_two_fingers_time: float | None = None
        self.mouse_active = False

    def process_frame(
        self,
        static_detection: Detection | None,
        index_detection: Detection | None,
        mouse_detection: Detection | None,
        frame_w: int,
        frame_h: int,
    ) -> FrameState:
        mouse_sequence_armed = (
            self.last_two_fingers_time is not None
            and (time.monotonic() - self.last_two_fingers_time) <= MOUSE_SEQUENCE_WINDOW_SECONDS
        )

        static_vote_candidate: str | None = None
        static_vote_streak = 0

        # --- Lane 1: Mouse cursor (two_fingers) ---
        if mouse_detection is not None:
            self.mouse_active = True
            self.mouse_miss_frames = 0
            self.last_two_fingers_time = time.monotonic()
            self.active_gesture = MOUSE_POINTER_CLASS
            screen_x, screen_y = self.cursor_controller.update(
                mouse_detection.centroid[0],
                mouse_detection.centroid[1],
                mouse_detection.bbox_width,
                mouse_detection.bbox_height,
                frame_w,
                frame_h,
            )
            pyautogui.moveTo(screen_x, screen_y, _pause=False)
        elif self.cursor_controller.active and self.mouse_miss_frames < MOUSE_GRACE_FRAMES:
            self.mouse_miss_frames += 1
            self.mouse_active = True
            pos = self.cursor_controller.current_position()
            if pos is not None:
                pyautogui.moveTo(pos[0], pos[1], _pause=False)
        else:
            self.mouse_active = False
            self.mouse_miss_frames = 0
            self.cursor_controller.reset()

        # --- Lane 2: Static gestures (thumbs / palm / fist) ---
        if static_detection is not None:
            gesture = static_detection.class_name
            self.active_gesture = gesture

            if gesture in STATIC_ONLY_CLASSES:
                static_vote_candidate = self.static_voter.candidate
                static_vote_streak = self.static_voter.streak
                confirmed = self.static_voter.update(gesture)
                if confirmed is not None and _cooldown_elapsed(self.last_action_time):
                    self.last_action_time = fire_action(confirmed, self.last_action_time, self.centroid_queue)
                    self.last_action_label = confirmed
                    self.swipe_smoother.reset()
                    self.swipe_miss_frames = 0

            elif gesture in MOUSE_SEQUENCE_TRIGGER_CLASSES and mouse_sequence_armed:
                self.static_voter.reset()
                confirmed = self.sequence_voter.update(gesture)
                static_vote_candidate = self.sequence_voter.candidate
                static_vote_streak = self.sequence_voter.streak
                if confirmed is not None and _cooldown_elapsed(self.last_action_time):
                    click_key = _sequence_click_key(confirmed)
                    if click_key:
                        self.last_action_time = fire_action(click_key, self.last_action_time, self.centroid_queue)
                        self.last_action_label = click_key
                        self.last_two_fingers_time = None
                        self.sequence_voter.reset()

            elif gesture in MOUSE_SEQUENCE_TRIGGER_CLASSES:
                self.sequence_voter.reset()
                static_vote_candidate = self.static_voter.candidate
                static_vote_streak = self.static_voter.streak
                confirmed = self.static_voter.update(gesture)
                if confirmed is not None and _cooldown_elapsed(self.last_action_time):
                    self.last_action_time = fire_action(confirmed, self.last_action_time, self.centroid_queue)
                    self.last_action_label = confirmed
                    self.swipe_smoother.reset()
                    self.swipe_miss_frames = 0
        else:
            self.static_voter.reset()
            self.sequence_voter.reset()

        # Static gesture held blocks swipe tracking to avoid accidental media keys
        blocks_swipe = (STATIC_BLOCKS_SWIPE and static_detection is not None) or mouse_detection is not None

        # --- Lane 3: Swipe tracking (index_finger_up) ---
        if index_detection is not None and not blocks_swipe:
            self.swipe_miss_frames = 0
            if not self.swipe_smoother.active:
                self.centroid_queue.clear()

            sx, sy, sbw, sbh = self.swipe_smoother.update(
                index_detection.centroid[0],
                index_detection.centroid[1],
                index_detection.bbox_width,
                index_detection.bbox_height,
            )
            self.centroid_queue.append((sx, sy, sbw, sbh))

            swipe = detect_swipe(self.centroid_queue)
            if swipe is not None and _cooldown_elapsed(self.last_action_time):
                self.last_action_time = fire_action(swipe, self.last_action_time, self.centroid_queue)
                self.last_action_label = swipe
                self.active_gesture = swipe
                self.swipe_smoother.reset()
        elif self.swipe_smoother.active:
            self.swipe_miss_frames += 1
            if self.swipe_miss_frames > TRACKING_GRACE_FRAMES:
                self.swipe_smoother.reset()
                self.centroid_queue.clear()
                self.swipe_miss_frames = 0

        is_tracking = index_detection is not None or (
            self.swipe_smoother.active and self.swipe_miss_frames <= TRACKING_GRACE_FRAMES
        )

        return FrameState(
            active_gesture=self.active_gesture,
            last_action_label=self.last_action_label,
            last_action_time=self.last_action_time,
            is_tracking=is_tracking,
            mouse_active=self.mouse_active,
            mouse_sequence_armed=mouse_sequence_armed,
            queue_len=len(self.centroid_queue),
            static_vote_candidate=static_vote_candidate,
            static_vote_streak=static_vote_streak,
        )
