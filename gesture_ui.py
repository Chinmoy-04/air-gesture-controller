"""
gesture_ui.py — OpenCV UI for the air-gesture controller (shared Windows + Pi).

Layout: video feed (left) + sidebar legend/status (right).
UI_LITE mode (default on Pi) uses 1280x720, simpler fonts, and no alpha compositing.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from gesture_engine import (
    CENTROID_QUEUE_LENGTH,
    COOLDOWN_SECONDS,
    CONFIDENCE_THRESHOLD,
    DRAW_CENTROID_TRAIL,
    MotionSample,
    STATIC_VOTE_FRAMES,
    action_label,
    control_profile_summary,
    fullscreen_mode_summary,
    get_control_profile,
    get_legend_rows,
    remaining_cooldown,
)

WINDOW_TITLE = "Air-Gesture Controller"
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
SIDEBAR_WIDTH = 480
SIDEBAR_SUPERSAMPLE = 2
SIDEBAR_STATUS_HEIGHT = 208
HEADER_HEIGHT = 64
FOOTER_HEIGHT = 36
UI_FONT = cv2.FONT_HERSHEY_DUPLEX
WINDOW_ALWAYS_ON_TOP = True
UI_LITE = False
_display_canvas: np.ndarray | None = None

# Full-quality defaults (restored by configure_ui_lite(False)).
_UI_FULL = {
    "WINDOW_WIDTH": 1440,
    "WINDOW_HEIGHT": 900,
    "SIDEBAR_WIDTH": 480,
    "SIDEBAR_SUPERSAMPLE": 2,
    "SIDEBAR_STATUS_HEIGHT": 208,
    "HEADER_HEIGHT": 64,
    "FOOTER_HEIGHT": 36,
    "UI_FONT": cv2.FONT_HERSHEY_DUPLEX,
}

_UI_LITE = {
    "WINDOW_WIDTH": 1280,
    "WINDOW_HEIGHT": 720,
    "SIDEBAR_WIDTH": 320,
    "SIDEBAR_SUPERSAMPLE": 1,
    "SIDEBAR_STATUS_HEIGHT": 156,
    "HEADER_HEIGHT": 44,
    "FOOTER_HEIGHT": 26,
    "UI_FONT": cv2.FONT_HERSHEY_SIMPLEX,
}


def configure_ui_lite(enabled: bool = True) -> None:
    """Use a smaller, cheaper UI layout (intended for Raspberry Pi)."""
    global UI_LITE, WINDOW_WIDTH, WINDOW_HEIGHT, SIDEBAR_WIDTH, SIDEBAR_SUPERSAMPLE
    global SIDEBAR_STATUS_HEIGHT, HEADER_HEIGHT, FOOTER_HEIGHT, UI_FONT, _display_canvas

    UI_LITE = enabled
    _display_canvas = None
    settings = _UI_LITE if enabled else _UI_FULL
    WINDOW_WIDTH = settings["WINDOW_WIDTH"]
    WINDOW_HEIGHT = settings["WINDOW_HEIGHT"]
    SIDEBAR_WIDTH = settings["SIDEBAR_WIDTH"]
    SIDEBAR_SUPERSAMPLE = settings["SIDEBAR_SUPERSAMPLE"]
    SIDEBAR_STATUS_HEIGHT = settings["SIDEBAR_STATUS_HEIGHT"]
    HEADER_HEIGHT = settings["HEADER_HEIGHT"]
    FOOTER_HEIGHT = settings["FOOTER_HEIGHT"]
    UI_FONT = settings["UI_FONT"]

COLOR_BG = (28, 26, 24)
COLOR_PANEL = (42, 38, 36)
COLOR_PANEL_BORDER = (72, 68, 64)
COLOR_HEADER = (32, 52, 88)
COLOR_ACCENT = (255, 180, 60)
COLOR_TEXT = (248, 248, 248)
COLOR_TEXT_MUTED = (168, 164, 158)
COLOR_SUCCESS = (100, 220, 120)
COLOR_WARNING = (80, 180, 255)
COLOR_TRACKING = (255, 200, 0)
COLOR_HIGHLIGHT = (60, 140, 255)


def setup_display_window(title: str = WINDOW_TITLE) -> None:
    # WINDOW_GUI_NORMAL disables OpenCV's Qt-backend toolbar/statusbar (the pan,
    # zoom, save, and print icons that otherwise clutter the title bar area).
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(title, flags)
    cv2.resizeWindow(title, WINDOW_WIDTH, WINDOW_HEIGHT)
    apply_profile_window_mode(title)


def allow_system_hotkeys(title: str = WINDOW_TITLE) -> None:
    """Drop always-on-top so Alt+Tab and the Start menu can appear above us."""
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 0)


def apply_profile_window_mode(title: str = WINDOW_TITLE) -> None:
    """Presentation keeps the window on top; desktop does not (OS shortcuts need focus).
    On Linux, WND_PROP_TOPMOST steals focus — never set it in desktop mode, and
    only set it on non-Linux platforms in presentation mode.
    """
    import sys
    from gesture_engine import get_control_profile

    is_linux = sys.platform.startswith("linux")
    if get_control_profile()[0] == "desktop":
        allow_system_hotkeys(title)
    elif WINDOW_ALWAYS_ON_TOP and not is_linux:
        # Only raise window on non-Linux; on Linux this steals keyboard focus
        cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)


def wire_os_action_hooks(title: str = WINDOW_TITLE) -> None:
    from os_actions import register_before_action

    register_before_action(lambda: allow_system_hotkeys(title))


def _s(value: float | int, ui_scale: int) -> int:
    return int(value * ui_scale)


def _draw_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = COLOR_TEXT,
    thickness: int = 1,
    font: int = UI_FONT,
    shadow: bool = True,
    ui_scale: int = 1,
) -> None:
    x, y = xy
    scale = max(0.1, scale * max(1, ui_scale))  # Qt backend rejects scale <= 0 on Pi
    thickness = max(1, thickness * max(1, ui_scale))
    line_type = cv2.LINE_8 if UI_LITE else cv2.LINE_AA
    if shadow and not UI_LITE:
        offset = ui_scale
        cv2.putText(
            img, text, (x + offset, y + offset), font, scale, (0, 0, 0),
            thickness + ui_scale, line_type,
        )
    cv2.putText(img, text, (x, y), font, scale, color, thickness, line_type)


def _fill_rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    if alpha >= 1.0 or UI_LITE:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_panel(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill: tuple[int, int, int] = COLOR_PANEL,
    border: tuple[int, int, int] = COLOR_PANEL_BORDER,
    ui_scale: int = 1,
) -> None:
    _fill_rect(img, x1, y1, x2, y2, fill)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, ui_scale, cv2.LINE_AA)


def _fit_frame_to_area(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_NEAREST if UI_LITE else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def _draw_legend_row(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    row_height: int,
    gesture_key: str | None,
    title: str,
    hint: str,
    binding: str,
    active_key: str | None,
    tracking: bool,
    mouse_active: bool,
    ui_scale: int = 1,
) -> int:
    if gesture_key is None:
        _draw_text(
            canvas, title, (x + _s(14, ui_scale), y + row_height // 3),
            scale=0.58, color=COLOR_ACCENT, thickness=2, ui_scale=ui_scale,
        )
        cv2.line(
            canvas,
            (x + _s(10, ui_scale), y + row_height - _s(4, ui_scale)),
            (x + width - _s(10, ui_scale), y + row_height - _s(4, ui_scale)),
            COLOR_PANEL_BORDER, ui_scale, cv2.LINE_AA,
        )
        return y + row_height

    is_active = (
        gesture_key == active_key
        or (gesture_key == "two_fingers" and mouse_active)
        or (gesture_key == "click_left" and active_key == "click_left")
        or (gesture_key == "click_right" and active_key == "click_right")
    )

    row_y1 = y + _s(2, ui_scale)
    row_y2 = y + row_height - _s(2, ui_scale)
    if is_active and not UI_LITE:
        _fill_rect(canvas, x + _s(8, ui_scale), row_y1, x + width - _s(8, ui_scale), row_y2, COLOR_HIGHLIGHT, alpha=0.35)
        cv2.rectangle(
            canvas,
            (x + _s(8, ui_scale), row_y1),
            (x + width - _s(8, ui_scale), row_y2),
            COLOR_ACCENT, ui_scale, cv2.LINE_AA,
        )
    elif is_active:
        cv2.rectangle(
            canvas,
            (x + _s(8, ui_scale), row_y1),
            (x + width - _s(8, ui_scale), row_y2),
            COLOR_ACCENT, max(1, ui_scale), cv2.LINE_8,
        )

    dot_color = COLOR_SUCCESS if is_active else COLOR_TEXT_MUTED
    cv2.circle(
        canvas,
        (x + _s(22, ui_scale), y + row_height // 2),
        _s(5, ui_scale), dot_color, -1, cv2.LINE_AA,
    )

    title_y = y + max(_s(16, ui_scale), row_height // 4)
    hint_y = y + max(_s(30, ui_scale), (row_height * 3) // 5)
    bind_y1 = y + max(_s(10, ui_scale), row_height // 6)
    bind_y2 = y + max(_s(28, ui_scale), row_height // 2 + _s(4, ui_scale))
    bind_text_y = y + max(_s(24, ui_scale), row_height // 2 + _s(2, ui_scale))

    _draw_text(
        canvas, title, (x + _s(38, ui_scale), title_y),
        scale=0.58, color=COLOR_TEXT if is_active else (220, 218, 214),
        thickness=2, ui_scale=ui_scale,
    )
    if row_height >= _s(44, ui_scale):
        _draw_text(
            canvas, hint, (x + _s(38, ui_scale), hint_y),
            scale=0.44, color=COLOR_TEXT_MUTED, shadow=False, ui_scale=ui_scale,
        )

    binding_scale = 0.48 * ui_scale
    binding_size = cv2.getTextSize(binding, UI_FONT, binding_scale, ui_scale)[0]
    bind_x = x + width - binding_size[0] - _s(18, ui_scale)
    _fill_rect(
        canvas,
        bind_x - _s(6, ui_scale), bind_y1,
        x + width - _s(12, ui_scale), bind_y2,
        (55, 50, 48),
    )
    _draw_text(
        canvas, binding, (bind_x, bind_text_y),
        scale=0.48, color=COLOR_ACCENT if is_active else COLOR_TEXT_MUTED,
        shadow=False, thickness=2, ui_scale=ui_scale,
    )
    return y + row_height


def _render_sidebar_surface(
    active_key: str | None,
    tracking: bool,
    mouse_active: bool,
    mouse_sequence_armed: bool,
    last_action: str | None,
    last_action_time: float,
    queue_len: int,
    static_vote_candidate: str | None,
    static_vote_streak: int,
) -> np.ndarray:
    ss = SIDEBAR_SUPERSAMPLE
    sw = SIDEBAR_WIDTH * ss
    sh = (WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT) * ss
    surf = np.full((sh, sw, 3), COLOR_PANEL, dtype=np.uint8)

    status_h = _s(SIDEBAR_STATUS_HEIGHT, ss)
    legend_top = _s(40, ss) if UI_LITE else _s(68, ss)
    legend_bottom = sh - status_h - _s(10, ss)
    legend_rows = get_legend_rows()
    available_h = legend_bottom - legend_top
    row_h = available_h // max(1, len(legend_rows))
    row_h = max(_s(34, ss), min(row_h, _s(56, ss)))

    _draw_panel(surf, 0, 0, sw, legend_bottom + _s(4, ss), fill=COLOR_PANEL, ui_scale=ss)

    _draw_text(
        surf, "GESTURE LEGEND", (_s(18, ss), _s(30, ss)),
        scale=0.72, color=COLOR_ACCENT, thickness=2, ui_scale=ss,
    )
    if not UI_LITE:
        _draw_text(
            surf,
            "Perform a gesture in front of the camera",
            (_s(18, ss), _s(52, ss)),
            scale=0.42,
            color=COLOR_TEXT_MUTED,
            shadow=False,
            ui_scale=ss,
        )

    y = legend_top
    for row in legend_rows:
        gesture_key, title, hint, binding = row
        y = _draw_legend_row(
            surf, 0, y, sw, row_h,
            gesture_key, title, hint, binding,
            active_key, tracking, mouse_active, ui_scale=ss,
        )

    div_y = legend_bottom + _s(6, ss)
    cv2.line(surf, (_s(12, ss), div_y), (sw - _s(12, ss), div_y), COLOR_ACCENT, _s(2, ss), cv2.LINE_AA)

    card_y1 = sh - status_h
    _draw_panel(surf, _s(10, ss), card_y1, sw - _s(10, ss), sh - _s(8, ss), fill=(32, 30, 28), ui_scale=ss)

    _draw_text(surf, "LIVE STATUS", (_s(22, ss), card_y1 + _s(26, ss)), scale=0.56, color=COLOR_ACCENT, thickness=2, ui_scale=ss)

    cooldown_left = remaining_cooldown(last_action_time)
    ready = cooldown_left <= 0
    status_label = "READY" if ready else f"COOLDOWN {cooldown_left:.1f}s"
    status_color = COLOR_SUCCESS if ready else COLOR_WARNING
    _draw_text(
        surf, status_label, (_s(22, ss), card_y1 + _s(50, ss)),
        scale=0.58, color=status_color, thickness=2, ui_scale=ss,
    )

    bar_x1, bar_x2 = _s(22, ss), sw - _s(22, ss)
    bar_y = card_y1 + _s(64, ss)
    bar_h = _s(10, ss)
    cv2.rectangle(surf, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (50, 46, 44), -1)
    if not ready:
        progress = 1.0 - (cooldown_left / COOLDOWN_SECONDS)
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * progress)
        cv2.rectangle(surf, (bar_x1, bar_y), (fill_x, bar_y + bar_h), COLOR_WARNING, -1)

    line_y = card_y1 + _s(84, ss)
    line_gap = _s(18, ss)

    if get_control_profile()[0] == "presentation":
        track_text = "Swipe tracking: ON" if tracking else "Swipe tracking: off"
        track_color = COLOR_TRACKING if tracking else COLOR_TEXT_MUTED
        _draw_text(surf, track_text, (_s(22, ss), line_y), scale=0.42, color=track_color, shadow=False, ui_scale=ss)
        line_y += line_gap
    else:
        mouse_text = "Mouse mode: ON" if mouse_active else "Mouse mode: off"
        mouse_color = COLOR_ACCENT if mouse_active else COLOR_TEXT_MUTED
        _draw_text(surf, mouse_text, (_s(22, ss), line_y), scale=0.42, color=mouse_color, shadow=False, ui_scale=ss)
        line_y += line_gap

    profile_text = f"Profile: {control_profile_summary()}"
    _draw_text(
        surf, profile_text, (_s(22, ss), line_y),
        scale=0.38, color=COLOR_ACCENT, shadow=False, ui_scale=ss,
    )
    line_y += line_gap

    if get_control_profile()[0] == "presentation":
        _draw_text(
            surf, f"Fullscreen: {fullscreen_mode_summary()}", (_s(22, ss), line_y),
            scale=0.38, color=COLOR_TEXT_MUTED, shadow=False, ui_scale=ss,
        )
        line_y += line_gap

    if mouse_sequence_armed:
        _draw_text(
            surf, "Click armed (show fist / palm)", (_s(22, ss), line_y),
            scale=0.38, color=COLOR_WARNING, shadow=False, ui_scale=ss,
        )
        line_y += line_gap
    elif static_vote_candidate and static_vote_streak > 0:
        vote_text = f"Confirming: {static_vote_candidate} ({static_vote_streak}/{STATIC_VOTE_FRAMES})"
        _draw_text(
            surf, vote_text, (_s(22, ss), line_y),
            scale=0.38, color=COLOR_ACCENT, shadow=False, ui_scale=ss,
        )
        line_y += line_gap
    else:
        _draw_text(
            surf,
            f"Motion buffer: {queue_len}/{CENTROID_QUEUE_LENGTH}",
            (_s(22, ss), line_y),
            scale=0.42,
            color=COLOR_TEXT_MUTED,
            shadow=False,
            ui_scale=ss,
        )
        line_y += line_gap

    if last_action:
        action_text = action_label(last_action)
        if len(action_text) > 38:
            action_text = action_text[:35] + "..."
        _draw_text(
            surf, f"Last: {action_text}", (_s(22, ss), line_y),
            scale=0.40, color=COLOR_TEXT, shadow=False, ui_scale=ss,
        )

    return surf if ss == 1 else cv2.resize(surf, (SIDEBAR_WIDTH, sh // ss), interpolation=cv2.INTER_AREA)


def _draw_sidebar(
    canvas: np.ndarray,
    active_key: str | None,
    tracking: bool,
    mouse_active: bool,
    mouse_sequence_armed: bool,
    last_action: str | None,
    last_action_time: float,
    queue_len: int,
    static_vote_candidate: str | None,
    static_vote_streak: int,
) -> None:
    x0 = WINDOW_WIDTH - SIDEBAR_WIDTH
    sidebar = _render_sidebar_surface(
        active_key, tracking, mouse_active, mouse_sequence_armed,
        last_action, last_action_time, queue_len,
        static_vote_candidate, static_vote_streak,
    )
    y0 = HEADER_HEIGHT
    canvas[y0 : y0 + sidebar.shape[0], x0 : x0 + sidebar.shape[1]] = sidebar


def _draw_header(canvas: np.ndarray) -> None:
    _fill_rect(canvas, 0, 0, WINDOW_WIDTH, HEADER_HEIGHT, COLOR_HEADER)
    cv2.line(canvas, (0, HEADER_HEIGHT - 1), (WINDOW_WIDTH, HEADER_HEIGHT - 1), COLOR_ACCENT, 2)
    title = "Air-Gesture Controller" if UI_LITE else "Interactive Air-Gesture Touchless Controller"
    _draw_text(canvas, title, (20, HEADER_HEIGHT - 12), scale=0.62, color=COLOR_TEXT, thickness=1)

    if not UI_LITE:
        badge = f"conf >= {CONFIDENCE_THRESHOLD:.0%}   cooldown {COOLDOWN_SECONDS:.1f}s"
        badge_size = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        bx = WINDOW_WIDTH - SIDEBAR_WIDTH - badge_size[0] - 28
        _fill_rect(canvas, bx - 8, 16, bx + badge_size[0] + 8, 48, (24, 40, 68), alpha=0.9)
        _draw_text(canvas, badge, (bx, 40), scale=0.42, color=(200, 220, 255), shadow=False)


def _draw_footer(canvas: np.ndarray) -> None:
    y0 = WINDOW_HEIGHT - FOOTER_HEIGHT
    _fill_rect(canvas, 0, y0, WINDOW_WIDTH, WINDOW_HEIGHT, (22, 20, 18))
    _draw_text(
        canvas,
        "P = profile  |  M = fullscreen  |  Q = quit" if UI_LITE
        else "P = profile (slides / desktop)  |  M = fullscreen (presentation)  |  Q = quit",
        (20, y0 + FOOTER_HEIGHT - 10),
        scale=0.44,
        color=COLOR_TEXT_MUTED,
        shadow=False,
    )


def compose_display(
    video_frame: np.ndarray,
    active_key: str | None,
    last_action: str | None,
    last_action_time: float,
    tracking: bool,
    mouse_active: bool,
    mouse_sequence_armed: bool,
    queue_len: int,
    static_vote_candidate: str | None,
    static_vote_streak: int,
) -> np.ndarray:
    """Compose the full window: header + scaled video + profile-aware sidebar."""
    global _display_canvas
    if UI_LITE:
        if _display_canvas is None or _display_canvas.shape[:2] != (WINDOW_HEIGHT, WINDOW_WIDTH):
            _display_canvas = np.empty((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas = _display_canvas
        canvas[:] = COLOR_BG
    else:
        canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), COLOR_BG, dtype=np.uint8)
    _draw_header(canvas)
    _draw_footer(canvas)

    video_area_w = WINDOW_WIDTH - SIDEBAR_WIDTH - 32
    video_area_h = WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT - 32
    scaled = _fit_frame_to_area(video_frame, video_area_w, video_area_h)

    vx = 16 + (video_area_w - scaled.shape[1]) // 2
    vy = HEADER_HEIGHT + 16 + (video_area_h - scaled.shape[0]) // 2

    _draw_panel(canvas, vx - 8, vy - 8, vx + scaled.shape[1] + 8, vy + scaled.shape[0] + 8, fill=(18, 16, 14))
    canvas[vy : vy + scaled.shape[0], vx : vx + scaled.shape[1]] = scaled

    _draw_sidebar(
        canvas, active_key, tracking, mouse_active, mouse_sequence_armed,
        last_action, last_action_time, queue_len,
        static_vote_candidate, static_vote_streak,
    )
    return canvas


def draw_centroid_trail(frame: np.ndarray, centroid_queue: deque[MotionSample]) -> None:
    """Overlay the swipe centroid trail on the video frame (presentation mode)."""
    if len(centroid_queue) < 2:
        return

    points = [(s[0], s[1]) for s in centroid_queue]
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (255, 200, 0), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 165, 255), -1)