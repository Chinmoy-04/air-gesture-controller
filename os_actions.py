"""
os_actions.py — Desktop / OS control actions (Linux Pi and Windows).

Used when the gesture controller is in "desktop" profile.

Display server detection
------------------------
Pi OS Bookworm uses Wayland (labwc compositor) by default.
  - xdotool  : works only for injecting keys into X11 / XWayland windows.
               Cannot send global WM shortcuts (Alt+Tab, Super) to Wayland.
  - ydotool  : Wayland-native equivalent; requires uinput kernel module.
               Install:  sudo apt install ydotool
                         sudo modprobe uinput   (or add 'uinput' to /etc/modules)
                         sudo usermod -a -G input $USER  (then re-login)
  - wmctrl   : talks X11 window list; works on both X11 and XWayland for
               window focus / close, even under Wayland.
               Install:  sudo apt install wmctrl
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable

_before_action_callbacks: list[Callable[[], None]] = []
_last_other_window: str | None = None
_gesture_window_title: str = "Gesture Controller"
_tracker_active: bool = False

# X11 DISPLAY (used by xdotool / wmctrl / cv2)
_DISPLAY: str = os.environ.get("DISPLAY", ":0")
# Wayland display socket (non-empty when running under Wayland)
_WAYLAND_DISPLAY: str = os.environ.get("WAYLAND_DISPLAY", "")


def register_before_action(callback: Callable[[], None]) -> None:
    """Register a hook run before OS hotkeys (e.g. drop always-on-top on our window)."""
    _before_action_callbacks.append(callback)


def _prepare_for_hotkey() -> None:
    for cb in _before_action_callbacks:
        cb()


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_wayland() -> bool:
    return bool(_WAYLAND_DISPLAY)


# ---------------------------------------------------------------------------
# Subprocess environment helpers
# ---------------------------------------------------------------------------

def _xenv() -> dict[str, str]:
    """Env dict with DISPLAY and WAYLAND_DISPLAY guaranteed set."""
    env = dict(os.environ)
    env.setdefault("DISPLAY", _DISPLAY)
    if _WAYLAND_DISPLAY:
        env.setdefault("WAYLAND_DISPLAY", _WAYLAND_DISPLAY)
    return env


def _run(*cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=_xenv(),
    )


# ---------------------------------------------------------------------------
# xdotool  (X11 key injection — works for XWayland apps, NOT global WM keys on Wayland)
# ---------------------------------------------------------------------------

def _xdotool(*args: str) -> None:
    if shutil.which("xdotool"):
        _run("xdotool", *args)


def _xdotool_key(key: str) -> None:
    _xdotool("key", "--clearmodifiers", key)


def _get_active_wid() -> str | None:
    if not shutil.which("xdotool"):
        return None
    r = subprocess.run(
        ["xdotool", "getactivewindow"],
        capture_output=True, text=True, env=_xenv(),
    )
    return r.stdout.strip() if r.returncode == 0 else None


def _get_window_name(wid: str) -> str:
    r = subprocess.run(
        ["xdotool", "getwindowname", wid],
        capture_output=True, text=True, env=_xenv(),
    )
    return r.stdout.strip() if r.returncode == 0 else ""


# ---------------------------------------------------------------------------
# ydotool  (Wayland key injection — needs uinput kernel module + group access)
# ---------------------------------------------------------------------------

def _ydotool_key(*keys: str) -> None:
    """Send a key combo via ydotool (Wayland).  keys = ["alt", "Tab"] etc."""
    if not shutil.which("ydotool"):
        return
    # ydotool expects key names joined with '+' for combos
    _run("ydotool", "key", "+".join(keys))


# ---------------------------------------------------------------------------
# wmctrl  (X11 protocol window list — works even under XWayland on Wayland)
# ---------------------------------------------------------------------------

def _wmctrl_list_other_windows() -> list[str]:
    """Return wmctrl window IDs for every window that is NOT ours."""
    if not shutil.which("wmctrl"):
        return []
    r = subprocess.run(
        ["wmctrl", "-l"], capture_output=True, text=True, env=_xenv(),
    )
    if r.returncode != 0:
        return []
    wids = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        title = parts[3]
        if _gesture_window_title not in title:
            wids.append(parts[0])
    return wids


def _wmctrl_focus(wid: str) -> None:
    if shutil.which("wmctrl"):
        _run("wmctrl", "-i", "-a", wid)


def _wmctrl_close(wid: str) -> None:
    if shutil.which("wmctrl"):
        _run("wmctrl", "-i", "-c", wid)


# ---------------------------------------------------------------------------
# Background window tracker (polls every 500 ms — never blocks main loop)
# ---------------------------------------------------------------------------

def start_window_tracker(gesture_window_title: str = "Gesture Controller") -> None:
    global _tracker_active, _gesture_window_title
    _gesture_window_title = gesture_window_title
    if not is_linux():
        return
    if not shutil.which("xdotool") and not shutil.which("wmctrl"):
        print("[os_actions] Neither xdotool nor wmctrl found — window tracking disabled.")
        return
    _tracker_active = True
    t = threading.Thread(target=_tracker_loop, daemon=True, name="window-tracker")
    t.start()
    print("[os_actions] Window tracker started (polls every 500 ms)")


def stop_window_tracker() -> None:
    global _tracker_active
    _tracker_active = False


def _tracker_loop() -> None:
    global _last_other_window
    while _tracker_active:
        time.sleep(0.5)
        prev = _last_other_window
        # Prefer wmctrl (works on Wayland too); fall back to xdotool
        if shutil.which("wmctrl"):
            wids = _wmctrl_list_other_windows()
            if wids:
                _last_other_window = wids[-1]  # most-recently listed = most recent
        elif shutil.which("xdotool"):
            wid = _get_active_wid()
            if wid and _gesture_window_title not in _get_window_name(wid):
                _last_other_window = wid
        if _last_other_window != prev:
            name = _get_window_name(_last_other_window) if _last_other_window else "(none)"
            print(f"[os_actions] tracked window changed -> {_last_other_window} \"{name}\"")


def _focus_last_window() -> None:
    if not _last_other_window or not is_linux():
        return
    if shutil.which("wmctrl"):
        _wmctrl_focus(_last_other_window)
    else:
        _xdotool("windowactivate", "--sync", _last_other_window)
    time.sleep(0.12)


# ---------------------------------------------------------------------------
# Volume  (wpctl → pactl → amixer → xdotool key)
# ---------------------------------------------------------------------------

def linux_volume(direction: str) -> None:
    up = direction == "up"
    if shutil.which("wpctl"):
        _run("wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+" if up else "5%-")
    elif shutil.which("pactl"):
        _run("pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%" if up else "-5%")
    elif shutil.which("amixer"):
        _run("amixer", "sset", "Master", "5%+" if up else "5%-")
    else:
        _xdotool_key("XF86AudioRaiseVolume" if up else "XF86AudioLowerVolume")


# ---------------------------------------------------------------------------
# Media play/pause
# ---------------------------------------------------------------------------

def linux_play_pause() -> None:
    if shutil.which("playerctl"):
        _run("playerctl", "play-pause")
    else:
        _xdotool_key("XF86AudioPlay")


# ---------------------------------------------------------------------------
# Seek / page navigation (Left/Right arrow to last focused window)
# ---------------------------------------------------------------------------

def linux_seek(direction: str) -> None:
    key = "Right" if direction == "forward" else "Left"
    linux_send_key_to_focused(key)


# ---------------------------------------------------------------------------
# Generic "send key to the last focused (non-gesture) window"
# Used for presentation-mode page navigation and fullscreen toggles.
# ---------------------------------------------------------------------------

_KEY_NAME_MAP = {
    "pagedown": "Page_Down",
    "pageup": "Page_Up",
    "f11": "F11",
    "f": "f",
    "esc": "Escape",
}


def _to_xdotool_key(pyautogui_style_key: str) -> str:
    """Convert a pyautogui-style key/hotkey string (e.g. 'ctrl+l', 'pagedown')
    into an xdotool key spec."""
    if "+" in pyautogui_style_key:
        *mods, last = pyautogui_style_key.split("+")
        last = _KEY_NAME_MAP.get(last, last)
        return "+".join([*mods, last])
    return _KEY_NAME_MAP.get(pyautogui_style_key, pyautogui_style_key)


def linux_send_key_to_focused(pyautogui_style_key: str) -> None:
    """Focus the last non-gesture window, then send a key to it directly.

    NOTE: xdotool/wmctrl can only see X11 / XWayland windows. A window
    belonging to a native-Wayland client (common on Pi OS Bookworm/labwc)
    is invisible to these tools no matter what — if this silently does
    nothing, switch the Pi session to X11 via:
        sudo raspi-config -> Advanced Options -> Wayland -> X11
    """
    key = _to_xdotool_key(pyautogui_style_key)
    _focus_last_window()
    if shutil.which("xdotool") and _last_other_window:
        print(f"[os_actions] sending key '{key}' to window {_last_other_window}")
        _xdotool("key", "--window", _last_other_window, "--clearmodifiers", key)
    else:
        print(f"[os_actions] no tracked window yet — sending '{key}' to whatever has focus "
              f"(likely our own window if you haven't clicked/alt-tabbed to the target app)")
        _xdotool_key(key)


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------

def _hotkey(*keys: str) -> None:
    import pyautogui
    pyautogui.hotkey(*keys)


def _press(key: str) -> None:
    import pyautogui
    pyautogui.press(key)


def _win_key_combo(vk_modifier: int, vk_key: int) -> None:
    import ctypes
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_modifier, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(vk_key, 0, 0, 0)
    user32.keybd_event(vk_key, 0, KEYEVENTF_KEYUP, 0)
    time.sleep(0.05)
    user32.keybd_event(vk_modifier, 0, KEYEVENTF_KEYUP, 0)


# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------

def print_linux_capabilities() -> None:
    session = os.environ.get("XDG_SESSION_TYPE", "unknown")
    print(f"─── Linux system-control capabilities  [session: {session}] ───")
    tools = {
        "xdotool":    "X11 key injection (arrow keys, seek, XWayland windows)",
        "wmctrl":     "window list / focus / close  (works on X11 AND Wayland via XWayland)",
        "lxpanelctl": "LXDE run dialog — used for app launcher gesture on Pi",
        "wpctl":      "volume — PipeWire/WirePlumber",
        "pactl":      "volume — PulseAudio",
        "amixer":     "volume — ALSA fallback",
        "playerctl":  "media play-pause (MPRIS, D-Bus)",
    }
    for tool, desc in tools.items():
        status = "OK     " if shutil.which(tool) else "MISSING"
        print(f"  {tool:12s} [{status}]  {desc}")
    if session == "wayland" and not shutil.which("wmctrl"):
        print("  Wayland detected — install wmctrl for window switching:")
        print("    sudo apt install wmctrl")
    elif not shutil.which("wmctrl"):
        print("  Tip: install wmctrl for reliable window switching:  sudo apt install wmctrl")
    print("─" * 60)


# ---------------------------------------------------------------------------
# Desktop gesture actions
# ---------------------------------------------------------------------------

def open_terminal() -> None:
    if is_linux():
        for cmd in (
            ["lxterminal"], ["gnome-terminal"], ["xfce4-terminal"],
            ["x-terminal-emulator"], ["xterm"],
        ):
            if shutil.which(cmd[0]):
                subprocess.Popen(cmd, start_new_session=True)
                return
        _xdotool_key("ctrl+alt+t")
        return
    if shutil.which("wt"):
        subprocess.Popen(["wt"], start_new_session=True)
        return
    subprocess.Popen(["cmd"], start_new_session=True,
                     creationflags=subprocess.CREATE_NEW_CONSOLE)


def switch_application() -> None:
    """Cycle to the next window; uses wmctrl (works on Wayland) or ydotool/xdotool."""
    _prepare_for_hotkey()
    if is_linux():
        # wmctrl approach: enumerate other windows and focus the next one
        if shutil.which("wmctrl"):
            wids = _wmctrl_list_other_windows()
            print(f"[os_actions] switch_application: wmctrl sees {len(wids)} other window(s): {wids}")
            if not wids:
                print("[os_actions]   No other windows visible to wmctrl. If the target app is")
                print("[os_actions]   running natively on Wayland, it is invisible to X11 tools —")
                print("[os_actions]   switch the Pi session to X11 (raspi-config > Advanced Options > Wayland).")
                return
            if _last_other_window and _last_other_window in wids:
                idx = (wids.index(_last_other_window) + 1) % len(wids)
            else:
                idx = 0
            _wmctrl_focus(wids[idx])
            return
        # Fallback: ydotool (Wayland) or xdotool (X11)
        if is_wayland() and shutil.which("ydotool"):
            _ydotool_key("alt", "Tab")
        else:
            _xdotool_key("alt+Tab")
        return
    _win_key_combo(0x12, 0x09)


def show_desktop() -> None:
    if is_linux():
        if is_wayland() and shutil.which("ydotool"):
            _ydotool_key("super", "d")
        else:
            _xdotool_key("super+d")
        return
    _hotkey("win", "d")


def close_active_window() -> None:
    """Close the last non-gesture window."""
    _prepare_for_hotkey()
    if is_linux():
        if _last_other_window and shutil.which("wmctrl"):
            _wmctrl_close(_last_other_window)
            return
        _focus_last_window()
        if is_wayland() and shutil.which("ydotool"):
            _ydotool_key("alt", "F4")
        else:
            _xdotool_key("alt+F4")
        return
    _win_key_combo(0x12, 0x73)


def open_app_launcher() -> None:
    """Open an app launcher / run dialog.
    On Pi/Wayland without ydotool: opens the LXDE run dialog, then falls back
    to the file manager, then a terminal.
    """
    _prepare_for_hotkey()
    if is_linux():
        # lxpanelctl is the LXDE run-command dialog (works on Pi OS without ydotool)
        if shutil.which("lxpanelctl"):
            subprocess.Popen(["lxpanelctl", "run"], start_new_session=True)
            return
        # rofi / dmenu are lightweight launchers sometimes installed on Pi
        for launcher in (["rofi", "-show", "run"], ["dmenu_run"]):
            if shutil.which(launcher[0]):
                subprocess.Popen(launcher, start_new_session=True,
                                 env=_xenv())
                return
        # ydotool Super key (Wayland) — only if installed
        if is_wayland() and shutil.which("ydotool"):
            _ydotool_key("super")
            return
        # X11 fallback: xdotool Super key
        _xdotool_key("super")
        return
    _press("win")


def execute_desktop_gesture(gesture: str) -> None:
    if gesture == "thumbs_up":
        switch_application()
    elif gesture == "thumbs_down":
        open_terminal()
    elif gesture == "open_palm":
        open_app_launcher()
    elif gesture == "fist_close":
        close_active_window()
    else:
        raise ValueError(f"Unknown desktop gesture: {gesture}")
