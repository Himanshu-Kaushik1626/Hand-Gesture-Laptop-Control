"""
mouse_controller.py
-------------------
Translates hand landmark data into concrete mouse actions using PyAutoGUI.
Implements smoothing, click cooldown, and all gesture mappings.
"""

import time
import pyautogui
import numpy as np
from hand_tracking import HandTracker

# Safety – PyAutoGUI won't raise on FAIL_SAFE by default; let's keep it False
# so moving to corners doesn't kill the app while developing.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0          # We handle our own timing

# ─────────────────────────── constants ────────────────────────────────────────
CLICK_COOLDOWN   = 0.5   # seconds between clicks
SCROLL_COOLDOWN  = 0.08  # seconds between scroll events
DRAG_HOLD_TIME   = 0.4   # seconds index+thumb must overlap to start drag
PINCH_THRESHOLD  = 40    # px distance for left-click pinch
RIGHT_THRESHOLD  = 40    # px distance for right-click gesture
SCROLL_THRESHOLD = 50    # px distance for two-finger scroll trigger

# Smoothing: exponential moving average coefficient (higher = snappier)
ALPHA = 0.25


class MouseController:
    """
    Consumes a HandTracker instance each frame and dispatches PyAutoGUI calls.
    """

    def __init__(self, cam_w: int = 640, cam_h: int = 480,
                 sensitivity: float = 1.0):
        self.cam_w       = cam_w
        self.cam_h       = cam_h
        self.sensitivity = sensitivity          # 0.5 – 3.0

        # Screen resolution
        self.screen_w, self.screen_h = pyautogui.size()

        # Smoothed cursor position (initialised to screen centre)
        self.smooth_x = self.screen_w / 2
        self.smooth_y = self.screen_h / 2

        # State tracking
        self._last_click_time   = 0.0
        self._last_rclick_time  = 0.0
        self._last_scroll_time  = 0.0
        self._drag_start_time   = 0.0
        self._dragging          = False
        self._prev_gesture      = None
        self._prev_scroll_y     = None

        # Public status (read by UI)
        self.current_gesture    = "None"
        self.fps                = 0.0
        self._frame_times       = []

    # ──────────────────────── public API ──────────────────────────────────────

    def process(self, tracker: HandTracker, hand_detected: bool) -> str:
        """
        Call once per frame.  Returns a gesture label string.
        """
        self._update_fps()

        if not hand_detected:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False
            self.current_gesture = "No Hand"
            return self.current_gesture

        fingers = tracker.fingers_up_list()  # [thumb, idx, mid, ring, pinky]

        index_tip  = tracker.get_index_tip()
        thumb_tip  = tracker.get_thumb_tip()
        middle_tip = tracker.get_middle_tip()

        # ── 1. Cursor Move (only index finger up) ─────────────────────────────
        if fingers[1] and not fingers[2]:
            self._move_cursor(index_tip, tracker)
            self.current_gesture = "Move Cursor"

        # ── 2. Left Click  (index + thumb pinch) ──────────────────────────────
        dist_pinch = tracker.distance(index_tip, thumb_tip)
        if dist_pinch < PINCH_THRESHOLD:
            now = time.time()
            if now - self._last_click_time > CLICK_COOLDOWN:
                pyautogui.click()
                self._last_click_time = now
                self.current_gesture  = "Left Click"

        # ── 3. Right Click  (middle + thumb pinch while index is up) ──────────
        elif fingers[1] and fingers[2] and not fingers[3]:
            dist_rc = tracker.distance(middle_tip, thumb_tip)
            if dist_rc < RIGHT_THRESHOLD:
                now = time.time()
                if now - self._last_rclick_time > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    self._last_rclick_time = now
                    self.current_gesture   = "Right Click"
            else:
                self.current_gesture = "Move Cursor"

        # ── 4. Scroll  (index + middle up, close together) ────────────────────
        elif fingers[1] and fingers[2] and not fingers[0]:
            if middle_tip and index_tip:
                mid_gap = tracker.distance(index_tip, middle_tip)
                if mid_gap < SCROLL_THRESHOLD:
                    # Use vertical change of index tip for scroll direction
                    norm_y = index_tip[1] / tracker.img_shape[0]
                    if self._prev_scroll_y is not None:
                        delta = self._prev_scroll_y - norm_y  # + = up
                        now = time.time()
                        if abs(delta) > 0.005 and now - self._last_scroll_time > SCROLL_COOLDOWN:
                            clicks = int(delta * 20)
                            if clicks != 0:
                                pyautogui.scroll(clicks)
                            self._last_scroll_time = now
                    self._prev_scroll_y  = norm_y
                    self.current_gesture = "Scroll"
        else:
            self._prev_scroll_y = None

        # ── 5. Drag  (fist: all fingers down + hold) ──────────────────────────
        if not any(fingers[1:]):           # index, middle, ring, pinky all down
            if not self._dragging:
                if self._drag_start_time == 0.0:
                    self._drag_start_time = time.time()
                elif time.time() - self._drag_start_time > DRAG_HOLD_TIME:
                    pyautogui.mouseDown()
                    self._dragging = True
                    self.current_gesture = "Drag"
            else:
                # While dragging, move to index position
                self._move_cursor(index_tip, tracker)
                self.current_gesture = "Drag"
        else:
            self._drag_start_time = 0.0
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging     = False

        return self.current_gesture

    # ──────────────────────── helpers ─────────────────────────────────────────

    def _move_cursor(self, pos, tracker: HandTracker):
        """Map camera pixel -> screen pixel with smoothing."""
        if pos is None:
            return
        h, w = tracker.img_shape
        # Normalise, apply sensitivity relative to screen
        nx = np.clip(pos[0] / w, 0.05, 0.95)
        ny = np.clip(pos[1] / h, 0.05, 0.95)

        # Scale sensitivity around centre
        cx, cy = 0.5, 0.5
        sx = cx + (nx - cx) * self.sensitivity
        sy = cy + (ny - cy) * self.sensitivity
        sx = np.clip(sx, 0.0, 1.0)
        sy = np.clip(sy, 0.0, 1.0)

        target_x = sx * self.screen_w
        target_y = sy * self.screen_h

        # Exponential moving average smoothing
        self.smooth_x += ALPHA * (target_x - self.smooth_x)
        self.smooth_y += ALPHA * (target_y - self.smooth_y)

        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

    def _update_fps(self):
        now = time.time()
        self._frame_times.append(now)
        # Keep only last 30 samples
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        self.fps = len(self._frame_times)

    def set_sensitivity(self, value: float):
        """Update sensitivity from UI slider (value 0.5 – 3.0)."""
        self.sensitivity = float(value)
