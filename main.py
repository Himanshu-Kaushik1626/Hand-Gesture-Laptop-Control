"""
main.py
-------
Entry point for the Virtual Mouse Hand-Gesture Control system.
Starts the CustomTkinter UI, manages the webcam loop in a background
thread, and wires HandTracker + MouseController together.
"""

import threading
import cv2

from hand_tracking   import HandTracker
from mouse_controller import MouseController
from ui               import App


class VirtualMouseApp:
    """Orchestrates the webcam thread, tracker, and controller."""

    CAM_INDEX = 0          # Change if you have multiple cameras
    CAM_W, CAM_H = 640, 480

    def __init__(self):
        self._running   = False
        self._thread    = None
        self.tracker    = None
        self.controller = None

        # Build the UI and register callbacks
        self.ui = App(
            on_start       = self.start_camera,
            on_stop        = self.stop_camera,
            on_sensitivity = self._on_sensitivity,
        )

    # ──────────────────────── camera thread ───────────────────────────────────

    def start_camera(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._camera_loop,
                                         daemon=True, name="CamThread")
        self._thread.start()

    def stop_camera(self):
        self._running = False
        # Thread will stop on next iteration

    def _camera_loop(self):
        """Runs in a daemon thread.  Opens webcam, processes frames, queues UI updates."""
        cap = None

        # Try backends in order: default → DirectShow → Microsoft Media Foundation
        for backend in [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            if backend is None:
                cap = cv2.VideoCapture(self.CAM_INDEX)
            else:
                cap = cv2.VideoCapture(self.CAM_INDEX, backend)
            if cap.isOpened():
                break
            cap.release()

        if cap is None or not cap.isOpened():
            self.ui.push_status("Camera Error", False, 0)
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise buffer lag

        # Drain warm-up frames (camera often returns black for first few frames)
        for _ in range(8):
            cap.read()

        self.tracker    = HandTracker(max_hands=1)
        self.controller = MouseController(cam_w=self.CAM_W, cam_h=self.CAM_H,
                                          sensitivity=self.ui.sens_slider.get())

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so movement feels natural
            frame = cv2.flip(frame, 1)

            # Hand tracking
            frame, hand_detected = self.tracker.process_frame(frame)

            # Mouse control
            gesture = self.controller.process(self.tracker, hand_detected)

            # Overlay: FPS + gesture text on the frame itself
            self._draw_overlay(frame, gesture, hand_detected,
                               self.controller.fps)

            # Send to UI
            self.ui.push_frame(frame)
            self.ui.push_status(gesture, hand_detected, self.controller.fps)

        cap.release()
        if self.tracker:
            self.tracker.release()
        self._running = False

    # ──────────────────────── overlay drawing ─────────────────────────────────

    @staticmethod
    def _draw_overlay(frame, gesture: str, hand_detected: bool, fps: float):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent top bar
        cv2.rectangle(overlay, (0, 0), (w, 48), (13, 17, 23), -1)
        cv2.addWeighted(frame, 0.25, overlay, 0.75, 0, frame)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 229, 255), 2)

        # Gesture
        cv2.putText(frame, f"Gesture: {gesture}", (120, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (74, 222, 128), 2)

        # Hand status badge (bottom-right)
        status_text  = "Hand Detected" if hand_detected else "No Hand"
        status_color = (74, 222, 128) if hand_detected else (248, 113, 113)
        cv2.putText(frame, status_text, (w - 200, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # ──────────────────────── sensitivity callback ────────────────────────────

    def _on_sensitivity(self, value):
        if self.controller:
            self.controller.set_sensitivity(value)

    # ──────────────────────── run ─────────────────────────────────────────────

    def run(self):
        self.ui.protocol("WM_DELETE_WINDOW", self._on_close)
        self.ui.mainloop()

    def _on_close(self):
        self.stop_camera()
        self.ui.destroy()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = VirtualMouseApp()
    app.run()
