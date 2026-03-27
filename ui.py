"""
ui.py
-----
Modern dark-mode dashboard built with CustomTkinter.
Runs the webcam loop in a background thread and pushes frames + status
to the UI via a thread-safe queue.
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

# ──────────────────────────── appearance ──────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─────────────────────── palette / design tokens ──────────────────────────────
BG_DARK   = "#0d1117"
BG_CARD   = "#161b22"
BG_CARD2  = "#1c2230"
ACCENT    = "#00e5ff"
ACCENT2   = "#4ade80"
DANGER    = "#f87171"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
BORDER    = "#30363d"

GESTURE_ICONS = {
    "Move Cursor": "🖱️",
    "Left Click" : "👆",
    "Right Click": "✌️",
    "Scroll"     : "🔄",
    "Drag"       : "✊",
    "No Hand"    : "🤚",
    "None"       : "—",
}


# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    """Main application window."""

    QUEUE_INTERVAL_MS = 15          # UI poll interval

    def __init__(self, on_start=None, on_stop=None, on_sensitivity=None):
        super().__init__()

        self.on_start       = on_start or (lambda: None)
        self.on_stop        = on_stop  or (lambda: None)
        self.on_sensitivity = on_sensitivity or (lambda v: None)

        # Thread-safe queues filled by the webcam thread
        self.frame_queue  = queue.Queue(maxsize=3)
        self.status_queue = queue.Queue(maxsize=20)

        self._build_ui()
        self._poll_queues()

    # ──────────────────────── UI construction ─────────────────────────────────

    def _build_ui(self):
        self.title("🖐  Virtual Mouse – Hand Gesture Control")
        self.geometry("1180x720")
        self.minsize(900, 600)
        self.configure(fg_color=BG_DARK)
        self.resizable(True, True)

        # ── Top header bar ─────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="  🖐  Virtual Mouse Control",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left", padx=20, pady=10)

        self.fps_label = ctk.CTkLabel(
            header,
            text="FPS: --",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=TEXT_SEC,
        )
        self.fps_label.pack(side="right", padx=20)

        self.clock_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=TEXT_SEC,
        )
        self.clock_label.pack(side="right", padx=10)
        self._tick_clock()

        # ── Main body ──────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        body.pack(fill="both", expand=True, padx=14, pady=(8, 14))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: webcam panel
        self._build_camera_panel(body)

        # Right: control + status panel
        self._build_side_panel(body)

    def _build_camera_panel(self, parent):
        cam_outer = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=14,
                                 border_width=1, border_color=BORDER)
        cam_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        cam_outer.rowconfigure(1, weight=1)
        cam_outer.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cam_outer, text="📷  Live Webcam Feed",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            text_color=TEXT_PRI,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        # Frame to hold the canvas (so it stretches nicely)
        cam_frame = ctk.CTkFrame(cam_outer, fg_color="#000000", corner_radius=8)
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        cam_frame.rowconfigure(0, weight=1)
        cam_frame.columnconfigure(0, weight=1)

        self.cam_canvas = tk.Canvas(cam_frame, bg="#000000",
                                    highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0, sticky="nsew")
        self.cam_canvas.bind("<Configure>", self._on_canvas_resize)

        self._canvas_size = (640, 480)
        self._cam_photo   = None

        # Placeholder text when camera is off
        self.cam_placeholder = ctk.CTkLabel(
            cam_frame,
            text="📷\n\nCamera is Off\nPress  Start  to begin",
            font=ctk.CTkFont("Segoe UI", 16),
            text_color=TEXT_SEC,
        )
        self.cam_placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _build_side_panel(self, parent):
        side = ctk.CTkFrame(parent, fg_color=BG_DARK, corner_radius=0)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        for r in range(5):
            side.rowconfigure(r, weight=0)
        side.rowconfigure(5, weight=1)

        # ── Hand Status card ──────────────────────────────────────────────
        self.status_card = ctk.CTkFrame(side, fg_color=BG_CARD, corner_radius=12,
                                        border_width=1, border_color=BORDER)
        self.status_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(self.status_card, text="Detection Status",
                     font=ctk.CTkFont("Segoe UI", 12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=14, pady=(10, 2))

        self.status_dot = ctk.CTkLabel(
            self.status_card, text="⬤  No Hand Detected",
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
            text_color=DANGER,
        )
        self.status_dot.pack(anchor="w", padx=14, pady=(0, 10))

        # ── Current Gesture card ─────────────────────────────────────────
        gest_card = ctk.CTkFrame(side, fg_color=BG_CARD, corner_radius=12,
                                 border_width=1, border_color=BORDER)
        gest_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(gest_card, text="Current Gesture",
                     font=ctk.CTkFont("Segoe UI", 12), text_color=TEXT_SEC
                     ).pack(anchor="w", padx=14, pady=(10, 0))

        self.gest_icon = ctk.CTkLabel(
            gest_card, text="🤚",
            font=ctk.CTkFont("Segoe UI", 40),
        )
        self.gest_icon.pack(pady=(4, 0))

        self.gest_label = ctk.CTkLabel(
            gest_card, text="No Hand",
            font=ctk.CTkFont("Segoe UI", 18, "bold"),
            text_color=ACCENT,
        )
        self.gest_label.pack(pady=(0, 10))

        # ── Gesture legend ─────────────────────────────────────────────
        leg_card = ctk.CTkFrame(side, fg_color=BG_CARD2, corner_radius=12,
                                border_width=1, border_color=BORDER)
        leg_card.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(leg_card, text="Gesture Map",
                     font=ctk.CTkFont("Segoe UI", 12), text_color=TEXT_SEC,
                     ).pack(anchor="w", padx=14, pady=(10, 4))

        gesture_map = [
            ("🖱️", "Move Cursor", "Index finger only"),
            ("👆", "Left Click",  "Pinch index + thumb"),
            ("✌️", "Right Click", "Pinch middle + thumb"),
            ("🔄", "Scroll",      "Index + middle (close)"),
            ("✊", "Drag",        "Fist (hold 0.4 s)"),
        ]
        for icon, name, hint in gesture_map:
            row_f = ctk.CTkFrame(leg_card, fg_color="transparent")
            row_f.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row_f, text=icon, font=ctk.CTkFont(size=16),
                         width=28).pack(side="left")
            ctk.CTkLabel(row_f, text=name,
                         font=ctk.CTkFont("Segoe UI", 12, "bold"),
                         text_color=TEXT_PRI).pack(side="left", padx=(4, 0))
            ctk.CTkLabel(row_f, text=f"  {hint}",
                         font=ctk.CTkFont("Segoe UI", 11),
                         text_color=TEXT_SEC).pack(side="left")

        ctk.CTkFrame(leg_card, height=8, fg_color="transparent").pack()

        # ── Sensitivity slider ─────────────────────────────────────────
        sens_card = ctk.CTkFrame(side, fg_color=BG_CARD, corner_radius=12,
                                 border_width=1, border_color=BORDER)
        sens_card.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ctk.CTkLabel(sens_card, text="Cursor Sensitivity",
                     font=ctk.CTkFont("Segoe UI", 12), text_color=TEXT_SEC,
                     ).pack(anchor="w", padx=14, pady=(10, 2))

        self.sens_val_label = ctk.CTkLabel(
            sens_card, text="1.0×",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=ACCENT,
        )
        self.sens_val_label.pack(anchor="e", padx=14)

        self.sens_slider = ctk.CTkSlider(
            sens_card, from_=0.5, to=3.0, number_of_steps=25,
            command=self._on_sensitivity_change,
            button_color=ACCENT, button_hover_color=ACCENT2,
            progress_color=ACCENT,
        )
        self.sens_slider.set(1.0)
        self.sens_slider.pack(fill="x", padx=14, pady=(0, 12))

        # ── Start / Stop buttons ───────────────────────────────────────
        btn_frame = ctk.CTkFrame(side, fg_color="transparent")
        btn_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶  Start",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color="#22c55e", hover_color="#16a34a",
            corner_radius=10, height=42,
            command=self._on_start,
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="⏹  Stop",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color=DANGER, hover_color="#dc2626",
            corner_radius=10, height=42,
            state="disabled",
            command=self._on_stop,
        )
        self.stop_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # ── Stats card ────────────────────────────────────────────────
        stats_card = ctk.CTkFrame(side, fg_color=BG_CARD2, corner_radius=12,
                                  border_width=1, border_color=BORDER)
        stats_card.grid(row=5, column=0, sticky="nsew")

        ctk.CTkLabel(stats_card, text="Session Stats",
                     font=ctk.CTkFont("Segoe UI", 12), text_color=TEXT_SEC,
                     ).pack(anchor="w", padx=14, pady=(10, 4))

        self._stats = {
            "Left Clicks":  0,
            "Right Clicks": 0,
            "Scrolls":      0,
            "Drags":        0,
        }
        self._stat_labels = {}
        for key, val in self._stats.items():
            r = ctk.CTkFrame(stats_card, fg_color="transparent")
            r.pack(fill="x", padx=14, pady=2)
            ctk.CTkLabel(r, text=key,
                         font=ctk.CTkFont("Segoe UI", 12), text_color=TEXT_SEC,
                         ).pack(side="left")
            lbl = ctk.CTkLabel(r, text=str(val),
                               font=ctk.CTkFont("Segoe UI", 12, "bold"),
                               text_color=ACCENT2)
            lbl.pack(side="right")
            self._stat_labels[key] = lbl

        ctk.CTkFrame(stats_card, height=8, fg_color="transparent").pack()

    # ──────────────────────── event callbacks ─────────────────────────────────

    def _on_start(self):
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.cam_placeholder.place_forget()
        self.on_start()

    def _on_stop(self):
        self.stop_btn.configure(state="disabled")
        self.start_btn.configure(state="normal")
        self.cam_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.on_stop()

    def _on_sensitivity_change(self, value):
        self.sens_val_label.configure(text=f"{float(value):.1f}×")
        self.on_sensitivity(value)

    def _on_canvas_resize(self, event):
        self._canvas_size = (event.width, event.height)

    # ──────────────────────── frame / status ingestion ────────────────────────

    def push_frame(self, bgr_frame):
        """Called from worker thread – non-blocking."""
        try:
            self.frame_queue.put_nowait(bgr_frame)
        except queue.Full:
            pass

    def push_status(self, gesture: str, hand_detected: bool, fps: float):
        """Called from worker thread – non-blocking."""
        try:
            self.status_queue.put_nowait((gesture, hand_detected, fps))
        except queue.Full:
            pass

    # ──────────────────────── polling loop ────────────────────────────────────

    def _poll_queues(self):
        # Drain the frame queue (show the latest frame only)
        latest_frame = None
        while not self.frame_queue.empty():
            try:
                latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None:
            self._display_frame(latest_frame)

        # Drain the status queue
        while not self.status_queue.empty():
            try:
                gesture, hand_detected, fps = self.status_queue.get_nowait()
                self._update_status(gesture, hand_detected, fps)
            except queue.Empty:
                break

        self.after(self.QUEUE_INTERVAL_MS, self._poll_queues)

    def _display_frame(self, bgr_frame):
        cw, ch = self._canvas_size
        if cw < 10 or ch < 10:
            return

        # Letterbox resize
        fh, fw = bgr_frame.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb)
        self._cam_photo = ImageTk.PhotoImage(img)

        self.cam_canvas.delete("all")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.cam_canvas.create_image(ox, oy, anchor="nw",
                                     image=self._cam_photo)

    def _update_status(self, gesture: str, hand_detected: bool, fps: float):
        if hand_detected:
            self.status_dot.configure(text=f"⬤  Hand Detected",
                                      text_color=ACCENT2)
        else:
            self.status_dot.configure(text=f"⬤  No Hand Detected",
                                      text_color=DANGER)

        icon = GESTURE_ICONS.get(gesture, "—")
        self.gest_icon.configure(text=icon)
        self.gest_label.configure(text=gesture)
        self.fps_label.configure(text=f"FPS: {fps:.0f}")

        # Update session stats
        if gesture == "Left Click":
            self._stats["Left Clicks"]  += 1
        elif gesture == "Right Click":
            self._stats["Right Clicks"] += 1
        elif gesture == "Scroll":
            self._stats["Scrolls"]      += 1
        elif gesture == "Drag":
            self._stats["Drags"]        += 1

        for key, lbl in self._stat_labels.items():
            lbl.configure(text=str(self._stats[key]))

    # ──────────────────────── clock ───────────────────────────────────────────

    def _tick_clock(self):
        t = time.strftime("%H:%M:%S")
        self.clock_label.configure(text=t)
        self.after(1000, self._tick_clock)
