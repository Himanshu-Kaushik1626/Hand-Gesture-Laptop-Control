"""
hand_tracking.py  (MediaPipe Tasks API – mediapipe >= 0.10, Python 3.13)
-------------------------------------------------------------------------
Uses HandLandmarker from mediapipe.tasks.python.vision.
All drawing is done with plain OpenCV (no mp.solutions dependency).
"""

import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks        import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── model path ────────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")

# ── landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
INDEX_PIP  = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP   = 16
RING_PIP   = 14
PINKY_TIP  = 20
PINKY_PIP  = 18

# MediaPipe hand skeleton connections (0-indexed)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),            # thumb
    (0,5),(5,6),(6,7),(7,8),            # index
    (5,9),(9,10),(10,11),(11,12),       # middle
    (9,13),(13,14),(14,15),(15,16),     # ring
    (13,17),(17,18),(18,19),(19,20),    # pinky
    (0,17),                             # palm base
]


class HandTracker:
    """
    Wraps MediaPipe Tasks HandLandmarker.
    Public interface is identical to the old mp.solutions version.
    """

    def __init__(self, max_hands: int = 1,
                 detection_conf: float = 0.7,
                 tracking_conf: float  = 0.7):

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=tracking_conf,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp  = 0          # strictly-increasing ms counter

        self.landmarks   = []         # list[NormalizedLandmark] for hand-0
        self.img_shape   = (480, 640) # (h, w) – updated each frame

    # ── core ──────────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Detect hands in *frame* (BGR uint8).
        Returns (annotated_frame, hand_detected_bool).
        """
        self.img_shape = frame.shape[:2]   # (h, w)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp += 33          # ~30 fps — must be strictly increasing
        result = self._landmarker.detect_for_video(mp_img, self._timestamp)

        self.landmarks = []
        if result.hand_landmarks:
            self.landmarks = result.hand_landmarks[0]
            self._draw_landmarks(frame)
            return frame, True

        return frame, False

    def _draw_landmarks(self, frame):
        """Draw skeleton using plain OpenCV — no mp.solutions needed."""
        h, w = self.img_shape

        # Convert normalised coords → pixel coords
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in self.landmarks]

        # Draw connections
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 170, 255), 2)

        # Draw landmark dots
        for pt in pts:
            cv2.circle(frame, pt, 5, (0, 255, 170), -1)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1)

    # ── position helpers ──────────────────────────────────────────────────────

    def get_landmark_pos(self, idx: int):
        if not self.landmarks or idx >= len(self.landmarks):
            return None
        h, w = self.img_shape
        lm = self.landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    def get_index_tip(self):   return self.get_landmark_pos(INDEX_TIP)
    def get_thumb_tip(self):   return self.get_landmark_pos(THUMB_TIP)
    def get_middle_tip(self):  return self.get_landmark_pos(MIDDLE_TIP)
    def get_ring_tip(self):    return self.get_landmark_pos(RING_TIP)
    def get_pinky_tip(self):   return self.get_landmark_pos(PINKY_TIP)

    # ── geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def distance(p1, p2) -> float:
        if p1 is None or p2 is None:
            return float('inf')
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # ── finger-up checks ──────────────────────────────────────────────────────

    def _finger_up(self, tip_idx: int, pip_idx: int) -> bool:
        if not self.landmarks:
            return False
        h, _ = self.img_shape
        return self.landmarks[tip_idx].y * h < self.landmarks[pip_idx].y * h

    def index_up(self)  -> bool: return self._finger_up(INDEX_TIP,  INDEX_PIP)
    def middle_up(self) -> bool: return self._finger_up(MIDDLE_TIP, MIDDLE_PIP)
    def ring_up(self)   -> bool: return self._finger_up(RING_TIP,   RING_PIP)
    def pinky_up(self)  -> bool: return self._finger_up(PINKY_TIP,  PINKY_PIP)

    def thumb_up(self) -> bool:
        if not self.landmarks:
            return False
        _, w = self.img_shape
        tip_x  = self.landmarks[THUMB_TIP].x * w
        base_x = self.landmarks[THUMB_TIP - 1].x * w
        return tip_x < base_x

    def fingers_up_list(self) -> list:
        """`[thumb, index, middle, ring, pinky]` – True = extended."""
        return [self.thumb_up(), self.index_up(), self.middle_up(),
                self.ring_up(), self.pinky_up()]

    # ── cleanup ───────────────────────────────────────────────────────────────

    def release(self):
        try:
            self._landmarker.close()
        except Exception:
            pass
