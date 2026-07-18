"""
Shared drowsiness-detection logic.

Encapsulates the MediaPipe Face Landmarker setup, EAR calculation, and
frame annotation so both the CLI script (main.py) and the Flask web app
(app.py) can reuse it without duplicating code.
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20
ALERT_COOLDOWN_SEC = 3
MODEL_PATH = "face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * image_w, lm.y * image_h))

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class DrowsinessDetector:
    """Wraps the Face Landmarker model and EAR-based drowsiness logic.

    Call process_frame(frame) once per captured frame. It returns:
        (annotated_frame, status_text, is_alert, ear_value)
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Missing '{MODEL_PATH}'. Run download_model.py first."
            )
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.closed_frames = 0
        self.last_alert_time = 0
        self.frame_idx = 0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(self.frame_idx * (1000 / 30))
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.frame_idx += 1

        status_text = "No face detected"
        status_color = (0, 165, 255)
        is_alert = False
        ear_value = None

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear_value = (left_ear + right_ear) / 2.0

            if ear_value < EAR_THRESHOLD:
                self.closed_frames += 1
            else:
                self.closed_frames = 0

            if self.closed_frames >= CONSEC_FRAMES:
                status_text = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)
                is_alert = True
                self.last_alert_time = time.time()
            else:
                status_text = f"EAR: {ear_value:.2f}"
                status_color = (0, 255, 0)

            for idx in LEFT_EYE + RIGHT_EYE:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (255, 255, 0), -1)

        cv2.putText(
            frame, status_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2
        )

        return frame, status_text, is_alert, ear_value
