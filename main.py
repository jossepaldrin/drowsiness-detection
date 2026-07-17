"""
Drowsiness Detection using OpenCV + MediaPipe Face Landmarker (Tasks API)
--------------------------------------------------------------------------
Detects eye landmarks in real time from a webcam feed, computes the
Eye Aspect Ratio (EAR), and raises an alert if the eyes stay closed
(EAR below threshold) for longer than a set number of consecutive frames.

Uses MediaPipe's current Face Landmarker Task API (the older
`mp.solutions.face_mesh` API was removed in recent mediapipe releases).

Requires the face_landmarker.task model file in the project folder —
see README.md for the download step.

Run:
    python main.py
Quit:
    press 'q' in the video window
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import winsound  # Windows-only beep; swap for playsound/simpleaudio on Linux/Mac

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.22          # below this = eye considered "closed"
CONSEC_FRAMES = 20            # frames closed in a row before alert triggers
ALERT_COOLDOWN_SEC = 3        # avoid spamming the alarm
MODEL_PATH = "face_landmarker.task"

# Face Landmarker uses the same 478-point mesh topology as the old Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    """Compute EAR for one eye given 6 landmark indices in order:
    [p1 (left corner), p2, p3, p4 (right corner), p5, p6]
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * image_w, lm.y * image_h))

    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def build_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing '{MODEL_PATH}'. Download it first — see README.md."
        )
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    landmarker = build_landmarker()

    closed_frames = 0
    last_alert_time = 0
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_idx * (1000 / 30))  # assume ~30 fps
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        status_text = "No face detected"
        status_color = (0, 165, 255)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= CONSEC_FRAMES:
                status_text = "DROWSINESS ALERT!"
                status_color = (0, 0, 255)
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN_SEC:
                    try:
                        winsound.Beep(1000, 500)
                    except RuntimeError:
                        pass  # non-Windows fallback: swap in playsound here
                    last_alert_time = now
            else:
                status_text = f"EAR: {avg_ear:.2f}"
                status_color = (0, 255, 0)

            for idx in LEFT_EYE + RIGHT_EYE:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (255, 255, 0), -1)

        cv2.putText(
            frame, status_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2
        )
        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
