# Drowsiness Detection using OpenCV + MediaPipe

Real-time drowsiness detection from a webcam feed. Uses MediaPipe Face Mesh
to locate eye landmarks, computes the Eye Aspect Ratio (EAR), and triggers
an audible alert if eyes stay closed for too many consecutive frames.

## How it works
1. Capture webcam frames with OpenCV.
2. Run MediaPipe's Face Landmarker (Tasks API) to get ~478 facial landmarks.
3. Extract 6 landmarks per eye and compute EAR:
   `EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`
4. If EAR drops below `EAR_THRESHOLD` for `CONSEC_FRAMES` frames in a row,
   flag drowsiness and sound an alert.

> Note: this uses MediaPipe's current **Face Landmarker Task API**, not the
> older `mp.solutions.face_mesh` API — recent mediapipe releases (0.10.30+)
> dropped the legacy solutions module entirely.

## Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python download_model.py   # one-time: downloads face_landmarker.task
python main.py
```

Press `q` to quit the video window.

## Tuning
- `EAR_THRESHOLD` (default 0.22): lower it if the alert fires too easily,
  raise it if it's missing genuinely closed eyes. Depends on your camera
  angle and face shape.
- `CONSEC_FRAMES` (default 20): roughly how many frames of closed eyes
  (at ~20-30 fps) before it counts as drowsiness, not just a blink.

## Notes
- Alert sound uses `winsound` (Windows-only). On Linux/Mac, swap it for
  `playsound` or `simpleaudio`.
- Works best with decent, front-facing lighting.

## Possible extensions
- Log drowsiness events with timestamps to a CSV.
- Add yawning detection using mouth landmarks.
- Add a head-tilt / nod detection as a second signal.
