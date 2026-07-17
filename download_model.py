"""
One-time setup script: downloads the MediaPipe Face Landmarker model file
needed by main.py. Run this once before running main.py.

    python download_model.py
"""

import requests

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
OUT_PATH = "face_landmarker.task"


def main():
    print(f"Downloading model from {MODEL_URL} ...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(OUT_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
