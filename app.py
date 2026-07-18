"""
Flask web app for the drowsiness detection project.

Streams the annotated webcam feed to the browser and exposes a
/status endpoint the front-end polls to show a live alert banner.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from detector import DrowsinessDetector

app = Flask(__name__)

# Shared state updated by the capture thread, read by Flask routes
state_lock = threading.Lock()
state = {
    "status_text": "Starting...",
    "is_alert": False,
    "ear": None,
}

latest_frame = None
frame_lock = threading.Lock()


def capture_loop():
    global latest_frame

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with state_lock:
            state["status_text"] = "Error: could not open webcam"
        return

    detector = DrowsinessDetector()

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        annotated, status_text, is_alert, ear_value = detector.process_frame(frame)

        with state_lock:
            state["status_text"] = status_text
            state["is_alert"] = is_alert
            state["ear"] = round(ear_value, 3) if ear_value is not None else None

        ok, buffer = cv2.imencode(".jpg", annotated)
        if ok:
            with frame_lock:
                latest_frame = buffer.tobytes()

        time.sleep(0.01)


def gen_frames():
    while True:
        with frame_lock:
            frame_bytes = latest_frame
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with state_lock:
        return jsonify(state)


if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(debug=False, threaded=True)
