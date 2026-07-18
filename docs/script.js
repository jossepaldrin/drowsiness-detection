import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// Config (mirrors the Python version's thresholds)
// ---------------------------------------------------------------------------
const EAR_THRESHOLD = 0.22;
const CONSEC_FRAMES = 20;
const ALERT_COOLDOWN_MS = 3000;

// Same 478-point face mesh topology as the Python Face Landmarker
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const banner = document.getElementById("banner");
const earValueEl = document.getElementById("ear-value");
const startBtn = document.getElementById("startBtn");

let faceLandmarker = null;
let closedFrames = 0;
let lastAlertTime = 0;
let running = false;

// Simple beep using the Web Audio API — no external audio file needed.
function playBeep() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();
  const oscillator = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  oscillator.type = "sine";
  oscillator.frequency.value = 1000;
  gain.gain.value = 0.15;
  oscillator.connect(gain);
  gain.connect(audioCtx.destination);
  oscillator.start();
  setTimeout(() => {
    oscillator.stop();
    audioCtx.close();
  }, 400);
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function eyeAspectRatio(landmarks, eyeIndices) {
  const [p1, p2, p3, p4, p5, p6] = eyeIndices.map((i) => landmarks[i]);
  const vertical1 = distance(p2, p6);
  const vertical2 = distance(p3, p5);
  const horizontal = distance(p1, p4);
  return (vertical1 + vertical2) / (2.0 * horizontal);
}

function setBanner(text, isAlert) {
  banner.textContent = text;
  banner.className = "banner " + (isAlert ? "alert" : "ok");
}

async function initFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

function drawEyeLandmarks(landmarks, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "cyan";
  for (const idx of [...LEFT_EYE, ...RIGHT_EYE]) {
    const lm = landmarks[idx];
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function detectLoop() {
  if (!running) return;

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (w === 0 || h === 0) {
    requestAnimationFrame(detectLoop);
    return;
  }
  canvas.width = w;
  canvas.height = h;

  const nowMs = performance.now();
  const result = faceLandmarker.detectForVideo(video, nowMs);

  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    const landmarks = result.faceLandmarks[0];

    const leftEar = eyeAspectRatio(landmarks, LEFT_EYE);
    const rightEar = eyeAspectRatio(landmarks, RIGHT_EYE);
    const avgEar = (leftEar + rightEar) / 2.0;

    earValueEl.textContent = avgEar.toFixed(2);
    drawEyeLandmarks(landmarks, w, h);

    if (avgEar < EAR_THRESHOLD) {
      closedFrames += 1;
    } else {
      closedFrames = 0;
    }

    if (closedFrames >= CONSEC_FRAMES) {
      setBanner("DROWSINESS ALERT!", true);
      const now = Date.now();
      if (now - lastAlertTime > ALERT_COOLDOWN_MS) {
        playBeep();
        lastAlertTime = now;
      }
    } else {
      setBanner(`Watching — EAR ${avgEar.toFixed(2)}`, false);
    }
  } else {
    ctx.clearRect(0, 0, w, h);
    earValueEl.textContent = "--";
    setBanner("No face detected", false);
    closedFrames = 0;
  }

  requestAnimationFrame(detectLoop);
}

async function start() {
  startBtn.disabled = true;
  setBanner("Loading model...", false);

  try {
    if (!faceLandmarker) {
      await initFaceLandmarker();
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve();
    });

    running = true;
    setBanner("Watching...", false);
    requestAnimationFrame(detectLoop);
  } catch (err) {
    console.error(err);
    setBanner("Error: " + err.message, true);
    startBtn.disabled = false;
  }
}

startBtn.addEventListener("click", start);
