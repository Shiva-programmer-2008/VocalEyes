/**
 * VocalEyes v3.0 — Frontend Application
 * Real-time object detection with WebSocket & TTS
 */

"use strict";

// ─────────────────────────────────────────────────────────────
// CONSTANTS & CONFIG
// ─────────────────────────────────────────────────────────────

const CONFIG = {
  FRAME_INTERVAL_MS: 100,      // How often to send frames (10fps)
  JPEG_QUALITY: 0.75,
  MAX_LOG_ENTRIES: 50,
  STATS_REFRESH_MS: 2000,
  RECONNECT_DELAY_MS: 3000,
  DANGER_BEEP_FREQ: 880,
  DANGER_BEEP_DURATION: 0.3,
  TTS_COOLDOWN_MS: 1500,
};

const PRIORITY_COLORS = {
  CRITICAL: "#ff2d55",
  HIGH:     "#ff9500",
  MEDIUM:   "#ffcc00",
  LOW:      "#34c759",
  INFO:     "#636366",
};

const KEYBOARD_SHORTCUTS = {
  "c": () => startCamera(),
  "s": () => stopCamera(),
  "u": () => document.getElementById("upload-input").click(),
  "r": () => resetStats(),
  "t": () => toggleTTS(),
  "m": () => toggleMute(),
  "?": () => toggleHelp(),
};

// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────

const state = {
  cameraActive: false,
  ttsEnabled: true,
  muted: false,
  language: "en",
  socket: null,
  stream: null,
  frameTimer: null,
  statsTimer: null,
  lastSpokenTime: 0,
  spokenTexts: new Map(),    // label -> timestamp
  audioCtx: null,
  detectionLog: [],
  totalFrames: 0,
  useWebSocket: false,
};

// ─────────────────────────────────────────────────────────────
// DOM REFERENCES
// ─────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const DOM = {
  video:          () => $("camera-feed"),
  canvas:         () => $("capture-canvas"),
  resultImg:      () => $("result-image"),
  uploadInput:    () => $("upload-input"),
  startBtn:       () => $("btn-start"),
  stopBtn:        () => $("btn-stop"),
  uploadBtn:      () => $("btn-upload"),
  resetBtn:       () => $("btn-reset"),
  ttsToggle:      () => $("toggle-tts"),
  muteToggle:     () => $("toggle-mute"),
  langSelect:     () => $("lang-select"),
  statusDot:      () => $("status-dot"),
  statusText:     () => $("status-text"),
  dangerBanner:   () => $("danger-banner"),
  dangerMsg:      () => $("danger-message"),
  detectionList:  () => $("detection-list"),
  eventLog:       () => $("event-log"),
  statTotal:      () => $("stat-total"),
  statDanger:     () => $("stat-danger"),
  statFps:        () => $("stat-fps"),
  statUptime:     () => $("stat-uptime"),
  processingTime: () => $("processing-time"),
  helpPanel:      () => $("help-panel"),
};

// ─────────────────────────────────────────────────────────────
// WEBSOCKET
// ─────────────────────────────────────────────────────────────

function initWebSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${location.host}`;

  try {
    state.socket = io(url, {
      transports: ["websocket"],
      reconnectionDelay: CONFIG.RECONNECT_DELAY_MS,
    });

    state.socket.on("connect", () => {
      state.useWebSocket = true;
      setStatus("connected", "WebSocket connected");
      logEvent("WebSocket connection established", "info");
    });

    state.socket.on("disconnect", () => {
      state.useWebSocket = false;
      setStatus("idle", "WebSocket disconnected — using HTTP fallback");
      logEvent("WebSocket disconnected", "warning");
    });

    state.socket.on("connected", data => {
      logEvent(data.message, "info");
    });

    state.socket.on("detection_result", data => {
      handleDetectionResult(data);
    });

    state.socket.on("stats_update", data => {
      updateStatsDisplay(data);
    });

    state.socket.on("error", data => {
      logEvent(`Server error: ${data.message}`, "error");
    });

  } catch (e) {
    console.warn("WebSocket unavailable, falling back to HTTP:", e);
    state.useWebSocket = false;
  }
}

// ─────────────────────────────────────────────────────────────
// CAMERA
// ─────────────────────────────────────────────────────────────

async function startCamera() {
  if (state.cameraActive) return;

  try {
    setStatus("loading", "Requesting camera access…");

    const constraints = {
      video: {
        width:  { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "environment",   // Prefer rear camera on mobile
      }
    };

    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    const video = DOM.video();
    video.srcObject = state.stream;
    video.style.display = "block";

    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    video.play();

    state.cameraActive = true;
    DOM.startBtn().disabled = true;
    DOM.stopBtn().disabled  = false;
    setStatus("active", "Live detection running");
    logEvent("Camera started", "info");

    // Start frame capture loop
    state.frameTimer = setInterval(captureAndSend, CONFIG.FRAME_INTERVAL_MS);

  } catch (err) {
    setStatus("error", `Camera error: ${err.message}`);
    logEvent(`Camera error: ${err.message}`, "error");
  }
}

function stopCamera() {
  if (!state.cameraActive) return;

  clearInterval(state.frameTimer);
  state.frameTimer = null;

  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }

  DOM.video().style.display = "none";
  state.cameraActive = false;
  DOM.startBtn().disabled = false;
  DOM.stopBtn().disabled  = true;
  hideDanger();
  setStatus("idle", "Camera stopped");
  logEvent("Camera stopped", "info");
}

// ─────────────────────────────────────────────────────────────
// FRAME CAPTURE & SEND
// ─────────────────────────────────────────────────────────────

function captureAndSend() {
  const video  = DOM.video();
  const canvas = DOM.canvas();

  if (!video || video.readyState < 2) return;

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataURL = canvas.toDataURL("image/jpeg", CONFIG.JPEG_QUALITY);

  if (state.useWebSocket && state.socket?.connected) {
    state.socket.emit("frame", { image: dataURL });
  } else {
    sendFrameHTTP(dataURL);
  }

  state.totalFrames++;
}

async function sendFrameHTTP(dataURL) {
  try {
    const res = await fetch("/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL }),
    });

    if (!res.ok) return;
    const data = await res.json();
    if (data.success) handleDetectionResult(data);

  } catch (e) {
    // Silently fail on network errors during live detection
  }
}

// ─────────────────────────────────────────────────────────────
// IMAGE UPLOAD
// ─────────────────────────────────────────────────────────────

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  const allowedTypes = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
  if (!allowedTypes.includes(file.type)) {
    logEvent("Invalid file type. Please select an image.", "error");
    return;
  }

  if (file.size > 16 * 1024 * 1024) {
    logEvent("File too large (max 16MB).", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  setStatus("loading", `Analyzing ${file.name}…`);

  fetch("/upload", { method: "POST", body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) throw new Error(data.error);
      handleDetectionResult(data);
      setStatus("idle", "Image analysis complete");
    })
    .catch(err => {
      setStatus("error", `Upload failed: ${err.message}`);
      logEvent(`Upload error: ${err.message}`, "error");
    });
}

// ─────────────────────────────────────────────────────────────
// DETECTION RESULT HANDLER
// ─────────────────────────────────────────────────────────────

function handleDetectionResult(data) {
  // Update annotated image
  if (data.image) {
    const img = DOM.resultImg();
    img.src = data.image;
    img.style.display = "block";
  }

  // Update processing time
  if (data.processing_time_ms !== undefined) {
    const el = DOM.processingTime();
    if (el) el.textContent = `${data.processing_time_ms.toFixed(1)} ms`;
  }

  const detections = data.detections || [];

  // Update detection panel
  renderDetections(detections);

  // Check for danger
  const dangers = detections.filter(d => d.is_danger);
  if (dangers.length > 0) {
    const topDanger = dangers.sort((a, b) => (b.urgency || 0) - (a.urgency || 0))[0];
    showDanger(topDanger.sentence || `DANGER: ${topDanger.label}!`);
    playBeep();
  } else {
    hideDanger();
  }

  // Speak alerts
  if (state.ttsEnabled && !state.muted) {
    speakDetections(detections);
  }

  // Log event
  if (detections.length > 0) {
    const labels = detections.map(d => d.label).join(", ");
    addToLog(`Detected: ${labels}`, "detection");
  }
}

// ─────────────────────────────────────────────────────────────
// UI RENDERING
// ─────────────────────────────────────────────────────────────

function renderDetections(detections) {
  const list = DOM.detectionList();
  if (!list) return;

  if (detections.length === 0) {
    list.innerHTML = `<div class="empty-state">No objects detected</div>`;
    return;
  }

  // Sort by urgency desc
  const sorted = [...detections].sort((a, b) => (b.urgency || 0) - (a.urgency || 0));

  list.innerHTML = sorted.map(det => {
    const color = PRIORITY_COLORS[det.priority] || PRIORITY_COLORS.LOW;
    const pct   = Math.round((det.urgency || 0) * 100);
    const conf  = Math.round((det.confidence || 0) * 100);
    return `
      <div class="detection-item" style="--accent:${color}">
        <div class="det-header">
          <span class="det-label">${escapeHtml(det.label)}</span>
          <span class="det-priority" style="color:${color}">${det.priority || "LOW"}</span>
        </div>
        <div class="det-meta">
          <span>${det.distance || "—"}</span>
          <span>${det.direction || "—"}</span>
          <span>${conf}% conf</span>
        </div>
        <div class="urgency-bar">
          <div class="urgency-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <div class="det-sentence">${escapeHtml(det.sentence || "")}</div>
      </div>`;
  }).join("");
}

function showDanger(message) {
  const banner = DOM.dangerBanner();
  const msg    = DOM.dangerMsg();
  if (!banner) return;
  if (msg) msg.textContent = message;
  banner.classList.add("visible");
  banner.setAttribute("aria-live", "assertive");
}

function hideDanger() {
  const banner = DOM.dangerBanner();
  if (banner) banner.classList.remove("visible");
}

function setStatus(type, text) {
  const dot  = DOM.statusDot();
  const span = DOM.statusText();
  if (!dot || !span) return;
  dot.className  = `status-dot status-${type}`;
  span.textContent = text;
}

function logEvent(message, type = "info") {
  const log = DOM.eventLog();
  if (!log) return;

  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = `log-entry log-${type}`;
  entry.innerHTML = `<span class="log-time">${time}</span> ${escapeHtml(message)}`;

  log.prepend(entry);

  // Trim log
  while (log.children.length > CONFIG.MAX_LOG_ENTRIES) {
    log.removeChild(log.lastChild);
  }
}

function addToLog(message, type) {
  logEvent(message, type);
}

// ─────────────────────────────────────────────────────────────
// STATS
// ─────────────────────────────────────────────────────────────

async function fetchStats() {
  try {
    const res  = await fetch("/stats");
    const data = await res.json();
    updateStatsDisplay(data);

    // Request WebSocket stats update too
    if (state.socket?.connected) {
      state.socket.emit("request_stats");
    }
  } catch (e) {
    // Silently fail
  }
}

function updateStatsDisplay(data) {
  if (DOM.statTotal())  DOM.statTotal().textContent  = data.total_detections  ?? "—";
  if (DOM.statDanger()) DOM.statDanger().textContent = data.danger_alerts     ?? "—";
  if (DOM.statFps())    DOM.statFps().textContent    = (data.avg_fps ?? 0).toFixed(1);
  if (DOM.statUptime()) {
    const s = Math.round(data.uptime_seconds ?? 0);
    const m = Math.floor(s / 60);
    DOM.statUptime().textContent = `${m}m ${s % 60}s`;
  }
}

async function resetStats() {
  await fetch("/stats/reset", { method: "POST" });
  logEvent("Statistics reset", "info");
  updateStatsDisplay({ total_detections: 0, danger_alerts: 0, avg_fps: 0, uptime_seconds: 0 });
}

// ─────────────────────────────────────────────────────────────
// TEXT-TO-SPEECH (Browser)
// ─────────────────────────────────────────────────────────────

function speakDetections(detections) {
  if (!("speechSynthesis" in window)) return;

  const now = Date.now();
  if (now - state.lastSpokenTime < CONFIG.TTS_COOLDOWN_MS) return;

  // Find most urgent unseen detection
  const sorted = [...detections]
    .filter(d => d.sentence)
    .sort((a, b) => (b.urgency || 0) - (a.urgency || 0));

  for (const det of sorted) {
    const key      = det.label;
    const lastSpoken = state.spokenTexts.get(key) || 0;
    const cooldown = det.priority === "CRITICAL" ? 2000
                   : det.priority === "HIGH"     ? 4000
                   : det.priority === "MEDIUM"   ? 8000
                   : 12000;

    if (now - lastSpoken > cooldown) {
      speakText(det.sentence, det.priority);
      state.spokenTexts.set(key, now);
      state.lastSpokenTime = now;
      break;
    }
  }
}

function speakText(text, priority = "LOW") {
  if (!("speechSynthesis" in window) || !text) return;

  const utt  = new SpeechSynthesisUtterance(text);
  utt.lang   = state.language;
  utt.rate   = priority === "CRITICAL" ? 1.3 : priority === "HIGH" ? 1.1 : 1.0;
  utt.volume = 1.0;

  // Cancel urgent speech for CRITICAL
  if (priority === "CRITICAL") {
    window.speechSynthesis.cancel();
  }

  window.speechSynthesis.speak(utt);
}

function toggleTTS() {
  state.ttsEnabled = !state.ttsEnabled;
  const btn = DOM.ttsToggle();
  if (btn) {
    btn.classList.toggle("active", state.ttsEnabled);
    btn.setAttribute("aria-pressed", state.ttsEnabled);
    btn.title = state.ttsEnabled ? "TTS On" : "TTS Off";
  }
  logEvent(`Text-to-speech ${state.ttsEnabled ? "enabled" : "disabled"}`, "info");
}

function toggleMute() {
  state.muted = !state.muted;
  const btn = DOM.muteToggle();
  if (btn) {
    btn.classList.toggle("active", state.muted);
    btn.setAttribute("aria-pressed", state.muted);
  }
  if (state.muted) window.speechSynthesis?.cancel();
}

// ─────────────────────────────────────────────────────────────
// AUDIO BEEP (Web Audio API)
// ─────────────────────────────────────────────────────────────

function playBeep() {
  if (state.muted) return;

  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    const ctx = state.audioCtx;
    const oscillator = ctx.createOscillator();
    const gainNode   = ctx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.type      = "square";
    oscillator.frequency.setValueAtTime(CONFIG.DANGER_BEEP_FREQ, ctx.currentTime);
    gainNode.gain.setValueAtTime(0.15, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + CONFIG.DANGER_BEEP_DURATION);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + CONFIG.DANGER_BEEP_DURATION);
  } catch (e) {
    // Audio not available
  }
}

// ─────────────────────────────────────────────────────────────
// LANGUAGE
// ─────────────────────────────────────────────────────────────

function setLanguage(lang) {
  state.language = lang;
  logEvent(`Language set to: ${lang}`, "info");
}

// ─────────────────────────────────────────────────────────────
// HELP PANEL
// ─────────────────────────────────────────────────────────────

function toggleHelp() {
  const panel = DOM.helpPanel();
  if (!panel) return;
  const visible = panel.classList.toggle("visible");
  panel.setAttribute("aria-hidden", !visible);
}

// ─────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ─────────────────────────────────────────────────────────────
// KEYBOARD SHORTCUTS
// ─────────────────────────────────────────────────────────────

document.addEventListener("keydown", e => {
  // Skip if typing in input
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;

  const handler = KEYBOARD_SHORTCUTS[e.key.toLowerCase()];
  if (handler) {
    e.preventDefault();
    handler();
  }
});

// ─────────────────────────────────────────────────────────────
// PWA SERVICE WORKER
// ─────────────────────────────────────────────────────────────

function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/static/sw.js")
      .then(() => console.log("Service Worker registered"))
      .catch(err => console.warn("SW registration failed:", err));
  }
}

// ─────────────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch("/health");
    const data = await res.json();
    if (data.status === "ok") {
      setStatus("idle", `Models loaded: ${data.models_loaded}`);
    }
  } catch (e) {
    setStatus("error", "Cannot reach server");
  }
}

// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────

function init() {
  // Wire up buttons
  DOM.startBtn()?.addEventListener("click", startCamera);
  DOM.stopBtn()?.addEventListener("click", stopCamera);
  DOM.uploadBtn()?.addEventListener("click", () => DOM.uploadInput()?.click());
  DOM.uploadInput()?.addEventListener("change", handleFileSelect);
  DOM.resetBtn()?.addEventListener("click", resetStats);
  DOM.ttsToggle()?.addEventListener("click", toggleTTS);
  DOM.muteToggle()?.addEventListener("click", toggleMute);
  DOM.langSelect()?.addEventListener("change", e => setLanguage(e.target.value));

  // Stop btn disabled initially
  if (DOM.stopBtn()) DOM.stopBtn().disabled = true;

  // Init WebSocket
  if (typeof io !== "undefined") {
    initWebSocket();
  } else {
    console.warn("Socket.IO not loaded — using HTTP polling");
  }

  // Initial health check
  checkHealth();

  // Start stats polling
  state.statsTimer = setInterval(fetchStats, CONFIG.STATS_REFRESH_MS);

  // PWA
  registerServiceWorker();

  logEvent("VocalEyes v3.0 initialized", "info");
}

document.addEventListener("DOMContentLoaded", init);