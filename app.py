from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import os
import base64
import numpy as np
import time
import torch
import json
from ocr_module import read_text
import threading
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ─────────────────────────────
# MODEL LOADING
# ─────────────────────────────
try:
    coco_model = YOLO("yolov8n.pt")
    logger.info("COCO model loaded")
except Exception as e:
    logger.error(f"COCO model error: {e}")
    coco_model = None

try:
    MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/train26/weights/best.pt")
    custom_model = YOLO(MODEL_PATH)
    logger.info("Custom model loaded")
except Exception as e:
    logger.warning(f"Custom model not found, using yolov8n fallback: {e}")
    custom_model = YOLO("yolov8n.pt")

CONF_THRESHOLD = 0.4

device = "cuda" if torch.cuda.is_available() else "cpu"

if coco_model:
    coco_model.to(device)

if custom_model:
    custom_model.to(device)

print(f"[INFO] Using device: {device}")

tracker = DeepSort(max_age=30)
# ─────────────────────────────
# PRIORITY SYSTEM
# ─────────────────────────────
PRIORITY_MAP = {
    "HIGH": ["car", "stairs", "truck", "bus", "motorcycle", "bicycle", "fire hydrant", "stop sign"],
    "MEDIUM": ["person", "doors", "dog", "cat"],
    "LOW": ["chair", "table", "couch", "bottle", "text-sign"],
}

def get_priority(label):
    for level, items in PRIORITY_MAP.items():
        if label in items:
            return level
    return "LOW"

# ─────────────────────────────
# DISTANCE ESTIMATION
# ─────────────────────────────
def estimate_distance(box_width, frame_width=640):
    ratio = box_width / frame_width
    if ratio > 0.55:
        return "very near"
    elif ratio > 0.30:
        return "near"
    elif ratio > 0.15:
        return "medium"
    else:
        return "far"

# ─────────────────────────────
# DIRECTION
# ─────────────────────────────
def estimate_direction(x_center, frame_width):
    third = frame_width / 3
    if x_center < third:
        return "left"
    elif x_center > 2 * third:
        return "right"
    else:
        return "ahead"

# ─────────────────────────────
# SENTENCE GENERATOR
# ─────────────────────────────
def generate_sentence(label, distance, direction, priority):
    if priority == "HIGH":
        if distance == "very near":
            return f"DANGER! {label} very close on your {direction}!"
        elif distance == "near":
            return f"Warning! {label} nearby on your {direction}."
        elif distance == "medium":
            return f"{label.capitalize()} approaching from the {direction}."
        else:
            return f"{label.capitalize()} detected on your {direction}."
    elif priority == "MEDIUM":
        if distance in ("very near", "near"):
            return f"Watch out — {label} on your {direction}."
        else:
            return f"{label.capitalize()} spotted {direction}."
    else:
        return f"{label.capitalize()} {distance} on your {direction}."

# ─────────────────────────────
# STATS TRACKER (in-memory)
# ─────────────────────────────
detection_history = deque(maxlen=100)
session_stats = {
    "total_detections": 0,
    "danger_alerts": 0,
    "session_start": time.time(),
    "top_objects": {}
}
stats_lock = threading.Lock()

# ─────────────────────────────
# SPEECH COOLDOWN MEMORY
# ─────────────────────────────
last_spoken = {}

def update_stats(results_list):
    with stats_lock:
        session_stats["total_detections"] += len(results_list)

        for r in results_list:
            label = r.get("label", "unknown")
            session_stats["top_objects"][label] = session_stats["top_objects"].get(label, 0) + 1

            if r.get("priority") == "HIGH" and r.get("distance") in ("very near", "near"):
                session_stats["danger_alerts"] += 1

        detection_history.append({
            "time": time.time(),
            "count": len(results_list),
            "objects": [r.get("label") for r in results_list]
        })

# ─────────────────────────────
# GLOBAL MEMORY
# ─────────────────────────────
object_memory = {}
last_depth_time = 0
cached_depth = None
frame_skip = 2
frame_count = 0
last_fps = 0
object_velocity = {}
velocity_smooth = {}
# ─────────────────────────────
# DEPTH MODEL (MiDaS)
# ─────────────────────────────
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    logger.info("MiDaS depth model loaded")
except Exception as e:
    logger.error(f"Depth model error: {e}")
    midas = None
    transform = None
# ─────────────────────────────
# DEPTH FUNCTION (MiDaS)
# ─────────────────────────────
def get_depth_map(frame):
    if transform is None or midas is None:
        return np.zeros((frame.shape[0], frame.shape[1]))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter

    return inter/union if union else 0

# ─────────────────────────────
# CORE DETECTION (FINAL VERSION)
# ─────────────────────────────
def process_frame(frame):

    start_time = time.time()

    global frame_count, frame_skip, last_fps 
    frame_count += 1

    if frame_count % frame_skip != 0:
        return frame, None

    global last_depth_time, cached_depth

    frame = cv2.resize(frame, (256, 256))
    h, w = frame.shape[:2]

    results_list = []
    detected_boxes = []
    detected_labels = set()

    annotated = frame.copy()

    # ───── Depth caching ─────
    if time.time() - last_depth_time > 1.0:
        cached_depth = get_depth_map(frame)
        last_depth_time = time.time()

    depth_map = cached_depth if cached_depth is not None else np.zeros((h, w))

    # ───── Duplicate filter ─────
    def is_duplicate(b1, b2, thr=0.5):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])

        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        union = area1 + area2 - inter

        return (inter/union) > thr if union else False

    # ───── YOLO detection ─────
    model_pairs = []
    if coco_model:
        model_pairs.append(coco_model(frame, conf=CONF_THRESHOLD, imgsz=256, device=device, verbose=False)[0])
    model_pairs.append(custom_model(frame, conf=CONF_THRESHOLD, verbose=False)[0])

    for results in model_pairs:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls = int(box.cls[0])
            label = results.names[cls]

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            bbox = [x1, y1, x2, y2]

            if any(is_duplicate(bbox, prev) for prev in detected_boxes):
                continue

            detected_boxes.append(bbox)
            detected_labels.add(label)

            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))
            depth_value = depth_map[cy, cx]

            # Distance
            if depth_value < 50:
                distance = "very near"
            elif depth_value < 150:
                distance = "near"
            elif depth_value < 300:
                distance = "medium"
            else:
                distance = "far"

            direction = estimate_direction(cx, w)
            priority = get_priority(label)

            results_list.append({
                "label": label,
                "distance": distance,
                "direction": direction,
                "priority": priority,
                "confidence": round(conf, 2),
                "sentence": generate_sentence(label, distance, direction, priority),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "cx": cx,
                "cy": cy,
                "depth": depth_value
            })

        annotated = results.plot(img=annotated)

    # ───── TRACKING ─────
    detections_for_tracker = []
    for r in results_list:
        x1, y1, x2, y2 = r["bbox"]
        detections_for_tracker.append(([x1, y1, x2-x1, y2-y1], r["confidence"], r["label"]))

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        best_iou = 0
        matched = None

        # ✅ FIXED LOOP
        for rdata in results_list:
            bx1, by1, bx2, by2 = rdata["bbox"]
            score = iou([bx1, by1, bx2, by2], [l, t, r, b])

            if score > best_iou:
                best_iou = score
                matched = rdata

        if best_iou < 0.3 or matched is None:
            continue

        label = matched["label"]
        depth_value = matched["depth"]
        direction = matched["direction"]

        obj_key = f"{label}_{track_id}"
        collision = False

        prev = object_memory.get(obj_key)

        if prev:
            prev_depth, prev_time = prev
            dt = time.time() - prev_time

            if dt > 0:
                raw_velocity = (prev_depth - depth_value) / dt

                prev_v = velocity_smooth.get(obj_key, raw_velocity)
                velocity = 0.7 * prev_v + 0.3 * raw_velocity

                velocity_smooth[obj_key] = velocity
                object_velocity[obj_key] = velocity

                if velocity > 20 and depth_value < 120 and direction == "ahead":
                    collision = True

                if velocity > 0:
                    ttc = depth_value / velocity
                else:
                    ttc = float('inf')

                if ttc < 2:
                    collision = True

        object_memory[obj_key] = (depth_value, time.time())

        if collision:
            matched["sentence"] = f"⚠ Collision risk! {label} approaching ahead!"

        cv2.putText(annotated, f'ID {track_id}', (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # ───── NAVIGATION ─────
    left = right = center = False

    for r in results_list:
        if r["priority"] == "HIGH" and r["distance"] in ["near", "very near"]:
            if r["direction"] == "left":
                left = True
            elif r["direction"] == "right":
                right = True
            else:
                center = True

    if center:
        if not left:
            nav = "Move left"
        elif not right:
            nav = "Move right"
        else:
            nav = "Stop — path blocked"
    else:
        nav = "Path clear ahead"

    results_list.append({
        "label": "navigation",
        "distance": "",
        "direction": "",
        "priority": "LOW",
        "confidence": 1.0,
        "sentence": nav,
        "bbox": []
    })

    # ───── OCR ─────
    if "text-sign" in detected_labels:
        try:
            text = read_text(frame)
            if text:
                results_list.append({
                    "label": "text",
                    "distance": "",
                    "direction": "",
                    "priority": "LOW",
                    "confidence": 1.0,
                    "sentence": f"Sign reads: {text}",
                    "bbox": []
                })
        except:
            pass

    update_stats(results_list)

    # ───── Adaptive FPS ─────

    fps = 1 / (time.time() - start_time + 1e-6)
    last_fps = fps

    if fps < 10:
        frame_skip = 3
    elif fps < 20:
        frame_skip = 2
    else:
        frame_skip = 1

    # ───── Memory cleanup ─────
    if len(object_memory) > 100:
        keys = list(object_memory.keys())[:50]
        for k in keys:
            object_memory.pop(k, None)
            object_velocity.pop(k, None)
            velocity_smooth.pop(k, None)

    return annotated, results_list
# ─────────────────────────────
# ROUTES
# ─────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": coco_model is not None})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    path = os.path.join("uploads", file.filename)
    file.save(path)

    image = cv2.imread(path)
    if image is None:
        return jsonify({"error": "Could not read image"}), 400

    frame, results = process_frame(image)

    if results is None:
        results = []

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": f"data:image/jpeg;base64,{encoded}",
        "results": results,
        "count": len(results)
    })

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image data"}), 400

    try:
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Image decode error: {str(e)}"}), 400

    processed, results = process_frame(frame)

    if results is None:
        results = []

    _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": f"data:image/jpeg;base64,{encoded}",
        "results": results,
        "count": len(results)
    })

@app.route("/stats")
def stats():
    with stats_lock:
        uptime = round(time.time() - session_stats["session_start"])
        top = sorted(session_stats["top_objects"].items(), key=lambda x: x[1], reverse=True)[:5]
        return jsonify({
            "total_detections": session_stats["total_detections"],
            "danger_alerts": session_stats["danger_alerts"],
            "uptime_seconds": uptime,
            "top_objects": top,
            "recent_history": list(detection_history)[-10:]
        })

@app.route("/reset_stats", methods=["POST"])
def reset_stats():
    with stats_lock:
        session_stats["total_detections"] = 0
        session_stats["danger_alerts"] = 0
        session_stats["session_start"] = time.time()
        session_stats["top_objects"] = {}
        detection_history.clear()
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)