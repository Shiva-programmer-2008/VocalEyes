"""
main.py — VocalEyes standalone desktop runner
Supports webcam mode and image upload mode.
"""

import cv2
import time
import sys
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import pyttsx3

from context_logic import get_priority, is_danger
from sentence_generator import generate_sentence, reset_state
from ocr_module import read_text, reset as ocr_reset

# ─────────────────────────────
# TTS ENGINE
# ─────────────────────────────
engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

def speak(text: str):
    print(f"[SPEAK] {text}")
    engine.say(text)
    engine.runAndWait()

# ─────────────────────────────
# MODELS
# ─────────────────────────────
print("Loading models...")
coco_model = YOLO("yolov8s.pt")

try:
    custom_model = YOLO("runs/detect/train26/weights/best.pt")
    print("Custom model loaded.")
except Exception:
    print("Custom model not found, using yolov8n fallback.")
    custom_model = YOLO("yolov8n.pt")

CONF = 0.45

# ─────────────────────────────
# DISTANCE / DIRECTION
# ─────────────────────────────
def estimate_distance(box_width: float, frame_width: int) -> str:
    ratio = box_width / frame_width
    if ratio > 0.55:
        return "very near"
    elif ratio > 0.30:
        return "near"
    elif ratio > 0.15:
        return "medium"
    return "far"

def estimate_direction(x_center: float, frame_width: int) -> str:
    third = frame_width / 3
    if x_center < third:
        return "left"
    elif x_center > 2 * third:
        return "right"
    return "ahead"

# ─────────────────────────────
# FRAME PROCESSOR
# ─────────────────────────────
def process_frame(frame, do_speak=False, last_spoken_time=[0.0], do_ocr=False, last_object_time=[0.0]):
    h, w = frame.shape[:2]
    detected_labels = set()
    sentences = []
    danger = False

    coco_res = coco_model(frame, conf=CONF, verbose=False)[0]
    custom_res = custom_model(frame, conf=CONF, verbose=False)[0]

    for results in [coco_res, custom_res]:
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF:
                continue

            cls = int(box.cls[0])
            label = results.names[cls]

            if label in detected_labels:
                continue
            detected_labels.add(label)

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            box_width = x2 - x1
            x_center = (x1 + x2) / 2

            distance = estimate_distance(box_width, w)
            direction = estimate_direction(x_center, w)
            priority = get_priority(label)

            if is_danger(label, distance):
                danger = True

            obj = {"label": label, "direction": direction, "distance": distance}
            sentence = generate_sentence(obj, priority)

            if sentence:
                sentences.append(sentence)
                if do_speak and (time.time() - last_spoken_time[0] > 1.5):
                    speak(sentence)
                    last_spoken_time[0] = time.time()
                    last_object_time[0] = time.time()

    # OCR pass when no danger
    if do_ocr and not danger and time.time() - last_object_time[0] > 2:
        text = read_text(frame)
        if text:
            msg = f"Sign reads: {text}"
            sentences.append(msg)
            if do_speak:
                speak(msg)

    frame = coco_res.plot()
    frame = custom_res.plot(img=frame)

    return frame, sentences

# ─────────────────────────────
# FILE PICKER
# ─────────────────────────────
def choose_image() -> str | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()
    return path or None

# ─────────────────────────────
# MAIN
# ─────────────────────────────
print("\n╔══════════════════════╗")
print("║  VocalEyes Desktop   ║")
print("╠══════════════════════╣")
print("║  1 → Webcam mode     ║")
print("║  2 → Image upload    ║")
print("╚══════════════════════╝")

mode = input("\nSelect mode [1/2]: ").strip()

if mode == "2":
    # ── IMAGE MODE ──────────────────────────────
    path = choose_image()
    if not path:
        print("No file selected. Exiting.")
        sys.exit(0)

    image = cv2.imread(path)
    if image is None:
        print("Failed to load image.")
        sys.exit(1)

    reset_state()
    frame, results = process_frame(image, do_speak=True, do_ocr=True)

    print("\n[Detected Objects]")
    for r in results:
        print(" •", r)

    cv2.imshow("VocalEyes — Image Mode", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # ── WEBCAM MODE ──────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    reset_state()
    ocr_reset()

    speak("VocalEyes live detection started.")
    print("\n[Live Mode] Press Q to quit.\n")

    last_spoken_time = [0.0]
    last_object_time = [0.0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed, _ = process_frame(
            frame,
            do_speak=True,
            last_spoken_time=last_spoken_time,
            do_ocr=True,
            last_object_time=last_object_time
        )

        cv2.imshow("VocalEyes — Live", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Detection stopped.")
