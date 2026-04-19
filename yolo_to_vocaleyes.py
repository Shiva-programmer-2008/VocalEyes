import cv2
import time
from ultralytics import YOLO

from ocr_module import read_text
from context_logic import get_priority
from sentence_generator import generate_sentence
from speech import speak

# ----------------------------
# INITIALIZATION
# ----------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_object_time = 0

print("VocalEyes Live Detection Started")
speak("Vocal Eyes live object detection started")

ret, frame = cap.read()
frame_width = frame.shape[1]

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    danger_detected = False  # 🔥 reset every frame

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]
            box_width = x2 - x1

            center_x = (x1 + x2) / 2
            if center_x < frame_width / 3:
                direction = "left"
            elif center_x > 2 * frame_width / 3:
                direction = "right"
            else:
                direction = "center"

            if box_width > 300:
                distance = "near"
            elif box_width > 150:
                distance = "medium"
            else:
                distance = "far"

            obj = {
                "label": label,
                "direction": direction,
                "distance": distance
            }

            priority = get_priority(label)
            if priority == "HIGH":
                danger_detected = True

            sentence = generate_sentence(obj, priority)

            if sentence:
                print(sentence)
                speak(sentence)
                last_object_time = time.time()

        frame = r.plot()

    # ---------- OCR TEXT READING ----------
    if not danger_detected and time.time() - last_object_time > 2:
        text = read_text(frame)
        if text:
            print("Text detected:", text)
            speak(f"Reading text. {text}")

    cv2.imshow("VocalEyes - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
speak("Live detection stopped")