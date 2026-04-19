import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("YOLO Live Detection Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        frame = r.plot()  # draw bounding boxes

    cv2.imshow("VocalEyes - YOLO Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()