"""
import cv2
import json
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load parking slot coordinates
with open("slots.json", "r") as f:
    data = json.load(f)

# Handle dict {"Slot1": [...]} or list [[...]]
if isinstance(data, dict):
    parking_slots = list(data.values())
else:
    parking_slots = data

# Function to check if car is inside slot
def is_car_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return cv2.pointPolygonTest(np.array(slot_points, np.int32), car_center, False) >= 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera not found or not working")
        break

    # Run YOLO object detection
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    # Collect detected cars
    cars = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in [2, 3, 5, 7]:  # Car=2, Motorcycle=3, Bus=5, Truck=7
            cars.append((int(x1), int(y1), int(x2), int(y2)))

    # Check each parking slot
    for idx, slot in enumerate(parking_slots):
        occupied = any(is_car_in_slot(car, slot) for car in cars)
        color = (0, 0, 255) if occupied else (0, 255, 0)

        # Draw slot boundaries
        cv2.polylines(frame, [np.array(slot, np.int32)], True, color, 2)

        # Put status text
        status_text = f"Slot {idx+1}: {'Occupied' if occupied else 'Empty'}"
        cv2.putText(
            frame,
            status_text,
            (slot[0][0], slot[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Show result
    cv2.imshow("Parking Slot Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load parking slot coordinates
with open("slots.json", "r") as f:
    data = json.load(f)

# Handle dict {"Slot1": [...]} or list [[...]]
if isinstance(data, dict):
    parking_slots = list(data.values())
else:
    parking_slots = data

# Function to check if car is inside slot
def is_car_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return cv2.pointPolygonTest(np.array(slot_points, np.int32), car_center, False) >= 0

# Load input image
image_path = "image.png"   # 👈 replace with your image filename
frame = cv2.imread(image_path)

if frame is None:
    print(f"⚠️ Could not load image: {image_path}")
    exit()

# Run YOLO object detection
results = model(frame)
detections = results[0].boxes.data.cpu().numpy()

# Collect detected cars
cars = []
for det in detections:
    x1, y1, x2, y2, conf, cls = det
    if int(cls) in [2, 3, 5, 7]:  # Car=2, Motorcycle=3, Bus=5, Truck=7
        cars.append((int(x1), int(y1), int(x2), int(y2)))

# Check each parking slot
for idx, slot in enumerate(parking_slots):
    occupied = any(is_car_in_slot(car, slot) for car in cars)
    color = (0, 0, 255) if occupied else (0, 255, 0)

    # Draw slot boundaries
    cv2.polylines(frame, [np.array(slot, np.int32)], True, color, 2)

    # Put status text
    status_text = f"Slot {idx+1}: {'Occupied' if occupied else 'Empty'}"
    cv2.putText(
        frame,
        status_text,
        (slot[0][0], slot[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

# Show result
cv2.imshow("Parking Slot Detection (Image)", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

