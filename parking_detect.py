import cv2
import json
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient

# ----------------- MongoDB Setup -----------------
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["vehicle_system"]
slots_collection = db["slots"]

# ----------------- Load Slots from slots.json -----------------
with open("slots.json", "r") as f:
    slots_data = json.load(f)


parking_slots = []
for idx, (veh_type, points) in enumerate(slots_data.items(), start=1):
    parking_slots.append({
        "slotNumber": idx,
        "vehicleType": veh_type.lower(),
        "points": points
    })

# ----------------- Track Slot States -----------------
slot_states = {slot["slotNumber"]: {"occupied": False, "allocatedTo": None} for slot in parking_slots}

# ----------------- Helper: Update Slot -----------------
def update_slot(slot_number, occupied, allocated_to=None):
    """Update MongoDB only if state changes."""
    prev_state = slot_states[slot_number]
    if prev_state["occupied"] == occupied and prev_state["allocatedTo"] == allocated_to:
        return  # no change → skip DB update
    # Update DB
    slots_collection.update_one(
        {"slotNumber": slot_number},
        {"$set": {"occupied": occupied, "allocatedTo": allocated_to}}
    )
    # Update local state cache
    slot_states[slot_number] = {"occupied": occupied, "allocatedTo": allocated_to}
    if occupied:
        print(f"✅ DB Updated -> Slot {slot_number} OCCUPIED by {allocated_to}")
    else:
        print(f"✅ DB Updated -> Slot {slot_number} EMPTY")

# ----------------- YOLO Setup -----------------
model = YOLO("yolov8n.pt")

# YOLO class → vehicle type mapping
yolo_to_type = {
    2: "car",
    3: "bike",
    5: "bus",
    7: "suv"  # replaced truck with SUV
}

# ----------------- Vehicle Hierarchy -----------------
# Lower number = smaller vehicle
vehicle_hierarchy = {
    "bike": 1,
    "car": 2,
    "suv": 3,
    "bus": 4
}

# ----------------- Slot Check Function -----------------
def is_car_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    return cv2.pointPolygonTest(np.array(slot_points, np.int32), car_center, False) >= 0

# ----------------- Detection Loop -----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera not found or not working")
        break

    # Run YOLO detection
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    cars = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls_id = int(cls)
        if cls_id in yolo_to_type.keys():
            cars.append((int(x1), int(y1), int(x2), int(y2), cls_id))

    # Check each slot
    for slot in parking_slots:
        occupied = False
        allocated_to = None
        slot_type = slot["vehicleType"]

        for (x1, y1, x2, y2, cls_id) in cars:
            det_vehicle_type = yolo_to_type.get(cls_id, "car")

            # Check if vehicle is physically in slot
            if is_car_in_slot((x1, y1, x2, y2), slot["points"]):
                # Check hierarchy: only allow if vehicle fits slot level
                if vehicle_hierarchy[det_vehicle_type] <= vehicle_hierarchy[slot_type]:
                    occupied = True
                    allocated_to = det_vehicle_type
                    break  # one vehicle per slot

        # Update DB only if slot state changes
        update_slot(slot["slotNumber"], occupied, allocated_to)

        # Draw slot
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.polylines(frame, [np.array(slot["points"], np.int32)], True, color, 2)
        status_text = f"Slot {slot['slotNumber']} ({slot['vehicleType']}): {'Occupied' if occupied else 'Empty'}"
        cv2.putText(frame, status_text, (slot["points"][0][0], slot["points"][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow("Parking Slot Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
