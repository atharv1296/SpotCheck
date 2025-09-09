import json
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

# ----------------- Initialize Database -----------------
def init_slots():
    # Always reset collection
    slots_collection.delete_many({})
    print("Old slots cleared")

    docs = []
    for slot in parking_slots:
        docs.append({
            "slotNumber": slot["slotNumber"],
            "vehicleType": slot["vehicleType"],
            "allocatedTo": None,
            "occupied": False
        })
    slots_collection.insert_many(docs)
    print("✅ Fresh slots initialized in MongoDB")

if __name__ == "__main__":
    init_slots()
