# 🔧 How to Use Maintenance Status - Quick Guide

## What It Does
Slots marked as "Under Maintenance" are automatically **excluded** from vehicle assignment. The system won't assign vehicles to these slots until you mark them as available again.

---

## ⚡ Quick Commands (Easiest Way)

### Start Server
```powershell
cd "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard"
python manage.py runserver
```

### Check Status
```powershell
python manage.py maintenance status
```
Shows: Total, Available, Occupied, Maintenance counts

### Mark Slots for Maintenance

**Single slot:**
```powershell
python manage.py maintenance mark --slots TW1
```

**Multiple slots:**
```powershell
python manage.py maintenance mark --slots TW1 TW2 C5
```

**All two-wheeler slots:**
```powershell
python manage.py maintenance mark --type two_wheeler
```

### Return Slots to Service

**Single slot:**
```powershell
python manage.py maintenance unmark --slots TW1
```

**Multiple slots:**
```powershell
python manage.py maintenance unmark --slots TW1 TW2 C5
```

### List All Maintenance Slots
```powershell
python manage.py maintenance list
```

---

## 🌐 Via API (For Integration with Your Interface)

### Mark Maintenance
```powershell
# Using curl/PowerShell
curl -X POST http://127.0.0.1:8000/api/toggle-maintenance/ `
  -H "Content-Type: application/json" `
  -d '{\"slot_numbers\": [\"TW1\", \"C5\"], \"action\": \"maintenance\"}'
```

### Return to Service
```powershell
curl -X POST http://127.0.0.1:8000/api/toggle-maintenance/ `
  -H "Content-Type: application/json" `
  -d '{\"slot_numbers\": [\"TW1\"], \"action\": \"available\"}'
```

### Check Current Maintenance
```powershell
curl http://127.0.0.1:8000/api/toggle-maintenance/
```

---

## 📊 What You'll See in Dashboard

Your main dashboard now shows:
- **Available** (green) - Ready for vehicles
- **Occupied** (red) - Has a vehicle
- **Maintenance** (yellow/orange) - Under repair ← NEW!
- **Out of Service** (gray) - Permanently closed

The maintenance count appears in your dashboard summary.

---

## 🎯 Status Options

| Status | Can Assign Vehicles? | Use Case |
|--------|---------------------|----------|
| **available** | ✅ Yes | Normal operation |
| **occupied** | ❌ No (has vehicle) | Auto-set by system |
| **maintenance** | ❌ No | Daily/weekly cleaning, repairs |
| **out_of_service** | ❌ No | Permanent closure, damage |

---

## 💡 Real-World Examples

### Daily Cleaning (Morning)
```powershell
# Mark slots for cleaning
python manage.py maintenance mark --slots TW1 TW2 TW3

# System shows: "3 slots under maintenance"
# These slots won't be assigned to vehicles

# After cleaning
python manage.py maintenance unmark --slots TW1 TW2 TW3
```

### Emergency Repair
```powershell
# Slot C5 has damage
python manage.py maintenance mark --slots C5

# Check status
python manage.py maintenance list
# Shows: C5    Cars    Floor 1

# After repair
python manage.py maintenance unmark --slots C5
```

### Weekly Maintenance (All Two-Wheelers)
```powershell
# Friday evening - mark all TW slots
python manage.py maintenance mark --type two_wheeler

# Monday morning - return to service
python manage.py maintenance unmark --type two_wheeler
```

---

## 🔄 How It Works Automatically

1. **Vehicle tries to park** → System searches for available slots
2. **Maintenance slots are skipped** → Only truly available slots considered
3. **Vehicle gets assigned** → Next available non-maintenance slot
4. **Dashboard updates** → Shows current maintenance count

---

## 📱 Integration Example (JavaScript/Frontend)

```javascript
// Mark slot for maintenance
async function markMaintenance(slotNumber) {
    const response = await fetch('/api/toggle-maintenance/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            slot_numbers: [slotNumber],
            action: 'maintenance'
        })
    });
    const data = await response.json();
    console.log(data.message); // "Updated 1 slot(s) to maintenance"
}

// Return to service
async function returnToService(slotNumber) {
    const response = await fetch('/api/toggle-maintenance/', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            slot_numbers: [slotNumber],
            action: 'available'
        })
    });
    const data = await response.json();
    console.log(data.message);
}

// Get current maintenance list
async function getMaintenanceList() {
    const response = await fetch('/api/toggle-maintenance/');
    const data = await response.json();
    console.log('Maintenance slots:', data.maintenance_slots);
}
```

---

## ⚠️ Important Notes

1. **Cannot mark occupied slots** - System prevents this automatically
2. **Status persists** - Slots stay in maintenance until you change them
3. **Dashboard auto-updates** - Real-time counts include maintenance
4. **Backward compatible** - Old `is_occupied` field still works

---

## 🛠️ Troubleshooting

**Slot still being assigned?**
```powershell
# Check if it's really in maintenance
python manage.py maintenance list

# If not listed, mark it again
python manage.py maintenance mark --slots C5
```

**Can't mark slot?**
- Occupied slots cannot be marked (vehicle must exit first)
- Check slot number is correct

**Need to reset all?**
```powershell
# Return all maintenance slots to service
python manage.py shell -c "from parking_app.models import ParkingSlot; ParkingSlot.objects.filter(status='maintenance').update(status='available'); print('Done')"
```

---

## 📝 Command Reference

```powershell
# Status commands
python manage.py maintenance status      # Overview
python manage.py maintenance list        # List maintenance slots

# Mark commands
python manage.py maintenance mark --slots TW1 C5     # Specific slots
python manage.py maintenance mark --type two_wheeler # By type

# Unmark commands
python manage.py maintenance unmark --slots TW1      # Specific slots
python manage.py maintenance unmark --type car       # By type
```

---

**Ready to use! 🎉**

Test it:
```powershell
python manage.py maintenance status
python manage.py maintenance mark --slots TW1
python manage.py maintenance list
python manage.py maintenance unmark --slots TW1
```
