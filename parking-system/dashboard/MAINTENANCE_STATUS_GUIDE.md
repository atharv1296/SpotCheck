# 🔧 Maintenance Status Feature - User Guide

## Quick Start

The parking system now supports marking slots as "Under Maintenance" so they won't be assigned to vehicles automatically.

---

## 📍 Method 1: Using API Endpoint (Recommended for Interface Integration)

### Step 1: Start the Server
```powershell
cd "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard"
python manage.py runserver
```

### Step 2: Mark Slots for Maintenance via API

**Single Slot:**
```powershell
curl -X POST http://127.0.0.1:8000/api/toggle-maintenance/ `
  -H "Content-Type: application/json" `
  -d '{"slot_numbers": ["TW1"], "action": "maintenance"}'
```

**Multiple Slots:**
```powershell
curl -X POST http://127.0.0.1:8000/api/toggle-maintenance/ `
  -H "Content-Type: application/json" `
  -d '{"slot_numbers": ["TW1", "TW2", "C5"], "action": "maintenance"}'
```

**Return to Service:**
```powershell
curl -X POST http://127.0.0.1:8000/api/toggle-maintenance/ `
  -H "Content-Type: application/json" `
  -d '{"slot_numbers": ["TW1"], "action": "available"}'
```

**Check Current Maintenance Status:**
```powershell
curl http://127.0.0.1:8000/api/toggle-maintenance/
```

### API Response Example:
```json
{
  "success": true,
  "updated": 2,
  "failed": 0,
  "slots": [
    {
      "slot_number": "TW1",
      "old_status": "Available",
      "new_status": "Under Maintenance",
      "slot_type": "Two Wheeler"
    }
  ],
  "message": "Updated 2 slot(s) to maintenance"
}
```

---

## 📍 Method 2: Using Django Shell

```powershell
cd "C:\Users\athar\OneDrive\Desktop\TY - Sem 1\EDI\parking-system\dashboard"
python manage.py shell
```

Then run:

```python
from parking_app.models import ParkingSlot

# Mark slot TW1 as under maintenance
slot = ParkingSlot.objects.get(slot_number='TW1')
slot.status = 'maintenance'
slot.save()
print(f"✅ {slot.slot_number} is now under maintenance")

# Check all maintenance slots
maintenance_slots = ParkingSlot.objects.filter(status='maintenance')
print(f"Total slots under maintenance: {maintenance_slots.count()}")
for s in maintenance_slots:
    print(f"  - {s.slot_number} ({s.get_slot_type_display()})")

# Return slot to service
slot.status = 'available'
slot.save()
print(f"✅ {slot.slot_number} is now available")

# Exit
exit()
```



---

## 🎯 Status Options

Your parking slots can have 4 statuses:

| Status | Description | Available for Parking? |
|--------|-------------|----------------------|
| **Available** | Ready to accept vehicles | ✅ Yes |
| **Occupied** | Currently has a vehicle | ❌ No (auto-assigned) |
| **Under Maintenance** | Being repaired/maintained | ❌ No |
| **Out of Service** | Permanently closed | ❌ No |

---

## 📊 How It Works

### Automatic Behavior

1. **When assigning slots:**
   - System automatically **skips** maintenance and out-of-service slots
   - Only available slots are considered

2. **When vehicle exits:**
   - If slot status was 'occupied' → changes to 'available'
   - If slot was in maintenance → stays in maintenance

3. **Dashboard displays:**
   - Green: Available
   - Red: Occupied
   - Yellow/Orange: Under Maintenance
   - Gray: Out of Service

---

## 💡 Common Use Cases

### Scenario 1: Weekly Maintenance
```
TW5 needs cleaning every Friday
→ Mark as "Under Maintenance" Friday morning
→ Mark as "Available" Friday evening
```

### Scenario 2: Permanent Closure
```
C12 has structural damage
→ Mark as "Out of Service"
→ Won't be assigned until fixed and marked "Available"
```

### Scenario 3: Seasonal Closure
```
Close 10 outdoor slots in winter
→ Bulk select outdoor slots
→ Mark as "Out of Service"
→ Reopen in spring by marking "Available"
```

---

## 🔍 Verification

### Check Current Status

**Via API:**
```powershell
curl http://127.0.0.1:8000/api/toggle-maintenance/
```

**Via Command Line:**
```powershell
python manage.py shell -c "from parking_app.models import ParkingSlot; print(f'Maintenance: {ParkingSlot.objects.filter(status=\"maintenance\").count()}')"
```

**Via Dashboard:**
- Main dashboard shows count: "X slots under maintenance"
- Real-time view shows status badges

---

## 🛠️ Troubleshooting

### Slot still being assigned despite maintenance status?
Check the migration is applied:
```powershell
python manage.py showmigrations parking_app
```
Should show `[X] 0007_parkingslot_status`

### Can't change status via API?
- Make sure Django server is running
- Check the slot_number exists in database
- Verify you're not trying to mark an occupied slot as maintenance

### Status not showing in dashboard?
- Clear browser cache
- Restart Django server
- Check views are using `status` field queries

---

## 📝 Quick Commands Reference

```powershell
# Show migration status
python manage.py showmigrations parking_app

# Check slots by status
python manage.py shell -c "from parking_app.models import ParkingSlot; [print(f'{s}: {c}') for s, c in [('Available', ParkingSlot.objects.filter(status='available').count()), ('Occupied', ParkingSlot.objects.filter(status='occupied').count()), ('Maintenance', ParkingSlot.objects.filter(status='maintenance').count())]]"

# Bulk mark all TW slots as maintenance (example)
python manage.py shell -c "from parking_app.models import ParkingSlot; ParkingSlot.objects.filter(slot_type='two_wheeler').update(status='maintenance'); print('Done')"

# Bulk return all maintenance slots to available
python manage.py shell -c "from parking_app.models import ParkingSlot; ParkingSlot.objects.filter(status='maintenance').update(status='available'); print('Done')"
```

---

## 🎓 Example Workflow

```
1. Morning: Mark C5 as maintenance
   POST /api/toggle-maintenance/ {"slot_numbers": ["C5"], "action": "maintenance"}

2. System automatically excludes C5 from assignments
   User tries to park → Gets C6 instead

3. Dashboard shows: "1 slot under maintenance"
   GET /api/toggle-maintenance/ → shows C5 in maintenance list

4. Evening: Maintenance complete
   POST /api/toggle-maintenance/ {"slot_numbers": ["C5"], "action": "available"}

5. C5 is now available for new vehicles ✅
```

---

## Need Help?

- Check logs: `dashboard/dashboard.log`
- Test connection: `python manage.py test_oracle`
- View all slots: Visit admin panel or run `python manage.py shell`

---

**Feature ready to use! 🎉**
