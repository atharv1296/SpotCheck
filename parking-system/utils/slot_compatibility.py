"""
Slot Compatibility Algorithm for Parking System
Defines which vehicle types can park in which slot types
"""

# Slot compatibility matrix - which vehicles can use which slots
# Format: {vehicle_type: [compatible_slot_types_ordered_by_priority]}
SLOT_COMPATIBILITY = {
    'Two Wheeler': ['Two Wheeler', 'Cars', 'Large'],  # Two Wheeler can use all slots
    'Cars': ['Cars', 'Large'],                   # Cars can use Cars, Large
    'Large': ['Large']                                    # Large vehicles only in Large slots
}

def get_compatible_slot_types(vehicle_type):
    """
    Get compatible slot types for a vehicle type, ordered by priority
    Returns list of compatible slot types or empty list if invalid vehicle type
    """
    return SLOT_COMPATIBILITY.get(vehicle_type, [])

def can_vehicle_park_in_slot(vehicle_type, slot_type):
    """
    Check if a vehicle type can park in a slot type
    Returns True if compatible, False otherwise
    """
    compatible_slots = get_compatible_slot_types(vehicle_type)
    return slot_type in compatible_slots

def find_best_available_slot(vehicle_type, available_slots):
    """
    Find the best available slot for a vehicle type based on compatibility priority
    Returns slot_id and slot_number of the best match, or (None, None) if no suitable slot
    """
    compatible_types = get_compatible_slot_types(vehicle_type)
    
    if not compatible_types:
        return None, None  # Invalid vehicle type
    
    # Try to find slot in priority order (first compatible type is best match)
    for slot_type in compatible_types:
        for slot in available_slots:
            if slot['slot_type'] == slot_type:
                return slot['slot_id'], slot['slot_number']
    
    return None, None  # No compatible slot available

def get_slot_assignment_priority(vehicle_type, slot_type):
    """
    Get priority score for slot assignment (lower number = better match)
    Returns priority score or None if incompatible
    """
    compatible_types = get_compatible_slot_types(vehicle_type)
    
    if slot_type not in compatible_types:
        return None  # Incompatible
    
    # Priority is the index in the compatibility list (0 = best match)
    return compatible_types.index(slot_type)

def suggest_alternative_slots(vehicle_type, available_slots):
    """
    Suggest alternative slots when preferred type is not available
    Returns list of suggested alternative slots with compatibility info
    """
    compatible_types = get_compatible_slot_types(vehicle_type)
    suggestions = []
    
    for slot_type in compatible_types:
        slots_of_type = [s for s in available_slots if s['slot_type'] == slot_type]
        if slots_of_type:
            priority = compatible_types.index(slot_type)
            suggestions.append({
                'slot_type': slot_type,
                'available_count': len(slots_of_type),
                'priority_level': priority,
                'priority_label': get_priority_label(priority)
            })
    
    return sorted(suggestions, key=lambda x: x['priority_level'])

def get_priority_label(priority_level):
    """
    Get human-readable priority label
    """
    labels = {
        0: "Perfect match (ideal slot type)",
        1: "Good alternative",
        2: "Adequate option", 
        3: "Last resort option"
    }
    return labels.get(priority_level, "Compatible option")

def get_compatibility_rules():
    """
    Return the complete compatibility rules for display purposes
    """
    return SLOT_COMPATIBILITY.copy()    