# Forbes Marshall SpotCheck API Documentation

## Overview
The Forbes Marshall SpotCheck API provides comprehensive endpoints for parking management operations, real-time monitoring, and system administration.

**Base URL:** `http://localhost:8000/api/`
**Version:** v2.0.0
**Authentication:** CSRF token required for POST/PUT/DELETE operations

## Authentication

### CSRF Token
All write operations require a CSRF token. Include the token in:
- Header: `X-CSRFToken: <token>`
- Form data: `csrfmiddlewaretoken=<token>`

```javascript
// Get CSRF token from cookie or meta tag
const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

fetch('/api/parking/assign/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken
    },
    body: JSON.stringify(data)
});
```

## Endpoints

### Parking Operations

#### GET `/api/parking/status/`
Get current parking system status and statistics.

**Response:**
```json
{
    "success": true,
    "data": {
        "stats": {
            "total_slots": 100,
            "available": 65,
            "occupied": 30,
            "reserved": 3,
            "maintenance": 2,
            "occupancy_rate": 30.0
        },
        "slots": [
            {
                "id": 1,
                "slot_number": "A001",
                "slot_type": "cars",
                "status": "available",
                "floor_level": 1
            }
        ],
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

#### POST `/api/parking/assign/`
Assign a parking slot to a vehicle.

**Request Body:**
```json
{
    "vehicle_type": "cars",
    "license_plate": "MH12AB1234",
    "owner_name": "John Doe",
    "owner_contact": "+91-9876543210"
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "slot_id": 15,
        "slot_number": "A015",
        "vehicle_id": 123,
        "session_id": 456,
        "entry_time": "2024-01-15T10:30:00Z",
        "message": "Slot A015 assigned successfully"
    }
}
```

**Error Response:**
```json
{
    "success": false,
    "error": {
        "code": "NO_SLOTS_AVAILABLE",
        "message": "No available slots for vehicle type: cars",
        "details": {
            "vehicle_type": "cars",
            "available_alternatives": ["large"]
        }
    }
}
```

#### POST `/api/parking/release/{slot_id}/`
Release a parking slot.

**Parameters:**
- `slot_id` (integer): ID of the slot to release

**Response:**
```json
{
    "success": true,
    "data": {
        "slot_id": 15,
        "slot_number": "A015",
        "vehicle": {
            "license_plate": "MH12AB1234",
            "owner_name": "John Doe"
        },
        "session": {
            "entry_time": "2024-01-15T10:30:00Z",
            "exit_time": "2024-01-15T14:45:00Z",
            "duration_minutes": 255,
            <!-- fee field removed (Free parking system) -->
        },
        "message": "Slot A015 released successfully"
    }
}
```

#### GET `/api/parking/slots/`
Get all parking slots with optional filtering.

**Query Parameters:**
- `status` (string): Filter by status (available|occupied|reserved|maintenance)
- `slot_type` (string): Filter by type (two-wheeler|cars|large|disabled|vip)
- `floor_level` (integer): Filter by floor level
- `page` (integer): Page number for pagination
- `limit` (integer): Items per page (default: 50, max: 100)

**Response:**
```json
{
    "success": true,
    "data": {
        "slots": [
            {
                "id": 1,
                "slot_number": "A001",
                "slot_type": "cars",
                "status": "available",
                "floor_level": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "current_vehicle": null,
                "current_session": null
            },
            {
                "id": 2,
                "slot_number": "A002",
                "slot_type": "cars",
                "status": "occupied",
                "floor_level": 1,
                "created_at": "2024-01-01T00:00:00Z",
                "current_vehicle": {
                    "id": 123,
                    "license_plate": "MH12AB1234",
                    "owner_name": "John Doe"
                },
                "current_session": {
                    "id": 456,
                    "entry_time": "2024-01-15T10:30:00Z",
                    "duration_minutes": 45
                }
            }
        ],
        "pagination": {
            "page": 1,
            "limit": 50,
            "total_items": 100,
            "total_pages": 2,
            "has_next": true,
            "has_previous": false
        }
    }
}
```

#### GET `/api/parking/slots/{slot_id}/`
Get details of a specific parking slot.

**Response:**
```json
{
    "success": true,
    "data": {
        "slot": {
            "id": 1,
            "slot_number": "A001",
            "slot_type": "cars",
            "status": "occupied",
            "floor_level": 1,
            "created_at": "2024-01-01T00:00:00Z"
        },
        "current_vehicle": {
            "id": 123,
            "license_plate": "MH12AB1234",
            "vehicle_type": "cars",
            "owner_name": "John Doe",
            "owner_contact": "+91-9876543210"
        },
        "current_session": {
            "id": 456,
            "entry_time": "2024-01-15T10:30:00Z",
            "duration_minutes": 45
            <!-- estimated_fee field removed (Free parking system) -->
        },
        "history": [
            {
                "session_id": 455,
                "vehicle_license": "MH11XY9876",
                "entry_time": "2024-01-14T09:15:00Z",
                "exit_time": "2024-01-14T17:30:00Z",
                "duration_minutes": 495
            }
        ]
    }
}
```

### Vehicle Management

#### GET `/api/vehicles/`
Get all registered vehicles.

**Query Parameters:**
- `license_plate` (string): Search by license plate
- `vehicle_type` (string): Filter by vehicle type
- `owner_name` (string): Search by owner name
- `page` (integer): Page number
- `limit` (integer): Items per page

**Response:**
```json
{
    "success": true,
    "data": {
        "vehicles": [
            {
                "id": 123,
                "license_plate": "MH12AB1234",
                "vehicle_type": "cars",
                "owner_name": "John Doe",
                "owner_contact": "+91-9876543210",
                "created_at": "2024-01-10T08:00:00Z",
                "total_visits": 15,
                "current_session": {
                    "slot_number": "A015",
                    "entry_time": "2024-01-15T10:30:00Z"
                }
            }
        ],
        "pagination": {
            "page": 1,
            "limit": 50,
            "total_items": 250,
            "total_pages": 5
        }
    }
}
```

#### POST `/api/vehicles/`
Register a new vehicle.

**Request Body:**
```json
{
    "license_plate": "MH12AB1234",
    "vehicle_type": "cars",
    "owner_name": "John Doe",
    "owner_contact": "+91-9876543210"
}
```

#### GET `/api/vehicles/{vehicle_id}/`
Get details of a specific vehicle.

#### PUT `/api/vehicles/{vehicle_id}/`
Update vehicle information.

#### DELETE `/api/vehicles/{vehicle_id}/`
Delete a vehicle record.

### Session Management

#### GET `/api/sessions/`
Get parking sessions with filtering and pagination.

**Query Parameters:**
- `is_active` (boolean): Filter active/inactive sessions
- `vehicle_id` (integer): Filter by vehicle
- `slot_id` (integer): Filter by slot
- `date_from` (date): Start date filter (YYYY-MM-DD)
- `date_to` (date): End date filter (YYYY-MM-DD)
- `page` (integer): Page number
- `limit` (integer): Items per page

**Response:**
```json
{
    "success": true,
    "data": {
        "sessions": [
            {
                "id": 456,
                "vehicle": {
                    "license_plate": "MH12AB1234",
                    "owner_name": "John Doe"
                },
                "slot": {
                    "slot_number": "A015",
                    "floor_level": 1
                },
                "entry_time": "2024-01-15T10:30:00Z",
                "exit_time": null,
                "is_active": true,
                "duration_minutes": 45
                <!-- estimated_fee field removed (Free parking system) -->
            }
        ],
        "pagination": {
            "page": 1,
            "limit": 50,
            "total_items": 1000,
            "total_pages": 20
        }
    }
}
```

#### GET `/api/sessions/{session_id}/`
Get details of a specific parking session.

### Analytics

#### GET `/api/analytics/occupancy/`
Get occupancy analytics.

**Query Parameters:**
- `period` (string): Time period (hour|day|week|month|year)
- `date_from` (date): Start date
- `date_to` (date): End date
- `granularity` (string): Data granularity (hourly|daily|weekly)

**Response:**
```json
{
    "success": true,
    "data": {
        "occupancy_data": [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "total_slots": 100,
                "occupied": 45,
                "available": 52,
                "reserved": 2,
                "maintenance": 1,
                "occupancy_rate": 45.0
            }
        ],
        "summary": {
            "avg_occupancy_rate": 42.5,
            "peak_occupancy": 89.0,
            "peak_time": "2024-01-15T14:30:00Z",
            "lowest_occupancy": 12.0,
            "lowest_time": "2024-01-15T03:00:00Z"
        }
    }
}
```

<!-- Revenue endpoint removed (Free parking system) -->

#### GET `/api/analytics/popular-times/`
Get popular parking times.

#### GET `/api/analytics/vehicle-types/`
Get vehicle type distribution.

### System Health

#### GET `/api/health/`
System health check endpoint.

**Response:**
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z",
        "version": "2.0.0",
        "database": {
            "status": "connected",
            "response_time_ms": 5
        },
        "cache": {
            "status": "available",
            "hit_rate": 85.2
        },
        "websocket": {
            "status": "active",
            "connections": 12
        },
        "disk_space": {
            "available_gb": 45.2,
            "used_percent": 62.8
        },
        "memory": {
            "used_percent": 45.6,
            "available_mb": 2048
        }
    }
}
```

#### GET `/api/health/detailed/`
Detailed system health information.

### Data Export

#### GET `/api/export/parking-data/`
Export parking data in various formats.

**Query Parameters:**
- `format` (string): Export format (csv|xlsx|json|pdf)
- `date_from` (date): Start date
- `date_to` (date): End date
- `include_sessions` (boolean): Include session data
- `include_vehicles` (boolean): Include vehicle data

**Response:** File download or JSON with download URL

### Real-time Updates

#### WebSocket `/ws/parking/`
WebSocket endpoint for real-time updates.

**Message Types:**
```json
// Slot status update
{
    "type": "slot_update",
    "slot_id": 15,
    "slot_number": "A015",
    "status": "occupied",
    "vehicle": {
        "license_plate": "MH12AB1234"
    }
}

// Occupancy statistics update
{
    "type": "occupancy_change",
    "stats": {
        "total_slots": 100,
        "available": 64,
        "occupied": 31,
        "occupancy_rate": 31.0
    }
}

// New vehicle registration
{
    "type": "new_vehicle",
    "vehicle": {
        "id": 124,
        "license_plate": "MH13CD5678",
        "owner_name": "Jane Smith"
    }
}

// System alert
{
    "type": "system_alert",
    "level": "warning",
    "message": "High occupancy rate detected",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Error Handling

### Standard Error Response
```json
{
    "success": false,
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {
            "field": "Additional error details"
        },
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

### Common Error Codes
- `VALIDATION_ERROR`: Request validation failed
- `NOT_FOUND`: Resource not found
- `SLOT_NOT_AVAILABLE`: No available slots
- `SLOT_ALREADY_OCCUPIED`: Slot is already occupied
- `VEHICLE_NOT_FOUND`: Vehicle not registered
- `SESSION_NOT_ACTIVE`: No active parking session
- `INSUFFICIENT_PERMISSIONS`: Access denied
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `DATABASE_ERROR`: Database operation failed
- `SYSTEM_MAINTENANCE`: System under maintenance

### HTTP Status Codes
- `200`: Success
- `201`: Created successfully
- `400`: Bad request / Validation error
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not found
- `409`: Conflict (e.g., slot already occupied)
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Service unavailable

## Rate Limiting

API requests are rate-limited to prevent abuse:
- **General endpoints**: 100 requests per minute per IP
- **Write operations**: 20 requests per minute per IP
- **Export endpoints**: 5 requests per minute per IP

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per minute
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

## SDKs and Libraries

### JavaScript/Node.js
```javascript
const ForbesMarshallSpotCheck = require('forbes-marshall-spotcheck-sdk');

const client = new ForbesMarshallSpotCheck({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your-api-key' // If using API key authentication
});

// Assign a parking slot
const result = await client.parking.assign({
    vehicleType: 'cars',
    licensePlate: 'MH12AB1234',
    ownerName: 'John Doe',
    ownerContact: '+91-9876543210'
});
```

### Python
```python
from forbes_marshall_spotcheck import SpotCheckClient

client = SpotCheckClient(
    base_url='http://localhost:8000',
    api_key='your-api-key'  # If using API key authentication
)

# Get parking status
status = client.parking.get_status()
print(f"Available slots: {status['data']['stats']['available']}")

# Assign parking slot
result = client.parking.assign(
    vehicle_type='cars',
    license_plate='MH12AB1234',
    owner_name='John Doe',
    owner_contact='+91-9876543210'
)
```

## Webhooks

Configure webhooks to receive real-time notifications about parking events.

### Webhook Events
- `slot.assigned`: When a slot is assigned
- `slot.released`: When a slot is released
- `vehicle.registered`: When a new vehicle is registered
- `occupancy.threshold`: When occupancy crosses defined thresholds
- `system.alert`: System alerts and notifications

### Webhook Payload
```json
{
    "event": "slot.assigned",
    "timestamp": "2024-01-15T10:30:00Z",
    "data": {
        "slot_id": 15,
        "slot_number": "A015",
        "vehicle": {
            "license_plate": "MH12AB1234",
            "owner_name": "John Doe"
        },
        "session_id": 456
    },
    "signature": "sha256=webhook_signature"
}
```

## Support

For API support and documentation updates:
- **Email**: support@forbesmarshall.com
- **Documentation**: http://localhost:8000/api/docs/
- **GitHub Issues**: https://github.com/forbes-marshall/spotcheck/issues
- **API Status**: http://localhost:8000/api/health/

---

**Forbes Marshall SpotCheck API v2.0.0**  
*Intelligent Parking Management System*  
© 2024 Forbes Marshall. All rights reserved.
