from rest_framework import serializers
from .models import Vehicle, ParkingSlot, ParkingSession

class VehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vehicle
        fields = '__all__'

class ParkingSlotSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParkingSlot
        fields = '__all__'

class ParkingSessionSerializer(serializers.ModelSerializer):
    vehicle = VehicleSerializer(read_only=True)
    parking_slot = ParkingSlotSerializer(read_only=True)
    duration_minutes = serializers.SerializerMethodField()
    
    class Meta:
        model = ParkingSession
        fields = '__all__'
    
    def get_duration_minutes(self, obj):
        duration = obj.duration
        return int(duration.total_seconds() / 60)
