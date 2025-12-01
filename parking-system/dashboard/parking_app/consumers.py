# WebSocket consumers for real-time updates (optional)
# This file is for future WebSocket implementation

from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ParkingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add(
            "parking_updates",
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            "parking_updates",
            self.channel_name
        )

    async def parking_update(self, event):
        await self.send(text_data=json.dumps({
            'type': 'parking_update',
            'data': event['data']
        }))
