from channels.generic.websocket import AsyncJsonWebsocketConsumer

class MainConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        if self.scope['user'].is_anonymous:
            self.close()

        await self.channel_layer.group_add('connectedUsers', self.channel_name)
        await self.accept()


    async def disconnect(self, close_code):
        await self.channel_layer.group_discard('connectedUsers',self.channel_name)


    async def test_message(self, event):
        await self.send_json(event['content'])
    

    async def receive(self, text_data=None, bytes_data=None):
        # print(text_data)
        await self.send_json({
            'data': 'received',
        })