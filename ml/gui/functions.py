from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import json

#### Websockets
def ws_send_data(content = {}, group_name = 'connectedUsers'):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(group_name, {
        'type': 'test.message',
        'content': content,
    })


def send_to_gui(data = None, group_name = 'connectedUsers'):
    content = json.dumps({
        'data': data,
    })

    ws_send_data(content, group_name)