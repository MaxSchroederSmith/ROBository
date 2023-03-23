import uuid

import bluetooth
from bluetooth import *

def init_bt():
    service_name = "Dealt"
    #service_class = [uuid, bluetooth.SERIAL_PORT_CLASS]
    #service_profile = [bluetooth.SERIAL_PORT_PROFILE]
    MY_UUID = "00001101-0000-1000-8000-00805F9B34FB"
    port = 1
    
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_socket.bind(("", port))
    server_socket.listen(1)
    #bluetooth.advertise_service(server_socket, MY_UUID, service_classes = service_class, profiles=service_profile, provider = "me", description = "test")
    #port = server_socket.getsockname()[1]
    print("Waiting for connection on RFCOMM channel", port)
    client_socket, client_info = server_socket.accept()
    print("Accepted connection from", client_info)
    
    
    
def recieveData():
    data = client_socket.recv(1024).decode().strip()
    return data

def sendData(msg):
    client_sock.send(msg.encode())
    