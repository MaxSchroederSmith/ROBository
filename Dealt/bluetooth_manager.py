import uuid

import bluetooth
from bluetooth import *
import socket

def init_bt():
    service_name = "Dealt"
    service_class = [bluetooth.SERIAL_PORT_CLASS]
    service_profile = [bluetooth.SERIAL_PORT_PROFILE]
    MY_UUID = "00001101-0000-1000-8000-00805F9B34FB"
    port = 1
    
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("", bluetooth.PORT_ANY))
    server_socket.listen(1)
    
    port = server_socket.getsockname()[1]
    bluetooth.advertise_service(server_socket, MY_UUID, service_classes = service_class, profiles=service_profile, provider = "me", description = "test")
    print("Waiting for connection on RFCOMM channel", port)
    client_socket, client_info = server_socket.accept()
    print("Accepted connection from", client_info)
    return client_socket
    
    
def recieveData(client_socket):
    data = client_socket.recv(1024).decode().strip()
    return data

def sendData(msg, client_socket):
    client_socket.send(msg.encode())
    