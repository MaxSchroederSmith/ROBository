import uuid

import bluetooth

MY_UUID = "00001101-0000-1000-8000-00805F9B34FB"
server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_socket.bind(("", bluetooth.PORT_ANY))
server_socket.listen(1)

port = server_socket.getsockname()[1]

bluetooth.advertise_service(server_socket, "MyService", service_id=uuid.uuid1(),
                            service_classes=[MY_UUID, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE])

print("Waiting for connection on RFCOMM channel", port)

client_socket, client_info = server_socket.accept()
print("Accepted connection from", client_info)

while True:
    # Read data from the client socket
    data = client_socket.recv(1024).decode().strip()
    print('Received:', data)

    # Send data to the client socket
    response = 'Hello from Python!'
    client_socket.send(response.encode())
