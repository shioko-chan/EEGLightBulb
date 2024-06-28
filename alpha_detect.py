import json
import socket

host = "0.0.0.0"
port = 12345

server_socket = socket.socket(
    socket.AF_INET, socket.SOCK_DGRAM
)  # 从OpenBCI GUI 读取处理后的数据
server_socket.bind((host, port))
try:
    while True:
        mes = server_socket.recv(1024)
        obj = json.loads(mes.decode())
        aver = sum([li[2] for li in obj["data"]]) / len(obj["data"])
        if aver > 10.0:
            print("detected!")
            server_socket.sendto(b"light", ("192.168.137.6", 12345))
        else:
            print("not detected.")
            server_socket.sendto(b"snuff", ("192.168.137.6", 12345))
except KeyboardInterrupt:
    print("Data collection interrupted.")
except Exception as e:
    print(e)
