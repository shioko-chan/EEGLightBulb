import cv2
import socket

host = "0.0.0.0"
port = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((host, port))
img = cv2.imread("./0b86d7421a126103d40bc9ebf06944a591fd6fae_raw.jpg")
while True:
    cv2.imshow("1", img)
    cv2.waitKey(0)
    server_socket.sendto(b"toggle", ("192.168.137.6", 12345))
