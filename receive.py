import socket
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)


gpio_pin = 7
GPIO.setup(gpio_pin, GPIO.OUT, initial=GPIO.LOW)


host = "0.0.0.0"
port = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


server_socket.bind((host, port))

print(f"Listening on {host}:{port}...")

try:
    while True:
        data, addr = server_socket.recvfrom(1024)
        message = data.decode("utf-8")
        if message.strip().lower() == "toggle":
            GPIO.output(gpio_pin, not GPIO.input(gpio_pin))
            print(f"GPIO 7 state toggled: {GPIO.input(gpio_pin)}")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    GPIO.cleanup()
    server_socket.close()
