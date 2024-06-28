import threading
import queue
import time
import numpy as np
from pyOpenBCI import OpenBCICyton
from train import LSTMNet, pad_or_trim
import torch
from pathlib import Path
import socket

host = "0.0.0.0"
port = 12345

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((host, port))

input_size = 8  # 输入特征数
hidden_size = 64  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
num_classes = 2  # 输出类别数
train_name = "first"
save_path = Path(f"models/{train_name}")

model = LSTMNet(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load(save_path / "lstm_model_epoch10.pt"))
model.eval()

eeg_data_queue = queue.Queue()


def eeg_callback(sample):
    eeg_data_queue.put((time.time(), sample.channels_data))


board = OpenBCICyton(port="COM3", daisy=False)

eeg_thread = threading.Thread(target=board.start_stream, args=(eeg_callback,))
eeg_thread.start()

sampling_duration = 1
max_length = 0
with open(save_path / "config.txt", "r") as f:
    max_length = int(f.readline())
print(max_length)

try:
    while True:
        current_data = []
        start_time = time.time()
        while True:
            if not eeg_data_queue.empty():
                t, data = eeg_data_queue.get()
                if t < start_time:
                    continue
                if t > start_time + sampling_duration:
                    break
                current_data.append(data)
        current_data = np.array(current_data)
        try:
            data = pad_or_trim(current_data, max_length)
            data = np.array([data])
            data = torch.from_numpy(data).float()
            output = model(data)
            _, predicted = torch.max(output, 1)
            if int(predicted) == 0:
                server_socket.sendto(b"toggle", ("192.168.137.6", 12345))
        except:
            pass


except KeyboardInterrupt:
    print("Inference interrupted.")

board.stop_stream()
eeg_thread.join()
