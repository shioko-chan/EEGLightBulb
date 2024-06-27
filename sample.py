import threading
import queue
import time
import matplotlib.pyplot as plt
from pyOpenBCI import OpenBCICyton
import numpy as np
from pathlib import Path

eeg_data_queue = queue.Queue()


def eeg_callback(sample):
    eeg_data_queue.put((time.time(), sample.channels_data))


board = OpenBCICyton(port="COM3", daisy=False)

eeg_thread = threading.Thread(target=board.start_stream, args=(eeg_callback,))
eeg_thread.start()

num_classes = 2
samples_per_class = 10
sampling_duration = 1
eeg_data = []

class_labels = ["must have light", "not relative"]

try:
    for i in range(num_classes):
        if i == 0:
            continue
        for j in range(samples_per_class):
            print(
                f"###Get ready for {class_labels[i]}, sampling will start in 2 seconds..."
            )
            time.sleep(2)
            print(f"###Starting {class_labels[i]} sampling...")

            start_time = time.time()
            current_class_data = []
            while True:
                if not eeg_data_queue.empty():
                    t, data = eeg_data_queue.get()
                    if t < start_time:
                        continue
                    if t > start_time + sampling_duration:
                        break
                    current_class_data.append(data)

            eeg_data.append((i, current_class_data))

            print(f"###Finished {class_labels[i]} sampling.")
            time.sleep(1)

except KeyboardInterrupt:
    print("Data collection interrupted.")

board.stop_stream()
eeg_thread.join()

save_path = Path("./data/")
save_path.mkdir(exist_ok=True)

for label, data in eeg_data:
    data = np.array(data)
    data_path = save_path / f"{label}"
    data_path.mkdir(exist_ok=True)
    np.save(data_path / f"{round(time.time()*1000)}.npy", data)

print("EEG data and labels saved.")
