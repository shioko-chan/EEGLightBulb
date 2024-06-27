import threading
import queue
import time
import matplotlib.pyplot as plt
from pyOpenBCI import OpenBCICyton
import numpy as np
from pathlib import Path
from psychopy import visual, core, event
from scipy.signal import iirnotch, filtfilt, butter


eeg_data_queue = queue.Queue()


board = OpenBCICyton(port="COM3", daisy=False)
fs = 250.0

notch_freq = 50.0

quality_factor = 30.0

b, a = iirnotch(notch_freq, quality_factor, fs)

lowcut = 5.0
highcut = 50.0
order = 4
b, a = butter(order, [lowcut, highcut], btype="band", fs=fs)


def eeg_callback(sample):
    eeg_data_filtered = filtfilt(b, a, sample.channels_data, axis=0)
    eeg_data_filtered = filtfilt(b, a, eeg_data_filtered, axis=0)
    eeg_data_queue.put((time.time(), sample.channels_data))


eeg_thread = threading.Thread(target=board.start_stream, args=(eeg_callback,))
eeg_thread.start()

samples_per_class = 20
sampling_duration = 1

class_labels = ["Left", "Right", "Up", "Down"]
num_classes = len(class_labels)

win = visual.Window([800, 600], monitor="testMonitor", units="pix")

eeg_data = []

cue_text = visual.TextStim(win, text=f"Imagine movement of the ball")
cue_text.draw()
win.flip()
core.wait(5)

try:
    for i in range(num_classes):
        for j in range(samples_per_class):
            ball = visual.Circle(
                win=win, radius=10, fillColor="blue", lineColor="blue", units="pix"
            )
            start_time = time.time()
            x_pos = 0
            y_pos = 0
            if i == 0:
                x_speed = -4
                y_speed = 0
            if i == 1:
                x_speed = 4
                y_speed = 0
            if i == 2:
                x_speed = 0
                y_speed = -3
            if i == 3:
                x_speed = 0
                y_speed = 3

            while x_pos < 400 and x_pos > -400 and y_pos < 300 and y_pos > -300:
                x_pos += x_speed
                y_pos += y_speed
                ball.pos = (x_pos, y_pos)
                ball.draw()
                win.flip()
                core.wait(sampling_duration / 100)

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
            if j % 5 == 0 and j != 0:
                cue_text = visual.TextStim(win, text=f"break time for 5s")
                cue_text.draw()
                win.flip()
                core.wait(5)
        print("finish one sample.")


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
win.close()
core.quit()
