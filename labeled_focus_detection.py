import time
import numpy as np
import serial
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import welch

# === CONFIGURATION ===
board_id = BoardIds.CYTON_BOARD.value
params = BrainFlowInputParams()
params.serial_port = 'COM13'       # Your OpenBCI Dongle port
arduino_port = 'COM15'             # Your Arduino port
baud_rate = 9600
sampling_rate = 250
task_duration_sec = 60             # 1 minute per task

# === TASKS ===
tasks = [
    ("Reading", task_duration_sec),
    ("Staring", task_duration_sec)
]

# === Initialize Serial and BrainFlow ===
arduino = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # let Arduino settle

board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()
print("🧠 EEG Streaming Started...")

# === Data recording ===
eeg_channel_index = 0  # use first EEG channel (Fp1)
log_rows = []
start_time = time.time()

# === Focus Detection ===
def compute_focus(eeg_signal):
    freqs, psd = welch(eeg_signal, fs=sampling_rate, nperseg=128)
    alpha = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
    beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
    total = np.sum(psd)
    focus = 1 if total > 100 and beta > alpha else 0
    return alpha, beta, total, focus

try:
    for task_name, duration in tasks:
        print(f"\n📝 Task: {task_name} | Duration: {duration}s")
        for _ in range(duration * 5):  # 5 samples/sec = 0.2s interval
            data = board.get_current_board_data(sampling_rate)
            eeg_data = data[BoardShim.get_eeg_channels(board_id)[eeg_channel_index], :]

            alpha, beta, total, focus = compute_focus(eeg_data)

            # Send to Arduino
            arduino.write(f"{focus}\n".encode())

            # Log data
            t = time.time() - start_time
            log_rows.append([t, alpha, beta, total, focus, task_name])
            print(f"{t:.1f}s | {task_name:<8} | Focus: {focus} | α: {alpha:.2f} | β: {beta:.2f}")

            time.sleep(0.2)

    print("\n✅ Session complete.")

except KeyboardInterrupt:
    print("\n⛔ Interrupted manually.")

finally:
    board.stop_stream()
    board.release_session()
    arduino.close()

    df = pd.DataFrame(log_rows, columns=["Time", "Alpha", "Beta", "Total", "Focus", "Task"])
    df.to_csv("labeled_focus_log.csv", index=False)
    print("📁 Data saved to labeled_focus_log.csv")
