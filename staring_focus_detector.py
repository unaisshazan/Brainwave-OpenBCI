
import time
import numpy as np
import pandas as pd
import serial
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import welch

# === CONFIGURATION ===
board_id = BoardIds.CYTON_BOARD.value
params = BrainFlowInputParams()
params.serial_port = 'COM13'
arduino_port = 'COM15'
baud_rate = 9600
sampling_rate = 250

# Strict alpha/beta range considered as "focus" open bci unit is micro volt
# psd μV²/Hz
ALPHA_RANGE = (0, 15) #this is for Andy alpha 0.0 to 15 beta 0.1 to 20 Vesper 0 - 100, 0.1 - 100   Ryron 0 - 30 0.1 - 20   Leo 0 - 10 0 - 10  TiTi 0 - 15 0 - 10  
BETA_RANGE = (0.1, 10) #for ao 0 to 100

# === Setup Serial and EEG board ===
arduino = serial.Serial(arduino_port, baud_rate)
time.sleep(2)
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()
print("🧠 Streaming EEG with strict focus range detection...")

# === Setup Plot ===
plt.ion()
fig, ax = plt.subplots()
line_alpha, = ax.plot([], [], label='Alpha')
line_beta, = ax.plot([], [], label='Beta')
line_led, = ax.plot([], [], 'go', label='LED ON', markersize=5)
ax.set_title("Strict Alpha/Beta Range Focus Detection")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)

times, alpha_vals, beta_vals, leds = [], [], [], []
start_time = time.time()

try:
    while True:
        data = board.get_current_board_data(sampling_rate)
        eeg_ch = BoardShim.get_eeg_channels(board_id)[0]
        eeg_signal = data[eeg_ch, :]

        freqs, psd = welch(eeg_signal, fs=sampling_rate, nperseg=128)
        alpha = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
        beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])

        t = time.time() - start_time
        times.append(t)
        alpha_vals.append(alpha)
        beta_vals.append(beta)

        # Determine if within focused range
        is_focus = ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1] and BETA_RANGE[0] <= beta <= BETA_RANGE[1]
        leds.append(beta if is_focus else np.nan)

        # Send to Arduino
        arduino.write(f"{int(is_focus)}\n".encode())
        print(f"{t:.1f}s | Alpha: {alpha:.3f} | Beta: {beta:.3f} | {'🟢 LED ON' if is_focus else '⚪ LED OFF'}")

        # Keep last 50 points
        max_len = 50
        times, alpha_vals, beta_vals, leds = times[-max_len:], alpha_vals[-max_len:], beta_vals[-max_len:], leds[-max_len:]

        # Update Plot
        line_alpha.set_data(times, alpha_vals)
        line_beta.set_data(times, beta_vals)
        line_led.set_data(times, leds)
        ax.relim(), ax.autoscale_view()
        plt.pause(0.01)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    board.stop_stream()
    board.release_session()
    arduino.close()
    print("✅ Session ended cleanly.")

