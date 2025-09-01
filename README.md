# Ideal, Natural, & Flat-top -Sampling
# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
# Tools required

Google Colab

# Program
## Ideal Sampling
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

fs = 100
t = np.arange(0, 1, 1/fs) 
f = 5
signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

```
## Natural Sampling
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --- Parameters ---
fs = 1000  # Sampling frequency (Hz)
T = 1      # Duration (s)
t = np.arange(0, T, 1/fs)  # Time vector

# --- Message Signal ---
fm = 5  # Message signal frequency (Hz)
message_signal = np.sin(2 * np.pi * fm * t)

# --- Pulse Train ---
pulse_rate = 50  # Pulse rate (Hz)
pulse_width = int(fs / (2 * pulse_rate)) # Half the period
pulse_train = np.zeros_like(t)
pulse_indices = np.arange(0, len(t), int(fs / pulse_rate))
for i in pulse_indices:
    pulse_train[i : i + pulse_width] = 1

# --- Sampling and Reconstruction ---
# Natural Sampling (multiplying message by pulse train)
nat_sampled_signal = message_signal * pulse_train

# Zero-order hold reconstruction
reconstructed_signal = np.repeat(nat_sampled_signal[pulse_indices], int(fs / pulse_rate))
# Ensure the reconstructed signal has the same length as the original
reconstructed_signal = reconstructed_signal[:len(t)]

# Low-pass filter for smoother reconstruction
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

# Apply the filter
reconstructed_signal_filtered = lowpass_filter(reconstructed_signal, 10, fs)

# --- Plotting ---
plt.figure(figsize=(14, 10))

# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.legend()
plt.grid(True)

# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.title('Pulse Train')
plt.legend()
plt.grid(True)

# Naturally Sampled Signal
plt.subplot(4, 1, 3)
plt.plot(t, nat_sampled_signal, label='Naturally Sampled Signal')
plt.title('Naturally Sampled Signal')
plt.legend()
plt.grid(True)

# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal_filtered, label='Reconstructed Message Signal', color='green')
plt.title('Reconstructed Message Signal (Filtered)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```
## Flat Top Sampling
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000  # Sampling frequency (samples per second)
T = 1      # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector
fm = 5     # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)
pulse_rate = 50  # pulses per second
pulse_train_indices = np.arange(0, len(t), int(fs / pulse_rate))
pulse_train = np.zeros_like(t)
pulse_train[pulse_train_indices] = 1
flat_top_signal = np.zeros_like(t)
sample_times = t[pulse_train_indices]
pulse_width_samples = int(fs / (2 * pulse_rate)) # Adjust pulse width as needed

for i, sample_time in enumerate(sample_times):
    index = np.argmin(np.abs(t - sample_time))
    if index < len(message_signal):
        sample_value = message_signal[index]
        start_index = index
        end_index = min(index + pulse_width_samples, len(t))
        flat_top_signal[start_index:end_index] = sample_value

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

cutoff_freq = 2 * fm  # Nyquist rate or slightly higher
reconstructed_signal = lowpass_filter(flat_top_signal, cutoff_freq, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.stem(t[pulse_train_indices], pulse_train[pulse_train_indices], basefmt=" ", label='Ideal Sampling Instances')
plt.title('Ideal Sampling Instances')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.title('Flat-Top Sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label=f'Reconstructed Signal (Low-pass Filter, Cutoff={cutoff_freq} Hz)', color='green')
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
# Output Waveform
## Ideal Sampling
<img width="989" height="889" alt="ideal" src="https://github.com/user-attachments/assets/a1cb3c43-b847-4948-b0fa-6da9f663cea6" />


## Natural Sampling
<img width="1390" height="989" alt="natural" src="https://github.com/user-attachments/assets/b4537ca5-fc1c-4218-808f-e2000338d415" />

## Flat Top Sampling
<img width="1398" height="990" alt="flatop" src="https://github.com/user-attachments/assets/f3806566-5f29-4ed8-af05-109fedad03ef" />

# Results
Thus we consrtucted and reconsrtucted ideal, natural and flat top sampling using python code.

