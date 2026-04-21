# Experimental verification of Signal sampling using various types
# Aim
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
# Tools Required

- Google Colab
- Python
- NumPy Library
- Matplotlib Library
- Internet Connection
- Computer / Laptop

# Theory

Sampling is the process of converting a continuous-time analog signal into a discrete-time signal by taking the values of the signal at regular time intervals. It is widely used in digital communication systems, signal processing, and data conversion.

According to the Nyquist Sampling Theorem, the sampling frequency must be at least twice the highest frequency present in the message signal to reconstruct the original signal without distortion.

Types of Sampling
1. Ideal Sampling

In ideal sampling, the analog signal is multiplied by an impulse train of very short duration. The sampled output consists of instantaneous values of the signal at equally spaced intervals. It is theoretical and not practically realizable.

2. Natural Sampling

In natural sampling, the analog signal is sampled using pulses of finite width. During the pulse duration, the top of each pulse follows the shape of the input signal. It is more practical than ideal sampling.

3. Flat-top Sampling

In flat-top sampling, the sampled value remains constant during the pulse width. The amplitude of each sample equals the signal value at the start of sampling instant. This method is commonly used in practical sample-and-hold circuits.
# Program
```python
import numpy as np
import matplotlib.pyplot as plt
# Continuous signal parameters
fm = 5                      # Message frequency 5hz
fs = 50                     # Sampling frequency 50hz
t = np.linspace(0, 1, 1000) # Continuous time axis
ts = np.arange(0, 1, 1/fs)  # Sampled time axis

# Original analog signal
x = np.sin(2 * np.pi * fm * t)#x=sin(2pifmt)

# Sampled values
xs = np.sin(2 * np.pi * fm * ts)

# Flat-top sampling generation
flat_top = np.zeros_like(t)

for i in range(len(ts)-1):
    idx = np.where((t >= ts[i]) & (t < ts[i+1]))
    flat_top[idx] = xs[i]

# Plotting
plt.figure(figsize=(12,8))

# Original signal
plt.subplot(4,1,1)
plt.plot(t, x)
plt.title("Original Analog Signal")
plt.grid()

# Ideal Sampling
plt.subplot(4,1,2)
plt.stem(ts, xs, basefmt=" ")
plt.title("Ideal Sampling")
plt.grid()

# Natural Sampling (approximated)
plt.subplot(4,1,3)
plt.plot(t, x)
plt.stem(ts, xs, linefmt='r', markerfmt='ro', basefmt=" ")
plt.title("Natural Sampling")
plt.grid()

# Flat-top Sampling
plt.subplot(4,1,4)
plt.plot(t, flat_top)
plt.title("Flat-top Sampling")
plt.grid()

plt.tight_layout()
plt.show()
```
# Output Waveform
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/87df26f4-c05e-4037-b739-9c4fc9df7061" />


# Results
Thus we consrtucted and reconsrtucted ideal, natural and flat top sampling using python code.

