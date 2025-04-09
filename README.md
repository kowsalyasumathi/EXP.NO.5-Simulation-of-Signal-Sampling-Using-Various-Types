# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM: 
To study and analyze the sampling of natural, ideal and flat top sampling.

# SOFTWARE REQUIRED: 
Google colab software(python)

# ALGORITHMS:
step1: Generate a continuous signal using a sine wave.

step2: Apply uniform sampling by selecting fixed-interval samples.

step3: Apply random sampling by selecting random indices.

step4: Apply Platop sampling using probability-based selection.

step5: Plot the original signal and sampled points.

step6: Reconstruct the signal using resampling.



# PROGRAM
#impulse sampling
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
fs = 100
t = np.arange(0, 1, 1/fs) 
f = 6
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
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

```

#natural sampling
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
fs = 1000  
T = 1  
t = np.arange(0, T, 1 / fs)  
fm = 8  
message_signal = np.sin(2 * np.pi * fm * t)
pulse_rate = 50 
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:min(i + pulse_width, len(t))] = 1
nat_signal = message_signal * pulse_train
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(nat_signal, 10, fs) 
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```

#Flattop sampling
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
fs = 1000  
T = 1  
t = np.arange(0, T, 1/fs)  
fm = 4  
message_signal = np.sin(2 * np.pi * fm * t)
pulse_rate = 50 
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 4)  
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1
flat_top_signal = np.copy(message_signal)
for i in range(0, len(t), int(fs / pulse_rate)):
    flat_top_signal[i:i+pulse_width] = message_signal[i]  
sampled_signal = flat_top_signal[pulse_train == 1]
sample_times = t[pulse_train == 1]
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(t, flat_top_signal, label='Flat-Top Sampled Signal')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='red')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

# OUTPUT
## Impulse sampling
![image](https://github.com/user-attachments/assets/1cbc04b9-0f56-45e9-bac5-133c58176275)

## Natural sampling
![image](https://github.com/user-attachments/assets/cc51fac7-ee28-46ff-9d6b-c1bdf41188c7)

## Flattop sampling
![image](https://github.com/user-attachments/assets/98971591-07d3-4e75-8fdc-8aaa101bbdbf)



# RESULT / CONCLUSIONS:
Thus the sampling of natural, ideal and flattop sampling techniques were simulated and output waveform is obtained.


