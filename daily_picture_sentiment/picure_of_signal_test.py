from pymongo import MongoClient
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Parameters for the damped sinusoidal wave
A = 1               # Initial amplitude
lambda_decay = 0.5  # Decay rate
omega = 8 * np.pi   # Angular frequency (1 Hz frequency)
sampling_rate = 200  # Samples per second
duration = 5         # Duration of the signal in seconds

# Generate time and signal
time = np.linspace(0, duration, sampling_rate * duration)
signal = A * np.exp(-lambda_decay * time) * np.sin(omega * time)

# Add 1 second of zeros at the start and end
zero_padding = int(sampling_rate)  # 1 second of zeros
time = np.concatenate([np.linspace(-1, 0, zero_padding), time, np.linspace(duration, duration + 1, zero_padding)])
signal = np.concatenate([np.zeros(zero_padding), signal, np.zeros(zero_padding)])

# Frequency domain using FFT
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)
amplitude_spectrum = np.abs(fft_result)

# Only keep positive frequencies
positive_freq_idx = frequencies >= 0
frequencies = frequencies[positive_freq_idx]
amplitude_spectrum = amplitude_spectrum[positive_freq_idx]

# Reshape to create 2D mapping: Time (X), Frequency (Y), Amplitude (Z)
time_mesh, freq_mesh = np.meshgrid(time, frequencies)
z_mesh = np.abs(amplitude_spectrum[:, None] * np.sin(omega * time_mesh))

# Create a DataFrame for 2D histogram
df_2d = pd.DataFrame({
    'Time': time_mesh.flatten(),
    'Frequency': freq_mesh.flatten(),
    'Amplitude': z_mesh.flatten()
})

# Create a DataFrame for the time domain plot
df_time = pd.DataFrame({'Time': time, 'Amplitude': signal})

# Create a DataFrame for the frequency domain plot
df_freq = pd.DataFrame({'Frequency': frequencies, 'Amplitude': amplitude_spectrum})

# Time-domain plot
fig_time = px.line(df_time, x='Time', y='Amplitude', title='Signal in Time Domain',
                   labels={'Time': 'Time (s)', 'Amplitude': 'Amplitude'})
fig_time.show()

# Frequency-domain plot
fig_freq = px.line(df_freq, x='Frequency', y='Amplitude', title='Signal in Frequency Domain',
                   labels={'Frequency': 'Frequency (Hz)', 'Amplitude': 'Amplitude Spectrum'})
fig_freq.show()

# 2D histogram (time-frequency-amplitude domain)
fig_2d = px.density_heatmap(df_2d, x='Time', y='Frequency', z='Amplitude', nbinsx=50, nbinsy=50,
                            color_continuous_scale="Viridis", histfunc='avg',
                            title="2D Histogram of Signal in Time-Frequency-Amplitude Domain",
                            labels={'Time': 'Time (s)', 'Frequency': 'Frequency (Hz)', 'Amplitude': 'Amplitude'})
fig_2d.show()