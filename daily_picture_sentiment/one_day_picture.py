from pymongo import MongoClient
import pandas as pd
import numpy as np
import plotly.express as px

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')
db = client.tekinvestor
collection = db.pci_biotech_llama_10

# Query to fetch data for the exact date
# query_date = "2019-05-10"
query_date = "2019-05-11"
start_date = f"{query_date}T00:00:00.000Z"
end_date = f"{query_date}T23:59:59.999Z"

# Find data using a date range
data = list(collection.find({
    "post_published_date": {
        "$gte": pd.to_datetime(start_date),
        "$lt": pd.to_datetime(end_date)
    }
}, {"_id": 0}))

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Check if data is retrieved
if not df.empty:
    # Ensure date parsing for plotting
    df['post_published_date'] = pd.to_datetime(df['post_published_date'])

    # Plot time-domain data
    fig = px.scatter(df, x='post_published_date', y='credibility_value',
                     title=f'Data from {query_date}',
                     labels={'post_published_date': 'Published Date', 'credibility_value': 'Credibility Value'})
    fig.show()

    # Prepare data for FFT
    signal = df['credibility_value'].fillna(0).values  # Use 'credibility_value' as the signal

    # Perform FFT
    sampling_rate = 1  # Assuming 1 sample per unit time (adjust as needed)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)
    amplitude_spectrum = np.abs(fft_result)

    # Only keep positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    amplitude_spectrum = amplitude_spectrum[positive_freq_idx]

    # Create DataFrames for plotting
    df_time = pd.DataFrame({'Time': df['post_published_date'], 'Amplitude': signal})
    df_freq = pd.DataFrame({'Frequency': frequencies, 'Amplitude': amplitude_spectrum})

    # For 2D Histogram
    time_mesh, freq_mesh = np.meshgrid(
        (df['post_published_date'] - df['post_published_date'].min()).dt.total_seconds(),  # Convert to seconds
        frequencies
    )
    z_mesh = np.abs(amplitude_spectrum[:, None] * np.sin(2 * np.pi * freq_mesh * time_mesh))

    df_2d = pd.DataFrame({
        'Time': (df['post_published_date'].min() + pd.to_timedelta(time_mesh.flatten(), unit='s')),  # Convert back to datetime
        'Frequency': freq_mesh.flatten(),
        'Amplitude': z_mesh.flatten()
    })

    # Time-domain plot
    fig_time = px.line(df_time, x='Time', y='Amplitude', title='Signal in Time Domain',
                       labels={'Time': 'Time', 'Amplitude': 'Amplitude'})
    fig_time.show()

    # Frequency-domain plot
    fig_freq = px.line(df_freq, x='Frequency', y='Amplitude', title='Signal in Frequency Domain',
                       labels={'Frequency': 'Frequency (Hz)', 'Amplitude': 'Amplitude Spectrum'})
    fig_freq.show()

    # 2D histogram (time-frequency-amplitude domain)
    fig_2d = px.density_heatmap(df_2d, x='Time', y='Frequency', z='Amplitude', nbinsx=320, nbinsy=320,
                                color_continuous_scale="Viridis", histfunc='avg',
                                title="2D Histogram of Signal in Time-Frequency-Amplitude Domain",
                                labels={'Time': 'Time', 'Frequency': 'Frequency (Hz)', 'Amplitude': 'Amplitude'})
    fig_2d.show()

else:
    print(f"No data found for {query_date}.")
