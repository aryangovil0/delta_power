import mne
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
import os
import time
from termcolor import colored
import shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Prompt the user to input the path to the EDF file and the output directory
file_path = input("Enter the path to the EDF file: ")
output_dir = input("Enter the output directory: ")
num_threads = int(input("Enter the number of threads to use: "))
input_file_name = os.path.splitext(os.path.basename(file_path))[0]

# Load the EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

# Extract data information
info = raw.info
sampling_rate = info['sfreq']
channel_names = info['ch_names']

# Extract the real-time start date and time
meas_date = info['meas_date']
if meas_date is None:
    meas_date = datetime.now()
print(f"Recording start time: {meas_date}")

# Print the descriptives about the data
duration_seconds = raw.times[-1] - raw.times[0]
duration_minutes = duration_seconds / 60
duration_hours = duration_minutes / 60
print(f"Duration of the recording: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes, {duration_hours:.2f} hours)")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Number of channels: {info['nchan']}")
print("Channel names:")
for i, name in enumerate(channel_names):
    print(f"{i + 1}. {name}")
print(f"Data user info: {info['subject_info']}")

## Specify the channels of interest [IMPORTANT] ##
channels_of_interest = [
    'EEG C3-A2', 'EEG C4-A1', 'EEG O1-A2', 'EEG O2-A1', 
    'EEG A1-A2', 'EOG LOC-A2', 'EOG ROC-A1', 'EEG F3-A2', 'EEG F4-A1'
]

# Extract the data for the channels of interest
picks = mne.pick_channels(raw.info['ch_names'], include=channels_of_interest)
raw_corrected = raw.copy().filter(1, 60)

# Perform FFT on each EEG channel and plot them side by side
def process_fft(ch_name, pick, ax):
    data, times = raw_corrected[pick, :]
    data = data[0]  # because MNE returns a 2D array, we need a 1D array for FFT

    # Perform FFT
    freqs = np.fft.rfftfreq(len(data), d=1/sampling_rate)
    fft_vals = np.fft.rfft(data)

    # Plot FFT results
    ax.plot(freqs, np.abs(fft_vals))
    ax.set_title(ch_name)
    ax.set_xlabel('Frequency (Hz)')
    if ax == axes_fft[0]:
        ax.set_ylabel('Amplitude')

num_channels = len(channels_of_interest)
fig_fft, axes_fft = plt.subplots(1, num_channels, figsize=(20, 5), sharey=True)
fig_fft.suptitle('FFT of EEG Channels', fontsize=16)

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i, ch_name in enumerate(channels_of_interest):
        executor.submit(process_fft, ch_name, picks[i], axes_fft[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Perform Continuous Wavelet Transform (CWT) and extract delta power
def process_cwt(ch_name, pick, ax):
    data, times = raw_corrected[pick, :]
    data = data[0]  # MNE returns a 2D array, we need a 1D array for CWT

    # CWT
    widths = np.arange(1, 31)
    cwt_matrix, freqs = pywt.cwt(data, widths, 'mexh', sampling_period=1/sampling_rate)

    # Extract delta power (0.5-4 Hz)
    delta_idx = np.where((freqs >= 0.5) & (freqs <= 4))
    delta_power = np.sum(np.abs(cwt_matrix[delta_idx])**2, axis=0)
    delta_power_dict[ch_name] = delta_power

    # Plot delta power over time
    ax.plot(times, delta_power)
    ax.set_title(ch_name)
    ax.set_xlabel('Time (s)')
    if ax == axes_delta[0]:
        ax.set_ylabel('Power')

start_time = time.time()
fig_delta, axes_delta = plt.subplots(1, num_channels, figsize=(20, 5), sharey=True)
fig_delta.suptitle('Delta Power of EEG Channels', fontsize=16)

delta_power_dict = {'Time': raw_corrected.times}
delta_power_dict['Real Time'] = [meas_date + timedelta(seconds=t) for t in raw_corrected.times]

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i, ch_name in enumerate(channels_of_interest):
        executor.submit(process_cwt, ch_name, picks[i], axes_delta[i])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

end_time = time.time()
print(f"Time taken to perform Continuous Wavelet Transform (CWT) and extract delta power: {end_time - start_time:.2f} seconds")

# Prompts for displaying data
print("Display full .edf plot, fft, delta power plots? y/n")
display_raw_plot = input().lower() == 'y'
if display_raw_plot:
    raw.plot(n_channels=len(raw.ch_names), scalings='auto', title='Raw EEG Data', show=True, block=True)

#print("Display full .edf plot of the fully filtered and clean data? y/n")
#display_filtered_plot = input().lower() == 'y'
#if display_filtered_plot:
#    raw_corrected.plot(n_channels=len(raw_corrected.ch_names), scalings='auto', title='Filtered and Clean EEG Data', show=True, block=True)

# Prompts for saving data
print("Save filtered data (bandpassed) as EDF? y/n")
save_filtered_data = input().lower() == 'y'

print("Save FFT plots as image? y/n")
save_fft_image = input().lower() == 'y'

print("Save delta power plots as image? y/n")
save_delta_image = input().lower() == 'y'

print("Save delta power data to CSV? y/n")
save_delta_csv = input().lower() == 'y'

start_time_outputs = time.time()

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Save the filtered data to a new EDF file
if save_filtered_data:
    filtered_file_path = os.path.join(output_dir, f"{input_file_name}_filtered_data.edf")
    raw_corrected.export(filtered_file_path, fmt='edf', overwrite=True)
    print("Filtered data saved to " + filtered_file_path)

# Save the FFT plots as an image
if save_fft_image:
    fft_image_path = os.path.join(output_dir, f"{input_file_name}_fft_plots.png")
    fig_fft.savefig(fft_image_path)
    print("FFT plots saved as " + fft_image_path)

# Save the delta power plots as an image
if save_delta_image:
    delta_image_path = os.path.join(output_dir, f"{input_file_name}_delta_power_plots.png")
    fig_delta.savefig(delta_image_path)
    print("Delta power plots saved as " + delta_image_path)

# Save delta power data to CSV
if save_delta_csv:
    delta_power_df = pd.DataFrame(delta_power_dict)
    csv_file_path = os.path.join(output_dir, f"{input_file_name}_delta_power_data.csv")
    delta_power_df.to_csv(csv_file_path, index=False)
    print(f"Delta power data saved to {csv_file_path}")

end_time_outputs = time.time()
print(f"Time taken to save outputs: {end_time_outputs - start_time_outputs:.2f} seconds")

'''
MULTITHREAD FOR SAVE delta_power CSV; currently DISABLED (unvalidated)

# Save delta power data to CSV
if save_delta_csv:
    delta_power_df = pd.DataFrame(delta_power_dict)

    def save_chunk_to_csv(start, end):
        chunk_df = delta_power_df.iloc[start:end]
        chunk_df.to_csv(os.path.join(output_dir, f"{input_file_name}_delta_power_data_{start}_{end}.csv"), index=False)

    chunk_size = len(delta_power_df) // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(0, len(delta_power_df), chunk_size):
            executor.submit(save_chunk_to_csv, i, min(i + chunk_size, len(delta_power_df)))

    # Combine all chunked CSV files into one
    with open(os.path.join(output_dir, f"{input_file_name}_delta_power_data.csv"), 'w') as outfile:
        for i in range(0, len(delta_power_df), chunk_size):
            chunk_file = os.path.join(output_dir, f"{input_file_name}_delta_power_data_{i}_{min(i + chunk_size, len(delta_power_df))}.csv")
            with open(chunk_file, 'r') as infile:
                outfile.write(infile.read())
            os.remove(chunk_file)
    print(f"Delta power data saved to {os.path.join(output_dir, f'{input_file_name}_delta_power_data.csv')}")

end_time_outputs = time.time()
print(f"Time taken to save outputs: {end_time_outputs - start_time_outputs:.2f} seconds")
'''