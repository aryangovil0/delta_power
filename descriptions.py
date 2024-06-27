import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Input the path to the EDF file and the output directory
file_path = input("Enter the path to the EDF file: ")
input_file_name = os.path.splitext(os.path.basename(file_path))[0]

# Load the EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

# Extract data information
info = raw.info
sampling_rate = info['sfreq']
channel_names = info['ch_names']

# Extract header information
meas_date = info['meas_date']
nchan = info['nchan']
subject_info = info.get('subject_info', 'No subject information available')

# Print the header information
print("Header Information:")
print("-------------------")
print(f"Measurement Date: {meas_date}")
print(f"Subject Information: {subject_info}")
duration_seconds = raw.times[-1] - raw.times[0]
duration_minutes = duration_seconds / 60
duration_hours = duration_minutes / 60
print(f"Duration of the recording: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes, {duration_hours:.2f} hours)")
print(f"Number of Channels: {nchan}")
print(f"Sampling Rate: {sampling_rate} Hz")
print("Channel Names:")
for i, name in enumerate(channel_names):
    print(f"{i + 1}. {name}")

# Plot the raw data
raw.plot(n_channels=len(raw.ch_names), scalings='auto', title='Raw EEG Data', show=True, block=True)
