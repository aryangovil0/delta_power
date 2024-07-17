'''
Aryan Govil

'''

import mne
import os
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

channels_of_interest = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 
    'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 
    'P8', 'O1', 'O2'
]

# Function to apply bandpass filter
def bandpass_filter(raw, low_freq, high_freq):
    raw.filter(low_freq, high_freq, fir_design='firwin')
    return raw

# Function to apply ICA and remove artifacts
def apply_ica(raw, reference_channels, artifact_type):
    ica = ICA(n_components=19, random_state=97, max_iter=800)
    ica.fit(raw)
    
    if artifact_type == 'eog':
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=reference_channels)
        ica.exclude = eog_indices
    elif artifact_type == 'ecg':
        ecg_epochs = create_ecg_epochs(raw, ch_name=reference_channels)
        ecg_indices, ecg_scores = ica.find_bads_ecg(ecg_epochs)
        ica.exclude = ecg_indices
    elif artifact_type == 'emg':
        emg_indices, emg_scores = ica.find_bads_eog(raw, ch_name=reference_channels)  # Using EOG method for EMG channels for simplicity
        ica.exclude = emg_indices

    ica.apply(raw)
    return raw

# Main processing function
def process_eeg(file_path, output_folder):
    # Load the .edf file
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Select channels of interest
    raw.pick_channels(channels_of_interest)

    # Print descriptives about the data
    subject_info = raw.info['subject_info'] if 'subject_info' in raw.info else 'N/A'
    sampling_rate = raw.info['sfreq']
    duration_seconds = raw.times[-1] - raw.times[0]
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60
    print(f"Subject Information: {subject_info}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Duration of the recording: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes, {duration_hours:.2f} hours)")

    # Apply bandpass filter
    raw = bandpass_filter(raw, 0.1, 20.0)

    # Perform ICA for eye blinks
    raw = apply_ica(raw, ['E1', 'E2'], 'EOG')
    print('ICA complete for eye blink removal')

    # Perform ICA for cardiogenic artifacts
    raw = apply_ica(raw, ['ECG 1', 'ECG 2', 'ECG 3'], 'ECG')
    print('ICA complete for cardiogenic artifact removal')

    # Perform ICA for arm movements EXAMPLE, we can add or remove whatever for EMG signals
    raw = apply_ica(raw, ['Arm/L', 'Arm/R', 'Leg/L', 'Leg/R'], 'EMG')
    print('ICA complete for EMG recorded movement removal')

    # Save the cleaned data to a new .edf file
    base_name = os.path.basename(file_path)
    new_file_path = os.path.join(output_folder, base_name.replace('.edf', '_cleaned.edf'))
    mne.export.export_raw(new_file_path, raw, fmt='edf', overwrite=True)

    print(f'Cleaned data saved to {new_file_path}')

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing .edf files: ")
    output_folder = os.path.join(folder_path, 'cleaned')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    edf_files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]

    for edf_file in edf_files:
        file_path = os.path.join(folder_path, edf_file)
        process_eeg(file_path, output_folder)
