import os
import numpy as np
import gc
from wfdb import rdann, rdsamp
from scipy.signal import butter, filtfilt, resample, stft
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def load_ecg_file(filepath, annotation_path):
    ecg_data, metadata = rdsamp(filepath)
    ecg_data = ecg_data[10000:-10000]
    annotations = rdann(annotation_path, 'atr')
    print(f"Loaded ECG data with shape: {ecg_data.shape}")
    return ecg_data, annotations

def clean_ecg_data(ecg_data):
    num_nan = np.isnan(ecg_data).sum()
    num_inf = np.isinf(ecg_data).sum()
    print(f"NaN values before cleaning: {num_nan}, Inf values before cleaning: {num_inf}")
    ecg_data = np.nan_to_num(ecg_data, nan=0.0, posinf=0.0, neginf=0.0)
    return ecg_data

def remove_baseline_wander(ecg_data, sampling_rate, cutoff=0.5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=2, Wn=normal_cutoff, btype='highpass', analog=False)
    ecg_filtered = filtfilt(b, a, ecg_data, axis=0)
    print("Baseline wander removed.")
    return ecg_filtered

def resample_ecg(ecg_data, original_fs, target_fs):
    num_samples = int(ecg_data.shape[0] * target_fs / original_fs)
    ecg_resampled = resample(ecg_data, num_samples, axis=0)
    print(f"Resampled ECG data to {target_fs} Hz with shape: {ecg_resampled.shape}")
    return ecg_resampled

def butter_bandpass_filter(ecg_data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    ecg_filtered = filtfilt(b, a, ecg_data, axis=0)
    print(f"Applied bandpass filter: {lowcut}-{highcut} Hz.")
    return ecg_filtered

def merge_channels(ecg_data):
    if ecg_data.shape[1] != 2:
        print("Expected ECG data with 2 channels for merging.")
        return None
    merged_ecg = np.sqrt(np.sum(ecg_data ** 2, axis=1))
    print(f"Merged ECG channels into single-channel data with shape: {merged_ecg.shape}")
    # take the first 10 s to plot
    """ 
    ecg_data = ecg_data[:2000]
    merged_ecg = merged_ecg[:2000]

    plt.figure(figsize=(12, 6))
    plt.plot(ecg_data[:, 0], label='Lead 1')
    plt.plot(ecg_data[:, 1], label='Lead 2')
    plt.plot(merged_ecg, label='Merged')
    plt.title("ECG Signal")
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
    return merged_ecg

def amplitude_scaling(ecg_data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return ecg_data * scale_factor

def add_noise(ecg_data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, ecg_data.shape)
    return ecg_data + noise

def normalize_ecg_by_person(ecg_data):
    mean = np.mean(ecg_data, axis=0)
    std = np.std(ecg_data, axis=0)
    return (ecg_data - mean) / std    

def segment_ecg(ecg_data, patch_length, stride):
    num_samples = ecg_data.shape[0]
    patches = []
    for start in range(0, num_samples - patch_length + 1, stride):
        end = start + patch_length
        patch = ecg_data[start:end]
        patches.append(patch)
    patches = np.array(patches)
    print(f"Segmented ECG data into {patches.shape[0]} patches.")
    return patches

def load_annotations(annotations, filepath):
    af_intervals = []
    is_afib = False
    start_sample = None
    end_sample = None
    other_ev = None
    start_other_ev = None
    end_other_ev = None
    annotations_fix = annotations.aux_note
    annotations_sample = annotations.sample

    """
    # Define and truncate annotations
    annotations_fix = annotations.aux_note[10000:-10000]
    annotations_sample = annotations.sample[10000:-10000]
    print(annotations_sample)
    annotations_antifix_start = annotations.aux_note[:10000]
    annotations_antifix_end = annotations.aux_note[-10000:]

    for i in range(len(annotations_antifix_start)):
        if '(AFIB' in annotations_antifix_start[i] and not is_afib and not other_ev:
            is_afib = True
        elif is_afib and '(N' in annotations_antifix_start[i]:
            is_afib = False
        elif is_afib and '(VT' in annotations_antifix_start[i] or '(SVTA' in annotations_antifix_start[i]: #or '(B' in annotations_antifix_start[i]: #or '(T' in annotations_antifix_start[i]: or '(AB' in annotations_antifix_start[i]:
            is_afib = False
            other_ev = True
        elif other_ev and '(AFIB' in annotations_antifix_start[i]:
            is_afib = True
            other_ev = False
        elif other_ev and '(N' in annotations_antifix_start[i]:
            is_afib = False
            other_ev = False

    if other_ev:
        annotations_fix =['(VT'] + list(annotations_fix[1:])
    elif is_afib:
        annotations_fix = ['(AFIB'] + list(annotations_fix[1:])
    else:
        annotations_fix = list(annotations_fix)
    
    print(f"Loaded {len(annotations_fix)} annotations.")

    if is_afib:
        print("AFIB detected in the first 10 seconds.")
        is_afib = False
    if other_ev:
        print("Other event detected in the first 10 seconds.")
        other_ev = False
    """
    # Process annotations
    for i in range(len(annotations_fix)):
        aux_note = annotations_fix[i]
        if '(AFIB' in aux_note and not is_afib and not other_ev: 
            is_afib = True
            start_sample = annotations_sample[i]
            print(f"Start sample: {start_sample}")

        elif is_afib and '(N' in aux_note:  
            is_afib = False
            end_sample = annotations_sample[i]
            print(f"End sample: {end_sample}")
            if start_sample is not None and end_sample is not None: 
                af_intervals.append((start_sample, end_sample, 1))
                print(f"Added AF interval: {start_sample}-{end_sample}")
                print(f"Annotation: {aux_note}")
            start_sample = None
            end_sample = None

        elif is_afib and '(VT' in aux_note or '(SVTA' in aux_note: #or '(B' in aux_note: #or '(T' in aux_note:# or '(AB' in aux_note:
            is_afib = False
            end_sample = annotations_sample[i]
            print(f"End sample: {end_sample}")
            if start_sample is not None and end_sample is not None:
                af_intervals.append((start_sample, end_sample, 1))
                print(f"Added AF interval: {start_sample}-{end_sample}")
                print(f"Annotation: {aux_note}")
            start_sample = None
            end_sample = None
            other_ev = True
            start_other_ev = annotations_sample[i]
            print(f"Start other event sample: {start_other_ev}")

        elif other_ev and '(AFIB' in aux_note:
            end_other_ev = annotations_sample[i]
            print(f"End other event sample: {end_other_ev}")
            if start_other_ev is not None and end_other_ev is not None:
                af_intervals.append((start_other_ev, end_other_ev, 2))
                print(f"Added other event interval: {start_other_ev}-{end_other_ev}")
                print(f"Annotation: {aux_note}")
            start_other_ev = None
            end_other_ev = None
            other_ev = False
            is_afib = True
            start_sample = annotations_sample[i]
            print(f"Start sample: {start_sample}")

        elif other_ev and '(N' in aux_note:
            end_other_ev = annotations_sample[i]
            print(f"End other event sample: {end_other_ev}")
            if start_other_ev is not None and end_other_ev is not None:
                af_intervals.append((start_other_ev, end_other_ev, 2))
                print(f"Added other event interval: {start_other_ev}-{end_other_ev}")
                print(f"Annotation: {aux_note}")
            start_other_ev = None
            end_other_ev = None
            other_ev = False
        
        elif other_ev and '(VT' in aux_note or '(SVTA' in aux_note: #or '(B' in aux_note: #or '(T' in aux_note: #or '(AB' in aux_note:
            end_other_ev = annotations_sample[i]
            print(f"End other event sample: {end_other_ev}")
            if start_other_ev is not None and end_other_ev is not None:
                af_intervals.append((start_other_ev, end_other_ev, 2))
                print(f"Added other event interval: {start_other_ev}-{end_other_ev}")
                print(f"Annotation: {aux_note}")
            start_other_ev = annotations_sample[i]
            other_ev = False
            end_other_ev = None

    # Handle case where AFIB interval extends to the last annotation
    if is_afib and start_sample is not None:
        af_intervals.append((start_sample, annotations_sample[-1], 1))
        print(f"Added AF interval: {start_sample}-{annotations_sample[-1]}")
    
    # Merge overlapping or consecutive intervals
    merged_intervals = []
    for interval in af_intervals:
        # Check if interval param = 3:
        if interval.__len__() != 3:
            print(f"Skipping interval {interval} due to incorrect format.")
            continue
        start, end, cat = interval
        if merged_intervals and start <= merged_intervals[-1][1] and merged_intervals[-1][2] == cat:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end, cat))
        else:
            merged_intervals.append((start, end, cat))
    
    print(f"Extracted {len(merged_intervals)} AF intervals.")

    return merged_intervals

def adjust_af_intervals(af_intervals, original_fs, target_fs):
    adjusted_intervals = [
        (int(start * target_fs / original_fs), int(end * target_fs / original_fs))
        for start, end, cat in af_intervals
    ]
    print("Adjusted AF intervals for resampled data.")
    return adjusted_intervals

def assign_labels(patches, af_intervals, patch_length, ecg_start_offset):
    labels = []
    for i, patch in enumerate(patches):
        start = i * patch_length + ecg_start_offset
        end = start + patch_length
        label = 0
        for af_start, af_end, cat in af_intervals:
            if cat == 1:  # AFib
                overlap_start = max(start, af_start)
                overlap_end = min(end, af_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap / patch_length >= 0.5:
                    label = 1
                    break

            elif cat == 2:  # Other event
                overlap_start = max(start, af_start)
                overlap_end = min(end, af_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap / patch_length >= 0.5:
                    label = 2

        labels.append(label)
    labels = np.array(labels)

    # Summarize the labels
    tot_afib = np.sum(labels == 1)
    tot_other = np.sum(labels == 2)
    tot_normal = np.sum(labels == 0)

    print(f"Assigned labels to patches. AFib present in {tot_afib} patches, other events present in {tot_other} patches, normal present in {tot_normal} patches.")
    return labels

def save_patches(patches, labels, save_dir, filename_prefix):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{filename_prefix}_patches.npz")
    np.savez(save_path, patches=patches, labels=labels)
    print(f"Saved patches to {save_path}.")
    

def preprocess_and_save_ecg(filepath, annotation_path, save_dir, original_fs=200, target_fs=200):
    patch_length_seconds = 10
    stride_seconds = 10

    patch_length = int(patch_length_seconds * target_fs)
    stride = int(stride_seconds * target_fs)

    ecg_data, annotations = load_ecg_file(filepath, annotation_path)

    # Get the start offset for ECG data
    ecg_start_offset = 10000

    ecg_data = clean_ecg_data(ecg_data)
    #ecg_data = remove_baseline_wander(ecg_data, original_fs)
    #ecg_data = normalize_ecg_by_person(ecg_data)
    ecg_data = clean_ecg_data(ecg_data)

    ecg_data = butter_bandpass_filter(ecg_data, 0.5, 40, target_fs)
    ecg_data = merge_channels(ecg_data)
    if ecg_data is None:
        return
    ecg_data = clean_ecg_data(ecg_data)

    if original_fs != target_fs:
        ecg_data = resample_ecg(ecg_data, original_fs, target_fs)
        ecg_data = clean_ecg_data(ecg_data)

    af_intervals = load_annotations(annotations, filepath)
    if original_fs != target_fs:
        af_intervals = adjust_af_intervals(af_intervals, original_fs, target_fs)

    patches = segment_ecg(ecg_data, patch_length, stride)
    labels = assign_labels(patches, af_intervals, patch_length, ecg_start_offset)

    filename_prefix = os.path.splitext(os.path.basename(filepath))[0]
    save_patches(patches, labels, save_dir, filename_prefix)

    del ecg_data
    del patches
    gc.collect()


# Example usage
data_folder = './data/ATA/'
files = [f for f in os.listdir(data_folder) if f.endswith('.dat')]
#shuffle
#files = np.random.permutation(files)

for x in range(260):
    if x <= 200:
        file = files[x]
        save_directory = './preprocessed_data_ata_ecg_1c/'
        filepath = os.path.join(data_folder, file)
        filepath = filepath[:-4]
        annotation_path = filepath
        print(f"\nProcessing file: {file} for training")
        preprocess_and_save_ecg(filepath, annotation_path, save_directory, original_fs=200, target_fs=200)
        print('-'*50)
    else:
        file = files[x]
        save_directory = './preprocessed_data_ata_ecg_test_1c/'
        filepath = os.path.join(data_folder, file)
        filepath = filepath[:-4]
        annotation_path = filepath
        print(f"\nProcessing file: {file} for testing")
        preprocess_and_save_ecg(filepath, annotation_path, save_directory, original_fs=200, target_fs=200)
        print('-'*50)

"""
data_folder = './data/ATA_test/'
save_directory = './preprocessed_data_ata_ecg_test_1c/'
files = [f for f in os.listdir(data_folder) if f.endswith('.dat')]

for x in range(25):
    file = files[x]
    filepath = os.path.join(data_folder, file)
    filepath = filepath[:-4]
    annotation_path = filepath
    print(f"\nProcessing file: {file}")
    preprocess_and_save_ecg(filepath, annotation_path, save_directory, original_fs=200, target_fs=200)
"""
