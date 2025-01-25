import numpy as np
import glob
import os
from itertools import combinations

def initial_split(input_dir, temp_dir, group_count=16, batch_size=500):
    """ First pass to split data into grouped shuffled files. """
    os.makedirs(temp_dir, exist_ok=True)
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    buffer_data = [[] for _ in range(group_count)]  # Create buffers for each group
    batch_count = 0

    print(f"Starting initial split with {len(npz_files)} files...")

    # Distribute files into groups
    for idx, npz_file in enumerate(npz_files):
        with np.load(npz_file) as data:
            spectrograms = data["spectrograms"]
            labels = data["labels"]
            
            # Define the expected shape based on the first valid patch
            valid_patches = [p for p in spectrograms if p.shape == spectrograms[0].shape]
            expected_shape = valid_patches[0].shape if valid_patches else None

            for i, (spec, label) in enumerate(zip(spectrograms, labels)):
                # Skip if patch does not match expected shape
                if expected_shape and spec.shape != expected_shape:
                    print(f"Skipping patch with inconsistent shape {spec.shape} in file {npz_file}")
                    continue

                buffer_data[(idx + i) % group_count].append((spec, label))
                
                # Save data if it reaches batch size for each group
                if len(buffer_data[(idx + i) % group_count]) >= batch_size:
                    save_temp_file(buffer_data[(idx + i) % group_count], temp_dir, batch_count)
                    print(f"Saved batch {batch_count} for group {(idx + i) % group_count}.")
                    buffer_data[(idx + i) % group_count] = []  # Clear buffer after saving
                    batch_count += 1

    # Save any remaining data in each buffer group
    for group_idx, group in enumerate(buffer_data):
        if group:
            save_temp_file(group, temp_dir, batch_count)
            print(f"Saved final batch {batch_count} for group {group_idx}.")
            batch_count += 1

    print("Initial split complete.")

def save_temp_file(buffer_data, temp_dir, batch_count):
    """ Helper function to save buffered data to a temporary npz file. """
    # Filter out inconsistent data before saving
    spectrograms, labels = zip(*[(p, l) for p, l in buffer_data if p.shape == buffer_data[0][0].shape])
    spectrograms = np.array(spectrograms)  # Convert to a homogeneous array
    labels = np.array(labels)    # Convert labels similarly
    temp_file = os.path.join(temp_dir, f"temp_batch_{batch_count}.npz")
    np.savez_compressed(temp_file, spectrograms=spectrograms, labels=labels)


def final_shuffle(temp_dir, output_dir, group_count=16, max_patches_per_file=1000):
    """ Shuffle data between pairs of grouped datasets and save as 16 final shuffled files, with max 1000 patches each. """
    os.makedirs(output_dir, exist_ok=True)
    temp_files = glob.glob(os.path.join(temp_dir, "*.npz"))
    grouped_data = [[] for _ in range(group_count)]

    print(f"Loading data from {len(temp_files)} temp files...")

    # Load all data into separate lists for each group
    for idx, temp_file in enumerate(temp_files):
        with np.load(temp_file) as data:
            spectrograms = data["spectrograms"]
            labels = data["labels"]
            grouped_data[idx % group_count].extend(list(zip(spectrograms, labels)))
        print(f"Loaded data from {temp_file} into group {idx % group_count}.")

    print("Data loaded, starting pairwise shuffling...")

    # Pairwise shuffle between all pairs of groups
    for pair_count, (group1, group2) in enumerate(combinations(range(group_count), 2)):
        combined_data = grouped_data[group1] + grouped_data[group2]
        np.random.shuffle(combined_data)
        
        # Redistribute shuffled data back to both groups
        half = len(combined_data) // 2
        grouped_data[group1] = combined_data[:half]
        grouped_data[group2] = combined_data[half:]
        print(f"Shuffled pair {pair_count + 1}: group {group1} with group {group2}.")

    print("Pairwise shuffling complete, saving final shuffled groups...")

    # Save each group as separate files with a maximum of 1000 patches each
    for group_idx, group_data in enumerate(grouped_data):
        np.random.shuffle(group_data)
        total_files = (len(group_data) + max_patches_per_file - 1) // max_patches_per_file  # Calculate required number of files
        for file_idx in range(total_files):
            start = file_idx * max_patches_per_file
            end = min((file_idx + 1) * max_patches_per_file, len(group_data))
            file_data = group_data[start:end]
            save_final_file(file_data, output_dir, group_idx, file_idx)
            print(f"Saved file {file_idx} for group {group_idx} with {len(file_data)} patches.")

    print("Final shuffle and save complete.")

def save_final_file(buffer_data, output_dir, group_idx, file_idx):
    """ Helper function to save each final shuffled group to a separate npz file with max patches per file. """
    spectrograms, labels = zip(*buffer_data)
    final_file = os.path.join(output_dir, f"shuffled_group_{group_idx}_file_{file_idx}.npz")
    np.savez_compressed(final_file, spectrograms=spectrograms, labels=labels)

# Directories for input, temporary, and final output data
input_directory = "./preprocessed_data_ata_ecg_spec"
temp_directory = "./temp_dir_spec"
output_directory = "./shuffled_preprocessed_data_ata_ecg_spec"

# Execute the functions to achieve the full shuffle across 16 final datasets, each saved in files with max 1000 patches
initial_split(input_directory, temp_directory, group_count=16, batch_size=150000)
final_shuffle(temp_directory, output_directory, group_count=16, max_patches_per_file=10000)

# Directories for input, temporary, and final output data
input_directory = "./preprocessed_data_ata_ecg_test_spec"
temp_directory = "./temp_dir_spec_2"
output_directory = "./shuffled_preprocessed_data_ata_ecg_test_spec"

# Execute the functions to achieve the full shuffle across 16 final datasets, each saved in files with max 1000 patches
initial_split(input_directory, temp_directory, group_count=16, batch_size=150000)
final_shuffle(temp_directory, output_directory, group_count=16, max_patches_per_file=10000)
