import shap
import torch
import random
import os
from ECGDataset import ECGDataset
from ECGTransformer import ECGFormerForClassification
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device}")

# Define the model and load the weights
model_dim = 256
num_heads = 4
num_encoder_layers = 2
num_classes = 2
log_interval = 100
dropout_rate = 0.2
patch_length = 2000

model = ECGFormerForClassification(
        model_dim=model_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        input_length=patch_length
    ).to(device)
print("Model initialized.")

checkpoint_path = "./checkpoints_ata/ecgformer_ata_model_epoch_30.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)  # Fix weights loading
model.load_state_dict(checkpoint)
print(f"Model weights loaded from {checkpoint_path}")

# Load your data
test_folder_path = "./shuffled_preprocessed_data_ata_ecg_1c"
test_folder = []

# Randomly select files for testing
for i in range(len(os.listdir(test_folder_path))):
    random_file = random.choice(os.listdir(test_folder_path))
    if random_file not in test_folder:
        test_folder.append(os.path.join(test_folder_path, random_file))

random.shuffle(test_folder)
test_folder = test_folder[:20]

print(f"Test folder shuffled, {len(test_folder)} files loaded.")

test_dataset = ECGDataset(test_folder)
print(f"Test dataset created with {len(test_dataset.data)} samples.")

test_data = torch.tensor(test_dataset.data, dtype=torch.float32)
test_labels = torch.tensor(test_dataset.labels, dtype=torch.long)

# Filter valid test data (exclude label == 2)
valid_test_indices = test_labels != 2
test_data = test_data[valid_test_indices]
test_labels = test_labels[valid_test_indices]
print(f"Filtered data to {len(test_data)} samples (labels != 2).")

# Set the model to evaluation mode
model.eval()
print("Model set to evaluation mode.")

# Select a subset of your data for SHAP analysis (e.g., 100 samples)
sample_data = test_data[:100].to(device)
print(f"Selected {sample_data.shape[0]} samples for SHAP analysis.")
print(f"Shape of sample data: {sample_data.shape}")

# Wrap your model with a function that returns logits
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

wrapped_model = WrappedModel(model)
print("Wrapped model created.")

# Initialize the SHAP explainer
explainer = shap.DeepExplainer(wrapped_model, sample_data)
print("SHAP explainer initialized.")

# Select new data to explain (e.g., another 10 samples)
test_sample = test_data[100:2000].to(device)
print(f"Selected {test_sample.shape[0]} samples for explanation.")
print(f"Shape of test sample: {test_sample.shape}")

# Calculate SHAP values
shap_values = explainer.shap_values(test_sample, check_additivity=False)
print(f"SHAP values calculated for {test_sample.shape[0]} samples.")

# Example ECG signal (replace with your ECG signal list)
ecg_signal = test_sample[0].cpu().numpy()  # Move to CPU and convert to NumPy
original_class = test_labels[100].item()  # Get the true label for the first sample

# Get the model's prediction for the test sample
with torch.no_grad():
    predictions = model(test_sample)
predicted_class = torch.argmax(predictions[0]).item()

print(f"Original Class: {original_class}")
print(f"Predicted Class: {predicted_class}")

# Print the shape of SHAP values for debugging
print(f"SHAP values shape: {np.array(shap_values).shape}")

# Define how many images you want to save for each class combination (predicted and original class)
images_per_group = 5  # Set the number of images you want to save for each group

# Define the folder where the images will be saved
output_folder = "./shap_images_2"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Loop over the four combinations of predicted and original classes
for predicted_class_value in [0, 1]:
    for original_class_value in [0, 1]:
        # Select the samples where both predicted and original classes match the current values
        selected_indices = np.where((test_labels == original_class_value) & (predicted_class == predicted_class_value))[0]

        if len(selected_indices) == 0:
            print(f"No samples found for Predicted Class: {predicted_class_value}, Original Class: {original_class_value}. Skipping.")
            continue

        # Limit to `images_per_group` number of images from the selected indices
        selected_indices = selected_indices[:images_per_group]

        # Select a subset of data for SHAP analysis based on the indices
        selected_sample_data = test_data[selected_indices].to(device)

        print(f"Selected {selected_sample_data.shape[0]} samples for Predicted Class {predicted_class_value}, Original Class {original_class_value}.")
        
        # Calculate SHAP values for the selected samples
        shap_values = explainer.shap_values(selected_sample_data, check_additivity=False)
        print(f"SHAP values calculated for {selected_sample_data.shape[0]} samples.")

        # Loop through selected samples to generate individual SHAP plots
        for idx, sample in enumerate(selected_sample_data):
            ecg_signal = sample.cpu().numpy()  # Move to CPU and convert to NumPy
            original_class = test_labels[selected_indices[idx]].item()  # Get the true label for the sample
            predicted_class = torch.argmax(predictions[selected_indices[idx]]).item()  # Get the predicted class

            # Check if shap_values for the predicted class exist
            if len(shap_values) > predicted_class:
                shap_values_signal = shap_values[idx][:, predicted_class]  # Extract SHAP values for the current sample and predicted class
            else:
                print(f"Warning: No SHAP values found for predicted class {predicted_class} and sample {idx}. Skipping.")
                continue

            # Check lengths
            if len(shap_values_signal) != len(ecg_signal):
                raise ValueError(f"Mismatch: SHAP values length ({len(shap_values_signal)}) != ECG signal length ({len(ecg_signal)})")

            my_colors = ["blue", "grey", "red"]
            my_cmap = LinearSegmentedColormap.from_list("BlackDiverging", my_colors, N=256)

            # Suppose shap_values_signal is your SHAP array
            min_val = shap_values_signal.min()
            max_val = shap_values_signal.max()
            max_abs_val = max(abs(min_val), abs(max_val))

            # Make the colormap range symmetric about 0
            norm = plt.Normalize(vmin=-max_abs_val, vmax=max_abs_val)

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(shap_values_signal))

            for i in range(len(x) - 1):
                ax.plot(
                    x[i : i + 2],
                    ecg_signal[i : i + 2],
                    color=my_cmap(norm(shap_values_signal[i])),
                    linewidth=2,
                )

            # Create the colorbar using your custom colormap
            sm = plt.cm.ScalarMappable(norm=norm, cmap=my_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label="SHAP Value")

            ax.set_title(f"ECG Signal Colored by SHAP Values\nPredicted: {predicted_class}, Original: {original_class}")
            ax.set_xlabel("Time (index)")
            ax.set_ylabel("Amplitude")

            # Save the plot
            filename = f"shap_values_{predicted_class}_{original_class}_{idx}.png"
            plt.savefig(os.path.join(output_folder, filename), dpi=300)
            plt.close()


        print(f"Saved {images_per_group} images for Predicted Class {predicted_class_value}, Original Class {original_class_value}.")








