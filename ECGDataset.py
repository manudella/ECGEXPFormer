import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDatasetSpec(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = folder_path
        self.data, self.labels = self._load_all_data()

    def _load_all_data(self):
        all_data = []
        all_labels = []
        files_processed = 0
        total_files = len(self.file_list)
        
        for filepath in self.file_list:
            with np.load(filepath) as datas:
                data = datas['spectrograms']
                labels = datas['labels']
                
                if len(labels) != data.shape[0]:
                    print(f"[WARNING] Skipping file due to mismatch: {filepath}")
                    continue
                
                all_data.append(data)
                all_labels.append(labels)
                
                files_processed += 1
                print(f"[INFO] Files processed: {files_processed}/{total_files}", end='\r')
        
        # Concatenate all data and labels into single arrays
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.data[idx]
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(patch).any() or np.isinf(patch).any():
            raise ValueError(f"Data contains NaNs or Infs at index {idx}")

        patch = torch.tensor(patch, dtype=torch.float32)
        label = self.labels[idx]
        return patch, label

class ECGDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = folder_path
        self.data, self.labels = self._load_all_data()

    def _load_all_data(self):
        all_data = []
        all_labels = []
        files_processed = 0
        total_files = len(self.file_list)
        
        for filepath in self.file_list:
            with np.load(filepath) as datas:
                data = datas['patches']
                labels = datas['labels']
                
                if len(labels) != data.shape[0]:
                    print(f"[WARNING] Skipping file due to mismatch: {filepath}")
                    continue
                
                all_data.append(data)
                all_labels.append(labels)
                
                files_processed += 1
                print(f"[INFO] Files processed: {files_processed}/{total_files}", end='\r')
        
        # Concatenate all data and labels into single arrays
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.data[idx]
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(patch).any() or np.isinf(patch).any():
            raise ValueError(f"Data contains NaNs or Infs at index {idx}")

        patch = torch.tensor(patch, dtype=torch.float32)
        label = self.labels[idx]
        return patch, label