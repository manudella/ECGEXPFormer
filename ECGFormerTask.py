import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast, GradScaler
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset
from sklearn.utils import resample
import random
from ECGTransformer import ECGFormerForClassification, ECGFormerForSpecClassification
from ECGDataset import ECGDataset

# Device configuration
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Dataset and DataLoader initialization
    model_dim = 128
    num_heads = 4
    num_encoder_layers = 2
    num_classes = 2
    epochs = 200
    log_interval = 100
    batch_size = 64
    test_size = 0.2
    learning_rate = 0.0005 
    dropout_rate = 0.2
    patch_length = 2000

    folder_path = "./shuffled_preprocessed_data_ata_ecg_1c"
    folder = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    random.shuffle(folder)
    folder = folder
    json_file_path = 'random_files.json'
    with open(json_file_path, 'w') as json_file:
            json.dump(folder_path, json_file)
    chunk_size = 1000

    # Batch size for DataLoader initialization
    initial_batch_size = 1024

    # Create DataLoader
    dataset_start_time = time.time()

    # Load training dataset
    train_dataset = ECGDataset(folder)
    data = torch.tensor(train_dataset.data, dtype=torch.float32)
    labels = torch.tensor(train_dataset.labels, dtype=torch.long)

    # Filter out class '2' for binary classification
    valid_indices = labels != 2
    data = data[valid_indices]
    labels = labels[valid_indices]

    print("Unique labels after filtering:", torch.unique(labels))

    dataset_end_time = time.time()
    print('Dataset loaded')
    print(f'Time taken to load dataset: {dataset_end_time - dataset_start_time:.2f}s, with batch size: {initial_batch_size}')

    # Load test dataset
    test_folder_path = "./shuffled_preprocessed_data_ata_ecg_test_1c"
    test_folder = []

    while len(test_folder) < test_size * len(folder):
        random_file = random.choice(os.listdir(test_folder_path))
        if random_file not in test_folder:
            test_folder.append(os.path.join(test_folder_path, random_file))

    random.shuffle(test_folder)
    test_dataset = ECGDataset(test_folder)

    # Filter out class '2' in test dataset
    test_data = torch.tensor(test_dataset.data, dtype=torch.float32)
    test_labels = torch.tensor(test_dataset.labels, dtype=torch.long)
    valid_test_indices = test_labels != 2
    test_data = test_data[valid_test_indices]
    test_labels = test_labels[valid_test_indices]

    print(f"Using original training set")
    """
    # Downsampling function
    def downsample_indices(indices, labels):
        minority_class = 1
        majority_class = 0

        # Get indices for each class
        minority_indices = indices[labels[indices] == minority_class]
        majority_indices = indices[labels[indices] == majority_class]

        # Downsample majority class to match minority size
        majority_indices_downsampled = resample(
            majority_indices,
            replace=False,
            n_samples=len(minority_indices)*2,
            random_state=42
        )
        # Combine and shuffle
        downsampled_indices = torch.cat([minority_indices, majority_indices_downsampled])
        downsampled_indices = downsampled_indices[torch.randperm(len(downsampled_indices))]
        return downsampled_indices

    # Generate indices for training dataset
    train_indices = torch.arange(len(data))

    # Downsample the training set
    downsampled_indices = downsample_indices(train_indices, labels)
    train_dataset = Subset(train_dataset, downsampled_indices.tolist())
    data = data[downsampled_indices]
    labels = labels[downsampled_indices]

    print(f"Downsampled training set size: {len(train_dataset)}")
    """

    # Display dataset information
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    train_size = len(train_dataset)
    # Define parameters for training

    model = ECGFormerForClassification(
        model_dim=model_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        input_length=patch_length
    ).to(device)

    # Print the parameters
    print(f"Model dimension: {model_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Number of encoder layers: {num_encoder_layers}")
    
    print(f"Test set size: {len(test_dataset)}")
    print("Model created")

    # print model summary
    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Optimizer and loss function
    # Update optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        #weight_decay=0.01  # L2 regularization
    )
    criterion = nn.CrossEntropyLoss()
    # After optimizer definition
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_dataset) // batch_size,
        pct_start=0.3,
        div_factor=25
    )

    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            
        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
            
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7)

    # Training and Evaluation
    train_losses = []
    test_losses = []
    test_accuracies = []
    all_labels = []
    all_predictions = []
    epoch_loss = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        correct = 0
        total = 0
        model.train()
        batch_log_start_time = time.time()
        sensitivity_dividend = 0
        sensitivity_divisor = 0
        specificity_dividend = 0
        specificity_divisor = 0
        sensitivity_log = 0
        specificity_log = 0
        
        batch_total = len(train_dataset) // batch_size
        x = 0
        for batch_idx in range(0, len(train_dataset), batch_size):
            batch_start_time = time.time()
            data_chunk = data[batch_idx:batch_idx + batch_size].to(device)
            label_chunk = labels[batch_idx:batch_idx + batch_size].to(device)
            # Check if the data_chunk is compatible with the model
        
            outputs = model(data_chunk)
            if outputs is None:
                print(f'[INFO] Skipping batch {batch_idx} due to incompatible tensor size.')
                continue

            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, num_classes)
            label_chunk = label_chunk.view(-1)
            
            classification_loss = criterion(outputs, label_chunk)
            loss = classification_loss # vq_loss + classification_loss

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            # After loss.backward() but before optimizer.step():
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Optimizer step
            # Inside training loop, after optimizer.step():
            #scheduler.step()
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label_chunk.size(0)
            correct += predicted.eq(label_chunk).sum().item()
            
            specificity_dividend += confusion_matrix(label_chunk.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])[0, 0] 
            specificity_divisor += np.sum(label_chunk.cpu().numpy() == 0)
            sensitivity_dividend += confusion_matrix(label_chunk.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])[1, 1]
            sensitivity_divisor += np.sum(label_chunk.cpu().numpy() == 1)
            sensitivity_log = f'{sensitivity_dividend}/{sensitivity_divisor}'
            specificity_log = f'{specificity_dividend}/{specificity_divisor}'
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # print and rewrite on the same line afer deleting the print
            print(' '*100, end='\r')
            #print(f'Epoch {epoch}- Batch {batch_idx+(int(int(train_size/batch_size)/dataset_divisor))}/{int(train_size/batch_size)} - Loss: {loss.item()}, Accuracy: {100. * correct / total}%, Time: {batch_time:.2f}s', end='\r')       
            #print(f'Epoch {epoch}- Batch {x}/{int(train_size/batch_size)} - Loss: {loss.item()}, Accuracy: {100. * correct / total}%, Time: {batch_time:.2f}s', end='\r')
            print(f'Epoch {epoch}- Batch {x}/{int(train_size/batch_size)} '+'['+'='*int((x+1)/batch_total*25)+'>'+' '*(25-int((x+1)/batch_total*25))+ f'] Accuracy: {round(100. * correct / total, 4)}%, Precision: {sensitivity_log}, Recall: {specificity_log}, Loss: {epoch_loss}', end='\r')   
            """
            if x % log_interval == 0:
                # Log the specificity, and sensitivity over log interval
                sensitivity_log = f'{sensitivity_dividend}/{sensitivity_divisor}'
                specificity_log = f'{specificity_dividend}/{specificity_divisor}'
                batch_log_end_time = time.time()
                batch_log_time = batch_log_end_time - batch_log_start_time
                print(f'--------------> Epoch {epoch}- Batch {int(x)}/{int(train_size/batch_size)} - Loss: {loss.item()}, Accuracy: {100. * correct / total}%, Sensitivity: {sensitivity_log}, Specificity: {specificity_log}, Time({log_interval}): {batch_log_time:.2f}s')
                sensitivity_dividend = 0
                sensitivity_divisor = 0
                specificity_dividend = 0
                specificity_divisor = 0
                batch_log_start_time = time.time()
            """
            x += 1
        print(f'Epoch {epoch}- Batch {x}/{int(train_size/batch_size)} '+'['+'='*int((x+1)/batch_total*25)+'>'+' '*(25-int((x+1)/batch_total*25))+ f'] Accuracy: {round(100. * correct / total, 4)}%, Precision: {sensitivity_log}, Recall: {specificity_log}, Loss: {epoch_loss}')     

        epoch_loss /= len(train_dataset)
        epoch_end_time = time.time()
        print(f'Epoch {epoch}')
        print(f'Time for epoch {epoch}: {epoch_end_time - epoch_start_time:.2f}s')

        # Evaluation Phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_labels = []
        all_predictions= []
        sensitivity_dividend = 0
        sensitivity_divisor = 0
        specificity_dividend = 0
        specificity_divisor = 0
        with torch.no_grad():
            #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
            y=0
            for batch_idx in range(0, len(test_dataset), batch_size):
                # Forward pass
                #outputs, vq_loss = model(data_chunk)
                data_chunk = test_data[batch_idx:batch_idx+batch_size].to(device)
                label_chunk = test_labels[batch_idx:batch_idx+batch_size].to(device)
                outputs = model(data_chunk)
                if outputs is None:
                    print(f'[INFO] Skipping batch {batch_idx} due to incompatible tensor size.')
                    continue

                # Reshape outputs and labels for loss calculation
                outputs = outputs.view(-1, num_classes)
                label_chunk = label_chunk.view(-1)
                label_chunk = label_chunk.long()

                # Calculate the loss
                classification_loss = criterion(outputs, label_chunk)
                loss = classification_loss #vq_loss + classification_loss

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += label_chunk.size(0)
                test_correct += predicted.eq(label_chunk).sum().item()
                all_labels.extend(label_chunk.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                sensitivity_dividend += confusion_matrix(label_chunk.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])[1, 1]
                sensitivity_divisor += np.sum(label_chunk.cpu().numpy() == 1)
                specificity_dividend += confusion_matrix(label_chunk.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])[0, 0]
                specificity_divisor += np.sum(label_chunk.cpu().numpy() == 0)
                
                print(f'Test Batch {y}/{int(len(test_dataset)/batch_size)} - Loss: {test_loss/len(test_dataset)}, Accuracy: {100. * test_correct / test_total}%, Sensitivity: {sensitivity_dividend}/{sensitivity_divisor}, Specificity: {specificity_dividend}/{specificity_divisor}', end='\r')
                y += 1

            print(f'Test Batch {y}/{int(len(test_dataset)/batch_size)} - Loss: {test_loss/len(test_dataset)}, Accuracy: {100. * test_correct / test_total}%, Sensitivity: {sensitivity_dividend}/{sensitivity_divisor}, Specificity: {specificity_dividend}/{specificity_divisor}')
            test_loss /= len(test_dataset)
            test_acc = 100. * test_correct / test_total
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}%, Sensitivity: {sensitivity_dividend}/{sensitivity_divisor}, Specificity: {specificity_dividend}/{specificity_divisor}')
            # Save the model checkpoint
            if epoch == 1 or epoch == 2 or epoch == 3 or epoch == 5 or epoch % 10 == 0:
                checkpoint_path = f"./checkpoints_ata/ecgformer_ata_model_epoch_{epoch}.pth"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                conf_mat = confusion_matrix(all_labels, all_predictions, labels=[0, 1])

                plt.figure(figsize=(10, 7))
                sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')

                # Save the plot to a file
                # crate a folder to save the confusion matrix
                os.makedirs(f'./confusion_matrix_ata_tran', exist_ok=True)
                plt.savefig(f'./confusion_matrix_ata_tran/confusion_matrix_epoch_{epoch}.png')

                # Close the plot to free up memory
                plt.close()    
            if early_stopping(test_loss):
                print("Early stopping triggered")
                break 