import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import gc
import time
from torch.utils.data import DataLoader, Subset
from sklearn.utils import resample

from ECGDataset import ECGDatasetSpec
from ECGCNN import ECGCNNForClassification, CNNLSTMForClassification

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    test_size = 0.2
    # Load dataset
    folder_path = "./shuffled_preprocessed_data_ata_ecg_spec"
    folder = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    folder = folder[:50]
    random.shuffle(folder)

    test_folder_path = "./shuffled_preprocessed_data_ata_ecg_test_spec"
    test_folder = []
    while len(test_folder) < test_size * len(folder):
        random_file = random.choice(os.listdir(test_folder_path))
        if random_file not in test_folder:
            random_file = os.path.join(test_folder_path, random_file)
            test_folder.append(random_file)
    
    random.shuffle(test_folder)
    train_dataset = ECGDatasetSpec(folder)
    data = torch.tensor(train_dataset.data, dtype=torch.float32)
    labels = torch.tensor(train_dataset.labels, dtype=torch.long)

    # Filter out class '2' for binary classification
    valid_indices = labels != 2
    data = data[valid_indices]
    labels = labels[valid_indices]

    print("Unique labels after filtering:", torch.unique(labels))


    test_dataset = ECGDatasetSpec(test_folder)
    test_data = torch.tensor(test_dataset.data, dtype=torch.float32)
    test_labels = torch.tensor(test_dataset.labels, dtype=torch.long)
    valid_test_indices = test_labels != 2
    test_data = test_data[valid_test_indices]
    test_labels = test_labels[valid_test_indices]
  
    # Model initialization
    model = ECGCNNForClassification(
    input_channels=1,
    num_classes=2,
    block_config=[
        (32, 7, 1, 0.2),  
        (64, 5, 2, 0.2),   
        (128, 3, 2, 0.2),
        (256, 3, 2, 0.2),
    ],
    use_residual=True,
    global_pooling='avg').to(device)


    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Define optimizer and loss function
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
 
    # Training and evaluation parameters
    epochs = 100
    batch_size = 64
    train_size = len(train_dataset)
    num_classes = 2

    # Training and Evaluation
    train_losses = []
    test_losses = []
    test_accuracies = []
    all_labels = []
    all_predictions = []
    epoch_loss = 0

    def initialize_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    model.apply(initialize_weights)
    
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
            
            if data_chunk.size(0) == 0:
                print(f"Skipping empty batch at index {batch_idx}")
                continue

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
            optimizer.step()  # Optimizer step

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label_chunk.size(0)
            correct += predicted.eq(label_chunk).sum().item()
            
            specificity_dividend += confusion_matrix(label_chunk.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])[0, 0] 
            specificity_divisor += np.sum(label_chunk.cpu().numpy() == 0)
            sensitivity_dividend += confusion_matrix(label_chunk.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])[1, 1]
            sensitivity_divisor += np.sum(label_chunk.cpu().numpy() == 1)
            sensitivity_log = (sensitivity_dividend/sensitivity_divisor)*100
            specificity_log = (specificity_dividend/specificity_divisor)*100
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # print and rewrite on the same line afer deleting the print
            print(' '*100, end='\r')
            #print(f'Epoch {epoch}- Batch {batch_idx+(int(int(train_size/batch_size)/dataset_divisor))}/{int(train_size/batch_size)} - Loss: {loss.item()}, Accuracy: {100. * correct / total}%, Time: {batch_time:.2f}s', end='\r')       
            #print(f'Epoch {epoch}- Batch {x}/{int(train_size/batch_size)} - Loss: {loss.item()}, Accuracy: {100. * correct / total}%, Time: {batch_time:.2f}s', end='\r')
            print(f'Epoch {epoch}- Batch {x}/{int(train_size/batch_size)} '+'['+'='*int((x+1)/batch_total*25)+'>'+' '*(25-int((x+1)/batch_total*25))+ f'] Accuracy: {round(100. * correct / total, 4)}%, Precision: {sensitivity_log}%, Recall: {specificity_log}%', end='\r')     
    
            x += 1
        print(f'Epoch {epoch}- Batch {x}/{int(train_size/batch_size)} '+'['+'='*int((x+1)/batch_total*25)+'>'+' '*(25-int((x+1)/batch_total*25))+ f'] Accuracy: {round(100. * correct / total, 4)}%, Precision: {sensitivity_log}%, Recall: {specificity_log}%')     

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

                if data_chunk.size(0) == 0:
                    print(f"Skipping empty batch at index {batch_idx}")
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
                
                print(f'Test Batch {y}/{int(len(test_dataset)/batch_size)} - Loss: {test_loss/len(test_dataset)}, Accuracy: {100. * test_correct / test_total}%, Precision: {(sensitivity_dividend/sensitivity_divisor)*100}%, Recall: {(specificity_dividend/specificity_divisor)}%', end='\r')
                y += 1

            print(f'Test Batch {y}/{int(len(test_dataset)/batch_size)} - Loss: {test_loss/len(test_dataset)}, Accuracy: {100. * test_correct / test_total}%, Precision: {(sensitivity_dividend/sensitivity_divisor)*100}%, Recall: {(specificity_dividend/specificity_divisor)*100}%')
            test_loss /= len(test_dataset)
            test_acc = 100. * test_correct / test_total
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}%, Precision: {(sensitivity_dividend/sensitivity_divisor)*100}%, Recall: {(specificity_dividend/specificity_divisor)*100}%')
            
            # Save the model checkpoint
            if epoch == 1 or epoch == 2 or epoch == 3 or epoch == 5 or epoch % 10 == 0:
                checkpoint_path = f"./checkpoints_ata_CNN/ecgCNN_ata_model_epoch_{epoch}.pth"
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
                os.makedirs(f'./confusion_matrix_ata_CNN', exist_ok=True)
                plt.savefig(f'./confusion_matrix_ata_CNN/confusion_matrix_epoch_{epoch}.png')

                # Close the plot to free up memory
                plt.close()
