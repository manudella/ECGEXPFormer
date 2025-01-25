# ECG-Based Atrial Fibrillation (AF) Detection with Deep Learning and SHAP Explainability

This repository provides a concise pipeline for detecting Atrial Fibrillation (AF) from ECG recordings using two deep learning architectures: a **Transformer** and a **Convolutional Neural Network** (CNN). Both models operate on fixed-length ECG segments (e.g., 10-second patches) and output AF vs. normal (and optionally “other”) classifications. To make the decision-making process transparent, we employ **SHAP** (Shapley Additive Explanations) to highlight which time-points most influence AF predictions.

---

## Key Features

1. **Preprocessing Pipeline**  
   - **Filtering**: Baseline wander removal (high-pass) and bandpass filtering (e.g., 0.5–40 Hz).  
   - **Channel Merging** (optional): Merge two ECG leads into a single channel.  
   - **Normalization**: Each segment is standardized to zero mean and unit variance.  
   - **Segmentation**: Raw ECG is split into fixed-length patches (e.g., 10 s at 200 Hz).

2. **Deep Learning Models**  
   - **Transformer**: Utilizes multi-head self-attention to capture long-range dependencies in 1D ECG signals.  
   - **CNN**: Employs multiple convolutional blocks and pooling layers to learn local morphological features crucial for arrhythmia detection.

3. **Training Utilities**  
   - **Cross-Entropy Loss** for multi-class or binary classification.  
   - **Adam/AdamW Optimizers** with learning rate schedulers (OneCycleLR or StepLR).  
   - **Early Stopping**: Monitors validation performance to avoid overfitting.

4. **Explainability via SHAP**  
   - **DeepExplainer** for computing Shapley values from raw model logits.  
   - **Color-Coded Plots**: Visualize positive (red) or negative (blue) SHAP attributions per time sample on the ECG waveform.  
   - **Group and Save**: Automatically groups samples by predicted and true class, saving plots in a designated folder.

---
## Getting Started

### 1. Clone and Install

    git clone https://github.com/manudella/ECGEXPFormer.git
    cd ECGEXPFormer
    pip install -r requirements.txt

### 2. Data Preparation

- **Raw Data**: If you have raw `.dat` files, place them in `./data/` and update your preprocessing script accordingly.  
- **Preprocessed Data**: If you already have `.npz` patches, place them in `./preprocessed_data/`.
- **Shuffled Data**: Shuffle your patches that will be put in `./shuffled_preprocessed_data/`.

### 3. Train a Model

- **Transformer**:

      python ECGFormerTask.py

- **CNN**:

      python ECGCNNTask.py

Each script supports various hyperparameters (batch size, learning rate, etc.). 

### 4. SHAP Explainability

Once training finishes, load a checkpointed model and generate SHAP explanations:

    python ECGShap.py

This script saves color-coded ECG plots in `./shap_images/`, showing which time regions most strongly influence the AF probability.

---

## References

- **AF Detection & ECG Classification**  
  - [Rohr et al., 2022](https://doi.org/10.1088/1361-6579/ac7840)  
  - [Zihlmann et al., 2017](https://doi.org/10.22489/CinC.2017.070-060)  
  - [Che et al., 2021 (BMC Medical Informatics and Decision Making)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01546-2)  
- **Transformer in Medical Signals**  
  - [Hong et al., 2019 (IJCAI)](https://www.ijcai.org/Proceedings/2019/0130.pdf)  
  - [Zhang et al., 2021](https://doi.org/10.1007/s11517-020-02292-9)  
- **SHAP**  
  - [Lundberg and Lee, 2017](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

---

## License

This project is licensed under the [MIT License](LICENSE). Contributions and issues are welcome!

Feel free to adapt the data paths, hyperparameters, or model configurations to suit your specific ECG dataset. Feedback or pull requests improving the pipeline, the models, or the visualization utilities are highly appreciated.

