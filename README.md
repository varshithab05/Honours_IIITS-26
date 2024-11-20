# SARS-CoV-2 Variant Classification Using CNN with Explainability

## Overview
This project is part of my Honors work aimed at developing a deep learning-based approach for **classifying SARS-CoV-2 variants** using one-hot encoded genomic sequences. The research leverages **1D Convolutional Neural Networks (CNNs)** for efficient and accurate classification, with an added focus on explainability using **Layer-wise Relevance Propagation (LRP)**.

---

## Objectives
1. **Classification of SARS-CoV-2 Variants**: Use genomic sequences to predict the variant (e.g., Alpha, Beta, Gamma, Delta, Omicron).
2. **Deep Learning Explainability**: Provide model interpretability through **LRP** to understand which genomic regions influence the model's predictions.
3. **Performance Optimization**: Achieve high accuracy and efficiency through hyperparameter tuning, batch size adjustments, and learning rate scheduling.

---

## Dataset Details
- **Source**: SARS-CoV-2 genomic sequences.
- **Variants**:
  - **Alpha (B.1.1.7)**: 2000 sequences
  - **Beta (B.1.351)**: 2000 sequences
  - **Gamma (P.1)**: 2000 sequences
  - **Delta (B.1.617.2)**: 2000 sequences
  - **Omicron (B.1.1.529)**: 2000 sequences
- **Preprocessing**:
  - Removed invalid characters from sequences.
  - Padded sequences to a maximum length of **30255** with the character 'N'.
  - Converted sequences to one-hot encoded format.
  - Balanced dataset with 2000 samples per variant.

---

## Methodology

### 1. **Model Architecture**
- **Type**: 1D Convolutional Neural Network (CNN)
- **Components**:
  - 3 convolutional layers followed by max-pooling and dropout.
  - Fully connected (dense) layers:
    - Layer 1: 72 units
    - Layer 2: 32 units
    - Output: 5 neurons (one for each variant) with **Softmax activation**.
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

### 2. **Training and Validation**
- **Batch Size**: 16
- **Learning Rate**: Tuned dynamically using a scheduler.
- **Early Stopping**: Stops training if validation loss doesn't improve for 5 consecutive epochs.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

### 3. **Explainability** (On Going)
- Using **Layer-wise Relevance Propagation (LRP)** to find mutations.

---

## Results
- **Accuracy**:
  - TensorFlow/Keras Model: 98%
  - PyTorch Model: Improved from 84% to 89.9% with tuning.
- **Explainability**: LRP successfully identified significant genomic regions contributing to predictions.
