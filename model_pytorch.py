import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import ast  # For parsing string representation of lists
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("step1")

# Custom Dataset to load data from CSV files
class SARSCoV2Dataset(Dataset):
    def __init__(self, csv_files):
        self.data = []
        self.variant_to_idx = {
            "B.1.1.7": 0,
            "B.1.351": 1,
            "P.1": 2,
            "B.1.617.2": 3,
            "B.1.1.529": 4
        }
        for file in csv_files:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                sequence = ast.literal_eval(row['Sequence'])
                sequence = np.array(sequence, dtype=np.float32)
                variant = self.variant_to_idx[row['Variant']]
                self.data.append((sequence, variant))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence, variant = self.data[idx]
        return torch.tensor(sequence), torch.tensor(variant)

print("step2")

# CNN Model Definition
class InterSSPPCNN(nn.Module):
    def __init__(self, input_length):
        super(InterSSPPCNN, self).__init__()
        # First Conv1D layer
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)
        
        # Second Conv1D layer
        self.conv2 = nn.Conv1d(32, 8, kernel_size=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Third Conv1D layer
        self.conv3 = nn.Conv1d(8, 8, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        
        # Calculate the output size after the convolution and pooling layers
        conv_out_size = self.calculate_conv_output_size(input_length)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size * 8, 72)  # Dense layer with 72 units
        self.fc2 = nn.Linear(72, 32)  # Dense layer with 32 units
        self.output = nn.Linear(32, 5)  # Output layer with 5 neurons for classification
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification

    def calculate_conv_output_size(self, input_length):
        x = torch.zeros(1, 5, input_length)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        return x.size(2)

    def forward(self, x):
        # Apply Conv1D, MaxPooling, and Dropout layers in sequence
        x = self.dropout1(self.pool1(torch.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(torch.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(torch.relu(self.conv3(x))))
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activation
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Output layer with softmax activation
        x = self.softmax(self.output(x))
        return x

print("step3")

# Load data and split into training and validation sets
csv_files = ['sampled_alpha_sequences.csv', 'sampled_beta_sequences.csv', 'sampled_delta_sequences.csv', 'sampled_gamma_sequences.csv', 'sampled_omicron_sequences.csv']
dataset = SARSCoV2Dataset(csv_files)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize Model
input_length = 30255
model = InterSSPPCNN(input_length)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early Stopping Parameters
patience = 3  # Number of epochs with no improvement after which training will be stopped
best_val_accuracy = 0.0
epochs_without_improvement = 0

# Training Loop with Early Stopping
num_epochs = 10
train_accuracies = []
val_accuracies = []

print("Model training started🎉🎉🎉🎉")
for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.permute(0, 2, 1)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    
    train_accuracy = train_correct / train_total
    train_accuracies.append(train_accuracy)
    
    # Validation accuracy
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.permute(0, 2, 1)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_accuracy = val_correct / val_total
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early Stopping Check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0
        # Save the model with the best validation accuracy
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered")
            break

print("Training complete.")

# Load the best model for evaluation
model.load_state_dict(torch.load("best_model.pth"))

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

variant_to_idx = {
    "B.1.1.7": 0,
    "B.1.351": 1,
    "P.1": 2,
    "B.1.617.2": 3,
    "B.1.1.529": 4
}
# Evaluation with confusion matrix
def evaluate_model(model, dataloader, variant_to_idx):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences = sequences.permute(0, 2, 1)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(variant_to_idx.keys()), 
                yticklabels=list(variant_to_idx.keys()))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Run Evaluation on Validation Set
evaluate_model(model, val_loader, variant_to_idx)
