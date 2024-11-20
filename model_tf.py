import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Data
variant_files = ['sampled_alpha_sequences.csv', 'sampled_beta_sequences.csv', 'sampled_delta_sequences.csv', 'sampled_gamma_sequences.csv', 'sampled_omicron_sequences.csv']  # Replace with actual file paths
data = []
print("step1 done")

for file in variant_files:
    df = pd.read_csv(file)
    data.append(df)

combined_data = pd.concat(data, ignore_index=True)

# 2. Shuffle the Data
combined_data = shuffle(combined_data, random_state=42)

print("step2 done")

# 3. Prepare Data for CNN
# Convert the one-hot encoded sequences column from string to array format with memory optimization
def parse_sequence(seq_str):
    """Parses the JSON string sequence and converts it to a numpy array of type int8."""
    return np.array(json.loads(seq_str.replace("'", '"')), dtype=np.int8)

# Apply the parsing function to each sequence
X = np.array([parse_sequence(seq) for seq in combined_data['Sequence']])

y = pd.factorize(combined_data['Variant'])[0]  # Encode variant labels as integers
y = to_categorical(y)  # One-hot encode labels for classification

# Verify the input shape (sequence length, features)
input_shape = (X.shape[1], X.shape[2])  # Shape is (sequence length, 5) for one-hot encoded bases
print("step 3 done")

# 4. Define CNN Model using your provided architecture
def create_cnn_model(input_shape):
    model = models.Sequential()
    
    # First Conv1D layer
    model.add(layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    
    # Second Conv1D layer
    model.add(layers.Conv1D(filters=8, kernel_size=4, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    
    # Third Conv1D layer
    model.add(layers.Conv1D(filters=8, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(layers.Dropout(0.2))
    
    # Flatten the convolutional output
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(72, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    
    # Output layer with softmax for classification
    model.add(layers.Dense(y.shape[1], activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Create the model with the calculated input shape
model = create_cnn_model(input_shape)

print("model training start")
# 5. Train the Model with Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

print('model training end')
# Save the Model
model.save('sars_cov2_variant_classifier.h5')
print("model saaved")
# 6. Evaluate Model - Accuracy, Precision, Recall
# Split the data into train and validation for evaluation
# Here, we'll use 20% of the data as a validation set for simplicity
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on the validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes, target_names=pd.factorize(combined_data['Variant'])[1])

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
variant_labels = pd.factorize(combined_data['Variant'])[1]  # Get variant labels

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=variant_labels)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for SARS-CoV-2 Variant Classification')
plt.xticks(rotation=45)
plt.show()


# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Additional Combined Metrics Visualization
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Accuracy
axs[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
axs[0].set_title('Accuracy')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Accuracy')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Loss
axs[1].plot(history.history['loss'], label='Training Loss', color='red')
axs[1].plot(history.history['val_loss'], label='Validation Loss', color='purple')
axs[1].set_title('Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
