import pandas as pd
import numpy as np

# File paths
input_file_path = './dataset/gamma_seq_p2.csv'  # Update with your actual file name
output_file_path = 'encoded_dataset/one_hot_gamma_p2_sequences.csv'  # Output file name
batch_size = 1000  # Define batch size for processing
max_sequence_length = 30255  # Define max length for padding

# Define the one-hot encoding function
def one_hot_encode(sequence):
    encoding = {
        'A': [1, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0],
        'G': [0, 0, 1, 0, 0],
        'T': [0, 0, 0, 1, 0],
        'N': [0, 0, 0, 0, 1]
    }
    
    # Encode each character in the sequence and pad up to max length
    one_hot_matrix = [encoding.get(nucleotide, [0, 0, 0, 0, 1]) for nucleotide in sequence]
    
    # Pad if the sequence is shorter than max length
    padding_length = max_sequence_length - len(one_hot_matrix)
    if padding_length > 0:
        one_hot_matrix.extend([[0, 0, 0, 0, 1]] * padding_length)  # Pad with 'N' encoding

    return np.array(one_hot_matrix)

# Process CSV in batches
with pd.read_csv(input_file_path, chunksize=batch_size) as reader:
    for i, chunk in enumerate(reader):
        # One-hot encode the 'sequence' column, turning each into a (30255, 5) array
        chunk['Sequence'] = chunk['Sequence'].apply(lambda seq: one_hot_encode(seq).tolist())
        
        # Save the result to a CSV file
        if i == 0:
            chunk.to_csv(output_file_path, index=False, mode='w')  # Overwrite for the first batch
        else:
            chunk.to_csv(output_file_path, index=False, mode='a', header=False)  # Append for subsequent batches
        
        print(f"Processed and saved batch {i + 1} to CSV.")

print(f"One-hot encoded sequences saved to '{output_file_path}'.")
