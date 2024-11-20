# remove the charecters other than A C G T and N
import pandas as pd

# Function to clean sequences
def clean_sequence(sequence):
    return ''.join([char for char in sequence if char in 'ACGTN'])

file_path_ = './dataset/gamma_seq.csv'

df = pd.read_csv(file_path_)

# Clean the 'sequence' column
df['Sequence'] = df['Sequence'].apply(clean_sequence)

# Save the cleaned data back to the same CSV file
df.to_csv(file_path_, index=False)  # Overwrite the original file

print(f"Cleaned sequences saved back to {file_path_}")
