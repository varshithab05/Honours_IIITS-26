# add padding to the sequences
import pandas as pd

max_length = 30255

file_path = './dataset/gamma_seq.csv'

df = pd.read_csv(file_path)

def pad_sequence(sequence):
    return sequence.ljust(max_length, 'N')

df['Sequence'] = df['Sequence'].apply(pad_sequence)

df.to_csv(file_path, index=False)

print(f"Padded sequences saved back to {file_path}.")
