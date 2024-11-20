# find out max length
import pandas as pd

file_path_ = './dataset/gamma_seq.csv'
df = pd.read_csv(file_path_)
        
max_length = df['Sequence'].apply(len).max()
    
print(max_length)
