import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

CRITEO_PATH = r'C:\Users\arsal\OneDrive\Desktop\ad_efficiency\dac\train.txt'

def load_criteo(filepath=CRITEO_PATH, nrows=100000):
    print(f"Loading {nrows:,} rows from Criteo dataset...")
    
    num_cols = [f'n{i}' for i in range(13)]
    cat_cols = [f'c{i}' for i in range(26)]
    cols     = ['label'] + num_cols + cat_cols
    
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=cols,
        nrows=nrows
    )
    
    print(f"Loaded! Shape: {df.shape}")
    print(f"Click rate: {df['label'].mean()*100:.1f}%")
    
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('missing')
    
    X = df[num_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)
    X      = np.clip(X, -10, 10)
    
    print(f"Features shape: {X.shape}")
    print(f"Clicks:    {int(y.sum()):,}")
    print(f"No clicks: {int((1-y).sum()):,}")
    
    return torch.tensor(X, dtype=torch.float32), \
           torch.tensor(y, dtype=torch.float32)

if __name__ == "__main__":
    X, y = load_criteo()
    print("\nSuccess!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
