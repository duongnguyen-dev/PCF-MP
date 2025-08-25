import os
import pandas as pd
from natsort import natsorted
from configs import DATASET_DIR

def get_datasets():
    ds_dir = os.path.expanduser(DATASET_DIR)
    ds = []
    for f in os.listdir(ds_dir):
        if not f.endswith(".docx"):
            ds.append(f"{DATASET_DIR}/{f}")
    
    return natsorted(ds)

def load_time_series(folder_path):
    file_path =  [f for f in os.listdir(folder_path) if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(folder_path, file_path), skiprows=18, low_memory=True)

    # Remove first row if needed
    first_row = df.iloc[0]
    converted = pd.to_numeric(first_row, errors='coerce')
    if converted.isnull().any():
        df = df.drop(index=0).reset_index(drop=True)
        df = df.astype({
            'Time': float,
            'Fx': float,
            'Fy': float,
            'Fz': float
        })

    return df
