import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from natsort import natsorted
from configs import DATASET_DIR

def get_datasets(dataset_dir):
    ds_dir = os.path.expanduser(os.path.join(dataset_dir, 'data'))
    ds = []
    for f in os.listdir(ds_dir):
        ds.append(f"{dataset_dir}/data/{f}")
    
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

def visualize_dataset(index=None, start_time=None, end_time=None):
    ds = get_datasets(DATASET_DIR)
    if index == None:
        selected_index = random.randint(0, len(ds))
    else:
        selected_index = index
        
    df = pd.read_csv(ds[selected_index])
    taguchi_matrix_df = pd.read_csv(f"{DATASET_DIR}/taguchi_matrix.csv")
    selected_exp = taguchi_matrix_df.iloc[selected_index]
    
    if start_time != None and end_time != None and start_time >= 0:
        df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
        
    df.set_index(df.columns[0], inplace=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes = axes.flatten()
    
    for i, col in enumerate(df):
        axes[i].plot(df.index, df[col], label=col)
        axes[i].set_title(f"Vc:{selected_exp['Vc']} - ft:{selected_exp['ft']} - a:{selected_exp['a']} - b:{selected_exp['b']}")
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel(f"{col} (N)")
        axes[i].legend()

    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    visualize_dataset(index=0, start_time=1, end_time=1.05)