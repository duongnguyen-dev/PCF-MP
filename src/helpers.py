import os
import math
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from mlflow.sklearn import load_model
from loguru import logger
from natsort import natsorted
from configs import DATASET_DIR, SAMPLING_RATE
from dataset import CuttingForceTaguchiDataset

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

def read_file(dataset_dir: str, index=None, start_time=None, end_time=None):
    ds = get_datasets(dataset_dir)
    if index == None:
        selected_index = random.randint(0, len(ds))
    else:
        selected_index = index
        
    df = pd.read_csv(ds[selected_index])
    taguchi_matrix_df = pd.read_csv(f"{dataset_dir}/taguchi_matrix.csv")
    selected_exp = taguchi_matrix_df.iloc[selected_index]
    
    if start_time != None and end_time != None and start_time >= 0:
        df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
    
    return df, selected_exp

def calculate_rpm(vc: int, d: int):
    return math.floor((1000 * vc) / (math.pi * d))
    
def calculate_tpf(tn, rpm: int):
    """
    Calculating Tooth Passing Frequency that automatically identify the starting time of the cutting process
    """
    return np.sin((rpm / 60) * tn)
    
def make_window(x, horizon: int, window: int):
    """
    Turns a 1D array into a 2D array of sequential window
    - window: the number of past data
    - horizon: the number of future values that we want to predict based on the number of past values (window)
    """
    # Step 1: Create a window of specific window size (add horizon on the end)
    window_step = np.expand_dims(np.arange(window + horizon), axis=0)
    logger.info(f"Window step: {window_step}")
    
    # Step 2: Create a 2D array of multiple window steps
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window+ horizon -1)), axis=0).T
    logger.info(f"Window indexes: {window_indexes.shape}")
    
    # Step 3: Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]
    
    return windowed_array
    
def get_labelled_windows(x, horizon):
    """
    Creates labels for windowed dataset.
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """   
    return x[:, :-horizon], x[:, -horizon:, 1:4]


def visualize_preprocessed_dataset(dataset_dir, d: int, train_size: float, horizon: int, window: int, sampling_rate: int, index=None):
    df, selected_exp = read_file(dataset_dir, index)
    df.drop(df[(df['Fx'] == 0.0) & (df['Fy'] == 0.0) & (df['Fz'] == 0.0)].index, inplace=True)
    df = df.assign(Vc=selected_exp['Vc'], ft=selected_exp['ft'], a=selected_exp['a'], b=selected_exp['b'])
    rpm = calculate_rpm(df["Vc"][0], d)
    # Apply the function to the tn column
    df['tpf'] = calculate_tpf(df["Time"].values, rpm)

    split_size = int(train_size * len(df))
    df_array = df.iloc[:, :].values
    df_train, _ = df_array[:split_size], df_array[split_size:]
    train_windowed_array = make_window(df_train, horizon=horizon, window=window)
    
    X_train, y_train = get_labelled_windows(train_windowed_array, horizon=horizon)
    
    n_visualize = random.randint(0, 10)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes = axes.flatten()

    time = X_train[:n_visualize, :, 0]
    new_times = []
    for t in time:
        extended = np.append(t, t[-1]+(1/sampling_rate))
        new_times.append(extended)
    new_times = np.array(new_times)

    Fx = X_train[:n_visualize, :, 1]
    Fy = X_train[:n_visualize, :, 2]
    Fz = X_train[:n_visualize, :, 3]

    Fx_gt = y_train[:n_visualize, :, 0]
    Fy_gt = y_train[:n_visualize, :, 1]
    Fz_gt = y_train[:n_visualize, :, 2]
    
    axes[0].plot(new_times[:, :-1], Fx, color='blue')
    axes[0].plot(new_times[:, window:], Fx_gt, color='magenta', linestyle='--')
    axes[0].set_ylabel('Fx')
    axes[0].set_xlabel('Time')
    axes[0].set_title('Fx over time')
    axes[0].legend()

    # Vẽ x2 và b
    axes[1].plot(new_times[:, :window], Fy, color='green')
    axes[1].plot(new_times[:, -1], Fy_gt, color='magenta', linestyle='--')
    axes[1].set_ylabel('Fy')
    axes[1].set_xlabel('Time')
    axes[1].set_title('Fy over time')
    axes[1].legend()

    # Vẽ x3 và b
    axes[2].plot(new_times[:, :window], Fz, color='red')
    axes[2].plot(new_times[:, -1], Fz_gt, color='magenta', linestyle='--')
    axes[2].set_ylabel('Fz')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Fz over time')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
        
    

def visualize_dataset(dataset_dir, index=None, start_time=None, end_time=None):
    df, selected_exp = read_file(dataset_dir, index, start_time, end_time)
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

def visualize_tpf(dataset_dir, index, d, start_time, end_time):
    df, selected_exp = read_file(dataset_dir, index, start_time, end_time)
    df.drop(df[(df['Fx'] == 0.0) & (df['Fy'] == 0.0) & (df['Fz'] == 0.0)].index, inplace=True)
    df = df.assign(Vc=selected_exp['Vc'], ft=selected_exp['ft'], a=selected_exp['a'], b=selected_exp['b'])
    rpm = calculate_rpm(df['Vc'].values[0], d)
    # Apply the function to the tn column
    df['tpf'] = calculate_tpf(df["Time"].values, rpm)
    df.set_index('Time', inplace=True)

    df['tpf'].plot(figsize=(10, 5), marker='o', linestyle='-')
    plt.title('TPF over Time')
    plt.xlabel('Time')
    plt.ylabel('TPF')
    plt.grid(True)
    plt.show()

def visualize_pred(dataset_dir, index, start_time, end_time, d):
    df, selected_exp = read_file(dataset_dir, index, start_time, end_time)
    df.drop(df[(df['Fx'] == 0.0) & (df['Fy'] == 0.0) & (df['Fz'] == 0.0)].index, inplace=True)
    df = df.assign(Vc=selected_exp['Vc'], ft=selected_exp['ft'], a=selected_exp['a'], b=selected_exp['b'])
    rpm = calculate_rpm(df['Vc'].values[0], d)
    # Apply the function to the tn column
    df['tpf'] = calculate_tpf(df["Time"].values, rpm)

    X = df.drop(columns=["Fx", "Fy", "Fz"]).values
    y = df[["Fx", "Fy", "Fz"]].values
    with open('mlartifacts/462321641444534951/models/m-085472596aee43a3a337370cd919e73f/artifacts/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    y_pred = loaded_model.predict(X)
    print(y_pred, y)
    time_index = X[:, 0]

    df = pd.DataFrame({
        'Fx_GT': y[:, 0],
        'Fx_Pred': y_pred[:, 0],
        'Fy_GT': y[:, 1],
        'Fy_Pred': y_pred[:, 1],
        'Fz_GT': y[:, 2],
        'Fz_Pred': y_pred[:, 2],
        # Add more columns as needed
    }, index=time_index)

    df[['Fx_GT', 'Fx_Pred']].plot(color=['blue', 'orange'])
    df[['Fy_GT', 'Fy_Pred']].plot(color=['blue', 'orange'])
    df[['Fz_GT', 'Fz_Pred']].plot(color=['blue', 'orange'])
    plt.show()

if __name__ == "__main__":
    # visualize_preprocessed_dataset(DATASET_DIR, d=16, train_size=0.8, horizon=1, window=30, sampling_rate=SAMPLING_RATE, index=0)
    # visualize_tpf(DATASET_DIR, index=20, d=16, start_time=0, end_time=0.5)
    visualize_pred(DATASET_DIR, index=24, d=16, start_time=5, end_time=5.05)