import os
import math
import torch
import random
import numpy as np
import pandas as pd
from loguru import logger
from natsort import natsorted
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CuttingForceTaguchiDataset:
    """Cutting Force Dataset taken by using Taguchi method"""
    def __init__(self, root_dir: str, train_size: float, time_range: float=0.004, d: int = 16):
        """
        The project structure should have the following csv files:
        - taguchi_matrix.csv: This store the cutting condition of all the experiments to get cutting force data
        - /data/*.csv: Dataset file that include cutting force captured by the dynamometer in time series 
        NOTE: The number of *.csv files should equals to the number of rows in taguchi_matrix.csv file

        The class takes two arguments when being initialized:
        - root_dir: The directory to the dataset files 
        - transform: Additional transformations on the dataset
        """
        self.root_dir = root_dir
        self.train_size = train_size
        self.time_range = time_range
        self.d = d

        self.scaler = StandardScaler()

        result = self.prepare()
        if result != None:
            self.X_train, self.y_train, self.X_test, self.y_test = result
        else:
            raise ValueError("prepare() return None unexpectedly")

    def calculate_rpm(self, vc: int):
        return math.floor((1000 * vc) / (math.pi * self.d))
    
    @staticmethod
    def calculate_tpf(tn, rpm: int):
        """
        Calculating Tooth Passing Frequency that automatically identify the starting time of the cutting process
        """
        return np.sin((rpm / 60) * tn)
    
    def prepare(self):
        """
        For each file in that data folder, apply the following functions:
        - Remove zero values in the cutting force data
        - Split into train / test set and turn time-series data into a supervised learning problem (windowing)
        - Kalman filter to remove noise from the dataset (Unavailable)
        - Feature engineering: Create additional features in time and frequency domains (Unavailable)
        - Concatenate data
        - Apply normalization
        """
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")
        
        taguchi_matrix_path = os.path.join(self.root_dir, "taguchi_matrix.csv")
        if not os.path.exists(taguchi_matrix_path):
            raise FileNotFoundError(f"Taguchi Matrix file not found in the dataset folder: {taguchi_matrix_path}")
        
        taguchi_matrix_df = pd.read_csv(taguchi_matrix_path)
        if len(taguchi_matrix_df) == 0:
            return None

        data_dir = os.path.join(self.root_dir, "data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        data_files = natsorted(os.listdir(data_dir))

        if len(data_files) != len(taguchi_matrix_df):
            raise Exception("The number of dataset files doesn't match the number of experiments")
        
        list_X_train, list_X_test, list_y_train, list_y_test = [], [], [], []
        for i, f in enumerate(data_files):
            f_path = os.path.join(data_dir, f)
            df = pd.read_csv(f_path)

            # Remove zero values in the cutting force data
            df.drop(df[(df['Fx'] == 0.0) & (df['Fy'] == 0.0) & (df['Fz'] == 0.0)].index, inplace=True)
            
            if self.time_range != None:
                start_time = random.choice(df["Time"].values)
                end_time = start_time + self.time_range
                df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]

            # Attach Taguchi parameters (row i) to every row in this df
            for col in taguchi_matrix_df.columns:
                df[col] = taguchi_matrix_df.loc[i, col]

            rpm = self.calculate_rpm(df["Vc"].values[0])
            # Apply the function to the tn column
            df['tpf'] = self.calculate_tpf(df["Time"].values, rpm)
            # Split data into train and test set
            split_size = int(self.train_size * len(df))
        
            X = df.drop(columns=["Fx", "Fy", "Fz"])
            y = df[["Fx", "Fy", "Fz"]]
            X_train, X_test = X[:split_size], X[split_size:]
            y_train, y_test = y[:split_size], y[split_size:]

            list_X_train.append(X_train)
            list_X_test.append(X_test)
            list_y_train.append(y_train)
            list_y_test.append(y_test)

            logger.info(f"Processed {f} successfully.")
        # Concatenate all loaded dataframe
        combined_X_train = np.concatenate(list_X_train, axis=0)
        combined_X_test = np.concatenate(list_X_test, axis=0)
        combined_y_train = np.concatenate(list_y_train, axis=0)
        combined_y_test = np.concatenate(list_y_test, axis=0)
    
        return combined_X_train, combined_y_train, combined_X_test, combined_y_test

class CuttingForceWindowDataset():
    def __init__(self, root_dir: str, horizon: int, window: int, sampling_rate: int = 10000, d: int = 16, train_size: float = 0.8):
        self.root_dir = root_dir
        self.horizon = horizon
        self.window = window
        self.d = d
        self.sampling_rate = sampling_rate
        self.train_size = train_size
        self.scaler = StandardScaler()
        result = self.prepare()
        if result != None:
            self.X_train, self.y_train, self.X_test, self.y_test = result
        else:
            raise ValueError("prepare() return None unexpectedly")
    
    def calculate_rpm(self, vc: int):
        return math.floor((1000 * vc) / (math.pi * self.d))
    
    # def calculate_window(self, rpm: int):
    #     """
    #     Look back step (window)
    #     """
    #     return math.floor((self.sampling_rate * 60) / rpm)
    
    @staticmethod
    def calculate_tpf(tn, rpm: int):
        """
        Calculating Tooth Passing Frequency that automatically identify the starting time of the cutting process
        """
        return np.sin((rpm / 60) * tn)
    
    @staticmethod
    def make_window(x, horizon: int, window: int):
        """
        Turns a 1D array into a 2D array of sequential window
        - window: the number of past data
        - horizon: the number of future values that we want to predict based on the number of past values (window)
        """
        # Step 1: Create a window of specific window size (add horizon on the end)
        window_step = np.expand_dims(np.arange(window + horizon), axis=0)
        
        # Step 2: Create a 2D array of multiple window steps
        window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window+ horizon -1)), axis=0).T
        
        # Step 3: Index on the target array (time series) with 2D array of multiple window steps
        windowed_array = x[window_indexes]
        
        return windowed_array
        
    @staticmethod
    def get_labelled_windows(x, horizon):
        """
        Creates labels for windowed dataset.
        Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
        """   
        return x[:, :-horizon], x[:, -horizon:, 1:4]
    
    def prepare(self):
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory does not exist: {self.root_dir}")
        
        taguchi_matrix_path = os.path.join(self.root_dir, "taguchi_matrix.csv")
        if not os.path.exists(taguchi_matrix_path):
            raise FileNotFoundError(f"Taguchi Matrix file not found in the dataset folder: {taguchi_matrix_path}")
        
        taguchi_matrix_df = pd.read_csv(taguchi_matrix_path)
        if len(taguchi_matrix_df) == 0:
            return None

        data_dir = os.path.join(self.root_dir, "data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        ds = []
        for f in os.listdir(data_dir):
            ds.append(f"{data_dir}/{f}")
            
        data_files = natsorted(ds)

        if len(data_files) != len(taguchi_matrix_df):
            raise Exception("The number of dataset files doesn't match the number of experiments")

        list_X_train, list_X_test, list_y_train, list_y_test = [], [], [], []
        for i, f in enumerate(data_files[:1]):
            df = pd.read_csv(data_files[0]) 
            
            # Remove zero values in the cutting force data
            df.drop(df[(df['Fx'] == 0.0) & (df['Fy'] == 0.0) & (df['Fz'] == 0.0)].index, inplace=True)
            
            # Attach Taguchi parameters (row i) to every row in this df
            for col in taguchi_matrix_df.columns:
                df[col] = taguchi_matrix_df.loc[i, col]
            
            rpm = self.calculate_rpm(df["Vc"][0])
            # Apply the function to the tn column
            df['tpf'] = self.calculate_tpf(df["Time"].values, rpm)
            
            # Windowing split
            split_size = int(self.train_size * len(df))
            df_array = df.iloc[:, :].values
            df_train, df_test = df_array[:split_size], df_array[split_size:]

            # Normalization 
            train_mean = df_train.mean()
            train_std = df_train.std()

            df_train = (df_train - train_mean) / train_std
            df_test = (df_test - train_mean) / train_std

            train_windowed_array = self.make_window(df_train, horizon=self.horizon, window=self.window)
            test_windowed_array = self.make_window(df_test, horizon=self.horizon, window=self.window)
            
            X_train, y_train = self.get_labelled_windows(train_windowed_array, horizon=self.horizon)
            X_test, y_test = self.get_labelled_windows(test_windowed_array, horizon=self.horizon)
            print(X_train.shape)
            list_X_train.append(X_train)
            list_X_test.append(X_test)
            list_y_train.append(y_train)
            list_y_test.append(y_test)
        # X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        # X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Processed {data_files[0]} successfully.")

        # Concatenate all loaded dataframe
        combined_X_train = np.concatenate(list_X_train, axis=0)
        combined_X_test = np.concatenate(list_X_test, axis=0)
        combined_y_train = np.concatenate(list_y_train, axis=0)
        combined_y_test = np.concatenate(list_y_test, axis=0)
    
        return combined_X_train, combined_y_train, combined_X_test, combined_y_test

class PytorchWrapper(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].reshape(-1), self.y[idx].reshape(-1)

