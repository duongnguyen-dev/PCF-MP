import os
import torch
import numpy as np
import pandas as pd
from loguru import logger
from natsort import natsorted
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CuttingForceTaguchiDataset:
    """Cutting Force Dataset taken by using Taguchi method"""
    def __init__(self, root_dir: str, train_size: float):
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
        self.scaler = StandardScaler()

        result = self.prepare()
        if result != None:
            self.X_train, self.y_train, self.X_test, self.y_test = result
        else:
            raise ValueError("prepare() return None unexpectedly")

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

            # Attach Taguchi parameters (row i) to every row in this df
            for col in taguchi_matrix_df.columns:
                df[col] = taguchi_matrix_df.loc[i, col]
            
            # Split data into train and test set
            split_size = int(self.train_size * len(df))
            X = df.iloc[:, :].values
            y = df.iloc[:, 1:4].values
            X_train, y_train = X[:split_size], y[:split_size]        
            X_test, y_test = X[split_size:], y[split_size:]    
            # [TODO] Apply kalman filter for training set go here

            # [TODO]: Feature engineering go here

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

        # Normalization
        combined_train_scaled = self.scaler.fit_transform(combined_X_train)
        combined_test_scaled = self.scaler.transform(combined_X_test)
    
        return combined_train_scaled, combined_y_train, combined_test_scaled, combined_y_test
    
class PytorchWrapper(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

