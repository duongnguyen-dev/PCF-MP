import os
import torch
import mlflow
import numpy as np
from mlflow.pytorch import autolog, log_model
from dotenv import load_dotenv
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from models.base import BaseModel
from dataset import PytorchWrapper
from sklearn.metrics import mean_absolute_error, r2_score

class MLPRegression(nn.Module):
    def __init__(self, input_dim=8, output_dim=3, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)  # 3 outputs
        )
    
    def forward(self, x):
        return self.layers(x)
    
class MLPModel(BaseModel):
    def __init__(self, params, auto_log: bool):
        super().__init__(params)
        load_dotenv()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri is None:
            raise ValueError("MLFLOW_TRACKING_URI environment variable should not be None")
        else:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("MLP Experiments")

        self.model = self.build_model()
        if auto_log:
            logger.info("Enable autologging.")
            autolog()
    
    def build_model(self):
        return MLPRegression(**self.params)
    
    def train(self, X_train, y_train, X_test, y_test):
        device, epochs = self.params['device'], self.params['epochs']

        self.model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        with mlflow.start_run():
            train_ds = PytorchWrapper(X_train, y_train)
            test_ds = PytorchWrapper(X_test, y_test)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_ds, batch_size=32, num_workers=4, shuffle=False)
            self.model.to(device)
        
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                all_train_preds, all_train_targets = [], []

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * X_batch.size(0)
                    all_train_preds.append(outputs.detach().cpu().numpy())
                    all_train_targets.append(y_batch.cpu().numpy())

                train_loss /= len(train_loader)
                all_train_preds = np.vstack(all_train_preds)
                all_train_targets = np.vstack(all_train_targets)
                train_mae = mean_absolute_error(all_train_targets, all_train_preds)
                train_r2 = r2_score(all_train_targets, all_train_preds)

                # Validation
                self.model.eval()
                val_loss = 0.0
                all_val_preds, all_val_targets = [], []

                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)

                        all_val_preds.append(outputs.cpu().numpy())
                        all_val_targets.append(y_batch.cpu().numpy())

                val_loss /= len(test_loader)
                all_val_preds = np.vstack(all_val_preds)
                all_val_targets = np.vstack(all_val_targets)
                val_mae = mean_absolute_error(all_val_targets, all_val_preds)
                val_r2 = r2_score(all_val_targets, all_val_preds)

                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_r2": train_r2,
                    "val_r2": val_r2
                }
                self.log_metrics(metrics, step=epoch)

                print(f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                    f"Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f}")
                
                self._log_model()

    def _log_model(self, X_train=None):
        log_model(self.model, "mlp_model")
    
    def eval(self, X_test, y_test) -> dict:
        return {}

    @staticmethod
    def predict(model_uri: str, sample):
        pass
        