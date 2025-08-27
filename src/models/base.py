import mlflow
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, params):
        super().__init__()
        self.params = params

    @abstractmethod
    def build_model(self) -> object:
        """Modeling"""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_test, y_test):
        """Training Procedure"""
        pass

    @staticmethod
    @abstractmethod
    def predict(model_uri: str, sample):
        """Predicting a new sample"""
        pass

    @abstractmethod
    def eval(self, X_test, y_test) -> dict:
        """Evaluation procedure, must return a dict of metrics"""
        pass

    @abstractmethod
    def _log_model(self, X_train=None):
        """Log model using MLFlow"""
        pass

    @staticmethod
    def log_metrics(metrics, step=None):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics, step=step)
        
    def log_params(self):
        """Log parameters to MLflow"""
        mlflow.log_params(self.params)