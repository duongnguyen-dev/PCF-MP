import os
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from mlflow.sklearn import autolog, log_model, load_model
from loguru import logger
from dotenv import load_dotenv
from models.base import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, params, auto_log: bool):
        super().__init__(params)
        load_dotenv()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri is None:
            raise ValueError("MLFLOW_TRACKING_URI environment variable should not be None")
        else:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Random Forest Experiments")

        self.model = self.build_model()
        if auto_log:
            logger.info("Enable autologging.")
            autolog()

    def build_model(self):
        return RandomForestRegressor(**self.params)
    
    def train(self, X_train, y_train, X_test, y_test):
        with mlflow.start_run():
            self.model.fit(X_train, y_train)

            self.log_params()
            metrics = self.eval(X_test, y_test)
            self.log_metrics(metrics)
            mlflow.log_artifact("")
            self._log_model(np.expand_dims(X_train[0], axis=0))
    
    @staticmethod
    def predict(model_uri: str, sample):
        model = load_model(model_uri)
        if model == None:
            raise ValueError("Received None value after loading model from MLFlow")
        y_pred = model.predict(sample)
        return y_pred

    def eval(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        output_names = ["Fx", "Fy", "Fz"]

        r2_values = r2_score(y_test, y_pred, multioutput="raw_values")
        mae_values = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

        metrics = {f"{name}_r2": r2 for name, r2 in zip(output_names, r2_values)}
        metrics.update({f"{name}_mae": mae for name, mae in zip(output_names, mae_values)})
        
        return metrics

    def _log_model(self, X_train):
        log_model(
            sk_model=self.model,
            name="RF-model",
            input_example=X_train,
            registered_model_name="RF-reg-model"
        )