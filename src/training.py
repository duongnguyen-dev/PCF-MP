import os
import torch
import pysindy as ps    
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load
from models.sklearn.svr import SVRModel
from models.sklearn.random_forest import RandomForestModel
from models.torch.mlp import MLPModel
from dataset import CuttingForceTaguchiDataset, CuttingForceWindowDataset
from configs import DATASET_DIR, HORIZON

def train_svr_model():
    dataset = CuttingForceTaguchiDataset(root_dir=os.path.expanduser(DATASET_DIR), train_size=0.8)
    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test

    svr_hyperparams = {
        "kernel": "poly",       # default: 'rbf' — other options: 'linear', 'poly', 'sigmoid', 'precomputed'
        "degree": 3,           # default: 3 — used only with 'poly' kernel
        "gamma": "scale",      # default: 'scale' (since v0.22); alternative: 'auto' or float
        "coef0": 0.0,          # default: 0.0 — relevant for 'poly' and 'sigmoid'
        "tol": 0.0001,          # default: 0.001 — tolerance for stopping criterion
        "C": 0.5,              # default: 1.0 — regularization parameter
        "epsilon": 0.1,        # default: 0.1 — epsilon-tube in regression loss
        "verbose": True,      # default: False — enable verbose output
        "max_iter": 20000     # default: -1 — no iteration limit
    }

    model = SVRModel(params=svr_hyperparams, auto_log=False)

    model.train(X_train, y_train, X_test, y_test)
    metrics = model.eval(X_test, y_test)


def train_mlp_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 100
    window = 30
    horizon = 1
    dataset = CuttingForceTaguchiDataset(root_dir=os.path.expanduser(DATASET_DIR),
                                        train_size=0.8)
    
    params = {
        "device": device,
        "epochs": epochs,
        "input_dim": 6
    }

    model = MLPModel(params, auto_log=False)

    model.train(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)

def train_randomforest_model():
    dataset = CuttingForceTaguchiDataset(root_dir=os.path.expanduser(DATASET_DIR),
                                        train_size=0.8)
    rf_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42
    }
    model = RandomForestModel(rf_params, auto_log=False)
    model.train(dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test)

if __name__ == "__main__":
    train_randomforest_model()