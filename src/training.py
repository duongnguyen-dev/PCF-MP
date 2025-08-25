import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.sklearn.svr import SVRModel
from dataset import CuttingForceTaguchiDataset
from configs import DATASET_DIR

dataset = CuttingForceTaguchiDataset(root_dir=os.path.expanduser(DATASET_DIR), train_size=0.8)
X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test


svr_hyperparams = {
    "kernel": "rbf",       # default: 'rbf' — other options: 'linear', 'poly', 'sigmoid', 'precomputed'
    "degree": 3,           # default: 3 — used only with 'poly' kernel
    "gamma": "scale",      # default: 'scale' (since v0.22); alternative: 'auto' or float
    "coef0": 0.0,          # default: 0.0 — relevant for 'poly' and 'sigmoid'
    "tol": 0.001,          # default: 0.001 — tolerance for stopping criterion
    "C": 1.0,              # default: 1.0 — regularization parameter
    "epsilon": 0.1,        # default: 0.1 — epsilon-tube in regression loss
    "shrinking": True,     # default: True — whether to use the shrinking heuristic
    "cache_size": 200,     # default: 200 (MB) — size of kernel cache
    "verbose": True,      # default: False — enable verbose output
    "max_iter": 300     # default: -1 — no iteration limit
}

model = SVRModel(params=svr_hyperparams, auto_log=False)

model.train(X_train, y_train)
metrics = model.eval(X_test, y_test)
print(metrics)