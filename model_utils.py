import numpy as np
from joblib import load

def load_scaler(path="data/scaler.pkl"):
    return load(path)

def load_model_parameters(w_path="data/lr_weights.npy", b_path="data/lr_bias.npy"):
    w = np.load(w_path)
    b = np.load(b_path).item()
    return w, b
