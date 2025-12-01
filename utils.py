# utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def handle_missing(df, strategy='drop', cols=None):
    if cols is None:
        cols = df.columns.tolist()
    if strategy == 'drop':
        return df.dropna(subset=cols).reset_index(drop=True)
    if strategy == 'ffill':
        return df[cols].ffill().bfill()
    return df[cols].fillna(0)

def scale_features(X, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)
        return Xs, scaler
    else:
        return scaler.transform(X), scaler

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
