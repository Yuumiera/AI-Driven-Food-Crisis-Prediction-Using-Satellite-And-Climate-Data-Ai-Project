
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_dataset(data, window_size=6):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def scale_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler
