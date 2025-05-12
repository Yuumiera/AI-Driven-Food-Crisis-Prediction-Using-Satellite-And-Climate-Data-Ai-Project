
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mae, mse

def plot_predictions(y_true, y_pred, title='LSTM Prediction'):
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label='Gerçek NDVI')
    plt.plot(y_pred, label='Tahmin NDVI', linestyle='--')
    plt.title(title)
    plt.xlabel('Zaman Adımı')
    plt.ylabel('NDVI')
    plt.legend()
    plt.grid(True)
    plt.show()
