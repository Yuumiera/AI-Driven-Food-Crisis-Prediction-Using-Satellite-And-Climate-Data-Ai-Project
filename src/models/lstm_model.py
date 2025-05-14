import ee
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.gee.data_loader import get_monthly_ndvi
from src.preprocessing.data_prep import create_dataset

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run_lstm_pipeline():
    # Bölge tanımı
    region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', 'Turkey')).geometry()

    # NDVI verisi çek
    ndvi_df = get_monthly_ndvi(region, start_year=2020, end_year=2024, scale=250)
    ndvi_df.dropna(inplace=True)

    # Tarih kolonunu oluştur
    ndvi_df["date"] = pd.to_datetime(ndvi_df["year"].astype(str) + "-" + ndvi_df["month"].astype(str) + "-01")
    ndvi_df.set_index("date", inplace=True)

    # Normalizasyon
    scaler = MinMaxScaler()
    scaled_ndvi = scaler.fit_transform(ndvi_df[["ndvi"]])

    # Veri seti oluştur
    window_size = 6
    X, y = create_dataset(scaled_ndvi, window_size)

    # Modeli eğit
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=200, batch_size=8, verbose=0)

    # Tahmin yap
    y_pred = model.predict(X)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y)

    # Değerlendirme
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    print(f"LSTM MAE: {mae:.4f}, MSE: {mse:.4f}")

    # Grafik
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_inv, label="Actual NDVI")
    plt.plot(y_pred_inv, label="Predicted NDVI")
    plt.legend()
    plt.title("LSTM NDVI Forecast")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Kaydet
    model.save("/content/drive/MyDrive/food_crisis_prediction2/models/lstm_ndvi_model.h5")
    print("LSTM model saved.")
