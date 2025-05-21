import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

def get_monthly_modis_ndvi(region, start_year, end_year):
    collection = (ee.ImageCollection("MODIS/006/MOD13Q1")
                  .filterBounds(region)
                  .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
                  .select("NDVI"))

    ndvi_list = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            image = collection \
                .filter(ee.Filter.calendarRange(month, month, 'month')) \
                .filter(ee.Filter.calendarRange(year, year, 'year')) \
                .mean()
            try:
                ndvi = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=250,
                    maxPixels=1e13
                ).get("NDVI").getInfo()
                ndvi = ndvi / 10000.0 if ndvi is not None else None
            except:
                ndvi = None
            ndvi_list.append({'year': year, 'month': month, 'ndvi': ndvi})
    return pd.DataFrame(ndvi_list)

def create_dataset(data, window_size=6):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def main(country_name="Turkey", start_year=2020, end_year=2024):
    start_time = time.time()

    print("🌍 Earth Engine başlatılıyor...")
    ee.Initialize()

    print(f"📡 NDVI verisi alınıyor: {country_name}")
    region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
               .filter(ee.Filter.eq('country_na', country_name)).geometry()
    ndvi_df = get_monthly_modis_ndvi(region, start_year, end_year)
    ndvi_df.dropna(inplace=True)

    ndvi_df["date"] = pd.to_datetime(ndvi_df["year"].astype(str) + "-" + ndvi_df["month"].astype(str) + "-01")
    ndvi_df.set_index("date", inplace=True)
    ndvi = ndvi_df[["ndvi"]].values

    # Normalizasyon
    scaler = MinMaxScaler()
    ndvi_scaled = scaler.fit_transform(ndvi)

    # Dataset
    X, y = create_dataset(ndvi_scaled)
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, batch_size=8, verbose=0)

    # Tahmin
    y_pred = model.predict(X)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y)

    # Forecast 5 ay
    future_steps = 5
    last_window = X[-1]
    future_preds = []
    for _ in range(future_steps):
        pred = model.predict(np.expand_dims(last_window, axis=0), verbose=0)[0][0]
        future_preds.append(pred)
        last_window = np.roll(last_window, -1, axis=0)
        last_window[-1] = pred
    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Metrikler
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    print(f"✅ MAE: {mae:.4f}, MSE: {mse:.4f}")
    print(f"📈 Son tahmin edilen NDVI: {future_preds_inv[-1][0]:.4f}")

    # Grafik
    plt.figure(figsize=(12, 4))
    plt.plot(y_true_inv, label="Gerçek NDVI")
    plt.plot(y_pred_inv, label="Model Tahmini")
    plt.plot(range(len(y_pred_inv), len(y_pred_inv) + future_steps),
             future_preds_inv, label="Gelecek 5 Ay", linestyle="--", marker='o')
    plt.title(f"LSTM NDVI Forecast ({country_name})")
    plt.xlabel("Zaman")
    plt.ylabel("NDVI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Kayıt
    model_path = f"/content/drive/MyDrive/food_crisis_prediction2/models/lstm_ndvi_model_modis_{country_name}.h5"
    model.save(model_path)
    print(f"💾 Model kaydedildi: {model_path}")

    # Süre
    duration = time.time() - start_time
    print(f"⏱️ Tahmin süresi: {duration:.2f} saniye")

# Çalıştırmak için:
#main("Brazil")  # veya "Turkey", "India", vs.


