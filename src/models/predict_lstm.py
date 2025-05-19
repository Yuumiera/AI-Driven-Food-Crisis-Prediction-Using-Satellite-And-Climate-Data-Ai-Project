import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
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

def forecast_lstm_ndvi(model_path, country_name="Brazil", start_year=2020, end_year=2024):
    start_time = time.time()
    ee.Initialize()

    # Bölge ve veri
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

    # Son pencere
    window_size = 6
    last_window = ndvi_scaled[-window_size:]
    last_window = np.expand_dims(last_window, axis=0)

    # Model yükle
    model = load_model(model_path, compile=False)

    # 5 aylık forecast
    future_preds = []
    for _ in range(5):
        pred = model.predict(last_window, verbose=0)[0][0]
        future_preds.append(pred)
        new_window = np.roll(last_window[0], -1, axis=0)
        new_window[-1] = pred
        last_window = np.expand_dims(new_window, axis=0)

    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Grafik
    plt.figure(figsize=(12, 4))
    plt.plot(ndvi, label="Gerçek NDVI")
    plt.plot(range(len(ndvi), len(ndvi) + 5), future_preds_inv, label="Tahmin (5 Ay)", linestyle='--', marker='o')
    plt.title(f"{country_name} - LSTM NDVI Forecast")
    plt.xlabel("Zaman")
    plt.ylabel("NDVI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"📈 Son tahmin edilen NDVI: {future_preds_inv[-1][0]:.4f}")
    print(f"⏱️ Süre: {time.time() - start_time:.2f} saniye")
