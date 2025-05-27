# #!/usr/bin/env python3
# """
# LSTM tabanlı sıcaklık ve zaman girdileriyle kuraklık skoru tahmini ve
# sıcaklık trendi + LSTM tahmini modülü.

# - ERA5 verisini çeker,
# - Sıcaklık trendini görselleştirir,
# - LSTM ile kaydırmalı pencere (sliding window) üzerinden tahmin yapar,
#   her bir tahmini konsola basar ve grafik olarak kaydeder.
# - Komut satırından bölge ve tarih aralığı ile çalıştırılabilir.
# """
# import os
# import sys
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# from io import BytesIO
# import base64
# import pandas as pd
# import ee
# import argparse
# import matplotlib.dates as mdates

# # Earth Engine kimlik doğrulama
# ee.Authenticate(auth_mode='notebook')
# ee.Initialize()

# def get_era5_temp_precip(region, start_date, end_date):
#     collection = (ee.ImageCollection('ECMWF/ERA5/DAILY')
#                   .filterDate(start_date, end_date)
#                   .filterBounds(region))
#     def extract_vals(img):
#         date = img.date().format('YYYY-MM-dd')
#         temp = img.select('mean_2m_air_temperature') \
#                   .reduceRegion(ee.Reducer.mean(), region, 10000) \
#                   .get('mean_2m_air_temperature')
#         precip = img.select('total_precipitation') \
#                      .reduceRegion(ee.Reducer.mean(), region, 10000) \
#                      .get('total_precipitation')
#         return ee.Feature(None, {'date': date, 'temp': temp, 'precip': precip})
#     features = collection.map(extract_vals).getInfo()['features']
#     data = [
#         (f['properties']['date'],
#          f['properties']['temp'],
#          f['properties']['precip'])
#         for f in features
#         if f['properties']['temp'] is not None and f['properties']['precip'] is not None
#     ]
#     return data

# # Model ve scaler yolları
# BASE_DIR      = os.path.dirname(__file__)
# MODEL_PATH    = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/models/lstm_temprature_model.h5"
# SCALER_X_PATH = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/scaler_X.pkl"
# SCALER_Y_PATH = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/scaler_y.pkl"

# # Model ve scaler'ları yükle
# try:
#     model     = load_model(MODEL_PATH, compile=False)
#     scaler_X  = pickle.load(open(SCALER_X_PATH, "rb"))
#     scaler_y  = pickle.load(open(SCALER_Y_PATH, "rb"))
#     print("Model ve scaler'lar başarıyla yüklendi.")
# except Exception as e:
#     print(f"Model yükleme hatası: {str(e)}")
#     sys.exit(1)

# # LSTM kaydırmalı pencere boyutu
# WINDOW_SIZE = 10

# def sliding_lstm_preds(df, window=WINDOW_SIZE):
#     """
#     df içinden son window günün (temp, month, week) dizilerini kullanarak
#     LSTM modeliyle tahminler üretir.
#     Döner: (tarihler, tahminler)
#     """
#     # Kelvin → Celsius
#     temps_C   = df['temp'].values - 273.15
#     months    = df['month'].values
#     weeks_iso = df['date'].dt.isocalendar().week.values

#     print(f"\nDebug - Veri boyutları:")
#     print(f"temps_C shape: {temps_C.shape}")
#     print(f"months shape: {months.shape}")
#     print(f"weeks_iso shape: {weeks_iso.shape}")

#     seqs = []
#     for i in range(window, len(df)):
#         raw_seq = np.vstack([
#             temps_C[i-window:i],
#             months[i-window:i],
#             weeks_iso[i-window:i]
#         ]).T  # (window, 3)
#         seqs.append(raw_seq)
#     X = np.array(seqs)  # (N, window, 3)

#     print(f"\nDebug - X shape: {X.shape}")

#     flat        = X.reshape(-1, 3)                # (N*window, 3)
#     flat_scaled = scaler_X.transform(flat)        # (N*window, 3)
#     X_scaled    = flat_scaled.reshape(-1, window, 3)  # (N, window, 3)

#     print(f"\nDebug - X_scaled shape: {X_scaled.shape}")

#     y_scaled = model.predict(X_scaled)             # (N, 1) veya (N, window, 1)
#     print(f"\nDebug - y_scaled shape: {y_scaled.shape}")
    
#     preds    = scaler_y.inverse_transform(y_scaled)[:, 0]  # (N,)
#     print(f"\nDebug - preds shape: {preds.shape}")
#     print(f"Debug - İlk 5 tahmin: {preds[:5]}")

#     dates = df['date'].iloc[window:].dt.to_pydatetime()
#     return dates, preds

# def plot_and_save(df_dates, df_temps, dates_pred, lstm_preds, region_key):
#     """
#     Sıcaklık ve kuraklık skorlarını ayrı grafik dosyalarında çizer.
#     """
#     print(f"\nDebug - Grafik verileri:")
#     print(f"df_dates shape: {len(df_dates)}")
#     print(f"df_temps shape: {len(df_temps)}")
#     print(f"dates_pred shape: {len(dates_pred)}")
#     print(f"lstm_preds shape: {len(lstm_preds)}")
    
#     # Sıcaklık grafiği
#     plt.figure(figsize=(12, 6))
#     plt.plot(df_dates, df_temps, color='tab:blue', label='Temperature')
#     plt.title(f'Temperature Trend - {region_key}')
#     plt.ylabel('Temperature (K)')
#     plt.grid(True)
#     plt.legend()
#     plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     temp_fname = f'temperature_trend_{region_key}.png'
#     plt.savefig(temp_fname)
#     print(f"\nSıcaklık grafiği kaydedildi: {temp_fname}")
#     plt.close()
    
#     # Kuraklık skoru grafiği
#     if len(lstm_preds):
#         print("\nDebug - LSTM tahminleri çiziliyor...")
#         plt.figure(figsize=(12, 6))
#         plt.plot(dates_pred, lstm_preds, color='tab:red', label='Drought Score')
#         plt.title(f'Drought Score Prediction - {region_key}')
#         plt.ylabel('Drought Score')
#         plt.grid(True)
#         plt.legend()
#         plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
#         plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#         plt.xticks(rotation=45)
#         plt.tight_layout()
        
#         drought_fname = f'drought_score_{region_key}.png'
#         plt.savefig(drought_fname)
#         print(f"Kuraklık skoru grafiği kaydedildi: {drought_fname}")
#         plt.close()
#     else:
#         print("\nDebug - LSTM tahminleri boş!")

# def plot_temperature(dates, temps, region_key):
#     """Sıcaklık grafiğini oluşturur ve base64 olarak döner"""
#     dates = pd.to_datetime(dates)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(dates, temps, color='tab:blue', label='Temperature')
#     ax.set_title(f'Temperature Trend - {region_key}')
#     ax.set_ylabel('Temperature (K)')
#     ax.set_xlabel('Date')
#     ax.grid(True)
#     ax.legend()

#     # 🔥 X ekseni: 6 ay aralık, yyyy-mm formatı, yatay
#     locator = mdates.MonthLocator(interval=6)
#     formatter = mdates.DateFormatter('%Y-%m')
#     ax.xaxis.set_major_locator(locator)
#     ax.xaxis.set_major_formatter(formatter)
#     fig.autofmt_xdate(rotation=0)  # Otomatik yerleştir, 0 derece

#     plt.tight_layout()

#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')

# def plot_drought_score(dates, scores, region_key, ndvi_values=None):
#     """Kuraklık skoru grafiğini oluşturur ve base64 olarak döner"""
#     dates = pd.to_datetime(dates)

#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
#     # Main plot for drought scores
#     ax1.plot(dates, scores, color='tab:red', label='Drought Score')
#     ax1.set_title(f'Drought Score Prediction - {region_key}')
#     ax1.set_ylabel('Drought Score')
#     ax1.grid(True)
#     ax1.legend()

#     # X-axis formatting for main plot
#     locator = mdates.MonthLocator(interval=6)
#     formatter = mdates.DateFormatter('%Y-%m')
#     ax1.xaxis.set_major_locator(locator)
#     ax1.xaxis.set_major_formatter(formatter)
#     fig.autofmt_xdate(rotation=0)

#     # Color indicators subplot
#     if ndvi_values is not None:
#         # Calculate drought percentages based on NDVI
#         ndvi_colors = []
#         for ndvi in ndvi_values:
#             if ndvi >= 0.5:
#                 ndvi_colors.append('green')
#             elif ndvi >= 0.2:
#                 ndvi_colors.append('yellow')
#             else:
#                 ndvi_colors.append('red')
        
#         # Calculate drought percentages based on LSTM scores
#         lstm_colors = []
#         for score in scores:
#             if score >= 0.5:
#                 lstm_colors.append('green')
#             elif score >= 0.2:
#                 lstm_colors.append('yellow')
#             else:
#                 lstm_colors.append('red')
        
#         # Create color bars
#         ax2.bar(dates, [1] * len(dates), color=ndvi_colors, label='NDVI Status')
#         ax2.bar(dates, [0.5] * len(dates), color=lstm_colors, label='LSTM Status')
#         ax2.set_ylabel('Drought Status')
#         ax2.set_yticks([0.25, 0.75])
#         ax2.set_yticklabels(['LSTM', 'NDVI'])
#         ax2.legend(loc='upper right')
        
#         # Remove x-axis labels from color bar subplot
#         ax2.set_xticklabels([])
    
#     plt.tight_layout()

#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')

# def get_lstm_trend_and_plot(region_key):
#     """
#     Belirli bir bölge için LSTM trendini ve grafiklerini üretir.
#     Returns:
#         tuple: (scores, plot_base64)
#     """
#     # Son 1 yıllık veriyi al
#     end_date = '2020-12-31'
#     start_date = '2014-01-01'
    
#     region_dict = {
#         "munich":      ee.Geometry.Point([11.5761, 48.1371]).buffer(20000),
#         "sanliurfa":   ee.Geometry.Point([37.1674, 38.7926]).buffer(20000),
#         "punjab":      ee.Geometry.Point([75.3412, 31.1471]).buffer(20000),
#         "gujarat":     ee.Geometry.Point([72.5714, 23.0225]).buffer(20000),
#         "yunnan":      ee.Geometry.Point([102.7123, 23.0225]).buffer(20000),
#         "nsw":         ee.Geometry.Point([151.2153, -33.8568]).buffer(20000),
#         "cordoba":     ee.Geometry.Point([-64.1810, -31.4135]).buffer(20000),
#         "gauteng":     ee.Geometry.Point([28.0473, -26.2041]).buffer(20000),
#         "kano":        ee.Geometry.Point([8.4922, 12.0526]).buffer(20000),
#         "addis_ababa": ee.Geometry.Point([38.7577, 9.0450]).buffer(20000),
#         "iowa":        ee.Geometry.Point([-93.6208, 41.5236]).buffer(20000),
#         # "zacatecas":   ee.Geometry.Point([-102.6993, 22.7735]).buffer(20000),
#         # "mato_grosso": ee.Geometry.Point([-54.6900, -12.6425]).buffer(20000)
#     }
    
#     if region_key not in region_dict:
#         raise ValueError(f"Bilinmeyen bölge: {region_key}")
    
#     print(f"ERA5 verisi çekiliyor: {region_key} ({start_date} - {end_date})")
#     region = region_dict[region_key]
#     data = get_era5_temp_precip(region, start_date, end_date)
    
#     df = pd.DataFrame(data, columns=['date', 'temp', 'precip'])
#     df['date']  = pd.to_datetime(df['date'])
#     df['month'] = df['date'].dt.month
#     df['year']  = df['date'].dt.year
    
#     if df.empty:
#         raise ValueError("Hiç veri bulunamadı!")
    
#     # LSTM tahminleri
#     if len(df) >= WINDOW_SIZE:
#         dates_pred, lstm_preds = sliding_lstm_preds(df, WINDOW_SIZE)
#         # NDVI değerlerini hesapla (örnek olarak sıcaklık değerlerini normalize ederek)
#         ndvi_values = (df['temp'].values[WINDOW_SIZE:] - df['temp'].values[WINDOW_SIZE:].min()) / \
#                      (df['temp'].values[WINDOW_SIZE:].max() - df['temp'].values[WINDOW_SIZE:].min())
#     else:
#         raise ValueError(f"LSTM tahmini için en az {WINDOW_SIZE} gün veri gerekli.")
    
#     # Grafikleri oluştur ve base64'e çevir
#     temp_plot = plot_temperature(df['date'].values, df['temp'].values, region_key)
#     drought_plot = plot_drought_score(dates_pred, lstm_preds, region_key, ndvi_values)
    
#     return lstm_preds, {'temperature_plot': temp_plot, 'drought_plot': drought_plot}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="ERA5 verisiyle sıcaklık ve yağış analizi + LSTM tahmini"
#     )
#     parser.add_argument('--region', type=str, required=True,
#                         help='Bölge adı (ör: munich, şanlıurfa, punjab, gujarat, yunnan, nsw, cordoba, gauteng, kano, addis_ababa, iowa, zacatecas, mato_grosso)')
#     parser.add_argument('--start', type=str, required=True,
#                         help='Başlangıç tarihi (YYYY-MM-DD)')
#     parser.add_argument('--end', type=str, required=True,
#                         help='Bitiş tarihi (YYYY-MM-DD)')
#     args = parser.parse_args()
    
#     try:
#         scores, plots = get_lstm_trend_and_plot(args.region)
#         print(f"\nSon kuraklık skoru: {scores[-1]:.2f}")
#         print("Grafikler oluşturuldu.")
#     except Exception as e:
#         print(f"Hata: {str(e)}")
#         sys.exit(1)


#!/usr/bin/env python3
"""
LSTM tabanlı sıcaklık ve zaman girdileriyle kuraklık skoru tahmini ve
sıcaklık trendi + LSTM tahmini modülü.

- ERA5 verisini çeker,
- Sıcaklık trendini görselleştirir,
- LSTM ile kaydırmalı pencere (sliding window) üzerinden tahmin yapar,
  her bir tahmini konsola basar ve grafik olarak kaydeder.
- Komut satırından bölge ve tarih aralığı ile çalıştırılabilir.
"""
import os
import sys
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import ee
import argparse
import matplotlib.dates as mdates

# Earth Engine kimlik doğrulama
ee.Authenticate(auth_mode='notebook')
ee.Initialize()

def get_era5_temp_precip(region, start_date, end_date):
    collection = (ee.ImageCollection('ECMWF/ERA5/DAILY')
                  .filterDate(start_date, end_date)
                  .filterBounds(region))
    def extract_vals(img):
        date = img.date().format('YYYY-MM-dd')
        temp = img.select('mean_2m_air_temperature') \
                  .reduceRegion(ee.Reducer.mean(), region, 10000) \
                  .get('mean_2m_air_temperature')
        precip = img.select('total_precipitation') \
                     .reduceRegion(ee.Reducer.mean(), region, 10000) \
                     .get('total_precipitation')
        return ee.Feature(None, {'date': date, 'temp': temp, 'precip': precip})
    features = collection.map(extract_vals).getInfo()['features']
    data = [
        (f['properties']['date'],
         f['properties']['temp'],
         f['properties']['precip'])
        for f in features
        if f['properties']['temp'] is not None and f['properties']['precip'] is not None
    ]
    return data

# Model ve scaler yolları
BASE_DIR      = os.path.dirname(__file__)
MODEL_PATH    = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/models/lstm_temprature_model.h5"
SCALER_X_PATH = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/scaler_X.pkl"
SCALER_Y_PATH = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/scaler_y.pkl"

# Model ve scaler'ları yükle
try:
    model     = load_model(MODEL_PATH, compile=False)
    scaler_X  = pickle.load(open(SCALER_X_PATH, "rb"))
    scaler_y  = pickle.load(open(SCALER_Y_PATH, "rb"))
    print("Model ve scaler'lar başarıyla yüklendi.")
except Exception as e:
    print(f"Model yükleme hatası: {str(e)}")
    sys.exit(1)

# LSTM kaydırmalı pencere boyutu
WINDOW_SIZE = 10

def sliding_lstm_preds(df, window=WINDOW_SIZE):
    """
    df içinden son window günün (temp, month, week) dizilerini kullanarak
    LSTM modeliyle tahminler üretir.
    Döner: (tarihler, tahminler)
    """
    # Kelvin → Celsius
    temps_C   = df['temp'].values - 273.15
    months    = df['month'].values
    weeks_iso = df['date'].dt.isocalendar().week.values

    print(f"\nDebug - Veri boyutları:")
    print(f"temps_C shape: {temps_C.shape}")
    print(f"months shape: {months.shape}")
    print(f"weeks_iso shape: {weeks_iso.shape}")

    seqs = []
    for i in range(window, len(df)):
        raw_seq = np.vstack([
            temps_C[i-window:i],
            months[i-window:i],
            weeks_iso[i-window:i]
        ]).T  # (window, 3)
        seqs.append(raw_seq)
    X = np.array(seqs)  # (N, window, 3)

    print(f"\nDebug - X shape: {X.shape}")

    flat        = X.reshape(-1, 3)                # (N*window, 3)
    flat_scaled = scaler_X.transform(flat)        # (N*window, 3)
    X_scaled    = flat_scaled.reshape(-1, window, 3)  # (N, window, 3)

    print(f"\nDebug - X_scaled shape: {X_scaled.shape}")

    y_scaled = model.predict(X_scaled)             # (N, 1) veya (N, window, 1)
    print(f"\nDebug - y_scaled shape: {y_scaled.shape}")
    
    preds    = scaler_y.inverse_transform(y_scaled)[:, 0]  # (N,)
    print(f"\nDebug - preds shape: {preds.shape}")
    print(f"Debug - İlk 5 tahmin: {preds[:5]}")

    dates = df['date'].iloc[window:].dt.to_pydatetime()
    return dates, preds

def plot_and_save(df_dates, df_temps, dates_pred, lstm_preds, region_key):
    """
    Sıcaklık ve kuraklık skorlarını ayrı grafik dosyalarında çizer.
    """
    print(f"\nDebug - Grafik verileri:")
    print(f"df_dates shape: {len(df_dates)}")
    print(f"df_temps shape: {len(df_temps)}")
    print(f"dates_pred shape: {len(dates_pred)}")
    print(f"lstm_preds shape: {len(lstm_preds)}")
    
    # Sıcaklık grafiği
    plt.figure(figsize=(12, 6))
    plt.plot(df_dates, df_temps, color='tab:blue', label='Temperature')
    plt.title(f'Temperature Trend - {region_key}')
    plt.ylabel('Temperature (K)')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    temp_fname = f'temperature_trend_{region_key}.png'
    plt.savefig(temp_fname)
    print(f"\nSıcaklık grafiği kaydedildi: {temp_fname}")
    plt.close()
    
    # Kuraklık skoru grafiği
    if len(lstm_preds):
        print("\nDebug - LSTM tahminleri çiziliyor...")
        plt.figure(figsize=(12, 6))
        plt.plot(dates_pred, lstm_preds, color='tab:red', label='Drought Score')
        plt.title(f'Drought Score Prediction - {region_key}')
        plt.ylabel('Drought Score')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        drought_fname = f'drought_score_{region_key}.png'
        plt.savefig(drought_fname)
        print(f"Kuraklık skoru grafiği kaydedildi: {drought_fname}")
        plt.close()
    else:
        print("\nDebug - LSTM tahminleri boş!")

def get_lstm_trend_and_plot(region_key):
    """
    Belirli bir bölge için LSTM trendini ve grafiklerini üretir.
    Returns:
        tuple: (scores, plot_base64)
    """
    # Son 1 yıllık veriyi al
    end_date = '2020-12-31'
    start_date = '2014-01-01'
    
    region_dict = {
        "munich":      ee.Geometry.Point([11.5761, 48.1371]).buffer(20000),
        "sanliurfa":   ee.Geometry.Point([37.1674, 38.7926]).buffer(20000),
        "punjab":      ee.Geometry.Point([75.3412, 31.1471]).buffer(20000),
        "gujarat":     ee.Geometry.Point([72.5714, 23.0225]).buffer(20000),
        "yunnan":      ee.Geometry.Point([102.7123, 23.0225]).buffer(20000),
        "nsw":         ee.Geometry.Point([151.2153, -33.8568]).buffer(20000),
        "cordoba":     ee.Geometry.Point([-64.1810, -31.4135]).buffer(20000),
        "gauteng":     ee.Geometry.Point([28.0473, -26.2041]).buffer(20000),
        "kano":        ee.Geometry.Point([8.4922, 12.0526]).buffer(20000),
        "addis_ababa": ee.Geometry.Point([38.7577, 9.0450]).buffer(20000),
        "iowa":        ee.Geometry.Point([-93.6208, 41.5236]).buffer(20000),
        "zacatecas":   ee.Geometry.Point([-102.6993, 22.7735]).buffer(20000),
        "mato_grosso": ee.Geometry.Point([-54.6900, -12.6425]).buffer(20000)
    }
    
    if region_key not in region_dict:
        raise ValueError(f"Bilinmeyen bölge: {region_key}")
    
    print(f"ERA5 verisi çekiliyor: {region_key} ({start_date} - {end_date})")
    region = region_dict[region_key]
    data = get_era5_temp_precip(region, start_date, end_date)
    
    df = pd.DataFrame(data, columns=['date', 'temp', 'precip'])
    df['date']  = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year']  = df['date'].dt.year
    
    if df.empty:
        raise ValueError("Hiç veri bulunamadı!")
    
    # LSTM tahminleri
    if len(df) >= WINDOW_SIZE:
        dates_pred, lstm_preds = sliding_lstm_preds(df, WINDOW_SIZE)
    else:
        raise ValueError(f"LSTM tahmini için en az {WINDOW_SIZE} gün veri gerekli.")
    
    # Grafikleri oluştur ve base64'e çevir
    temp_plot = plot_temperature(df['date'].values, df['temp'].values, region_key)
    drought_plot = plot_drought_score(dates_pred, lstm_preds, region_key)
    
    return lstm_preds, {'temperature_plot': temp_plot, 'drought_plot': drought_plot}

def plot_temperature(dates, temps, region_key):
    """Sıcaklık grafiğini oluşturur ve base64 olarak döner"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, temps, color='tab:blue', label='Temperature')
    plt.title(f'Temperature Trend - {region_key}')
    plt.ylabel('Temperature (K)')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Grafiği base64'e çevir
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_drought_score(dates, scores, region_key):
    """Kuraklık skoru grafiğini oluşturur ve base64 olarak döner"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, scores, color='tab:red', label='Drought Score')
    plt.title(f'Drought Score Prediction - {region_key}')
    plt.ylabel('Drought Score')
    plt.grid(True)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Grafiği base64'e çevir
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ERA5 verisiyle sıcaklık ve yağış analizi + LSTM tahmini"
    )
    parser.add_argument('--region', type=str, required=True,
                        help='Bölge adı (ör: munich, şanlıurfa, punjab, gujarat, yunnan, nsw, cordoba, gauteng, kano, addis_ababa, iowa, zacatecas, mato_grosso)')
    parser.add_argument('--start', type=str, required=True,
                        help='Başlangıç tarihi (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                        help='Bitiş tarihi (YYYY-MM-DD)')
    args = parser.parse_args()
    
    try:
        scores, plots = get_lstm_trend_and_plot(args.region)
        print(f"\nSon kuraklık skoru: {scores[-1]:.2f}")
        print("Grafikler oluşturuldu.")
    except Exception as e:
        print(f"Hata: {str(e)}")
        sys.exit(1)