"""
LSTM tabanlı sıcaklık ve zaman girdileriyle kuraklık skoru tahmini modülü.
Bu modül, model ve scaler'ları yükler ve tahmin fonksiyonu sunar.
Doğrudan çalıştırılırsa CLI üzerinden test edilebilir.
"""
import os
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
ee.Authenticate(auth_mode='notebook')
ee.Initialize()

def get_era5_temp_precip(region, start_date, end_date):
    collection = (ee.ImageCollection('ECMWF/ERA5/DAILY')
                  .filterDate(start_date, end_date)
                  .filterBounds(region))
    def extract_vals(img):
        date = img.date().format('YYYY-MM-dd')
        temp = img.select('mean_2m_air_temperature').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10000).get('mean_2m_air_temperature')
        precip = img.select('total_precipitation').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10000).get('total_precipitation')
        return ee.Feature(None, {'date': date, 'temp': temp, 'precip': precip})
    features = collection.map(extract_vals).getInfo()['features']
    data = [(f['properties']['date'], f['properties']['temp'], f['properties']['precip'])
            for f in features if f['properties']['temp'] is not None and f['properties']['precip'] is not None]
    return data


# Örnek veri: [('2023-06-01', 25.3), ...]
region = ee.Geometry.Point([32.8541, 39.9333])
start_date = '2023-01-01'
end_date = '2023-12-31'
data = get_era5_temp_precip(region, start_date, end_date)
df = pd.DataFrame(data, columns=['date', 'temp', 'precip'])
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Model ve scaler'ları bir kez yükle
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/lstm_temprature_model.h5')
SCALER_X_PATH = os.path.join(os.path.dirname(__file__), '../scaler_X.pkl')
SCALER_Y_PATH = os.path.join(os.path.dirname(__file__), '../scaler_y.pkl')

model = load_model(MODEL_PATH, compile=False)
scaler_X = pickle.load(open(SCALER_X_PATH, "rb"))
scaler_y = pickle.load(open(SCALER_Y_PATH, "rb"))

def predict_drought_score(temp: float, month: int, week: int) -> float:
    """
    LSTM modelini kullanarak kuraklık skorunu tahmin eder.
    Args:
        temp (float): Son gün sıcaklık (°C)
        month (int): Ay (1-12)
        week (int): Hafta (1-53)
    Returns:
        float: Tahmini kuraklık skoru
    """
    raw = np.array([[temp, month, week]])
    x_scaled = scaler_X.transform(raw)
    X_input = np.repeat(x_scaled[np.newaxis, :, :], 10, axis=1)  # (1,10,3)
    y_scaled = model.predict(X_input)
    score = float(scaler_y.inverse_transform(y_scaled)[0, 0])
    return score

def predict_drought_trend(temps, months, weeks):
    """
    LSTM modelini kullanarak birden fazla zaman noktası için kuraklık skorları tahmin eder.
    Args:   
        temps (list of float): Sıcaklık değerleri
        months (list of int): Ay değerleri
        weeks (list of int): Hafta değerleri
    Returns:
        list of float: Tahmini kuraklık skorları
    """
    scores = []
    for temp, month, week in zip(temps, months, weeks):
        score = predict_drought_score(temp, month, week)
        scores.append(score)
    return scores

def plot_lstm_drought_trend(dates, scores, region_name):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, scores, marker='o', label='LSTM Drought Score')
    plt.title(f'LSTM Drought Trend - {region_name}')
    plt.xlabel('Date')
    plt.ylabel('Drought Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return plot_base64

# Örnek: 12 aylık trend tahmini ve plot (test amaçlı)
def get_lstm_trend_and_plot(region_name):
    # Burada örnek olarak sabit sıcaklık, ay ve hafta değerleri kullanıyoruz
    months = list(range(1, 13))
    weeks = [23]*12
    temps = [25.0]*12
    dates = [f'2025-{m:02d}' for m in months]
    scores = predict_drought_trend(temps, months, weeks)
    plot_base64 = plot_lstm_drought_trend(dates, scores, region_name)
    return scores, plot_base64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERA5 verisiyle sıcaklık ve yağış analizi")
    parser.add_argument('--region', type=str, required=True, help='Bölge adı (ör: ankara)')
    parser.add_argument('--start', type=str, required=True, help='Başlangıç tarihi (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='Bitiş tarihi (YYYY-MM-DD)')
    args = parser.parse_args()

    region_dict = {
        "ankara": ee.Geometry.Point([32.8541, 39.9333]),
        "istanbul": ee.Geometry.Point([28.9784, 41.0082]).buffer(10000),
        "munich": ee.Geometry.Point([11.5761, 48.1371]).buffer(20000),
        "şanlıurfa": ee.Geometry.Point([37.1674, 38.7926]).buffer(20000),
        "punjab": ee.Geometry.Point([75.3412, 31.1471]).buffer(20000),
        "gujarat": ee.Geometry.Point([72.5714, 23.0225]).buffer(20000),
    }
    if args.region not in region_dict:
        raise ValueError(f"Bilinmeyen bölge: {args.region}. Tanımlı bölgeler: {list(region_dict.keys())}")
    region = region_dict[args.region]

    print(f"ERA5 verisi çekiliyor: {args.region} ({args.start} - {args.end})")
    data = get_era5_temp_precip(region, args.start, args.end)
    df = pd.DataFrame(data, columns=['date', 'temp', 'precip'])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    if df.empty:
        print("Hiç veri bulunamadı!")
        exit()

    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f"Veri aralığı: {min_date.date()} - {max_date.date()}")

    last_row = df.iloc[-1]
    print(f"Son gün sıcaklık: {last_row['temp']:.2f} K, Son gün yağış: {last_row['precip']:.2f} m")

    # Son yıl ve Haziran sonrası
    last_year = max_date.year
    start_june = pd.Timestamp(year=last_year, month=6, day=1)
    if max_date < start_june:
        print(f"{last_year} yılında Haziran sonrası veri yok.")
    else:
        mask = (df['date'] >= start_june) & (df['date'] <= max_date)
        df_filtered = df.loc[mask]
        monthly_means = df_filtered.groupby(['year', 'month']).agg({'temp': 'mean', 'precip': 'mean'}).reset_index()
        print(f"\n{last_year} Haziran-{max_date.strftime('%B')} Aylık Ortalama Sıcaklık ve Yağış:")
        print(monthly_means)

    # LSTM tahmini için yeterli veri yoksa uyarı
    window_size = 10
    lstm_preds = []
    if len(df) > window_size:
        for i in range(len(df) - window_size):
            window = df['temp'].values[i:i+window_size]
            month = df['month'].values[i+window_size-1]
            week = df['date'].dt.isocalendar().week.values[i+window_size-1]
            raw = np.array([[window[-1], month, week]])
            x_scaled = scaler_X.transform(raw)
            X_input = np.repeat(x_scaled[np.newaxis, :, :], 10, axis=1)
            y_scaled = model.predict(X_input)
            pred = float(scaler_y.inverse_transform(y_scaled)[0, 0])
            lstm_preds.append(pred)
    else:
        print("LSTM tahmini için yeterli veri yok.")

    # Yeni tarih aralığı oluştur (ör: 2025-01-01'den veri uzunluğu kadar ay/gün)
    fake_start = pd.Timestamp('2025-01-01')
    date_range = pd.date_range(start=fake_start, periods=len(df), freq='D')  # Günlük veri için
    date_range_lstm = date_range[window_size:]  # LSTM tahminleri için

    plt.figure(figsize=(10,5))
    plt.plot(date_range, df['temp'], label='Temperature (K)')
    if len(lstm_preds) > 0:
        plt.plot(date_range_lstm, lstm_preds, label='LSTM Prediction', linestyle='--')

    plt.legend()
    plt.title(f'Temperature Trend & LSTM Prediction - {args.region} (2025 Takvimiyle)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (K)')

    # X eksenini ay olarak göster
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()