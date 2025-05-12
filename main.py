import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction')

import pandas as pd
from src.gee.gee_handler import init_gee
from src.gee.data_loader import get_monthly_ndvi
from src.models.lstm_model import build_lstm_model
from src.preprocessing.data_prep import create_dataset, scale_data
from src.evaluation.metrics import calculate_metrics, plot_predictions

#  GEE Oturumunu Başlat
init_gee()

import ee
#  Veri Çek
region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', 'Turkey')).geometry()
ndvi_df = get_monthly_ndvi(region, start_year=2020, end_year=2024, scale=250)
ndvi_df.dropna(inplace=True)
print(" NDVI Verisi Alındı!")

#  Veriyi İşle
ndvi_df["date"] = pd.to_datetime(ndvi_df["year"].astype(str) + "-" + ndvi_df["month"].astype(str) + "-01")
ndvi_df.set_index("date", inplace=True)

scaled_ndvi, scaler = scale_data(ndvi_df[["ndvi"]])

window_size = 6
X, y = create_dataset(scaled_ndvi, window_size)

# LSTM Modelini Kur ve Eğit
model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

#  Tahmin Yap
y_pred = model.predict(X)
y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y)

#  Sonuçları Değerlendir
mae, mse = calculate_metrics(y_true_inv, y_pred_inv)
print(f" MAE: {mae:.4f}, MSE: {mse:.4f}")

#  Grafik Çiz
plot_predictions(y_true_inv, y_pred_inv)

#  Modeli Kaydet
model.save("/content/drive/MyDrive/food_crisis_prediction/models/lstm_ndvi_model.h5")
print(" Model Kaydedildi.")
