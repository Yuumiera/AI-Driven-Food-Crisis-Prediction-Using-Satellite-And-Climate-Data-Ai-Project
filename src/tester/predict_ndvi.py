import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import base64
from io import BytesIO
import logging
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Region to CSV file mapping
REGION_TO_CSV = {
    "sanliurfa": "ndvi_punjab.csv",  # Geçici olarak Punjab verilerini kullan
    "avustralia": "ndvi_nsw.csv",
    "munich": "ndvi_munich.csv",
    "cordoba": "ndvi_cordoba.csv",
    "gauteng": "ndvi_gauteng.csv",
    "kano": "ndvi_kano.csv",
    "addis_ababa": "ndvi_addis_ababa.csv",
    "yunnan": "ndvi_yunnan.csv",
    "punjab": "ndvi_punjab.csv",
    "gujarat": "ndvi_gujarat.csv"
}

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_csv_path(region_name):
    """Get the CSV file path for a given region"""
    logger.debug(f"Looking up CSV path for region: {region_name}")
    logger.debug(f"Available regions: {list(REGION_TO_CSV.keys())}")
    
    region_key = region_name.lower()
    logger.debug(f"Normalized region name: {region_key}")
    
    csv_filename = REGION_TO_CSV.get(region_key)
    if not csv_filename:
        error_msg = f"No CSV file mapping found for region: {region_name}. Available regions are: {list(REGION_TO_CSV.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    csv_path = os.path.join("notebooks", "ndvi_data", csv_filename)
    logger.debug(f"Constructed CSV path: {csv_path}")
    return csv_path

def generate_ndvi_plot(true_values, predicted_values, future_predictions, region_name, dates):
    """Generate NDVI plot and return as base64 string"""
    logger.debug(f"Generating NDVI plot for region: {region_name}")
    try:
        plt.figure(figsize=(14, 7))  # Grafik boyutunu büyüttüm
        
        # Tarih aralığını oluştur
        start_date = pd.to_datetime('2020-01-01')
        date_range = pd.date_range(start=start_date, periods=len(true_values), freq='MS')  # MS: Month Start
        future_dates = pd.date_range(start=date_range[-1] + pd.DateOffset(months=1), 
                                   periods=len(future_predictions), freq='MS')
        
        # Ana grafik
        plt.plot(date_range, true_values, label="Actual NDVI", linewidth=2)
        plt.plot(date_range, predicted_values, label="Predicted NDVI", linewidth=2)
        plt.plot(future_dates, future_predictions, label="Future Predictions", 
                linestyle="--", marker="o", linewidth=2)
        
        # X ekseni formatı
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Her yıl için bir tick
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Yıl-Ay formatı
        plt.gcf().autofmt_xdate()  # Tarih etiketlerini eğ
        
        plt.title(f"NDVI Prediction – {region_name.capitalize()}", pad=20, fontsize=14)
        plt.xlabel("Date", labelpad=10, fontsize=12)
        plt.ylabel("NDVI", labelpad=10, fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Y ekseni sınırlarını ayarla
        all_values = true_values + predicted_values + future_predictions
        y_min = min(all_values) - 0.1
        y_max = max(all_values) + 0.1
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)  # DPI'ı artırdım
        plt.close()
        buffer.seek(0)
        plot_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        
        logger.debug("Plot generated successfully")
        return plot_image
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        raise

def predict_ndvi(
    region_name,
    csv_folder="notebooks/ndvi_data",
    model_path=os.path.join(PROJECT_ROOT, "models", "lstm_ndvi_model.h5"),  # Absolute path kullan
    window_size=6,
    future_months=5,
    generate_plot=True
):
    logger.info(f"Starting NDVI prediction for region: {region_name}")
    logger.debug(f"Using model path: {model_path}")
    logger.debug(f"Model file exists: {os.path.exists(model_path)}")
    
    # --- CSV dosyasını bul ---
    try:
        logger.debug("Attempting to get CSV file path")
        csv_file = get_csv_path(region_name)
        logger.debug(f"CSV file path: {csv_file}")
    except ValueError as e:
        logger.error(f"ValueError in get_csv_path: {str(e)}")
        raise FileNotFoundError(str(e))
    
    if not os.path.exists(csv_file):
        error_msg = f"CSV file not found at path: {csv_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.debug("Reading CSV file")
    try:
        df = pd.read_csv(csv_file).dropna()
        logger.debug(f"CSV file read successfully. Shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        logger.debug(f"First few rows:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise

    try:
        ndvi = df["ndvi"].values.reshape(-1, 1)
        logger.debug(f"NDVI data shape: {ndvi.shape}")
    except KeyError as e:
        logger.error(f"Column 'ndvi' not found in CSV. Available columns: {df.columns.tolist()}")
        raise

    # --- Ölçekleme ---
    logger.debug("Starting data scaling")
    scaler = MinMaxScaler()
    ndvi_scaled = scaler.fit_transform(ndvi)
    logger.debug(f"Scaled data shape: {ndvi_scaled.shape}")

    # --- Giriş / Çıkış dizileri ---
    logger.debug("Creating input/output sequences")
    X = []
    for i in range(len(ndvi_scaled) - window_size):
        X.append(ndvi_scaled[i:i + window_size])
    X = np.array(X)
    y = ndvi_scaled[window_size:]
    logger.debug(f"Input shape: {X.shape}, Output shape: {y.shape}")

    # --- Modeli yükle ---
    logger.debug(f"Loading model from: {model_path}")
    try:
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at path: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        model = load_model(model_path, compile=False)
        logger.debug("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    # --- Tahmin ---
    logger.debug("Making predictions")
    try:
        y_pred = model.predict(X, verbose=0)
        y_true = scaler.inverse_transform(y)
        y_pred_inv = scaler.inverse_transform(y_pred)
        logger.debug("Predictions completed successfully")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

    # --- Son NDVI değeri ---
    last_ndvi = y_pred_inv[-1][0]
    logger.debug(f"Last NDVI value: {last_ndvi}")

    # --- Metrikler ---
    logger.debug("Calculating metrics")
    try:
        mae = mean_absolute_error(y_true, y_pred_inv)
        mse = mean_squared_error(y_true, y_pred_inv)
        r2 = r2_score(y_true, y_pred_inv)
        logger.debug(f"Metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

    # --- Gelecek tahmin ---
    logger.debug("Making future predictions")
    try:
        last_window = ndvi_scaled[-window_size:]
        future_preds = []
        for i in range(future_months):
            logger.debug(f"Predicting month {i+1}/{future_months}")
            input_seq = last_window.reshape(1, window_size, 1)
            pred_scaled = model.predict(input_seq, verbose=0)
            future_preds.append(pred_scaled[0][0])
            last_window = np.concatenate([last_window[1:], pred_scaled.reshape(1, 1)], axis=0)

        future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        logger.debug("Future predictions completed")
    except Exception as e:
        logger.error(f"Error during future predictions: {str(e)}")
        raise

    # --- Gelecek aylar ---
    logger.debug("Calculating future dates")
    try:
        last_date = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str)).max()
        future_dates = [last_date + relativedelta(months=i + 1) for i in range(future_months)]
        future_labels = [d.strftime("%Y-%m") for d in future_dates]
        logger.debug(f"Future dates: {future_labels}")
    except Exception as e:
        logger.error(f"Error calculating future dates: {str(e)}")
        raise

    # CSV'den tarihleri al
    try:
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
        dates = df['date'].tolist()
        logger.debug(f"Date range: {dates[0]} to {dates[-1]}")
    except Exception as e:
        logger.error(f"Error processing dates: {str(e)}")
        raise

    # --- Grafik oluştur ---
    plot_image = None
    if generate_plot:
        logger.debug("Generating plot")
        try:
            plot_image = generate_ndvi_plot(
                y_true.flatten().tolist(),
                y_pred_inv.flatten().tolist(),
                future_preds_inv.flatten().tolist(),
                region_name,
                dates
            )
            logger.debug("Plot generated successfully")
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            raise

    # --- Çıktılar ---
    logger.debug("Preparing results dictionary")
    results = {
        "region": region_name,
        "last_ndvi": float(last_ndvi),
        "true_values": y_true.flatten().tolist(),
        "predicted_values": y_pred_inv.flatten().tolist(),
        "future_predictions": future_preds_inv.flatten().tolist(),
        "future_ndvi": {label: float(val[0]) for label, val in zip(future_labels, future_preds_inv)},
        "metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2)
        }
    }
    
    if plot_image:
        results["plot_image"] = plot_image

    logger.info(f"NDVI prediction completed successfully for region: {region_name}")
    return results
