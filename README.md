# AI-Driven Food Crisis Prediction Using Satellite and Climate Data

An end-to-end system that combines satellite imagery, climate reanalysis and real-time news to forecast droughts and food-security risks before they hit.

---

## Technical Overview

### Data Sources

* **Satellite Data**: Google Earth Engine for NDVI, EVI and land-cover indices
* **Climate Data**: ERA5 reanalysis (temperature, precipitation, soil moisture, radiation)
* **News Data**: NewsAPI feed + BERT-based sentiment & topic modeling

### AI Models & Pipelines

1. **CNN (ResNet-18)**

   * **Input**: Multi-band satellite tiles (NDVI, temperature, precipitation)
   * **Output**: Drought / no-drought classification + severity score
   * **Metrics**: 99.7% accuracy, F1-score 0.98

2. **Bi-LSTM for NDVI Prediction**

   * **Input**: 30-day windows of climate & vegetation indices
   * **Output**: 7-day NDVI forecast
   * **Metrics**:

     * **Training**: MAE 0.069, MSE 0.0088, R² 0.7194
     * **Validation**: MAE 0.0729, MSE 0.0104, R² 0.7265

3. **Bi-LSTM for ERA5 Forecasting**

   * **Input**: Historical ERA5 climate variables
   * **Output**: Drought-risk forecast
   * **Metrics**: MAE 0.0647, MSE 0.0069, R² 0.8453

4. **News Sentiment & Risk Scoring**

   * **Preprocessing**: Tokenization, stop-word removal, agricultural term tagging
   * **Model**: Fine-tuned BERT + custom keyword extractor
   * **Output**: Daily risk index & trending topics dashboard

---

## Tech Stack

* **Backend**: Python 3.8+, Flask 3.0
* **ML**: TensorFlow 2.x, scikit-learn
* **Data**: NumPy, Pandas, xarray
* **Visualization**: Matplotlib, OpenCV, Leaflet.js
* **Frontend**: HTML5, CSS3, vanilla JS (+ D3.js)

---

## Features

* **Interactive Map**: Draw a polygon → view historical & forecast drought maps
* **Time-Series Viewer**: Click a point → plot NDVI & rainfall trends
* **News Dashboard**: Live sentiment gauge + risk-term word-cloud
* **Ensemble Engine**: CNN + LSTMs + news → unified food-security score

---

## Project Structure

```
├── ai_model/
│   ├── cnn_model.py
│   ├── lstm_ndvi.py          # NDVI LSTM implementation
│   ├── lstm_era5.py         # ERA5 LSTM implementation
│   └── news_analyzer.py
├── src/
│   ├── data/                # ingest & preprocess scripts
│   ├── gee/                 # Earth Engine calls
│   └── evaluation/          # metrics & plots
├── notebooks/               # EDA & prototyping
├── frontend/
│   ├── static/              # CSS, JS, images
│   └── templates/           # Flask HTML
├── models/                  # saved weights
├── results/                 # sample outputs
├── app.py                   # Flask entrypoint
├── main.py                  # CLI for train/infer
└── requirements.txt
```

---

## Prerequisites

* Python 3.8+
* Google Earth Engine account & authenticated API
* ERA5 API key (CDS)
* ≥16 GB RAM + GPU recommended
* ≥100 GB disk space

---

## Installation

```bash
git clone https://github.com/ahmetbekir22/AI-Driven-Food-Crisis-Prediction.git
cd AI-Driven-Food-Crisis-Prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GEE_CREDENTIALS="path/to/credentials.json"
export ERA5_API_KEY="your_era5_key"
```

---

## Usage

### Web App

```bash
python app.py
```

Open `http://localhost:5000`

### CLI

```bash
# Train CNN
python main.py --train --model cnn --epochs 50 --batch 32

# Train NDVI LSTM
python main.py --train --model ndvi_lstm --epochs 100 --batch 64

# Train ERA5 LSTM
python main.py --train --model era5_lstm --epochs 100 --batch 64

# Inference
python main.py --infer --region "Sahara" --forecast 7
```

---

## Experiment Setup

* **Hardware**: NVIDIA GPU, 16 GB RAM
* **Optimizers**: Adam (CNN lr=1e-4, LSTMs lr=1e-3)
* **Batch Sizes**: 32 (CNN), 64 (LSTMs)
* **Epochs**: up to 50 (CNN), 100 (LSTMs)
* **Early Stopping**: monitor val\_loss, patience = 5 epochs

---

## Model Training & Evaluation

1. **Prepare Data**

   ```bash
   python src/data/prepare_data.py --raw data/raw --out data/processed
   ```
2. **Training**

   * CNN: `--epochs 50 --batch 32`
   * NDVI LSTM: `--epochs 100 --batch 64`
   * ERA5 LSTM: `--epochs 100 --batch 64`
3. **Evaluation**

   ```bash
   python src/evaluation/evaluate_models.py --model_dir models/
   ```

### Performance Metrics

#### CNN

* **Accuracy**: 99.7 %
* **F1-Score**: 0.98

#### NDVI LSTM

* **Training**:

  * MAE: 0.069
  * MSE: 0.0088
  * R²: 0.7194
* **Validation**:

  * MAE: 0.0729
  * MSE: 0.0104
  * R²: 0.7265

#### ERA5 LSTM

* **MAE**: 0.0647
* **MSE**: 0.0069
* **R² Score**: 0.8453

---

## Summary & Conclusions

With early stopping (patience 5) to prevent overfitting, our ResNet-18 CNN achieves 99.7 % accuracy on drought classification. The Bi-LSTM NDVI model forecasts vegetation index with validation MAE 0.0729, and the ERA5 LSTM forecasts climate-driven drought risk with MAE 0.0647. Combined with real-time news sentiment, this tool enables robust, early food-security warnings.

---
