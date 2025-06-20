# AI-Driven Food Crisis Prediction Using Satellite and Climate Data

A robust end-to-end system that leverages satellite imagery, climate reanalysis, and real-time news analytics to forecast droughts and food-security risks before they escalate. This project combines deep learning, geospatial analysis, and NLP to deliver actionable early warnings for food crises.

---

## 🚀 Key Features

- **Drought Risk Mapping**: Deep learning (CNN) on satellite NDVI and climate data for pixel-level drought classification.
- **NDVI & Climate Forecasting**: Bi-LSTM models predict vegetation (NDVI) and drought risk from climate time series.
- **News Sentiment Analysis**: Real-time news scraping and BERT-based sentiment/topic modeling for food security risk signals.
- **Interactive Web Dashboard**: Visualize drought maps, trends, and news-driven risk indices for any region.
- **Modular CLI**: Train, evaluate, and infer with all models from the command line.

---

## 🏗️ Architecture Overview

### Data Sources
- **Satellite**: Google Earth Engine (NDVI, EVI, land cover)
- **Climate**: ERA5 reanalysis (temperature, precipitation, soil moisture, radiation)
- **News**: NewsAPI + NLP (BERT/TextBlob)

### Model Pipelines
- **CNN (ResNet-18)**: Classifies drought severity from satellite tiles.
- **Bi-LSTM (NDVI)**: Predicts 7-day NDVI from 30-day climate/vegetation windows.
- **Bi-LSTM (ERA5)**: Forecasts drought risk from historical climate variables.
- **News Analyzer**: Scrapes, tags, and scores news for food/climate crisis signals.

### System Flow
1. **Data Ingestion** → 2. **Preprocessing** → 3. **Model Inference/Training** → 4. **Visualization & Dashboard**

---

## 📁 Project Structure

```
├── ai_model/
│   ├── news_analyzer.py         # News scraping & sentiment analysis
│   └── requirements.txt         # NLP dependencies
├── src/
│   ├── analyze_drought.py       # Drought risk analysis (CNN)
│   ├── ananlys_era5.py          # LSTM/ERA5 drought forecasting
│   ├── gee/                     # Google Earth Engine data loaders
│   ├── preprocessing/           # Data prep & patch generation
│   ├── train/                   # Model training scripts
│   ├── tester/                  # Inference & testing
│   ├── evaluation/              # Metrics & plots
│   └── models/                  # Model architectures & training/inference code
├── notebooks/                   # EDA, prototyping, and data extraction
│   └── ndvi_data/               # Example NDVI CSVs for regions
├── models/                      # Saved model weights (e.g., .h5, .pth files)
├── results/                     # Output maps, plots, and reports
├── frontend/                    # Static web assets (HTML, JS, CSS)
├── app.py                       # Flask web app entrypoint
├── main.py                      # CLI for training/inference
├── requirements.txt             # Main dependencies
└── README.md
```

> **Note:**
> - `src/models/` contains all model architectures (CNN, LSTM, etc.) and their training/inference scripts.
> - `models/` (root directory) stores only the saved weights of trained models (not code).

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Google Earth Engine account & credentials
- ERA5 API key (CDS)
- ≥16 GB RAM (GPU recommended)
- ≥100 GB disk space

### Environment Setup
```bash
git clone https://github.com/ahmetbekir22/AI-Driven-Food-Crisis-Prediction.git
cd AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r ai_model/requirements.txt
```

### API Keys & Credentials
```bash
export GEE_CREDENTIALS="path/to/credentials.json"
export ERA5_API_KEY="your_era5_key"
```

---

## 🖥️ Usage

### Web Application
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### Command-Line Interface (CLI)
```bash
# Train CNN
python main.py --train --model cnn --epochs 50 --batch 32

# Train NDVI LSTM
python main.py --train --model ndvi_lstm --epochs 100 --batch 64

# Train ERA5 LSTM
python main.py --train --model era5_lstm --epochs 100 --batch 64

# Inference (example)
python main.py --infer --region "Sanliurfa" --forecast 7
```

---

## 🏋️ Model Training & Evaluation

### Data Preparation
```bash
python src/preprocessing/data_prep.py --raw data/raw --out data/processed
```

### Training
- CNN: `--epochs 50 --batch 32`
- NDVI LSTM: `--epochs 100 --batch 64`
- ERA5 LSTM: `--epochs 100 --batch 64`

### Evaluation
```bash
python src/evaluation/metrics.py --model_dir models/
```

#### Performance Metrics
- **CNN**: Accuracy 99.7%, F1-Score 0.98
- **NDVI LSTM**: MAE 0.0729, MSE 0.0104, R² 0.7265 (validation)
- **ERA5 LSTM**: MAE 0.0647, MSE 0.0069, R² 0.8453

---

## 📊 Visualization & Outputs
- **Drought Maps**: Heatmaps and risk classifications for each region
- **Time-Series Plots**: NDVI and temperature/drought trends
- **News Dashboard**: Sentiment gauge and risk-term word clouds
- **Results**: All outputs saved in the `results/` directory

---

## 🤝 Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes.

---

## 📄 License
This project is licensed under the MIT License.

---

## 📬 Contact
- **Author**: Ahmet Bekir
- **GitHub**: [ahmetbekir22](https://github.com/ahmetbekir22)
- **Email**: [your-email@example.com]

---
