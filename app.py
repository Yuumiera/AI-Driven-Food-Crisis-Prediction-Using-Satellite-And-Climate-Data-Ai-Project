from flask import Flask, jsonify, request, send_from_directory
from src.analyze_drought import analyze_drought_for_region
from src.tester.predict_ndvi import predict_ndvi
from src.ananlys_era5 import get_lstm_trend_and_plot
import logging
import traceback
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='frontend')

# Global dictionary to store LSTM calculation status and results
lstm_calculations = {}
executor = ThreadPoolExecutor(max_workers=4)  # Maksimum 4 eşzamanlı hesaplama

def calculate_lstm(region):
    try:
        scores, plots = get_lstm_trend_and_plot(region)
        lstm_calculations[region] = {
            'status': 'completed',
            'data': {
                'score': float(scores[-1]),
                'temperature_plot': plots['temperature_plot'],
                'drought_plot': plots['drought_plot']
            }
        }
    except Exception as e:
        logger.error(f"Error calculating LSTM for region {region}: {str(e)}")
        logger.error(traceback.format_exc())
        lstm_calculations[region] = {
            'status': 'error',
            'error': str(e)
        }

# Serve static files from frontend directory
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/analyze_drought')
def analyze_drought():
    region = request.args.get('region')
    if not region:
        return jsonify({'error': 'Region parameter is required'}), 400

    try:
        logger.debug(f"Starting drought analysis for region: {region}")
        result = analyze_drought_for_region(region)
        if result is None or result[0] is None:
            logger.error(f"No results found for region: {region}")
            return jsonify({'error': f'Could not analyze region: {region}'}), 404

        drought_ratio, heatmap_image, classification_image, stats = result
        logger.debug(f"Analysis completed successfully for region: {region}")
        return jsonify({
            'drought_ratio': float(drought_ratio),
            'heatmap_image': heatmap_image,
            'classification_image': classification_image,
            'stats': stats
        })

    except Exception as e:
        logger.error(f"Error analyzing region {region}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ndvi')
def predict_ndvi_endpoint():
    region = request.args.get('region')
    if not region:
        return jsonify({'error': 'Region parameter is required'}), 400

    try:
        logger.debug(f"Starting NDVI prediction for region: {region}")
        results = predict_ndvi(region, generate_plot=True)
        if results is None:
            logger.error(f"No results found for region: {region}")
            return jsonify({'error': f'Could not predict NDVI for region: {region}'}), 404

        logger.debug(f"NDVI prediction completed successfully for region: {region}")
        return jsonify(results)

    except FileNotFoundError as e:
        logger.error(f"Data file not found for region {region}: {str(e)}")
        return jsonify({'error': f'Data file not found for region {region}: {str(e)}'}), 404
    except Exception as e:
        logger.error(f"Error during NDVI prediction for region {region}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error during NDVI prediction: {str(e)}'}), 500

@app.route('/predict_lstm_drought')
def predict_lstm_drought():
    region = request.args.get('region')
    if not region:
        return jsonify({'error': 'Region parameter is required'}), 400
    
    # Check if calculation is already in progress
    if region in lstm_calculations:
        if lstm_calculations[region]['status'] == 'completed':
            return jsonify(lstm_calculations[region]['data'])
        elif lstm_calculations[region]['status'] == 'error':
            return jsonify({'error': lstm_calculations[region]['error']}), 500
        else:
            return jsonify({'status': 'calculating'})
    
    # Start new calculation with timeout
    lstm_calculations[region] = {'status': 'calculating'}
    try:
        future = executor.submit(calculate_lstm, region)
        future.result(timeout=300)  # 5 dakika timeout
    except TimeoutError:
        lstm_calculations[region] = {
            'status': 'error',
            'error': 'Calculation timeout - please try again'
        }
        return jsonify({'error': 'Calculation timeout - please try again'}), 500
    except Exception as e:
        logger.error(f"Error in LSTM calculation for region {region}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
    return jsonify({'status': 'calculating'})

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0') 