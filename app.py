from flask import Flask, jsonify, request, send_from_directory
from src.analyze_drought import analyze_drought_for_region
from src.tester.predict_ndvi import predict_ndvi

app = Flask(__name__, static_folder='frontend')

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
        result = analyze_drought_for_region(region)
        if result is None or result[0] is None:
            return jsonify({'error': f'Could not analyze region: {region}'}), 404

        drought_ratio, heatmap_image, classification_image, stats = result
        return jsonify({
            'drought_ratio': float(drought_ratio),
            'heatmap_image': heatmap_image,
            'classification_image': classification_image,
            'stats': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ndvi')
def predict_ndvi_endpoint():
    region = request.args.get('region')
    if not region:
        return jsonify({'error': 'Region parameter is required'}), 400

    try:
        results = predict_ndvi(region, generate_plot=True)
        if results is None:
            return jsonify({'error': f'Could not predict NDVI for region: {region}'}), 404

        return jsonify(results)

    except FileNotFoundError as e:
        return jsonify({'error': f'Data file not found for region {region}: {str(e)}'}), 404
    except Exception as e:
        return jsonify({'error': f'Error during NDVI prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0') 