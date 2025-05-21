from flask import Flask, jsonify, request, send_from_directory
import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from src.analyze_drought import analyze_drought_for_region

# Configure matplotlib for non-interactive backend
plt.ioff()  # Turn off interactive mode

app = Flask(__name__, static_folder='frontend')

# Serve static files from frontend directory
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# API endpoint for drought analysis
@app.route('/analyze_drought')
def analyze_drought():
    region = request.args.get('region')
    if not region:
        return jsonify({'error': 'Region parameter is required'}), 400

    try:
        # Direct call without multiprocessing
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
        plt.close('all')  # Clean up on error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0') 