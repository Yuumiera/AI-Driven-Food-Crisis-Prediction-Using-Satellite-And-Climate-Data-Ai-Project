import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import base64
from io import BytesIO
import gc
import torch
from src.models.cnn_model import build_cnn_model

def analyze_drought_for_region(region_name):
    """
    Analyzes the .tif prediction map for the given region name,
    generates heatmap, prints statistics and returns drought ratio.
    
    Args:
        region_name (str): Region name (e.g., 'sanliurfa')
    
    Returns:
        tuple: (drought_ratio, heatmap_image, normalized_image, stats_dict)
    """
    # Configure matplotlib for this process
    plt.ioff()
    
    try:
        print(f"Analyzing region: {region_name}")  # Debug print
        
        # Region to TIF file mapping
        REGION_TO_TIF = {
            "sanliurfa": "test_tifs/sanliurfa_05-2025.tif",
            "nsw": "test_tifs/nsw_05-2025.tif",  # NSW, Avustralya
            "munich": "test_tifs/munich_05-2025.tif",
            "cordoba": "test_tifs/cordoba_05-2025.tif",
            "gauteng": "test_tifs/gauteng_05-2025.tif",
            "kano": "test_tifs/kano_05-2025.tif",
            "addis_ababa": "test_tifs/addis_ababa_05-2025.tif",
            "yunnan": "test_tifs/yunnan_02-2025.tif",
            "punjab": "test_tifs/punjab_05-2025.tif",
            "gujarat": "test_tifs/gujarat_05-2025.tif",
            "iowa": "test_tifs/iowa_05-2025.tif",
            "zacatecas": "test_tifs/zacatecas_05-2025.tif",
            "mato_grosso": "test_tifs/mato_grosso_05-2025.tif"
        }

        tif_path = REGION_TO_TIF[region_name]
        print(f"Looking for file: {tif_path}")  # Debug print
        print(f"File exists: {os.path.exists(tif_path)}")  # Debug print
        
        if region_name not in REGION_TO_TIF:
            print(f"❌ Warning: No tif file defined for '{region_name}'.")
            return None, None, None, None

        if not os.path.exists(tif_path):
            print(f"❌ File not found: {tif_path}")
            print(f"Current working directory: {os.getcwd()}")  # Debug print
            return None, None, None, None

        # Load and prepare model
        print("Loading CNN model...")  # Debug print
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_cnn_model().to(device)
        model_path = os.path.join("models", "cnn_resnet18_drought.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")  # Debug print

        # Read TIF file
        print("Reading TIF file...")  # Debug print
        with rasterio.open(tif_path) as src:
            ndvi = src.read(1)
            profile = src.profile

        # Preprocess NDVI data
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = (ndvi + 1) / 2  # Normalize to 0-1 range

        # Define drought severity thresholds
        thresholds = {
            'severe': 0.3,    # Severe drought threshold
            'moderate': 0.5,  # Moderate drought threshold
            'mild': 0.7      # Mild drought threshold
        }

        # Calculate drought ratios for different severity levels
        drought_masks = {
            'severe': ndvi < thresholds['severe'],
            'moderate': (ndvi >= thresholds['severe']) & (ndvi < thresholds['moderate']),
            'mild': (ndvi >= thresholds['moderate']) & (ndvi < thresholds['mild'])
        }

        # Calculate weighted drought ratio
        weights = {
            'severe': 1.0,
            'moderate': 0.6,
            'mild': 0.3
        }

        total_weighted_drought = sum(
            np.sum(mask) * weight 
            for mask, weight in zip(drought_masks.values(), weights.values())
        )
        
        drought_ratio = total_weighted_drought / ndvi.size

        # Calculate statistics
        stats = {
            'mean_score': float(np.mean(ndvi)),
            'max_score': float(np.max(ndvi)),
            'min_score': float(np.min(ndvi)),
            'std_score': float(np.std(ndvi)),
            'drought_ratio': float(drought_ratio * 100),
            'severe_drought_ratio': float(np.sum(drought_masks['severe']) / ndvi.size * 100),
            'moderate_drought_ratio': float(np.sum(drought_masks['moderate']) / ndvi.size * 100),
            'mild_drought_ratio': float(np.sum(drought_masks['mild']) / ndvi.size * 100)
        }
        print("Statistics calculated")  # Debug print

        # Generate heatmap visualization
        print("Generating heatmap...")  # Debug print
        fig = plt.figure(figsize=(8, 6), dpi=72)
        plt.imshow(ndvi, cmap='YlOrRd')
        plt.colorbar(label="Drought Score")
        plt.title(f"{region_name.capitalize()} - Drought Heatmap")
        
        heatmap_buffer = BytesIO()
        fig.savefig(heatmap_buffer, format='png', bbox_inches='tight', dpi=72)
        plt.close(fig)
        heatmap_buffer.seek(0)
        heatmap_image = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
        heatmap_buffer.close()
        print("Heatmap generated")  # Debug print

        # Generate classification visualization
        print("Generating classification map...")  # Debug print
        fig = plt.figure(figsize=(8, 6), dpi=72)
        classified = np.zeros_like(ndvi)
        classified[ndvi > 0.7] = 2  # Severe
        classified[(ndvi > 0.5) & (ndvi <= 0.7)] = 1  # Moderate

        cmap = plt.get_cmap('OrRd', 3)
        plt.imshow(classified, cmap=cmap)
        plt.colorbar(ticks=[0, 1, 2], 
                    label='Drought Level\n(0: Normal, 1: Moderate, 2: Severe)')
        plt.title("Drought Classification")
        
        classification_buffer = BytesIO()
        fig.savefig(classification_buffer, format='png', bbox_inches='tight', dpi=72)
        plt.close(fig)
        classification_buffer.seek(0)
        classification_image = base64.b64encode(classification_buffer.getvalue()).decode('utf-8')
        classification_buffer.close()
        print("Classification map generated")  # Debug print

        # Clean up
        plt.close('all')
        gc.collect()

        return drought_ratio, heatmap_image, classification_image, stats

    except Exception as e:
        print(f"Error during analysis: {str(e)}")  # Debug print
        plt.close('all')
        gc.collect()
        return None, None, None, None
    finally:
        plt.close('all')
        gc.collect()