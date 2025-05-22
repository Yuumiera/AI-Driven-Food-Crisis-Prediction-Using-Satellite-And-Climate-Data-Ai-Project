# import numpy as np
# import rasterio
# import matplotlib
# matplotlib.use('Agg')  # Set backend before importing pyplot
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# import os
# import base64
# from io import BytesIO
# import gc
# import torch
# from src.models.cnn_model import build_cnn_model

# def analyze_drought_for_region(region_name):
#     """
#     Analyzes the .tif prediction map for the given region name,
#     generates heatmap, prints statistics and returns drought ratio.
    
#     Args:
#         region_name (str): Region name (e.g., 'sanliurfa')
    
#     Returns:
#         tuple: (drought_ratio, heatmap_image, normalized_image, stats_dict)
#     """
#     # Configure matplotlib for this process
#     plt.ioff()
    
#     try:
#         print(f"Analyzing region: {region_name}")  # Debug print
        
#         # Region to TIF file mapping
#         REGION_TO_TIF = {
#             "sanliurfa": "test_tifs/sanliurfa_05-2025.tif",
#             "nsw": "test_tifs/nsw_05-2025.tif",  # NSW, Avustralya
#             "munich": "test_tifs/munich_05-2025.tif",
#             "cordoba": "test_tifs/cordoba_05-2025.tif",
#             "gauteng": "test_tifs/gauteng_05-2025.tif",
#             "kano": "test_tifs/kano_05-2025.tif",
#             "addis_ababa": "test_tifs/addis_ababa_05-2025.tif",
#             "yunnan": "test_tifs/yunnan_02-2025.tif",
#             "punjab": "test_tifs/punjab_05-2025.tif",
#             "gujarat": "test_tifs/gujarat_05-2025.tif",
#             "iowa": "test_tifs/iowa_05-2025.tif",
#             "zacatecas": "test_tifs/zacatecas_05-2025.tif",
#             "mato_grosso": "test_tifs/mato_grosso_05-2025.tif"
#         }

#         tif_path = REGION_TO_TIF[region_name]
#         print(f"Looking for file: {tif_path}")  # Debug print
#         print(f"File exists: {os.path.exists(tif_path)}")  # Debug print
        
#         if region_name not in REGION_TO_TIF:
#             print(f"❌ Warning: No tif file defined for '{region_name}'.")
#             return None, None, None, None

#         if not os.path.exists(tif_path):
#             print(f"❌ File not found: {tif_path}")
#             print(f"Current working directory: {os.getcwd()}")  # Debug print
#             return None, None, None, None

#         # Load and prepare model
#         print("Loading CNN model...")  # Debug print
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = build_cnn_model().to(device)
#         model_path = os.path.join("models", "cnn_resnet18_drought.pth")
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()
#         print("Model loaded successfully")  # Debug print

#         # Read TIF file
#         print("Reading TIF file...")  # Debug print
#         with rasterio.open(tif_path) as src:
#             drought_prediction = src.read(1)  # Model's drought prediction
#             profile = src.profile

#         # Normalize prediction values to 0-1 range
#         drought_prediction = np.clip(drought_prediction, 0, 1)

#         # Calculate drought ratios for different severity levels
#         drought_masks = {
#             'severe': drought_prediction >= 0.7,    # Severe drought (70% risk)
#             'moderate': (drought_prediction >= 0.45) & (drought_prediction < 0.7),  # Moderate drought (45-70% risk)
#             'mild': (drought_prediction >= 0.25) & (drought_prediction < 0.45)  # Mild drought (25-45% risk)
#         }

#         # Calculate weighted drought ratio
#         weights = {
#             'severe': 1.0,
#             'moderate': 0.6,
#             'mild': 0.3
#         }

#         total_weighted_drought = sum(
#             np.sum(mask) * weight 
#             for mask, weight in zip(drought_masks.values(), weights.values())
#         )
        
#         drought_ratio = total_weighted_drought / drought_prediction.size

#         # Calculate statistics
#         stats = {
#             'mean_score': float(np.mean(drought_prediction)),
#             'max_score': float(np.max(drought_prediction)),
#             'min_score': float(np.min(drought_prediction)),
#             'std_score': float(np.std(drought_prediction)),
#             'drought_ratio': float(drought_ratio * 100),
#             'severe_drought_ratio': float(np.sum(drought_masks['severe']) / drought_prediction.size * 100),
#             'moderate_drought_ratio': float(np.sum(drought_masks['moderate']) / drought_prediction.size * 100),
#             'mild_drought_ratio': float(np.sum(drought_masks['mild']) / drought_prediction.size * 100)
#         }
#         print("Statistics calculated")  # Debug print

#         # Generate heatmap visualization
#         print("Generating heatmap...")  # Debug print
#         fig = plt.figure(figsize=(8, 6), dpi=72)
        
#         # Create custom colormap for drought visualization
#         colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green -> Orange -> Red
#         n_bins = 100
#         custom_cmap = LinearSegmentedColormap.from_list("custom_drought", colors, N=n_bins)
        
#         # Plot with fixed vmin and vmax for consistent color scale
#         plt.imshow(drought_prediction, cmap=custom_cmap, vmin=0, vmax=1)
#         cbar = plt.colorbar(label="Drought Risk")
        
#         # Add threshold lines to colorbar
#         cbar.ax.axhline(y=0.25, color='white', linestyle='--', linewidth=1)
#         cbar.ax.axhline(y=0.55, color='white', linestyle='--', linewidth=1)
        
#         # Add threshold labels
#         cbar.ax.text(1.5, 0.25, '25%', color='white', ha='left', va='center')
#         cbar.ax.text(1.5, 0.55, '55%', color='white', ha='left', va='center')
        
#         plt.title(f"{region_name.capitalize()} - Drought Risk Map")
        
#         heatmap_buffer = BytesIO()
#         fig.savefig(heatmap_buffer, format='png', bbox_inches='tight', dpi=72)
#         plt.close(fig)
#         heatmap_buffer.seek(0)
#         heatmap_image = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
#         heatmap_buffer.close()
#         print("Heatmap generated")  # Debug print

#         # Generate classification visualization
#         print("Generating classification map...")  # Debug print
#         fig = plt.figure(figsize=(8, 6), dpi=72)
#         classified = np.zeros_like(drought_prediction)
#         classified[drought_prediction < 0.25] = 0  # Low risk (Green)
#         classified[(drought_prediction >= 0.25) & (drought_prediction < 0.55)] = 1  # Medium risk (Orange)
#         classified[drought_prediction >= 0.55] = 2  # High risk (Red)

#         cmap = plt.get_cmap('RdYlGn_r', 3)  # Red -> Orange -> Green
#         plt.imshow(classified, cmap=cmap)
#         plt.colorbar(ticks=[0, 1, 2], 
#                     label='Drought Risk Level\n(0: Low, 1: Medium, 2: High)')
#         plt.title("Drought Risk Classification")
        
#         classification_buffer = BytesIO()
#         fig.savefig(classification_buffer, format='png', bbox_inches='tight', dpi=72)
#         plt.close(fig)
#         classification_buffer.seek(0)
#         classification_image = base64.b64encode(classification_buffer.getvalue()).decode('utf-8')
#         classification_buffer.close()
#         print("Classification map generated")  # Debug print

#         # Clean up
#         plt.close('all')
#         gc.collect()

#         return drought_ratio, heatmap_image, classification_image, stats

#     except Exception as e:
#         print(f"Error during analysis: {str(e)}")  # Debug print
#         plt.close('all')
#         gc.collect()
#         return None, None, None, None
#     finally:
#         plt.close('all')
#         gc.collect()

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
        fig, ax = plt.subplots(figsize=(8, 6), dpi=72)
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("NDVI")
        ax.set_title(f"{region_name.capitalize()} - NDVI Map")
        fig.tight_layout()
        heatmap_buffer = BytesIO()
        fig.savefig(heatmap_buffer, format='png', dpi=72)
        plt.close(fig)
        heatmap_buffer.seek(0)
        heatmap_image = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
        heatmap_buffer.close()
        print("Heatmap generated")  # Debug print

        # Generate classification visualization
        print("Generating classification map...")  # Debug print
        fig, ax = plt.subplots(figsize=(8, 6), dpi=72)
        classified = np.zeros_like(ndvi)
        classified[ndvi > 0.7] = 2  # Severe
        classified[(ndvi > 0.5) & (ndvi <= 0.7)] = 1  # Moderate

        cmap = plt.get_cmap('OrRd', 3)
        im = ax.imshow(classified, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.set_label('Drought Level\n(0: Normal, 1: Moderate, 2: Severe)')
        ax.set_title("Drought Classification")
        fig.tight_layout()
        classification_buffer = BytesIO()
        fig.savefig(classification_buffer, format='png', dpi=72)
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