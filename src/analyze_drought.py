import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
import gc

def analyze_drought_for_region(region_name):
    """
    Analyzes the .tif prediction map for the given region name,
    generates heatmap, prints statistics and returns drought ratio.
    
    Args:
        region_name (str): Region name (e.g., 'sanliurfa')
    
    Returns:
        tuple: (drought_ratio, heatmap_image, drought_classification_image, stats_dict)
    """
    # Configure matplotlib for this process
    plt.ioff()
    
    try:
        print(f"Analyzing region: {region_name}")  # Debug print
        
        # 🌍 Static mapping list
        region_to_tif = {
            "sanliurfa": "some_tifs/ndvi_patch_201908_predicted.tif",
            "punjab": "some_tifs/ndvi_patch_afghanistan_202307_predicted.tif",
            "avustralia": "some_tifs/ndvi_patch_australia_202307_predicted.tif",
            "munich": "some_tifs/ndvi_patch_europe_202307-0000000000-0000000000_predicted.tif",
            "california": "some_tifs/ndvi_patch_california_202307_predicted.tif",
            "iowa": "some_tifs/iowa_05-2025.tif",
            "kano": "some_tifs/kano_05-2025.tif",
            "zacatecas": "some_tifs/zacatecas_05-2025.tif",
            "gauteng": "some_tifs/gauteng_05-2025.tif"
        }

        tif_path = region_to_tif[region_name]
        print(f"Looking for file: {tif_path}")  # Debug print
        print(f"File exists: {os.path.exists(tif_path)}")  # Debug print
        
        if region_name not in region_to_tif:
            print(f"❌ Warning: No tif file defined for '{region_name}'.")
            return None, None, None, None

        if not os.path.exists(tif_path):
            print(f"❌ File not found: {tif_path}")
            print(f"Current working directory: {os.getcwd()}")  # Debug print
            return None, None, None, None

        # 🔄 Read Tif with downsampling for faster processing
        print("Opening TIF file...")  # Debug print
        with rasterio.open(tif_path) as src:
            # Calculate downsampling factor based on image size
            scale_factor = max(1, min(4, src.width // 1000))  # Adjust scale based on image width
            out_shape = (
                src.height // scale_factor,
                src.width // scale_factor
            )
            
            # Read downsampled data
            prediction = src.read(
                1,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.bilinear
            )
        print("TIF file read successfully")  # Debug print

        # 🔬 Drought Analysis
        threshold = 0.5
        drought_mask = prediction > threshold
        drought_ratio = np.sum(drought_mask) / drought_mask.size
        print(f"Drought ratio calculated: {drought_ratio}")  # Debug print

        # 📊 Statistics
        stats = {
            'mean_score': float(np.mean(prediction)),
            'max_score': float(np.max(prediction)),
            'min_score': float(np.min(prediction)),
            'std_score': float(np.std(prediction)),
            'drought_ratio': float(drought_ratio * 100)
        }
        print("Statistics calculated")  # Debug print

        # Generate heatmap image with optimized settings
        print("Generating heatmap...")  # Debug print
        fig = plt.figure(figsize=(8, 6), dpi=72)  # Reduced DPI for faster rendering
        plt.imshow(prediction, cmap='YlOrRd')
        plt.colorbar(label="Drought Score")
        plt.title(f"{region_name.capitalize()} - Drought Heatmap")
        
        # Save heatmap to base64 with optimized settings
        heatmap_buffer = BytesIO()
        fig.savefig(heatmap_buffer, format='png', bbox_inches='tight', dpi=72)
        plt.close(fig)  # Explicitly close the figure
        heatmap_buffer.seek(0)
        heatmap_image = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
        heatmap_buffer.close()  # Explicitly close the buffer
        del heatmap_buffer  # Explicitly delete the buffer
        print("Heatmap generated")  # Debug print

        # Generate drought classification image with optimized settings
        print("Generating classification map...")  # Debug print
        fig = plt.figure(figsize=(8, 6), dpi=72)  # Reduced DPI for faster rendering
        classified = np.zeros_like(prediction)
        classified[prediction > 0.7] = 2  # Severe
        classified[(prediction > 0.5) & (prediction <= 0.7)] = 1  # Moderate

        cmap = plt.get_cmap('OrRd', 3)
        plt.imshow(classified, cmap=cmap)
        plt.colorbar(ticks=[0, 1, 2], 
                    label='Drought Level\n(0: Normal, 1: Moderate, 2: Severe)')
        plt.title("Drought Classification")
        
        # Save classification to base64 with optimized settings
        classification_buffer = BytesIO()
        fig.savefig(classification_buffer, format='png', bbox_inches='tight', dpi=72)
        plt.close(fig)  # Explicitly close the figure
        classification_buffer.seek(0)
        classification_image = base64.b64encode(classification_buffer.getvalue()).decode('utf-8')
        classification_buffer.close()  # Explicitly close the buffer
        del classification_buffer  # Explicitly delete the buffer
        print("Classification map generated")  # Debug print

        # Clean up
        plt.close('all')
        gc.collect()  # Force garbage collection

        return drought_ratio, heatmap_image, classification_image, stats

    except Exception as e:
        print(f"Error during analysis: {str(e)}")  # Debug print
        plt.close('all')  # Clean up on error
        gc.collect()  # Force garbage collection
        return None, None, None, None
    finally:
        # Final cleanup
        plt.close('all')
        gc.collect()  # Force garbage collection 