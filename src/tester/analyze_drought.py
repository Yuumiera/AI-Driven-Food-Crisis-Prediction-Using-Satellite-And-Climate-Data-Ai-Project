import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import os
import sys
import gc  # Garbage collector
from rasterio.windows import Window

def process_chunk(chunk_data, threshold):
    """Process a chunk of data and return statistics."""
    kurak_mask = chunk_data > threshold
    return {
        'mean': np.mean(chunk_data),
        'max': np.max(chunk_data),
        'min': np.min(chunk_data),
        'std': np.std(chunk_data),
        'drought_pixels': np.sum(kurak_mask),
        'total_pixels': chunk_data.size
    }

def analyze_drought_for_region(region_name, chunk_size=1000):
    """
    Analyze drought for a region by processing the TIF file in chunks.
    
    Args:
        region_name (str): Region name (e.g., 'sanliurfa', 'avustralia')
        chunk_size (int): Size of chunks to process at once
    
    Returns:
        float: Drought ratio (between 0.0 and 1.0)
    """
    try:
        region_to_tif = {
            "sanliurfa": "ndvi_patch_201908_predicted.tif",
            "avustralia": "ndvi_patch_australia_202307_predicted.tif",
            "california": "ndvi_patch_california_202307_predicted.tif",
        }

        tif_dir = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/some_tifs"
        tif_path = os.path.join(tif_dir, region_to_tif[region_name])

        if not os.path.exists(tif_path):
            print(f"❌ File not found: {tif_path}")
            return None

        print(f"📂 Reading file: {tif_path}")
        with rasterio.open(tif_path) as src:
            # Get image dimensions
            height = src.height
            width = src.width
            print(f"Image size: {width}x{height}")

            # Initialize statistics
            total_stats = {
                'mean': 0,
                'max': float('-inf'),
                'min': float('inf'),
                'std': 0,
                'drought_pixels': 0,
                'total_pixels': 0
            }

            # Process image in chunks
            for i in range(0, height, chunk_size):
                for j in range(0, width, chunk_size):
                    # Calculate window size
                    win_height = min(chunk_size, height - i)
                    win_width = min(chunk_size, width - j)
                    
                    # Read chunk
                    window = Window(j, i, win_width, win_height)
                    chunk = src.read(1, window=window)
                    
                    # Process chunk
                    chunk_stats = process_chunk(chunk, 0.5)
                    
                    # Update total statistics
                    total_stats['max'] = max(total_stats['max'], chunk_stats['max'])
                    total_stats['min'] = min(total_stats['min'], chunk_stats['min'])
                    total_stats['drought_pixels'] += chunk_stats['drought_pixels']
                    total_stats['total_pixels'] += chunk_stats['total_pixels']
                    
                    # Update mean and std (weighted by chunk size)
                    chunk_weight = chunk.size / (width * height)
                    total_stats['mean'] += chunk_stats['mean'] * chunk_weight
                    total_stats['std'] += chunk_stats['std'] * chunk_weight
                    
                    # Clear memory
                    del chunk
                    gc.collect()

            print("✅ File processed successfully")

            # Calculate final statistics
            drought_ratio = total_stats['drought_pixels'] / total_stats['total_pixels']
            
            stats = {
                'mean_score': float(total_stats['mean']),
                'max_score': float(total_stats['max']),
                'min_score': float(total_stats['min']),
                'std_score': float(total_stats['std']),
                'drought_ratio': float(drought_ratio * 100)
            }

            # Generate visualizations with reduced resolution
            print("🖼️ Generating visualizations...")
            
            # Read a downsampled version for visualization
            scale_factor = 4  # Reduce resolution by factor of 4
            vis_height = height // scale_factor
            vis_width = width // scale_factor
            
            with rasterio.open(tif_path) as src:
                # Read downsampled data
                prediction = src.read(
                    1,
                    out_shape=(vis_height, vis_width),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                # Ensure prediction is 2D
                if prediction.ndim == 3:
                    prediction = prediction[0]
                elif prediction.ndim == 1:
                    prediction = prediction.reshape((vis_height, vis_width))

                # Generate heatmap
                print("Creating heatmap...")
                fig = plt.figure(figsize=(8, 6), dpi=100)
                plt.imshow(prediction, cmap='YlOrRd')
                plt.colorbar(label="Drought Score")
                plt.title(f"{region_name.capitalize()} - Drought Heatmap")
                plt.savefig(f'heatmap_1{region_name}.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                gc.collect()

                # Generate drought mask
                print("Creating drought mask...")
                kurak_mask = prediction > 0.5
                fig = plt.figure(figsize=(8, 6), dpi=100)
                plt.imshow(kurak_mask, cmap='Reds')
                plt.title("Drought Areas (> 0.5)")
                plt.savefig('kurak_mask.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                gc.collect()

                # Generate classification map
                print("Creating classification map...")
                classified = np.zeros_like(prediction)
                classified[prediction > 0.7] = 2  # Severe
                classified[(prediction > 0.5) & (prediction <= 0.7)] = 1  # Moderate

                fig = plt.figure(figsize=(8, 6), dpi=100)
                cmap = plt.get_cmap('OrRd', 3)
                plt.imshow(classified, cmap=cmap)
                plt.colorbar(ticks=[0, 1, 2], label='Drought Level')
                plt.title("Drought Classification")
                plt.savefig('classified.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
                gc.collect()

            # Print statistics
            print(f"\n📊 Drought Analysis ({region_name}):")
            print(f"- Mean Score: {stats['mean_score']:.3f}")
            print(f"- Maximum Score: {stats['max_score']:.3f}")
            print(f"- Minimum Score: {stats['min_score']:.3f}")
            print(f"- Standard Deviation: {stats['std_score']:.3f}")
            print(f"- Drought Ratio (> 0.5): {stats['drought_ratio']:.2f}%")

            return drought_ratio

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a region name as an argument.")
        print("Example: python analyze_drought.py sanliurfa")
        sys.exit(1)

    region = sys.argv[1]
    print(f"\n🔍 Analyzing region: {region}...")
    ratio = analyze_drought_for_region(region)
    
    if ratio is not None:
        print(f"\n✅ Calculated drought ratio for {region}: {ratio:.3f}")
    else:
        print(f"\n❌ Failed to analyze region {region}.")