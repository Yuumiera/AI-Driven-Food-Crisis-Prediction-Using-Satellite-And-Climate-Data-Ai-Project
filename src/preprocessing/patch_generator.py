import os
import numpy as np
import rasterio

def generate_patches(tif_folder, patch_size=128, stride=128, threshold=0.4):
    patches = []
    labels = []
    tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif')]

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            ndvi = src.read(1)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = (ndvi + 1) / 2  # Normalize [0, 1]
        
        for i in range(0, ndvi.shape[0] - patch_size + 1, stride):
            for j in range(0, ndvi.shape[1] - patch_size + 1, stride):
                patch = ndvi[i:i+patch_size, j:j+patch_size]
                mean_val = np.mean(patch)
                label = 1 if mean_val < threshold else 0
                patches.append(patch)
                labels.append(label)
                
    return np.array(patches), np.array(labels)
