import os
import numpy as np
import rasterio

def save_patches_to_npy(tif_folder, save_path, patch_size=128, stride=512, threshold=0.55):
    patches = []
    labels = []
    tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif')]

    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            ndvi = src.read(1)

        ndvi = np.nan_to_num(ndvi, nan=0.0)
        ndvi = np.clip(ndvi, -1, 1)
        ndvi = (ndvi + 1) / 2  # Normalize to [0,1]

        for i in range(0, ndvi.shape[0] - patch_size + 1, stride):
            for j in range(0, ndvi.shape[1] - patch_size + 1, stride):
                patch = ndvi[i:i+patch_size, j:j+patch_size]
                
                # Tamamen sıfır olan patch'leri atla
                if np.all(patch == 0):
                    continue

                mean_val = np.mean(patch)
                label = 1 if mean_val < threshold else 0
                patches.append(patch)
                labels.append(label)

    patches = np.array(patches)
    labels = np.array(labels)

    np.save(os.path.join(save_path, "patches.npy"), patches)
    np.save(os.path.join(save_path, "labels.npy"), labels)

    kurak = np.sum(labels)
    saglikli = len(labels) - kurak
    print(f"✅ Patch ve label dosyaları kaydedildi. Toplam Patch: {len(patches)}, Kurak: {kurak}, Sağlıklı: {saglikli}")
