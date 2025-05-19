import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

import os
import torch
import numpy as np
import rasterio
from models.cnn_model import build_cnn_model

def run_cnn_pipeline_multi(tif_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_cnn_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]

    for tif_name in tif_files:
        tif_path = os.path.join(tif_folder, tif_name)
        with rasterio.open(tif_path) as src:
            ndvi = src.read(1)

        ndvi = np.clip(ndvi, -1, 1)
        ndvi = (ndvi + 1) / 2

        patch_size = 128
        stride = 128
        h, w = ndvi.shape
        heatmap = np.zeros((h // stride, w // stride))
        all_targets, all_preds, all_scores = [], [], []

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = ndvi[i:i+patch_size, j:j+patch_size]
                if np.isnan(patch).any():
                    continue

                mean_val = np.mean(patch)
                label = 1 if mean_val < 0.45 else 0
                tensor_patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor_patch)
                    prob = torch.sigmoid(output).item()

                pred = int(prob > 0.5)

                all_scores.append(prob)
                all_preds.append(pred)
                all_targets.append(label)
                heatmap[i // stride, j // stride] = prob

        results.append({
            "file": tif_name,
            "heatmap": heatmap,
            "y_true": all_targets,
            "y_pred": all_preds,
            "y_scores": all_scores
        })

    return results
