import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import rasterio

from src.preprocessing.lazy_dataset import LazyNDVIDataset

def build_cnn_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def run_cnn_pipeline():
    data_dir = "/content/drive/MyDrive/food_crisis_prediction2/data/processed"
    model_path = "/content/drive/MyDrive/food_crisis_prediction2/models/cnn_resnet18_drought.pth"
    tif_path = "/content/drive/MyDrive/GEE_NDVI/ndvi_patch_202307.tif"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Heatmap üretimi için NDVI yükle
    with rasterio.open(tif_path) as src:
        ndvi = src.read(1)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi = (ndvi + 1) / 2  # normalize to 0–1

    patch_size = 128
    stride = 128
    h, w = ndvi.shape
    heatmap = np.zeros((h // stride, w // stride))
    all_targets = []
    all_preds = []
    all_scores = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = ndvi[i:i + patch_size, j:j + patch_size]
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

    return heatmap, all_targets, all_preds, all_scores

if __name__ == "__main__":
    print("Running CNN pipeline...")
    heatmap, y_true, y_pred, y_scores = run_cnn_pipeline()
    print("Heatmap shape:", heatmap.shape)
