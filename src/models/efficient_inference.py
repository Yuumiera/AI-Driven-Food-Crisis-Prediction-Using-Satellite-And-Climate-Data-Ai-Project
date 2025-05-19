import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
from skimage.util import view_as_windows
import timm

# Model tanımı

def build_efficientnet():
    model = timm.create_model('efficientnet_b0', pretrained=False, in_chans=1, num_classes=1)
    return model

# Tahmin fonksiyonu

def run_efficientnet_inference(tif_path, model_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with rasterio.open(tif_path) as src:
        ndvi = src.read(1)

    ndvi = np.nan_to_num(ndvi, nan=0.0)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi = (ndvi + 1) / 2

    patch_size = 128
    stride = 128
    h, w = ndvi.shape
    heatmap = np.zeros((h // stride, w // stride))
    y_true, y_pred, y_scores = [], [], []

    patches = view_as_windows(ndvi, (patch_size, patch_size), step=stride)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            if np.all(patch == 0):
                continue

            label = 1 if np.mean(patch) < threshold else 0
            tensor_patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor_patch)
                prob = torch.sigmoid(out).item()

            pred = int(prob > 0.5)
            heatmap[i, j] = prob
            y_scores.append(prob)
            y_pred.append(pred)
            y_true.append(label)

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    plt.title("EfficientNet Drought Prediction Heatmap")
    plt.colorbar(label="Drought Probability")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tif_path = "/content/drive/MyDrive/GEE_NDVI/ndvi_patch_202307.tif"
    model_path = "/content/drive/MyDrive/food_crisis_prediction2/models/cnn_efficientnet_drought.pth"
    run_efficientnet_inference(tif_path, model_path)
