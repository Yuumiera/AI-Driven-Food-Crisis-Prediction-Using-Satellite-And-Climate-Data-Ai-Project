import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from datetime import datetime

RESULT_DIR = "/content/drive/MyDrive/food_crisis_prediction2/results"
os.makedirs(RESULT_DIR, exist_ok=True)

def save_and_show(fig, filename):
    save_path = os.path.join(RESULT_DIR, filename)
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)

def draw_heatmap(heatmap, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="Drought Probability (0–1)")
    plt.title(f"Heatmap {title_suffix}")
    plt.tight_layout()
    save_and_show(fig, f"heatmap{title_suffix}.png")

def draw_confusion_matrix(y_true, y_pred, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.tight_layout()
    save_and_show(fig, f"confusion{title_suffix}.png")

def draw_roc_auc(y_true, y_scores, title_suffix=""):
    if len(set(y_true)) < 2:
        print("Skipping ROC curve: not enough class diversity.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_and_show(fig, f"roc_auc{title_suffix}.png")

def draw_heatmap_and_confusion(heatmap, y_true, y_pred, y_scores, title_suffix=""):
    draw_heatmap(heatmap, title_suffix)
    draw_confusion_matrix(y_true, y_pred, title_suffix)
    draw_roc_auc(y_true, y_scores, title_suffix)
