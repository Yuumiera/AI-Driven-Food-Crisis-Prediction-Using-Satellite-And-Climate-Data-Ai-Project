import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def draw_heatmap(heatmap):
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(label="Drought Probability (0–1)")
    plt.title("CNN Drought Prediction Heatmap")
    plt.tight_layout()
    plt.show()

def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - CNN")
    plt.tight_layout()
    plt.show()

def draw_roc_auc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - CNN Drought Classifier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def draw_heatmap_and_confusion(heatmap, y_true, y_pred, y_scores):
    draw_heatmap(heatmap)
    draw_confusion_matrix(y_true, y_pred)
    draw_roc_auc(y_true, y_scores)
