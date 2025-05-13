import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

from src.preprocessing.lazy_dataset import LazyNDVIDataset

def build_cnn_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def train_cnn_model(train_loader, device, epochs=20, lr=0.0001, patience=3):
    model = build_cnn_model().to(device)

    labels_np = train_loader.dataset.labels
    kurak = np.sum(labels_np)
    saglikli = len(labels_np) - kurak
    pos_weight_value = min(saglikli / kurak, 5)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    print(f"Pos Weight (Clamped): {pos_weight_value:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_size = len(train_loader.dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = Subset(train_loader.dataset, train_indices)
    val_subset = Subset(train_loader.dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    best_loss = float('inf')
    counter = 0
    best_model_wts = None

    all_preds = []
    all_targets = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = torch.clamp(outputs, -10, 10)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(" NaN Loss Detected. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(outputs).cpu().detach().numpy() > 0.5
            correct += (preds.flatten() == labels.cpu().numpy().flatten()).sum()
            total += labels.size(0)

        if total == 0:
            print(" No valid training batches. Exiting epoch.")
            break

        train_loss = running_loss / total
        train_acc = correct / total

        # ----- Validation -----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds.clear()
        all_targets.clear()

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                outputs = torch.clamp(outputs, -10, 10)
                batch_loss = criterion(outputs, labels)

                if torch.isnan(batch_loss):
                    continue

                val_loss += batch_loss.item() * imgs.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds.flatten().tolist())
                all_targets.extend(labels.cpu().numpy().flatten().tolist())
                val_correct += (preds.flatten() == labels.cpu().numpy().flatten()).sum()
                val_total += labels.size(0)

        if val_total == 0:
            print(" No valid validation batches. Skipping evaluation.")
            break

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(" Early stopping triggered.")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    else:
        print(" No valid model weights found due to NaN losses.")

    # ----- Final Evaluation -----
    print("\n Evaluation Report on Validation Set:")
    binary_preds = [int(p > 0.5) for p in all_preds]
    print(confusion_matrix(all_targets, binary_preds))
    print(classification_report(all_targets, binary_preds, digits=4))

    return model

if __name__ == "__main__":
    print(" CNN Lazy Training Started...")

    data_dir = "/content/drive/MyDrive/food_crisis_prediction2/data/processed"
    dataset = LazyNDVIDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_cnn_model(train_loader, device, epochs=20, patience=5)

    save_path = "/content/drive/MyDrive/food_crisis_prediction2/models/cnn_resnet18_drought.pth"
    torch.save(model.state_dict(), save_path)
    print(f" Model Kaydedildi: {save_path}")
