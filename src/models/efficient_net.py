import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import timm

# Dataset tanımı
class EfficientNDVIDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx].astype(np.float32)
        patch = np.expand_dims(patch, axis=0)  # 1 kanal
        label = self.labels[idx]
        return torch.tensor(patch), torch.tensor(label, dtype=torch.float32)

# EfficientNet-B0 Modeli
def build_efficientnet():
    model = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=1)
    return model

# Eğitim Fonksiyonu
def train_efficientnet(data_dir, model_save_path, epochs=20, batch_size=8, lr=0.0003):
    patches = np.load(os.path.join(data_dir, "patches.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    dataset = EfficientNDVIDataset(patches, labels)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(0.2 * len(dataset))
    train_idx, val_idx = indices[split:], indices[:split]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet().to(device)

    kurak = np.sum(labels)
    saglikli = len(labels) - kurak
    pos_weight = torch.tensor([min(saglikli / kurak, 5)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_model = None
    patience = 3
    stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)

                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict().copy()
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("Early stopping.")
                break

    if best_model:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ EfficientNet model saved to: {model_save_path}")
    else:
        print("⚠️ No best model was found.")

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, digits=4))

if __name__ == "__main__":
    train_efficientnet(
        data_dir="/content/drive/MyDrive/food_crisis_prediction2/data/processed_effnet",
        model_save_path="/content/drive/MyDrive/food_crisis_prediction2/models/cnn_efficientnet_drought.pth",
        epochs=20
    )
