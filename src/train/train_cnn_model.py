import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.cnn_model import build_cnn_model  # modelini buradan alıyoruz
from dataset import LazyNDVIDataset           # Dataset class'ı senin tanımına göre

def train_model(
    data_path="data",
    model_save_path="models/cnn_resnet18_drought.pth",
    epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    patience=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Veri yükle
    dataset = LazyNDVIDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, loss ve optimizer
    model = build_cnn_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"📦 Model improved and saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print("⏹️ Early stopping triggered.")
                break
