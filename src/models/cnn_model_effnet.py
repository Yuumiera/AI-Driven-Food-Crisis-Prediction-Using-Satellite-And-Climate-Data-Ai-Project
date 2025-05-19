import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def build_efficientnet_model():
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model

def train_efficientnet_model(train_loader, device, epochs=20, lr=0.0001, patience=3):
    model = build_efficientnet_model().to(device)

    labels_np = train_loader.dataset.labels
    kurak = np.sum(labels_np)
    saglikli = len(labels_np) - kurak
    pos_weight_value = min(saglikli / kurak, 5)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    print(f"Pos Weight (Clamped): {pos_weight_value:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset_size = len(train_loader.dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = Subset(train_loader.dataset, train_indices)
    val_subset = Subset(train_loader.dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    best_loss = float('inf')
    best_model_wts = None
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            outputs = torch.clamp(outputs, -10, 10)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            train_correct += (preds.flatten() == labels.cpu().numpy().flatten()).sum()
            total += labels.size(0)

        train_loss /= total
        train_acc = train_correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                outputs = torch.clamp(outputs, -10, 10)
                vloss = criterion(outputs, labels)

                if torch.isnan(vloss):
                    continue

                val_loss += vloss.item() * imgs.size(0)
                preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                val_correct += (preds.flatten() == labels.cpu().numpy().flatten()).sum()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    else:
        print("No valid weights found due to NaNs.")

    # Confusion matrix (isteğe bağlı)
    print("\nFinal validation results:")
    binary_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outputs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds = (outputs > 0.5).astype(int).flatten()
            binary_preds.extend(preds)
            all_targets.extend(labels.numpy().flatten())

    print(confusion_matrix(all_targets, binary_preds))
    print(classification_report(all_targets, binary_preds, digits=4))

    return model
