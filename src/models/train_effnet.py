import sys
sys.path.append('/content/drive/MyDrive/food_crisis_prediction2')

import torch
from torch.utils.data import DataLoader
from src.models.cnn_model_effnet import build_efficientnet_model, train_efficientnet_model
from src.preprocessing.lazy_dataset_effnet import LazyNDVIDataset
import os

def main():
    print("EfficientNet Training Started...")

    #  Veri Yolu
    data_dir = "/content/drive/MyDrive/food_crisis_prediction2/data/processed"
    dataset = LazyNDVIDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    #  Cihaz Seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Eğitim
    model = train_efficientnet_model(train_loader, device, epochs=20, patience=5)

    #  Model Kaydetme
    save_path = "/content/drive/MyDrive/food_crisis_prediction2/models/cnn_efficientnet_drought.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()
