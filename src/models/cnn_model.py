import sys
sys.path.append('/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data')


import torch
import os
import numpy as np
import rasterio
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import gc

# --- MODEL TANIMI ---
def build_cnn_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


# --- EĞİTİM FONKSİYONU ---
class LazyNDVIDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.patches = np.load(os.path.join(data_dir, "patches.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img = self.patches[idx].astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Tek kanal
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(label, dtype=torch.float32)
def train_model(train_path,
                val_path,
                model_save_path,
                epochs=20,
                batch_size=32,
                learning_rate=1e-4,
                patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LazyNDVIDataset(train_path)
    val_dataset = LazyNDVIDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = build_cnn_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f}")

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

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"📦 Model saved to {model_save_path}")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs} epochs.")
            if no_improve_epochs >= patience:
                print("⏹️ Early stopping triggered.")
                break

def run_cnn_pipeline_multi(tif_folder):
    import gc
    from tqdm import tqdm
    import rasterio

    model_path = "/Users/ahmetbekir/AI-Driven-Food-Crisis-Prediction-Using-Satellite-And-Climate-Data/models/cnn_resnet18_drought.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_cnn_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.half()  # Float16 hızlandırma

    tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif")]

    for tif_name in tif_files:
        tif_path = os.path.join(tif_folder, tif_name)

        with rasterio.open(tif_path) as src:
            ndvi = src.read(1)
            profile = src.profile

        ndvi = np.clip(ndvi, -1, 1)
        ndvi = (ndvi + 1) / 2

        patch_size = 128
        stride = 128
        h, w = ndvi.shape

        patches, coords, labels = [], [], []

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = ndvi[i:i+patch_size, j:j+patch_size]
                if np.isnan(patch).any():
                    continue
                mean_val = np.mean(patch)
                label = 1 if mean_val < 0.45 else 0
                patch_tensor = patch.astype(np.float16)[np.newaxis, :, :]
                patches.append(patch_tensor)
                labels.append(label)
                coords.append((i, j))

        print(f"📂 Processing {tif_name} — {len(patches)} patches")

        if len(patches) == 0:
            print("⚠️ No valid patches found.")
            continue

        # Mini-batch inference
        probs = []
        batch_size = 512
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            batch_tensor = torch.tensor(np.stack(batch)).to(device).half()
            with torch.no_grad():
                output = model(batch_tensor)
                batch_probs = torch.sigmoid(output).cpu().numpy().flatten()
                probs.extend(batch_probs)
            del batch_tensor
            torch.cuda.empty_cache()

        # Heatmap doldur
        heatmap = np.zeros((h // stride, w // stride), dtype=np.float32)
        for idx, (i, j) in enumerate(coords):
            heatmap[i // stride, j // stride] = probs[idx]

        # GeoTIFF çıktısı
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        out_path = os.path.join(tif_folder, tif_name.replace(".tif", "_predicted.tif"))
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(heatmap, 1)

        print(f"✅ Saved: {out_path}")
        gc.collect()
    return True


# import os
# import numpy as np
# import torch
# import rasterio
# from torchvision import models
# import torch.nn as nn
# from tqdm import tqdm
# import gc

# def build_cnn_model():
#     model = models.resnet18(weights='IMAGENET1K_V1')
#     model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model.fc = nn.Linear(model.fc.in_features, 1)
#     return model

# def generate_predicted_tifs(
#     tif_folder,
#     model_path,
#     output_suffix="_predicted.tif",
#     patch_size=128,
#     stride=128,
#     batch_size=512,
#     device=None
# ):
#     """
#     Verilen klasördeki .tif dosyalarını CNN modeliyle işleyip, tahmin (_predicted) tif'leri üretir.

#     Args:
#         tif_folder (str): NDVI .tif dosyalarının bulunduğu klasör
#         model_path (str): Eğitilmiş CNN modelin yol (pth uzantılı)
#         output_suffix (str): Çıktı dosyasının uzantı eki (varsayılan '_predicted.tif')
#         patch_size (int): CNN giriş yaması boyutu
#         stride (int): Yama geçiş adımı
#         batch_size (int): İnferans batch boyutu
#         device (torch.device): GPU varsa belirt, yoksa otomatik seçilir
#     """
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Modeli yükle
#     model = build_cnn_model().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     model = model.half()  # İnferans için hız avantajı

#     tif_files = [f for f in os.listdir(tif_folder) if f.endswith(".tif") and output_suffix not in f]

#     for tif_name in tqdm(tif_files):
#         tif_path = os.path.join(tif_folder, tif_name)

#         with rasterio.open(tif_path) as src:
#             ndvi = src.read(1)
#             profile = src.profile

#         # NDVI normalize
#         ndvi = np.clip(ndvi, -1, 1)
#         ndvi = (ndvi + 1) / 2

#         h, w = ndvi.shape
#         patches, coords = [], []

#         for i in range(0, h - patch_size + 1, stride):
#             for j in range(0, w - patch_size + 1, stride):
#                 patch = ndvi[i:i+patch_size, j:j+patch_size]
#                 if np.isnan(patch).any():
#                     continue
#                 patch_tensor = patch.astype(np.float16)[np.newaxis, :, :]  # [1, H, W]
#                 patches.append(patch_tensor)
#                 coords.append((i, j))

#         if len(patches) == 0:
#             print(f"⚠️ {tif_name} içinde geçerli yama bulunamadı.")
#             continue

#         # Tahmin et
#         probs = []
#         for i in range(0, len(patches), batch_size):
#             batch = patches[i:i+batch_size]
#             batch_tensor = torch.tensor(np.stack(batch)).to(device).half()
#             with torch.no_grad():
#                 output = model(batch_tensor)
#                 batch_probs = torch.sigmoid(output).cpu().numpy().flatten()
#                 probs.extend(batch_probs)
#             del batch_tensor
#             torch.cuda.empty_cache()

#         # Heatmap oluştur
#         heatmap = np.zeros((h // stride, w // stride), dtype=np.float32)
#         for idx, (i, j) in enumerate(coords):
#             heatmap[i // stride, j // stride] = probs[idx]

#         # Tif yaz
#         profile.update(dtype=rasterio.float32, count=1, compress='lzw')
#         out_path = os.path.join(tif_folder, tif_name.replace(".tif", output_suffix))
#         with rasterio.open(out_path, 'w', **profile) as dst:
#             dst.write(heatmap, 1)

#         print(f"✅ {tif_name} → {os.path.basename(out_path)} oluşturuldu.")

#         gc.collect()

#     print("🎯 Tüm tif dosyaları işlendi.")
