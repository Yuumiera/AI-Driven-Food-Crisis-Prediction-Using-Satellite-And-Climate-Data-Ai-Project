import ee

# GEE bağlantısı
try:
    ee.Initialize()
    print("✅ GEE zaten başlatıldı.")
except Exception:
    print("🔐 Authenticate gerekiyor.")
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize()

# NDVI Export Fonksiyonu
def export_ndvi_patch(region, start_date, end_date, filename, scale=10):
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )

    size = collection.size().getInfo()
    if size == 0:
        print(f"⚠️ No images found for {filename}. Skipping.")
        return

    image = (
        collection
        .map(lambda img: img.unmask(0))
        .map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
        .median()
    )

    task = ee.batch.Export.image.toDrive(
        image=image.select('NDVI'),
        description=filename,
        folder='GEE_NDVI',
        fileNamePrefix=filename,
        region=region,
        scale=scale,
        maxPixels=1e13
    )
    task.start()
    print(f"🚀 Export task started for {filename}")

# 📍 Avrupa bölgesi (örnek: Macaristan civarı / Orta Avrupa)
region = ee.Geometry.Rectangle([16.0, 46.0, 21.0, 49.0])  # [minLng, minLat, maxLng, maxLat]

# 🗓️ Yaz ayları (kuraklık gözlemi için)
months = [
    ('2023-07-01', '2023-07-31'),
]

# 🔄 Batch Export
for start, end in months:
    filename = f"ndvi_patch_europe_{start[:7].replace('-', '')}"
    export_ndvi_patch(region, start, end, filename)
