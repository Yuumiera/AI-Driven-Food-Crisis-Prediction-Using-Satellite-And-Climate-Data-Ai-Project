import ee

# ✅ GEE bağlantısı (auth varsa skip)
try:
    ee.Initialize()
    print("✅ GEE zaten başlatıldı.")
except Exception:
    print("🔐 Authenticate gerekiyor.")
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize()

# ✅ NDVI Export Fonksiyonu
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
    print(f"✅ Export task started for {filename}")

# ✅ Somali – Dar ve kurak alan
region = ee.Geometry.Rectangle([42.0, 1.5, 43.0, 2.5])

# ⏳ Yaz ayları (kurak mevsim)
months = [
    ('2022-07-01', '2022-07-31'),
    ('2023-07-01', '2023-07-31'),
]

# ✅ Tüm batch export'ları başlat
for start, end in months:
    filename = f"ndvi_patch_somali_{start[:7].replace('-', '')}"
    export_ndvi_patch(region, start, end, filename)
