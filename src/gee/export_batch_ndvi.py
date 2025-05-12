import ee
ee.Authenticate(auth_mode='notebook')
ee.Initialize()

def export_ndvi_patch(region, start_date, end_date, filename, scale=10):
    image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterDate(start_date, end_date)
             .filterBounds(region)
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
             .map(lambda img: img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
             .median())

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

# 📍 Afrika Kurak Bölge (Örneğin Somali)
region = ee.Geometry.Rectangle([42.0, 1.0, 46.0, 5.0])  # Somali civarı

months = [
    ('2022-07-01', '2022-07-31'),
    ('2023-07-01', '2023-07-31'),
]

for start, end in months:
    filename = f"ndvi_patch_africa_{start[:7].replace('-', '')}"
    export_ndvi_patch(region, start, end, filename)

