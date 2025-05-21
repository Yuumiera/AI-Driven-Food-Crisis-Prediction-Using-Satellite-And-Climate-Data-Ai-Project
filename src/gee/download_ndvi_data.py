import ee
import os
import pandas as pd
from src.gee.ndvi_utils import mask_s2_clouds, add_ndvi

def init_gee():
    """Initialize Earth Engine"""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

def get_monthly_ndvi(geom, start_year, end_year, scale, buffer_radius=50000):
    """Get monthly NDVI data for a region"""
    results = []
    reduced_geom = geom.centroid().buffer(buffer_radius)
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-28"
            image = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filterBounds(reduced_geom)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                     .map(mask_s2_clouds)
                     .map(add_ndvi)
                     .median())
            try:
                ndvi = image.select('NDVI').reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=reduced_geom,
                    scale=scale,
                    maxPixels=1e13
                ).getInfo().get('NDVI')
            except Exception:
                ndvi = None
            results.append({"year": year, "month": month, "ndvi": ndvi})
    return pd.DataFrame(results)

def download_all_regions():
    """Download NDVI data for all regions"""
    # Create ndvi_data directory if it doesn't exist
    os.makedirs("ndvi_data", exist_ok=True)

    # Define regions and their coordinates
    regions = {
        "sanliurfa": ee.Geometry.Rectangle([38.5, 36.5, 39.5, 37.5]),
        "avustralia": ee.Geometry.Rectangle([133.0, -25.0, 134.0, -24.0]),
        "punjab": ee.Geometry.Rectangle([74.0, 31.0, 75.0, 32.0]),
        "munich": ee.Geometry.Rectangle([11.2, 47.9, 12.0, 48.4]),
        "california": ee.Geometry.Rectangle([-122.0, 37.0, -121.0, 38.0]),
        "iowa": ee.Geometry.Rectangle([-94.0, 42.0, -93.0, 43.0]),
        "kano": ee.Geometry.Rectangle([8.0, 11.0, 9.0, 12.0]),
        "zacatecas": ee.Geometry.Rectangle([-103.0, 22.0, -102.0, 23.0]),
        "gauteng": ee.Geometry.Rectangle([27.0, -26.0, 28.0, -25.0])
    }

    # Initialize Earth Engine
    print("Initializing Earth Engine...")
    init_gee()
    print("✅ Earth Engine initialized")

    # Download data for each region
    for name, geom in regions.items():
        save_path = f"ndvi_data/ndvi_{name}.csv"
        if os.path.exists(save_path):
            print(f"✅ {name}: Data already exists, skipping...")
            continue

        print(f"📥 {name}: Downloading NDVI data...")
        try:
            df = get_monthly_ndvi(geom, start_year=2020, end_year=2025, scale=10)
            df.dropna(inplace=True)
            df.to_csv(save_path, index=False)
            print(f"✅ {name}: Data saved to {save_path}")
        except Exception as e:
            print(f"❌ {name}: Error downloading data - {str(e)}")

if __name__ == "__main__":
    download_all_regions() 