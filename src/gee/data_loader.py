
import ee
import pandas as pd
from src.gee.ndvi_utils import mask_s2_clouds, add_ndvi

def get_monthly_ndvi(geom, start_year, end_year, scale, buffer_radius=50000):
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
