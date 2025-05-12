
import ee

def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_mask = 1 << 10
    cirrus_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_mask).eq(0).And(qa.bitwiseAnd(cirrus_mask).eq(0))
    return image.updateMask(mask)

def add_ndvi(image):
    nir = image.select('B8').divide(10000)
    red = image.select('B4').divide(10000)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi)
