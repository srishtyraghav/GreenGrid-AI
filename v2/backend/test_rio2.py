from rio_tiler.io import Reader
from rio_tiler.colormap import cmap

raster_path = "../data/phase3/2026/ndvi_30m.tif"
with Reader(raster_path) as src:
    img = src.preview() # gets whole image preview
    print("Min before:", img.data.min(), "Max:", img.data.max())
    img.rescale(in_range=((-1, 1),))
    print("Min after:", img.data.min(), "Max:", img.data.max())
