from rio_tiler.io import Reader
from rio_tiler.colormap import cmap
import os

raster_path = "../data/phase3/2026/ndvi_30m.tif"
with Reader(raster_path) as src:
    img = src.tile(21868, 13745, 15) # Example tile for Delhi
    print("Shape before:", img.data.shape)
    img.rescale(in_range=((-1, 1),))
    cm = cmap.get("RdYlGn")
    buffer = img.render(colormap=cm, img_format="PNG")
    print("Rendered successfully. Buffer size:", len(buffer))
