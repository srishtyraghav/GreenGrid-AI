from rio_tiler.io import Reader
from rio_tiler.colormap import cmap

raster_path = "../data/phase3/2026/ndvi_30m.tif"
with Reader(raster_path) as src:
    img = src.preview()
    img.rescale(in_range=((-1, 1),))
    cm = cmap.get("rdylgn")
    buffer = img.render(colormap=cm, img_format="PNG")
    with open("test_ndvi.png", "wb") as f:
        f.write(buffer)
print("Saved test_ndvi.png")
