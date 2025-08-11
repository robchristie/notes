import numpy as np
import rasterio

def load_raster_as_float(path, robust=(2,98), per_channel=True):
    with rasterio.open(path) as src:
        arr = src.read()
    arr = arr.astype(np.float32)
    def _stretch(x, p):
        if p is None:
            lo, hi = float(x.min()), float(x.max())
        else:
            lo, hi = np.percentile(x, p)
        if hi <= lo: hi = lo + 1.0
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)
    if per_channel:
        for c in range(arr.shape[0]):
            arr[c] = _stretch(arr[c], robust)
    else:
        arr = _stretch(arr, robust)
    return arr
