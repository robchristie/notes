import numpy as np, torch, rasterio
from PIL import Image
from utils.imread_rs import load_raster_as_float
from models.sensor_adapter import SensorAdapter

def rs_path_to_rgb_uint8(path, adapter_init='pca', adapter_sample_paths=None, robust=(2,98), device='cuda'):
    try:
        with rasterio.open(path) as src:
            bands = src.count; dtype = src.dtypes[0]
        if bands == 3 and dtype in ('uint8',):
            return Image.open(path).convert('RGB')
    except Exception:
        pass
    arr = load_raster_as_float(path, robust=robust)
    C = arr.shape[0]
    adapter = SensorAdapter(C, 3, init=adapter_init, pca_sample_paths=adapter_sample_paths or [path], robust=robust).to(device)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        y = adapter(x).clamp(0,1).squeeze(0).cpu().numpy()
    y = (np.transpose(y, (1,2,0)) * 255.0).round().astype(np.uint8)
    return Image.fromarray(y, mode='RGB')
