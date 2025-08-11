from pathlib import Path
import json
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

def tile_geotiff(src_path, out_dir, tile_size=1024, stride=1024, min_var=10.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        W, H = src.width, src.height
        profile = src.profile
        meta = {"crs": getattr(src, "crs", None), "transform": getattr(src, "transform", None)}
        ext_meta = {}
        tid = 0
        for y in tqdm(range(0, H - tile_size + 1, stride)):
            for x in range(0, W - tile_size + 1, stride):
                win = Window(x, y, tile_size, tile_size)
                chip = src.read(window=win)
                if np.var(chip.astype(np.float32)) < min_var:
                    continue
                t_profile = profile.copy()
                t_profile.update({
                    "height": tile_size, "width": tile_size,
                    "transform": rasterio.windows.transform(win, src.transform)
                })
                tif_path = out_dir / f"tile_{tid:09d}.tif"
                with rasterio.open(tif_path, "w", **t_profile) as dst:
                    dst.write(chip)
                geo = {
                    "src": str(src_path), "tile_id": tid,
                    "window": [x, y, tile_size, tile_size],
                    "crs": str(meta["crs"]), "transform": list(t_profile["transform"]),
                }
                geo.update(ext_meta)
                (out_dir / f"tile_{tid:09d}.json").write_text(json.dumps(geo))
                tid += 1

if __name__ == "__main__":
    import fire
    fire.Fire(tile_geotiff)
