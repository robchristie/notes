import torch, torch.nn as nn
import numpy as np, random, rasterio

class SensorAdapter(nn.Module):
    def __init__(self, in_ch, out_ch=3, init='avg', pca_sample_paths=None, pca_pixel_samples=200_000, robust=(2,98)):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        with torch.no_grad():
            W = self.proj.weight; B = self.proj.bias
            if init == 'avg':
                W.zero_(); W[:, :, 0, 0] = 1.0 / in_ch; B.zero_()
            elif init == 'first3' and in_ch >= 3:
                W.zero_(); B.zero_()
                for o in range(min(out_ch, in_ch)): W[o, o, 0, 0] = 1.0
            elif init == 'pca':
                w, b = self._pca_init(in_ch, pca_sample_paths or [], pca_pixel_samples, robust)
                W.zero_(); B.zero_()
                W[:, :, 0, 0] = torch.from_numpy(w).float()
                B[:] = torch.from_numpy(b).float()
    def forward(self, x):
        return self.proj(x).clamp(0,1)
    @staticmethod
    def _pca_init(in_ch, paths, n_pixels, robust):
        if not paths:
            w = np.ones((3, in_ch), dtype=np.float32) / in_ch
            b = np.zeros((3,), dtype=np.float32)
            return w, b
        rng = random.Random(1337); samples = []; needed = n_pixels
        for p in paths:
            try:
                with rasterio.open(p) as src:
                    arr = src.read().astype(np.float32)
                for c in range(arr.shape[0]):
                    lo, hi = np.percentile(arr[c], robust) if robust else (arr[c].min(), arr[c].max())
                    if hi <= lo: hi = lo + 1.0
                    arr[c] = np.clip((arr[c]-lo)/(hi-lo), 0, 1)
                H, W = arr.shape[1], arr.shape[2]; total = H*W; take = min(needed, total)
                if take <= 0: break
                import numpy as np
                idx = np.array(rng.sample(range(total), take))
                vec = arr.reshape(arr.shape[0], -1)[:, idx].T
                samples.append(vec); needed -= take
                if needed <= 0: break
            except Exception:
                continue
        if not samples:
            w = np.ones((3, in_ch), dtype=np.float32) / in_ch
            b = np.zeros((3,), dtype=np.float32)
            return w, b
        import numpy as np
        X = np.concatenate(samples, axis=0); mu = X.mean(axis=0, keepdims=True); Xc = X - mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False); comps = Vt[:3]
        for i in range(comps.shape[0]): comps[i] /= (np.linalg.norm(comps[i]) + 1e-8)
        b = - (comps @ mu.T).reshape(-1)
        return comps.astype(np.float32), b.astype(np.float32)
