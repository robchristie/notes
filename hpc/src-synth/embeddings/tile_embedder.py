from pathlib import Path
import json, torch, numpy as np, open_clip, rasterio
from tqdm import tqdm
from PIL import Image
from utils.imread_rs import load_raster_as_float
from models.sensor_adapter import SensorAdapter

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

class TileEmbedder:
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', clip_ckpt=None, device='cuda',
                 use_adapter=True, adapter_init='pca', adapter_sample_paths=None, robust=(2,98)):
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        if clip_ckpt:
            sd = torch.load(clip_ckpt, map_location=device); self.model.load_state_dict(sd['model'], strict=False)
        self.model.eval()
        self.use_adapter = use_adapter; self.adapter = None
        self.adapter_init = adapter_init; self.adapter_sample_paths = adapter_sample_paths
        self.robust = robust
    @torch.no_grad()
    def _embed_rgb_pil(self, path):
        from torchvision import transforms
        im = Image.open(path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711]),
        ])
        t = preprocess(im).unsqueeze(0).to(self.device)
        img_f, _, _ = self.model(t, None)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)
        return img_f.squeeze(0).cpu().numpy()
    @torch.no_grad()
    def _embed_rs(self, path):
        arr = load_raster_as_float(path, robust=self.robust)  # [C,H,W]
        C = arr.shape[0]
        if self.adapter is None or self.adapter.proj.weight.shape[1] != C:
            sample_paths = self.adapter_sample_paths or [path]
            self.adapter = SensorAdapter(C, 3, init=self.adapter_init, pca_sample_paths=sample_paths, robust=self.robust).to(self.device)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        x = self.adapter(x)
        x = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        x = (x - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)
        img_f, _, _ = self.model(x, None)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)
        return img_f.squeeze(0).cpu().numpy()
    def embed_image(self, img_path):
        try:
            with rasterio.open(img_path) as src:
                bands = src.count; dtype = src.dtypes[0]
            if (bands == 3 and dtype in ('uint8',)):
                return self._embed_rgb_pil(img_path)
            else:
                return self._embed_rs(img_path)
        except Exception:
            return self._embed_rgb_pil(img_path)

def main(tiles_dir, out_dir='runs/pilot/tile_embeds', clip_ckpt='ckpts/clip/clip_ep5.pt',
         use_adapter=True, adapter_init='pca', adapter_sample_glob=None, robust_low=2, robust_high=98):
    tiles = sorted(Path(tiles_dir).glob('*.tif'))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    sample_paths = None
    if adapter_sample_glob:
        root = Path(tiles_dir).parent
        sample_paths = [str(p) for p in root.glob(adapter_sample_glob)]
    E = TileEmbedder(clip_ckpt=clip_ckpt, use_adapter=use_adapter, adapter_init=adapter_init,
                     adapter_sample_paths=sample_paths, robust=(robust_low, robust_high))
    mani = []
    for i, p in enumerate(tqdm(tiles, desc='Embedding tiles')):
        try:
            vec = E.embed_image(str(p))
        except Exception as e:
            print('skip', p, e); continue
        import numpy as np
        binpath = out / f"{i:09d}.npy"
        np.save(binpath, vec)
        mani.append({'tile_id': i, 'tile_path': str(p), 'embed_path': str(binpath), 'meta': {}})
    (out / 'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in mani))
    print('Wrote manifest with', len(mani), 'tiles to', out)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
