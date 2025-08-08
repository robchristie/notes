import os, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ms_ssim
import lpips
import open_clip

class ImgEncoder:
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', clip_ckpt='ckpts/clip/clip_ep5.pt', device='cuda'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        if clip_ckpt and Path(clip_ckpt).exists():
            sd = torch.load(clip_ckpt, map_location=device)
            self.model.load_state_dict(sd['model'], strict=False)
        self.model.eval()
    @torch.no_grad()
    def embed(self, pil_img):
        t = self.preprocess(pil_img.convert('RGB')).unsqueeze(0).to(self.device)
        v, _, _ = self.model(t, None)
        v = torch.nn.functional.normalize(v, dim=-1)
        return v.squeeze(0)

def pil_to_torch(img):
    arr = torch.from_numpy(np.array(img)).float()/255.0
    return arr.permute(2,0,1).unsqueeze(0)

def psnr(a, b):
    a = np.asarray(a).astype(np.float32); b = np.asarray(b).astype(np.float32)
    mse = np.mean((a-b)**2)
    if mse == 0: return 99.0
    return 20*np.log10(255.0/np.sqrt(mse))

@torch.no_grad()
def msssim(a, b):
    a = pil_to_torch(a).to('cuda'); b = pil_to_torch(b).to('cuda')
    return float(ms_ssim(a, b, data_range=1.0).item())

@torch.no_grad()
def lpips_dist(a, b, net='vgg'):
    loss_fn = lpips.LPIPS(net=net).to('cuda')
    A = (pil_to_torch(a)*2-1).to('cuda'); B = (pil_to_torch(b)*2-1).to('cuda')
    return float(loss_fn(A, B).item())

def jxl_roundtrip(src_path, out_dir, distance):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jxl = out_dir / f"tmp_d{distance:.2f}.jxl"
    png = out_dir / f"tmp_d{distance:.2f}.png"
    enc = subprocess.run(['cjxl', src_path, str(jxl), '-d', str(distance), '--num_threads', str(os.cpu_count() or 8)], capture_output=True)
    if enc.returncode != 0: raise RuntimeError(enc.stderr.decode())
    dec = subprocess.run(['djxl', str(jxl), str(png)], capture_output=True)
    if dec.returncode != 0: raise RuntimeError(dec.stderr.decode())
    return str(png)

def evaluate_images(img_paths, distances, clip_ckpt='ckpts/clip/clip_ep5.pt', tmp_dir='.jxl_tmp'):
    enc = ImgEncoder(clip_ckpt=clip_ckpt)
    results = []
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    for p in img_paths:
        orig = Image.open(p).convert('RGB')
        v_orig = enc.embed(orig)
        for d in distances:
            try: recon_path = jxl_roundtrip(p, tmp_dir, d)
            except Exception as e: print('JXL failed on', p, e); continue
            recon = Image.open(recon_path).convert('RGB')
            _psnr = psnr(orig, recon)
            _ssim = ssim(np.array(orig), np.array(recon), channel_axis=2, data_range=255)
            _mss = msssim(orig, recon)
            _lp  = lpips_dist(orig, recon)
            v_rec = enc.embed(recon)
            cos = torch.nn.functional.cosine_similarity(v_orig.unsqueeze(0), v_rec.unsqueeze(0)).item()
            results.append({'image': str(p), 'distance': float(d), 'PSNR': float(_psnr), 'SSIM': float(_ssim), 'MS_SSIM': float(_mss), 'LPIPS': float(_lp), 'EMB_COS': float(cos)})
    return results

def main(tile_glob, distances='0.6,0.8,1.0,1.2,1.5,2.0', out_json='runs/pilot/jxl_qa.json', clip_ckpt='ckpts/clip/clip_ep5.pt'):
    tiles = [str(p) for p in Path('.').glob(tile_glob)]
    if not tiles: raise SystemExit(f'No images matched {tile_glob}')
    dists = [float(x) for x in distances.split(',')]
    res = evaluate_images(tiles, dists, clip_ckpt)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_json,'w'), indent=2)
    print('Wrote', out_json, 'rows:', len(res))

if __name__ == '__main__':
    import fire
    fire.Fire(main)
