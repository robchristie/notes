from pathlib import Path
import json
import torch
from PIL import Image
import open_clip
import numpy as np
from tqdm import tqdm

class TileEmbedder:
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', clip_ckpt=None, device='cuda'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        if clip_ckpt:
            sd = torch.load(clip_ckpt, map_location=device)
            self.model.load_state_dict(sd['model'], strict=False)
        self.model.eval()
    @torch.no_grad()
    def embed_image(self, img_path):
        im = Image.open(img_path).convert('RGB')
        t = self.preprocess(im).unsqueeze(0).to(self.device)
        img_f, _, _ = self.model(t, None)
        img_f = torch.nn.functional.normalize(img_f, dim=-1)
        return img_f.squeeze(0).cpu().numpy()

def main(tiles_dir, out_dir='runs/pilot/tile_embeds', clip_ckpt='ckpts/clip/clip_ep5.pt'):
    tiles = sorted(Path(tiles_dir).glob('*.tif'))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    E = TileEmbedder(clip_ckpt=clip_ckpt)
    mani = []
    for i, p in enumerate(tqdm(tiles, desc='Embedding tiles')):
        try:
            vec = E.embed_image(str(p))
        except Exception as e:
            print('skip', p, e); continue
        binpath = out / f"{i:09d}.npy"
        np.save(binpath, vec)
        mani.append({'tile_id': i, 'tile_path': str(p), 'embed_path': str(binpath), 'meta': {}})
    (out / 'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in mani))
    print('Wrote manifest with', len(mani), 'tiles to', out)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
