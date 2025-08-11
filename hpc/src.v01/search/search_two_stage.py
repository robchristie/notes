import json
import faiss, torch
from pathlib import Path
import open_clip
from PIL import Image
import numpy as np

class QueryEncoder:
    def __init__(self, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', clip_ckpt='ckpts/clip/clip_ep5.pt', device='cuda'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        if clip_ckpt and Path(clip_ckpt).exists():
            sd = torch.load(clip_ckpt, map_location=device)
            self.model.load_state_dict(sd['model'], strict=False)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(model_name)
    @torch.no_grad()
    def text(self, prompt):
        toks = self.tok([prompt]).to(self.device)
        _, tf, _ = self.model(None, toks)
        return torch.nn.functional.normalize(tf, dim=-1).cpu().numpy()
    @torch.no_grad()
    def image(self, img_path):
        im = Image.open(img_path).convert('RGB')
        t = self.preprocess(im).unsqueeze(0).to(self.device)
        vf, _, _ = self.model(t, None)
        return torch.nn.functional.normalize(vf, dim=-1).cpu().numpy()

def load_index(index_dir):
    idx = faiss.read_index(str(Path(index_dir)/'tiles.ivfpq'))
    idmap = json.loads(Path(index_dir,'idmap.json').read_text())
    return idx, idmap

def search_tiles(index_dir, query_vec, topk=2000):
    idx, idmap = load_index(index_dir)
    D, I = idx.search(query_vec.astype('float32'), topk)
    hits = []
    for i, d in zip(I[0], D[0]):
        if i < 0: continue
        hits.append({'score': float(d), 'tile_id': int(idmap[i]['tile_id']), 'tile_path': idmap[i]['tile_path']})
    return hits

def main(index_dir='runs/pilot/tile_index', text=None, image=None, topk=2000, clip_ckpt='ckpts/clip/clip_ep5.pt'):
    QE = QueryEncoder(clip_ckpt=clip_ckpt)
    if text:
        q = QE.text(text)
    elif image:
        q = QE.image(image)
    else:
        raise SystemExit('Provide --text or --image')
    hits = search_tiles(index_dir, q, topk=topk)
    outp = Path(index_dir)/'tile_hits.json'
    Path(outp).write_text(json.dumps(hits, indent=2))
    print('Wrote', outp, 'topk:', len(hits))

if __name__ == '__main__':
    import fire
    fire.Fire(main)
