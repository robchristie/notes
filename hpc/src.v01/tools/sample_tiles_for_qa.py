import json, random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

def box_area(bb):
    x,y,w,h = bb
    return max(0.0, w) * max(0.0, h)

def load_coco(gt_json):
    coco = json.load(open(gt_json,'r'))
    id2img = {im['id']: im for im in coco['images']}
    img2anns = {}
    for a in coco['annotations']:
        img2anns.setdefault(a['image_id'], []).append(a)
    return id2img, img2anns

def is_small(bb, imw, imh, max_abs=32*32, max_rel=0.001):
    A = box_area(bb)
    return (A <= max_abs) or (A <= max_rel * (imw*imh))

def edge_density(img_path):
    try:
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None: return 0.0
        e = cv2.Canny(im, 50, 150)
        return float((e>0).mean())
    except Exception:
        return 0.0

def sample(gt_json, tiles_root, out_list='runs/pilot/qa_tiles.txt', n_pos=1500, n_neg=500, max_abs=32*32, max_rel=0.001, shuffle=True):
    tiles_root = Path(tiles_root)
    id2img, img2anns = load_coco(gt_json)
    positives = []; negatives = []
    for img_id, im in id2img.items():
        p = tiles_root / im['file_name']
        if not p.exists(): continue
        anns = img2anns.get(img_id, [])
        if anns:
            small = False
            W = im.get('width', 0) or Image.open(p).size[0]
            H = im.get('height', 0) or Image.open(p).size[1]
            for a in anns:
                if is_small(a['bbox'], W, H, max_abs=max_abs, max_rel=max_rel):
                    small = True; break
            if small: positives.append(str(p))
        else:
            negatives.append(str(p))
    if shuffle: random.shuffle(positives); random.shuffle(negatives)
    neg_scores = [(edge_density(n), n) for n in negatives]
    neg_scores.sort(reverse=True)
    hard_negs = [n for _, n in neg_scores[:n_neg*3]]
    sel_pos = positives[:n_pos]; sel_neg = hard_negs[:n_neg]
    out = Path(out_list); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(sel_pos + sel_neg))
    print(f"Selected {len(sel_pos)} positives and {len(sel_neg)} negatives → {out}")

if __name__ == '__main__':
    import fire
    fire.Fire(sample)
