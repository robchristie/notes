I've appended a **Tile Sampler** that preferentially selects tiles with labeled small objects, so the JPEG XL QA process targets the hard cases rather than random backgrounds.

---

## A14.4 Stratified Small‑Object Tile Sampler (tools/sample\_small\_object\_tiles.py)

```python
import json, random
from pathlib import Path
from collections import defaultdict

# Assumes COCO‑style annotations
# Sample tiles containing small objects preferentially, but include some background for control.

def load_coco(ann_path):
    return json.load(open(ann_path, 'r'))

def build_tile_index(coco, min_area_px=0, max_area_px=4096):
    idx = defaultdict(list)
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        area = w * h
        if min_area_px <= area <= max_area_px:
            idx[ann['image_id']].append(ann)
    return idx

def sample_tiles(coco, tile_dir, out_list, num_pos=1000, num_bg=500):
    tile_dir = Path(tile_dir)
    id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    tile_index = build_tile_index(coco)
    
    pos_ids = list(tile_index.keys())
    bg_ids = [i for i in id_to_filename if i not in tile_index]
    
    pos_sample = random.sample(pos_ids, min(num_pos, len(pos_ids)))
    bg_sample = random.sample(bg_ids, min(num_bg, len(bg_ids)))
    
    all_ids = pos_sample + bg_sample
    random.shuffle(all_ids)
    
    with open(out_list, 'w') as f:
        for img_id in all_ids:
            f.write(str(tile_dir / id_to_filename[img_id]) + '\n')
    print(f"Wrote {len(all_ids)} paths to {out_list}")

def main(coco_json, tile_dir, out_list, num_pos=1000, num_bg=500):
    coco = load_coco(coco_json)
    sample_tiles(coco, tile_dir, out_list, num_pos, num_bg)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
```

### Usage

```bash
python tools/sample_small_object_tiles.py \
  --coco_json data/annotations.json \
  --tile_dir data/tiles \
  --out_list runs/pilot/sampled_tiles.txt \
  --num_pos 1500 --num_bg 500

# Then pass the list to the JPEG XL QA script:
python tools/codec_qa_jxl.py --tile_glob "$(cat runs/pilot/sampled_tiles.txt)" \
  --distances '0.6,0.8,1.0,1.2,1.5,2.0' \
  --out_json runs/pilot/jxl_qa.json \
  --clip_ckpt ckpts/clip/clip_ep5.pt
```

**Why:** This focuses your QA on tiles most likely to show compression‑induced degradation for small/rare targets, while still including a baseline of background tiles to avoid overfitting compression settings to only positives.



---

## A15) QA Sampler — Preferentially pick tiles with **small objects** (and hard backgrounds)

Use this to build a focused set for codec QA (so compression decisions are driven by the hardest cases).

### A15.1 Small‑Object Sampler from COCO (tools/sample\_tiles\_for\_qa.py)

Selects tiles with **small** GT boxes (by absolute pixels or fraction of image area) and mixes in **hard negatives** (background tiles with high edge density). Writes a file list you can feed to A14.

```python
# tools/sample_tiles_for_qa.py
import json, random, argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

"""
Outputs a newline‑separated list of image paths selected for QA.

Selection:
- Positives: images with at least one GT bbox meeting "small" criteria.
- Negatives: images with no GT (or ignoring GT), ranked by edge density (Canny) to get hard backgrounds.
"""

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
    # max_abs in pixels^2, max_rel as fraction of image area
    A = box_area(bb)
    return (A <= max_abs) or (A <= max_rel * (imw*imh))


def edge_density(img_path):
    try:
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return 0.0
        e = cv2.Canny(im, 50, 150)
        return float((e>0).mean())
    except Exception:
        return 0.0


def sample(gt_json, tiles_root, out_list='runs/pilot/qa_tiles.txt', n_pos=1500, n_neg=500,
           max_abs=32*32, max_rel=0.001, shuffle=True, require_labels=True):
    tiles_root = Path(tiles_root)
    id2img, img2anns = load_coco(gt_json)

    positives = []
    negatives = []

    for img_id, im in id2img.items():
        p = tiles_root / im['file_name']
        if not p.exists():
            continue
        anns = img2anns.get(img_id, [])
        if anns:
            # any small box?
            small = False
            W = im.get('width', 0) or Image.open(p).size[0]
            H = im.get('height', 0) or Image.open(p).size[1]
            for a in anns:
                if is_small(a['bbox'], W, H, max_abs=max_abs, max_rel=max_rel):
                    small = True; break
            if small:
                positives.append(str(p))
        else:
            negatives.append(str(p))

    if shuffle:
        random.shuffle(positives); random.shuffle(negatives)

    # Rank negatives by edge density (hard backgrounds first)
    neg_scores = [(edge_density(n), n) for n in negatives]
    neg_scores.sort(reverse=True)
    hard_negs = [n for _, n in neg_scores[:n_neg*3]]  # preselect top‑3x

    sel_pos = positives[:n_pos]
    sel_neg = hard_negs[:n_neg]

    out = Path(out_list); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('
'.join(sel_pos + sel_neg))
    print(f"Selected {len(sel_pos)} positives and {len(sel_neg)} negatives → {out}")

if __name__ == '__main__':
    import fire
    fire.Fire(sample)
```

### A15.2 Using the sampler with A14

```bash
# Build a focused QA set: lots of small objects + hard backgrounds
python tools/sample_tiles_for_qa.py \
  --gt_json /data/anns/coco_multiclass.json \
  --tiles_root /data/tiles \
  --out_list runs/pilot/qa_tiles.txt \
  --n_pos 2000 --n_neg 1000 --max_abs $((24*24)) --max_rel 0.0006

# Run codec QA only on those tiles
python - <<'PY'
from pathlib import Path
paths = [p.strip() for p in Path('runs/pilot/qa_tiles.txt').read_text().splitlines() if p.strip()]
# write a temp glob list by symlinking, or modify codec_qa to accept a list. Quick hack:
import json; json.dump(paths, open('runs/pilot/qa_list.json','w'))
PY

# If you prefer, modify tools/codec_qa_jxl.py to accept --list_json runs/pilot/qa_list.json
# or just point --tile_glob to a folder containing symlinks to those tiles.
```

**Notes:**

- Adjust `max_abs`/`max_rel` to match your notion of “small” in pixels or relative to tile size.
- For multi‑category balancing, extend the sampler to stratify by `category_id` (say the word and I’ll add it).

