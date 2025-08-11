import json, argparse, csv
from pathlib import Path

def load_rows(path):
    p = Path(path)
    rows = []
    if p.suffix.lower() == '.csv':
        with open(p, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append({'filename': r['filename'], 'bbox': [float(r['xmin']), float(r['ymin']), float(r['xmax']), float(r['ymax'])], 'category': r['category']})
    else:
        with open(p, 'r') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line); rows.append(r)
    return rows

def to_coco(rows, images_dir, class_agnostic=False):
    images = {}; annotations = []; categories = {}; ann_id = 1; img_id = 1
    if class_agnostic:
        categories = {1: {'id':1,'name':'object'}}
        cat2id = {'object': 1}
    else:
        cat2id = {}
    for r in rows:
        fn = r['filename']
        if fn not in images:
            images[fn] = {'id': img_id, 'file_name': fn, 'width': 0, 'height': 0}
            img_id += 1
        x0,y0,x1,y1 = r['bbox']; w,h = max(0.0,x1-x0), max(0.0,y1-y0)
        cat = 'object' if class_agnostic else r['category']
        if not class_agnostic and cat not in cat2id:
            cid = len(cat2id)+1; cat2id[cat] = cid
        cid = 1 if class_agnostic else cat2id[cat]
        annotations.append({'id': ann_id, 'image_id': images[fn]['id'], 'category_id': cid, 'bbox': [float(x0), float(y0), float(w), float(h)], 'area': float(w*h), 'iscrowd': 0})
        ann_id += 1
    if not class_agnostic:
        categories = [{'id': cid, 'name': name} for name, cid in cat2id.items()]
    else:
        categories = [categories[1]]
    return {'images': list(images.values()), 'annotations': annotations, 'categories': categories}

def main(inp, out_json, images_dir, class_agnostic=False):
    rows = load_rows(inp)
    coco = to_coco(rows, images_dir, class_agnostic)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out_json,'w'))
    print('Wrote', out_json, 'with', len(coco['images']), 'images and', len(coco['annotations']), 'annotations')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
