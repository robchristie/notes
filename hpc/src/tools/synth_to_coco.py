import json, argparse
from pathlib import Path
from PIL import Image
def main(imgs_dir, meta, class_name, out_json):
    imgs_dir = Path(imgs_dir)
    rows = [json.loads(l) for l in Path(meta).read_text().splitlines() if l.strip()]
    images = []; annotations = []; categories = [{'id':1,'name':class_name}]
    img_id = 1; ann_id = 1
    for r in rows:
        fn = f"{r['id']:07d}.png"; ip = imgs_dir / fn
        if not ip.exists(): continue
        with Image.open(ip) as im: w, h = im.size
        images.append({'id': img_id, 'file_name': fn, 'width': w, 'height': h})
        x0,y0,x1,y1 = r['bbox']
        annotations.append({'id': ann_id, 'image_id': img_id, 'category_id': 1,
                            'bbox': [float(x0), float(y0), float(x1-x0), float(y1-y0)],
                            'area': float((x1-x0)*(y1-y0)), 'iscrowd': 0})
        img_id += 1; ann_id += 1
    coco = {'images': images, 'annotations': annotations, 'categories': categories}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True); json.dump(coco, open(out_json,'w'))
    print(f"Wrote {out_json} with {len(images)} images and {len(annotations)} annotations")
if __name__ == '__main__':
    import fire; fire.Fire(main)
