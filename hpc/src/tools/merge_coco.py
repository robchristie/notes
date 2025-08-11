import json, argparse, random
from pathlib import Path
def load_coco(p): return json.load(open(p,'r'))
def remap_ids(coco, start_img_id, start_ann_id, source_tag):
    images, annots, img_id_map = [], [], {}
    next_img, next_ann = start_img_id, start_ann_id
    for im in coco['images']:
        img_id_map[im['id']] = next_img
        im2 = dict(im); im2['id'] = next_img; im2['source'] = source_tag; images.append(im2); next_img += 1
    for a in coco['annotations']:
        a2 = dict(a); a2['id'] = next_ann; a2['image_id'] = img_id_map[a['image_id']]; annots.append(a2); next_ann += 1
    return images, annots, next_img, next_ann
def main(real_json, synth_json, images_root_real, images_root_synth, out_json, synth_frac=0.30, seed=123):
    random.seed(seed); real = load_coco(real_json); synth = load_coco(synth_json)
    name2id = {}
    for c in real.get('categories', []): name2id.setdefault(c['name'], len(name2id)+1)
    for c in synth.get('categories', []): name2id.setdefault(c['name'], len(name2id)+1)
    cats = [{'id': cid, 'name': name} for name, cid in name2id.items()]
    def remap_cats(coco):
        old = {c['id']: c['name'] for c in coco['categories']}
        for a in coco['annotations']: a['category_id'] = name2id[old[a['category_id']]]
        coco['categories'] = cats
    remap_cats(real); remap_cats(synth)
    target_synth = int((synth_frac / (1.0 - synth_frac)) * len(real['images']))
    if target_synth < len(synth['images']):
        keep_ids = set(im['id'] for im in random.sample(synth['images'], target_synth))
        synth['images'] = [im for im in synth['images'] if im['id'] in keep_ids]
        synth['annotations'] = [a for a in synth['annotations'] if a['image_id'] in keep_ids]
    images, annotations = [], []
    next_img_id, next_ann_id = 1, 1
    r_imgs, r_anns, next_img_id, next_ann_id = remap_ids(real, next_img_id, next_ann_id, 'real')
    s_imgs, s_anns, next_img_id, next_ann_id = remap_ids(synth, next_img_id, next_ann_id, 'synthetic')
    images.extend(r_imgs); images.extend(s_imgs); annotations.extend(r_anns); annotations.extend(s_anns)
    merged = {'images': images, 'annotations': annotations, 'categories': cats, 'images_root': {'real': str(images_root_real), 'synthetic': str(images_root_synth)} }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True); json.dump(merged, open(out_json,'w'))
    print(f"Wrote {out_json} with {len(images)} images")
if __name__ == '__main__':
    import fire; fire.Fire(main)
