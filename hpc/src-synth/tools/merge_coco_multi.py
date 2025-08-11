import json, argparse, random
from pathlib import Path
def load_json(p): return json.load(open(p,'r'))
def unify_categories(dsets):
    name2id = {}
    for ds in dsets:
        for c in ds.get('categories', []):
            name2id.setdefault(c['name'], len(name2id)+1)
    cats = [{'id': cid, 'name': name} for name, cid in name2id.items()]
    for ds in dsets:
        old = {c['id']: c['name'] for c in ds.get('categories', [])}
        for a in ds.get('annotations', []):
            a['category_id'] = name2id[old[a['category_id']]]
        ds['categories'] = cats
    return cats
def remap_ids(coco, start_img_id, start_ann_id, source_tag):
    images, annots, img_id_map = [], [], {}
    next_img, next_ann = start_img_id, start_ann_id
    for im in coco['images']:
        img_id_map[im['id']] = next_img
        im2 = dict(im); im2['id'] = next_img; im2['source'] = source_tag; images.append(im2); next_img += 1
    for a in coco['annotations']:
        a2 = dict(a); a2['id'] = next_ann; a2['image_id'] = img_id_map[a['image_id']]; annots.append(a2); next_ann += 1
    return images, annots, next_img, next_ann
def main(real_json, synth_spec, out_json, synth_frac=0.30, seed=123):
    random.seed(seed)
    real = load_json(real_json); specs = json.loads(synth_spec)
    synth_sets = []
    for s in specs:
        ds = load_json(s['json']); ds['_name'] = s.get('name','synthetic'); ds['_images_root'] = s.get('images_root',''); synth_sets.append(ds)
    cats = unify_categories([real] + synth_sets)
    n_real = len(real['images']); target_synth = int((synth_frac / max(1e-9,(1.0 - synth_frac))) * n_real)
    all_synth_imgs = []
    for ds in synth_sets:
        for im in ds['images']:
            im['_ds_name'] = ds['_name']; all_synth_imgs.append((ds, im))
    if target_synth < len(all_synth_imgs):
        keep = set(im['id'] for _, im in random.sample(all_synth_imgs, target_synth))
        for ds in synth_sets:
            ds['images'] = [im for im in ds['images'] if im['id'] in keep]
            keep_ids = set(im['id'] for im in ds['images'])
            ds['annotations'] = [a for a in ds['annotations'] if a['image_id'] in keep_ids]
    images, annotations = [], []; images_root = {'real': ''}
    next_img_id, next_ann_id = 1, 1
    r_imgs, r_anns, next_img_id, next_ann_id = remap_ids(real, next_img_id, next_ann_id, 'real')
    images.extend(r_imgs); annotations.extend(r_anns)
    for ds in synth_sets:
        tag = f"synthetic:{ds['_name']}"
        s_imgs, s_anns, next_img_id, next_ann_id = remap_ids(ds, next_img_id, next_ann_id, tag)
        images.extend(s_imgs); annotations.extend(s_anns); images_root[tag] = ds.get('_images_root','')
    merged = {'images': images, 'annotations': annotations, 'categories': cats, 'images_root': images_root}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True); json.dump(merged, open(out_json,'w'))
    print(f"Wrote {out_json} with {len(images)} images")
if __name__ == '__main__':
    import fire; fire.Fire(main)
