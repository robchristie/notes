import json, argparse, numpy as np
from pycocotools.coco import COCO
def iou_xyxy(a, b):
    ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
    xa0, ya0, xa1, ya1 = max(ax0,bx0), max(ay0,by0), min(ax1,bx1), min(ay1,by1)
    iw, ih = max(0, xa1-xa0), max(0, ya1-ya0); inter = iw*ih
    ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter/ua if ua>0 else 0.0
def eval_prompts(coco_gt_json, det_json, prompt_to_cat, iou_thresh=0.5, ks=(1,5,10)):
    coco = COCO(coco_gt_json); preds = json.load(open(det_json,'r'))
    img2gt = {}
    for ann in coco.anns.values():
        key = (ann['image_id'], ann['category_id'])
        img2gt.setdefault(key, []).append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
    results = {}
    for prompt, cat_id in prompt_to_cat.items():
        dets = []
        for item in preds:
            iid = item['image_id']
            for d in item['detections']:
                if d.get('label') == prompt:
                    dets.append((iid, d['bbox'], d['score']))
        if not dets:
            results[prompt] = {'AP': 0.0, **{f'P@{k}':0.0 for k in ks}}; continue
        dets.sort(key=lambda x: -x[2]); tp, fp, matched = [], [], set()
        for iid, bb, sc in dets:
            gtlist = img2gt.get((iid, cat_id), []); ious = [iou_xyxy(bb, g) for g in gtlist]
            j = int(np.argmax(ious)) if ious else -1
            if ious and ious[j] >= iou_thresh and (iid,cat_id,j) not in matched:
                tp.append(1); fp.append(0); matched.add((iid,cat_id,j))
            else:
                tp.append(0); fp.append(1)
        tp = np.cumsum(tp); fp = np.cumsum(fp)
        total_gt = sum(len(v) for k,v in img2gt.items() if k[1]==cat_id)
        recall = tp / max(1, total_gt); precision = tp / np.maximum(1, tp+fp)
        ap = 0.0
        for t in np.linspace(0,1,101):
            p = precision[recall>=t].max() if np.any(recall>=t) else 0; ap += p
        ap /= 101.0
        p_at = {}
        for k in ks:
            hits = 0
            for iid,bb,sc in dets[:k]:
                gtlist = img2gt.get((iid, cat_id), [])
                if any(iou_xyxy(bb,g)>=iou_thresh for g in gtlist): hits += 1
            p_at[f'P@{k}'] = hits/float(k)
        results[prompt] = {'AP': float(ap), **{k: float(v) for k,v in p_at.items()}}
    return results
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('coco_gt'); ap.add_argument('det_json'); ap.add_argument('prompt_to_cat_json'); ap.add_argument('--out_json', default=None)
    args = ap.parse_args(); import json as js; ptc = js.load(open(args.prompt_to_cat_json,'r')); out = eval_prompts(args.coco_gt, args.det_json, ptc)
    print(json.dumps(out, indent=2)); 
    if args.out_json: json.dump(out, open(args.out_json,'w'))
