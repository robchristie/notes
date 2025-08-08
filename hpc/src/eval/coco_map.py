from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, argparse

def coco_eval(gt_json, pred_json, iou_type='bbox'):
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = {
        'AP': float(coco_eval.stats[0]),
        'AP50': float(coco_eval.stats[1]),
        'AP75': float(coco_eval.stats[2]),
        'AP_small': float(coco_eval.stats[3]),
        'AP_medium': float(coco_eval.stats[4]),
        'AP_large': float(coco_eval.stats[5]),
    }
    return stats

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('gt_json'); ap.add_argument('pred_json')
    ap.add_argument('--iou_type', default='bbox'); ap.add_argument('--out_json', default=None)
    args = ap.parse_args()
    stats = coco_eval(args.gt_json, args.pred_json, args.iou_type)
    print(json.dumps(stats, indent=2))
    if args.out_json:
        with open(args.out_json,'w') as f: json.dump(stats, f)
