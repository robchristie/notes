import json, sys

def to_coco(preds_json):
    data = json.load(open(preds_json, 'r'))
    out = []
    for item in data:
        iid = item['image_id']
        for d in item['detections']:
            rec = {
                'image_id': iid,
                'category_id': d['category_id'] if 'category_id' in d else 1,
                'bbox': d['bbox'],
                'score': d['score'],
            }
            out.append(rec)
    return out

if __name__ == '__main__':
    inp, outp = sys.argv[1], sys.argv[2]
    json.dump(to_coco(inp), open(outp,'w'))
