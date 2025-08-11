import json, sys
def to_coco(preds_json):
    data = json.load(open(preds_json, 'r')); out = []
    for item in data:
        iid = item['image_id']
        for d in item['detections']:
            out.append({'image_id': iid, 'category_id': d.get('category_id', 1), 'bbox': d['bbox'], 'score': d['score']})
    return out
if __name__ == '__main__':
    inp, outp = sys.argv[1], sys.argv[2]; json.dump(to_coco(inp), open(outp,'w'))
