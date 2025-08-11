import yaml, json
from pathlib import Path
from det.open_vocab_match import OVMatcher, crop_boxes
from det.d2_open_vocab import infer as d2_infer
from det.detr_open_vocab import DETROpenVocab
from det.sam_proposer import SAMProposer

def main(cfg_path='config/config.yaml'):
    cfg = yaml.safe_load(open(cfg_path,'r'))
    out_dir = Path(cfg['paths']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    prompts = cfg['ov']['prompts']
    detections = []
    tiles = sorted(Path(cfg['paths']['tiles_dir']).glob('*.tif'))[:200]
    if cfg['proposer']['type'] == 'detectron2':
        ckpt = cfg['proposer']['d2_ckpt']
        for iid, tile in enumerate(tiles):
            dets = d2_infer(str(tile), ckpt, prompts)
            detections.append({'image_id': iid, 'detections': [{'bbox': b, 'score': p, 'label': prompts[c]} for (b,c,p) in dets]})
    elif cfg['proposer']['type'] == 'detr':
        detr = DETROpenVocab(cfg['proposer']['detr_ckpt'])
        for iid, tile in enumerate(tiles):
            dets = detr.detect(str(tile), prompts, obj_score=cfg['proposer']['obj_score'])
            detections.append({'image_id': iid, 'detections': [{'bbox': b, 'score': p, 'label': prompts[c]} for (b,c,p) in dets]})
    elif cfg['proposer']['type'] == 'sam':
        sam = SAMProposer(cfg['proposer']['sam_type'], cfg['proposer']['sam_ckpt'])
        matcher = OVMatcher(clip_ckpt=cfg['clip']['ckpt'])
        for iid, tile in enumerate(tiles):
            boxes, _ = sam.propose(str(tile))
            crops = crop_boxes(str(tile), boxes)
            sim = matcher.score(crops, text_prompts=prompts)['text'].softmax(1)
            dets = []
            for i, b in enumerate(boxes):
                c = int(sim[i].argmax().item()); p = float(sim[i].max().item())
                dets.append((b,c,p))
            detections.append({'image_id': iid, 'detections': [{'bbox': b, 'score': p, 'label': prompts[c]} for (b,c,p) in dets]})
    out_json = out_dir / 'detections_open_vocab.json'
    json.dump(detections, open(out_json,'w'))
    print('Wrote', out_json)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
