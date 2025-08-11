import json, yaml, subprocess, shlex
from pathlib import Path
def maybe_auto_merge(cfg):
    dm = cfg.get('data', {}).get('auto_merge', None)
    if not dm: return cfg
    real_json = dm['real_json']; synth_spec = dm['synth']; out_json = dm['out_json']; synth_frac = float(dm.get('synth_frac', 0.30))
    cmd = f"python tools/merge_coco_multi.py --real_json {shlex.quote(real_json)} --synth_spec {shlex.quote(json.dumps(synth_spec))} --out_json {shlex.quote(out_json)} --synth_frac {synth_frac}"
    print('[auto-merge]', cmd); subprocess.check_call(cmd, shell=True); cfg['paths']['anns_coco'] = out_json; return cfg
def main(cfg_path='config/config_with_automerge.yaml'):
    cfg = yaml.safe_load(open(cfg_path,'r')); cfg = maybe_auto_merge(cfg)
    from det.d2_open_vocab import infer as d2_infer
    from det.detr_open_vocab import DETROpenVocab
    from det.sam_proposer import SAMProposer
    from det.open_vocab_match import OVMatcher, crop_boxes
    tiles = sorted(Path(cfg['paths']['tiles_dir']).glob('*.tif'))[:200]; prompts = cfg['ov']['prompts']
    sa = cfg.get('sensor_adapter', {}); adapter_init = sa.get('init', 'pca')
    detections = []
    if cfg['proposer']['type'] == 'detectron2':
        ckpt = cfg['proposer']['d2_ckpt']
        for iid, tile in enumerate(tiles):
            dets = d2_infer(str(tile), ckpt, prompts, adapter_init=adapter_init, adapter_sample_paths=[str(tile)])
            detections.append({'image_id': iid, 'detections': [{'bbox': b, 'score': p, 'label': prompts[c]} for (b,c,p) in dets]})
    elif cfg['proposer']['type'] == 'detr':
        detr = DETROpenVocab(cfg['proposer']['detr_ckpt'])
        for iid, tile in enumerate(tiles):
            dets = detr.detect(str(tile), prompts, obj_score=cfg['proposer']['obj_score'], adapter_init=adapter_init, adapter_sample_paths=[str(tile)])
            detections.append({'image_id': iid, 'detections': [{'bbox': b, 'score': p, 'label': prompts[c]} for (b,c,p) in dets]})
    elif cfg['proposer']['type'] == 'sam':
        sam = SAMProposer(cfg['proposer']['sam_type'], cfg['proposer']['sam_ckpt']); matcher = OVMatcher(clip_ckpt=cfg['clip']['ckpt'])
        for iid, tile in enumerate(tiles):
            boxes, _ = sam.propose(str(tile), adapter_init=adapter_init, adapter_sample_paths=[str(tile)])
            crops = crop_boxes(str(tile), boxes); sim = matcher.score(crops, text_prompts=prompts)['text'].softmax(1)
            dets=[]; 
            for i,b in enumerate(boxes):
                c=int(sim[i].argmax().item()); p=float(sim[i].max().item()); dets.append((b,c,p))
            detections.append({'image_id': iid, 'detections': [{'bbox': b, 'score': p, 'label': prompts[c]} for (b,c,p) in dets]})
    out_dir = Path(cfg['paths']['out_dir']); out_dir.mkdir(parents=True, exist_ok=True); out_json = out_dir / 'detections_open_vocab.json'
    json.dump(detections, open(out_json,'w')); print('Wrote', out_json)
if __name__ == '__main__':
    import fire; fire.Fire(main)
