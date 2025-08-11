import os, cv2, numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from det.open_vocab_match import OVMatcher
from utils.rs_to_rgb import rs_path_to_rgb_uint8

def infer(tile_path, predictor_ckpt, prompts, base_cfg='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
          adapter_init='pca', adapter_sample_paths=None):
    cfg = get_cfg(); cfg.merge_from_file(model_zoo.get_config_file(base_cfg))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1; cfg.MODEL.WEIGHTS = predictor_ckpt; cfg.MODEL.DEVICE = 'cuda'
    predictor = DefaultPredictor(cfg)
    rgb = rs_path_to_rgb_uint8(tile_path, adapter_init=adapter_init, adapter_sample_paths=adapter_sample_paths)
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    outputs = predictor(bgr)
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
    matcher = OVMatcher(clip_ckpt='ckpts/clip/clip_ep5.pt')
    W, H = rgb.size; crops = []
    for x0,y0,x1,y1 in boxes:
        x0 = max(0, int(x0-4)); y0 = max(0, int(y0-4))
        x1 = min(W, int(x1+4)); y1 = min(H, int(y1+4))
        crops.append(rgb.crop((x0,y0,x1,y1)))
    scores = matcher.score(crops, text_prompts=prompts)['text'].softmax(dim=1).cpu().numpy()
    cls = np.argmax(scores, axis=1); prob = np.max(scores, axis=1)
    return [(boxes[i], int(cls[i]), float(prob[i])) for i in range(len(boxes))]
