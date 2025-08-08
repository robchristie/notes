import os
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from det.open_vocab_match import OVMatcher, crop_boxes

def train(img_dir, ann_file, out_dir='ckpts/d2', base_cfg='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', iters=9000,
          ims_per_batch=8, base_lr=5e-4):
    name = 'coco_objectness'
    register_coco_instances(name, {}, ann_file, img_dir)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_cfg))
    cfg.DATASETS.TRAIN = (name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = iters
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_cfg)
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def infer(tile_path, predictor_ckpt, prompts, base_cfg='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_cfg))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = predictor_ckpt
    cfg.MODEL.DEVICE = 'cuda'
    predictor = DefaultPredictor(cfg)
    outputs = predictor(tile_path)
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
    matcher = OVMatcher(clip_ckpt='ckpts/clip/clip_ep5.pt')
    crops = crop_boxes(tile_path, boxes)
    scores = matcher.score(crops, text_prompts=prompts)['text'].softmax(dim=1).cpu().numpy()
    cls = np.argmax(scores, axis=1); prob = np.max(scores, axis=1)
    return [(boxes[i], int(cls[i]), float(prob[i])) for i in range(len(boxes))]

if __name__ == "__main__":
    import fire
    fire.Fire({'train': train, 'infer': infer})
