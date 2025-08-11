from pathlib import Path
import torch, numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torchvision

def masks_to_boxes(masks):
    boxes = []
    for m in masks:
        y, x = np.where(m)
        if len(x) == 0: continue
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        boxes.append([int(x0), int(y0), int(x1), int(y1)])
    return boxes

def nms(boxes, iou_thresh=0.5):
    if len(boxes) == 0: return []
    b = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.ones((len(boxes),))
    keep = torchvision.ops.nms(b, scores, iou_thresh)
    return [boxes[i] for i in keep.tolist()]

class SAMProposer:
    def __init__(self, sam_type='vit_h', sam_ckpt='/path/to/sam_vit_h_4b8939.pth', device='cuda'):
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        sam.to(device)
        self.gen = SamAutomaticMaskGenerator(
            sam, points_per_side=24, pred_iou_thresh=0.86,
            stability_score_thresh=0.92, box_nms_thresh=0.5,
            min_mask_region_area=64,
        )
    @torch.no_grad()
    def propose(self, tile_path):
        im = np.array(Image.open(tile_path).convert('RGB'))
        masks = self.gen.generate(im)
        boxes = masks_to_boxes([m['segmentation'] for m in masks])
        boxes = nms(boxes, 0.5)
        return boxes, masks

if __name__ == "__main__":
    import fire
    fire.Fire(SAMProposer)
