from transformers import DetrImageProcessor, DetrForObjectDetection
import torch, numpy as np
from utils.rs_to_rgb import rs_path_to_rgb_uint8
from det.open_vocab_match import OVMatcher

class DETROpenVocab:
    def __init__(self, hf_ckpt='facebook/detr-resnet-50', device='cuda'):
        self.device = device
        self.proc = DetrImageProcessor.from_pretrained(hf_ckpt)
        self.model = DetrForObjectDetection.from_pretrained(hf_ckpt).to(device)
        self.model.eval()
        self.matcher = OVMatcher(clip_ckpt='ckpts/clip/clip_ep5.pt', device=device)
    @torch.no_grad()
    def detect(self, image_path, text_prompts, obj_score=0.1, adapter_init='pca', adapter_sample_paths=None):
        im = rs_path_to_rgb_uint8(image_path, adapter_init=adapter_init, adapter_sample_paths=adapter_sample_paths)
        inputs = self.proc(images=im, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0].softmax(-1)[..., :-1]
        boxes  = outputs.pred_boxes[0]
        scores, _ = logits.max(-1)
        keep = (scores > obj_score).nonzero(as_tuple=True)[0]
        boxes = boxes[keep].cpu().numpy().tolist()
        W, H = im.size; xyxy = []
        for cx,cy,w,h in boxes:
            x0 = (cx - w/2)*W; x1 = (cx + w/2)*W
            y0 = (cy - h/2)*H; y1 = (cy + h/2)*H
            xyxy.append([int(x0), int(y0), int(x1), int(y1)])
        crops = []
        for x0,y0,x1,y1 in xyxy:
            x0 = max(0, int(x0-4)); y0 = max(0, int(y0-4))
            x1 = min(W, int(x1+4)); y1 = min(H, int(y1+4))
            crops.append(im.crop((x0,y0,x1,y1)))
        sim = self.matcher.score(crops, text_prompts=text_prompts)['text'].softmax(1).cpu().numpy()
        cls = np.argmax(sim, axis=1); prob = np.max(sim, axis=1)
        return [(xyxy[i], int(cls[i]), float(prob[i])) for i in range(len(xyxy))]
