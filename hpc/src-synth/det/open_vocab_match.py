from PIL import Image
import torch, open_clip

class OVMatcher:
    def __init__(self, clip_ckpt=None, model_name='ViT-B-16', pretrained='laion2b_s34b_b88k', device='cuda'):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        if clip_ckpt:
            sd = torch.load(clip_ckpt, map_location=device); self.model.load_state_dict(sd['model'], strict=False)
        self.tokenizer = open_clip.get_tokenizer(model_name); self.model.eval()
    @torch.no_grad()
    def embed_text(self, prompts):
        toks = self.tokenizer(prompts).to(self.device); _, txt_f, _ = self.model(None, toks)
        return torch.nn.functional.normalize(txt_f, dim=-1)
    @torch.no_grad()
    def embed_images(self, imgs):
        ims = torch.stack([self.preprocess(im) for im in imgs]).to(self.device)
        img_f, _, _ = self.model(ims, None)
        return torch.nn.functional.normalize(img_f, dim=-1)
    @torch.no_grad()
    def score(self, crops, text_prompts=None, visual_prototypes=None):
        feats = self.embed_images(crops); scores = {}
        if text_prompts:
            txt = self.embed_text(text_prompts); scores['text'] = feats @ txt.t()
        if visual_prototypes:
            vp = self.embed_images(visual_prototypes); scores['visual'] = feats @ vp.t()
        return scores

def crop_boxes(tile_path, boxes, pad=4):
    im = Image.open(tile_path).convert('RGB'); W, H = im.size; crops = []
    for x0,y0,x1,y1 in boxes:
        x0 = max(0, int(x0-pad)); y0 = max(0, int(y0-pad))
        x1 = min(W, int(x1+pad)); y1 = min(H, int(y1+pad))
        crops.append(im.crop((x0,y0,x1,y1)))
    return crops
