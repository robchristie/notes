import os, json, shutil
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np, torch, open_clip
def clip_score(img,prompt,model_name='ViT-B-16',pretrained='laion2b_s34b_b88k',ckpt=None,device='cuda'):
    model,_,preprocess=open_clip.create_model_and_transforms(model_name,pretrained=pretrained,device=device)
    if ckpt and Path(ckpt).exists(): sd=torch.load(ckpt, map_location=device); model.load_state_dict(sd['model'], strict=False)
    model.eval(); tok=open_clip.get_tokenizer(model_name)([prompt]).to(device)
    with torch.no_grad():
        im=preprocess(img).unsqueeze(0).to(device); img_f,txt_f,_=model(im,tok)
        img_f=torch.nn.functional.normalize(img_f,dim=-1); txt_f=torch.nn.functional.normalize(txt_f,dim=-1)
        return float((img_f@txt_f.t()).item())
def local_contrast(img): g=img.convert('L'); stat=ImageStat.Stat(g); return float(stat.stddev[0])/255.0
def embed(img,model_name='ViT-B-16',pretrained='laion2b_s34b_b88k',ckpt=None,device='cuda'):
    model,_,preprocess=open_clip.create_model_and_transforms(model_name,pretrained=pretrained,device=device)
    if ckpt and Path(ckpt).exists(): sd=torch.load(ckpt, map_location=device); model.load_state_dict(sd['model'], strict=False)
    model.eval()
    with torch.no_grad():
        im=preprocess(img).unsqueeze(0).to(device); img_f,_,_=model(im,None)
        return torch.nn.functional.normalize(img_f,dim=-1).cpu().numpy()[0]
def run(imgs,meta,out,prompt,clip_ckpt=None,min_clip=0.28,min_contrast=0.05,min_box_px=12,dedup_thresh=0.04):
    outp=Path(out); (outp/'imgs').mkdir(parents=True,exist_ok=True); keep_meta=[]
    rows=[json.loads(l) for l in Path(meta).read_text().splitlines() if l.strip()]; kept=[]
    for r in rows:
        ip=Path(imgs)/f"{r['id']:07d}.png"
        if not ip.exists(): continue
        img=Image.open(ip).convert('RGB'); x0,y0,x1,y1=r.get('bbox',[0,0,0,0])
        if (x1-x0)<min_box_px or (y1-y0)<min_box_px: continue
        if local_contrast(img)<min_contrast: continue
        if clip_score(img,prompt,ckpt=clip_ckpt)<min_clip: continue
        f=embed(img,ckpt=clip_ckpt)
        if any(np.dot(f,g)>(1.0-dedup_thresh) for g in kept): continue
        kept.append(f); shutil.copy(str(ip), str(outp/'imgs'/ip.name)); keep_meta.append(r)
    Path(outp/'meta.jsonl').write_text("\n".join(json.dumps(r) for r in keep_meta)); print("Kept",len(keep_meta),"of",len(rows),"->",outp)
if __name__=='__main__':
    import fire; fire.Fire(run)
