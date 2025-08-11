import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
def resample(img, scale):
    h, w = img.shape[:2]; nh, nw = max(1,int(h*scale)), max(1,int(w*scale))
    return cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
def mtf_blur(img, sigma_px=0.6, anisotropy=1.0):
    if sigma_px<=0: return img
    kx=int(6*sigma_px+1)|1; ky=int(6*(sigma_px*anisotropy)+1)|1
    return cv2.GaussianBlur(img,(kx,ky),sigmaX=sigma_px,sigmaY=sigma_px*anisotropy)
def add_noise(img, read_std=0.002, shot_gain=0.02):
    shot=np.random.normal(0,np.sqrt(np.clip(img,0,1)*shot_gain),img.shape)
    read=np.random.normal(0,read_std,img.shape)
    return np.clip(img+shot+read,0,1)
def quantize(img,bits=12):
    maxv=(1<<bits)-1; return np.round(img*maxv).astype(np.uint16)
def robust_scale_uint16(q,bits=12,p1=2,p99=98):
    vals=q.astype(np.float32); lo=np.percentile(vals,p1); hi=np.percentile(vals,p99); hi=max(hi,lo+1.0)
    y=np.clip((vals-lo)/(hi-lo),0,1); return (y*255.0).round().astype(np.uint8)
def process_dir(in_dir,out_dir,src_gsd_cm=25,tgt_gsd_cm=25,mtf_sigma_px=0.6,anisotropy=1.0,read_noise=0.002,shot_gain=0.02,bits=12,rescale_to_uint8=True):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    for p in sorted(list(Path(in_dir).glob('*.png'))+list(Path(in_dir).glob('*.jpg'))):
        img=np.array(Image.open(p).convert('RGB'),dtype=np.float32)/255.0
        scale=tgt_gsd_cm/max(1e-6,src_gsd_cm); img=resample(img,scale); img=mtf_blur(img,mtf_sigma_px,anisotropy); img=add_noise(img,read_noise,shot_gain)
        q=quantize(img,bits)
        if rescale_to_uint8: Image.fromarray(robust_scale_uint16(q,bits)).save(out/f"{p.stem}.png")
        else: Image.fromarray(q,mode='I;16').save(out/f"{p.stem}.png")
    print("Done:", out)
if __name__=='__main__':
    import fire; fire.Fire(process_dir)
