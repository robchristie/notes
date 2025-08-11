import os, random, json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
def random_mask(w, h, size_range=(24,72), shape='ellipse'):
    size = random.randint(size_range[0], size_range[1]); x0 = random.randint(0, max(1, w-size)); y0 = random.randint(0, max(1, h-size))
    x1, y1 = x0 + size, y0 + size; m = Image.new("L", (w, h), 0); d = ImageDraw.Draw(m)
    (d.ellipse if shape=='ellipse' else d.rectangle)([x0,y0,x1,y1], fill=255); return m, (x0,y0,x1,y1)
def load_inpaint(model_id, device='cuda', sdxl=False):
    Pipe = StableDiffusionXLInpaintPipeline if sdxl else StableDiffusionInpaintPipeline
    pipe = Pipe.from_pretrained(model_id, torch_dtype=torch.float16).to(device); pipe.enable_xformers_memory_efficient_attention(); return pipe
def maybe_load_lora(pipe, lora_path, weight=0.8):
    if lora_path and os.path.exists(lora_path): pipe.load_lora_weights(lora_path); pipe.fuse_lora(weights=weight)
def generate(bg_dir, out_dir, prompt, negative_prompt='', model='runwayml/stable-diffusion-inpainting', sdxl=False, lora_path=None, mask_size_px=(24,72), num=100, seed=123, strength=0.85, guidance_scale=7.0):
    random.seed(seed); out = Path(out_dir); (out/'imgs').mkdir(parents=True, exist_ok=True); (out/'masks').mkdir(parents=True, exist_ok=True)
    bgs = sorted(list(Path(bg_dir).glob('*.tif'))+list(Path(bg_dir).glob('*.png'))+list(Path(bg_dir).glob('*.jpg'))); 
    if not bgs: raise SystemExit(f'No backgrounds in {bg_dir}')
    pipe = load_inpaint(model, 'cuda', sdxl=sdxl); maybe_load_lora(pipe, lora_path)
    for i in range(num):
        bg = Image.open(random.choice(bgs)).convert('RGB').resize((512,512), Image.BICUBIC); mask, bbox = random_mask(512, 512, size_range=mask_size_px, shape='ellipse')
        gen = pipe(prompt=prompt, negative_prompt=negative_prompt, image=bg, mask_image=mask, strength=strength, guidance_scale=guidance_scale).images[0]
        gen.save(out/'imgs'/f'{i:07d}.png'); mask.save(out/'masks'/f'{i:07d}.png'); (out/'meta.jsonl').open('a').write(json.dumps({'id':i,'bbox':bbox,'prompt':prompt})+'\n')
    print('Done:', out)
if __name__ == '__main__':
    import fire; fire.Fire(generate)
