import os, random, json
from pathlib import Path
from PIL import Image
import numpy as np, cv2, torch
from diffusers import StableDiffusionImg2ImgPipeline
def poisson_blend(bg, chip, center):
    bg_bgr = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR); chip_bgr = cv2.cvtColor(np.array(chip), cv2.COLOR_RGB2BGR)
    mask = 255*np.ones(chip_bgr.shape[:2], dtype=np.uint8); blended = cv2.seamlessClone(chip_bgr, bg_bgr, mask, center, cv2.MIXED_CLONE)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
def run(bg_dir, chip_dir, out_dir, prompt='aerial photo, top-down', sd_model=None, strength=0.2, num=1000, seed=123):
    random.seed(seed); out = Path(out_dir); (out/'imgs').mkdir(parents=True, exist_ok=True); (out/'meta.jsonl').touch()
    bgs = sorted(list(Path(bg_dir).glob('*.tif'))+list(Path(bg_dir).glob('*.png'))+list(Path(bg_dir).glob('*.jpg'))); chips = sorted(list(Path(chip_dir).glob('*.png'))+list(Path(chip_dir).glob('*.jpg')))
    if not bgs or not chips: raise SystemExit('Missing backgrounds or chips'); pipe = None
    if sd_model:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model, torch_dtype=torch.float16).to('cuda'); pipe.enable_xformers_memory_efficient_attention()
    for i in range(num):
        bg = Image.open(random.choice(bgs)).convert('RGB').resize((512,512), Image.BICUBIC); chip = Image.open(random.choice(chips)).convert('RGB')
        bw, bh = bg.size; cw, ch = chip.size; scale = random.uniform(0.3, 0.8); chip = chip.resize((int(cw*scale), int(ch*scale)), Image.BICUBIC)
        x = random.randint(0, bw - chip.size[0]); y = random.randint(0, bh - chip.size[1])
        blended = poisson_blend(bg, chip, (x + chip.size[0]//2, y + chip.size[1]//2)); out_img = blended if pipe is None else pipe(prompt=prompt, image=blended, strength=strength, guidance_scale=5.0).images[0]
        out_img.save(out/'imgs'/f'{i:07d}.png'); (out/'meta.jsonl').open('a').write(json.dumps({'id':i,'bbox':[x,y,x+chip.size[0],y+chip.size[1]]})+'\n')
    print('Done:', out)
if __name__ == '__main__':
    import fire; fire.Fire(run)
