# Synthetic Data Playbook (Diffusion + Sensor Model)

## One-command pipeline
```bash
bash tools/pipeline_synth.sh \
  ship \
  "white cargo ship, top-down, aerial" \
  /data/tiles_neg \
  runs/synth \
  runwayml/stable-diffusion-inpainting \
  ckpts/clip/clip_ep5.pt \
  50000
```
**Outputs**
- Images: `runs/synth/ship_inpaint/filtered/imgs/*.png`
- COCO: `runs/synth/ship_inpaint/coco/instances_ship.json`

## Tuning checklist
- LoRA rank 4–16 (start 8). Mask size per class scale (e.g., 24–72 px).
- Inpaint strength 0.75–0.9 (watch for washed backgrounds).
- QC: `min_clip=0.28`, `min_contrast=0.05`, `dedup_thresh=0.04` (tighten if fakes slip through).
- Sensor model: match edge slope/FFT to real tiles.
- Mix ratio: start **30% synthetic** per epoch, cap ≤50%; include all-real minibatches.
- Keep metadata (seed, prompt, bbox) for regen/debug.
