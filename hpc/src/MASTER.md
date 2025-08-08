# Gigapixel Open‑Vocabulary Satellite Detection — Functional Bundle

This repository contains a **single-node starter** for open‑vocabulary small‑object detection and search over very large satellite imagery.

## Contents (functional)
1) **Core pipeline**
   - `data/tiler.py` — tile gigapixel GeoTIFFs with geo sidecars
   - `ssl/pretrain_mae.py` — toy multi‑GPU MAE pretraining scaffold
   - `pdf/extract_pairs.py` — extract (image, text) pairs from PDFs
   - `vl/finetune_clip.py` — domain CLIP fine‑tune on extracted pairs

2) **Proposals & Open‑Vocab**
   - `det/proposals_rpn.py` — class‑agnostic RPN (Torchvision Faster‑RCNN) trainer
   - `det/sam_proposer.py` — SAM‑based generic proposer
   - `det/d2_open_vocab.py` — Detectron2 head (class‑agnostic) + CLIP matching
   - `det/detr_open_vocab.py` — DETR‑based proposals + CLIP matching
   - `det/open_vocab_match.py` — text/visual prototype scoring

3) **Index & Search**
   - `embeddings/tile_embedder.py` — tile‑level embeddings manifest
   - `index/tile_faiss_ivfpq.py` — IVF‑PQ tile index builder
   - `index/build_and_query.py` — proposal‑embedding FAISS example
   - `search/search_two_stage.py` — two‑stage search (tile coarse → proposal fine-ready)

4) **Evaluation & Reporting**
   - `eval/coco_map.py` — COCO mAP with JSON output
   - `eval/to_coco_preds.py` — convert detections to COCO format
   - `eval/ov_retrieval.py` — open‑vocab retrieval AP + P@K with JSON output
   - `eval/report.py` — Markdown/HTML report with quick chart
   - `tools/pilot.sh` — one‑command pilot (run → eval → report)
   - `tools/run_from_config.py` — YAML runner

5) **Dataset Utilities**
   - `tools/build_coco_from_tiles.py` — build COCO JSON from CSV/JSON labels
   - `tools/make_prompt_map.py` — derive `prompt_to_cat.json` from COCO categories

6) **Compression QA**
   - `tools/codec_qa_jxl.py` — JXL round‑trip + PSNR/SSIM/MS‑SSIM/LPIPS + Embedding drift
   - `eval/summarize_jxl_qa.py` — aggregate & recommend distance

7) **Samplers**
   - `tools/sample_tiles_for_qa.py` — sample tiles with **small objects** + hard negatives

8) **Config & Requirements**
   - `config/config.yaml` — switch between Detectron2 / DETR / SAM
   - `config/prompt_to_cat.json` — template mapping
   - `requirements.txt`

## Quickstart
```bash
# 0) install deps (CUDA PyTorch, then)
pip install -r requirements.txt

# 1) Tile a test GeoTIFF
python data/tiler.py /path/to/huge.tif /data/tiles --tile_size=1024 --stride=1024

# 2) Pretrain toy MAE (10 GPUs on one node)
torchrun --nproc_per_node=10 ssl/pretrain_mae.py --tiles_dir /data/tiles --out_dir ckpts/mae --batch_size 192 --epochs 2

# 3) Extract PDF pairs & fine‑tune CLIP
python pdf/extract_pairs.py /data/docs/domain.pdf /data/pairs/domain1
python vl/finetune_clip.py --pairs_jsonl /data/pairs/domain1/pairs.jsonl --model_name ViT-B-16 --pretrained laion2b_s34b_b88k --epochs 1

# 4) Train proposals (Detectron2) or run SAM
pip install 'git+https://github.com/facebookresearch/detectron2.git'
python det/d2_open_vocab.py train --img_dir /data/tiles --ann_file /data/anns/coco_objectness.json --out_dir ckpts/d2 --iters 1000

# 5) Run pilot via YAML
bash tools/pilot.sh config/config.yaml
```

> Note: most scripts are intentionally **minimal** so you can verify I/O & throughput first. Swap in production backbones (DINOv2/MAE), add mixed precision & data readers as needed.
