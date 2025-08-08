# Gemma 3 Pan-and-Scan Benchmark Launcher

This is a tiny, no-frills launcher to sweep **pan-and-scan** settings on Gemma 3 (27B) using 🤗 Transformers.
It measures per-batch latency percentiles, rough tokens/sec, and overall throughput (req/s).

> **Hardware**: tuned for A100 40GB. Use bf16 + FlashAttention-2 and shard with `device_map=auto` (2 GPUs is a sweet spot).

## Install

```bash
# Python 3.10+ recommended
pip install -r requirements.txt

# (Optional) FlashAttention-2:
# pip install flash-attn --no-build-isolation
```

You also need HF auth if the model is gated:
```bash
huggingface-cli login
```

## Run (single process, 2×GPU shard)

Prepare a folder of test images (PDF pages as PNG/JPG, screenshots of long docs, receipts, infographics).

```bash
python bench.py \
  --model_id google/gemma-3-27b-it \
  --precision bf16 \
  --attn_impl flash_attention_2 \
  --device_map auto \
  --images "/data/docs/*.png" \
  --prompts prompts.txt \
  --num_samples 40 \
  --batch_size 2 \
  --max_new_tokens 256 \
  --grid_min_ratio 2.0,2.5 \
  --grid_max_crops 4,6 \
  --grid_min_crop_size 448 \
  --output_dir runs
```

**Baseline (no pan-and-scan)**:
```bash
python bench.py --images "/data/docs/*.png" --no_pan_and_scan
```

Outputs a CSV in `runs/` with results per setting row.

## Max throughput on 10×A100 (recommended pattern)

Instead of sharding one model across all 10 GPUs, spin up **five replicas** of the script, each using 2 GPUs via CUDA visibility.

Example (bash pseudo-code):
```bash
# Replica 0 on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 python bench.py --images "/data/docs/*.png" --output_dir runs/rep0 &
# Replica 1 on GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 python bench.py --images "/data/docs/*.png" --output_dir runs/rep1 &
# ...
wait
```

If you want a single front-end, put these replicas behind an HTTP router (e.g., nginx + a trivial FastAPI wrapper) or a simple queue,
but for **pure benchmarking** just run the launcher N times and compare CSVs.

## Tips

- Keep `batch_size` small (1–2) for lower p95 latency; increase for raw throughput.
- Use `bf16` on A100. 8-bit/4-bit can free VRAM but won’t usually be faster.
- Pan-and-scan adds compute proportional to crop count—cap `--grid_max_crops` at 4–6 for speed-sensitive use.
- For doc-heavy sets, lower `--grid_min_ratio` to trigger pan-and-scan more often (quality↑, speed↓).

## CSV Columns

- `latency_p50_ms`, `latency_p95_ms`: Per-batch latency (includes preproc + generate).
- `throughput_req_s`: Completed requests per second over the whole run.
- `tokens_per_s_mean`: Crude generated tokens per second (approx).

## Known caveats

- Tokens/sec is approximate because image tokens complicate input length accounting.
- For the best FA2 speedups, ensure `flash-attn` is installed for your CUDA/PyTorch version.
- If you see memory issues, reduce `batch_size` or shard across 3–4 GPUs via `device_map=auto`.
