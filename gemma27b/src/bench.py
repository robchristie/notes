#!/usr/bin/env python3
import argparse, time, os, math, glob, csv, itertools, statistics, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def load_images(glob_path: str, limit: int = None) -> List[Image.Image]:
    paths = sorted(glob.glob(glob_path))
    if not paths:
        raise FileNotFoundError(f"No images matched: {glob_path}")
    if limit:
        paths = paths[:limit]
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    return imgs

def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return ["Describe the key information in this document."]
    return lines

def set_torch_perf_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def humanize_time(sec: float) -> str:
    if sec < 1e-3: return f"{sec*1e6:.1f} µs"
    if sec < 1: return f"{sec*1e3:.2f} ms"
    return f"{sec:.2f} s"

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 pan-and-scan benchmark launcher")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--precision", type=str, choices=["bf16","fp16"], default="bf16")
    parser.add_argument("--attn_impl", type=str, choices=["flash_attention_2","sdpa"], default="flash_attention_2")
    parser.add_argument("--device_map", type=str, default="auto", help='"auto" or explicit like "cuda:0"')
    parser.add_argument("--images", type=str, required=True, help="Glob for input images (e.g., '/data/docs/*.png')")
    parser.add_argument("--prompts", type=str, default="prompts.txt", help="Path to prompts.txt")
    parser.add_argument("--num_samples", type=int, default=20, help="Total requests to run (sampled with replacement)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pan_and_scan", action="store_true", help="Disable pan-and-scan (baseline)")
    parser.add_argument("--grid_min_ratio", type=str, default="2.0,2.5", help="Comma list for pan_and_scan_min_ratio_to_activate")
    parser.add_argument("--grid_max_crops", type=str, default="4,6", help="Comma list for pan_and_scan_max_num_crops")
    parser.add_argument("--grid_min_crop_size", type=str, default="448", help="Comma list for pan_and_scan_min_crop_size")
    parser.add_argument("--save_generations", action="store_true")
    parser.add_argument("--warmup_batches", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_torch_perf_flags()

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    print(f"[INFO] Loading model: {args.model_id}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, padding_side="left")

    img_list = load_images(args.images)
    prompts = load_prompts(args.prompts)

    # Build request pool
    reqs = []
    for i in range(args.num_samples):
        img = random.choice(img_list)
        text = random.choice(prompts)
        reqs.append((text, img))

    # Grids
    if args.no_pan_and_scan:
        grids = [dict(do_pan_and_scan=False)]
    else:
        ratios = [float(x) for x in args.grid_min_ratio.split(",") if x.strip()]
        max_crops = [int(x) for x in args.grid_max_crops.split(",") if x.strip()]
        min_sizes = [int(x) for x in args.grid_min_crop_size.split(",") if x.strip()]
        grids = list(product_dict(
            do_pan_and_scan=[True],
            pan_and_scan_min_ratio_to_activate=ratios,
            pan_and_scan_max_num_crops=max_crops,
            pan_and_scan_min_crop_size=min_sizes,
        ))

    print(f"[INFO] Running {len(grids)} setting(s) × {args.num_samples} requests, batch_size={args.batch_size}")
    run_id = int(time.time())
    csv_path = out_dir / f"results_{run_id}.csv"
    fieldnames = [
        "run_id","model_id","precision","attn_impl","device_map",
        "do_pan_and_scan","min_ratio","max_crops","min_crop_size",
        "batch_size","max_new_tokens","temperature","top_p",
        "num_samples","warmup_batches",
        "latency_p50_ms","latency_p95_ms","throughput_req_s","tokens_per_s_mean"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for grid in grids:
            settings = {
                "do_pan_and_scan": grid.get("do_pan_and_scan", False),
                "pan_and_scan_min_ratio_to_activate": grid.get("pan_and_scan_min_ratio_to_activate"),
                "pan_and_scan_max_num_crops": grid.get("pan_and_scan_max_num_crops"),
                "pan_and_scan_min_crop_size": grid.get("pan_and_scan_min_crop_size"),
                "add_generation_prompt": True,
                "return_tensors": "pt",
            }

            # Warmup
            print(f"\n[INFO] Warmup for settings: {grid}")
            for _ in range(args.warmup_batches):
                batch = reqs[:args.batch_size]
                inputs = processor(
                    text=[t for (t,_) in batch],
                    images=[im for (_,im) in batch],
                    **settings
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
                with torch.inference_mode():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        temperature=0.0,
                        do_sample=False,
                        use_cache=True,
                    )
                torch.cuda.synchronize()

            # Timed run
            latencies = []
            tok_speeds = []
            start_all = time.perf_counter()

            # Process in batches
            for i in range(0, len(reqs), args.batch_size):
                batch = reqs[i:i+args.batch_size]
                inputs = processor(
                    text=[t for (t,_) in batch],
                    images=[im for (_,im) in batch],
                    **settings
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=(args.temperature > 0),
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                latency = t1 - t0
                latencies.append(latency)

                # crude tokens/sec: avg generated length / latency
                # (inputs['input_ids'] may not equal prompt tokens due to image tokens; this is an approximation)
                try:
                    gen_tokens = out.shape[1]
                    tok_speeds.append(gen_tokens / latency)
                except Exception:
                    pass

                if args.save_generations:
                    dec = processor.batch_decode(out, skip_special_tokens=True)
                    for j, txt in enumerate(dec):
                        with open(out_dir / f"gen_{run_id}_{i+j}.txt", "w", encoding="utf-8") as fo:
                            fo.write(txt)

            total_time = time.perf_counter() - start_all
            throughput = len(reqs) / total_time if total_time > 0 else float("nan")
            p50 = statistics.median(latencies) * 1000.0
            p95 = statistics.quantiles(latencies, n=100)[94] * 1000.0 if len(latencies) >= 20 else max(latencies)*1000.0
            tps_mean = statistics.mean(tok_speeds) if tok_speeds else float("nan")

            print(f"[RESULT] do_pan_and_scan={settings['do_pan_and_scan']} "
                  f"min_ratio={settings.get('pan_and_scan_min_ratio_to_activate')} "
                  f"max_crops={settings.get('pan_and_scan_max_num_crops')} "
                  f"min_crop_size={settings.get('pan_and_scan_min_crop_size')} | "
                  f"p50={p50:.1f} ms p95={p95:.1f} ms | throughput={throughput:.2f} req/s | tokens/s≈{tps_mean:.1f}")

            row = {
                "run_id": run_id,
                "model_id": args.model_id,
                "precision": args.precision,
                "attn_impl": args.attn_impl,
                "device_map": args.device_map,
                "do_pan_and_scan": settings['do_pan_and_scan'],
                "min_ratio": settings.get("pan_and_scan_min_ratio_to_activate"),
                "max_crops": settings.get("pan_and_scan_max_num_crops"),
                "min_crop_size": settings.get("pan_and_scan_min_crop_size"),
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_samples": args.num_samples,
                "warmup_batches": args.warmup_batches,
                "latency_p50_ms": f"{p50:.2f}",
                "latency_p95_ms": f"{p95:.2f}",
                "throughput_req_s": f"{throughput:.4f}",
                "tokens_per_s_mean": f"{tps_mean:.2f}",
            }
            writer.writerow(row)

    print(f"\n[DONE] Wrote CSV: {csv_path}")

if __name__ == "__main__":
    main()
