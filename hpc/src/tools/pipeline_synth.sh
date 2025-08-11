#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 6 ]; then
  echo "Usage: $0 <CLASS> <PROMPT> <BG_DIR> <OUT_ROOT> <SD_INPAINT_MODEL> <CLIP_CKPT> [NUM=20000]"; exit 1
fi
CLASS=$1; PROMPT=$2; BG_DIR=$3; OUT_ROOT=$4; SD_MODEL=$5; CLIP_CKPT=$6; NUM=${7:-20000}
OUT_SYNTH="${OUT_ROOT}/${CLASS}_inpaint"; OUT_SENSOR="${OUT_SYNTH}/sensorized"; OUT_FILTER="${OUT_SYNTH}/filtered"; COCO_DIR="${OUT_SYNTH}/coco"
mkdir -p "$OUT_SYNTH" "$OUT_SENSOR" "$OUT_FILTER" "$COCO_DIR"
python tools/gen_inpaint.py --bg_dir "$BG_DIR" --out_dir "$OUT_SYNTH" --prompt "$PROMPT" --model "$SD_MODEL" --num "$NUM" --mask_size_px 24,72 --strength 0.85
python tools/sensor_model.py --in_dir "$OUT_SYNTH/imgs" --out_dir "$OUT_SENSOR" --src_gsd_cm 25 --tgt_gsd_cm 50 --mtf_sigma_px 0.7 --anisotropy 1.0 --read_noise 0.002 --shot_gain 0.02 --bits 12
python tools/synth_qc.py --imgs "$OUT_SENSOR" --meta "$OUT_SYNTH/meta.jsonl" --out "$OUT_FILTER" --prompt "$PROMPT" --clip_ckpt "$CLIP_CKPT" --min_clip 0.28 --min_contrast 0.05 --min_box_px 12 --dedup_thresh 0.04
python tools/synth_to_coco.py --imgs_dir "$OUT_FILTER/imgs" --meta "$OUT_FILTER/meta.jsonl" --class_name "$CLASS" --out_json "$COCO_DIR/instances_${CLASS}.json"
echo "Done. Synth images: $OUT_FILTER/imgs  |  COCO: $COCO_DIR/instances_${CLASS}.json"
