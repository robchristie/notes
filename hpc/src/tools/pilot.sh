#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-config/config.yaml}
RUN_DIR=${2:-runs/pilot}
GT_JSON=${3:-/data/anns/coco_objectness.json}
PROMPT_MAP=${4:-config/prompt_to_cat.json}

python tools/run_from_config.py --cfg_path "$CFG"
python eval/to_coco_preds.py "$RUN_DIR/detections_open_vocab.json" "$RUN_DIR/coco_preds.json"
python eval/coco_map.py "$GT_JSON" "$RUN_DIR/coco_preds.json" --out_json "$RUN_DIR/coco_stats.json"
python eval/ov_retrieval.py "$GT_JSON" "$RUN_DIR/detections_open_vocab.json" "$PROMPT_MAP" --out_json "$RUN_DIR/ov_stats.json"
python eval/report.py --run_dir "$RUN_DIR" --run_name pilot_single_node --proposer detectron2 --prompts_csv 'ship,airplane,helicopter'
echo "Report: $RUN_DIR/report.md  |  $RUN_DIR/report.html"
