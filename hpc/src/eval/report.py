import json, argparse
from pathlib import Path
import matplotlib.pyplot as plt

MD_TMPL = """# Open‑Vocabulary Detection Report

**Run:** {run_name}  
**Proposer:** {proposer}  
**Prompts:** {prompts}  

## COCO mAP

| Metric | Value |
|---|---:|
| AP | {AP:.3f} |
| AP50 | {AP50:.3f} |
| AP75 | {AP75:.3f} |
| AP_small | {AP_small:.3f} |
| AP_medium | {AP_medium:.3f} |
| AP_large | {AP_large:.3f} |

## Open‑Vocab Retrieval

| Prompt | AP | P@1 | P@5 | P@10 |
|---|---:|---:|---:|---:|
{ov_rows}
"""

def render_markdown(run_name, proposer, prompts, coco_stats, ov_stats, out_md):
    rows = []
    for p in prompts:
        s = ov_stats.get(p, { 'AP':0, 'P@1':0, 'P@5':0, 'P@10':0 })
        rows.append(f"| {p} | {s.get('AP',0):.3f} | {s.get('P@1',0):.3f} | {s.get('P@5',0):.3f} | {s.get('P@10',0):.3f} |")
    md = MD_TMPL.format(run_name=run_name, proposer=proposer, prompts=', '.join(prompts), ov_rows='\n'.join(rows), **coco_stats)
    Path(out_md).write_text(md)
    return md

def simple_bar_chart(ov_stats, out_png):
    prompts = list(ov_stats.keys())
    ap = [ov_stats[p]['AP'] for p in prompts]
    plt.figure(figsize=(8,4)); plt.bar(prompts, ap); plt.ylabel('AP'); plt.title('Open‑Vocab AP by Prompt'); plt.tight_layout(); plt.savefig(out_png)

def main(run_dir, run_name, proposer, prompts_csv):
    run = Path(run_dir)
    coco_stats = json.loads((run/'coco_stats.json').read_text()) if (run/'coco_stats.json').exists() else {'AP':0,'AP50':0,'AP75':0,'AP_small':0,'AP_medium':0,'AP_large':0}
    ov_stats = json.loads((run/'ov_stats.json').read_text()) if (run/'ov_stats.json').exists() else {}
    prompts = [p.strip() for p in prompts_csv.split(',') if p.strip()]
    md = render_markdown(run_name, proposer, prompts, coco_stats, ov_stats, run/'report.md')
    if ov_stats: simple_bar_chart(ov_stats, run/'ov_ap.png')
    html = f"""<html><head><meta charset='utf-8'><title>{run_name} Report</title></head><body><pre>{md}</pre>{'<img src=\'ov_ap.png\' />' if ov_stats else ''}</body></html>"""
    (run/'report.html').write_text(html)
    print('Wrote', run/'report.md', 'and', run/'report.html')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
