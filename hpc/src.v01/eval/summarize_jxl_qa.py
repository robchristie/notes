import json
from pathlib import Path
import numpy as np

def summarize(rows):
    by_d = {}
    for r in rows:
        d = r['distance']
        by_d.setdefault(d, {'PSNR':[], 'SSIM':[], 'MS_SSIM':[], 'LPIPS':[], 'EMB_COS':[]})
        for k in by_d[d].keys(): by_d[d][k].append(r[k])
    table = []
    for d, m in sorted(by_d.items()):
        row = {'distance': d}
        for k, vals in m.items():
            row[k+'_mean'] = float(np.mean(vals)); row[k+'_p05'] = float(np.percentile(vals, 5))
            row[k+'_p50']  = float(np.percentile(vals, 50)); row[k+'_p95'] = float(np.percentile(vals, 95))
        table.append(row)
    return table

def recommend(table, ssim_min=0.98, emb_cos_min=0.995):
    good = [t for t in table if t['SSIM_p50'] >= ssim_min and t['EMB_COS_p50'] >= emb_cos_min]
    if not good: return None
    return sorted(good, key=lambda x: x['distance'])[-1]

def main(in_json='runs/pilot/jxl_qa.json', out_json='runs/pilot/jxl_qa_summary.json', ssim_min=0.98, emb_cos_min=0.995):
    rows = json.load(open(in_json,'r'))
    table = summarize(rows); rec = recommend(table, ssim_min, emb_cos_min)
    out = {'summary': table, 'recommendation': rec}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json,'w'), indent=2)
    print('Wrote', out_json)
    if rec: print('Recommended distance:', rec['distance'])
    else: print('No setting met thresholds; consider relaxing.')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
