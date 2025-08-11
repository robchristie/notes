import faiss, json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import numpy as np

def load_manifest(manifest_path):
    rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
    vecs = []
    for r in rows:
        v = np.load(r['embed_path']).astype('float32')
        vecs.append(v)
    return rows, np.vstack(vecs)

def build_ivfpq(vecs, nlist=4096, m=16, nbits=8):
    d = vecs.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    rs = np.random.RandomState(123)
    train = vecs[rs.choice(len(vecs), size=min(200000, len(vecs)), replace=False)]
    index.train(train); index.add(vecs)
    return index

def main(manifest, out_dir='runs/pilot/tile_index', nlist=4096, m=16, nbits=8, gpu=True):
    rows, vecs = load_manifest(manifest)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    index = build_ivfpq(vecs, nlist=nlist, m=m, nbits=nbits)
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    cpu = faiss.index_gpu_to_cpu(index) if gpu else index
    faiss.write_index(cpu, str(Path(out_dir)/'tiles.ivfpq'))
    idmap = [{'tile_id': i, 'tile_path': rows[i]['tile_path']} for i in range(len(rows))]
    Path(out_dir, 'idmap.json').write_text(json.dumps(idmap))
    print('Index written to', out_dir)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
