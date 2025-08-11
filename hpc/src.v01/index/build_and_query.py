import faiss, torch, json
from det.open_vocab_match import OVMatcher, crop_boxes

class EmbeddingIndexer:
    def __init__(self, clip_ckpt, dim=512, gpu=True):
        self.matcher = OVMatcher(clip_ckpt=clip_ckpt)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        if gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        self.meta = []
    def add_tile(self, tile_path, boxes):
        crops = crop_boxes(tile_path, boxes)
        feats = self.matcher.embed_images(crops).cpu().numpy()
        self.index.add(feats)
        self.meta.extend([(tile_path, tuple(b)) for b in boxes])
    def query_text(self, prompt, k=20):
        q = self.matcher.embed_text([prompt]).cpu().numpy()
        D, I = self.index.search(q, k)
        return [(float(D[0,i]), self.meta[I[0,i]]) for i in range(min(k, len(I[0])))]
