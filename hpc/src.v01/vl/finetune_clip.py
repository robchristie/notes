import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import open_clip
from tqdm import tqdm

class PairSet(Dataset):
    def __init__(self, pairs_jsonl, img_size=224):
        self.rows = [json.loads(l) for l in Path(pairs_jsonl).read_text(encoding="utf-8").splitlines()]
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        txt = r["text"] if r["text"] else "satellite image"
        return self.t(img), txt

@torch.no_grad()
def accuracy(image_features, text_features):
    sims = (image_features @ text_features.t())
    preds = sims.argmax(dim=1)
    return (preds == torch.arange(len(preds), device=preds.device)).float().mean()

def finetune(pairs_jsonl, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", batch_size=256, epochs=2, lr=1e-5, out_dir="ckpts/clip"):
    device = "cuda"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    ds = PairSet(pairs_jsonl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.02)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        for imgs, texts in tqdm(dl):
            imgs = imgs.to(device)
            toks = tokenizer(texts).to(device)
            img_f, txt_f, logit_scale = model(imgs, toks)
            logits = (img_f @ txt_f.t()) * logit_scale.exp()
            labels = torch.arange(len(imgs), device=device)
            loss = 0.5*(torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.t(), labels))
            opt.zero_grad(); loss.backward(); opt.step()
        torch.save({"epoch": ep+1, "model": model.state_dict()}, out / f"clip_ep{ep+1}.pt")

if __name__ == "__main__":
    import fire
    fire.Fire(finetune)
