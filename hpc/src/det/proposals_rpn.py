import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path

class CocoLike(Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = Path(img_dir)
        self.ann = json.loads(Path(ann_file).read_text())
        self.id2img = {im['id']: im for im in self.ann['images']}
        self.img2anns = {}
        for ann in self.ann['annotations']:
            self.img2anns.setdefault(ann['image_id'], []).append(ann)
        self.ids = list(self.id2img.keys())
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        im = self.id2img[self.ids[i]]
        img = Image.open(self.img_dir / im['file_name']).convert('RGB')
        boxes = []
        for a in self.img2anns.get(im['id'], []):
            x,y,bw,bh = a['bbox']
            boxes.append([x,y,x+bw,y+bh])
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.ones((len(boxes),), dtype=torch.int64)}
        return torchvision.transforms.functional.to_tensor(img), target

class ClassAgnosticRCNN(FasterRCNN):
    def __init__(self):
        backbone = torchvision.models.resnet50(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        anchor = AnchorGenerator(sizes=((16, 32, 64, 128),), aspect_ratios=((0.5,1.0,2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        super().__init__(backbone, num_classes=2, rpn_anchor_generator=anchor, box_roi_pool=roi_pooler)
        self.roi_heads.box_predictor.cls_score = nn.Linear(1024, 2)

def train(img_dir, ann_file, out_path='ckpts/rpn.pt', epochs=10, bs=4):
    ds = CocoLike(img_dir, ann_file)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))
    model = ClassAgnosticRCNN().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for images, targets in dl:
            images = [im.cuda() for im in images]
            targets = [{k:v.cuda() for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {ep+1}, loss {loss.item():.4f}")
    torch.save(model.state_dict(), out_path)

if __name__ == "__main__":
    import fire
    fire.Fire(train)
