import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from timm.models.vision_transformer import vit_base_patch16_224

class TileDataset(Dataset):
    def __init__(self, tiles_dir, size=224):
        self.paths = [p for p in Path(tiles_dir).glob("*.tif")]
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.t(img)

class MAE(pl.LightningModule):
    def __init__(self, lr=1e-4, mask_ratio=0.6):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = vit_base_patch16_224()
        dim = self.encoder.head.in_features
        self.encoder.head = nn.Identity()
        self.decoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3*224*224))
    def forward(self, x):
        B,C,H,W = x.shape
        mask = (torch.rand(B,1,1,1, device=x.device) > self.hparams.mask_ratio).float()
        x_masked = x * mask
        z = self.encoder(x_masked)
        rec = self.decoder(z).view(B,3,224,224)
        return rec
    def training_step(self, batch, _):
        x = batch
        rec = self(x)
        loss = torch.mean((x - rec)**2)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_steps if self.trainer.max_steps else 1000)
        return {"optimizer": opt, "lr_scheduler": sched}

def main(tiles_dir, out_dir="ckpts/mae", batch_size=192, epochs=2, num_workers=8):
    ds = TileDataset(tiles_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    model = MAE()
    from pathlib import Path
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=out, save_last=True, save_top_k=1, monitor="train/loss", mode="min")
    lrmon = LearningRateMonitor()
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", devices=-1,
                         strategy=DDPStrategy(find_unused_parameters=True),
                         precision="bf16-mixed", callbacks=[ckpt, lrmon])
    trainer.fit(model, dl)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
