import fitz
from pathlib import Path
import re, json, io
from PIL import Image
from tqdm import tqdm

def extract(pdf_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("blocks")
        imgs = page.get_images(full=True)
        for i, img in enumerate(imgs):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_name = f"p{pno:03d}_img{i:02d}.png"
            pil.save(out / img_name)
            ibox = page.get_image_bbox(xref)
            caption = ""
            best_dy = 1e9
            for (x0,y0,x1,y1,txt,*rest) in blocks:
                if txt.strip() == "": continue
                if y0 >= ibox.y1 and (y0 - ibox.y1) < best_dy:
                    caption = txt.strip(); best_dy = y0 - ibox.y1
            if caption == "":
                for (_,_,_,_,txt,*_) in blocks:
                    if re.search(r"(Figure|Fig\.|Table)\s*\d+", txt):
                        caption = txt.strip(); break
            pairs.append({"image": str(out / img_name), "text": caption})
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in pairs: f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import fire
    fire.Fire(extract)
