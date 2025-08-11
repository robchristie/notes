import json
from pathlib import Path

def main(coco_json, out, alias=None):
    coco = json.load(open(coco_json,'r'))
    cats = coco['categories']
    name2id = {c['name']: c['id'] for c in cats}
    mapping = {}
    if alias:
        alias = json.loads(alias)
        for prompt, canonical in alias.items():
            if canonical not in name2id:
                raise SystemExit(f"Alias canonical '{canonical}' not in COCO categories: {list(name2id)}")
            mapping[prompt] = name2id[canonical]
    for n, i in name2id.items(): mapping.setdefault(n, i)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(mapping, open(out,'w'), indent=2)
    print('Wrote', out)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
