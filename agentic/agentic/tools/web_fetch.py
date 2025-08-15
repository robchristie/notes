from __future__ import annotations
import hashlib, json, pathlib, time, urllib.parse, urllib.robotparser
import httpx
from pydantic import BaseModel
from ..config import span, settings

SNAP_DIR = pathlib.Path(".snapshots"); SNAP_DIR.mkdir(exist_ok=True)

class WebFetchIn(BaseModel):
    url: str

class WebFetchOut(BaseModel):
    url: str
    ts: float
    status: int
    headers: dict
    text: str

async def web_fetch(inp: WebFetchIn) -> WebFetchOut:
    key = hashlib.sha256(inp.url.encode()).hexdigest()
    snap = SNAP_DIR / f"web_{key}.json"
    if snap.exists():
        return WebFetchOut(**json.loads(snap.read_text()))
    if settings.dry_run:
        return WebFetchOut(url=inp.url, ts=time.time(), status=200, headers={}, text="(dry_run)")

    parsed = urllib.parse.urlparse(inp.url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser(robots_url)
    try:
        rp.read()
        if not rp.can_fetch("agentic-starter", inp.url):
            raise PermissionError(f"Robots disallow: {inp.url}")
    except Exception:
        pass

    async with httpx.AsyncClient(timeout=30) as client, span("tool.web_fetch", url=inp.url):
        r = await client.get(inp.url)
        r.raise_for_status()
        out = WebFetchOut(
            url=inp.url, ts=time.time(), status=r.status_code, headers=dict(r.headers), text=r.text
        )
    snap.write_text(out.model_dump_json())
    return out
