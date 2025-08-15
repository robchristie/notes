from __future__ import annotations
import hashlib, json, pathlib, time
from typing import Optional
import httpx
from pydantic import BaseModel
from ..config import settings, span

SNAP_DIR = pathlib.Path(".snapshots"); SNAP_DIR.mkdir(exist_ok=True)

class BraveSearchIn(BaseModel):
    q: str
    freshness: Optional[str] = None
    country: str = settings.brave_default_country

class BraveSearchOut(BaseModel):
    query: BraveSearchIn
    ts: float
    results: dict

async def brave_search(inp: BraveSearchIn) -> BraveSearchOut:
    key = hashlib.sha256(inp.model_dump_json().encode()).hexdigest()
    snap = SNAP_DIR / f"brave_{key}.json"
    if snap.exists():
        return BraveSearchOut(**json.loads(snap.read_text()))
    if settings.dry_run:
        return BraveSearchOut(query=inp, ts=time.time(), results={"dry_run": True})

    headers = {"X-Subscription-Token": settings.brave_api_key}
    if settings.openrouter_http_referrer:
        headers["HTTP-Referer"] = settings.openrouter_http_referrer
    if settings.openrouter_x_title:
        headers["X-Title"] = settings.openrouter_x_title
    params = {"q": inp.q, "country": inp.country}

    async with httpx.AsyncClient(timeout=30) as client, span("tool.brave_search", **params):
        r = await client.get("https://api.search.brave.com/res/v1/web/search",
                             params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
    out = BraveSearchOut(query=inp, ts=time.time(), results=data)
    snap.write_text(out.model_dump_json())
    return out
