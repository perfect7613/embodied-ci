"""The HTTP API the web studio talks to. One app, two hosts:
Modal (infra/modal_app.py, runs spawn a Modal function) and a self-hosted GPU
box (infra/local_server.py, runs start a thread). Everything it serves is read
from the data root: run logs (JSONL), episode media, candidates, registry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Callable

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel


class HFReq(BaseModel):
    repo_id: str
    token: str  # used for this one push; never stored or logged
    private: bool = True


class RunReq(BaseModel):
    model: str = "vjepa2-ac-vitg"
    task: str = "lift"
    rounds: int = 2
    use_agents: bool = True
    reset: bool = False


def make_app(root: str, start: Callable[[str, int, bool, bool], str], fresh: Callable[..., None],
             models: list[dict], tasks: list[dict], backend: str, commit: Callable[[], None] = lambda: None) -> FastAPI:
    """Runs are private: a browser sends a random id (x-owner); only its hash is stored, next to the run log.
    A run is visible to its owner, or to everyone if listed in {root}/public.json (the demo runs)."""
    web = FastAPI(title="EmbodiedCI API")
    web.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def read_events(run_id: str, after: int = 0) -> list[dict]:
        p = f"{root}/runs/{run_id}.jsonl"
        if "/" in run_id or not os.path.exists(p):
            return []
        with open(p) as f:
            lines = f.readlines()
        return [dict(json.loads(l), seq=i) for i, l in enumerate(lines) if i >= after]

    def owner_hash(owner: str | None) -> str | None:
        return hashlib.sha256(owner.encode()).hexdigest()[:32] if owner and len(owner) >= 16 else None

    def public() -> set[str]:
        p = f"{root}/public.json"
        return set(json.load(open(p))) if os.path.exists(p) else set()

    def visible(run_id: str, h: str | None) -> bool:
        if run_id in public():
            return True
        p = f"{root}/runs/{run_id}.owner"
        return bool(h) and "/" not in run_id and os.path.exists(p) and open(p).read().strip() == h

    @web.get("/health")
    def health():
        return {"ok": True, "backend": backend}

    @web.get("/models")
    def get_models():
        return models

    @web.get("/tasks")
    def get_tasks():
        return tasks

    @web.post("/runs")
    def run(req: RunReq, x_owner: str | None = Header(None)):
        h = owner_hash(x_owner)
        if not h:
            raise HTTPException(400, "missing x-owner")
        if req.model not in {m["id"] for m in models} or req.task not in {t["id"] for t in tasks}:
            raise HTTPException(400, "unknown model or task")
        run_id = time.strftime("run-%Y%m%d-%H%M%S")
        os.makedirs(f"{root}/runs", exist_ok=True)
        with open(f"{root}/runs/{run_id}.owner", "w") as f:
            f.write(h)
        commit()
        return {"run_id": run_id, "call_id": start(run_id, req.rounds, req.use_agents, req.reset)}

    @web.get("/runs")
    def runs(x_owner: str | None = Header(None)):
        fresh()
        h = owner_hash(x_owner)
        d = f"{root}/runs"
        files = sorted(os.listdir(d), reverse=True) if os.path.exists(d) else []
        return [{"run_id": f[:-6], "updated": os.path.getmtime(f"{d}/{f}")} for f in files
                if f.endswith(".jsonl") and visible(f[:-6], h)]

    @web.get("/runs/{run_id}/events")
    def events(run_id: str, after: int = 0, x_owner: str | None = Header(None)):
        fresh()
        if not visible(run_id, owner_hash(x_owner)):
            raise HTTPException(404)
        return read_events(run_id, after)

    @web.get("/runs/{run_id}/stream")
    async def stream(run_id: str, after: int = 0, owner: str | None = None, x_owner: str | None = Header(None)):
        if not visible(run_id, owner_hash(x_owner or owner)):  # EventSource can't set headers: ?owner= too
            raise HTTPException(404)

        async def gen():
            seq = after
            while True:
                fresh()
                for e in read_events(run_id, seq):
                    seq = e["seq"] + 1
                    yield f"data: {json.dumps(e)}\n\n"
                    if e["kind"] == "run_end":
                        return
                await asyncio.sleep(2)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @web.get("/episodes/{eid}.{ext}")
    def episode_media(eid: str, ext: str):
        if ext not in ("mp4", "png") or "/" in eid:
            raise HTTPException(404)
        p = f"{root}/episodes/{eid}.{ext}"
        if not os.path.exists(p):
            fresh(0.5)
        if not os.path.exists(p):
            raise HTTPException(404)
        return FileResponse(p, headers={"cache-control": "public, max-age=86400, immutable"})

    @web.get("/candidates/{cid}")
    def candidate(cid: str):
        from embodied.candidate import Candidate

        fresh()
        if "/" in cid:
            raise HTTPException(404)
        try:
            c = Candidate.load(root, cid)
        except FileNotFoundError:
            raise HTTPException(404)
        return {"id": c.id, "parent": c.parent, "note": c.note, "guardrails": c.guardrails,
                "head": {k: list(v.shape) for k, v in c.head.items()}}

    @web.get("/lineage")
    def get_lineage():
        from infra.export import lineage

        fresh()
        return lineage(root)

    @web.get("/compare/{a}/{b}")
    def get_compare(a: str, b: str, suite: str = "heldout"):
        from infra.export import compare

        if "/" in a or "/" in b or suite not in ("heldout", "nominal", "regression"):
            raise HTTPException(400)
        fresh()
        return compare(root, a, b, suite)

    @web.get("/candidates/{cid}/bundle.zip")
    def download(cid: str, base: str | None = None):
        from infra.export import zip_bundle

        if "/" in cid or (base and "/" in base):
            raise HTTPException(404)
        fresh()
        try:
            path = zip_bundle(root, cid, base)
        except FileNotFoundError:
            raise HTTPException(404)
        return FileResponse(path, filename=f"embodiedci-{cid}.zip", media_type="application/zip")

    @web.post("/candidates/{cid}/export/hf")
    def export_hf(cid: str, req: HFReq, base: str | None = None):
        import re

        from infra.export import push_hf

        if "/" in cid or not re.fullmatch(r"[\w.-]+/[\w.-]+", req.repo_id):
            raise HTTPException(400, "repo_id must look like user/name")
        fresh()
        try:
            url = push_hf(root, cid, req.repo_id, req.token, req.private, base)
        except FileNotFoundError:
            raise HTTPException(404)
        except Exception as e:  # surface HF errors (bad token, no write access) without echoing the token
            raise HTTPException(502, f"Hugging Face push failed: {type(e).__name__}: {str(e).replace(req.token, '***')[:300]}")
        return {"url": url}

    @web.get("/registry")
    def registry():
        fresh()
        p = f"{root}/registry.json"
        return json.load(open(p)) if os.path.exists(p) else []

    @web.get("/champion")
    def champion():
        fresh()
        p = f"{root}/champion.json"
        return json.load(open(p)) if os.path.exists(p) else None

    return web
