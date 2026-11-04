"""Modal app. GPU policy: H100 only.

The JEPA model lives here because fixing it needs its weights as tensors.
Sim runs in the same container as the model (robosuite on CPU, EGL render),
so a rollout never ships frames across the network.

    modal run infra/modal_app.py::smoke              # model + sim check
    modal run infra/modal_app.py::run --rounds 3      # full CI loop
    modal deploy infra/modal_app.py                   # API for the web app
"""

from __future__ import annotations

import os

import modal

app = modal.App("embodied-ci")
vol = modal.Volume.from_name("embodied-ci-data", create_if_missing=True)
baseten = modal.Secret.from_dotenv(__file__.rsplit("/infra/", 1)[0] + "/.env")  # local .env, never committed

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang", "linux-libc-dev", "libgl1", "libegl1", "libgles2", "libglib2.0-0", "libosmesa6", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "timm", "einops", "scipy", "numpy<2", "tqdm", "pyyaml",
        "mujoco==3.2.6", "robosuite==1.5.1", "imageio[ffmpeg]", "openai",
    )
    .run_commands("git clone --depth 1 https://github.com/facebookresearch/vjepa2 /opt/vjepa2")
    .pip_install("fastapi[standard]", "python-dotenv", "huggingface_hub>=0.26", "transformers>=4.57,<6", "pillow")  # transformers: Meta SAM 3 labels
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "TORCH_HOME": "/data/torch", "VJEPA_REPO": "/opt/vjepa2"})
    .add_local_python_source("embodied", "harness", "infra")
)

GPU = dict(image=image, gpu="H100", volumes={"/data": vol}, timeout=3600)


@app.function(**GPU)
def smoke() -> dict:
    """Model + sim check: expert success per drift; does JEPA surprise rank the true action below wrong ones?"""
    import time

    import imageio
    import numpy as np
    import torch

    from embodied.sim import DRIFTS, LiftTask, scripted_expert
    from embodied.world_model import WorldModel

    t0 = time.time()
    wm = WorldModel.load()
    out = {"load_s": round(time.time() - t0, 1)}
    task = LiftTask()
    out["expert_success"] = {}
    for drift in DRIFTS:
        ok = 0
        imageio.imwrite(f"/data/drift_{drift}.png", np.concatenate(list(task.reset(0, drift).views), 1))
        for seed in range(6):
            o = task.reset(seed, drift)
            while not task.done:
                o = task.step(scripted_expert(o))
            ok += task.success
        out["expert_success"][drift] = f"{ok}/6"

    # surprise ranking on an expert trajectory: true action vs 6 perturbed actions, per step
    o = task.reset(0)
    frames, poses, acts = [o.frame], [o.pose], []
    while not task.done:
        a = scripted_expert(o)
        o = task.step(a)
        frames.append(o.frame), poses.append(o.pose), acts.append(a)
    imageio.mimwrite("/data/smoke_expert.mp4", frames, fps=10, macro_block_size=8)
    z = wm.encode(np.stack(frames))
    ranks, gaps = [], []
    rng = np.random.default_rng(0)
    for t in range(0, len(acts), 2):
        cands = [acts[t]] + [acts[t] + np.r_[rng.normal(0, 0.03, 3), 0, 0, 0, 0].astype(np.float32) for _ in range(6)]
        A = torch.from_numpy(np.stack(cands)).cuda().view(7, 1, 7)
        P = torch.from_numpy(poses[t]).cuda().view(1, 1, 7).repeat(7, 1, 1)
        zp = wm.predict(z[t].view(1, 1, *z.shape[1:]).repeat(7, 1, 1, 1), A, P)
        r = wm.residual(zp, z[t + 1].unsqueeze(0)).mean(-1).cpu().numpy()
        ranks.append(int((r < r[0]).sum()))
        gaps.append(float(r[1:].mean() - r[0]))
    out["true_action_rank_mean(0=best,6=worst)"] = round(float(np.mean(ranks)), 2)
    out["true_action_best_frac"] = round(float(np.mean(np.array(ranks) == 0)), 2)
    out["mean_residual_gap"] = round(float(np.mean(gaps)), 4)
    out["episode_len"] = len(acts)
    vol.commit()
    return out


# ---------------------------------------------------------------------------
# Engine: the JEPA model + sim on H100. Rollouts and fine-tuning run here.
# ---------------------------------------------------------------------------
# One H100 by default. GPU-seconds of work are the same at any parallelism; each extra container adds a
# model load (~30 s) and an idle tail (scaledown_window), so 1 is cheapest and 8 is fastest wall-clock.
ENGINE_GPUS = int(os.environ.get("EMBODIED_GPUS", "1"))


@app.cls(**GPU, memory=32768, max_containers=ENGINE_GPUS, scaledown_window=60, secrets=[baseten])  # .env: HF_TOKEN for gated SAM 3
class Engine:
    @modal.enter()
    def load(self) -> None:
        from embodied.rollout import Runner
        from embodied.world_model import WorldModel

        self.runner = Runner(WorldModel.load())

    @modal.method()
    def run(self, cand, specs: list) -> list[dict]:
        out = self.runner.run(cand, specs)
        vol.commit()
        return out

    @modal.method()
    def fit_probes(self, seeds: list[int], drifts: list[str], label_source: str = "sim", readout: int = 0) -> tuple[dict, dict]:
        from embodied.probes import fit_probes

        return fit_probes(self.runner.wm, self.runner.task, seeds, drifts, label_source=label_source, readout=readout)

    @modal.method()
    def eval_probes(self, probes: dict, seeds: list[int], drifts: list[str]) -> dict:
        from embodied.probes import eval_probes

        return eval_probes(self.runner.wm, self.runner.task, probes, seeds, drifts)


class RemoteEngine:
    """What the harness sees: parallel rollouts across H100 containers + Volume reads."""

    CHUNK = 6

    def __init__(self):
        self.e = Engine()

    def run(self, cand, specs: list) -> list[dict]:
        chunks = [specs[i : i + self.CHUNK] for i in range(0, len(specs), self.CHUNK)]
        out = [r for res in self.e.run.starmap([(cand, c) for c in chunks]) for r in res]
        vol.reload()
        return out

    def fit_probes(self, seeds, drifts, label_source="sim", readout=0):
        return self.e.fit_probes.remote(seeds, drifts, label_source, readout)

    def eval_probes(self, probes, seeds, drifts):
        calls = [self.e.eval_probes.spawn(probes, seeds, [d]) for d in drifts]  # one drift per container
        return {k: v for c in calls for k, v in c.get().items()}

    def episode(self, eid: str) -> dict:
        import numpy as np

        with np.load(f"/data/episodes/{eid}.npz") as d:
            return {k: d[k] for k in d.files}

    def frame(self, eid: str):
        import imageio.v3 as iio

        return iio.imread(f"/data/episodes/{eid}.png")


# ---------------------------------------------------------------------------
# CI loop: the harness runs server-side on CPU; tools call the H100 Engine.
# ---------------------------------------------------------------------------
@app.function(image=image, cpu=2, memory=8192, secrets=[baseten], volumes={"/data": vol}, timeout=6 * 3600)
def ci(rounds: int = 3, run_id: str | None = None, use_agents: bool = True, reset: bool = False, labels: str = "sim",
       first_round: int = 1, lean: bool = False) -> list[dict]:
    import os
    import shutil

    from harness.ci import Harness

    if reset:
        for p in ("champion.json", "registry.json", "train_ids.json", "v0.json"):
            if os.path.exists(f"/data/{p}"):
                os.remove(f"/data/{p}")
        shutil.rmtree("/data/candidates", ignore_errors=True)
    h = Harness(RemoteEngine(), "/data", run_id=run_id, commit=vol.commit, label_source=labels, lean=lean)
    h.log("run_start", "", rounds=rounds, model="vjepa2-ac-vitg", task="lift", use_agents=use_agents)
    if rounds == 0:  # bootstrap only: probes + controller v0, scored, then stop
        h.bootstrap()
    out = [h.round(k, use_agents=use_agents) for k in range(first_round, first_round + rounds)]  # k picks fresh raw seeds
    h.log("run_end", "", rounds=out)
    vol.commit()
    return out


# ---------------------------------------------------------------------------
# API for the web app. Reads the run log + artifacts from the Volume.
# ---------------------------------------------------------------------------
from infra.catalog import MODELS, TASKS  # noqa: E402


@app.function(image=image, volumes={"/data": vol}, secrets=[baseten])
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    import threading
    import time

    from infra.api import make_app

    lock, last = threading.Lock(), [0.0]

    def fresh(max_age: float = 3.0) -> None:
        """Throttled Volume reload: many parallel media requests must not reload at once."""
        with lock:
            if time.time() - last[0] > max_age:
                try:
                    vol.reload()
                except Exception:
                    pass
                last[0] = time.time()

    def start(run_id: str, rounds: int, use_agents: bool, reset: bool) -> str:
        return ci.spawn(rounds=rounds, run_id=run_id, use_agents=use_agents, reset=reset).object_id

    return make_app("/data", start, fresh, MODELS, TASKS, backend="modal", commit=vol.commit)


@app.local_entrypoint()
def check():
    import json

    print(json.dumps(smoke.remote(), indent=2))


@app.local_entrypoint()
def run(rounds: int = 3, use_agents: bool = True, reset: bool = False, labels: str = "sim", first_round: int = 1, lean: bool = False):
    """labels: "sim" (simulator segmentation), "sam" (Meta SAM 3 weights on our GPU) or "sam3.1" (Meta SAM 3.1 via the
    Meta Model API, capped by META_API_BUDGET_USD). SAM labels come from raw frames, as on a real robot."""
    import json

    print(json.dumps(ci.remote(rounds=rounds, use_agents=use_agents, reset=reset, labels=labels, first_round=first_round, lean=lean), indent=2, default=str))
