"""Self-hosted EmbodiedCI on any CUDA GPU box (the SSH deploy target).

    uv run --extra engine python -m infra.local_server --port 8700 --data ./data

Same API as the Modal deployment; the engine runs in-process on the local GPU,
and CI runs start in a background thread. Reach it through an SSH tunnel:
    ssh -N -L 8700:localhost:8700 you@gpu-box
then point the studio at http://localhost:8700 (Deploy page -> Connect).
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path


class LocalEngine:
    """The engine seam's second adapter: in-process, one GPU, no remote calls."""

    def __init__(self, root: str):
        from embodied.rollout import Runner
        from embodied.world_model import WorldModel

        self.runner = Runner(WorldModel.load(), data_root=root)
        self.root = root
        self.lock = threading.Lock()

    def run(self, cand, specs):
        with self.lock:
            return self.runner.run(cand, specs)

    def fit_probes(self, seeds, drifts, label_source="sim", readout=0):
        from embodied.probes import fit_probes

        with self.lock:
            return fit_probes(self.runner.wm, self.runner.task, seeds, drifts, label_source=label_source, readout=readout)

    def eval_probes(self, probes, seeds, drifts):
        from embodied.probes import eval_probes

        with self.lock:
            return eval_probes(self.runner.wm, self.runner.task, probes, seeds, drifts)

    def episode(self, eid: str) -> dict:
        import numpy as np

        with np.load(f"{self.root}/episodes/{eid}.npz") as d:
            return {k: d[k] for k in d.files}

    def frame(self, eid: str):
        import imageio.v3 as iio

        return iio.imread(f"{self.root}/episodes/{eid}.png")


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    from infra.api import make_app

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--host", default="127.0.0.1")  # tunnel in over SSH; don't expose publicly
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()
    load_dotenv()
    root = str(Path(args.data).resolve())
    Path(root).mkdir(parents=True, exist_ok=True)
    engine: list[LocalEngine] = []

    def start(run_id: str, rounds: int, use_agents: bool, reset: bool) -> str:
        def go():
            import shutil

            from harness.ci import Harness

            if not engine:
                engine.append(LocalEngine(root))
            if reset:
                for p in ("champion.json", "registry.json", "train_ids.json", "v0.json"):
                    Path(root, p).unlink(missing_ok=True)
                shutil.rmtree(Path(root, "candidates"), ignore_errors=True)
            h = Harness(engine[0], root, run_id=run_id)
            h.log("run_start", "", rounds=rounds, model="vjepa2-ac-vitg", task="lift", use_agents=use_agents, backend="ssh")
            if rounds == 0:
                h.bootstrap()
            out = [h.round(k, use_agents=use_agents) for k in range(1, rounds + 1)]
            h.log("run_end", "", rounds=out)

        threading.Thread(target=go, daemon=True).start()
        return f"local-{run_id}"

    from infra.catalog import MODELS, TASKS

    uvicorn.run(make_app(root, start, lambda *a: None, MODELS, TASKS, backend="ssh"), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
