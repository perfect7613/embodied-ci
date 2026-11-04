import json

from fastapi.testclient import TestClient

from infra.api import make_app


def test_runs_are_private_to_their_owner(tmp_path):
    root = str(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "run-demo.jsonl").write_text('{"kind":"run_start"}\n')
    (tmp_path / "public.json").write_text(json.dumps(["run-demo"]))
    started = []

    def start(run_id, *a):
        (tmp_path / "runs" / f"{run_id}.jsonl").write_text('{"kind":"run_start"}\n')
        started.append(run_id)
        return "call"

    app = make_app(root, start, lambda *a: None, [{"id": "vjepa2-ac-vitg"}], [{"id": "lift"}], "test")
    c = TestClient(app)
    alice, bob = {"x-owner": "a" * 32}, {"x-owner": "b" * 32}
    assert c.post("/runs", json={}).status_code == 400  # no owner, no run
    rid = c.post("/runs", json={}, headers=alice).json()["run_id"]
    ids = lambda h: {r["run_id"] for r in c.get("/runs", headers=h).json()}
    assert ids(alice) == {"run-demo", rid}
    assert ids(bob) == {"run-demo"}
    assert c.get(f"/runs/{rid}/events", headers=bob).status_code == 404
    assert c.get(f"/runs/{rid}/events", headers=alice).status_code == 200
    assert "a" * 32 not in (tmp_path / "runs" / f"{rid}.owner").read_text()  # only the hash is stored
