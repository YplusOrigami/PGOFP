"""Matrix manifests: every planned / running / completed / failed / skipped run with its config hash and reason."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.provenance import utc_now

STATUSES = ("planned", "running", "completed", "failed", "skipped", "interrupted")


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


class MatrixManifest:
    def __init__(self, matrix_dir: str | Path, matrix_id: str, runs: Optional[List[Dict[str, Any]]] = None):
        self.dir = Path(matrix_dir)
        self.matrix_id = matrix_id
        self.runs: List[Dict[str, Any]] = runs or []
        self.created = utc_now()

    @property
    def path(self) -> Path:
        return self.dir / "manifest.json"

    def save(self) -> None:
        write_json_atomic(self.path, {"matrix_id": self.matrix_id, "created_utc": self.created, "updated_utc": utc_now(),
                                      "counts": self.counts(), "runs": self.runs})

    @staticmethod
    def load(path: str | Path) -> "MatrixManifest":
        d = json.loads(Path(path).read_text())
        m = MatrixManifest(Path(path).parent, d["matrix_id"], d["runs"])
        m.created = d.get("created_utc", m.created)
        return m

    def counts(self) -> Dict[str, int]:
        c = {s: 0 for s in STATUSES}
        for r in self.runs:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return next((r for r in self.runs if r["run_key"] == key), None)

    def write_failures(self) -> None:
        bad = [r for r in self.runs if r["status"] in ("failed", "skipped", "interrupted")]
        write_json_atomic(self.dir / "failures.json", bad)
