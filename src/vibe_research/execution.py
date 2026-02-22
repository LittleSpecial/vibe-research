from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_experiment(run_dir: Path) -> int:
    script = run_dir / "experiment.sh"
    if not script.exists():
        msg = {
            "status": "skipped",
            "reason": "experiment.sh not found",
            "hint": "Create runs/<RUN_ID>/experiment.sh or generate it in next iteration.",
        }
        (run_dir / "results.json").write_text(json.dumps(msg, indent=2), encoding="utf-8")
        return 0

    script.chmod(0o755)
    proc = subprocess.run(["bash", str(script)], cwd=str(run_dir), check=False)
    return proc.returncode
