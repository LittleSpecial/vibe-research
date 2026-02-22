from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as _ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")
REPO_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
STAGE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def repo_root_default() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _age_seconds(iso_ts: str) -> int | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    return max(0, int((datetime.now() - dt).total_seconds()))


def _tail_lines(path: Path, count: int = 120) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max(1, count) :]


def _read_text_head(path: Path, max_chars: int = 1600) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n... [truncated]"


def _collect_agent_notes(run_dir: Path, max_notes: int = 12, per_note_chars: int = 1200) -> list[dict]:
    agents_dir = run_dir / "agents"
    if not agents_dir.exists():
        return []

    notes: list[dict] = []
    files = sorted([p for p in agents_dir.iterdir() if p.is_file() and p.suffix == ".md"])
    for p in files[: max(1, max_notes)]:
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        except OSError:
            mtime = ""
        notes.append(
            {
                "name": p.name,
                "updated_at": mtime,
                "preview": _read_text_head(p, max_chars=per_note_chars),
            }
        )
    return notes


def _list_run_dirs(root: Path) -> list[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    out: list[Path] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.name == "__pycache__":
            continue
        out.append(p)
    out.sort(key=lambda x: x.name, reverse=True)
    return out


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    meta = _read_json(run_dir / "meta.json")
    updated_at = str(status.get("updated_at", "") or "")
    state = str(status.get("state", "unknown") or "unknown")
    age_sec = _age_seconds(updated_at)
    likely_stale = bool(
        state in {"running", "waiting_feedback"} and age_sec is not None and age_sec >= 900
    )
    return {
        "run_id": run_dir.name,
        "state": state,
        "stage": status.get("stage", "unknown"),
        "step": status.get("step", "?"),
        "total_steps": status.get("total_steps", "?"),
        "message": status.get("message", ""),
        "updated_at": updated_at,
        "age_seconds": age_sec,
        "likely_stale": likely_stale,
        "topic": status.get("topic", meta.get("topic", "")),
    }


def _run_detail(root: Path, run_id: str, tail: int) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")

    status = _read_json(run_dir / "status.json")
    plan = []
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        try:
            parsed = json.loads(plan_path.read_text(encoding="utf-8"))
            if isinstance(parsed, list):
                plan = parsed
        except (OSError, json.JSONDecodeError):
            plan = []

    feedback_dir = run_dir / "feedback"
    feedback_files = []
    if feedback_dir.exists():
        feedback_files = sorted([p.name for p in feedback_dir.iterdir() if p.is_file()])
    gate_stage = str(status.get("stage", "") or "")
    waiting_feedback = bool(status.get("state") == "waiting_feedback" and gate_stage)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "progress_tail": _tail_lines(run_dir / "progress.log", tail),
        "orchestrator_log_tail": _tail_lines(root / "runs" / ".orchestrator_logs" / "latest.log", 60),
        "plan_len": len(plan),
        "has_idea": (run_dir / "idea.md").exists(),
        "has_plan": (run_dir / "plan.json").exists(),
        "has_implementation": (run_dir / "implementation.md").exists(),
        "has_experiment": (run_dir / "experiment.sh").exists(),
        "feedback_files": feedback_files,
        "waiting_feedback": waiting_feedback,
        "feedback_stage": gate_stage if waiting_feedback else "",
        "agent_notes": _collect_agent_notes(run_dir),
        "remote_submit": _read_json(run_dir / "remote_submit.json"),
        "remote_submit_error": _read_json(run_dir / "remote_submit_error.json"),
    }


def _latest_run_id(root: Path) -> str:
    dirs = _list_run_dirs(root)
    if dirs:
        runs = [_summarize_run(p) for p in dirs]
        active_runs = [r for r in runs if str(r.get("state", "")) not in {"stale_stopped"}]
        pool = active_runs or runs
        pool.sort(
            key=lambda x: (
                str(x.get("updated_at", "") or ""),
                str(x.get("run_id", "") or ""),
            ),
            reverse=True,
        )
        top = pool[0] if pool else {}
        rid = str(top.get("run_id", "") or "")
        if rid:
            return rid

    latest = root / "runs" / "LATEST_RUN"
    if latest.exists():
        txt = latest.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            return txt
    return ""


def _safe_run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise ValueError("invalid run id")
    return run_id


def _safe_host(host: str) -> str:
    if not HOST_RE.match(host):
        raise ValueError("invalid host")
    return host


def _safe_repo(repo: str) -> str:
    if not REPO_RE.match(repo):
        raise ValueError("invalid repo")
    return repo


def _safe_stage(stage: str) -> str:
    if not STAGE_RE.match(stage):
        raise ValueError("invalid stage")
    return stage


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": cp.returncode == 0,
            "code": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    except FileNotFoundError as e:
        return {"ok": False, "code": 127, "stdout": "", "stderr": str(e)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": 124, "stdout": "", "stderr": f"command timeout ({timeout}s)"}


def _remote_probe(
    root: Path,
    run_id: str,
    host: str,
    remote_repo: str,
    job_id: str,
) -> dict[str, Any]:
    ssh_script = Path(
        os.getenv(
            "ZX_SSH_SCRIPT",
            str(Path.home() / ".codex/skills/paracloud-zx-ssh-workflow/scripts/ssh_zx.sh"),
        )
    )
    if not ssh_script.exists():
        return {
            "ok": False,
            "error": f"ssh helper missing: {ssh_script}",
            "host": host,
            "remote_repo": remote_repo,
            "run_id": run_id,
            "job_id": job_id,
        }

    base = [str(ssh_script), host, "--repo", remote_repo]

    status_cmd = (
        "python3 - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"p=Path('runs/{run_id}/status.json')\n"
        "if not p.exists():\n"
        "  print('status.json not found')\n"
        "else:\n"
        "  try:\n"
        "    d=json.loads(p.read_text())\n"
        "  except Exception:\n"
        "    d={}\n"
        "  print(f\"{d.get('updated_at','?')} | state={d.get('state','?')} | stage={d.get('stage','?')} | step={d.get('step','?')}/{d.get('total_steps','?')} | msg={d.get('message','')}\")\n"
        "PY"
    )
    status_res = _run_cmd(base + [status_cmd], timeout=20)

    queue_out = ""
    detail_out = ""
    log_out = ""
    warnings: list[str] = []

    if job_id and job_id.isdigit():
        q = _run_cmd(
            base
            + [
                (
                    "squeue -j "
                    f"{job_id} -o '%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R' 2>/dev/null "
                    "| sed -n '1,5p'"
                )
            ],
            timeout=20,
        )
        queue_out = q.get("stdout", "")
        if not q.get("ok"):
            warnings.append(q.get("stderr", "squeue failed"))

        detail = _run_cmd(
            base
            + [
                (
                    "scontrol show job "
                    f"{job_id} 2>/dev/null | tr ' ' '\\n' "
                    "| grep -E 'JobState=|RunTime=|TimeLimit=|NumNodes=|NumCPUs=|GRES=|NodeList='"
                )
            ],
            timeout=20,
        )
        detail_out = detail.get("stdout", "")
        if not detail.get("ok"):
            warnings.append(detail.get("stderr", "scontrol failed"))

        logs = _run_cmd(
            base
            + [
                f"ls -1t logs/*{job_id}*.out logs/*{job_id}*.err 2>/dev/null | head -n 2 | xargs -r tail -n 60"
            ],
            timeout=20,
        )
        log_out = logs.get("stdout", "")
        if not logs.get("ok"):
            warnings.append(logs.get("stderr", "log tail failed"))
    else:
        logs = _run_cmd(
            base + ["ls -1t logs/*.out logs/*.err 2>/dev/null | head -n 2 | xargs -r tail -n 60"],
            timeout=20,
        )
        log_out = logs.get("stdout", "")
        if not logs.get("ok"):
            warnings.append(logs.get("stderr", "log tail failed"))

    return {
        "ok": bool(status_res.get("ok")),
        "host": host,
        "remote_repo": remote_repo,
        "run_id": run_id,
        "job_id": job_id,
        "status_line": status_res.get("stdout", ""),
        "queue": queue_out,
        "detail": detail_out,
        "log_tail": log_out,
        "warnings": [w for w in warnings if w],
        "status_error": status_res.get("stderr", ""),
    }


def _write_feedback(root: Path, run_id: str, stage: str, action: str, note: str) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_id}")

    stage = _safe_stage(stage)
    if action not in {"approve", "revise"}:
        raise ValueError("action must be approve or revise")

    feedback_dir = run_dir / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    note_path = feedback_dir / f"{stage}.md"
    if note.strip():
        note_path.write_text(note.strip() + "\n", encoding="utf-8")

    marker = feedback_dir / f"{stage}.{action}"
    marker.write_text(f"{datetime.now().isoformat()}\n", encoding="utf-8")
    return {"ok": True, "stage": stage, "action": action, "note_written": bool(note.strip())}


def _mark_stale_run(root: Path, run_id: str, reason: str) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return {"run_id": run_id, "ok": False, "error": "status.json not found"}

    status = _read_json(status_path)
    prev_state = str(status.get("state", "unknown") or "unknown")
    now = datetime.now().isoformat()
    status["state"] = "stale_stopped"
    status["message"] = reason
    status["stale_from_state"] = prev_state
    status["stale_marked_at"] = now
    status["updated_at"] = now
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        with (run_dir / "progress.log").open("a", encoding="utf-8") as f:
            f.write(f"{now} | stale cleanup: {reason}\n")
    except OSError:
        pass

    return {"run_id": run_id, "ok": True, "prev_state": prev_state}


def _cleanup_stale_runs(root: Path, threshold_seconds: int = 1800, dry_run: bool = False) -> dict[str, Any]:
    threshold = max(60, int(threshold_seconds))
    runs = [_summarize_run(p) for p in _list_run_dirs(root)]
    candidates = [
        r
        for r in runs
        if str(r.get("state", "")) in {"running", "waiting_feedback"}
        and isinstance(r.get("age_seconds"), int)
        and int(r["age_seconds"]) >= threshold
    ]
    # Oldest first for predictable cleanup.
    candidates.sort(key=lambda x: int(x.get("age_seconds", 0)), reverse=True)

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "threshold_seconds": threshold,
            "count": len(candidates),
            "candidates": [str(x.get("run_id", "")) for x in candidates],
        }

    updated: list[dict] = []
    for r in candidates:
        run_id = str(r.get("run_id", "") or "")
        if not run_id:
            continue
        updated.append(
            _mark_stale_run(
                root,
                run_id=run_id,
                reason=f"marked stale by cleanup API (age >= {threshold}s)",
            )
        )
    return {
        "ok": True,
        "dry_run": False,
        "threshold_seconds": threshold,
        "count": len(updated),
        "updated": updated,
    }


def _start_cycle(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = str(payload.get("config", "configs/local.toml")).strip() or "configs/local.toml"
    topic = str(payload.get("topic", "")).strip()
    interactive = bool(payload.get("interactive", True))
    agent_count = int(payload.get("agent_count", 0))
    feedback_timeout = int(payload.get("feedback_timeout", 0))

    mode = "interactive" if interactive else "noninteractive"
    cmd = [
        str(root / "scripts" / "start_cycle.sh"),
        config,
        topic,
        mode,
        str(agent_count),
        str(feedback_timeout),
    ]
    res = _run_cmd(cmd, cwd=root, timeout=60)
    combined = (res.get("stdout", "") + "\n" + res.get("stderr", "")).strip()
    m = re.search(r"RUN_ID=([A-Za-z0-9._-]+)", combined)
    run_id = m.group(1) if m else _latest_run_id(root)
    return {
        "ok": res.get("ok", False),
        "run_id": run_id,
        "code": res.get("code", 1),
        "output": combined,
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vibe Research Control Room</title>
  <style>
    :root {
      --bg-0: #06080d;
      --bg-1: #0f1724;
      --bg-2: #142031;
      --panel: rgba(19, 27, 39, 0.86);
      --panel-strong: rgba(15, 21, 30, 0.92);
      --border: rgba(122, 159, 186, 0.32);
      --line: rgba(103, 177, 201, 0.22);
      --text: #dceaf3;
      --muted: #8da9bc;
      --accent: #4ad3ff;
      --accent-2: #83ffbe;
      --warn: #ffcc66;
      --danger: #ff8f8f;
      --shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      font-family: "Iosevka Aile", "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 520px at 110% -20%, rgba(74, 211, 255, 0.18), transparent 65%),
        radial-gradient(820px 440px at -10% 110%, rgba(131, 255, 190, 0.14), transparent 68%),
        linear-gradient(160deg, var(--bg-0), var(--bg-1) 42%, var(--bg-2));
      min-height: 100vh;
    }
    .grain::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
      background-size: 3px 3px, 3px 3px;
      mix-blend-mode: overlay;
      opacity: 0.13;
      z-index: 0;
    }
    .wrap {
      position: relative;
      z-index: 1;
      max-width: 1380px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 14px;
      grid-template-columns: 340px 1fr;
      grid-template-rows: auto auto 1fr;
      grid-template-areas:
        "head head"
        "left controls"
        "left main";
    }
    .head {
      grid-area: head;
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      box-shadow: var(--shadow);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .title {
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-family: "Avenir Next Condensed", "DIN Alternate", "IBM Plex Sans", sans-serif;
      font-weight: 700;
      font-size: 18px;
    }
    .subtitle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }
    .clock {
      color: var(--accent);
      font-size: 13px;
      font-family: "Iosevka", "Fira Code", monospace;
      white-space: nowrap;
    }
    .panel {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel h3 {
      margin: 0;
      padding: 10px 12px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-2);
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(74, 211, 255, 0.1), transparent 55%);
    }
    .left {
      grid-area: left;
      min-height: 620px;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .controls {
      grid-area: controls;
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(260px, 1fr));
    }
    .main {
      grid-area: main;
      min-height: 520px;
      display: grid;
      gap: 12px;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr;
    }
    .run-list {
      padding: 8px;
      overflow: auto;
      max-height: 84vh;
    }
    .run-item {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      margin: 8px 0;
      background: rgba(18, 30, 44, 0.66);
      cursor: pointer;
      transition: transform 130ms ease, border-color 130ms ease, background 130ms ease;
    }
    .run-item:hover { transform: translateY(-1px); border-color: var(--accent); }
    .run-item.active {
      border-color: var(--accent);
      background: rgba(26, 48, 70, 0.75);
      box-shadow: inset 0 0 0 1px rgba(74, 211, 255, 0.25);
    }
    .run-id {
      font-family: "Iosevka", "Fira Code", monospace;
      font-size: 12px;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .chips { margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; }
    .chip {
      font-size: 11px;
      font-family: "Iosevka", "Fira Code", monospace;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--muted);
      background: rgba(10, 16, 24, 0.7);
    }
    .state-running { color: var(--accent); }
    .state-completed { color: var(--accent-2); }
    .state-failed { color: var(--danger); }
    .state-waiting_feedback { color: var(--warn); }
    .state-stale { color: var(--danger); }
    .block {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(116, 159, 179, 0.16);
    }
    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 8px;
      font-size: 12px;
      line-height: 1.35;
    }
    .kv .k { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.06em; }
    .kv .v { font-family: "Iosevka", "Fira Code", monospace; word-break: break-word; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #d4e6f5;
      font-family: "Iosevka", "Fira Code", monospace;
      max-height: 360px;
      overflow: auto;
      background: rgba(7, 12, 18, 0.82);
      border: 1px solid rgba(100, 153, 178, 0.18);
      border-radius: 8px;
      padding: 10px;
    }
    .muted { color: var(--muted); font-size: 12px; }
    label { display: block; font-size: 12px; margin: 8px 0 4px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    input, textarea, select {
      width: 100%;
      border: 1px solid rgba(126, 167, 189, 0.34);
      background: rgba(5, 10, 15, 0.78);
      color: var(--text);
      border-radius: 8px;
      padding: 8px 9px;
      font-size: 13px;
      font-family: "Iosevka", "Fira Code", monospace;
    }
    textarea { min-height: 88px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .btn-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
    button {
      border: 1px solid rgba(109, 166, 187, 0.4);
      background: linear-gradient(180deg, rgba(74, 211, 255, 0.14), rgba(74, 211, 255, 0.06));
      color: #d7f6ff;
      border-radius: 8px;
      padding: 8px 10px;
      cursor: pointer;
      font-size: 12px;
      font-family: "Avenir Next Condensed", "DIN Alternate", sans-serif;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    button:hover { filter: brightness(1.1); }
    button.secondary {
      background: linear-gradient(180deg, rgba(131, 255, 190, 0.15), rgba(131, 255, 190, 0.05));
      border-color: rgba(131, 255, 190, 0.32);
      color: #dcffe9;
    }
    button.warn {
      background: linear-gradient(180deg, rgba(255, 204, 102, 0.2), rgba(255, 204, 102, 0.06));
      border-color: rgba(255, 204, 102, 0.4);
      color: #fff0ce;
    }
    .badge {
      font-family: "Iosevka", "Fira Code", monospace;
      color: var(--accent);
      border: 1px solid rgba(74, 211, 255, 0.3);
      background: rgba(14, 29, 40, 0.8);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 11px;
      display: inline-block;
    }
    .notice {
      font-size: 12px;
      min-height: 16px;
      color: var(--muted);
      margin-top: 8px;
    }
    .notice.warn { color: var(--warn); }
    .notice.ok { color: var(--accent-2); }
    @media (max-width: 1120px) {
      .wrap {
        grid-template-columns: 1fr;
        grid-template-areas:
          "head"
          "controls"
          "left"
          "main";
      }
      .controls { grid-template-columns: 1fr; }
      .main { grid-template-columns: 1fr; }
      .left { min-height: 360px; }
    }
  </style>
</head>
<body class="grain">
  <div class="wrap">
    <div class="head">
      <div>
        <div class="title">Vibe Research Control Room</div>
        <div class="subtitle">Local orchestration + remote A100 execution monitor</div>
      </div>
      <div>
        <div class="clock" id="clock"></div>
        <div class="muted"><span class="badge">auto refresh 3s</span></div>
      </div>
    </div>

    <section class="panel left">
      <h3>Runs</h3>
      <div class="run-list" id="runList"></div>
    </section>

    <section class="panel">
      <h3>Start New Cycle</h3>
      <div class="block">
        <label>Topic</label>
        <input id="topicInput" placeholder="e.g. RLHF credit assignment with long horizon feedback" />
        <div class="row">
          <div>
            <label>Config</label>
            <input id="configInput" value="configs/local.toml" />
          </div>
          <div>
            <label>Agent Count (0 = config default)</label>
            <input id="agentCountInput" value="0" />
          </div>
        </div>
        <div class="row">
          <div>
            <label>Interactive</label>
            <select id="interactiveInput">
              <option value="true" selected>true</option>
              <option value="false">false</option>
            </select>
          </div>
          <div>
            <label>Feedback Timeout (sec, 0 = wait)</label>
            <input id="feedbackTimeoutInput" value="0" />
          </div>
        </div>
        <div class="btn-row">
          <button id="startBtn">Start Cycle</button>
          <button class="secondary" id="refreshRunsBtn">Refresh Runs</button>
          <button class="warn" id="cleanStaleBtn">Clean Stale</button>
        </div>
        <div class="row">
          <div>
            <label>Stale Threshold (sec)</label>
            <input id="staleThresholdInput" value="1800" />
          </div>
          <div>
            <label>Run Filter</label>
            <select id="runFilterInput">
              <option value="all">show all</option>
              <option value="hide_stale" selected>hide stale</option>
            </select>
          </div>
        </div>
        <div class="notice" id="startNotice"></div>
      </div>
    </section>

    <section class="panel">
      <h3>Live Feedback Gate</h3>
      <div class="block">
        <div class="notice warn" id="gateNotice">No active feedback gate.</div>
        <label>Stage</label>
        <select id="stageSelect">
          <option value="literature">literature</option>
          <option value="idea">idea</option>
          <option value="planning">planning</option>
          <option value="implementation">implementation</option>
          <option value="ideation_agents">ideation_agents</option>
        </select>
        <label>Revision Note (optional for approve)</label>
        <textarea id="feedbackNote" placeholder="Write note, then click Revise for this stage."></textarea>
        <div class="btn-row">
          <button class="secondary" id="approveBtn">Approve</button>
          <button class="warn" id="reviseBtn">Revise</button>
        </div>
        <div class="notice" id="feedbackNotice"></div>
      </div>
    </section>

    <section class="panel main">
      <div>
        <h3>Local Run Detail</h3>
        <div class="block">
          <div class="kv">
            <div class="k">Run ID</div><div class="v" id="runIdCell">-</div>
            <div class="k">State</div><div class="v" id="runStateCell">-</div>
            <div class="k">Stage</div><div class="v" id="runStageCell">-</div>
            <div class="k">Step</div><div class="v" id="runStepCell">-</div>
            <div class="k">Gate</div><div class="v" id="runGateCell">-</div>
            <div class="k">Updated</div><div class="v" id="runUpdatedCell">-</div>
            <div class="k">Message</div><div class="v" id="runMsgCell">-</div>
          </div>
        </div>
        <div class="block">
          <div class="muted">Progress Tail</div>
          <pre id="progressBox"></pre>
        </div>
        <div class="block">
          <div class="muted">Agent Debate Notes</div>
          <pre id="agentsBox"></pre>
        </div>
      </div>

      <div>
        <h3>Remote Status</h3>
        <div class="block">
          <div class="row">
            <div>
              <label>Host</label>
              <input id="remoteHostInput" value="zw1" />
            </div>
            <div>
              <label>Remote Repo</label>
              <input id="remoteRepoInput" value="vibe-research" />
            </div>
          </div>
          <div class="btn-row">
            <button id="refreshRemoteBtn">Refresh Remote</button>
          </div>
          <div class="notice" id="remoteNotice"></div>
        </div>
        <div class="block">
          <div class="muted">Remote Status Line</div>
          <pre id="remoteStatusBox"></pre>
        </div>
        <div class="block">
          <div class="muted">Queue / Job Detail</div>
          <pre id="remoteQueueBox"></pre>
        </div>
        <div class="block">
          <div class="muted">Remote Log Tail</div>
          <pre id="remoteLogBox"></pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    const state = {
      selectedRun: null,
      runs: [],
    };

    function fmtNow() {
      const d = new Date();
      return d.toLocaleString();
    }

    function updateClock() {
      const clock = document.getElementById("clock");
      clock.textContent = fmtNow();
    }

    async function apiGet(path) {
      const r = await fetch(path);
      if (!r.ok) {
        throw new Error(await r.text());
      }
      return await r.json();
    }

    async function apiPost(path, payload) {
      const r = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {}),
      });
      if (!r.ok) {
        throw new Error(await r.text());
      }
      return await r.json();
    }

    function renderRuns() {
      const root = document.getElementById("runList");
      root.innerHTML = "";
      const filter = document.getElementById("runFilterInput").value;
      const runs = (filter === "hide_stale")
        ? state.runs.filter(r => !r.likely_stale)
        : state.runs;
      for (const run of runs) {
        const div = document.createElement("div");
        div.className = "run-item" + (state.selectedRun === run.run_id ? " active" : "");
        const cls = "state-" + String(run.state || "").replace(/[\\s]+/g, "_");
        const staleChip = run.likely_stale ? `<span class="chip state-stale">stale?</span>` : "";
        const age = (run.age_seconds ?? null);
        const ageTxt = (age === null) ? "" : ` | age=${age}s`;
        div.innerHTML = `
          <div class="run-id">${run.run_id}</div>
          <div class="chips">
            <span class="chip ${cls}">${run.state || "unknown"}</span>
            <span class="chip">${run.stage || "?"}</span>
            <span class="chip">${run.step || "?"}/${run.total_steps || "?"}</span>
            ${staleChip}
          </div>
          <div class="muted" style="margin-top:6px;">${run.updated_at || ""}${ageTxt}</div>
        `;
        div.onclick = () => {
          state.selectedRun = run.run_id;
          renderRuns();
          refreshSelectedRun();
          refreshRemote();
        };
        root.appendChild(div);
      }
    }

    function setStageIfExists(stage) {
      const sel = document.getElementById("stageSelect");
      const options = Array.from(sel.options).map(o => o.value);
      if (options.includes(stage)) {
        sel.value = stage;
      }
    }

    function setText(id, txt) {
      document.getElementById(id).textContent = txt || "-";
    }

    async function refreshRuns() {
      try {
        const data = await apiGet("/api/runs?limit=50");
        state.runs = data.runs || [];
        if (!state.selectedRun) {
          state.selectedRun = data.latest_run || (state.runs[0] ? state.runs[0].run_id : null);
        } else if (!state.runs.some(x => x.run_id === state.selectedRun)) {
          state.selectedRun = state.runs[0] ? state.runs[0].run_id : null;
        }
        renderRuns();
      } catch (e) {
        document.getElementById("startNotice").textContent = "refresh runs failed: " + e.message;
      }
    }

    async function refreshSelectedRun() {
      if (!state.selectedRun) { return; }
      try {
        const data = await apiGet(`/api/run/${encodeURIComponent(state.selectedRun)}?tail=160`);
        const s = data.status || {};
        setText("runIdCell", data.run_id);
        setText("runStateCell", s.state);
        setText("runStageCell", s.stage);
        setText("runStepCell", `${s.step ?? "?"}/${s.total_steps ?? "?"}`);
        setText("runGateCell", data.waiting_feedback ? data.feedback_stage : "-");
        setText("runUpdatedCell", s.updated_at);
        setText("runMsgCell", s.message);
        document.getElementById("progressBox").textContent =
          (data.progress_tail || []).join("\\n") || "(no progress yet)";

        const notes = data.agent_notes || [];
        document.getElementById("agentsBox").textContent = notes.length
          ? notes.map(n => `### ${n.name}\\n${n.preview || "(empty)"}`).join("\\n\\n---\\n\\n")
          : "(no agent notes yet)";

        const gateNotice = document.getElementById("gateNotice");
        const feedbackNotice = document.getElementById("feedbackNotice");
        if (data.waiting_feedback && data.feedback_stage) {
          setStageIfExists(data.feedback_stage);
          gateNotice.textContent = `Waiting feedback for stage: ${data.feedback_stage}. Click Approve or Revise below.`;
          gateNotice.className = "notice warn";
          feedbackNotice.textContent = `Pending: ${data.feedback_stage}`;
          feedbackNotice.className = "notice warn";
        } else {
          gateNotice.textContent = "No active feedback gate.";
          gateNotice.className = "notice";
          if (!feedbackNotice.textContent.startsWith("ok:")) {
            feedbackNotice.textContent = "";
            feedbackNotice.className = "notice";
          }
        }
      } catch (e) {
        document.getElementById("progressBox").textContent = "load run failed: " + e.message;
      }
    }

    async function refreshRemote() {
      if (!state.selectedRun) { return; }
      const host = document.getElementById("remoteHostInput").value.trim() || "zw1";
      const repo = document.getElementById("remoteRepoInput").value.trim() || "vibe-research";
      try {
        const data = await apiGet(`/api/remote/${encodeURIComponent(state.selectedRun)}?host=${encodeURIComponent(host)}&repo=${encodeURIComponent(repo)}`);
        document.getElementById("remoteNotice").textContent =
          data.ok ? "remote refresh ok" : ("remote refresh issue: " + (data.error || data.status_error || "unknown"));
        document.getElementById("remoteStatusBox").textContent = data.status_line || "(empty)";
        const queue = [data.queue || "", data.detail || ""].filter(Boolean).join("\\n");
        document.getElementById("remoteQueueBox").textContent = queue || "(no queue detail)";
        document.getElementById("remoteLogBox").textContent = data.log_tail || "(no remote logs)";
      } catch (e) {
        document.getElementById("remoteNotice").textContent = "remote refresh failed: " + e.message;
      }
    }

    async function startCycle() {
      const topic = document.getElementById("topicInput").value.trim();
      const config = document.getElementById("configInput").value.trim() || "configs/local.toml";
      const agentCount = parseInt(document.getElementById("agentCountInput").value.trim() || "0", 10) || 0;
      const interactive = document.getElementById("interactiveInput").value === "true";
      const feedbackTimeout = parseInt(document.getElementById("feedbackTimeoutInput").value.trim() || "0", 10) || 0;

      const notice = document.getElementById("startNotice");
      notice.textContent = "starting...";
      try {
        const data = await apiPost("/api/start-cycle", {
          topic, config, agent_count: agentCount, interactive, feedback_timeout: feedbackTimeout
        });
        notice.textContent = data.ok
          ? `started: ${data.run_id}`
          : `start failed(code=${data.code}): ${data.output}`;
        if (data.run_id) {
          state.selectedRun = data.run_id;
        }
        await refreshRuns();
        await refreshSelectedRun();
      } catch (e) {
        notice.textContent = "start failed: " + e.message;
      }
    }

    async function cleanStaleRuns() {
      const notice = document.getElementById("startNotice");
      const threshold = parseInt(document.getElementById("staleThresholdInput").value.trim() || "1800", 10) || 1800;
      notice.textContent = "cleaning stale runs...";
      try {
        const data = await apiPost("/api/runs/cleanup-stale", {
          threshold_seconds: threshold,
          dry_run: false
        });
        notice.textContent = `stale cleanup done: ${data.count || 0} run(s) marked stale`;
        await refreshRuns();
        await refreshSelectedRun();
      } catch (e) {
        notice.textContent = "stale cleanup failed: " + e.message;
      }
    }

    async function sendFeedback(action) {
      if (!state.selectedRun) { return; }
      const stage = document.getElementById("stageSelect").value;
      const note = document.getElementById("feedbackNote").value;
      const notice = document.getElementById("feedbackNotice");
      notice.textContent = `${action}...`;
      try {
        const data = await apiPost(`/api/run/${encodeURIComponent(state.selectedRun)}/feedback`, {
          stage, action, note
        });
        notice.textContent = `ok: ${data.stage}.${data.action}`;
        notice.className = "notice ok";
        await refreshSelectedRun();
      } catch (e) {
        notice.textContent = "feedback failed: " + e.message;
        notice.className = "notice warn";
      }
    }

    function boot() {
      updateClock();
      setInterval(updateClock, 1000);
      document.getElementById("startBtn").onclick = startCycle;
      document.getElementById("refreshRunsBtn").onclick = refreshRuns;
      document.getElementById("cleanStaleBtn").onclick = cleanStaleRuns;
      document.getElementById("runFilterInput").onchange = renderRuns;
      document.getElementById("refreshRemoteBtn").onclick = refreshRemote;
      document.getElementById("approveBtn").onclick = () => sendFeedback("approve");
      document.getElementById("reviseBtn").onclick = () => sendFeedback("revise");

      refreshRuns().then(refreshSelectedRun).then(refreshRemote);
      setInterval(async () => {
        await refreshRuns();
        await refreshSelectedRun();
      }, 3000);
      setInterval(async () => {
        await refreshRemote();
      }, 6000);
    }

    boot();
  </script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    repo_root: Path = repo_root_default()
    default_host: str = "zw1"
    default_remote_repo: str = "vibe-research"

    def log_message(self, fmt: str, *args: object) -> None:
        # Keep server output compact for long-running monitor mode.
        _ = fmt, args

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_html(INDEX_HTML)
                return
            if path == "/healthz":
                self._send_json({"ok": True, "time": datetime.now().isoformat()})
                return
            if path == "/api/runs":
                limit = int(query.get("limit", ["50"])[0])
                runs_all = [_summarize_run(p) for p in _list_run_dirs(self.repo_root)]
                runs_all.sort(
                    key=lambda x: (
                        str(x.get("updated_at", "") or ""),
                        str(x.get("run_id", "") or ""),
                    ),
                    reverse=True,
                )
                runs = runs_all[: max(1, limit)]
                self._send_json(
                    {
                        "runs": runs,
                        "latest_run": _latest_run_id(self.repo_root),
                        "time": datetime.now().isoformat(),
                    }
                )
                return
            if path == "/api/latest-run":
                self._send_json({"run_id": _latest_run_id(self.repo_root)})
                return

            m_run = re.fullmatch(r"/api/run/([A-Za-z0-9._-]+)", path)
            if m_run:
                run_id = _safe_run_id(m_run.group(1))
                tail = int(query.get("tail", ["120"])[0])
                self._send_json(_run_detail(self.repo_root, run_id, tail))
                return

            m_remote = re.fullmatch(r"/api/remote/([A-Za-z0-9._-]+)", path)
            if m_remote:
                run_id = _safe_run_id(m_remote.group(1))
                detail = _run_detail(self.repo_root, run_id, 80)
                submit = detail.get("remote_submit", {})
                if not isinstance(submit, dict):
                    submit = {}
                host = _safe_host(
                    (query.get("host", [""])[0] or str(submit.get("host", "")) or self.default_host)
                )
                repo = _safe_repo(
                    (
                        query.get("repo", [""])[0]
                        or str(submit.get("remote_repo", ""))
                        or self.default_remote_repo
                    )
                )
                q_job = query.get("job_id", [""])[0].strip()
                job_id = q_job or str(submit.get("job_id", "")).strip()
                out = _remote_probe(self.repo_root, run_id, host, repo, job_id)
                self._send_json(out)
                return

            self._send_error_json(HTTPStatus.NOT_FOUND, f"route not found: {path}")
        except ValueError as e:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(e))
        except FileNotFoundError as e:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:  # pragma: no cover - defensive fallback
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")) or 0)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {e}")
            return

        try:
            if path == "/api/start-cycle":
                out = _start_cycle(self.repo_root, payload)
                code = HTTPStatus.OK if out.get("ok") else HTTPStatus.INTERNAL_SERVER_ERROR
                self._send_json(out, status=code)
                return

            if path == "/api/runs/cleanup-stale":
                threshold_seconds = int(payload.get("threshold_seconds", 1800))
                dry_run = bool(payload.get("dry_run", False))
                out = _cleanup_stale_runs(
                    self.repo_root,
                    threshold_seconds=threshold_seconds,
                    dry_run=dry_run,
                )
                self._send_json(out)
                return

            m_feedback = re.fullmatch(r"/api/run/([A-Za-z0-9._-]+)/feedback", path)
            if m_feedback:
                run_id = _safe_run_id(m_feedback.group(1))
                stage = str(payload.get("stage", "")).strip() or "idea"
                action = str(payload.get("action", "")).strip()
                note = str(payload.get("note", ""))
                out = _write_feedback(self.repo_root, run_id, stage, action, note)
                self._send_json(out)
                return

            self._send_error_json(HTTPStatus.NOT_FOUND, f"route not found: {path}")
        except ValueError as e:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(e))
        except FileNotFoundError as e:
            self._send_error_json(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:  # pragma: no cover - defensive fallback
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = body.encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"ok": False, "error": message}, status=status)


class FastThreadingHTTPServer(_ThreadingHTTPServer):
    """
    Avoid DNS reverse lookup in HTTPServer.server_bind().
    Python's default HTTPServer calls socket.getfqdn(host), which can block for
    a long time in restricted network environments.
    """

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


ThreadingHTTPServer = FastThreadingHTTPServer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vibe-monitor")
    p.add_argument("--root", default="", help="repo root (default: auto-detect from package path)")
    p.add_argument("--host", default="127.0.0.1", help="bind address")
    p.add_argument("--port", type=int, default=8787, help="HTTP port")
    p.add_argument("--remote-host", default="zw1", help="default remote host for /api/remote")
    p.add_argument("--remote-repo", default="vibe-research", help="default remote repo for /api/remote")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve() if args.root else repo_root_default()

    class Handler(MonitorHandler):
        pass

    Handler.repo_root = root
    Handler.default_host = args.remote_host
    Handler.default_remote_repo = args.remote_repo

    server = FastThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(
        f"[{datetime.now().isoformat()}] vibe-monitor running at http://{args.host}:{args.port} "
        f"(root={root})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
