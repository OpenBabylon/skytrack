#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
skytrack.sweep  –  real 'sky-tune' / 'skygrid' engine.

Supports two modes:
  • mode: grid       – Cartesian product of hyper-params
  • mode: benchmark  – Run one script across many MODEL_IDs

YAML keys (top level)
---------------------
mode: grid | benchmark   # default = grid
sweep:                     # infrastructure controls
  name: my-sweep
  template: sky_task.yaml  # single-job task file
  max_parallel: 2
  max_retries: 2
grid: { … }                # used when mode == grid
benchmark:                 # used when mode == benchmark
  script: scripts/eval.py
  models: [modelA, modelB]
slug_pattern: |            # python format str
  {MODE}_lr{LR:g}_gbs{GBS}_ep{EPOCHS}{lora_suffix}_{uid}
"""

from __future__ import annotations
import itertools, uuid, yaml, subprocess, time, json, sys
from pathlib import Path
from typing import Dict, Iterable, Any
import subprocess, time, json, sys, yaml, itertools, uuid, re

_STATUS_RE = re.compile(r"^\s*(\S+)\s+(RUNNING|INIT)\b")
NUM_RE = re.compile(r"^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?$")


def _running() -> list[str]:
    out = subprocess.check_output(["sky", "status", "-v"], text=True)
    return [m.group(1) for line in out.splitlines()
            if (m := _STATUS_RE.match(line))]

# def _running() -> list[str]:
#     """Return list of cluster names whose status is RUNNING or INIT."""
#     out = subprocess.check_output(["sky", "status", "-v"], text=True)
#     running = []
#     for line in out.splitlines():
#         m = _STATUS_RE.match(line)
#         if m:
#             name, status = m.groups()
#             running.append(name)
#     return running

            # list of cluster names
TASKS_DIR  = Path(".sky_tasks")
STATE_PATH = TASKS_DIR / ".retry_state.json"

# --------------------------------------------------------------------------- #
# Helper: load YAML safely
# --------------------------------------------------------------------------- #
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)

# --------------------------------------------------------------------------- #
# Helper: currently RUNNING/INIT VMs
# --------------------------------------------------------------------------- #
# def _running() -> list[dict]:
#     out = subprocess.check_output(["sky", "status", "--format", "json"])
#     jobs = yaml.safe_load(out) or []
#     return [j for j in jobs if j["status"] in ("RUNNING", "INIT")]

# --------------------------------------------------------------------------- #
# Build matrix of env-dicts for each job
# --------------------------------------------------------------------------- #
def _matrix(cfg: dict) -> Iterable[dict]:
    mode = cfg.get("mode", "grid")
    if mode == "grid":
        keys, vals = zip(*cfg["grid"].items())
        for combo in itertools.product(*vals):
            yield dict(zip(keys, combo))
    elif mode == "benchmark":
        script = cfg["benchmark"]["script"]
        for mid in cfg["benchmark"]["models"]:
            yield {"MODEL_ID": mid, "BENCH_SCRIPT": script}
    else:
        raise ValueError(f"Unknown mode {mode}")

# --------------------------------------------------------------------------- #
def sweep(cfg_path: str | Path):
    cfg = _load_yaml(cfg_path)
    sweep_cfg = cfg["sweep"]
    slug_fmt  = cfg["slug_pattern"].strip()

    TASKS_DIR.mkdir(exist_ok=True)
    state: dict[str, int | str] = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())

    def save_state(): STATE_PATH.write_text(json.dumps(state, indent=2))

    for env in _matrix(cfg):
        # --- try to convert numeric strings to float just for slugging
        env_for_slug = {
            k: (float(v) if isinstance(v, str) and NUM_RE.match(v) else v)
            for k, v in env.items()
        }
        env_for_slug["uid"] = uuid.uuid4().hex[:4]
        env_for_slug["lora_suffix"] = f"_r{env.get('LORA_R')}" if env.get("MODE") == "lora" else ""

        slug = slug_fmt.format(**env_for_slug)

        # keep original strings for ENV injection
        env["uid"] = env_for_slug["uid"]
        env["lora_suffix"] = env_for_slug["lora_suffix"]

        # -------- concurrency gate -----------------------------------------
        while len(_running()) >= sweep_cfg["max_parallel"]:
            print("⏳ concurrency limit reached – sleeping 60 s")
            time.sleep(60)

        # -------- build per-job task YAML ----------------------------------
        task = _load_yaml(sweep_cfg["template"])
        task.setdefault("envs", {}).update({k: str(v) for k, v in env.items()})
        task["envs"]["WANDB_RUN_NAME"] = slug

        task_path = TASKS_DIR / f"{slug}.yaml"
        with task_path.open("w") as f:
            yaml.safe_dump(task, f)

        # -------- launch ----------------------------------------------------
        print(f"🚀 launch {slug}")
        try:
            subprocess.check_call(["sky", "launch", "-d", "--name", slug, task_path])
            state.setdefault(slug, 0)          # init retry counter
        except subprocess.CalledProcessError:
            state[slug] = state.get(slug, 0) + 1
            print(f"🚨 immediate launch failure (attempt {state[slug]})")
        save_state()

    # -------- monitor / retry loop -----------------------------------------
    while True:
        out = subprocess.check_output(["sky", "status", "--format", "json"])
        jobs = yaml.safe_load(out) or []
        active = [j for j in jobs if j["status"] in ("RUNNING", "INIT")]
        failed = [j for j in jobs if j["status"] == "FAILED"]
        succeeded = [j for j in jobs if j["status"] == "SUCCEEDED"]

        for j in succeeded:
            state[j["name"]] = "DONE"
        save_state()

        for j in failed:
            cnt = state.get(j["name"], 0)
            if isinstance(cnt, int) and cnt < sweep_cfg["max_retries"]:
                print(f"🔄 retry {j['name']} (attempt {cnt+1})")
                subprocess.call([
                    "sky", "launch", "-d", "--name", j["name"],
                    TASKS_DIR / f"{j['name']}.yaml"
                ])
                state[j["name"]] = cnt + 1
                save_state()
            else:
                print(f"✖️ perm-failed {j['name']}")

        if not active and not failed:
            print("🎉 all jobs finished")
            break

        time.sleep(120)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m skytrack.sweep <sweep.yaml>")
    sweep(sys.argv[1])
