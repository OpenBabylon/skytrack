"""Utility for bundling multiple files/dirs into a single W&B artifact."""
from pathlib import Path
from typing import Iterable
import wandb

def log_artifacts(paths: Iterable[str | Path],
                  artifact_name: str = "bundle",
                  artifact_type: str = "experiment"):
    run = wandb.run or wandb.init(project="skytrack")
    art = wandb.Artifact(artifact_name, type=artifact_type)
    for p in paths:
        p = Path(p)
        if p.is_dir():
            art.add_dir(str(p))
        elif p.is_file():
            art.add_file(str(p))
    run.log_artifact(art)
    print(f"[SkyTrack] Logged {artifact_name} → W&B")
