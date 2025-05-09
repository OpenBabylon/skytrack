"""Lightweight loader that tries OmegaConf, then YAML, then JSON."""
from pathlib import Path
import json, yaml

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

def load(path):
    path = Path(path)
    if OmegaConf is not None:
        try:
            return OmegaConf.load(path)
        except Exception:
            pass
    with open(path, "r") as f:
        if path.suffix == ".json":
            return json.load(f)
        return yaml.safe_load(f)
