"""
SkyTrack – Lightweight experiment-tracking helpers built on
Weights & Biases (W&B) and SkyPilot.

Import & quick‑start::

    import skytrack as st
    run = st.init({"project": "demo", "run_name": "test"})
    st.log({"loss": 0.42}, step=1)

SkyTrack's goal is to give you **zero‑boilerplate, repeatable logging**
across multiple repos and clusters, while staying 100 % compatible with
raw ``wandb`` calls and SkyPilot job files.
"""
from .logging import init, log, log_gradients, log_lr
from .callbacks import SkyTrackCallback
from .artifacts import log_artifacts
from wandb import watch as watch
from .sweep import sweep  # re-export

__all__ = [
    "init", "log", "log_gradients", "log_lr",
    "SkyTrackCallback", "log_artifacts"
]

__version__ = "0.1.0"


def _cli():
    """Real sky-tune command:  sky-tune <sweep.yaml>"""
    import argparse
    p = argparse.ArgumentParser(description="SkyPilot-based hyper-sweep")
    p.add_argument("yaml", help="Sweep YAML file")
    args = p.parse_args()
    sweep(args.yaml)