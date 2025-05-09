"""Core helpers for W&B run initialisation and lightweight metric logging."""
from __future__ import annotations
from typing import Mapping, Any
import subprocess, os, wandb

_RUN = None  # singleton to avoid duplicate init() calls

# --------------------------------------------------------------------- #
# Utilities                                                              #
# --------------------------------------------------------------------- #

def _git_rev() -> str:
    """Return current git commit hash or ``'unknown'`` if not in repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #

def init(cfg: Mapping[str, Any] | None = None):
    """Idempotent W&B initialisation with a sensible default dashboard.

    Parameters
    ----------
    cfg
        Any mapping (dict, OmegaConf) of parameters. Contents are
        forwarded to W&B ``run.config`` for easy filtering.

    Returns
    -------
    wandb.sdk.wandb_run.Run
        The active W&B run.
    """
    global _RUN
    if _RUN is not None:
        return _RUN

    cfg = dict(cfg or {})
    cfg.setdefault("git_commit", _git_rev())

    _RUN = wandb.init(
        project=cfg.get("project", "skytrack"),
        entity=cfg.get("entity"),
        name=cfg.get("run_name"),
        config=cfg,
    )
    _setup_dashboard()
    return _RUN

# --------------------------------------------------------------------- #
# Dashboard & metric helpers                                             #
# --------------------------------------------------------------------- #

def _setup_dashboard():
    """Define common metric patterns so W&B auto-generates summaries."""
    # Minimise all losses
    wandb.define_metric("loss/*", summary="min")
    # Maximise accuracies / rewards
    wandb.define_metric("accuracy*", summary="max")
    wandb.define_metric("reward*", summary="max")
    # Learning rate: keep last value
    wandb.define_metric("lr*", summary="last")

# Shorthand alias: st.log({...}) is same as wandb.log({...})
log = wandb.log

def log_gradients(model, step: int, every: int = 100):
    """Log L2‑norm of gradients every *every* steps."""
    if step % every:
        return
    grads = {
        f"grad/{n}": p.grad.norm().item()
        for n, p in model.named_parameters() if p.grad is not None
    }
    if grads:
        wandb.log(grads, step=step)

def log_lr(optimizer, step: int):
    """Log learning‑rate(s) of an optimizer."""
    for i, group in enumerate(optimizer.param_groups):
        wandb.log({f"lr/group_{i}": group["lr"]}, step=step)
