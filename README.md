# SkyTrack 🚀

**SkyTrack** is a lightweight, modular toolkit for monitoring and managing
machine‑learning training experiments at scale. Built for projects using
Weights & Biases (W&B), SkyPilot, and modern preference‑optimization
methods (ORPO, DPO, GRPO).

## ✨ Features

* 📊 Auto‑generated W&B dashboards with standardised metrics
* 🛰️ Seamless integration with SkyPilot job launches & sweeps
* 🔍 Helpers to log gradients, learning‑rates, and system stats
* 📦 Artifacts utility to bundle configs, checkpoints, tokenizers
* ⚙️ HuggingFace `TrainerCallback` for zero‑boilerplate logging

## 🚀 Quick Start

```bash
pip install git+https://github.com/your-org/skytrack.git
```

```python
import skytrack as st
run = st.init({"project": "demo", "run_name": "exp‑1"})
st.log({"loss": 0.12}, step=1)
run.finish()
```

## SkyPilot grid example

```bash
sky tune skypilot/skytrack_sweep.yaml
```

## License

Apache‑2.0
