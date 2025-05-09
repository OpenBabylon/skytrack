"""Hugging Face Trainer callback → forwards logs to W&B automatically."""
from transformers import TrainerCallback
import wandb

class SkyTrackCallback(TrainerCallback):
    """Stream HF ``Trainer`` logs into Weights & Biases."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)
