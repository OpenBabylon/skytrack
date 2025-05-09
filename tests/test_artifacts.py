import skytrack.artifacts as sa
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os, wandb

def test_log_artifact(tmp_path):
    file = tmp_path / "foo.txt"
    file.write_text("hello")
    run = wandb.init(mode="offline")
    sa.log_artifacts([file], artifact_name="tmp_art", artifact_type="unit-test")
    run.finish()
