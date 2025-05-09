"""Toy demo: linear regression with SkyTrack logging."""
import torch, skytrack as st, random

cfg = {"project": "skytrack-demo", "run_name": "linear"}
run = st.init(cfg)

# Dataset
xs = torch.randn(1024, 1)
ys = 3.0 * xs + 0.5 + 0.1 * torch.randn_like(xs)

model = torch.nn.Linear(1, 1)
opt   = torch.optim.SGD(model.parameters(), lr=0.03)

for step in range(2000):
    opt.zero_grad()
    pred = model(xs)
    loss = torch.nn.functional.mse_loss(pred, ys)
    loss.backward()
    opt.step()

    # Log
    st.log({"loss/train": loss.item()}, step=step)
    st.log_lr(opt, step)
    st.log_gradients(model, step, every=100)
run.finish()
