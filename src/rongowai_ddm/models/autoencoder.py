"""A small MLP autoencoder that compresses the 200 normalized pixels to a latent vector.

Used as the "learned representation" baseline: the latent codes are fed to the
same tabular models as the hand-crafted features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

PIXEL_SCALE = 200.0  # normalized pixels average 1/200; rescale so the mean is one


class DDMAutoencoder(nn.Module):
    def __init__(self, n_in: int = 200, hidden: int = 128, latent: int = 20) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X: np.ndarray,
    X_val: np.ndarray | None = None,
    latent: int = 20,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
) -> tuple[DDMAutoencoder, dict]:
    """Train on sum-normalized pixel vectors ``(N, 200)``; returns the model and the loss history."""
    device = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = DDMAutoencoder(latent=latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32) * PIXEL_SCALE)
    xv = (
        torch.from_numpy(np.asarray(X_val, dtype=np.float32) * PIXEL_SCALE).to(device)
        if X_val is not None
        else None
    )
    history: dict[str, list[float]] = {"train": [], "val": []}
    n = xt.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, batch_size):
            xb = xt[perm[s : s + batch_size]].to(device, non_blocking=True)
            loss = nn.functional.mse_loss(model(xb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.shape[0]
        sched.step()
        history["train"].append(tot / n)
        if xv is not None:
            model.eval()
            with torch.no_grad():
                history["val"].append(float(nn.functional.mse_loss(model(xv), xv)))
    return model.cpu(), history


@torch.no_grad()
def encode(
    model: DDMAutoencoder, X: np.ndarray, device: str = "cuda", batch_size: int = 65536
) -> np.ndarray:
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    out = []
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32) * PIXEL_SCALE)
    for s in range(0, xt.shape[0], batch_size):
        out.append(model.encoder(xt[s : s + batch_size].to(device)).cpu().numpy())
    model.cpu()
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def reconstruct(model: DDMAutoencoder, X: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(X, dtype=np.float32) * PIXEL_SCALE)
    return (model.cpu().eval()(xt).numpy() / PIXEL_SCALE).astype(np.float32)
