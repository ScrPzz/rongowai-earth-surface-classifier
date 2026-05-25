"""Conv2d autoencoder for DDM compression (Phase 1.2 — MLP-encoder replacement).

The DDM is intrinsically a 2-D signal (5 Doppler bins × 40 delay bins). The
production encoder is a fully-connected MLP that must rediscover spatial
locality from untied weights; a small 2-D CNN bakes locality in for free.

Architecture (small by design — 5×40 is a tiny image)::

    ConvEncoder:
        (B, 1, 5, 40)
        → Conv2d(1, 16, k=3, pad=1) → ReLU
        → Conv2d(16, 32, k=3, pad=1) → ReLU
        → Flatten → Linear(32*5*40 = 6400, 20) → ReLU
        → (B, 20)

    ConvDecoder (symmetric):
        (B, 20)
        → Linear(20, 32*5*40) → reshape (B, 32, 5, 40) → ReLU
        → ConvTranspose2d(32, 16, k=3, pad=1) → ReLU
        → ConvTranspose2d(16, 1, k=3, pad=1)
        → (B, 1, 5, 40)

Caveats:

* No pooling on the Doppler axis — only 5 bins, pooling would collapse it.
* No final non-linearity on the decoder — outputs reconstruct MinMax-scaled
  inputs in [0, 1] and the MSE loss handles the range naturally.
* Latent dim 20 to match the production encoder for drop-in comparability.

Reference: Wang et al. (2021). "GNSS-R Delay/Doppler Map Compression Method
Using a Denoising Convolutional Autoencoder." IGARSS 2021.
DOI:10.1109/GNSSR53802.2021.9617706.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """Conv2d encoder producing a ``latent_dim``-D bottleneck."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 40, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if x.dim() == 2 and x.shape[1] == 200:
            x = x.view(-1, 1, 5, 40)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(
                f"ConvEncoder expects (B, 200) or (B, 5, 40) or (B, 1, 5, 40); "
                f"got {tuple(x.shape)}"
            )
        h = self.conv(x)
        h = self.flatten(h)
        return self.fc(h)


class ConvDecoder(nn.Module):
    """Symmetric Conv2dTranspose decoder."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 32 * 5 * 40)
        self.deconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D401
        h = self.fc(z).view(-1, 32, 5, 40)
        return self.deconv(h)


class ConvAutoencoder(nn.Module):
    """Encoder + decoder pair, returning the reconstruction by default."""

    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        z = self.encoder(x)
        return self.decoder(z)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _self_test() -> None:
    """Forward + backward sanity test on synthetic input."""
    model = ConvAutoencoder(latent_dim=20)

    # Single sample, flat 200-D input
    x_flat = torch.randn(1, 200)
    z = model.encoder(x_flat)
    assert z.shape == (1, 20), f"encoder out {z.shape}"
    x_recon = model.decoder(z)
    assert x_recon.shape == (1, 1, 5, 40), f"decoder out {x_recon.shape}"

    # Batched forward + MSE backward
    batch = torch.randn(32, 200)
    recon = model(batch)
    assert recon.shape == (32, 1, 5, 40)
    loss = nn.MSELoss()(recon.view(32, -1), batch.view(32, -1))
    loss.backward()

    print("=== conv_encoder._self_test ===")
    print(f"  encoder params: {parameter_count(model.encoder):,}")
    print(f"  decoder params: {parameter_count(model.decoder):,}")
    print(f"  total params  : {parameter_count(model):,}")
    print(f"  MSE on random input: {loss.item():.4f}")
    print(f"  recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
    print("  ✓ forward + backward OK on (B=32, 200) random input")


if __name__ == "__main__":
    _self_test()
