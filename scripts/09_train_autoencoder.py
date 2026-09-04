"""Step 9 (optional): learn a 20-D latent representation of the raw pixels.

Trains an MLP autoencoder on train-pool pixels, encodes every row and writes
``data/latent.parquet`` keyed by (flight_idx, epoch, channel). The latent
vector is then available to the ablation script as the ``latent`` set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from _common import Timer, base_parser, load_config, save_json

from rongowai_ddm.data.dataset import load_samples, raw_pixel_matrix
from rongowai_ddm.data.splits import SPLIT_TRAIN
from rongowai_ddm.models.autoencoder import encode, reconstruct, train_autoencoder

KEYS = ["flight_idx", "epoch", "channel"]


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--latent", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args)
    fl = pd.read_parquet(cfg.flight_table_path)

    df = load_samples(cfg.samples_dir, groups=("raw",), flight_table=fl, include_analysis=False)
    X = raw_pixel_matrix(df)
    is_train = (df["split"] == SPLIT_TRAIN).to_numpy()
    fold = df["fold_id"].to_numpy()
    rng = np.random.default_rng(cfg.split.seed)
    tr_idx = np.flatnonzero(is_train & (fold != 0))
    va_idx = np.flatnonzero(is_train & (fold == 0))
    tr_idx = rng.choice(tr_idx, min(args.n_train, len(tr_idx)), replace=False)
    va_idx = rng.choice(va_idx, min(100_000, len(va_idx)), replace=False)
    with Timer("autoencoder"):
        model, history = train_autoencoder(
            X[tr_idx],
            X[va_idx],
            latent=args.latent,
            epochs=args.epochs,
            device=args.device,
            seed=cfg.split.seed,
        )
    Z = encode(model, X, device=args.device)
    rec = reconstruct(model, X[va_idx[:20_000]])
    rel_err = float(
        np.linalg.norm(rec - X[va_idx[:20_000]], axis=1).mean()
        / np.linalg.norm(X[va_idx[:20_000]], axis=1).mean()
    )
    lat = df[KEYS].copy()
    for j in range(Z.shape[1]):
        lat[f"latent_{j:02d}"] = Z[:, j]
    lat.to_parquet(cfg.work_dir / "data" / "latent.parquet", index=False)
    torch.save(model.state_dict(), cfg.results_dir / "autoencoder.pt")
    save_json(
        {
            "latent": args.latent,
            "epochs": args.epochs,
            "n_train": int(len(tr_idx)),
            "history": history,
            "val_relative_reconstruction_error": rel_err,
        },
        cfg.results_dir / "autoencoder_report.json",
    )
    print(f"validation MSE {history['val'][-1]:.5f}, relative reconstruction error {rel_err:.3f}")


if __name__ == "__main__":
    main()
