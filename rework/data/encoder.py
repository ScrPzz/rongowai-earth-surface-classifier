"""Production autoencoder: 200→100→20 MLP encoder + symmetric decoder.

Vendored from the ``Encoder``/``Decoder`` definitions in
``deliver/[ETL]geok_compressed_dataset_generator_encoders.ipynb``. The class
architecture is byte-identical so the existing trained weights at
``geo_k/raw_counts/encoder_enh.pth`` can be loaded.

Normalisation contract (replicated from production ``DDMProcessor``):
    encoded_row = encoder( MinMaxScaler.fit_transform( raw_ddm_row * 1e13 ) )

NB: production fits a fresh MinMaxScaler **per file**, not globally. We keep
that quirk for baseline fidelity; correcting it is a candidate Phase-1 item.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    """200-D → 20-D MLP encoder, ReLU between layers and at the output."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class Decoder(nn.Module):
    """20-D → 200-D MLP decoder, symmetric to :class:`Encoder` (no final ReLU)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def _ensure_pickle_path() -> None:
    """Allow ``torch.load`` of pickled ``Encoder`` instances saved under __main__.

    The notebook that produced ``encoder_enh.pth`` pickled the model in the
    ``__main__`` module context. To deserialize that file outside the
    notebook, we expose this module's :class:`Encoder` / :class:`Decoder`
    under ``__main__`` so pickle can resolve them.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    if not hasattr(main_mod, "Encoder"):
        main_mod.Encoder = Encoder
    if not hasattr(main_mod, "Decoder"):
        main_mod.Decoder = Decoder


def load_encoder(
    model_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Encoder:
    """Load encoder weights from disk, supporting both state-dict and pickled-model formats.

    Tries state-dict first (cleaner / safer). Falls back to pickled-model
    deserialisation if needed, registering :class:`Encoder` under ``__main__``
    so the original notebook-pickled .pth files can still be opened.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Encoder weights file not found: {model_path}")

    device = torch.device(device)
    encoder = Encoder()

    try:
        # weights_only=True works for state-dicts and refuses to execute pickle code.
        state = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(state, dict):
            encoder.load_state_dict(state)
        else:
            raise TypeError("loaded object is not a state dict")
    except Exception:
        # Fallback: full pickled model. Register classes under __main__ so the
        # unpickler can find them, then extract state.
        _ensure_pickle_path()
        loaded = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(loaded, "state_dict"):
            encoder.load_state_dict(loaded.state_dict())
        elif isinstance(loaded, dict):
            encoder.load_state_dict(loaded)
        else:
            raise TypeError(
                f"could not extract Encoder state from {model_path}: "
                f"got object of type {type(loaded).__name__}"
            )

    encoder.to(device)
    encoder.eval()
    return encoder


def compress_array(
    raw_ddm: np.ndarray,
    encoder: Encoder,
    *,
    fit_scaler_per_call: bool = True,
    batch_size: int = 32,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Normalize and compress a (N, 200) raw_ddm array to (N, 20) latent codes.

    Replicates the production normalisation: ``raw * 1e13 → MinMaxScaler.fit_transform``.

    Parameters
    ----------
    raw_ddm
        Float array shaped (N, 200) — flattened 5×40 DDMs.
    encoder
        A loaded :class:`Encoder` instance (already on ``device``).
    fit_scaler_per_call
        If ``True`` (production default), fits a fresh MinMaxScaler on the
        input. If ``False``, the caller must have already scaled the data to
        [0, 1]; the input is passed straight to the encoder.
    batch_size
        Batch size for the encoder forward pass. Default matches production.
    device
        Torch device for inference.

    Returns
    -------
    np.ndarray
        Encoded codes shaped (N, 20), dtype float32.
    """
    if raw_ddm.ndim != 2 or raw_ddm.shape[1] != 200:
        raise ValueError(
            f"raw_ddm must be (N, 200); got shape {raw_ddm.shape}"
        )

    if fit_scaler_per_call:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw_ddm.astype(np.float64) * 1e13)
    else:
        scaled = raw_ddm

    device = torch.device(device)
    tensor = torch.from_numpy(np.asarray(scaled, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(tensor), batch_size=batch_size, shuffle=False
    )

    encoder.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            chunks.append(encoder(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)
