from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - allow preprocessing without torch installed
    torch = None

if torch is None:
    ArrayLike = Union[Sequence[float], np.ndarray]
else:
    ArrayLike = Union[Sequence[float], np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Canonical mapping between action token IDs and continuous action values."""

    action_token_start: int = 31744
    action_token_end: int = 31999
    action_value_min: float = -1.0
    action_value_max: float = 1.0
    action_dim: int = 7
    mapping_style: str = "openvla_digitize"

    @property
    def num_bins(self) -> int:
        return int(self.action_token_end - self.action_token_start + 1)

    def validate(self) -> None:
        if self.action_token_end < self.action_token_start:
            raise ValueError(
                "action_token_end must be >= action_token_start, got "
                f"{self.action_token_end} < {self.action_token_start}."
            )
        if self.action_value_max <= self.action_value_min:
            raise ValueError(
                "action_value_max must be > action_value_min, got "
                f"{self.action_value_max} <= {self.action_value_min}."
            )
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.mapping_style not in ("openvla_digitize", "linear"):
            raise ValueError(
                "mapping_style must be one of {'openvla_digitize', 'linear'}, got "
                f"{self.mapping_style}."
            )
        if self.mapping_style == "openvla_digitize" and self.num_bins < 2:
            raise ValueError("openvla_digitize requires at least 2 bins.")

    def _bins_np(self) -> np.ndarray:
        return np.linspace(
            float(self.action_value_min),
            float(self.action_value_max),
            int(self.num_bins),
            dtype=np.float32,
        )

    def _bin_centers_np(self) -> np.ndarray:
        bins = self._bins_np()
        return (bins[:-1] + bins[1:]) / 2.0

    def looks_like_token_ids(self, values: ArrayLike, atol: float = 1e-3) -> bool:
        x = np.asarray(values, dtype=np.float64)
        if x.size == 0:
            return False
        rounded = np.rint(x)
        if not np.allclose(x, rounded, atol=atol, rtol=0.0):
            return False
        x_int = rounded.astype(np.int64)
        return bool(
            x_int.min() >= self.action_token_start and x_int.max() <= self.action_token_end
        )

    def token_ids_to_continuous(
        self, token_ids: ArrayLike, strict: bool = True
    ) -> ArrayLike:
        self.validate()
        if self.mapping_style == "linear":
            return self._token_ids_to_continuous_linear(token_ids=token_ids, strict=strict)
        return self._token_ids_to_continuous_openvla(token_ids=token_ids, strict=strict)

    def _token_ids_to_continuous_linear(
        self, token_ids: ArrayLike, strict: bool = True
    ) -> ArrayLike:
        span_ids = float(self.action_token_end - self.action_token_start)
        span_values = float(self.action_value_max - self.action_value_min)
        if torch is not None and torch.is_tensor(token_ids):
            x = token_ids.to(dtype=torch.float32)
            if strict:
                rounded = torch.round(x)
                if not torch.allclose(x, rounded, atol=1e-3, rtol=0.0):
                    raise ValueError("token_ids contains non-integer-like values.")
                x = rounded
                if (x < self.action_token_start).any() or (x > self.action_token_end).any():
                    raise ValueError(
                        "token_ids out of range: "
                        f"[{int(x.min().item())}, {int(x.max().item())}] not in "
                        f"[{self.action_token_start}, {self.action_token_end}]."
                    )
            else:
                x = torch.clamp(x, self.action_token_start, self.action_token_end)
            return self.action_value_min + ((x - self.action_token_start) / span_ids) * span_values

        x_np = np.asarray(token_ids, dtype=np.float64)
        if strict:
            rounded = np.rint(x_np)
            if not np.allclose(x_np, rounded, atol=1e-3, rtol=0.0):
                raise ValueError("token_ids contains non-integer-like values.")
            x_np = rounded
            if x_np.min() < self.action_token_start or x_np.max() > self.action_token_end:
                raise ValueError(
                    "token_ids out of range: "
                    f"[{int(x_np.min())}, {int(x_np.max())}] not in "
                    f"[{self.action_token_start}, {self.action_token_end}]."
                )
        else:
            x_np = np.clip(x_np, self.action_token_start, self.action_token_end)
        cont = self.action_value_min + ((x_np - self.action_token_start) / span_ids) * span_values
        if isinstance(token_ids, np.ndarray):
            return cont.astype(np.float32)
        return cont.astype(np.float32).tolist()

    def _token_ids_to_continuous_openvla(
        self, token_ids: ArrayLike, strict: bool = True
    ) -> ArrayLike:
        centers_np = self._bin_centers_np()
        max_center_idx = int(centers_np.shape[0] - 1)
        if torch is not None and torch.is_tensor(token_ids):
            x = token_ids.to(dtype=torch.float32)
            if strict:
                rounded = torch.round(x)
                if not torch.allclose(x, rounded, atol=1e-3, rtol=0.0):
                    raise ValueError("token_ids contains non-integer-like values.")
                x = rounded
                if (x < self.action_token_start).any() or (x > self.action_token_end).any():
                    raise ValueError(
                        "token_ids out of range: "
                        f"[{int(x.min().item())}, {int(x.max().item())}] not in "
                        f"[{self.action_token_start}, {self.action_token_end}]."
                    )
            else:
                x = torch.clamp(x, self.action_token_start, self.action_token_end)
            discretized = (self.action_token_end - x + 1.0)
            idx = torch.clamp(discretized - 1.0, 0.0, float(max_center_idx)).to(dtype=torch.long)
            centers_t = torch.as_tensor(centers_np, device=x.device, dtype=torch.float32)
            out = centers_t[idx]
            return out

        x_np = np.asarray(token_ids, dtype=np.float64)
        if strict:
            rounded = np.rint(x_np)
            if not np.allclose(x_np, rounded, atol=1e-3, rtol=0.0):
                raise ValueError("token_ids contains non-integer-like values.")
            x_np = rounded
            if x_np.min() < self.action_token_start or x_np.max() > self.action_token_end:
                raise ValueError(
                    "token_ids out of range: "
                    f"[{int(x_np.min())}, {int(x_np.max())}] not in "
                    f"[{self.action_token_start}, {self.action_token_end}]."
                )
        else:
            x_np = np.clip(x_np, self.action_token_start, self.action_token_end)

        discretized = self.action_token_end - x_np + 1.0
        idx = np.clip(discretized - 1.0, a_min=0.0, a_max=float(max_center_idx)).astype(np.int64)
        cont = centers_np[idx]
        if isinstance(token_ids, np.ndarray):
            return cont.astype(np.float32)
        return cont.astype(np.float32).tolist()

    def continuous_to_token_ids(self, actions: ArrayLike, clamp: bool = True) -> ArrayLike:
        self.validate()
        if self.mapping_style == "linear":
            return self._continuous_to_token_ids_linear(actions=actions, clamp=clamp)
        return self._continuous_to_token_ids_openvla(actions=actions, clamp=clamp)

    def _continuous_to_token_ids_linear(
        self, actions: ArrayLike, clamp: bool = True
    ) -> ArrayLike:
        span_ids = float(self.action_token_end - self.action_token_start)
        span_values = float(self.action_value_max - self.action_value_min)
        if torch is not None and torch.is_tensor(actions):
            x = actions.to(dtype=torch.float32)
            if clamp:
                x = torch.clamp(x, self.action_value_min, self.action_value_max)
            ratio = (x - self.action_value_min) / span_values
            ids = torch.round(self.action_token_start + ratio * span_ids).to(dtype=torch.long)
            if clamp:
                ids = torch.clamp(ids, self.action_token_start, self.action_token_end)
            return ids

        x_np = np.asarray(actions, dtype=np.float64)
        if clamp:
            x_np = np.clip(x_np, self.action_value_min, self.action_value_max)
        ratio = (x_np - self.action_value_min) / span_values
        ids = np.rint(self.action_token_start + ratio * span_ids).astype(np.int64)
        if clamp:
            ids = np.clip(ids, self.action_token_start, self.action_token_end)
        if isinstance(actions, np.ndarray):
            return ids
        return ids.tolist()

    def _continuous_to_token_ids_openvla(
        self, actions: ArrayLike, clamp: bool = True
    ) -> ArrayLike:
        bins_np = self._bins_np()
        if torch is not None and torch.is_tensor(actions):
            x = actions.to(dtype=torch.float32)
            if clamp:
                x = torch.clamp(x, self.action_value_min, self.action_value_max)
            bins_t = torch.as_tensor(bins_np, device=x.device, dtype=torch.float32)
            # Align with np.digitize(..., right=False), including edge handling.
            discretized = torch.bucketize(x, bins_t, right=False)
            discretized = torch.where(
                x <= bins_t[0],
                torch.ones_like(discretized),
                discretized,
            )
            discretized = torch.where(
                x >= bins_t[-1],
                torch.full_like(discretized, int(self.num_bins)),
                discretized,
            )
            ids = (self.action_token_end - discretized + 1).to(dtype=torch.long)
            if clamp:
                ids = torch.clamp(ids, self.action_token_start, self.action_token_end)
            return ids

        x_np = np.asarray(actions, dtype=np.float64)
        if clamp:
            x_np = np.clip(x_np, self.action_value_min, self.action_value_max)
        discretized = np.digitize(x_np, bins_np)
        ids = (self.action_token_end - discretized + 1).astype(np.int64)
        if clamp:
            ids = np.clip(ids, self.action_token_start, self.action_token_end)
        if isinstance(actions, np.ndarray):
            return ids
        return ids.tolist()
