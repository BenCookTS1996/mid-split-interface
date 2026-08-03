"""Exact per-generation band scoring for the GA (gate 2, increments 1–2).

The GA fitness (`genetic_global._obj_viol` and the lock-stepped `numba_kernels._fused_eval`)
adds a per-MID month-band VIOLATION using a crude volume-ratio PROXY:

    proj_proxy(mid, month) = bval × (MID_volume(candidate) / MID_baseline_volume)
    viol += _pen( max(proj/ceil − 1, 0) ) · wm[mid] · pmul      (+ the floor side)

This module replaces `proj_proxy` with the EXACT pro-rata projection (via the validated,
numba-accelerated `PopulationBandProjector.project_pop_numba`), keeping the penalty SHAPE
(`_pen`: fixed hit + quadratic/exponential, tolerance dust-guard) and weights (`wm[mid] · pmul`)
byte-identical to `_obj_viol`. It is computed ONCE per generation for the whole population.

Two pieces, both pure-NumPy/scipy and fully unit-testable off the live pipeline:

  1. `build_col_incidence` / `shares_to_prop_raw` — aggregate the GA's per-column decoded shares
     (N = cell×gateway rows) onto the projector's prop-keys ((cur,bank,[rpgt],vampMid)),
     i.e. the same grouping `_prop_items_from_gran` does, as a sparse (K×N) matmul.
  2. `ExactBandPenalty` — from per-band specs (midl, months, metric, ceil, floor, weight) it
     projects the population and returns the exact band violation per candidate, using the
     SAME `_pen` as `_obj_viol`.

Wiring (increment 3, separate) removes the proxy band term from the kernel + `_obj_viol` and
adds `ExactBandPenalty.penalty(prop_raw)` to `viol` per generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import scipy.sparse as _sp
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False


def build_col_incidence(col_propkeys: Sequence[str], prop_keys: Sequence[str]):
    """Sparse (K × N) 0/1 incidence mapping each GA column j → its prop-key row i, so
    `prop_raw = (incidence @ shares.T).T`. Columns whose prop-key isn't in the projector
    (e.g. a gateway with no vampMid) are dropped, exactly like `_prop_items_from_gran.dropna`.

    col_propkeys : (N,) prop-key string per GA share column, built with the SAME rule as
                   `PopulationBandProjector.prop_keys` ("cur|bank|mid" or "cur|bank|rpgt|mid",
                   stripped/lower-cased on cur & mid).
    prop_keys    : projector.prop_keys (ordered).
    """
    kpos = {str(k): i for i, k in enumerate(prop_keys)}
    K = max(len(prop_keys), 1)
    N = max(len(col_propkeys), 1)
    rows = []
    cols = []
    for j, pk in enumerate(col_propkeys):
        i = kpos.get(str(pk))
        if i is not None:
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=float)
    if _HAVE_SCIPY:
        return _sp.csr_matrix((data, (rows, cols)), shape=(K, N))
    dense = np.zeros((K, N), dtype=float)
    dense[np.asarray(rows, int), np.asarray(cols, int)] = 1.0
    return dense


def shares_to_prop_raw(shares: np.ndarray, incidence) -> np.ndarray:
    """(P, N) decoded shares → (P, K) prop_raw = (incidence @ shares.T).T (sparse-safe)."""
    shares = np.ascontiguousarray(shares, dtype=float)
    if shares.ndim == 1:
        shares = shares[None, :]
    return np.asarray((incidence @ shares.T).T)


@dataclass
class BandSpec:
    """One GA band, in projector coordinates. `weight` = wm[mid] · pmul (priority × volume),
    matching `_obj_viol`'s `_wm[_mi] * _pmul`. `months` are the periods summed for the metric."""
    midl: str
    months: tuple
    metric: str                 # "vamp" or "txn"
    ceil: Optional[float]
    floor: Optional[float]
    weight: float


class ExactBandPenalty:
    """Exact per-generation band violation, drop-in for the proxy band term in `_obj_viol`.

    `penalty(prop_raw)` → (P,) violation to ADD to `viol`, using the SAME `_pen` shape and
    `weight` (wm·pmul) as `_obj_viol`, but with the EXACT projected band value instead of
    `bval × volume-ratio`.
    """

    def __init__(self, projector, specs: Sequence[BandSpec], *,
                 breach_fixed: float = 0.0, breach_quad: float = 1.0,
                 breach_shape: str = "quadratic", use_numba: bool = True):
        self.projector = projector
        self.specs = list(specs)
        self.bfix = float(breach_fixed or 0.0)
        self.qwt = float(breach_quad or 1.0)
        self.pexp = (str(breach_shape).lower() == "exponential")
        self.use_numba = bool(use_numba)
        self._bo = {b: i for i, b in enumerate(projector.band_order)}

    def _pen(self, ov):
        # IDENTICAL to genetic_global._obj_viol._pen (tolerance dust-guard + fixed + smooth).
        ov = np.where(ov > 1e-9, ov, 0.0)
        if self.pexp:
            return self.bfix * (ov > 0.0) + self.qwt * (np.exp(np.minimum(ov, 50.0)) - 1.0)
        return self.bfix * (ov > 0.0) + self.qwt * ov * ov

    def project(self, prop_raw):
        if self.use_numba:
            return self.projector.project_pop_numba(prop_raw)
        return self.projector.project_pop(prop_raw)

    def penalty(self, prop_raw) -> np.ndarray:
        prop_raw = np.ascontiguousarray(prop_raw, dtype=float)
        if prop_raw.ndim == 1:
            prop_raw = prop_raw[None, :]
        P = prop_raw.shape[0]
        vamp, txn = self.project(prop_raw)                      # (P, B) each
        out = np.zeros(P, dtype=float)
        for s in self.specs:
            mat = txn if s.metric == "txn" else vamp
            val = np.zeros(P, dtype=float)
            for m in s.months:
                col = self._bo.get((s.midl, int(m)))
                if col is not None:
                    val += mat[:, col]
            if s.ceil is not None:
                out += self._pen(np.maximum(val / max(float(s.ceil), 1e-9) - 1.0, 0.0)) * s.weight
            if s.floor is not None and float(s.floor) > 0:
                out += self._pen(np.maximum(1.0 - val / max(float(s.floor), 1e-9), 0.0)) * s.weight
        return out
