"""Exact per-generation band scoring for the GA (gate 2, increments 1–2).

The GA fitness (`genetic_global._obj_viol` and the lock-stepped `numba_kernels._fused_eval`)
adds a per-MID month-band VIOLATION using a crude volume-ratio PROXY:

    proj_proxy(mid, month) = bval × (MID_volume(candidate) / MID_baseline_volume)
    viol += _pen( max(proj/ceil − 1, 0) ) · wm[mid] · pmul      (+ the floor side)

This module replaces `proj_proxy` with the EXACT pro-rata projection (via the validated,
numba-accelerated `PopulationBandProjector.project_pop_numba`), keeping the penalty SHAPE
(`_pen`: fixed hit + quadratic/exponential, tolerance dust-guard) and weights (`wm[mid] · pmul`)
byte-identical to `_obj_viol`. It is computed ONCE per generation for the whole population.

ANALOGY: a "band" is a speed limit for a MID in a given month. The old proxy guessed how fast
you were going from a rough rule of thumb; this module reads the exact speedometer (the true
pro-rata projection) instead — but keeps the SAME fine schedule (a flat penalty the instant you
cross the limit, plus a fast-growing surcharge the further over you are).

Two pieces, both pure-NumPy/scipy and fully unit-testable off the live pipeline:

  1. `build_col_incidence` / `shares_to_prop_raw` — aggregate the GA's per-column decoded shares
     (N = cell×gateway rows) onto the projector's prop-keys ((cur,bank,[rpgt],vampMid)),
     i.e. the same grouping `_prop_items_from_gran` does, as a sparse (K×N) matmul.
  2. `ExactBandPenalty` — from per-band specs (midl, months, metric, ceil, floor, weight) it
     projects the population and returns the exact band violation per candidate, using the
     SAME `_pen` as `_obj_viol`.

This wiring is now LIVE (under `ctx['exact_bands']`): it removes the proxy band term from the
kernel + `_obj_viol` and adds `ExactBandPenalty.penalty(prop_raw)` to `viol` per generation.
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


# [FN-026]
def build_col_incidence(col_propkeys: Sequence[str], prop_keys: Sequence[str]):
    """Build the sparse (K × N) 0/1 lookup that rolls GA columns up to projector rows.

    Each GA share column j belongs to exactly one prop-key i (its cur|bank|[rpgt]|MID bucket).
    This returns a matrix with a single 1 at (i, j) for every column, so that
    `prop_raw = (incidence @ shares.T).T` sums each candidate's gateway shares into their MID
    buckets in one matmul — like a spreadsheet that totals individual receipts into per-category
    columns. Columns whose prop-key isn't in the projector (e.g. a gateway with no vampMid) are
    simply left out, exactly like `_prop_items_from_gran.dropna`.

    col_propkeys : (N,) prop-key string per GA share column, built with the SAME rule as
                   `PopulationBandProjector.prop_keys` ("cur|bank|mid" or "cur|bank|rpgt|mid",
                   stripped/lower-cased on cur & mid).
    prop_keys    : projector.prop_keys (ordered).
    """
    key_to_row = {str(key): row for row, key in enumerate(prop_keys)}
    row_count = max(len(prop_keys), 1)
    col_count = max(len(col_propkeys), 1)
    row_idx = []
    col_idx = []
    for col, prop_key in enumerate(col_propkeys):
        row = key_to_row.get(str(prop_key))
        if row is not None:
            row_idx.append(row)
            col_idx.append(col)
    ones = np.ones(len(row_idx), dtype=float)
    if _HAVE_SCIPY:
        return _sp.csr_matrix((ones, (row_idx, col_idx)), shape=(row_count, col_count))
    dense_matrix = np.zeros((row_count, col_count), dtype=float)
    dense_matrix[np.asarray(row_idx, int), np.asarray(col_idx, int)] = 1.0
    return dense_matrix


# [FN-027]
def shares_to_prop_raw(shares: np.ndarray, incidence) -> np.ndarray:
    """(P, N) decoded shares → (P, K) prop_raw = (incidence @ shares.T).T (sparse-safe).

    Rolls each candidate's per-gateway shares up to its per-MID-bucket totals using the
    incidence lookup built above. Accepts a single share vector or a whole population.
    """
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
    `bval × volume-ratio`. In effect it projects the whole population once, then for every
    band checks how far each candidate is over (ceil) or under (floor) the limit and charges
    the fine.
    """

    # [FN-028]
    def __init__(self, projector, specs: Sequence[BandSpec], *,
                 breach_fixed: float = 0.0, breach_quad: float = 1.0,
                 breach_shape: str = "quadratic", use_numba: bool = True):
        self.projector = projector
        self.specs = list(specs)
        # `breach_fixed` = the flat penalty the instant a band is crossed;
        # `breach_quad`  = the strength of the growing surcharge beyond that;
        # `penalty_is_exponential` picks the exponential surcharge instead of quadratic.
        self.breach_fixed = float(breach_fixed or 0.0)
        self.breach_quad = float(breach_quad or 1.0)
        self.penalty_is_exponential = (str(breach_shape).lower() == "exponential")
        self.use_numba = bool(use_numba)
        # Map each (midl, period) band to its column in the projector's output.
        self._band_to_index = {band: i for i, band in enumerate(projector.band_order)}

    # [FN-029]
    def _pen(self, overshoot):
        """The fine schedule for being `overshoot` fraction over a band (0 = at/under the limit).

        IDENTICAL to genetic_global._obj_viol._pen: a tiny dust-guard (ignore sub-1e-9 rounding),
        then a flat hit the moment you cross plus a smoothly-growing surcharge (quadratic by
        default, exponential if selected). Kept byte-identical so the GA scores the same.
        """
        overshoot = np.where(overshoot > 1e-9, overshoot, 0.0)
        if self.penalty_is_exponential:
            return self.breach_fixed * (overshoot > 0.0) + self.breach_quad * (np.exp(np.minimum(overshoot, 50.0)) - 1.0)
        return self.breach_fixed * (overshoot > 0.0) + self.breach_quad * overshoot * overshoot

    # [FN-030]
    def project(self, prop_raw):
        """Project the population to exact per-band VAMP/Txn values (numba path by default)."""
        if self.use_numba:
            return self.projector.project_pop_numba(prop_raw)
        return self.projector.project_pop(prop_raw)

    # [FN-031]
    def penalty(self, prop_raw) -> np.ndarray:
        """(P, K) prop_raw → (P,) total band violation to add to each candidate's `viol`."""
        prop_raw = np.ascontiguousarray(prop_raw, dtype=float)
        if prop_raw.ndim == 1:
            prop_raw = prop_raw[None, :]
        candidate_count = prop_raw.shape[0]
        vamp, txn = self.project(prop_raw)                      # (P, B) each
        penalties = np.zeros(candidate_count, dtype=float)
        for spec in self.specs:
            metric_values = txn if spec.metric == "txn" else vamp
            # Sum the metric across the band's months for every candidate.
            band_total = np.zeros(candidate_count, dtype=float)
            for month in spec.months:
                band_col = self._band_to_index.get((spec.midl, int(month)))
                if band_col is not None:
                    band_total += metric_values[:, band_col]
            # Over-the-ceiling side and under-the-floor side, each scaled by the band weight.
            if spec.ceil is not None:
                penalties += self._pen(np.maximum(band_total / max(float(spec.ceil), 1e-9) - 1.0, 0.0)) * spec.weight
            if spec.floor is not None and float(spec.floor) > 0:
                penalties += self._pen(np.maximum(1.0 - band_total / max(float(spec.floor), 1e-9), 0.0)) * spec.weight
        return penalties

    # [FN-032]
    def report(self, prop_raw) -> list:
        """Per-band projected value ('Now') for a single candidate (first of the population).

        Read-only counterpart to `penalty()` — the SAME exact projection, but returns the raw
        projected metric per band instead of the fine, so callers (e.g. the run-log breakdown)
        can show target vs Now per constraint that reconciles with the search's own breach.
        Returns a list aligned with `self.specs`: dicts of midl / months / metric / ceil / floor /
        now."""
        prop_raw = np.ascontiguousarray(prop_raw, dtype=float)
        if prop_raw.ndim == 1:
            prop_raw = prop_raw[None, :]
        vamp, txn = self.project(prop_raw)
        rows = []
        for spec in self.specs:
            metric_values = txn if spec.metric == "txn" else vamp
            band_total = 0.0
            for month in spec.months:
                band_col = self._band_to_index.get((spec.midl, int(month)))
                if band_col is not None:
                    band_total += float(metric_values[0, band_col])
            rows.append({"midl": spec.midl, "months": tuple(int(m) for m in spec.months),
                         "metric": spec.metric,
                         "ceil": (None if spec.ceil is None else float(spec.ceil)),
                         "floor": (None if spec.floor is None else float(spec.floor)),
                         "now": band_total})
        return rows
