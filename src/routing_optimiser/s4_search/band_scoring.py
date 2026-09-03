"""Exact per-generation band scoring for the GA (gate 2, increments 1–2).

The GA fitness (`seed_search._obj_viol` and the lock-stepped `numba_kernels._fused_eval`)
adds a per-MID month-band VIOLATION. NOTE (2026-08-19bb): the paragraph below describes the
volume-ratio PROXY this module was built to REPLACE — it is history, not what runs. The only
penalty class here is `ExactBandPenalty`, the full-matrix GA is called with
`band_penalty_fn=ExactBandPenalty.penalty` and WITHOUT the `mid_bands` proxy hook, and the run
log's five-rung chain reads identically at every rung with RECONCILIATION ERROR 0 — which a proxy
could not produce. The proxy text is kept only to explain what the exact scorer superseded:

    proj_proxy(mid, month) = bval × (MID_volume(candidate) / MID_baseline_volume)
    viol += _pen( max(proj/ceil − 1, 0) ) · wm[mid] · pmul      (+ the floor side)

This module replaces `proj_proxy` with the EXACT pro-rata projection (via the validated,
numba-accelerated `PopulationBandProjector.project_pop_numba`), keeping the penalty SHAPE
(`_pen`: fixed hit + quadratic/exponential, tolerance dust-guard) byte-identical to
`_obj_viol`. It is computed ONCE per generation for the whole population.

WEIGHTS: `BandSpec.weight` is the PRIORITY multiplier ALONE — `GAP**(1-priority)` with GAP=8, so
prio-1 = 1.0, prio-2 = 0.125, prio-3 = 0.015625. Corrected 2026-08-19aa: this file previously said
`wm[mid] · pmul (priority × volume)` in two places, but the live construction site
(tab_2_routing_engine.py, `_specs.append(_BSpec(..., weight=float(_pmulx)))`) passes the priority
multiplier only. The volume term `wm[mid]` IS computed on the PROXY band path (`_vmul` alongside
`_pmul`) and is simply not carried onto the exact path. Whether that omission was deliberate is
not recorded anywhere; this docstring now describes what the code does, and changing the weighting
itself would change the search and is a separate decision.

ANALOGY: a "band" is a speed limit for a MID in a given month. The old proxy guessed how fast
you were going from a rough rule of thumb; this module reads the exact speedometer (the true
pro-rata projection) instead — but keeps the SAME fine schedule (a flat penalty the instant you
cross the limit, plus a fast-growing surcharge the further over you are).

Two pieces, both pure-NumPy/scipy and fully unit-testable off the live pipeline:

  1. `build_col_incidence` / `shares_to_prop_raw` — aggregate the GA's per-column decoded shares
     (N = profile×gateway rows) onto the projector's prop-keys ((cur,bank,[rpgt],vampMid)),
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

# scipy is a HARD requirement, not an optional accelerator (2026-08-19aa). It used to be
# wrapped in try/except with a dense NumPy fallback in `build_col_incidence`; at the live scaffold
# size that fallback allocates a 212,557 × 245,409 float64 matrix = ~417 GB, so it never actually
# degraded gracefully — it died with a MemoryError far from the cause, or (on a small enough
# problem) silently ran a different code path. An ImportError here names the real problem on line
# one. Ben's call, 2026-08-19aa: if it fails, crash.
import scipy.sparse as _sp
import time as _bs_time

# ── 19jg: WHERE THE BAND-PENALTY ROW ACTUALLY GOES ───────────────────────────────────────
# [eval-cost] charges "band projection + penalty" 418.5 ms/call - the largest single row in
# the whole search. Then 19jd measured the KERNEL at 139.6 ms and [lift-ab] the whole
# projection at 166.5 ms, so SIXTY PERCENT of that row is not the projection at all, and
# nothing had ever measured it. These three counters split what `penalty` itself does; tab_2
# adds the `shares -> prop_raw` step in front of them and prints [bpf-inside].
#
# Timers only. Nothing computed here moved, and three perf_counter calls against a 400 ms
# body are not measurable.
_BP_T = {"contig": 0.0, "project": 0.0, "specs": 0.0, "n": 0}


def bpf_timing():
    """The [bpf-inside] counters, as a plain dict. Read by tab_2 at the end of the search."""
    return dict(_BP_T)


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
    # ALWAYS sparse as of 2026-08-19aa. The dense fallback that used to sit here is deleted, not
    # merely unreachable: at the live size it was a ~417 GB allocation masquerading as a graceful
    # degradation.
    return _sp.csr_matrix((ones, (row_idx, col_idx)), shape=(row_count, col_count))


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
    """One GA band, in projector coordinates. `months` are the periods summed for the metric.

    `weight` = the PRIORITY multiplier only (GAP**(1-priority), GAP=8: prio-1 1.0, prio-2 0.125,
    prio-3 0.015625). NOT priority × volume — see the module docstring. `_obj_viol` on the proxy
    path uses `_wm[_mi] * _pmul` (volume × priority); the exact path passes `_pmul` alone."""
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
        self._pr_buf = None        # 19jk: the reused C-contiguous prop_raw buffer
        self._pr_stat = {"copied": 0, "passthru": 0, "alloc": 0}

    # [FN-029]
    def _pen(self, overshoot):
        """The fine schedule for being `overshoot` fraction over a band (0 = at/under the limit).

        IDENTICAL to seed_search._obj_viol._pen: a tiny dust-guard (ignore sub-1e-9 rounding),
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

    # [FN-030b]
    def _contig(self, prop_raw):
        """`np.ascontiguousarray(prop_raw, dtype=float)`, into a buffer this instance reuses.

        WHY (19jk). [bpf-inside] on the 21:57 run put this ONE LINE at 148 ms per call and
        57.6s of the run - a quarter of the band row, and it computes nothing. The caller hands
        over `(incidence @ shares.T).T`, which is a TRANSPOSED VIEW, so making it C-contiguous
        is a full-width strided rewrite of a (35 x 323,063) array. 90 MB, 338 times, and every
        one of them was a fresh allocation whose pages had to be faulted in before they could
        be written. This does not remove the rewrite - that needs the array built the other way
        round in the first place, which changes the order the sparse matmul sums in and is a
        separate, riskier change. It removes the ALLOCATION: ~30 GB of churn over a run.

        MEASURED AT THE LIVE SHAPE before shipping, so the expectation is on record rather than
        implied. 35 x 323,063, the same transposed view, best of 25 after 5 warm-ups:

            np.ascontiguousarray (allocates)   177.4 ms min / 207.7 ms mean
            np.copyto into a reused buffer     145.2 ms min / 159.9 ms mean
                                            => 1.22x on the min, 1.30x on the mean

        So this is worth roughly a quarter of the row, not the whole of it - about 14s of the
        21:57 run's 57.6s. The other 145 ms IS the transpose, and only route two removes that.

        A COPY IS A COPY. `np.copyto` moves the same values in the same order into memory of
        the same dtype; there is no arithmetic here to reassociate, so this is bit-identical by
        construction rather than by measurement. The test asserts it anyway, on F-order,
        C-order, 1-D, float32 and integer inputs.

        AN ALREADY-C-CONTIGUOUS float64 ARRAY IS RETURNED UNTOUCHED, which is exactly what
        `np.ascontiguousarray` did with it - so a caller that already hands over the right
        layout still pays nothing, and the `passthru` counter says how often that happens.

        THE BUFFER ESCAPES INTO `project`, and that is safe but worth stating. `project_pop_numba`
        keeps a REFERENCE to its proposal for [lift-ab] and [proj-inside] to replay; that
        reference now points at this buffer, so it holds the LAST call's values. It always did -
        `_stash_ab` is called on every projection - so "the last real proposal" means the same
        thing it did before. The buffer is per-instance and `penalty` is not called
        concurrently: rowpar threads the delivery transform, not this, and the projector's
        parallelism is inside the numba kernel."""
        _a = np.asarray(prop_raw)
        if _a.dtype == np.float64 and _a.flags["C_CONTIGUOUS"]:
            self._pr_stat["passthru"] += 1
            return _a
        _b = self._pr_buf
        if _b is None or _b.shape != _a.shape:
            _b = self._pr_buf = np.empty(_a.shape, np.float64)
            self._pr_stat["alloc"] += 1
        np.copyto(_b, _a)
        self._pr_stat["copied"] += 1
        return _b

    # [FN-031]
    def penalty(self, prop_raw, detail_out=None) -> np.ndarray:
        """(P, K) prop_raw → (P,) total band violation to add to each candidate's `viol`.

        `detail_out`: optional dict. When given, it receives
            detail_out["per_spec"] : (P, n_specs) float — this spec's WEIGHTED penalty per
                                     candidate, i.e. the exact quantity summed into the return
                                     value, split out instead of discarded.
            detail_out["specs"]    : the spec list, so a caller can map columns → midl.
        Added 2026-08-19ab for BREACH-TARGETED MUTATION: the GA needs to know WHICH bands are
        still breached in order to aim mutation at the profiles feeding them, and this loop already
        computes exactly that before throwing it away. The projection is the expensive part and has
        already run, so the detail is free.

        BIT-IDENTICAL to the pre-19ab total: each `+=` into `penalties` adds the SAME expression it
        added before (stored in a temp and reused for the detail, never recomputed and never
        reassociated). Do not "tidy" this into a single accumulation — the two separate `+=` are
        load-bearing for reproducing earlier runs."""
        _bp_t0 = _bs_time.perf_counter()
        prop_raw = self._contig(prop_raw)
        if prop_raw.ndim == 1:
            prop_raw = prop_raw[None, :]
        candidate_count = prop_raw.shape[0]
        _bp_t1 = _bs_time.perf_counter()
        vamp, txn = self.project(prop_raw)                      # (P, B) each
        _bp_t2 = _bs_time.perf_counter()
        penalties = np.zeros(candidate_count, dtype=float)
        per_spec = (np.zeros((candidate_count, len(self.specs)), dtype=float)
                    if detail_out is not None else None)
        for spec_index, spec in enumerate(self.specs):
            metric_values = txn if spec.metric == "txn" else vamp
            # Sum the metric across the band's months for every candidate.
            band_total = np.zeros(candidate_count, dtype=float)
            for month in spec.months:
                band_col = self._band_to_index.get((spec.midl, int(month)))
                if band_col is not None:
                    band_total += metric_values[:, band_col]
            # Over-the-ceiling side and under-the-floor side, each scaled by the band weight.
            if spec.ceil is not None:
                _over = self._pen(np.maximum(band_total / max(float(spec.ceil), 1e-9) - 1.0, 0.0)) * spec.weight
                penalties += _over
                if per_spec is not None:
                    per_spec[:, spec_index] += _over
            if spec.floor is not None and float(spec.floor) > 0:
                _under = self._pen(np.maximum(1.0 - band_total / max(float(spec.floor), 1e-9), 0.0)) * spec.weight
                penalties += _under
                if per_spec is not None:
                    per_spec[:, spec_index] += _under
        if detail_out is not None:
            detail_out["per_spec"] = per_spec
            detail_out["specs"] = self.specs
        # 19jg: `contig` is the C-contiguous copy of the (P, K) prop array the caller just
        # built - K is the prop-key count, not the row count, so this is a full-width copy of
        # a transposed sparse-matmul result and is a candidate for the missing 60%.
        _BP_T["contig"] += _bp_t1 - _bp_t0
        _BP_T["project"] += _bp_t2 - _bp_t1
        _BP_T["specs"] += _bs_time.perf_counter() - _bp_t2
        _BP_T["n"] += 1
        return penalties

    # [FN-032]
    def report(self, prop_raw) -> list:
        """Per-band projected value ('Now') for a single candidate (first of the population).

        Read-only counterpart to `penalty()` — the SAME exact projection, but returns the raw
        projected metric per band instead of the fine, so callers (e.g. the run-log breakdown)
        can show target vs Now per constraint that reconciles with the search's own breach.
        Returns a list aligned with `self.specs`: dicts of midl / months / metric / ceil / floor /
        now."""
        # 19jk: the same reused buffer as `penalty`. The two are called back-to-back by the
        # progress heartbeat and each consumes its result before the other runs, so sharing one
        # buffer cannot interleave; `report` is ~48 calls a run, but they are 148 ms each.
        prop_raw = self._contig(prop_raw)
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
