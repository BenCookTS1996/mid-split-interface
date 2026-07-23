"""
Genetic-algorithm router — CROSS-CELL per-vampMid tilt search.

`run_midtilt_ga` is the live entry point. Its genome is ONE tilt θ_m per vampMid
(~20 numbers), which shifts that MID's volume from its HIGH-risk cells toward its
LOW-risk cells — directly controlling the per-vampMid CROSS-cell VAMP rate (the
actual constraint) at a tiny, fast search dimension. The split is decoded from the
revenue reference as  share_g ∝ ref_g · exp(−θ_{mid(g)} · z_g)  (z_g = risk
standardised WITHIN the MID), renormalised per cell so freed volume redistributes
in proportion to the revenue reference (revenue-efficient recipients).

Fitness (maximised) = expected_revenue − λ · risk_penalty (per-vampMid aggregate
VAMP rate + per-MID volume caps + max-share / floor), vectorised. Deterministic
given `seed`. The caller (tab 3) uses this GA for the compliant (dial-0) endpoint,
guards it against greedy (only adopting it when compliant AND higher-revenue), and
blends it with the revenue reference across the slider.

(The earlier raw-share global GA, per-cell-tilt reparam GA and NSGA-II variants were
removed once the cross-cell per-MID tilt superseded them.)
"""
from __future__ import annotations

import numpy as np

__build__ = "2026-07-19-genetic-band-priority-mult"


def _mid_sums(vol, mid_rows, M):
    """Per-MID column sums of `vol` (P, N) -> (P, M). Loops over the ~20 MIDs
    (vectorised per MID) — far faster than np.add.at on 49k rows."""
    out = np.empty((vol.shape[0], M), dtype=float)
    for m in range(M):
        r = mid_rows[m]
        out[:, m] = vol[:, r].sum(axis=1) if len(r) else 0.0
    return out


def _fitness(pop, ctx, lam):
    """Vectorised fitness. Revenue is the SAME quantity tab 4 shows as incremental
    revenue (maximising it ≡ maximising the delta vs a fixed baseline): each row's
    `rev_coef` = 30D-attempts × raw gateway success rate × avg ticket, so revenue =
    Σ share·rev_coef. All penalties are in those $-revenue units, so λ∈[0,1] is
    meaningful: λ=1 values a risk breach at ~1× the revenue it earns; λ=0 ignores
    risk. Max-share / floor carry a fixed heavy weight so they hold at every λ.
    RISK terms (VAMP rate, per-MID volume) use the FORECAST volume basis, matching
    the VAMP projection. Returns (P,) fitness."""
    cv, risk, rc = ctx["cell_vol"], ctx["risk"], ctx["rev_coef"]
    mid_rows, M = ctx["mid_rows"], ctx["n_mid"]
    rev_row = pop * rc[None, :]                           # tab-4-aligned revenue per row
    revenue = rev_row.sum(axis=1)

    # A breach costs MID_revenue × (BREACH_FIXED · breached + over²): a big FIXED hit the
    # instant a cap is crossed (so the GA treats a cap almost like a wall), plus a
    # QUADRATIC term so deeper breaches hurt sharply more. `over` is the RELATIVE breach
    # (actual / limit − 1). Zero penalty while compliant.
    risk_pen = np.zeros(pop.shape[0], dtype=float)
    _bfix = float(ctx.get("breach_fixed", 50.0))
    _bands = ctx.get("midband")
    if M and (ctx["vamp_cap"] is not None or ctx["mid_vol_cap"] is not None or _bands):
        vol = pop * cv[None, :]                           # forecast volume (for risk)
        midv = _mid_sums(vol, mid_rows, M)
        midrev = _mid_sums(rev_row, mid_rows, M)          # MID revenue = penalty scale
        if ctx["vamp_cap"] is not None:                   # per-vampMid aggregate VAMP rate
            midvr = _mid_sums(vol * risk[None, :], mid_rows, M)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(midv > 1e-12, midvr / midv, 0.0)
            over = np.maximum(rate / max(ctx["vamp_cap"], 1e-9) - 1.0, 0.0)
            risk_pen += (midrev * (_bfix * (over > 1e-12) + over ** 2)).sum(axis=1)
        if ctx["mid_vol_cap"] is not None:                # per-MID volume / projected caps
            _cap_v = np.where(ctx["mid_vol_cap"] > 0, ctx["mid_vol_cap"], np.inf)
            over_v = np.maximum(midv / _cap_v[None, :] - 1.0, 0.0)
            risk_pen += (midrev * (_bfix * (over_v > 1e-12) + over_v ** 2)).sum(axis=1)
        if _bands:                                        # month-specific per-MID PROJECTED bands
            # Each candidate's projected per-MID VAMP/Txn for the rule's month(s) is estimated
            # by a volume-ratio proxy: projected ≈ baseline_projected × (MID volume / baseline
            # MID volume). Tilting a MID up shrinks its volume → shrinks its projected metric,
            # so the GA can evolve toward the bands. (vamp_pct rules are scale-invariant under this
            # proxy → excluded by the caller and left to the post-GA enforcement.)
            #
            # FIXED + QUADRATIC penalty ($-scaled by MID revenue): a fixed `band_fixed` hit the
            # instant a band is breached (either side), PLUS a `band_weight` quadratic in the
            # relative breach so deeper misses hurt progressively more. The fixed hit is kept
            # BELOW the VAMP cap's wall (250) so the hard compliance cap still outranks the bands.
            # Trade-off: the fixed element can cost some conversion when a band is genuinely
            # unreachable — mitigated now by the gain lever (reach) and the dial-0 floor clamp.
            _band_w = float(ctx.get("band_weight", 8.0))
            _band_fix = float(ctx.get("band_fixed", 20.0))    # fixed hit on ANY band breach
            _bvol = ctx.get("mid_base_vol")
            with np.errstate(divide="ignore", invalid="ignore"):
                _fmid = (np.where(_bvol[None, :] > 1e-12, midv / _bvol[None, :], 1.0)
                         if _bvol is not None else np.ones_like(midv))
            for _b in _bands:
                # band tuple: (mid_index, baseline_proj, ceiling, floor[, var_mult[, prio_mult]]).
                # var_mult scales ONLY the quadratic (VAMP bands harder than txn); prio_mult scales
                # the WHOLE penalty (priority: lower-priority constraints get a smaller weight, so
                # they yield first when the set is infeasible). Both default to 1.0.
                _mi, _bval, _ceil, _floor = _b[0], _b[1], _b[2], _b[3]
                _vmul = float(_b[4]) if len(_b) > 4 else 1.0
                _pmul = float(_b[5]) if len(_b) > 5 else 1.0
                _proj = _fmid[:, _mi] * float(_bval)
                if _ceil is not None:
                    _ov = np.maximum(_proj / max(float(_ceil), 1e-9) - 1.0, 0.0)
                    risk_pen += midrev[:, _mi] * _pmul * (_band_fix * (_ov > 1e-12) + _band_w * _vmul * _ov ** 2)
                if _floor is not None and float(_floor) > 0:
                    _un = np.maximum(1.0 - _proj / max(float(_floor), 1e-9), 0.0)
                    risk_pen += midrev[:, _mi] * _pmul * (_band_fix * (_un > 1e-12) + _band_w * _vmul * _un ** 2)

    # Structural (max-share / floor) — $-equivalent (rev_coef), fixed heavy weight.
    shape = (np.maximum(pop - ctx["max_share"], 0.0) * rc[None, :]).sum(axis=1)
    if ctx["floor"] > 0:
        shape += (np.maximum(ctx["floor"] - pop, 0.0) * (ctx["elig"] * rc)[None, :]).sum(axis=1)

    fit = revenue - lam * risk_pen - ctx["shape_mult"] * shape

    # Optional RISK-MINIMISATION secondary objective (used only for the SAFE compliant
    # endpoint of the slider). It subtracts mu × aggregate expected VAMP count, so among
    # equally-compliant splits the GA prefers the one that also carries LESS total risk —
    # tilting each MID further toward its low-risk cells even below the cap. The caller
    # auto-scales mu (risk_min_w) to trade a bounded slice of revenue for lower risk;
    # default 0 leaves the pure revenue objective unchanged. (compliant-frontier)
    _rmw = float(ctx.get("risk_min_w", 0.0))
    if _rmw > 0.0:
        _vol = pop * cv[None, :]
        _vfr = ctx.get("vamp_floor_route")
        if _vfr is not None and M:
            # CLAMP at the VAMP floor: only reward reducing the VAMP that sits ABOVE each MID's
            # routing-space floor (derived from its two-sided VAMP band floor). Once a MID is at
            # its floor the risk-min term stops pulling, so dial-0 risk-min no longer drives VAMP
            # BELOW the band — keeping the two-sided VAMP ranges satisfiable at dial 0.
            _midvr = _mid_sums(_vol * risk[None, :], mid_rows, M)         # (P, M) routing VAMP/MID
            _excess = np.maximum(_midvr - np.asarray(_vfr, float)[None, :], 0.0)
            fit = fit - _rmw * _excess.sum(axis=1)
        else:
            _total_vamp = (_vol * risk[None, :]).sum(axis=1)   # aggregate expected VAMP count
            fit = fit - _rmw * _total_vamp
    return fit


# ---------------------------------------------------------------------------
# CROSS-CELL per-MID tilt search (the efficient one that can actually beat greedy).
#
# The per-cell tilt above reweights WITHIN a cell, so it can't move a vampMid's
# CROSS-cell aggregate VAMP rate — which is exactly the constraint. This version
# searches ONE parameter per vampMid: a cross-cell tilt θ_m that shifts that MID's
# volume from its HIGH-risk cells toward its LOW-risk cells:
#
#     share_g ∝ ref_g · exp(−θ_{mid(g)} · z_g)   (z_g = risk standardised WITHIN the MID)
#
# Raising θ_m pulls MID m out of its riskiest cells (dropping its aggregate rate)
# and the freed share redistributes per cell in proportion to the revenue reference
# (revenue-efficient recipients — unlike greedy, which dumps onto the lowest-rate
# gateway). Genome = n_mid (~20) dims, so the search is tiny and fast, AND it targets
# the real per-MID cross-cell constraint, so it can retain more revenue at compliance
# than the greedy shave on MIDs whose risk varies across cells.
# ---------------------------------------------------------------------------
def _risk_z_per_mid(risk, mid_rows, n_mid, N):
    """Standardise each vampMid's per-cell risk across ITS rows, so θ_m tilts that
    MID toward its own lower-risk cells."""
    z = np.zeros(N, dtype=float)
    for m in range(n_mid):
        r = mid_rows[m]
        if len(r) == 0:
            continue
        rr = risk[r]
        sd = rr.std()
        z[r] = (rr - rr.mean()) / sd if sd > 1e-12 else 0.0
    return z


def _cap_floor_shares(X, cell_starts, cell_counts, elig, cap, floor):
    """HARD per-cell max-share cap + exploration floor on shares X (P, N), vectorised.
    Cells are contiguous segments (cell_starts / cell_counts). Applies the floor first
    (lift eligible gateways to a per-cell-clamped floor, renormalise), then WATER-FILLS
    any over-cap excess into the same cell's under-cap eligible gateways — enforced LAST,
    so every row exits with share <= cap. Cells with < 2 eligible gateways can't be capped
    (a single gateway must be 1.0), matching the export's _cap_shares. Guarantees the GA
    only ever evaluates deployable splits (no search/output mismatch)."""
    elig_row = elig[None, :] > 0.5
    n_elig_cell = np.repeat(np.add.reduceat(elig.astype(float), cell_starts), cell_counts)  # (N,)
    if floor > 0.0:
        # Per-cell floor clamped to 1/n_elig so n_elig×floor <= 1 stays feasible. Lift any
        # below-floor eligible gateway to the floor, and take the deficit from ABOVE-floor
        # eligible gateways in proportion to their room above the floor (water-fill DOWN) — so
        # the sum stays 1 and the floored values are NOT shrunk back under by a global renorm.
        fl = np.minimum(floor, np.where(n_elig_cell > 0, 1.0 / np.maximum(n_elig_cell, 1.0),
                                        0.0))[None, :]
        for _ in range(50):
            under = elig_row & (X < fl - 1e-12)
            if not under.any():
                break
            deficit_cell = np.repeat(np.add.reduceat(np.where(under, fl - X, 0.0), cell_starts,
                                                     axis=1), cell_counts, axis=1)
            X = np.where(under, fl, X)
            give = np.where(elig_row & (~under) & (X > fl + 1e-12), X - fl, 0.0)
            give_cell = np.repeat(np.add.reduceat(give, cell_starts, axis=1), cell_counts, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X - np.where(give_cell > 1e-12, give * deficit_cell / give_cell, 0.0)
    if cap < 1.0:
        capN = np.where(n_elig_cell >= 2, cap, 1.0)[None, :]     # single-gateway cells uncapped
        for _ in range(50):
            over = X > capN + 1e-12
            if not over.any():
                break
            excess_col = np.where(over, X - capN, 0.0)
            excess_cell = np.repeat(np.add.reduceat(excess_col, cell_starts, axis=1),
                                    cell_counts, axis=1)
            X = np.where(over, capN, X)
            room = np.where(elig_row & (~over) & (X < capN - 1e-12), capN - X, 0.0)
            room_cell = np.repeat(np.add.reduceat(room, cell_starts, axis=1), cell_counts, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X + np.where(room_cell > 1e-12, room * excess_cell / room_cell, 0.0)
    return X


def _decode_midtilt(theta, ref, z_mid, mid_id, cell_starts, cell_counts, elig,
                    cap=1.0, floor=0.0, gain=None):
    """theta (P, M) -> shares (P, N). Per-row tilt uses the row's MID via mid_id.

    `gain` (P, M), optional, is a per-MID overall-presence knob: share ∝ ref · exp(−θ·z) ·
    exp(g_mid). θ decides WHERE a MID sits (which of its cells); g decides HOW MUCH of it
    there is across ALL its cells — a raised g pulls share to that MID in every cell (taking
    it from other MIDs), which is the cross-MID volume move the tilt alone can't make. The
    max-share cap and exploration floor are enforced HARD here (see _cap_floor_shares), so
    every candidate the GA scores is already a deployable split."""
    tr = theta[:, mid_id]                                     # (P, N)
    w = ref[None, :] * np.exp(-tr * z_mid[None, :]) * elig[None, :]
    if gain is not None:
        w = w * np.exp(gain[:, mid_id])                       # per-MID overall-presence gain
    seg = np.add.reduceat(w, cell_starts, axis=1)
    seg = np.where(seg > 1e-12, seg, 1.0)
    X = w / np.repeat(seg, cell_counts, axis=1)
    if cap < 1.0 or floor > 0.0:
        X = _cap_floor_shares(X, cell_starts, cell_counts, elig, float(cap), float(floor))
    return X


def run_midtilt_ga(ctx, lam, *, pop_size=40, generations=80, mutation_rate=0.3,
                   mutation_sigma=1.0, seed=42, elite_frac=0.2, auto=True,
                   patience=12, sigma_min=0.05, sigma_max=4.0, success_window=5,
                   theta_max=25.0, gain_max=2.0, warm_start=None):
    """Evolve a per-vampMid genome = [θ (cross-cell tilt) | g (overall-presence gain)],
    maximising revenue − λ·risk_penalty. Genome is 2·n_mid dims (still tiny), so it stays
    fast; the gain block lets a MID gain/shed net volume across cells (cross-MID moves the
    tilt alone can't make). `warm_start` seeds known-good genomes (e.g. a prior run's winner)
    into the population — free reach without more compute. Returns (best_shares (N,), info)
    with info['genome'] for warm-starting the next run. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    N = ctx["n_row"]; M = int(ctx["n_mid"])
    cs = np.asarray(ctx["cell_starts"]); cc = np.asarray(ctx["cell_counts"])
    elig = np.asarray(ctx["elig"], float)
    ref = np.asarray(ctx["ref_share"], float)
    mid_id = np.asarray(ctx["mid_id"])
    z_mid = _risk_z_per_mid(np.asarray(ctx["risk"], float), ctx["mid_rows"], M, N)
    _cap = float(ctx.get("max_share", 1.0) or 1.0)      # HARD max-share (enforced in decode)
    _floor = float(ctx.get("floor", 0.0) or 0.0)        # HARD exploration floor
    D = 2 * M                                            # genome = [θ | g]
    gain_max = float(gain_max)

    def _clip(x):
        x = np.array(x, dtype=float, copy=True)
        x[..., :M] = np.clip(x[..., :M], 0.0, theta_max)     # tilt >= 0
        x[..., M:] = np.clip(x[..., M:], -gain_max, gain_max)  # gain symmetric
        return x

    def _decode(genome):
        return _decode_midtilt(genome[:, :M], ref, z_mid, mid_id, cs, cc, elig, _cap, _floor,
                               gain=genome[:, M:])

    def fit_of(genome):
        return _fitness(_decode(genome), ctx, lam)

    pop = np.zeros((pop_size, D), dtype=float)           # member 0 = θ=0,g=0 = revenue reference
    if pop_size > 1:
        pop[1:, :M] = rng.uniform(0.0, theta_max * 0.5, size=(pop_size - 1, M))
        pop[1:, M:] = rng.uniform(-gain_max * 0.5, gain_max * 0.5, size=(pop_size - 1, M))
    if warm_start is not None:                           # seed prior winners after member 0
        _ws = np.atleast_2d(np.asarray(warm_start, float))
        if _ws.ndim == 2 and _ws.shape[1] == D and pop_size > 1:
            _k = min(len(_ws), pop_size - 1)
            pop[1:1 + _k] = _clip(_ws[:_k])
    fit = fit_of(pop)
    n_elite = max(1, int(round(elite_frac * pop_size)))
    sigma = float(max(mutation_sigma, sigma_max * 0.5)) if auto else float(mutation_sigma)
    _g_scale = (gain_max / theta_max) if theta_max > 1e-9 else 1.0   # gain noise ~ its range
    init_fit = float(fit.max()); best_fit = init_fit
    best_g = pop[int(np.argmax(fit))].copy()
    hist, stale = [], 0; _C = 0.85; gens_run = 0

    for g in range(generations):
        gens_run = g + 1
        order = np.argsort(-fit)
        elite = pop[order[:n_elite]].copy()

        def _pick():
            c = rng.integers(0, pop_size, size=3)
            return c[np.argmax(fit[c])]
        children = np.empty_like(pop)
        children[:n_elite] = elite
        for k in range(n_elite, pop_size):
            a, b = pop[_pick()], pop[_pick()]
            wq = rng.random(D)
            child = wq * a + (1 - wq) * b
            m = rng.random(D) < mutation_rate
            noise = rng.normal(0.0, sigma, size=D)
            noise[M:] *= _g_scale                        # scale gain mutations to the gain range
            child = np.where(m, child + noise, child)
            children[k] = _clip(child)
        pop = children
        fit = fit_of(pop)

        gb = float(fit.max()); improved = gb > best_fit + 1e-9
        if improved:
            best_fit = gb; best_g = pop[int(np.argmax(fit))].copy(); stale = 0
        else:
            stale += 1
        if auto:
            hist.append(1 if improved else 0)
            if len(hist) >= success_window:
                ps = float(np.mean(hist[-success_window:]))
                sigma = sigma / _C if ps > 0.2 else sigma * _C
                sigma = float(min(max(sigma, sigma_min), sigma_max))
            if stale >= patience:
                break

    best = _decode(best_g[None, :])[0]
    best_rev = float((best * ctx["rev_coef"]).sum())
    info = {
        "gens": int(gens_run), "gens_max": int(generations),
        "early_stopped": bool(auto and gens_run < generations),
        "sigma_final": float(sigma), "best_fit": float(best_fit), "init_fit": float(init_fit),
        "revenue": best_rev, "risk_cost": float(best_rev - best_fit), "dims": D,
        "genome": best_g.copy(),      # pass to warm_start on a subsequent run
    }
    return best, info
