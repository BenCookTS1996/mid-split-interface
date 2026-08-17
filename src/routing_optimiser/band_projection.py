"""EXACT, fast collapse of the per-MID cap projection used for GA band scoring.

Context
-------
The GA's per-MID month bands (e.g. "Adyen_TotalAVPro M5 VAMP ≤ 1,800") are defined on the
TRUE pro-rata projection. In `app/streamlit_app.py` that projection is `_project_capped`: a
two-cohort model over a large scaffold (`_T0` = the t0 rows, `_Pc` = the aged rows, one per
observation-month × age `t`). It is bit-exact but costs a few bincounts over ~1–2M rows per
call — fine for enforcement's few hundred calls, far too slow for the GA's ~1.2M evaluations,
which is why the search falls back to a crude "volume-ratio" proxy (the source of the large
proxy↔true gaps in the run log).

Key structural facts verified in `_project_capped`:
  * An aged row seen in month `per` at age `t` ORIGINATED in month `om = per − t`, and its
    movement is governed by the ORIGIN t0 row's movable fraction and share
    (`_Pc → _T0` join on `om == per`).
  * The moved-VAMP POOL is already aggregated to CELL grain (vampMid summed out):
    `pool(cell,per,t) = Σ_all-mids vc·pro_rata·fcp1` (the app's static `_Pc_movedvpool_a`),
    then re-split by the proposed share.
  * Caps are NOT applied inside the projection — the ceiling/floor is compared afterwards.
    So the forward map is LINEAR in the (per-cell-normalised) share.
  * **The movable fraction is GATED on the cell's proposed-share sum** (line 3951 of the app):
    `mv = where(psum > 0, pr·fcp, 0)`. In a cell where the candidate assigns ZERO to every
    gateway (zero-volume historical cell, or every gateway zeroed by the wallet/USA `emask`),
    NOTHING moves — each MID holds 100 % of its own VAMP. This gate is candidate-dependent,
    so the VAMP held-cohort is NOT a single static offset: it is `Σvc` (constant) MINUS a
    movement term `vc·mv` that is subtracted only when the origin cell is active (psum>0).

Therefore, for the count metrics:
    VAMP(mid,P) = Σ_pc vc                                    (static constant)
                − Σ_{origin t0 rows} (mv·Σvc) · [psum_origin>0]   (movement, psum-gated)
                + Σ_{origin t0 rows} pool_sum · vshare(origin)    (redistributed pool)
    Txn (mid,P) = Σ_{t0 cap rows}  ctot·base                         if psum_cell==0
                = Σ_{t0 cap rows}  ctot·(base·(1−mv) + moved_tot·pshare)  if psum_cell>0
where the per-row constants (`ctot`, `base`, `mv=pr·fcp`, `moved_tot`, `pool_sum`, `Σvc`) are
STATIC (precomputed once) and the only per-candidate inputs are the per-cell shares and the
`psum>0` active mask. `vshare`/`pshare` are the exact per-cell normalisations `_project_capped`
uses. This module precomputes the static pieces and evaluates the (piecewise-)linear form.

`BandProjector` precomputes the static collapse of `_project_capped` and evaluates it; the
collapse is exact for the VAMP/Txn COUNT metrics — including a cell whose candidate share is
entirely zero (the psum==0 path). (The `vamp_pct` RATE metric is out of scope: project VAMP
and volume separately and divide at the end.)

Inputs (both frames use lower-cased helper columns):
  T0   : cur, bin, rpgt, pmp, ctry, mid, midl, per, vi, vc, pr, fcp, bf, excl, emask, iscap
         (`_av` = per-row non-excluded VI used for the baseline share; `iscap`=MID has a band;
          `bf`=injected back-fill row; `excl`=switched-off; `emask`=wallet/USA-only masked)
  Pc   : cur, bin, rpgt, pmp, ctry, mid, midl, per, t, vc
  pool : Pc-aligned static array = cell-grain moved-VAMP pool (`_Pc_movedvpool_a`).
`prop` is a dict keyed (cur,bin,mid) — or (cur,bin,rpgt,mid) when by_rpgt — → proposed share.
Cell (`grpk`) = (cur,bin,rpgt,pmp,ctry,per); `ctot` = cell total VI at t0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Bumped when the projection signature/behaviour changes so stale bytecode is obvious in the run log.
__build__ = "2026-08-16-appearance-month-timing+maxshare-cap+subcell-propkey"

try:                                   # numba is optional — pure-NumPy path used if absent
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:                      # noqa: BLE001
    _HAVE_NUMBA = False

    # [FN-008]
    def _njit(*_a, **_k):
        # [FN-009]
        def _deco(f):
            return f
        return _deco

_GRPK = ["cur", "bin", "rpgt", "pmp", "ctry", "per"]


# [FN-010]
@_njit(cache=True)
def _pop_band_kernel(prop_raw, propidx, masked, gcode, base, mv_s, vcpos, ctot,
                     pc_org, pc_vc, pc_pool, pc_band, pc_heldfac, cap_row, cap_band, ncell, nband,
                     cap, vamp, txn, psum, vpsum, moved, pr, pshare, vshare, mvrow, nzc, exc, rsum):
    """Bit-identical numba equivalent of PopulationBandProjector.project_pop: flat passes over
    the reduced scaffold with per-cell scratch (ncell), no dense (P × nR) arrays. ~7× faster on
    the real scaffold. cap_row is pre-filtered to non-excl rows (excl txn contributions are 0).

    `cap` (max_share) < 1.0 folds in the per-sub-cell max-share water-fill (matches
    build_split_exports); cap >= 1.0 is a no-op. vshare is (re)derived from the capped routed
    share so the cap flows into VAMP exactly as the delivered projection does.

    All working arrays are passed IN and REUSED across calls (no per-generation allocation). Only
    accumulators need resetting: vamp/txn zeroed here; psum/moved/vpsum(=VAMP denom)/nzc/exc/rsum
    zeroed per candidate; pr/pshare/vshare/mvrow fully overwritten. Byte-identical to fresh alloc."""
    P = prop_raw.shape[0]; nR = propidx.shape[0]
    nA = pc_org.shape[0]; nC = cap_row.shape[0]
    vamp[:, :] = 0.0; txn[:, :] = 0.0
    for p in range(P):
        for c in range(ncell):
            psum[c] = 0.0; moved[c] = 0.0
        for r in range(nR):
            v = 0.0 if masked[r] else prop_raw[p, propidx[r]]
            pr[r] = v
            psum[gcode[r]] += v
        for r in range(nR):
            c = gcode[r]
            if psum[c] > 0.0:
                moved[c] += base[r] * mv_s[r]
        for r in range(nR):
            c = gcode[r]; ps = psum[c]
            if ps > 0.0:
                pshare[r] = pr[r] / ps; mvrow[r] = mv_s[r]
            else:
                pshare[r] = base[r]; mvrow[r] = 0.0
        # ---- per-sub-cell max-share water-fill (only cells with >=2 routed gateways) ----
        if cap < 1.0:
            for c in range(ncell):
                nzc[c] = 0.0
            for r in range(nR):
                c = gcode[r]
                if psum[c] > 0.0 and pshare[r] > 1e-12:
                    nzc[c] += 1.0
            for _sw in range(50):
                for c in range(ncell):
                    exc[c] = 0.0
                any_over = False
                for r in range(nR):
                    c = gcode[r]
                    if psum[c] > 0.0 and nzc[c] >= 2.0 and pshare[r] > cap + 1e-12:
                        exc[c] += pshare[r] - cap; any_over = True
                if not any_over:
                    break
                for r in range(nR):
                    c = gcode[r]
                    if psum[c] > 0.0 and nzc[c] >= 2.0 and pshare[r] > cap + 1e-12:
                        pshare[r] = cap
                for c in range(ncell):
                    rsum[c] = 0.0
                for r in range(nR):
                    c = gcode[r]
                    if psum[c] > 0.0 and nzc[c] >= 2.0 and pshare[r] > 1e-12 and pshare[r] < cap - 1e-12:
                        rsum[c] += cap - pshare[r]
                for r in range(nR):
                    c = gcode[r]
                    if (psum[c] > 0.0 and nzc[c] >= 2.0 and pshare[r] > 1e-12
                            and pshare[r] < cap - 1e-12 and rsum[c] > 1e-12):
                        pshare[r] += (cap - pshare[r]) / rsum[c] * exc[c]
        # ---- vshare from the (capped) ROUTED share (0 in inactive cells) ----
        for c in range(ncell):
            vpsum[c] = 0.0
        for r in range(nR):
            c = gcode[r]
            if psum[c] > 0.0 and vcpos[r] > 0.5:
                vpsum[c] += pshare[r]
        for r in range(nR):
            c = gcode[r]
            if psum[c] > 0.0 and vcpos[r] > 0.5 and vpsum[c] > 0.0:
                vshare[r] = pshare[r] / vpsum[c]
            else:
                vshare[r] = 0.0
        for j in range(nC):
            r = cap_row[j]; c = gcode[r]
            txn[p, cap_band[j]] += ctot[r] * (base[r] * (1.0 - mvrow[r]) + moved[c] * pshare[r])
        for j in range(nA):
            o = pc_org[j]
            # APPEARANCE-MONTH timing: held move = fcp[origin]·pro_rata[appearance] (pc_heldfac),
            # gated on the ORIGIN cell being routed (psum>0). Pool is pre-built appearance-timed.
            if o >= 0:
                mpc = pc_heldfac[j] if psum[gcode[o]] > 0.0 else 0.0
                psh = vshare[o]
            else:
                mpc = 0.0
                psh = 0.0
            vamp[p, pc_band[j]] += pc_vc[j] * (1.0 - mpc) + pc_pool[j] * psh
    return vamp, txn


# [FN-011]
def _prop_key(df: pd.DataFrame, by_rpgt: bool, by_subcell: bool = False) -> np.ndarray:
    """Build each row's bucket address:
      by_subcell → 'cur|bin|rpgt|pmp|ctry|mid'  (SUB-CELL decision grain)
      by_rpgt    → 'cur|bin|rpgt|mid'
      else       → 'cur|bin|mid'

    This is the key a proposed share is looked up by — the SAME string format the projector's
    `prop_keys` use, so a candidate's shares line up with the right rows (like a postcode that
    routes each share to the correct bucket). The sub-cell key includes pmp/ctry so a per-sub-cell
    share maps to exactly one scaffold row (no broadcast across sub-cells).
    """
    _cur = df["cur"].astype(str).str.strip().str.lower()
    _bin = df["bin"].astype(str).str.strip()
    _mid = df["mid"].astype(str).str.strip()
    if by_subcell:
        return (_cur + "|" + _bin + "|" + df["rpgt"].astype(str).str.strip().str.lower() + "|"
                + df["pmp"].astype(str).str.strip().str.lower() + "|"
                + df["ctry"].astype(str).str.strip().str.lower() + "|" + _mid).to_numpy()
    if by_rpgt:
        return (_cur + "|" + _bin + "|" + df["rpgt"].astype(str).str.strip().str.lower()
                + "|" + _mid).to_numpy()
    return (_cur + "|" + _bin + "|" + _mid).to_numpy()


# [FN-011b]
def _prop_key_str(k, by_rpgt: bool, by_subcell: bool = False) -> str:
    """Tuple→key string, matching `_prop_key`'s formats. SINGLE source so `_prop_raw` and
    `project_pop_from_props` can't diverge. Tuple layouts:
      by_subcell → (cur, bin, rpgt, pmp, ctry, mid)
      by_rpgt    → (cur, bin, rpgt, mid)
      else       → (cur, bin, mid)
    """
    if by_subcell:
        return "|".join([str(k[0]).strip().lower(), str(k[1]).strip(), str(k[2]).strip().lower(),
                         str(k[3]).strip().lower(), str(k[4]).strip().lower(), str(k[5]).strip()])
    if by_rpgt:
        return "|".join([str(k[0]).strip().lower(), str(k[1]).strip(),
                         str(k[2]).strip().lower(), str(k[3]).strip()])
    return "|".join([str(k[0]).strip().lower(), str(k[1]).strip(), str(k[2]).strip()])


# [FN-012]
def _prop_raw(T0: pd.DataFrame, prop: dict, by_rpgt: bool, by_subcell: bool = False) -> np.ndarray:
    """Look up each t0 row's proposed share from the `prop` dict by its bucket key.

    Rows that are switched off (`excl`) or wallet/USA-masked (`emask`) are forced to 0 —
    they can't receive routed volume, so they never enter the moved cohort.
    """
    keys = _prop_key(T0, by_rpgt, by_subcell)
    m = {}
    for k, v in prop.items():
        m[_prop_key_str(k, by_rpgt, by_subcell)] = float(v)
    raw = np.array([m.get(k, 0.0) for k in keys], dtype=float)
    raw = np.where(T0["excl"].to_numpy(bool), 0.0, raw)
    raw = np.where(T0["emask"].to_numpy(bool), 0.0, raw)
    return raw


# [FN-013]
def _static(T0: pd.DataFrame):
    """Precompute the per-row pieces that DON'T depend on the candidate (done once).

    Returns (gcode, ngc, base, ctot, mv_static):
      * gcode / ngc — an integer code per cell (grpk) + the number of cells, so cell-wise
        sums become fast bincounts (grouping rows into their cell "bins");
      * base        — each row's baseline share of its cell (non-excluded VI ÷ cell VI);
      * ctot        — the cell's total VI at t0, broadcast onto each of its rows;
      * mv_static   — the UNGATED movable fraction pr·fcp. The psum>0 gate ("did the
        candidate put any volume in this cell?") is applied per-candidate at eval time,
        matching `_project_capped` line 3951.
    """
    gcode = pd.factorize(T0[_GRPK].astype(str).agg("|".join, axis=1))[0]
    ngc = int(gcode.max()) + 1 if len(gcode) else 0
    av_sum = np.bincount(gcode, weights=T0["_av"].to_numpy(float), minlength=ngc)[gcode]
    base = np.where(av_sum > 0, T0["_av"].to_numpy(float) / np.where(av_sum > 0, av_sum, 1.0), 0.0)
    ctot = np.bincount(gcode, weights=T0["vi"].to_numpy(float), minlength=ngc)[gcode]
    # UNGATED movable fraction; gated on the candidate's per-cell psum>0 at eval time
    # (matches `_project_capped` line 3951: mv = where(psum>0, pr·fcp, 0)).
    mv_static = T0["pr"].to_numpy(float) * T0["fcp"].to_numpy(float)
    return gcode, ngc, base, ctot, mv_static


# [FN-014]
def _origin_map(T0: pd.DataFrame, Pc: pd.DataFrame) -> np.ndarray:
    """Each aged Pc row -> its ORIGIN t0 row index (om==per), excluding back-fill; -1 if none."""
    t0join = (T0["cur"] + "|" + T0["bin"] + "|" + T0["rpgt"] + "|" + T0["pmp"] + "|"
              + T0["ctry"] + "|" + T0["midl"] + "|" + T0["per"].astype(str)).to_numpy()
    valid = ~T0["bf"].to_numpy(bool)
    t0pos = pd.Series(np.where(valid)[0], index=t0join[valid])
    t0pos = t0pos[~t0pos.index.duplicated(keep="last")]
    om = (Pc["per"] - Pc["t"]).astype(int)
    pcjoin = (Pc["cur"] + "|" + Pc["bin"] + "|" + Pc["rpgt"] + "|" + Pc["pmp"] + "|"
              + Pc["ctry"] + "|" + Pc["midl"] + "|" + om.astype(str)).to_numpy()
    return t0pos.reindex(pcjoin).fillna(-1).to_numpy().astype(np.int64)


# [FN-015]
def _shares(T0, prop, by_rpgt, gcode, ngc, base, by_subcell=False):
    """Return (pshare, vshare, psum) for the candidate. `psum` is the per-row (broadcast
    per-cell) proposed-share sum; `psum>0` is the active mask that gates `mv`."""
    prop_raw = _prop_raw(T0, prop, by_rpgt, by_subcell)
    psum = np.bincount(gcode, weights=prop_raw, minlength=ngc)[gcode]
    pshare = np.array(base, dtype=float)
    np.divide(prop_raw, psum, out=pshare, where=psum > 0)
    vprop = prop_raw * (T0["vc"].to_numpy(float) > 0)
    vpsum = np.bincount(gcode, weights=vprop, minlength=ngc)[gcode]
    vshare = np.zeros_like(vprop)
    np.divide(vprop, vpsum, out=vshare, where=vpsum > 0)
    return pshare, vshare, psum


# NOTE: the `project_reference` oracle (a readable NumPy re-implementation of `_project_capped`)
# was removed — it was never called and no test asserted equivalence. `BandProjector` and
# `PopulationBandProjector` below are the live implementations of the same two-cohort math.


class BandProjector:
    """Precompute the static collapse of `_project_capped` for a set of banded (midl,period)
    pairs; `project(prop)` then evaluates the exact VAMP/Txn projection as a small
    (piecewise-)linear form. The only per-candidate inputs are the per-cell shares and the
    `psum>0` active mask that gates the movable fraction."""

    # [FN-017]
    def __init__(self, T0: pd.DataFrame, Pc: pd.DataFrame, pool: np.ndarray, bands,
                 by_rpgt: bool = False, by_subcell: bool = False):
        self.by_rpgt = by_rpgt
        self.by_subcell = by_subcell
        self.bands = {(str(m).strip().lower(), int(p)) for (m, p) in bands}
        self._T0 = T0.reset_index(drop=True)
        Pc = Pc.reset_index(drop=True)
        self._gcode, self._ngc, self._base, self._ctot, self._mv = _static(self._T0)
        pool = np.asarray(pool, float)

        # ---- VAMP: const Σvc  −  psum-gated movement  +  pool routed by ORIGIN vshare ---------
        # APPEARANCE-MONTH timing (see PopulationBandProjector): the held move for an aged row is
        # fcp[origin] × pro_rata[APPEARANCE cell], not pr[origin]·fcp[origin]. So each aged row's
        # held weight is fcp[origin]·prapp_row·vc (was mv_static[origin]·vc). Pool is appearance-timed
        # upstream. Keeps this diagnostic consistent with _project_capped / the population projector.
        pc_to_t0 = _origin_map(self._T0, Pc)
        pc_mid = Pc["midl"].to_numpy(); pc_per = Pc["per"].to_numpy().astype(int)
        pc_vc = Pc["vc"].to_numpy(float)
        _fcp_arr = (self._T0["fcp"].to_numpy(float) if "fcp" in self._T0.columns
                    else np.ones(len(self._T0)))
        _t0_ck = (self._T0["cur"].astype(str) + "|" + self._T0["bin"].astype(str) + "|"
                  + self._T0["rpgt"].astype(str) + "|" + self._T0["pmp"].astype(str) + "|"
                  + self._T0["ctry"].astype(str) + "|" + self._T0["per"].astype(str)).to_numpy()
        _pr_by_cell = pd.Series((self._T0["pr"].to_numpy(float) if "pr" in self._T0.columns
                                 else np.ones(len(self._T0))), index=_t0_ck)
        _pr_by_cell = _pr_by_cell[~_pr_by_cell.index.duplicated(keep="first")]
        _pc_ck = (Pc["cur"].astype(str) + "|" + Pc["bin"].astype(str) + "|" + Pc["rpgt"].astype(str)
                  + "|" + Pc["pmp"].astype(str) + "|" + Pc["ctry"].astype(str) + "|"
                  + Pc["per"].astype(str)).to_numpy()
        _pc_prapp = (_pr_by_cell.reindex(_pc_ck).fillna(0.0).to_numpy(float)
                     if len(Pc) else np.zeros(0, float))
        self._v_const = {}                       # key -> Σ vc over ALL aged rows in the band
        _v_hold = {}                             # key -> {origin_t0 -> Σ fcp[o]·prapp·vc}  (subtract if active)
        _v_pool = {}                             # key -> {origin_t0 -> Σ pool}          (× vshare[origin])
        for i in range(len(Pc)):
            key = (pc_mid[i], pc_per[i])
            if key not in self.bands:
                continue
            self._v_const[key] = self._v_const.get(key, 0.0) + pc_vc[i]
            o = int(pc_to_t0[i])
            if o >= 0:
                _v_hold.setdefault(key, {})
                _v_hold[key][o] = _v_hold[key].get(o, 0.0) + _fcp_arr[o] * _pc_prapp[i] * pc_vc[i]
                if pool[i] != 0.0:
                    _v_pool.setdefault(key, {})
                    _v_pool[key][o] = _v_pool[key].get(o, 0.0) + pool[i]
        # freeze to arrays for vectorised eval
        self._v_hold_o, self._v_hold_w, self._v_pool_o, self._v_pool_w = {}, {}, {}, {}
        for key in self.bands:
            h = _v_hold.get(key, {})
            self._v_hold_o[key] = np.array(list(h.keys()), dtype=np.int64)
            self._v_hold_w[key] = np.array(list(h.values()), dtype=float)
            p = _v_pool.get(key, {})
            self._v_pool_o[key] = np.array(list(p.keys()), dtype=np.int64)
            self._v_pool_w[key] = np.array(list(p.values()), dtype=float)

        # ---- TXN: per capped t0 row, piecewise on the cell's active mask -----------------------
        # active (psum>0):  ctot·(base·(1−mv) + moved_tot·pshare)
        # inactive (psum==0): ctot·base            (mv=0, moved_tot=0, pshare→base fallback)
        moved_tot = np.bincount(self._gcode, weights=self._base * self._mv,
                                minlength=self._ngc)[self._gcode]
        cap = self._T0["iscap"].to_numpy(bool)
        excl = self._T0["excl"].to_numpy(bool)
        t0_mid = self._T0["midl"].to_numpy(); t0_per = self._T0["per"].to_numpy().astype(int)
        _t_i, _t_const, _t_off, _t_coef = {}, {}, {}, {}
        for i in range(len(self._T0)):
            if not cap[i] or excl[i]:
                continue
            key = (t0_mid[i], t0_per[i])
            if key not in self.bands:
                continue
            _t_i.setdefault(key, []).append(i)
            _t_const.setdefault(key, []).append(self._ctot[i] * self._base[i])
            _t_off.setdefault(key, []).append(self._ctot[i] * self._base[i] * (1.0 - self._mv[i]))
            _t_coef.setdefault(key, []).append(self._ctot[i] * moved_tot[i])
        self._t_i, self._t_const, self._t_off, self._t_coef = {}, {}, {}, {}
        for key in self.bands:
            self._t_i[key] = np.array(_t_i.get(key, []), dtype=np.int64)
            self._t_const[key] = np.array(_t_const.get(key, []), dtype=float)
            self._t_off[key] = np.array(_t_off.get(key, []), dtype=float)
            self._t_coef[key] = np.array(_t_coef.get(key, []), dtype=float)

    # [FN-018]
    def project(self, prop: dict) -> dict:
        pshare, vshare, psum = _shares(self._T0, prop, self.by_rpgt,
                                       self._gcode, self._ngc, self._base,
                                       getattr(self, "by_subcell", False))
        active = psum > 0                                   # per T0 row (broadcast per cell)
        out = {}
        for key in self.bands:
            # VAMP
            ho = self._v_hold_o[key]
            vamp = self._v_const.get(key, 0.0)
            if len(ho):
                vamp -= float((self._v_hold_w[key] * active[ho]).sum())
            po = self._v_pool_o[key]
            if len(po):
                vamp += float((self._v_pool_w[key] * vshare[po]).sum())
            # TXN
            ti = self._t_i[key]
            if len(ti):
                a = active[ti]
                txn = float(np.where(a, self._t_off[key] + self._t_coef[key] * pshare[ti],
                                     self._t_const[key]).sum())
            else:
                txn = 0.0
            out[key] = [float(vamp), txn]
        return out


class PopulationBandProjector:
    """Exact `_project_capped` band values for a WHOLE population of candidate splits at once,
    restricted to just the sub-cells that feed the banded (midl,period) pairs.

    Same math as `BandProjector` (two-cohort held/moved, psum-gated
    movable fraction, per-sub-cell pshare/vshare renormalisation) but vectorised over P
    candidates with dense NumPy so it can be called inside the GA worker's fitness in place
    of the volume-ratio proxy. Only cells that are (a) a capped banded t0 cell or (b) an
    origin cell of a banded aged row are kept — but ALL MIDs of those cells are retained so
    the per-sub-cell normalisation (psum/vpsum) is exact.

    Interface
    ---------
    `prop_keys` : ordered list of the (cur,bin,mid) — or (cur,bin,rpgt,mid) — keys the caller
        must supply a proposed share for. Build the P×K `prop_raw` matrix in that column order
        (the worker aggregates its decoded per-gateway shares onto these keys once, via a
        precomputed column→key incidence, then calls `project_pop`).
    `project_pop(prop_raw)` -> (vamp[P,B], txn[P,B]) aligned to `band_order`.
    `project_pop_from_props(props)` : convenience for tests — builds `prop_raw` from a list of
        {key: share} dicts.
    """

    # [FN-019]
    def __init__(self, T0: pd.DataFrame, Pc: pd.DataFrame, pool: np.ndarray, bands,
                 by_rpgt: bool = False, max_share: float = 1.0, by_subcell: bool = False):
        self.by_rpgt = by_rpgt
        # SUB-CELL decision grain: prop keys include pmp/ctry (cur|bin|rpgt|pmp|ctry|mid) so a
        # per-sub-cell share maps to exactly one scaffold row instead of broadcasting across sub-cells.
        self.by_subcell = by_subcell
        # Per-sub-cell max-share CAP (matches build_split_exports): no gateway may exceed max_share
        # of a routed sub-cell; the excess water-fills onto the OTHER gateways already present.
        # 1.0 = OFF (backward-compatible). This is the SOLE driver of the scored-vs-delivered VAMP
        # residual (proven: reproduces tab-3 delivered within ±3), so folding it in makes the GA
        # fitness score the DELIVERED breach.
        self._cap = float(max_share) if max_share else 1.0
        bandset = {(str(m).strip().lower(), int(p)) for (m, p) in bands}
        T0 = T0.reset_index(drop=True)
        Pc = Pc.reset_index(drop=True)
        pool = np.asarray(pool, float)

        gcode_full, ngc_full, base_full, ctot_full, mv_full = _static(T0)
        pc_to_t0 = _origin_map(T0, Pc)
        t0_mid = T0["midl"].to_numpy(); t0_per = T0["per"].to_numpy().astype(int)
        pc_mid = Pc["midl"].to_numpy(); pc_per = Pc["per"].to_numpy().astype(int)
        iscap = T0["iscap"].to_numpy(bool)

        # banded aged rows + their origin cells; banded capped t0 rows + their cells (VECTORISED).
        # `bandset` should be the ACTUALLY-CONSTRAINED (midl,per) pairs only — restricting it here
        # is what lets the reduced scaffold shrink (fewer banded rows → fewer relevant cells).
        _bidx = (pd.MultiIndex.from_tuples(sorted(bandset)) if bandset
                 else pd.MultiIndex.from_arrays([[], []]))
        pc_band = pd.MultiIndex.from_arrays([pc_mid, pc_per]).isin(_bidx)
        t0_capband = iscap & pd.MultiIndex.from_arrays([t0_mid, t0_per]).isin(_bidx)
        _org = pc_to_t0[pc_band]
        rel = np.unique(np.concatenate([gcode_full[t0_capband],
                                        gcode_full[_org[_org >= 0]]]).astype(np.int64)) \
            if (t0_capband.any() or pc_band.any()) else np.zeros(0, np.int64)

        keep_t0 = np.isin(gcode_full, rel)                 # ALL mids of relevant cells
        t0_idx = np.where(keep_t0)[0]
        full2red = -np.ones(len(T0), np.int64); full2red[t0_idx] = np.arange(len(t0_idx))
        R = T0.iloc[t0_idx].reset_index(drop=True)

        # local per-row statics (recomputed on the reduced frame; base/ctot are cell sums, and
        # every row of each relevant cell is present, so they match the full-frame values)
        self._gcode, self._ngc, self._base, self._ctot, self._mv = _static(R)
        self._excl = R["excl"].to_numpy(bool)
        self._emask = R["emask"].to_numpy(bool)
        self._vcpos = (R["vc"].to_numpy(float) > 0).astype(float)
        # Retain the two movable-fraction FACTORS separately (mv = pr · fcp) so diagnostics can tell
        # whether a low movable fraction is driven by pro_rata (go-live phasing) or fcp1_frac (the
        # first-attempt reroutable slice). Default to 1.0 if a column is absent.
        self._pr = R["pr"].to_numpy(float) if "pr" in R.columns else np.ones(len(R))
        self._fcp = R["fcp"].to_numpy(float) if "fcp" in R.columns else np.ones(len(R))
        nR = len(R)

        # prop-key alignment (np.unique = sorted-unique + inverse index in C)
        keys = _prop_key(R, by_rpgt, by_subcell)
        if nR:
            uniq, self._propidx = np.unique(keys, return_inverse=True)
            self.prop_keys = [str(k) for k in uniq]
        else:
            self._propidx = np.zeros(0, np.int64); self.prop_keys = []
        self._K = len(self.prop_keys)

        # sparse cell-incidence (ngc × nR); cellsum(x) = (S @ x.T).T  (sparse@dense, C-fast)
        import scipy.sparse as _sp
        self._S = _sp.csr_matrix((np.ones(nR), (self._gcode, np.arange(nR))),
                                 shape=(max(self._ngc, 1), max(nR, 1)))

        # reduced Pc = banded aged rows only; remap origin to reduced t0 index
        pc_keep = np.where(pc_band)[0]
        self._pc_vc = Pc["vc"].to_numpy(float)[pc_keep]
        self._pc_pool = pool[pc_keep]
        _ofull = pc_to_t0[pc_keep]
        self._pc_org = np.where(_ofull >= 0, full2red[np.where(_ofull >= 0, _ofull, 0)], -1)
        pc_mid_k = pc_mid[pc_keep]; pc_per_k = pc_per[pc_keep]

        # ---- APPEARANCE-MONTH TIMING (#3 scored==delivered) -------------------------------------
        # The tab-3 DELIVERED projection (compute_vamp_prepost_granular) applies the go-live pro_rata
        # by the APPEARANCE month (the aged row's own period `per`), not the origination month, while
        # keeping fcp1_frac at ORIGINATION. So the movable fraction for an aged row is
        #   move = fcp[origin] × pro_rata[appearance]      (was pr[origin]·fcp[origin]).
        # Precompute, per reduced Pc row, the appearance-cell t0 pro_rata. Sourced from a t0 row of the
        # SAME sub-cell at the appearance period (cell key = cur|bin|rpgt|pmp|ctry|per, no mid — pro_rata
        # is a per-cell go-live weight). Verified bit-exact against compute_vamp_prepost_granular.
        _t0_cellkey = (T0["cur"].astype(str) + "|" + T0["bin"].astype(str) + "|" + T0["rpgt"].astype(str)
                       + "|" + T0["pmp"].astype(str) + "|" + T0["ctry"].astype(str) + "|"
                       + T0["per"].astype(str)).to_numpy()
        _t0_pr = (T0["pr"].to_numpy(float) if "pr" in T0.columns else np.ones(len(T0)))
        _pr_by_cell = pd.Series(_t0_pr, index=_t0_cellkey)
        _pr_by_cell = _pr_by_cell[~_pr_by_cell.index.duplicated(keep="first")]
        _pc_cellkey = (Pc["cur"].astype(str) + "|" + Pc["bin"].astype(str) + "|" + Pc["rpgt"].astype(str)
                       + "|" + Pc["pmp"].astype(str) + "|" + Pc["ctry"].astype(str) + "|"
                       + Pc["per"].astype(str)).to_numpy()[pc_keep]
        self._pc_prapp = (_pr_by_cell.reindex(_pc_cellkey).fillna(0.0).to_numpy(float)
                          if len(pc_keep) else np.zeros(0, float))

        # band order + per-band groupings (vectorised map to band index)
        self.band_order = sorted(bandset)
        self._B = len(self.band_order)
        _bpos = {b: k for k, b in enumerate(self.band_order)}
        self._pc_bandcol = (pd.Series(list(zip(pc_mid_k.tolist(), pc_per_k.tolist())))
                            .map(_bpos).to_numpy(np.int64) if len(pc_keep) else np.zeros(0, np.int64))
        Rmid = R["midl"].to_numpy(); Rper = R["per"].to_numpy().astype(int)
        cap_mask = (R["iscap"].to_numpy(bool)
                    & pd.MultiIndex.from_arrays([Rmid, Rper]).isin(_bidx)) if nR else np.zeros(0, bool)
        self._t_rows = np.where(cap_mask)[0]
        self._t_bandcol = (pd.Series(list(zip(Rmid[self._t_rows].tolist(),
                                              Rper[self._t_rows].tolist()))).map(_bpos).to_numpy(np.int64)
                           if len(self._t_rows) else np.zeros(0, np.int64))

    # [FN-020]
    def project_pop_from_props(self, props):
        kpos = {k: j for j, k in enumerate(self.prop_keys)}
        pr = np.zeros((len(props), self._K), dtype=float)
        for r, prop in enumerate(props):
            for k, v in prop.items():
                kk = _prop_key_str(k, self.by_rpgt, getattr(self, "by_subcell", False))
                j = kpos.get(kk)
                if j is not None:
                    pr[r, j] += float(v)
        return self.project_pop(pr)

    # [FN-021]
    def _nb_arrays(self):
        """Cast static arrays to the numba kernel's dtypes once; pre-filter excl txn rows
        (their contribution is 0, so dropping them is exact)."""
        if getattr(self, "_nbcache", None) is None:
            keep = (~self._excl[self._t_rows]) if len(self._t_rows) else np.zeros(0, bool)
            # appearance-timed held factor per reduced Pc row: fcp[origin]·pro_rata[appearance]
            # (guard origin<0 → 0; the kernel ignores those rows but avoids negative-index wrap).
            _oi = np.where(self._pc_org >= 0, self._pc_org, 0)
            _heldfac = np.where(self._pc_org >= 0, self._fcp[_oi] * self._pc_prapp, 0.0)
            self._nbcache = (
                self._propidx.astype(np.int64),
                (self._excl | self._emask),
                self._gcode.astype(np.int64),
                self._base.astype(np.float64), self._mv.astype(np.float64),
                self._vcpos.astype(np.float64), self._ctot.astype(np.float64),
                self._pc_org.astype(np.int64), self._pc_vc.astype(np.float64),
                self._pc_pool.astype(np.float64), self._pc_bandcol.astype(np.int64),
                _heldfac.astype(np.float64),
                self._t_rows[keep].astype(np.int64), self._t_bandcol[keep].astype(np.int64))
        return self._nbcache

    # [FN-022]
    def _nb_buffers(self, P):
        """Pre-allocated working buffers for the numba kernel, cached & REUSED across calls
        (removes tens of MB of per-generation alloc/free). The big scratch (psum/vpsum/moved
        sized ncell; pr/pshare/vshare/mvrow sized nR) is P-INDEPENDENT so it's allocated ONCE;
        only vamp/txn (P×B, tiny) reallocate when P changes (λ eval_ov vs P=1 score_of). The
        returned vamp/txn ARE these buffers — the caller must consume them before the next call
        (the search's _bands_pen reads them straight into the penalty, so that holds)."""
        # Reallocate when missing OR read-only. When this projector is shipped to a loky
        # worker, joblib memmaps its cached ndarrays as READ-ONLY, but the numba kernel
        # writes into these scratch buffers (psum[:]=0.0 …) → 'Cannot modify readonly array'
        # TypingError. A writeable re-alloc in the worker (re-cached on the worker's own copy,
        # reused for its lifetime) fixes it; the main process keeps its original buffers.
        # Numerically identical — same np.zeros scratch, fully overwritten each call.
        # NB: must test EVERY buffer, not just the first. joblib only memmaps arrays over its
        # size threshold (~1 MB), so the small per-cell scratch (psum/vpsum/moved, sized ncell)
        # can stay writeable while the large per-row scratch (pr/pshare/vshare/mvrow, sized nR)
        # is memmapped read-only — checking fixed[0] alone would miss that and the kernel fails.
        fixed = getattr(self, "_nbbuf_fixed", None)
        if fixed is None or not all(b.flags.writeable for b in fixed):
            nR = len(self._gcode); ncell = int(self._ngc)
            fixed = (np.zeros(ncell), np.zeros(ncell), np.zeros(ncell),     # psum, vpsum, moved
                     np.zeros(nR), np.zeros(nR), np.zeros(nR), np.zeros(nR),  # pr, pshare, vshare, mvrow
                     np.zeros(ncell), np.zeros(ncell), np.zeros(ncell))     # nzc, exc, rsum (cap water-fill)
            self._nbbuf_fixed = fixed
        vt = getattr(self, "_nbbuf_vt", None)
        if vt is None or vt[0] != int(P) or not (vt[1].flags.writeable and vt[2].flags.writeable):
            B = int(self._B)
            vt = (int(P), np.zeros((int(P), B)), np.zeros((int(P), B)))     # vamp, txn (outputs)
            self._nbbuf_vt = vt
        return (vt[1], vt[2]) + fixed

    # [FN-023]
    def project_pop_numba(self, prop_raw: np.ndarray):
        """Numba-accelerated project_pop — bit-identical, ~7× faster on the real scaffold.
        Falls back to the NumPy path if numba is unavailable or the scaffold is empty.
        Working arrays are pooled (see _nb_buffers); consume the result before the next call."""
        prop_raw = np.ascontiguousarray(prop_raw, dtype=np.float64)
        if not _HAVE_NUMBA or not len(self._gcode):
            return self.project_pop(prop_raw)
        a = self._nb_arrays()
        buf = self._nb_buffers(prop_raw.shape[0])
        return _pop_band_kernel(prop_raw, *a, int(self._ngc), int(self._B), float(self._cap), *buf)

    # [FN-024]
    def _cellsum(self, x):
        """(P, nR) -> (P, ngc) segment sum over cell codes via sparse matmul (C-fast)."""
        return np.asarray((self._S @ x.T).T)

    # [FN-024b]
    def _cap_pshare(self, pshare, act):
        """Per-sub-cell max-share water-fill on the (P, nR) normalised routed share `pshare`
        (sums to 1 per active cell). Bit-faithful to build_split_exports._cap_rows: only cells
        with >=2 non-zero routed gateways are capped; excess over `self._cap` is redistributed to
        the OTHER present gateways proportional to their remaining room, up to 50 sweeps. Inactive
        cells (act=False) are never touched. Returns a new capped array (input unmodified)."""
        cap = self._cap
        if cap >= 1.0:
            return pshare
        gc = self._gcode
        W = pshare.copy()
        # cappable = active cell with >=2 non-zero routed gateways (guard read ONCE, pre-loop)
        nz = self._cellsum((W > 1e-12).astype(float))[:, gc]
        cappable = act & (nz >= 2.0)
        if not cappable.any():
            return pshare
        for _ in range(50):
            over = (W > cap + 1e-12) & cappable
            if not over.any():
                break
            excess = self._cellsum(np.where(over, W - cap, 0.0))[:, gc]
            W = np.where(over, cap, W)
            recip = cappable & (~over) & (W > 1e-12) & (W < cap - 1e-12)
            room = np.where(recip, cap - W, 0.0)
            rs = self._cellsum(room)[:, gc]
            W = W + np.where(rs > 1e-12, room / np.where(rs > 1e-12, rs, 1.0) * excess, 0.0)
        return np.where(cappable, W, pshare)

    # [FN-025]
    def project_pop(self, prop_raw: np.ndarray):
        """prop_raw : (P, K) proposed share per `prop_keys`. Returns (vamp[P,B], txn[P,B])."""
        prop_raw = np.asarray(prop_raw, float)
        P = prop_raw.shape[0]
        if not len(self._gcode):                     # no constrained cells this build
            return np.zeros((P, self._B)), np.zeros((P, self._B))
        pr = prop_raw[:, self._propidx]                         # (P, nR)
        pr = np.where((self._excl | self._emask)[None, :], 0.0, pr)
        base = self._base[None, :]; mv_s = self._mv[None, :]
        ctot = self._ctot[None, :]

        psum = self._cellsum(pr)[:, self._gcode]                # (P, nR)
        act = psum > 0
        pshare = np.where(act, np.divide(pr, psum, out=np.zeros_like(pr), where=act), base)
        if self._cap < 1.0:                                     # per-sub-cell max-share cap
            pshare = self._cap_pshare(pshare, act)
        mv = np.where(act, mv_s, 0.0)
        # vshare from the (capped) ROUTED share only — 0 in inactive cells so no pool leaks there.
        routed = np.where(act, pshare, 0.0)
        vpr = routed * self._vcpos[None, :]
        vpsum = self._cellsum(vpr)[:, self._gcode]
        vact = vpsum > 0
        vshare = np.divide(vpr, vpsum, out=np.zeros_like(vpr), where=vact)

        moved_tot = self._cellsum(base * mv)[:, self._gcode]    # (P, nR)
        ptxn = ctot * (base * (1.0 - mv) + moved_tot * pshare)
        ptxn = np.where(self._excl[None, :], 0.0, ptxn)

        # VAMP over reduced aged rows — APPEARANCE-MONTH timing (see __init__): the movable
        # fraction is fcp[origin]·pro_rata[appearance], gated on the ORIGIN cell being routed
        # (act at the origin t0 row). The pool is pre-built appearance-timed upstream.
        o = self._pc_org
        ok = o >= 0
        oi = np.where(ok, o, 0)
        _heldfac = (self._fcp[oi] * self._pc_prapp)          # (nPc,) appearance-timed held factor
        move_pc = np.where(ok[None, :], act[:, oi] * _heldfac[None, :], 0.0)
        psh_pc = np.where(ok[None, :], vshare[:, oi], 0.0)
        vp = self._pc_vc[None, :] * (1.0 - move_pc) + self._pc_pool[None, :] * psh_pc

        P = prop_raw.shape[0]
        vamp = np.zeros((P, self._B)); txn = np.zeros((P, self._B))
        if len(self._pc_bandcol):
            np.add.at(vamp.T, self._pc_bandcol, vp.T)
        if len(self._t_bandcol):
            np.add.at(txn.T, self._t_bandcol, ptxn[:, self._t_rows].T)
        return vamp, txn
