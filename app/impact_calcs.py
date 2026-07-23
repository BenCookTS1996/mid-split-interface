"""Impact-tab calculations extracted from streamlit_app.py (behaviour unchanged):
VAMP pre/post projection from the saved exports, wallet-capability lookup, the
production split-template builder, and the small Streamlit cache wrappers that keep
the Impact tab fast. Kept here to keep streamlit_app.py smaller and more organised."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

__build__ = "2026-07-22-enforced-split-frame-for-eval"


def build_kill_eff(vamp2fids, fid_eff):
    """Build the hashable effective-date switch-off map for the projection.

    vamp2fids: {vampMid: set(normalised gatewayFid)}.
    fid_eff:   {normalised gatewayFid: effective_date str} for target=0 gateways.

    A vampMid is switched off only when EVERY gatewayFid mapping to it is off
    (same rule as excluded_mids). Its effective date is the LATEST among those
    gateways (it is fully off only once the last one switches off). Returns a
    sorted tuple of (vampMid, 'YYYY-MM-DD') so it is hashable for st.cache_data.
    """
    off = set(fid_eff or {})
    out = []
    for v, fids in (vamp2fids or {}).items():
        if fids and set(fids) <= off:
            ds = [pd.to_datetime(fid_eff[f], errors="coerce") for f in fids]
            ds = [d for d in ds if pd.notna(d)]
            if ds:
                out.append((str(v), str(max(ds).date())))
    return tuple(sorted(out))


def _mid_keep_fraction(vampmid_series, period_series, kill_eff, month_0):
    """Per-row RETAINED fraction (1 − kill) for effective-date-gated switch-offs.

    A switched-off vampMid keeps its full volume before its effective month, a
    mid-month pro-rated fraction in the effective month, and 0 afterwards — the
    same mid-month pro-rate the go-live phasing uses. Rows for vampMids not in
    kill_eff keep 1.0. `period` is the origination month index (0 = Month 0)."""
    import calendar as _cal
    n = len(vampmid_series)
    keep = np.ones(n, dtype=float)
    eff = {str(m): pd.to_datetime(d, errors="coerce") for m, d in (kill_eff or ())}
    eff = {m: d for m, d in eff.items() if pd.notna(d)}
    if not eff or month_0 is None:
        return keep
    m0 = pd.to_datetime(month_0)
    mids = np.asarray(vampmid_series, dtype=object)
    pers = np.asarray(period_series)
    cache = {}
    for i in range(n):
        ed = eff.get(str(mids[i]))
        if ed is None:
            continue
        per = int(pers[i])
        key = (str(mids[i]), per)
        if key not in cache:
            dt = m0 + pd.DateOffset(months=per)
            days = _cal.monthrange(dt.year, dt.month)[1]
            s = dt.replace(day=1)
            e = s + pd.Timedelta(days=days)
            if ed <= s:            # off for the whole month
                kf = 1.0
            elif ed >= e:          # not yet off this month
                kf = 0.0
            else:                  # switches off mid-month
                kf = (e - ed).days / days
            cache[key] = 1.0 - kf
        keep[i] = cache[key]
    return keep


@st.cache_data(show_spinner=False)
def compute_vamp_post_by_mid(tp_path, prop_items, month_0, go_live, excluded_mids=frozenset(),
                             kill_eff=()):
    """Derive the proposed-split VAMP forecast from the saved baseline export.

    NON-INVASIVE: re-scales vamp_t_period_export.csv's baseline (VAMP_Pre /
    VI_Txn_Pre) by each MID's proposed-vs-baseline transaction volume, phased in
    from the Split Go Live date. Transactions are conserved per cell; VAMPs move
    with the reallocated volume. The VAMP pipeline / actuarial engine is NOT run.

    prop_items: tuple of (Currency, BIN, vampMid, proposed_share) for the split.
    Returns a per-vampMid frame with VAMP M0-5 / VI Txn M0-5 and _Post variants.
    """
    import calendar as _cal
    tp = pd.read_csv(tp_path)
    for c in ["Currency", "vampMid"]:
        tp[c] = tp[c].astype(str).str.strip()
    tp["Currency"] = tp["Currency"].str.lower()
    tp["BIN"] = tp["BIN"].astype(str).str.strip()
    m0 = pd.to_datetime(month_0)
    gl = pd.to_datetime(go_live)

    prop = pd.DataFrame(list(prop_items), columns=["Currency", "BIN", "vampMid", "prop"])
    if not prop.empty:
        prop["Currency"] = prop["Currency"].astype(str).str.strip().str.lower()
        prop["BIN"] = prop["BIN"].astype(str).str.strip()
        prop["vampMid"] = prop["vampMid"].astype(str).str.strip()
        denom = prop.groupby(["Currency", "BIN"])["prop"].transform("sum").replace(0, np.nan)
        prop["prop"] = prop["prop"] / denom

    def _frac_after(m):
        if m < 0:
            return 0.0
        dt = m0 + pd.DateOffset(months=int(m))
        days = _cal.monthrange(dt.year, dt.month)[1]
        s = dt.replace(day=1)
        e = s + pd.Timedelta(days=days)
        return 1.0 if gl <= s else (0.0 if gl >= e else (e - gl).days / days)
    frac = {m: _frac_after(m) for m in range(-9, 6)}

    t0 = (tp[tp["t"] == 0]
          .groupby(["Currency", "BIN", "vampMid", "period"], as_index=False)
          .agg(pre_txn=("VI_Txn_Pre", "sum")))
    ct = t0.groupby(["Currency", "BIN", "period"], as_index=False).agg(cell_tot=("pre_txn", "sum"))
    t0 = t0.merge(ct, on=["Currency", "BIN", "period"]).merge(prop, on=["Currency", "BIN", "vampMid"], how="left")
    t0["f"] = t0["period"].map(frac)
    # vampMids switched off via gateway_volume_overrides are removed from BOTH the
    # pre-go-live retention and the proposed split; their volume redistributes to
    # the active gateways in the cell (transactions still conserved). The removal is
    # gated by each switch-off's effective_date (kill_eff): a switched-off vampMid
    # keeps its volume until its effective month, then drops (mid-month pro-rated).
    _keep = _mid_keep_fraction(t0["vampMid"], t0["period"], kill_eff, month_0)
    _dated = {m for m, _ in (kill_eff or ())}
    _binary = t0["vampMid"].isin(excluded_mids) & ~t0["vampMid"].isin(_dated)
    t0["_keep"] = np.where(_binary, 0.0, _keep)
    _have = t0["prop"].notna() & (t0["_keep"] > 0.0)
    t0["_active_pre"] = t0["pre_txn"] * t0["_keep"]
    t0["_active_tot"] = t0.groupby(["Currency", "BIN", "period"])["_active_pre"].transform("sum")
    t0["base_share"] = np.where(t0["_active_tot"] > 0, t0["_active_pre"] / t0["_active_tot"], 0.0)
    t0["prop_eff"] = np.where(_have, t0["prop"].fillna(0.0) * t0["_keep"], 0.0)
    t0["_prop_sum"] = t0.groupby(["Currency", "BIN", "period"])["prop_eff"].transform("sum")
    t0["prop_eff"] = np.where(t0["_prop_sum"] > 0, t0["prop_eff"] / t0["_prop_sum"], t0["base_share"])
    t0["post_txn"] = t0["cell_tot"] * ((1 - t0["f"]) * t0["base_share"] + t0["f"] * t0["prop_eff"])
    t0["r"] = np.where(t0["pre_txn"] > 0, t0["post_txn"] / t0["pre_txn"], 1.0)

    # VAMP conserved & redistributed by the volume share (pipeline-faithful). This legacy
    # non-prorata fallback has no fcp data, so the movable slice is the go-live fraction only.
    t0["_move"] = np.where(t0["_prop_sum"] > 0, t0["f"], 0.0)
    # VAMP redistribution only across MIDs that carry baseline VAMP (zero-VAMP MIDs stay 0).
    _midv = tp.groupby(["Currency", "BIN", "vampMid"])["VAMP_Pre"].sum().rename("_midv").reset_index()
    t0 = t0.merge(_midv, on=["Currency", "BIN", "vampMid"], how="left")
    t0["_vprop"] = t0["prop_eff"] * (t0["_midv"].fillna(0.0) > 0).astype(float)
    t0["_vpsum"] = t0.groupby(["Currency", "BIN", "period"])["_vprop"].transform("sum")
    t0["_vshare"] = np.where(t0["_vpsum"] > 0, t0["_vprop"] / t0["_vpsum"], 0.0)
    tp["orig_m"] = tp["period"] - tp["t"]
    _mv = t0[["Currency", "BIN", "vampMid", "period", "_move", "_vshare"]].rename(
        columns={"period": "orig_m", "_vshare": "_pshare"})
    tp["_cell_vamp"] = tp.groupby(["Currency", "BIN", "period", "t"])["VAMP_Pre"].transform("sum")
    tp = tp.merge(_mv, on=["Currency", "BIN", "vampMid", "orig_m"], how="left")
    tp["_move"] = tp["_move"].fillna(0.0)
    tp["_pshare"] = tp["_pshare"].fillna(0.0)
    tp["VAMP_Post_c"] = tp["VAMP_Pre"] * (1.0 - tp["_move"]) + tp["_cell_vamp"] * tp["_move"] * tp["_pshare"]

    vamp_pre = tp.groupby(["vampMid", "period"])["VAMP_Pre"].sum().unstack(fill_value=0.0)
    vamp_post = tp.groupby(["vampMid", "period"])["VAMP_Post_c"].sum().unstack(fill_value=0.0)
    txn_pre = t0.groupby(["vampMid", "period"])["pre_txn"].sum().unstack(fill_value=0.0)
    txn_post = t0.groupby(["vampMid", "period"])["post_txn"].sum().unstack(fill_value=0.0)

    out = pd.DataFrame({"vampMid": sorted(tp["vampMid"].unique())}).set_index("vampMid")
    for m in range(6):
        out[f"VAMP M{m}"] = vamp_pre[m] if m in vamp_pre.columns else 0.0
        out[f"VI Txn M{m}"] = txn_pre[m] if m in txn_pre.columns else 0.0
        out[f"VAMP Post M{m}"] = vamp_post[m] if m in vamp_post.columns else 0.0
        out[f"VI Txn Post M{m}"] = txn_post[m] if m in txn_post.columns else 0.0
    return out.fillna(0.0).reset_index()


@st.cache_data(show_spinner=False)
def compute_vamp_post_from_prorata(pp_path, prop_items, excluded_mids=frozenset(),
                                   kill_eff=(), month_0=None, scoped_rpgts=()):
    """Accurate proposed-split VAMP forecast using the pipeline pro-rata export.

    Method (per your spec): aggregate the baseline export EXCLUDING vampMid to get
    the per (Currency, BIN, RPGT, period) transaction pool, then redistribute the
    POST-go-live portion (the export's `pro_rata`, which the pipeline computed with
    its own RPGT-aware mid-month weighting) across gateways by the proposed share;
    the pre-go-live portion keeps the current split. Each gateway's baseline VAMPs
    scale by its resulting volume change at ORIGINATION month. No pipeline re-run.

    scoped_rpgts: if non-empty, the proposed split is applied ONLY to these RPGTs;
    every other RPGT is held at its current baseline split (post == pre). Empty ->
    the split applies to all RPGTs (the Currency x Bank decision hits every RPGT).
    """
    return _vamp_post_core(pd.read_csv(pp_path), prop_items, excluded_mids, kill_eff,
                           month_0, scoped_rpgts)


def _vamp_post_core(pp, prop_items, excluded_mids=frozenset(), kill_eff=(), month_0=None,
                    scoped_rpgts=()):
    """Core projection on a PRE-LOADED pro-rata dataframe, so the per-MID cap
    feedback loop can re-project candidate splits without re-reading the CSV."""
    pp = pp.copy()
    pp["Currency"] = pp["Currency"].astype(str).str.strip().str.lower()
    pp["BIN"] = pp["BIN"].astype(str).str.strip()
    pp["vampMid"] = pp["vampMid"].astype(str).str.strip()
    rpgt_col = "RPGT" if "RPGT" in pp.columns else "rpgt"
    pp["RPGT"] = pp[rpgt_col].astype(str)
    pp["pro_rata"] = pd.to_numeric(pp.get("pro_rata", 0.0), errors="coerce").fillna(0.0)
    # fcp1_frac: fraction of the cell the pipeline actually reroutes (fcpNumber==1 /
    # attempt==1 for restricted RPGTs). Missing (old export) -> 1.0 = prior behaviour.
    pp["fcp1_frac"] = pd.to_numeric(pp.get("fcp1_frac", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)

    # The export may now be split by paymentMethodProvider (wallet-aware). This
    # projection doesn't use it, so collapse it to one row per t-period key -
    # otherwise the rmap merge below fans out and double-counts.
    if "paymentMethodProvider" in pp.columns:
        pp = pp.groupby(["vampMid", "RPGT", "BIN", "Currency", "period", "t"], as_index=False).agg(
            vampCount=("vampCount", "sum"), VI_Txn_Count=("VI_Txn_Count", "sum"),
            pro_rata=("pro_rata", "first"), fcp1_frac=("fcp1_frac", "first"))

    # prop_items are 4-tuples (Currency, BIN, vampMid, prop_raw) at Bank×Currency grain,
    # or 5-tuples (Currency, BIN, RPGT, vampMid, prop_raw) at Bank×Currency×RPGT grain, so a
    # per-RPGT split is projected PER RPGT rather than one share applied across every RPGT.
    _pi = list(prop_items)
    _by_rpgt = bool(_pi) and len(_pi[0]) == 5
    prop = pd.DataFrame(_pi, columns=(["Currency", "BIN", "RPGT", "vampMid", "prop_raw"]
                                      if _by_rpgt else ["Currency", "BIN", "vampMid", "prop_raw"]))
    if not prop.empty:
        prop["Currency"] = prop["Currency"].astype(str).str.strip().str.lower()
        prop["BIN"] = prop["BIN"].astype(str).str.strip()
        prop["vampMid"] = prop["vampMid"].astype(str).str.strip()
        if _by_rpgt:
            prop["_rpgtl"] = prop["RPGT"].astype(str).str.strip().str.lower()
            prop = prop.drop(columns=["RPGT"])

    grp = ["Currency", "BIN", "RPGT", "period"]
    if _by_rpgt:
        _t0 = pp[pp["t"] == 0].copy()
        _t0["_rpgtl"] = _t0["RPGT"].astype(str).str.strip().str.lower()
        t0 = _t0.merge(prop, on=["Currency", "BIN", "_rpgtl", "vampMid"], how="left").drop(columns=["_rpgtl"])
    else:
        t0 = pp[pp["t"] == 0].merge(prop, on=["Currency", "BIN", "vampMid"], how="left")
    t0["prop_raw"] = t0["prop_raw"].fillna(0.0)
    # vampMids switched off via gateway_volume_overrides are excluded from BOTH the
    # pre-go-live retention and the proposed split; the cell total is unchanged so
    # their volume is redistributed to the active gateways (transactions conserved).
    # The removal is gated by each switch-off's effective_date (kill_eff): a
    # switched-off vampMid keeps its volume until its effective month, then drops
    # (mid-month pro-rated). vampMids in excluded_mids with no effective date are
    # removed for all periods (binary fallback).
    _keep = _mid_keep_fraction(t0["vampMid"], t0["period"], kill_eff, month_0)
    _dated = {m for m, _ in (kill_eff or ())}
    _binary = t0["vampMid"].isin(excluded_mids) & ~t0["vampMid"].isin(_dated)
    t0["_keep"] = np.where(_binary, 0.0, _keep)
    t0["prop_raw"] = t0["prop_raw"] * t0["_keep"]
    t0["_active_vi"] = t0["VI_Txn_Count"] * t0["_keep"]
    # Group the cell keys ONCE and reuse for all three per-cell sums (the 5-col key is
    # otherwise re-factorised per groupby). Bit-identical to grouping separately.
    _g = t0.groupby(grp)
    t0["cell_tot"] = _g["VI_Txn_Count"].transform("sum")
    t0["_active_tot"] = _g["_active_vi"].transform("sum")
    t0["base_share"] = np.where(t0["_active_tot"] > 0, t0["_active_vi"] / t0["_active_tot"], 0.0)
    # Renormalise proposed shares over the gateways present in each cell so the
    # redistribution conserves the cell's transactions; if no proposed shares map
    # to this cell, fall back to the current (baseline) split.
    t0["prop_sum"] = _g["prop_raw"].transform("sum")
    t0["prop_share"] = np.where(t0["prop_sum"] > 0, t0["prop_raw"] / t0["prop_sum"], t0["base_share"])
    # Movable fraction = go-live pro-rata × fcp1 cohort fraction: only that slice of the
    # cell takes the proposed share; the rest (pre-go-live + FCP2+/retries) stays baseline.
    # PER-MID movable fraction (fcp1_frac is per-vampMid now): move_mid = pro_rata × fcp1_frac.
    t0["_move"] = np.where(t0["prop_sum"] > 0, t0["pro_rata"] * t0["fcp1_frac"], 0.0)
    # VAMP redistribution share: proposed share renormalised over MIDs that CARRY baseline
    # VAMP, so zero-VAMP MIDs never receive VAMP and the cell's VAMP total is conserved.
    t0["_vprop"] = t0["prop_raw"] * (t0["vampCount"] > 0).astype(float)
    t0["_vpsum"] = t0.groupby(grp)["_vprop"].transform("sum")
    t0["_vshare"] = np.where(t0["_vpsum"] > 0, t0["_vprop"] / t0["_vpsum"], 0.0)

    # RPGT scope: hold RPGTs OUTSIDE the scoped set at their current baseline split
    # (post == pre) — they simply don't move.
    if scoped_rpgts:
        _scope = {str(r).strip().lower() for r in scoped_rpgts}
        _oos = ~t0["RPGT"].astype(str).str.strip().str.lower().isin(_scope)
        t0.loc[_oos, "_move"] = 0.0

    # TWO-COHORT volume (pipeline-faithful): each MID keeps (1-move_mid) of its OWN volume
    # on its own gateway; the pooled movable slice (Σ base_share × move) is redistributed by
    # the proposed share. Per-MID move captures that FCP2+/retry-heavy MIDs move less.
    t0["_bm"] = t0["base_share"] * t0["_move"]
    t0["_moved_tot"] = t0.groupby(grp)["_bm"].transform("sum")
    t0["post_txn"] = t0["cell_tot"] * (t0["base_share"] * (1 - t0["_move"])
                                       + t0["_moved_tot"] * t0["prop_share"])

    _mv = t0[["Currency", "BIN", "RPGT", "vampMid", "period", "_move", "_vshare"]].rename(
        columns={"period": "orig_m", "_vshare": "_pshare"})
    pp["orig_m"] = pp["period"] - pp["t"]
    pp = pp.merge(_mv, on=["Currency", "BIN", "RPGT", "vampMid", "orig_m"], how="left")
    pp["_move"] = pp["_move"].fillna(0.0)
    pp["_pshare"] = pp["_pshare"].fillna(0.0)
    # TWO-COHORT VAMP: hold (1-move) of each MID's VAMP; the pooled moved VAMP
    # (Σ vampCount × move) is redistributed by the VAMP-carrying proposed share.
    pp["_moved_v"] = pp["vampCount"] * pp["_move"]
    pp["_moved_vpool"] = pp.groupby(["Currency", "BIN", "RPGT", "period", "t"])["_moved_v"].transform("sum")
    pp["VAMP_Post_c"] = pp["vampCount"] * (1.0 - pp["_move"]) + pp["_moved_vpool"] * pp["_pshare"]

    vamp_pre = pp.groupby(["vampMid", "period"])["vampCount"].sum().unstack(fill_value=0.0)
    vamp_post = pp.groupby(["vampMid", "period"])["VAMP_Post_c"].sum().unstack(fill_value=0.0)
    txn_pre = t0.groupby(["vampMid", "period"])["VI_Txn_Count"].sum().unstack(fill_value=0.0)
    txn_post = t0.groupby(["vampMid", "period"])["post_txn"].sum().unstack(fill_value=0.0)
    out = pd.DataFrame({"vampMid": sorted(pp["vampMid"].unique())}).set_index("vampMid")
    for m in range(6):
        out[f"VAMP M{m}"] = vamp_pre[m] if m in vamp_pre.columns else 0.0
        out[f"VI Txn M{m}"] = txn_pre[m] if m in txn_pre.columns else 0.0
        out[f"VAMP Post M{m}"] = vamp_post[m] if m in vamp_post.columns else 0.0
        out[f"VI Txn Post M{m}"] = txn_post[m] if m in txn_post.columns else 0.0
    return out.fillna(0.0).reset_index()


def _dump_projection_diag(t0, pp_path, prop_items, enforced, by_rpgt):
    """EXCESSIVE diagnostics for the tab-3 vs tab-5 back-fill gap. Writes two files next to the
    pro-rata export: _proj_diag_rows.csv (every t0 sub-cell with all intermediates) and
    _proj_diag_summary.txt (prop_sum-per-cell stats, coarse-fallback/back-fill counts, a per-
    vampMid pre/post table, and a per-cell breakdown of every zero-baseline recipient). Never
    raises — diagnostics must not break the projection. OFF by default (heavy: writes a ~170MB
    rows CSV + reads the routed export/rules/mapping); set env ROUTING_PROJ_DIAG=1 to enable."""
    import os as _os
    if _os.environ.get("ROUTING_PROJ_DIAG", "0") != "1":
        return
    try:
        import datetime as _dt
        _dir = _os.path.dirname(_os.path.abspath(pp_path))
        _cols = [c for c in ["vampMid", "RPGT", "BIN", "Currency", "_pmp", "_ctry", "period",
                             "vampCount", "VI_Txn_Count", "cell_tot", "_at", "base_share",
                             "fcp1_frac", "pro_rata", "prop_raw", "prop_sum", "prop_share",
                             "_move", "_bm", "_moved_tot", "_vshare", "post_txn", "_keep",
                             "_psum_pre", "_prop_from_coarse", "_bf_inj"] if c in t0.columns]
        t0.sort_values(["Currency", "BIN", "RPGT", "period", "vampMid"])[_cols].to_csv(
            _os.path.join(_dir, "_proj_diag_rows.csv"), index=False)

        # ---- PER-GATEWAY (enforced-prop) SHARE DUMP: the EXACT shares the projection feeds in,
        # before the vampMid collapse — write to _proj_diag_enforced_prop.csv so it can be diffed
        # directly against the downloaded template (map its gateway columns → vampMid and compare).
        try:
            _pi = list(prop_items)
            if _pi:
                _n = len(_pi[0])
                _pcols = (["Currency", "BIN", "RPGT", "pmp", "Country", "vampMid", "prop_raw"] if _n == 7
                          else ["Currency", "BIN", "RPGT", "vampMid", "prop_raw"] if _n == 5
                          else ["Currency", "BIN", "vampMid", "prop_raw"])
                pd.DataFrame(_pi, columns=_pcols).to_csv(
                    _os.path.join(_os.path.dirname(_os.path.abspath(pp_path)),
                                  "_proj_diag_enforced_prop.csv"), index=False)
        except Exception:  # noqa: BLE001
            pass

        # ---- TARGETED CELL TRACE: full step-by-step for specific cell(s), written to
        # _proj_diag_trace.txt. Configure via env ROUTING_PROJ_TRACE = "currency|bin|rpgt"
        # (multiple separated by ';'); defaults to the WoodForest addon-sale cell under review.
        # Shows EVERY vampMid in the cell so you can see WoodForest's share vs the others and how
        # post_txn = cell_tot·(base_share·(1−move) + moved_tot·prop_share) is formed.
        try:
            _spec = _os.environ.get("ROUTING_PROJ_TRACE", "usd|400022|addon sale")
            _tl = []
            _tl.append(f"CELL TRACE  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}   spec='{_spec}'")
            _tcols = [c for c in ["period", "_pmp", "_ctry", "vampMid", "VI_Txn_Count", "cell_tot",
                                  "base_share", "fcp1_frac", "pro_rata", "prop_raw", "_psum_pre",
                                  "prop_sum", "prop_share", "_move", "_moved_tot", "post_txn",
                                  "_prop_from_coarse", "_bf_inj"] if c in t0.columns]
            for _s in _spec.split(";"):
                _p = [x.strip() for x in _s.split("|")]
                if len(_p) < 3:
                    continue
                _cur, _bin, _rp = _p[0].lower(), _p[1], _p[2].lower()
                _m = t0[(t0["Currency"].astype(str).str.lower() == _cur)
                        & (t0["BIN"].astype(str).str.strip() == _bin)
                        & (t0["RPGT"].astype(str).str.lower() == _rp)]
                _tl.append("")
                _tl.append(f"=== {_cur} / {_bin} / {_rp}  ({len(_m)} row(s)) ===")
                if _m.empty:
                    _tl.append("  (no rows — cell absent from the projection: check BIN/RPGT/currency "
                               "spelling, or the split doesn't route this cell)")
                    continue
                for _per in sorted(_m["period"].unique()):
                    _mp = _m[_m["period"] == _per]
                    _tl.append(f"  --- period {int(_per)} ---")
                    for _, r in _mp.sort_values("post_txn", ascending=False).iterrows():
                        _tl.append("   " + "  ".join(f"{c}={r[c]:.4f}" if isinstance(r[c], float)
                                                     else f"{c}={r[c]}" for c in _tcols))
                    _tl.append(f"   [cell totals] pre_VI={_mp['VI_Txn_Count'].sum():.2f} "
                               f"post_VI={_mp['post_txn'].sum():.2f}")
            with open(_os.path.join(_dir, "_proj_diag_trace.txt"), "w") as _tf:
                _tf.write("\n".join(str(x) for x in _tl))
        except Exception:  # noqa: BLE001
            pass

        # ---- AUTO-SAMPLE OF INCREASING PROFILES: find the cells where a chosen gateway's
        # addon-sale volume INCREASES in this projection (post VI > base VI) and dump each so the
        # tab-3-vs-tab-5 gap can be localised without hand-picking a BIN. For every selected
        # (Currency, BIN) cell it writes (a) per-period base-vs-post VI — directly comparable to
        # tab 5's monthly BIN row — and (b) the sub-cell decomposition showing WHERE the increase
        # comes from and whether each row is a back-fill injection (bf=1). Env-tunable:
        # ROUTING_PROJ_SAMPLE_MID (substring, default 'woodforest'),
        # ROUTING_PROJ_SAMPLE_RPGT (default 'addon sale'), ROUTING_PROJ_SAMPLE_N (default 8 cells).
        # Written to _proj_diag_sample.txt.
        try:
            _smid = _os.environ.get("ROUTING_PROJ_SAMPLE_MID", "woodforest").strip().lower()
            _srpgt = _os.environ.get("ROUTING_PROJ_SAMPLE_RPGT", "addon sale").strip().lower()
            _sn = int(_os.environ.get("ROUTING_PROJ_SAMPLE_N", "8") or "8")
            _sl = []
            _sl.append(f"INCREASING-PROFILE SAMPLE  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
            _sl.append(f"mid~'{_smid}'  rpgt='{_srpgt}'  top {_sn} (Currency,BIN) cells by net post-minus-base VI")
            _sl.append("For each BIN: compare per-period post VI to tab 5's monthly BIN row for this gateway;")
            _sl.append("the sub-cell rows show which pmp/Country sub-cell drives the increase (bf=1 => injected).")
            _w = t0[t0["vampMid"].astype(str).str.lower().str.contains(_smid, na=False)
                    & (t0["RPGT"].astype(str).str.lower() == _srpgt)].copy()
            if _w.empty:
                _sl.append(f"\n(no rows for mid~'{_smid}' rpgt='{_srpgt}' — check spelling / that the split routes it)")
            else:
                _w["_delta"] = _w["post_txn"].fillna(0.0) - _w["VI_Txn_Count"].fillna(0.0)
                _cellinc = (_w.groupby(["Currency", "BIN"], as_index=False)["_delta"].sum())
                _cellinc = _cellinc[_cellinc["_delta"] > 1e-6].sort_values("_delta", ascending=False).head(_sn)
                _sl.append(f"\n{len(_cellinc)} increasing cell(s) selected (of "
                           f"{int((_w.groupby(['Currency','BIN'])['_delta'].sum() > 1e-6).sum())} increasing total):")
                for _, cr in _cellinc.iterrows():
                    _sl.append(f"  {cr['Currency']}/{cr['BIN']}  net +{cr['_delta']:,.0f} VI")
                for _, cr in _cellinc.iterrows():
                    _cur, _bin = cr["Currency"], cr["BIN"]
                    _cw = _w[(_w["Currency"] == _cur) & (_w["BIN"] == _bin)]
                    _sl.append("")
                    _sl.append(f"================ {_cur} / {_bin} / {_srpgt} ================")
                    _ppv = _cw.groupby("period").agg(base=("VI_Txn_Count", "sum"),
                                                     post=("post_txn", "sum")).sort_index()
                    _sl.append("  per-period base vs post VI (compare 'post' to tab 5's BIN row):")
                    for _per, pr in _ppv.iterrows():
                        _sl.append(f"    P{int(_per)}: base={pr['base']:>9,.1f}  post={pr['post']:>9,.1f}"
                                   f"  d={pr['post'] - pr['base']:>+9,.1f}")
                    _sl.append("  sub-cell rows (base VI · cell_tot · prop_share · moved_tot · post · bf · coarse):")
                    for _, r in _cw.sort_values(["period", "post_txn"], ascending=[True, False]).iterrows():
                        _sl.append(f"    P{int(r['period'])} pmp={str(r.get('_pmp',''))[:9]:9s} "
                                   f"ctry={str(r.get('_ctry',''))[:8]:8s} base={r['VI_Txn_Count']:>8,.1f} "
                                   f"cell={r['cell_tot']:>9,.1f} pshare={float(r.get('prop_share', 0) or 0):.4f} "
                                   f"mov={float(r.get('_moved_tot', 0) or 0):.4f} post={r['post_txn']:>8,.1f} "
                                   f"bf={int(r.get('_bf_inj', 0) or 0)} coarse={int(r.get('_prop_from_coarse', 0) or 0)}")
            with open(_os.path.join(_dir, "_proj_diag_sample.txt"), "w") as _sf:
                _sf.write("\n".join(str(x) for x in _sl))
        except Exception:  # noqa: BLE001
            pass

        # ---- TAB3 vs TAB5 (routed) PER-CELL COMPARISON + ROOT-CAUSE SECTIONS. Auto-locates the
        # routed _validate export, the exported rules tab 5 read (PoolTargeted_Rules_*.xlsx), the
        # mapping_pct_export, the fid→vampMid map and the export manifest — all relative to pp_path.
        # Sections: (0) config/manifest echo, (A) per-vampMid Δ, (1) INPUT-SPLIT diff tab3-enforced
        # vs exported-rules, (2) full-cell side-by-side, (3) held/moved both sides, (4) finer-grain
        # renewal×fcp×attempt for the focus cell, (5) fcp1_frac provenance. Env: ROUTING_PROJ_TAB5CMP=0
        # to skip; ROUTING_PROJ_TAB5_EXPORT / _RULES_DIR / _BASIS overrides; ROUTING_PROJ_CMP_MID
        # (default 'braintree'). Writes _proj_diag_tab5_compare.txt. Heavy (reads 3M+ rows + 4 xlsx);
        # never raises.
        try:
            if _os.environ.get("ROUTING_PROJ_TAB5CMP", "1") != "0":
                import glob as _glob
                import json as _json
                _pn = _os.path.normpath(_os.path.abspath(pp_path)).split(_os.sep)
                _root = _os.sep.join(_pn[:_pn.index("data")]) if "data" in _pn else ""
                _sub = _pn[_pn.index("outputs") + 1:-1] if "outputs" in _pn else []
                if _sub and _sub[0] == "_validate":
                    _sub = _sub[1:]
                _R = _os.sep.join   # shorthand

                def _p(*parts):
                    return _R([_root] + list(parts)) if _root else _R(list(parts))
                _t5path = (_os.environ.get("ROUTING_PROJ_TAB5_EXPORT", "").strip()
                           or _p("data", "outputs", "_validate", *_sub, "vamp_t_period_export.csv"))
                _rules_dir = (_os.environ.get("ROUTING_PROJ_RULES_DIR", "").strip()
                              or _p("data", "rules", "_validate", *_sub))
                _map_path = _p("data", "outputs", "_validate", *_sub, "mapping_pct_export.csv")
                _mid_path = _p("data", "mappings", "Master_MID_List.csv")
                _man_path = _p("data", "exported_rules", "_export_manifest.json")
                _cmid = _os.environ.get("ROUTING_PROJ_CMP_MID", "braintree").strip().lower()

                def _kv(_s):
                    return _s.astype(str).str.strip().str.lower()

                def _cached_df(_cache, _srcs, _build):
                    # Return a cached DataFrame (pickle) if it's newer than every source; else
                    # rebuild + cache. Makes the slow xlsx/large-CSV reads a one-time cost.
                    try:
                        if _os.path.exists(_cache):
                            _cm = _os.path.getmtime(_cache)
                            if all(_os.path.exists(_s) and _os.path.getmtime(_s) <= _cm for _s in _srcs):
                                return pd.read_pickle(_cache), "cache"
                    except Exception:  # noqa: BLE001
                        pass
                    _df = _build()
                    try:
                        _df.to_pickle(_cache)
                    except Exception:  # noqa: BLE001
                        pass
                    return _df, "built"
                # fid -> vampMid map (for the rules xlsx, keyed by gatewayFid columns).
                _f2v = {}
                try:
                    _mdf = pd.read_csv(_mid_path)
                    _mdf.columns = [c.strip() for c in _mdf.columns]
                    _f2v = dict(zip(_mdf["gatewayFid"].astype(str).str.strip().str.lower(),
                                    _mdf["vampMid"].astype(str).str.strip()))
                except Exception:  # noqa: BLE001
                    pass

                _cl = []
                _cl.append(f"TAB3 vs TAB5 (routed) COMPARISON  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
                # ---- (0) CONFIG / MANIFEST ECHO (item 7): prove like-for-like. ----
                _cl.append(f"impact basis (env ROUTING_PROJ_BASIS): "
                           f"{_os.environ.get('ROUTING_PROJ_BASIS', 'unknown — set to No Compression / Compressed Rules')}")
                _cl.append(f"tab5 routed export: {_t5path}")
                _cl.append(f"exported rules dir: {_rules_dir}")
                try:
                    with open(_man_path) as _mf:
                        _man = _json.load(_mf)
                    _cl.append(f"export manifest: dial={_man.get('dial')} pools<={_man.get('max_pools')} "
                               f"engine={_man.get('engine')} brand={_man.get('brand')} "
                               f"go_live={_man.get('go_live')} max_share={_man.get('max_share')} "
                               f"built={_man.get('built_at')}  exp_sig={_man.get('exp_sig')}")
                except Exception:  # noqa: BLE001
                    _cl.append(f"export manifest: (not found at {_man_path})")

                if not _t5path or not _os.path.exists(_t5path):
                    _cl.append("(routed export not found — set ROUTING_PROJ_TAB5_EXPORT; skipping.)")
                else:
                    _u5 = {"vampMid", "RPGT", "BIN", "Currency", "paymentMethodProvider",
                           "Country", "period", "t", "VI_Txn_Pre", "VI_Txn_Post"}
                    _d5 = pd.read_csv(_t5path, usecols=lambda c: c.strip() in _u5, low_memory=False)
                    _d5.columns = [c.strip() for c in _d5.columns]
                    _d5 = _d5[_d5["t"] == 0].copy()      # VI_Txn lives at t=0
                    _routed = int(((_d5["VI_Txn_Post"] - _d5["VI_Txn_Pre"]).abs() > 1e-6).sum())
                    _cl.append(f"  routed export rows(t0)={len(_d5):,}  routed rows(Post!=Pre)={_routed:,}")
                    if _routed == 0:
                        _cl.append("  [WARN] routed export has ZERO routing (Post==Pre) — it is a BASELINE "
                                   "snapshot, NOT tab 5's routed output.")
                    _d5["_vml"] = _kv(_d5["vampMid"]); _d5["_rp"] = _kv(_d5["RPGT"])
                    _d5["_cur"] = _kv(_d5["Currency"]); _d5["_bn"] = _d5["BIN"].astype(str).str.strip()
                    _d5["_pm"] = _kv(_d5["paymentMethodProvider"]); _d5["_ct"] = _kv(_d5["Country"])
                    _K = ["_vml", "_rp", "_cur", "_bn", "_pm", "_ct", "period"]
                    _g5 = _d5.groupby(_K, observed=True).agg(
                        t5=("VI_Txn_Post", "sum"), t5_pre=("VI_Txn_Pre", "sum")).reset_index()
                    _tt = t0.copy()
                    _tt["_vml"] = _kv(_tt["vampMid"]); _tt["_rp"] = _kv(_tt["RPGT"])
                    _tt["_cur"] = _kv(_tt["Currency"]); _tt["_bn"] = _tt["BIN"].astype(str).str.strip()
                    _tt["_pm"] = _tt["_pmp"].astype(str).str.strip().str.lower()
                    _tt["_ct"] = _tt["_ctry"].astype(str).str.strip().str.lower()
                    _g3 = _tt.groupby(_K, observed=True).agg(
                        t3=("post_txn", "sum"), base=("VI_Txn_Count", "sum")).reset_index()
                    _cmp = _g3.merge(_g5, on=_K, how="outer").fillna(0.0)
                    _cmp["d"] = _cmp["t3"] - _cmp["t5"]
                    # ---- (A) per-vampMid × period headline: Δ = tab3 − tab5, worst |ΣΔ| first. ----
                    _pv = _cmp.groupby(["_vml", "period"]).agg(t3=("t3", "sum"), t5=("t5", "sum")).reset_index()
                    _pv["d"] = _pv["t3"] - _pv["t5"]
                    _mtot = (_pv.groupby("_vml").agg(ad=("d", lambda s: float(s.abs().sum())))
                             .reset_index().sort_values("ad", ascending=False))
                    _pers = sorted(int(p) for p in _pv["period"].unique())
                    _piv = _pv.pivot_table(index="_vml", columns="period", values="d", fill_value=0.0)
                    _cl.append("")
                    _cl.append("=== (A) per-vampMid Δ (tab3 − tab5) by period — worst |ΣΔ| first ===")
                    _cl.append("  vampMid                        " + " ".join(f"{('P' + str(p)):>9}" for p in _pers))
                    for _vm in _mtot["_vml"]:
                        _row = _piv.loc[_vm]
                        _cl.append(f"  {str(_vm)[:30]:30s} " + " ".join(f"{float(_row.get(p, 0.0)):>9,.0f}" for p in _pers))

                    # ---- (1) INPUT-SPLIT DIFF: tab3 enforced share vs the exported rules tab 5 read.
                    # Decides input-vs-application (and subsumes the compression-identity check). ----
                    _rules_share = None
                    if _os.environ.get("ROUTING_PROJ_RULESCMP", "1") == "0":
                        _cl.append("\n=== (1) INPUT-SPLIT DIFF: skipped (ROUTING_PROJ_RULESCMP=0) ===")
                    else:
                        try:
                            _rfiles = sorted(_glob.glob(_os.path.join(_rules_dir, "PoolTargeted_Rules_*.xlsx")))

                            def _build_rules():
                                _meta_cols = {"go live", "bin group", "brand", "rpgt", "currency", "bin",
                                              "paymentmethodprovider", "sticky", "country", "check", "dup check"}
                                _rr = []
                                for _rf in _rfiles:
                                    _rx = pd.read_excel(_rf)
                                    _rx.columns = [str(c).strip() for c in _rx.columns]
                                    _lc = {c.lower(): c for c in _rx.columns}
                                    if not all(k in _lc for k in ["rpgt", "currency", "bin", "paymentmethodprovider", "country"]):
                                        continue
                                    _gwc = [c for c in _rx.columns if c.lower() not in _meta_cols]
                                    _grp = [_lc["rpgt"], _lc["currency"], _lc["bin"], _lc["paymentmethodprovider"], _lc["country"]]
                                    _rg = _rx.groupby(_grp, observed=True)[_gwc].mean().reset_index()  # avg over STICKY dups
                                    _rl = _rg.melt(id_vars=_grp, value_vars=_gwc, var_name="fid", value_name="pct")
                                    _rl["_vml"] = _rl["fid"].astype(str).str.strip().str.lower().map(_f2v).fillna("").str.lower()
                                    _rl["_rp"] = _kv(_rl[_lc["rpgt"]]); _rl["_cur"] = _kv(_rl[_lc["currency"]])
                                    _rl["_bn"] = _rl[_lc["bin"]].astype(str).str.strip()
                                    _rl["_pm"] = _kv(_rl[_lc["paymentmethodprovider"]]); _rl["_ct"] = _kv(_rl[_lc["country"]])
                                    _rr.append(_rl[["_vml", "_rp", "_cur", "_bn", "_pm", "_ct", "pct"]])
                                _cols = ["_vml", "_rp", "_cur", "_bn", "_pm", "_ct", "rules_pct"]
                                if not _rr:
                                    return pd.DataFrame(columns=_cols)
                                return (pd.concat(_rr, ignore_index=True)
                                        .groupby(["_vml", "_rp", "_cur", "_bn", "_pm", "_ct"], observed=True)["pct"]
                                        .sum().reset_index().rename(columns={"pct": "rules_pct"}))
                            if _rfiles:
                                _rs, _src = _cached_df(_os.path.join(_dir, "_cache_rules_share.pkl"),
                                                       _rfiles, _build_rules)
                                _cl.append(f"  (rules parsed from {len(_rfiles)} xlsx, source={_src}; "
                                           "first build is slow, then cached by mtime)")
                                _rules_share = _rs if (_rs is not None and not _rs.empty) else None
                        except Exception as _e1:  # noqa: BLE001
                            _cl.append(f"\n=== (1) INPUT-SPLIT DIFF: rules read failed ({_e1}) ===")
                    if _rules_share is not None:
                        # tab3 enforced share per cell from prop_items (7-tuple), normalised to % per cell.
                        _pi = list(prop_items)
                        if _pi and len(_pi[0]) == 7:
                            _e3 = pd.DataFrame(_pi, columns=["Currency", "BIN", "RPGT", "pmp", "Country", "vampMid", "prop_raw"])
                            _e3["_vml"] = _kv(_e3["vampMid"]); _e3["_rp"] = _kv(_e3["RPGT"])
                            _e3["_cur"] = _kv(_e3["Currency"]); _e3["_bn"] = _e3["BIN"].astype(str).str.strip()
                            _e3["_pm"] = _kv(_e3["pmp"]); _e3["_ct"] = _kv(_e3["Country"])
                            _e3["prop_raw"] = pd.to_numeric(_e3["prop_raw"], errors="coerce").fillna(0.0)
                            _cellk = ["_rp", "_cur", "_bn", "_pm", "_ct"]
                            _csum = _e3.groupby(_cellk, observed=True)["prop_raw"].transform("sum")
                            _e3["t3_pct"] = np.where(_csum > 0, _e3["prop_raw"] * 100.0 / _csum, 0.0)
                            _e3s = _e3.groupby(["_vml"] + _cellk, observed=True)["t3_pct"].sum().reset_index()
                            _sp = _e3s.merge(_rules_share, on=["_vml"] + _cellk, how="outer").fillna(0.0)
                            _sp["dpct"] = _sp["t3_pct"] - _sp["rules_pct"]
                            _mx = float(_sp["dpct"].abs().max()) if len(_sp) else 0.0
                            _cl.append("")
                            _cl.append("=== (1) INPUT-SPLIT DIFF — tab3 enforced share vs EXPORTED RULES (per-cell %) ===")
                            _cl.append(f"  cells compared={len(_sp):,}  max |Δ%|={_mx:.2f}")
                            if _mx < 1.0:
                                _cl.append("  VERDICT: inputs MATCH (tab3 enforced ≈ exported rules, and the compressed "
                                           "split == the exported pools) ⇒ the divergence is APPLICATION-side (projection "
                                           "vs AllocationEngine), NOT the split.")
                            else:
                                _cl.append("  VERDICT: inputs DIFFER ⇒ the split tab 3 projects is NOT the exported rules "
                                           "tab 5 ran. Biggest share gaps below (re-export / check basis & compression).")
                            _cl.append("  per-vampMid mean |Δ%| (worst first):")
                            _mv = (_sp.groupby("_vml").agg(mad=("dpct", lambda s: float(s.abs().mean())),
                                   n=("dpct", "size")).reset_index().sort_values("mad", ascending=False).head(20))
                            for _, r in _mv.iterrows():
                                _cl.append(f"    {str(r['_vml'])[:30]:30s} mean|Δ%|={r['mad']:>6.2f}  cells={int(r['n'])}")
                            _cl.append("  top 25 cells by |Δ%| (t3% vs rules%):")
                            _tp = _sp.reindex(_sp["dpct"].abs().sort_values(ascending=False).index).head(25)
                            for _, r in _tp.iterrows():
                                _cl.append(f"    {str(r['_vml'])[:22]:22s} {r['_cur']}/{r['_bn']}/{str(r['_rp'])[:12]:12s} "
                                           f"pmp={str(r['_pm'])[:8]:8s} ctry={str(r['_ct'])[:7]:7s} "
                                           f"t3%={r['t3_pct']:>6.2f} rules%={r['rules_pct']:>6.2f} Δ%={r['dpct']:>+6.2f}")
                            _sp_focus = _sp[_sp["_vml"].str.contains(_cmid, na=False)]
                        else:
                            _cl.append("\n=== (1) INPUT-SPLIT DIFF: prop_items not 7-tuple (enforced grain); skipped ===")
                            _sp_focus = None
                    else:
                        _sp_focus = None
                        _cl.append("\n=== (1) INPUT-SPLIT DIFF: exported rules not found in "
                                   f"{_rules_dir} (PoolTargeted_Rules_*.xlsx); skipped ===")

                    # focus cell = the focus MID's single biggest-|Δ| OUTPUT cell (for sections 2-5).
                    _fcell = None
                    _fc = _cmp[_cmp["_vml"].str.contains(_cmid, na=False)]
                    # Prefer a cell where the focus MID has a real baseline (base>0) so the finer-grain
                    # / provenance sections (4/5) actually have mapping rows for it.
                    _fcb = _fc[_fc["base"] > 0]
                    _pick = _fcb if len(_fcb) else _fc
                    if len(_pick):
                        _fr = _pick.reindex(_pick["d"].abs().sort_values(ascending=False).index).iloc[0]
                        _fcell = (_fr["_cur"], _fr["_bn"], _fr["_rp"], _fr["_pm"], _fr["_ct"])

                    # ---- (2) FULL-CELL side-by-side (ALL gateways) for the top-divergent cells. ----
                    _cl.append("")
                    _cl.append("=== (2) FULL-CELL side-by-side (all gateways) — top 6 divergent cells (P1) ===")
                    _c1 = _cmp[_cmp["period"] == (1 if 1 in _pers else _pers[0])]
                    _cellcols = ["_cur", "_bn", "_rp", "_pm", "_ct"]
                    _cellmag = (_c1.groupby(_cellcols)["d"].agg(lambda s: float(s.abs().sum()))
                                .reset_index().sort_values("d", ascending=False).head(6))
                    for _, cr in _cellmag.iterrows():
                        _sel = ((_c1["_cur"] == cr["_cur"]) & (_c1["_bn"] == cr["_bn"]) & (_c1["_rp"] == cr["_rp"])
                                & (_c1["_pm"] == cr["_pm"]) & (_c1["_ct"] == cr["_ct"]))
                        _cc2 = _c1[_sel]
                        _t3tot = _cc2["t3"].sum(); _t5tot = _cc2["t5"].sum()
                        _cl.append(f"  ── {cr['_cur']}/{cr['_bn']}/{cr['_rp']} pmp={cr['_pm']} ctry={cr['_ct']}  "
                                   f"[cell tot t3={_t3tot:,.0f} t5={_t5tot:,.0f}]")
                        for _, r in _cc2.reindex(_cc2["d"].abs().sort_values(ascending=False).index).iterrows():
                            _s3 = (r["t3"] / _t3tot) if _t3tot > 0 else 0.0
                            _s5 = (r["t5"] / _t5tot) if _t5tot > 0 else 0.0
                            _cl.append(f"       {str(r['_vml'])[:26]:26s} base={r['base']:>7,.0f} "
                                       f"t3={r['t3']:>7,.0f}({_s3*100:>5.1f}%) t5={r['t5']:>7,.0f}({_s5*100:>5.1f}%) "
                                       f"Δ={r['d']:>+7,.0f}")

                    # ---- (3) HELD/MOVED decomposition, both sides, for the focus MID's cells. ----
                    _cl.append("")
                    _cl.append(f"=== (3) HELD/MOVED — focus MID ~'{_cmid}' (tab3 exact; tab5 net pre→post) ===")
                    _tf = _tt[_tt["_vml"].str.contains(_cmid, na=False)].copy()
                    if len(_tf):
                        _mvf = (pd.to_numeric(_tf.get("pro_rata", 1.0), errors="coerce").fillna(1.0)
                                * pd.to_numeric(_tf.get("fcp1_frac", 1.0), errors="coerce").fillna(1.0))
                        _tf["_held3"] = pd.to_numeric(_tf["cell_tot"], errors="coerce").fillna(0.0) * \
                            pd.to_numeric(_tf.get("base_share", 0.0), errors="coerce").fillna(0.0) * (1.0 - _mvf)
                        _tf["_movein3"] = pd.to_numeric(_tf["cell_tot"], errors="coerce").fillna(0.0) * \
                            pd.to_numeric(_tf.get("_moved_tot", 0.0), errors="coerce").fillna(0.0) * \
                            pd.to_numeric(_tf.get("prop_share", 0.0), errors="coerce").fillna(0.0)
                        _tf["_mv"] = _mvf
                        # DECISIVE FIELDS: prop_share (the redistribution share the projection actually
                        # applies), prop_raw (the enforced share after merge), and the merge/coarse/bf
                        # flags. If prop_share≈0 while the exported rules give this MID ~X%, the enforced
                        # share did NOT reach this sub-cell row (merge miss) — vs a genuine held-cohort
                        # model effect. coarse=1 => filled from the pmp/Country-agnostic fallback (not the
                        # exact rule); bf=1 => injected back-fill row.
                        _agg3 = dict(held3=("_held3", "sum"), movein3=("_movein3", "sum"),
                                     mv=("_mv", "mean"), post3=("post_txn", "sum"),
                                     base=("VI_Txn_Count", "sum"))
                        if "prop_share" in _tf.columns:
                            _agg3["pshare"] = ("prop_share", "mean")
                        if "prop_raw" in _tf.columns:
                            _agg3["praw"] = ("prop_raw", "sum")
                        if "_prop_from_coarse" in _tf.columns:
                            _agg3["coarse"] = ("_prop_from_coarse", "max")
                        if "_bf_inj" in _tf.columns:
                            _agg3["bf"] = ("_bf_inj", "max")
                        _hm3 = _tf.groupby(_K, observed=True).agg(**_agg3).reset_index()
                        _hm = _hm3.merge(_g5, on=_K, how="left").fillna(0.0)
                        # Attach the EXPORTED RULE % (tab 5's INPUT share for this MID/cell) beside
                        # tab 3's enforced prop_raw (tab 3's INPUT). THE decisive check: if praw ≈
                        # rules% but the OUTPUT still diverges, it's application (the two-cohort model);
                        # if praw ≠ rules%, the enforced split tab 3 projects differs from the rules
                        # tab 5 ran (input problem) — for THIS exact cell, not an average.
                        if _rules_share is not None:
                            _hm = _hm.merge(_rules_share, on=["_vml", "_rp", "_cur", "_bn", "_pm", "_ct"], how="left")
                        if "rules_pct" not in _hm.columns:
                            _hm["rules_pct"] = float("nan")
                        _hm["t5_net"] = _hm["t5"] - _hm["t5_pre"]
                        _hm = _hm[_hm["period"] == (1 if 1 in _pers else _pers[0])]
                        _hm = _hm.reindex((_hm["post3"] - _hm["t5"]).abs().sort_values(ascending=False).index).head(15)
                        _cl.append("   cell (P1): tab3 held+in=post | mv · prop_share · praw(tab3 in) · rules%(tab5 in) · coarse · bf | tab5 pre→post")
                        for _, r in _hm.iterrows():
                            _cl.append(f"    {r['_cur']}/{r['_bn']}/{str(r['_rp'])[:10]:10s} pmp={str(r['_pm'])[:8]:8s} "
                                       f"ctry={str(r['_ct'])[:7]:7s} | held={r['held3']:>6,.0f}+in={r['movein3']:>6,.0f}"
                                       f"=post{r['post3']:>6,.0f} | mv={r['mv']:.3f} psh={float(r.get('pshare', 0)):.4f} "
                                       f"praw={float(r.get('praw', 0)):.2f} rules%={float(r.get('rules_pct', float('nan'))):.2f} "
                                       f"c={int(r.get('coarse', 0))} bf={int(r.get('bf', 0))}"
                                       f" | t5 {r['t5_pre']:>6,.0f}→{r['t5']:>6,.0f}(net{r['t5_net']:>+6,.0f})")

                    # ---- (4) FINER GRAIN (renewal × fcp × attempt) + (5) fcp1_frac provenance,
                    # for the focus cell, straight from mapping_pct_export. ----
                    if (_fcell is not None and _os.path.exists(_map_path)
                            and _os.environ.get("ROUTING_PROJ_FINEGRAIN", "1") != "0"):
                        try:
                            _um = {"rpgt", "Currency", "BIN", "paymentMethodProvider", "Country",
                                   "renewal_number", "fcpNumber", "gatewayFid", "attemptNumber", "trx_count"}

                            def _build_map():
                                _m = pd.read_csv(_map_path, usecols=lambda c: c.strip() in _um, low_memory=False)
                                _m.columns = [c.strip() for c in _m.columns]
                                return _m
                            _mp, _msrc = _cached_df(_os.path.join(_dir, "_cache_mapping.pkl"), [_map_path], _build_map)
                            _fcur, _fbin, _frp, _fpm, _fct = _fcell
                            _msel = ((_kv(_mp["Currency"]) == _fcur) & (_mp["BIN"].astype(str).str.strip() == _fbin)
                                     & (_kv(_mp["rpgt"]) == _frp) & (_kv(_mp["paymentMethodProvider"]) == _fpm)
                                     & (_kv(_mp["Country"]) == _fct))
                            _mc = _mp[_msel].copy()
                            _mc["_vml"] = _mc["gatewayFid"].astype(str).str.strip().str.lower().map(_f2v).fillna(_mc["gatewayFid"]).str.lower()
                            _mc["_tc"] = pd.to_numeric(_mc["trx_count"], errors="coerce").fillna(0.0)
                            _cl.append("")
                            _cl.append(f"=== (4) FINER GRAIN for focus cell {_fcur}/{_fbin}/{_frp} pmp={_fpm} ctry={_fct} "
                                       "(mapping_pct_export renewal×fcp×attempt) ===")
                            _mfoc = _mc[_mc["_vml"].str.contains(_cmid, na=False)]
                            _cl.append(f"  focus MID ~'{_cmid}' rows: {len(_mfoc)}  Σtrx={_mfoc['_tc'].sum():,.0f}")
                            _byd = (_mc.assign(_ren=_mc["renewal_number"].astype(str), _fcp=_mc["fcpNumber"].astype(str),
                                               _att=_mc["attemptNumber"].astype(str))
                                    .groupby(["_vml", "_ren", "_fcp", "_att"], observed=True)["_tc"].sum().reset_index())
                            _bf = _byd[_byd["_vml"].str.contains(_cmid, na=False)].sort_values("_tc", ascending=False).head(12)
                            for _, r in _bf.iterrows():
                                _cl.append(f"    {str(r['_vml'])[:24]:24s} renewal={str(r['_ren'])[:10]:10s} "
                                           f"fcp={str(r['_fcp'])[:3]:3s} attempt={str(r['_att'])[:3]:3s} trx={r['_tc']:>8,.0f}")
                            # ---- (5) fcp1_frac provenance for the focus cell (from the SAME mapping) ----
                            _restr = _frp in ("monthly initial", "annual sub sale", "upgrades")
                            _mc["_fcp1"] = _mc["fcpNumber"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                            _mc["_att1"] = _mc["attemptNumber"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
                            _elig = (_mc["_fcp1"] == "1") & ((not _restr) | (_mc["_att1"] == "1"))
                            _mc["_eltc"] = np.where(_elig, _mc["_tc"], 0.0)
                            _prov = _mc.groupby("_vml", observed=True).agg(tot=("_tc", "sum"), el=("_eltc", "sum")).reset_index()
                            _prov["mapping_movable"] = np.where(_prov["tot"] > 0, _prov["el"] / _prov["tot"], np.nan)
                            # tab3 fcp1_frac for this cell (mean over its sub-rows).
                            _t3f = _tt[(_tt["_cur"] == _fcur) & (_tt["_bn"] == _fbin) & (_tt["_rp"] == _frp)
                                       & (_tt["_pm"] == _fpm) & (_tt["_ct"] == _fct)]
                            _t3fmap = (_t3f.groupby("_vml")["fcp1_frac"].mean().to_dict()
                                       if "fcp1_frac" in _t3f.columns else {})
                            _cl.append(f"  (5) fcp1_frac PROVENANCE (restricted RPGT={_restr}): movable = fcp1"
                                       + (" & attempt1" if _restr else "") + " share, from mapping vs tab3")
                            for _, r in _prov.sort_values("tot", ascending=False).head(12).iterrows():
                                _t3v = _t3fmap.get(r["_vml"], float("nan"))
                                _cl.append(f"    {str(r['_vml'])[:24]:24s} mapping_movable={r['mapping_movable']:.3f} "
                                           f"tab3_fcp1={_t3v:.3f}  (Σtrx={r['tot']:,.0f})")
                        except Exception as _e4:  # noqa: BLE001
                            _cl.append(f"=== (4)/(5) finer-grain read failed ({_e4}) ===")
                with open(_os.path.join(_dir, "_proj_diag_tab5_compare.txt"), "w") as _cf:
                    _cf.write("\n".join(str(x) for x in _cl))
        except Exception:  # noqa: BLE001
            pass

        _cc = [c for c in ["Currency", "BIN", "RPGT", "_pmp", "_ctry", "period"] if c in t0.columns]
        _cells = t0.drop_duplicates(_cc)
        L = []
        L.append(f"PROJECTION DIAGNOSTICS  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
        L.append(f"pp_path: {pp_path}")
        L.append(f"prop_items={len(list(prop_items))}  enforced={enforced}  by_rpgt={by_rpgt}")
        L.append(f"t0 rows={len(t0):,}  distinct cells={len(_cells):,}")
        L.append("")
        # prop_raw is on a PERCENT scale (0-100), so a healthy cell sums to ~100. The projection
        # is renormalised to 100 (prop_sum≈100 everywhere post-fix); the interesting signal is how
        # far the sum was OFF 100 BEFORE renorm (_psum_pre) — that's the coarse-fill / _keep shift.
        _post = _cells["prop_sum"]; _posta = _post[_post > 1e-9]
        L.append("=== prop_sum PER CELL — PERCENT scale, healthy ≈ 100 (post-renorm) ===")
        if len(_posta):
            L.append(f"  active cells={len(_posta):,}  min={_posta.min():.2f}  max={_posta.max():.2f}  "
                     f"mean={_posta.mean():.2f}  median={_posta.median():.2f}  (all should be ~100 after renorm)")
        if "_psum_pre" in _cells.columns:
            _pre = _cells["_psum_pre"]; _prea = _pre[_pre > 1e-9]
            if len(_prea):
                _dev = ((_prea - 100.0).abs() / 100.0)
                L.append(f"  PRE-renorm sum: min={_prea.min():.2f} max={_prea.max():.2f} "
                         f"mean={_prea.mean():.2f} median={_prea.median():.2f}")
                for _t in (0.05, 0.20, 0.50, 1.0):
                    L.append(f"  cells |pre_sum-100| > {int(_t*100)}%: {int((_dev > _t).sum()):,}")
        L.append("")
        if "_prop_from_coarse" in t0.columns:
            L.append(f"prop_raw filled from COARSE pmp/Country fallback: "
                     f"{int((t0['_prop_from_coarse'] > 0).sum()):,} rows")
        if "_bf_inj" in t0.columns:
            L.append(f"injected zero-baseline BACK-FILL rows: {int((t0['_bf_inj'] > 0).sum()):,} rows")
        L.append("")
        L.append("=== PER-vampMid (all t0 sub-cells summed): base vs post VI, Σprop, avg share ===")
        _agg = {"base_vi": ("VI_Txn_Count", "sum"), "post_vi": ("post_txn", "sum"),
                "sum_prop_raw": ("prop_raw", "sum"), "avg_prop_share": ("prop_share", "mean"),
                "n_cells": ("cell_tot", "size")}
        if "_bf_inj" in t0.columns:
            _agg["backfill_rows"] = ("_bf_inj", "sum")
        if "_prop_from_coarse" in t0.columns:
            _agg["coarse_rows"] = ("_prop_from_coarse", "sum")
        g = t0.groupby("vampMid", as_index=False).agg(**_agg).sort_values("post_vi", ascending=False)
        for _, r in g.iterrows():
            L.append(f"  {str(r['vampMid'])[:30]:30s} base={r['base_vi']:>12,.0f} post={r['post_vi']:>12,.0f}"
                     f"  d={r['post_vi'] - r['base_vi']:>+12,.0f}  Σprop={r['sum_prop_raw']:>8.3f}"
                     f"  avgShare={r['avg_prop_share']:.3f}  cells={int(r['n_cells'])}"
                     + (f"  bf={int(r['backfill_rows'])}" if "backfill_rows" in g.columns else "")
                     + (f"  coarse={int(r['coarse_rows'])}" if "coarse_rows" in g.columns else ""))
        L.append("")
        # ---- REROUTE DECOMPOSITION: where each MID's post VI comes from. This is the decisive
        # view for tab-3-vs-tab-5: post = held + moved-in. `reach` = the reroutable pool the MID
        # can draw from across its RECIPIENT cells (Σ cell_tot·moved_tot where prop_share>0); if a
        # MID's reach is far below what the pipeline gives it, it's a RECIPIENT-COVERAGE gap
        # (present in too few / too small cells), NOT a share or arithmetic gap. ----
        try:
            _d = t0.copy()
            _mv = _d.get("_move", pd.Series(0.0, index=_d.index)).fillna(0.0)
            _mt = _d.get("_moved_tot", pd.Series(0.0, index=_d.index)).fillna(0.0)
            _bs = _d.get("base_share", pd.Series(0.0, index=_d.index)).fillna(0.0)
            _psh = _d.get("prop_share", pd.Series(0.0, index=_d.index)).fillna(0.0)
            _ct = _d["cell_tot"].fillna(0.0)
            _d["_held"] = _ct * _bs * (1.0 - _mv)
            _d["_movedout"] = _ct * _bs * _mv
            _d["_movedin"] = _ct * _mt * _psh
            _d["_reach"] = _ct * _mt * (_psh > 1e-12)        # reroutable pool it can draw from
            _d["_is_recip"] = (_psh > 1e-12).astype(int)
            _rd = _d.groupby("vampMid", as_index=False).agg(
                held=("_held", "sum"), moved_out=("_movedout", "sum"),
                moved_in=("_movedin", "sum"), reach=("_reach", "sum"),
                recip_cells=("_is_recip", "sum")).sort_values("moved_in", ascending=False)
            L.append("=== REROUTE DECOMPOSITION per vampMid (post = held + moved_in; reach = pool it")
            L.append("    can draw from in its recipient cells; fill% = moved_in/reach) ===")
            for _, r in _rd.iterrows():
                _fill = (r["moved_in"] / r["reach"]) if r["reach"] > 1e-9 else 0.0
                L.append(f"  {str(r['vampMid'])[:30]:30s} held={r['held']:>11,.0f} "
                         f"moved_out={r['moved_out']:>11,.0f} moved_in={r['moved_in']:>11,.0f} "
                         f"reach={r['reach']:>12,.0f} fill={_fill:.2f} recip_cells={int(r['recip_cells']):>6}")
            L.append(f"  [conservation] Σmoved_in={_rd['moved_in'].sum():,.0f}  "
                     f"Σmoved_out={_rd['moved_out'].sum():,.0f}  (should match)")
            L.append("")
        except Exception as _e2:  # noqa: BLE001
            L.append(f"(reroute decomposition failed: {_e2})")
            L.append("")
        # ---- PER-MID × PERIOD post VI (M0-M5) — paste-comparable to tab 5's monthly columns ----
        try:
            _pp = t0.copy()
            _pp["_pvi"] = np.where(_pp["period"].notna(), _pp["post_txn"].fillna(0.0), 0.0)
            _piv = _pp.pivot_table(index="vampMid", columns="period", values="post_txn",
                                   aggfunc="sum", fill_value=0.0)
            L.append("=== PER-MID × PERIOD post VI Txn (compare directly to tab 5 monthly columns) ===")
            _cols_p = sorted([c for c in _piv.columns])
            L.append("  vampMid                        " + " ".join(f"P{int(c):>9}" for c in _cols_p))
            _piv = _piv.reindex(_piv.sum(axis=1).sort_values(ascending=False).index)
            for _mid, _row in _piv.iterrows():
                L.append(f"  {str(_mid)[:30]:30s} " + " ".join(f"{_row[c]:>10,.0f}" for c in _cols_p))
            L.append("")
        except Exception as _e3:  # noqa: BLE001
            L.append(f"(per-period pivot failed: {_e3})")
            L.append("")
        # ---- ENFORCED-PROP COVERAGE per vampMid (straight from prop_items) — how many distinct
        # sub-cells the split routes each MID into, and its total share mass. A back-fill gateway
        # under-covered HERE (vs the exported templates) is the recipient-coverage smoking gun. ----
        try:
            _pi = list(prop_items)
            if _pi and len(_pi[0]) == 7:
                _pdf = pd.DataFrame(_pi, columns=["Currency", "BIN", "RPGT", "pmp", "ctry", "vampMid", "prop_raw"])
                _pc = _pdf.groupby("vampMid").agg(
                    subcells=("prop_raw", "size"), total_prop=("prop_raw", "sum"),
                    distinct_bins=("BIN", "nunique")).sort_values("total_prop", ascending=False)
                L.append("=== ENFORCED-PROP COVERAGE per vampMid (from the split feeding this projection) ===")
                for _mid, r in _pc.iterrows():
                    L.append(f"  {str(_mid)[:30]:30s} subcells={int(r['subcells']):>7}  "
                             f"distinct_BINs={int(r['distinct_bins']):>6}  Σprop%={r['total_prop']:>12,.0f}")
                L.append("")
        except Exception as _e4:  # noqa: BLE001
            L.append(f"(enforced-prop coverage failed: {_e4})")
            L.append("")
        L.append("=== ZERO-BASELINE RECIPIENTS (base VI≈0, post VI>0) — per-cell breakdown (top 80) ===")
        L.append("    columns: vampMid · Cur/BIN/RPGT · pmp · ctry · P<period> · cell_tot · prop_raw ·")
        L.append("    prop_sum · prop_share · moved_tot · post_txn · coarse · bf")
        _rec = t0[(t0["VI_Txn_Count"] <= 1e-9) & (t0["post_txn"] > 1e-9)].sort_values(
            "post_txn", ascending=False).head(80)
        for _, r in _rec.iterrows():
            L.append(f"  {str(r['vampMid'])[:22]:22s} {str(r.get('Currency',''))}/{str(r.get('BIN',''))}/"
                     f"{str(r['RPGT'])[:12]:12s} pmp={str(r.get('_pmp',''))[:8]:8s} ctry={str(r.get('_ctry',''))[:8]:8s}"
                     f" P{int(r['period'])} cell={r['cell_tot']:>10,.0f} praw={r['prop_raw']:.4f}"
                     f" psum={r['prop_sum']:.4f} pshare={r['prop_share']:.4f} mov={r['_moved_tot']:.4f}"
                     f" post={r['post_txn']:>10,.0f} c={int(r.get('_prop_from_coarse', 0))} bf={int(r.get('_bf_inj', 0))}")
        with open(_os.path.join(_dir, "_proj_diag_summary.txt"), "w") as _f:
            _f.write("\n".join(str(x) for x in L))
    except Exception as _e:  # diagnostics must never break the projection
        try:
            import traceback as _tb
            with open(_os.path.join(_os.path.dirname(_os.path.abspath(pp_path)),
                                    "_proj_diag_ERROR.txt"), "w") as _f:
                _f.write(f"{_e}\n{_tb.format_exc()}")
        except Exception:
            pass


def _inject_backfill_rows(pp, prop):
    """#3 ZERO-BASELINE BACK-FILL: build_split_exports can route to gateways (e.g. <2-gateway
    back-fill fallbacks) that have NO baseline row in a cell. The LEFT merge drops them, so their
    routed volume wrongly redistributes to present MIDs. Re-inject them into `pp` as zero-baseline
    t=0 rows (vampCount=0, VI=0) so they RECEIVE the routed volume; VAMP stays 0 for them (no
    historical VAMP to redistribute). Scoped to the enforced (7-tuple) path — only it back-fills.
    """
    # Presence is judged at the pmp/Country SUB-CELL grain (Currency, BIN, RPGT, pmp, Country),
    # NOT the coarse cell — because the enforced table routes per sub-cell, and a MID present in
    # ONE sub-cell but routed volume in ANOTHER (e.g. WoodForest has baseline only in
    # non_gp_ap/non-usa but the template gives it 97% in non_gp_ap/usa) has no row there to
    # receive it, so its routed sub-cell volume wrongly redistributes to the present MIDs.
    # GUARD: only inject into (pmp, Country) sub-cells that actually EXIST in the baseline for
    # the coarse cell — never invent a sub-cell from a pmp/Country label the baseline lacks (a
    # pure label mismatch is handled by the hierarchical coarse fallback downstream, not here),
    # which is what previously twinned MIDs across mismatched sub-cells.
    subk = ["Currency", "BIN", "_rpgtl", "_pmp", "_ctry"]
    b = pp.copy()
    b["_rpgtl"] = b["RPGT"].astype(str).str.strip().str.lower()
    b["_vml"] = b["vampMid"].astype(str).str.strip().str.lower()
    present = set(map(tuple, b[subk + ["_vml"]].drop_duplicates().to_numpy()))
    valid_sub = set(map(tuple, b[subk].drop_duplicates().to_numpy()))
    # Global _vml -> proper-case vampMid, so a MID that exists elsewhere in the export keeps its
    # display name and merges cleanly (no lower-case twin) on the final collapse.
    name_map = b.drop_duplicates("_vml").set_index("_vml")["vampMid"].to_dict()
    # Representative RPGT / go-live pro_rata / fcp1 per (sub-cell, period), lowest-t row.
    reps = (b.sort_values("t").drop_duplicates(subk + ["period"])
            [subk + ["RPGT", "period", "pro_rata", "fcp1_frac"]])
    pc = prop[subk + ["_vml"]].drop_duplicates()
    # missing = enforced (sub-cell, MID) with no baseline row in that sub-cell, AND the sub-cell
    # itself exists in the baseline (so we don't fabricate sub-cells from label mismatches).
    miss = pc[[(tuple(r) not in present) and (tuple(r[:5]) in valid_sub)
               for r in pc.to_numpy()]]
    if "_bf_inj" not in pp.columns:
        pp = pp.copy(); pp["_bf_inj"] = 0
    if miss.empty:
        return pp
    new = reps.merge(miss, on=subk, how="inner")
    if new.empty:
        return pp
    new["vampMid"] = new["_vml"].map(name_map).fillna(new["_vml"])   # proper case if known
    new["vampCount"] = 0.0
    new["VI_Txn_Count"] = 0.0
    new["t"] = 0
    new["_bf_inj"] = 1   # DIAGNOSTIC flag: this is an injected zero-baseline back-fill row
    new = new[["vampMid", "RPGT", "BIN", "Currency", "_pmp", "_ctry", "period", "t",
               "vampCount", "VI_Txn_Count", "pro_rata", "fcp1_frac", "_bf_inj"]]
    return pd.concat([pp, new], ignore_index=True, sort=False)


@st.cache_data(show_spinner=False)
def compute_vamp_prepost_granular(pp_path, prop_items, excluded_mids=frozenset(),
                                  kill_eff=(), month_0=None, scoped_rpgts=(),
                                  wallet_incapable=frozenset(), usa_only=frozenset(),
                                  exploration_floor=0.0):
    """Per-ROW baseline vs proposed VAMP / VI-Txn from the pro-rata export.

    Routes at the (vampMid, RPGT, BIN, Currency, pmp, Country) sub-cell grain when the
    export carries paymentMethodProvider / Country, applying the pipeline's static
    enforcement (wallet-incapable gateways can't serve wallet pmp; USA-only gateways can't
    serve Non-USA) so the projection tracks the pipeline more closely. Result is collapsed
    back to (vampMid, RPGT, BIN, Currency, period, t) for the filterable table.
    """
    pp = pd.read_csv(pp_path)
    pp["Currency"] = pp["Currency"].astype(str).str.strip().str.lower()
    pp["BIN"] = pp["BIN"].astype(str).str.strip()
    pp["vampMid"] = pp["vampMid"].astype(str).str.strip()
    rpgt_col = "RPGT" if "RPGT" in pp.columns else "rpgt"
    pp["RPGT"] = pp[rpgt_col].astype(str)
    pp["pro_rata"] = pd.to_numeric(pp.get("pro_rata", 0.0), errors="coerce").fillna(0.0)
    pp["fcp1_frac"] = pd.to_numeric(pp.get("fcp1_frac", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    # Keep pmp / Country sub-cells (default '_all_' when the export lacks them) so the
    # projection can apply the pipeline's per-sub-cell wallet / USA-only enforcement.
    pp["_pmp"] = (pp["paymentMethodProvider"].astype(str).str.strip().str.lower()
                  if "paymentMethodProvider" in pp.columns else "_all_")
    pp["_ctry"] = (pp["Country"].astype(str).str.strip().str.lower()
                   if "Country" in pp.columns else "_all_")
    pp = pp.groupby(["vampMid", "RPGT", "BIN", "Currency", "_pmp", "_ctry", "period", "t"],
                    as_index=False).agg(
        vampCount=("vampCount", "sum"), VI_Txn_Count=("VI_Txn_Count", "sum"),
        pro_rata=("pro_rata", "first"), fcp1_frac=("fcp1_frac", "first"))

    # prop_items 4-tuples (…, vampMid, prop_raw) or 5-tuples (…, RPGT, vampMid, prop_raw)
    # at Bank×Currency×RPGT grain — see _vamp_post_core for the rationale.
    # prop_items: 4-tuples (…, vampMid, prop_raw), 5-tuples (…, RPGT, vampMid, prop_raw), or
    # 7-tuples (Currency, BIN, RPGT, pmp, Country, vampMid, prop_raw) = ENFORCED shares from
    # enforced_prop_items (already capped / wallet / USA-Non-USA / back-filled → no masking).
    _pi = list(prop_items)
    _n = len(_pi[0]) if _pi else 0
    _enforced = (_n == 7)
    _by_rpgt = (_n == 5) or _enforced
    _cols = (["Currency", "BIN", "RPGT", "_pmp", "_ctry", "vampMid", "prop_raw"] if _enforced
             else ["Currency", "BIN", "RPGT", "vampMid", "prop_raw"] if _by_rpgt
             else ["Currency", "BIN", "vampMid", "prop_raw"])
    prop = pd.DataFrame(_pi, columns=_cols)
    # Match vampMid CASE-INSENSITIVELY: the enforced prop's vampMids come from a lower-cased
    # fid2vamp map, while the export's are proper-case — join on a lower-cased key (`_vml`) on
    # BOTH sides so casing can never break the merge (which zeroed every share → post==pre).
    prop["Currency"] = prop["Currency"].astype(str).str.strip().str.lower()
    prop["BIN"] = prop["BIN"].astype(str).str.strip()
    prop["_vml"] = prop["vampMid"].astype(str).str.strip().str.lower()
    prop = prop.drop(columns=["vampMid"])
    if _by_rpgt:
        prop["_rpgtl"] = prop["RPGT"].astype(str).str.strip().str.lower()
        prop = prop.drop(columns=["RPGT"])
    if _enforced:
        prop["_pmp"] = prop["_pmp"].astype(str).str.strip().str.lower()
        prop["_ctry"] = prop["_ctry"].astype(str).str.strip().str.lower()
        pp = _inject_backfill_rows(pp, prop)   # #3 add zero-baseline back-fill target rows

    grp = ["Currency", "BIN", "RPGT", "_pmp", "_ctry", "period"]
    if _enforced:
        _t0 = pp[pp["t"] == 0].copy()
        _t0["_rpgtl"] = _t0["RPGT"].astype(str).str.strip().str.lower()
        _t0["_vml"] = _t0["vampMid"].astype(str).str.strip().str.lower()
        t0 = _t0.merge(prop, on=["Currency", "BIN", "_rpgtl", "_pmp", "_ctry", "_vml"], how="left")
        t0["_prop_from_coarse"] = 0.0   # DIAGNOSTIC flag
        # Fallback for pmp/Country label mismatches: fill unmatched sub-cells from the
        # pmp/Country-agnostic enforced share so nothing silently drops to zero.
        # HIERARCHICAL fallback for sub-cells whose exact (pmp, Country) didn't match: keep the
        # FINEST available grain so a Country/pmp-specific gateway (e.g. WoodForest — USA / non-
        # GooglePay dominant) keeps its true share instead of a cross-pmp/Country mean that halves
        # it. Try Country-keep, then pmp-keep, then fully agnostic — finest first.
        for _fk in (["Currency", "BIN", "_rpgtl", "_ctry", "_vml"],
                    ["Currency", "BIN", "_rpgtl", "_pmp", "_vml"],
                    ["Currency", "BIN", "_rpgtl", "_vml"]):
            if not t0["prop_raw"].isna().any():
                break
            _cm = prop.groupby(_fk, as_index=False)["prop_raw"].mean().rename(columns={"prop_raw": "_pc"})
            t0 = t0.merge(_cm, on=_fk, how="left")
            _fill = t0["prop_raw"].isna() & t0["_pc"].notna()
            t0.loc[_fill, "_prop_from_coarse"] = 1.0
            t0["prop_raw"] = t0["prop_raw"].fillna(t0["_pc"])
            t0 = t0.drop(columns=["_pc"])
        t0 = t0.drop(columns=["_rpgtl", "_vml"])
    elif _by_rpgt:
        _t0 = pp[pp["t"] == 0].copy()
        _t0["_rpgtl"] = _t0["RPGT"].astype(str).str.strip().str.lower()
        _t0["_vml"] = _t0["vampMid"].astype(str).str.strip().str.lower()
        t0 = _t0.merge(prop, on=["Currency", "BIN", "_rpgtl", "_vml"], how="left").drop(
            columns=["_rpgtl", "_vml"])
    else:
        _t0 = pp[pp["t"] == 0].copy()
        _t0["_vml"] = _t0["vampMid"].astype(str).str.strip().str.lower()
        t0 = _t0.merge(prop, on=["Currency", "BIN", "_vml"], how="left").drop(columns=["_vml"])
    t0["prop_raw"] = t0["prop_raw"].fillna(0.0)
    if "_prop_from_coarse" not in t0.columns:   # DIAGNOSTIC flags (always present)
        t0["_prop_from_coarse"] = 0.0
    if "_bf_inj" not in t0.columns:
        t0["_bf_inj"] = 0
    # Effective-date-gated switch-off (see _vamp_post_core / _mid_keep_fraction).
    _keep = _mid_keep_fraction(t0["vampMid"], t0["period"], kill_eff, month_0)
    _dated = {m for m, _ in (kill_eff or ())}
    _binary = t0["vampMid"].isin(excluded_mids) & ~t0["vampMid"].isin(_dated)
    t0["_keep"] = np.where(_binary, 0.0, _keep)
    t0["prop_raw"] = t0["prop_raw"] * t0["_keep"]
    # PIPELINE ENFORCEMENT (static masks): wallet-incapable gateways can't serve wallet-pmp
    # sub-cells; USA-only gateways can't serve Non-USA sub-cells — zero their proposed share
    # there, so the renormalised split matches what the pipeline actually routes.
    # Static masks only for RAW prop_items — enforced shares already have them baked in.
    _wc_s = {str(x).strip().lower() for x in (wallet_incapable or set())}
    _uo_s = {str(x).strip().lower() for x in (usa_only or set())}
    if (_wc_s or _uo_s) and not _enforced:
        _ml = t0["vampMid"].astype(str).str.strip().str.lower()
        _wallet = t0["_pmp"].isin(["googlepay", "applepay"])
        _nonusa = ~t0["_ctry"].isin(["usa", "us", "_all_", ""])
        _emask = (_wallet & _ml.isin(_wc_s)) | (_nonusa & _ml.isin(_uo_s))
        t0["prop_raw"] = np.where(_emask, 0.0, t0["prop_raw"])
    t0["_av"] = t0["VI_Txn_Count"] * t0["_keep"]
    _g = t0.groupby(grp)   # group cell keys ONCE, reuse for all three sums (bit-identical)
    t0["cell_tot"] = _g["VI_Txn_Count"].transform("sum")
    t0["_at"] = _g["_av"].transform("sum")
    t0["base_share"] = np.where(t0["_at"] > 0, t0["_av"] / t0["_at"], 0.0)
    # Renormalise each cell's proposed shares back to a clean 100 budget after the coarse
    # pmp/Country fill and _keep zeroing may have pushed the per-cell sum off 100 (diagnostic:
    # _psum_pre keeps the pre-renorm sum). NOTE: prop_share below is prop_raw/prop_sum, which is
    # scale-invariant, so this does not change the projected split — it only keeps prop_sum ≈ 100
    # so the shares read as a clean percentage and no downstream code can assume a stale budget.
    t0["_psum_pre"] = _g["prop_raw"].transform("sum")
    t0["prop_raw"] = np.where(t0["_psum_pre"] > 0, t0["prop_raw"] * 100.0 / t0["_psum_pre"], t0["prop_raw"])
    t0["prop_sum"] = t0.groupby(grp)["prop_raw"].transform("sum")
    t0["prop_share"] = np.where(t0["prop_sum"] > 0, t0["prop_raw"] / t0["prop_sum"], t0["base_share"])
    # EXPLORATION FLOOR (replicate the AllocationEngine): every ELIGIBLE gateway in a routed cell
    # keeps >= floor of the redistributed share, then renormalise. This is the primary reason a
    # 0%-rule incumbent (e.g. Braintree in a restricted RPGT) still retains volume in tab 5 — the
    # flat exported rule drives it to ~0, but the engine floors it. Eligible = present in the cell
    # (base_share>0 or prop_raw>0), not switched-off (_keep>0), and NOT wallet/USA-masked (so the
    # floor never un-masks an ineligible gateway). floor=0 → unchanged (backward-compatible).
    _efloor = float(exploration_floor or 0.0)
    if _efloor > 0.0:
        _wc_f = {str(x).strip().lower() for x in (wallet_incapable or set())}
        _uo_f = {str(x).strip().lower() for x in (usa_only or set())}
        _mlf = t0["vampMid"].astype(str).str.strip().str.lower()
        _emask_f = ((t0["_pmp"].isin(["googlepay", "applepay"]) & _mlf.isin(_wc_f))
                    | ((~t0["_ctry"].isin(["usa", "us", "_all_", ""])) & _mlf.isin(_uo_f)))
        _elig_f = (((t0["base_share"] > 0) | (t0["prop_raw"] > 0)) & (t0["_keep"] > 0)
                   & (~_emask_f) & (t0["prop_sum"] > 0))
        _nef = t0.assign(_ef=_elig_f.astype(float)).groupby(grp)["_ef"].transform("sum")
        _flc = np.where(_nef > 0, np.minimum(_efloor, 1.0 / np.maximum(_nef, 1.0)), 0.0)
        t0["prop_share"] = np.where(_elig_f, np.maximum(t0["prop_share"], _flc), t0["prop_share"])
        _psh_sum = t0.groupby(grp)["prop_share"].transform("sum")   # renormalise cells we floored
        _do_renorm = (t0["prop_sum"] > 0) & (_psh_sum > 0)
        t0["prop_share"] = np.where(_do_renorm, t0["prop_share"] / _psh_sum, t0["prop_share"])
    # Movable fraction = go-live pro-rata × fcp1 cohort fraction (see _vamp_post_core).
    _p = t0["pro_rata"] * t0["fcp1_frac"]
    t0["post_txn"] = t0["cell_tot"] * ((1 - _p) * t0["base_share"] + _p * t0["prop_share"])
    # PER-MID movable fraction + VAMP-carrying redistribution share (see _vamp_post_core).
    t0["_move"] = np.where(t0["prop_sum"] > 0, t0["pro_rata"] * t0["fcp1_frac"], 0.0)
    t0["_vprop"] = t0["prop_raw"] * (t0["vampCount"] > 0).astype(float)
    t0["_vpsum"] = t0.groupby(grp)["_vprop"].transform("sum")
    t0["_vshare"] = np.where(t0["_vpsum"] > 0, t0["_vprop"] / t0["_vpsum"], 0.0)

    # RPGT scope: hold non-scoped RPGTs at their current baseline split (post == pre).
    if scoped_rpgts:
        _scope = {str(r).strip().lower() for r in scoped_rpgts}
        _oos = ~t0["RPGT"].astype(str).str.strip().str.lower().isin(_scope)
        t0.loc[_oos, "_move"] = 0.0

    # TWO-COHORT volume (per-MID held on own gateway; pooled movable slice redistributed).
    t0["_bm"] = t0["base_share"] * t0["_move"]
    t0["_moved_tot"] = t0.groupby(grp)["_bm"].transform("sum")
    t0["post_txn"] = t0["cell_tot"] * (t0["base_share"] * (1 - t0["_move"])
                                       + t0["_moved_tot"] * t0["prop_share"])

    _sub = ["Currency", "BIN", "RPGT", "_pmp", "_ctry"]
    # #2 GO-LIVE TIMING: the pipeline applies the go-live weight by the APPEARANCE month
    # (target month m), not origination. So take the rule×cohort factor (_gf = fcp1 × has-rule
    # × scope, WITHOUT pro_rata) at the ORIGINATION cell, and multiply by the go-live pro_rata
    # of the APPEARANCE month (the t0 pro_rata at that period). VI-txn (t=0) is unchanged
    # because appearance == origination there.
    t0["_gf"] = np.where(t0["prop_sum"] > 0, t0["fcp1_frac"], 0.0)
    if scoped_rpgts:
        t0.loc[_oos, "_gf"] = 0.0
    _prapp = t0[_sub + ["period", "pro_rata"]].drop_duplicates(_sub + ["period"]).rename(
        columns={"pro_rata": "_pr_app"})
    _mv = t0[_sub + ["vampMid", "period", "_gf", "_vshare"]].rename(
        columns={"period": "orig_m", "_vshare": "_pshare"})
    pp["orig_m"] = pp["period"] - pp["t"]
    pp = pp.merge(_mv, on=_sub + ["vampMid", "orig_m"], how="left")        # factor at origination
    pp = pp.merge(_prapp, on=_sub + ["period"], how="left")               # go-live wt at appearance
    pp["_gf"] = pp["_gf"].fillna(0.0)
    pp["_pshare"] = pp["_pshare"].fillna(0.0)
    pp["_pr_app"] = pp["_pr_app"].fillna(0.0)
    # Originated before the window (orig_m<0) never moves; otherwise move = factor × appearance wt.
    pp["_move"] = np.where(pp["orig_m"] >= 0, pp["_gf"] * pp["_pr_app"], 0.0)
    pp["VAMP_Pre"] = pp["vampCount"]
    pp["_moved_v"] = pp["vampCount"] * pp["_move"]
    pp["_moved_vpool"] = pp.groupby(_sub + ["period", "t"])["_moved_v"].transform("sum")
    pp["VAMP_Post"] = pp["vampCount"] * (1.0 - pp["_move"]) + pp["_moved_vpool"] * pp["_pshare"]

    _tp = t0[_sub + ["vampMid", "period", "post_txn"]]
    pp = pp.merge(_tp, on=_sub + ["vampMid", "period"], how="left")
    pp["VI_Txn_Pre"] = np.where(pp["t"] == 0, pp["VI_Txn_Count"], 0.0)
    pp["VI_Txn_Post"] = np.where(pp["t"] == 0, pp["post_txn"].fillna(0.0), 0.0)
    _dump_projection_diag(t0, pp_path, prop_items, _enforced, _by_rpgt)   # heavy diagnostics
    # Collapse the pmp / Country sub-cells back to the reported grain (sums are exact).
    return (pp.groupby(["vampMid", "RPGT", "BIN", "Currency", "period", "t"], as_index=False)
              [["VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]].sum())


def mid_table_from_granular(gran):
    """Per-vampMid VAMP / VI-Txn M0–5 (pre & post) table, derived by AGGREGATING the
    granular pre/post frame from compute_vamp_prepost_granular. Because it reads the
    same projection, it is numerically identical to compute_vamp_post_from_prorata —
    so the Impact tab can run ONE granular projection and reuse it for both the
    filterable detail AND this per-MID table (instead of two projections)."""
    if gran is None or getattr(gran, "empty", True):
        cols = ["vampMid"] + [f"{p} M{m}" for m in range(6)
                              for p in ("VAMP", "VI Txn", "VAMP Post", "VI Txn Post")]
        return pd.DataFrame(columns=cols)
    _vp = gran.groupby(["vampMid", "period"])["VAMP_Pre"].sum().unstack(fill_value=0.0)
    _vq = gran.groupby(["vampMid", "period"])["VAMP_Post"].sum().unstack(fill_value=0.0)
    _tp = gran.groupby(["vampMid", "period"])["VI_Txn_Pre"].sum().unstack(fill_value=0.0)
    _tq = gran.groupby(["vampMid", "period"])["VI_Txn_Post"].sum().unstack(fill_value=0.0)
    out = pd.DataFrame({"vampMid": sorted(gran["vampMid"].unique())}).set_index("vampMid")
    for m in range(6):
        out[f"VAMP M{m}"] = _vp[m] if m in _vp.columns else 0.0
        out[f"VI Txn M{m}"] = _tp[m] if m in _tp.columns else 0.0
        out[f"VAMP Post M{m}"] = _vq[m] if m in _vq.columns else 0.0
        out[f"VI Txn Post M{m}"] = _tq[m] if m in _tq.columns else 0.0
    return out.fillna(0.0).reset_index()


def process_wallet_incapable(mid_list_path):
    """Set of gatewayFids (lowercased) that CANNOT process wallet (GOOGLEPAY /
    APPLEPAY), read from a processWallet-style column in Master_MID_List.

    Robust to column-name variants (any column whose normalised name contains
    'wallet'). Only EXPLICIT false-like values (FALSE/F/0/NO/N) mark a gateway
    incapable; blanks/unknown default to capable (so we never over-restrict).
    """
    if not mid_list_path or not os.path.exists(mid_list_path):
        return set()
    try:
        _m = pd.read_csv(mid_list_path)
    except Exception:
        return set()
    _norm = {c: str(c).lower().replace(" ", "").replace("_", "") for c in _m.columns}
    _gcol = next((c for c, n in _norm.items() if n == "gatewayfid"), None)
    _wcol = next((c for c, n in _norm.items() if "processwallet" in n), None) \
        or next((c for c, n in _norm.items() if "wallet" in n), None)
    if not _gcol or not _wcol:
        return set()
    _false = {"false", "f", "0", "no", "n"}
    _vals = _m[_wcol].astype(str).str.strip().str.lower()
    return set(_m.loc[_vals.isin(_false), _gcol].astype(str).str.strip().str.lower())


# --- Performance: cache the heavy, deterministic per-rerun computations. Moving the
# slider or a table filter triggers a FULL Streamlit rerun; without caching we re-read
# the pro-rata/thermometer files and re-run the VAMP projection every time. These
# wrappers are keyed on file path + mtime + the (hashable) split signature, so results
# are byte-identical — only recomputed when the inputs actually change.
def _cache_data(**kw):
    _dec = getattr(st, "cache_data", None) or getattr(st, "experimental_memo", None)
    return _dec(**kw) if _dec is not None else (lambda f: f)


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@_cache_data(show_spinner=False)
def _c_read_parquet(path, _m):
    return pd.read_parquet(path)


@_cache_data(show_spinner=False)
def _c_vamp_post_prorata(pp_path, _m, prop_items, excluded_mids, kill_eff=(), month_0=None,
                         scoped_rpgts=()):
    return compute_vamp_post_from_prorata(pp_path, prop_items, excluded_mids, kill_eff,
                                          month_0, scoped_rpgts)


@_cache_data(show_spinner=False)
def _c_prepost_granular(pp_path, _m, prop_items, excluded_mids, kill_eff=(), month_0=None,
                        scoped_rpgts=(), wallet_incapable=frozenset(), usa_only=frozenset(),
                        exploration_floor=0.0):
    return compute_vamp_prepost_granular(pp_path, prop_items, excluded_mids, kill_eff,
                                         month_0, scoped_rpgts, wallet_incapable, usa_only,
                                         exploration_floor=exploration_floor)


def build_split_exports(split, brand, go_live, wallet_incapable=frozenset(), fid2vamp=None,
                        mid_list_path=None, usa_only=frozenset(), country_pres=None,
                        max_share=0.97):
    """Build the production template (one DataFrame per Brand×RPGT) from a split.

    Wide format matching the uploaded template: one row per (BIN, currency, Country,
    paymentMethodProvider) with gateway weight columns (%), a `Check` column, etc.

    Enforcement applied to every row (so the template can't route in ways the engine
    forbids):
      * Wallet pmp (GOOGLEPAY/APPLEPAY): zero any wallet-incapable gateway, renorm.
      * Country: each cell is split into USA and/or Non-USA rows from the attempts
        `country` field (country_pres). USA-only gateways (usa_only) appear in USA
        rows ONLY — zeroed and renormalised in Non-USA rows.
      * Max share: no gateway exceeds `max_share`; the excess is redistributed to the
        OTHER gateways ALREADY in the split (never activates a new gateway). Only
        applied when ≥2 gateways are present — a genuinely single-gateway cell can't
        be capped without a fallback, so it's left at 100% (and flagged by Check).
    """
    fid2vamp = fid2vamp or {}
    wallet_incapable = set(wallet_incapable or [])
    usa_only = {str(x).strip().lower() for x in (usa_only or set())}
    country_pres = country_pres or {}
    _cap = float(max_share) if max_share else 1.0
    # Source of truth: read processWallet straight from Master_MID_List so the
    # export enforces it even if the routing run didn't populate the set.
    for _f in process_wallet_incapable(mid_list_path):
        wallet_incapable.add(_f)
        _vm = fid2vamp.get(_f)
        if _vm:
            wallet_incapable.add(_vm)
    # Master-MID lookups (fid → currency, fid → active) so a wallet / Non-USA row that would
    # collapse to <2 gateways can be back-filled with currency-matched, active, country-valid
    # (and wallet-capable, for wallet rows) fallback gateways instead of going 100% or empty.
    _fid_cur, _fid_active = {}, {}
    try:
        if mid_list_path and os.path.exists(mid_list_path):
            _mm = pd.read_csv(mid_list_path)
            _cc = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mm.columns}
            _gx, _cx, _ax = _cc.get("gatewayfid"), _cc.get("currency"), _cc.get("isactive")
            if _gx and _cx:
                for _i in range(len(_mm)):
                    _f = str(_mm[_gx].iloc[_i]).strip().lower()
                    if _f and _f not in ("", "nan", "none"):
                        _fid_cur.setdefault(_f, str(_mm[_cx].iloc[_i]).strip().lower())
                        if _ax:
                            _fid_active.setdefault(_f, str(_mm[_ax].iloc[_i]).strip().lower() in ("true", "1", "yes", "t", "y"))
    except Exception:  # noqa: BLE001
        _fid_cur, _fid_active = {}, {}
    df = split.copy()
    df["RPGT"] = df["rpgt"].astype(str)
    df["Currency"] = df["currency"].astype(str).str.upper()
    df["BIN"] = df["bank"].astype(str)
    df["gateway"] = df["gateway"].astype(str)
    df["share"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0)
    gateways = sorted(df["gateway"].unique().tolist())
    _pmps = ["GOOGLEPAY", "APPLEPAY", "non_gp_ap"]

    def _incap(gw):
        g = gw.strip().lower()
        return g in wallet_incapable or fid2vamp.get(g, "") in wallet_incapable

    def _is_usa_only(gw):
        g = gw.strip().lower()
        return g in usa_only or fid2vamp.get(g, "") in usa_only

    def _valid_candidates(cur_l, country, is_wallet):
        """Template-column gateways eligible to serve this (currency, country, pmp): currency-
        matched, active, USA-only excluded for Non-USA, and wallet-capable for wallet rows.
        Used to back-fill rows that would otherwise collapse to <2 gateways."""
        out = []
        for gw in gateways:
            g = gw.strip().lower()
            if _fid_cur.get(g) != cur_l:                       # must match the cell's currency
                continue
            if _fid_active and not _fid_active.get(g, True):    # must be active
                continue
            if country == "Non-USA" and _is_usa_only(gw):       # USA-only can't serve Non-USA
                continue
            if is_wallet and _incap(gw):                        # wallet rows: wallet-capable only
                continue
            out.append(gw)
        return out

    def _cap_shares(series):
        """Cap each share at `_cap`, water-filling the excess into the OTHER gateways
        already present (by remaining room). No-op for <2 gateways (can't cap without
        a fallback) or if the cap is 1.0. Returns a share Series summing to 1."""
        v = series.to_numpy(float).copy()
        s = v.sum()
        if s <= 0:
            return series * 0.0
        v = v / s
        nz = v > 1e-12
        if _cap < 1.0 and int(nz.sum()) >= 2:
            for _ in range(50):
                over = v > _cap + 1e-12
                if not over.any():
                    break
                excess = float((v[over] - _cap).sum())
                v[over] = _cap
                recip = nz & (~over) & (v < _cap - 1e-12)   # only existing gateways with room
                room = np.where(recip, _cap - v, 0.0)
                if room.sum() <= 1e-12:
                    break
                v = v + room / room.sum() * excess
        return pd.Series(v, index=series.index)

    def _countries_for(cur, bin_):
        _u, _n = country_pres.get((str(cur).strip().lower(), str(bin_).strip()), (None, None))
        if _u is None and _n is None:      # no attempts country info → emit both (safe default)
            return ["USA", "Non-USA"]
        cs = []
        if (_u or 0) > 0:
            cs.append("USA")
        if (_n or 0) > 0:
            cs.append("Non-USA")
        return cs or ["Non-USA"]

    out = {}
    for rpgt, g_rpgt in df.groupby("RPGT"):
        rows = []
        for (cur, bin_), cell in g_rpgt.groupby(["Currency", "BIN"]):
            base = cell.groupby("gateway")["share"].sum()
            _s = base.sum()
            base = base / _s if _s > 0 else base
            for country in _countries_for(cur, bin_):
                cbase = base.copy()
                if country == "Non-USA":               # USA-only gateways can't serve Non-USA
                    for gw in list(cbase.index):
                        if _is_usa_only(gw):
                            cbase[gw] = 0.0
                    _cs = cbase.sum()
                    cbase = cbase / _cs if _cs > 0 else cbase
                for pmp in _pmps:
                    _iswallet = pmp in ("GOOGLEPAY", "APPLEPAY")
                    sh = cbase.copy()
                    if _iswallet and wallet_incapable:
                        for gw in list(sh.index):
                            if _incap(gw):
                                sh[gw] = 0.0
                        _ws = sh.sum()
                        sh = sh / _ws if _ws > 0 else sh * 0.0
                    # Back-fill so no row is 100%-to-one or empty: if the wallet/country filtering
                    # left <2 gateways with share, pull in valid fallback gateways (currency-matched,
                    # active, country-valid, wallet-capable for wallet rows) and distribute.
                    _nz = [g for g in gateways if float(sh.get(g, 0.0)) > 1e-9]
                    if len(_nz) < 2 and _fid_cur:
                        _cands = _valid_candidates(str(cur).strip().lower(), country, _iswallet)
                        if _cands:
                            sh = pd.Series({g: float(sh.get(g, 0.0)) for g in gateways})   # reindex to all cols
                            _have_tot = float(sh[_cands].sum())
                            if _have_tot <= 1e-9:                       # nothing valid had share → uniform over candidates
                                for g in _cands:
                                    sh[g] = 1.0 / len(_cands)
                            else:                                       # keep the primary; give the rest a fallback slice
                                _add = [g for g in _cands if float(sh.get(g, 0.0)) <= 1e-9]
                                if _add:
                                    _resid = max(1.0 - _have_tot, 0.05)
                                    for g in _add:
                                        sh[g] = _resid / len(_add)
                            _ss = float(sh.sum())
                            sh = sh / _ss if _ss > 0 else sh
                    sh = _cap_shares(sh)   # enforce max share (no 100% when ≥2 gateways)
                    row = {"GO LIVE": go_live, "Brand": brand, "RPGT": rpgt, "Currency": cur,
                           "BIN": bin_, "paymentMethodProvider": pmp, "STICKY": "Both", "Country": country}
                    # Round to 2dp, push the residual onto the largest UNDER-cap weight so the
                    # row sums to EXACTLY 100.00 without pushing any gateway over the cap.
                    _rnd = {gw: round(float(sh.get(gw, 0.0)) * 100.0, 2) for gw in gateways}
                    _rsum = round(sum(_rnd.values()), 2)
                    if _rsum > 1e-9 and abs(_rsum - 100.0) > 1e-9:
                        _cappct = round(_cap * 100.0, 2)
                        _cands = [g for g in gateways if _rnd[g] > 0 and _rnd[g] < _cappct - 1e-9]
                        _gwmax = max(_cands, key=lambda g: _rnd[g]) if _cands else max(gateways, key=lambda g: _rnd[g])
                        _rnd[_gwmax] = round(_rnd[_gwmax] + (100.0 - _rsum), 2)
                    for gw in gateways:
                        row[gw] = _rnd[gw]
                    row["Check"] = round(sum(row[gw] for gw in gateways), 2)
                    rows.append(row)
        rdf = pd.DataFrame(rows)
        if not rdf.empty:
            _key = rdf[gateways].round(2).astype(str).agg("|".join, axis=1)
            _codes = {k: f"condition_{i+1}" for i, k in enumerate(dict.fromkeys(_key))}
            rdf["BIN GROUP"] = _key.map(_codes)
            rdf["DUP CHECK"] = 1
        _cols = (["GO LIVE", "BIN GROUP", "Brand", "RPGT", "Currency", "BIN",
                  "paymentMethodProvider", "STICKY", "Country", "Check"] + gateways + ["DUP CHECK"])
        out[(brand, rpgt)] = rdf.reindex(columns=_cols)
    return out


def enforced_prop_items(split, brand, go_live, wallet_incapable=frozenset(), fid2vamp=None,
                        mid_list_path=None, usa_only=frozenset(), country_pres=None,
                        max_share=0.97):
    """Proposed shares AFTER the pipeline's enforcement — cap, wallet-incapable zeroing,
    USA/Non-USA split, and <2-gateway BACK-FILL — taken straight from build_split_exports'
    output, at (Currency, BIN, RPGT, pmp, Country, vampMid) grain. Feeding these into the
    projection reproduces the pipeline's back-fill gateways (WoodForest/Authorize) that the
    raw optimiser split never assigned. Returns a tuple of 7-tuples (hashable for caching)."""
    fid2vamp = dict(fid2vamp or {})
    # Ensure a gatewayFid -> vampMid map (build from Master_MID_List if the caller didn't pass
    # one) — otherwise the gateway columns stay as raw FIDs and never match the export's vampMid,
    # so the projection sees no proposed shares and shows post == pre.
    if mid_list_path and os.path.exists(mid_list_path):
        try:
            _mm = pd.read_csv(mid_list_path)
            _cc = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mm.columns}
            _gx, _vx = _cc.get("gatewayfid"), _cc.get("vampmid")
            if _gx and _vx:
                for _f, _v in zip(_mm[_gx].astype(str).str.strip().str.lower(),
                                  _mm[_vx].astype(str).str.strip()):
                    if _f and _f not in ("", "nan", "none"):
                        fid2vamp.setdefault(_f, _v)
        except Exception:  # noqa: BLE001
            pass
    templates = build_split_exports(
        split, brand, go_live, wallet_incapable=wallet_incapable, fid2vamp=fid2vamp,
        mid_list_path=mid_list_path, usa_only=usa_only, country_pres=country_pres,
        max_share=max_share)
    _meta = {"GO LIVE", "BIN GROUP", "Brand", "RPGT", "Currency", "BIN",
             "paymentMethodProvider", "STICKY", "Country", "Check", "DUP CHECK"}
    frames = []
    for (_brand, _rpgt), wdf in templates.items():
        if wdf is None or getattr(wdf, "empty", True):
            continue
        gw_cols = [c for c in wdf.columns if c not in _meta]
        _idv = [c for c in ["Currency", "BIN", "paymentMethodProvider", "Country"] if c in wdf.columns]
        m = wdf.melt(id_vars=_idv, value_vars=gw_cols, var_name="_gw", value_name="prop_raw")
        m["RPGT"] = str(_rpgt)
        frames.append(m)
    if not frames:
        return tuple()
    allm = pd.concat(frames, ignore_index=True)
    allm["prop_raw"] = pd.to_numeric(allm["prop_raw"], errors="coerce").fillna(0.0)
    allm = allm[allm["prop_raw"] > 0].copy()
    allm["Currency"] = allm["Currency"].astype(str).str.strip().str.lower()
    allm["BIN"] = allm["BIN"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    allm["_pmp"] = (allm["paymentMethodProvider"].astype(str).str.strip().str.lower()
                    if "paymentMethodProvider" in allm.columns else "_all_")
    allm["_ctry"] = (allm["Country"].astype(str).str.strip().str.lower()
                     if "Country" in allm.columns else "_all_")
    allm["vampMid"] = (allm["_gw"].astype(str).str.strip().str.lower().map(fid2vamp)
                       .fillna(allm["_gw"].astype(str).str.strip()))
    agg = allm.groupby(["Currency", "BIN", "RPGT", "_pmp", "_ctry", "vampMid"],
                       as_index=False)["prop_raw"].sum()
    return tuple(agg.itertuples(index=False, name=None))


def enforced_split_frame(split, brand, go_live, wallet_incapable=frozenset(), fid2vamp=None,
                         mid_list_path=None, usa_only=frozenset(), country_pres=None,
                         max_share=0.97):
    """Gateway-grain version of :func:`enforced_prop_items`.

    Returns the proposed split AFTER the pipeline's enforcement (cap, wallet-incapable
    zeroing, USA/Non-USA split, <2-gateway BACK-FILL) as a ``[rpgt, currency, bank, gateway,
    share]`` DataFrame — the SAME enforcement the VAMP projection uses, but keeping the
    gatewayFid so the revenue / success-rate views can reflect the ACTUAL routed gateways
    (e.g. the WoodForest / Authorize back-fill the raw optimiser split never assigned).

    ``bank`` holds the BIN from the export (collapsed to a parent bank downstream via
    bin_to_bank, exactly like the raw split). pmp / Country variants are pooled by MEAN share
    per BIN cell (each variant already sums to 1, so the pooled shares sum to ~1). Share is
    re-normalised per (rpgt, currency, bank) cell. Empty frame if the split yields no rows.
    """
    cols = ["rpgt", "currency", "bank", "gateway", "share"]
    templates = build_split_exports(
        split, brand, go_live, wallet_incapable=wallet_incapable, fid2vamp=fid2vamp,
        mid_list_path=mid_list_path, usa_only=usa_only, country_pres=country_pres,
        max_share=max_share)
    _meta = {"GO LIVE", "BIN GROUP", "Brand", "RPGT", "Currency", "BIN",
             "paymentMethodProvider", "STICKY", "Country", "Check", "DUP CHECK"}
    frames = []
    for (_brand, _rpgt), wdf in templates.items():
        if wdf is None or getattr(wdf, "empty", True):
            continue
        gw_cols = [c for c in wdf.columns if c not in _meta]
        _idv = [c for c in ["Currency", "BIN", "paymentMethodProvider", "Country"] if c in wdf.columns]
        m = wdf.melt(id_vars=_idv, value_vars=gw_cols, var_name="gateway", value_name="w")
        m["rpgt"] = str(_rpgt)
        frames.append(m)
    if not frames:
        return pd.DataFrame(columns=cols)
    allm = pd.concat(frames, ignore_index=True)
    allm["w"] = pd.to_numeric(allm["w"], errors="coerce").fillna(0.0)
    allm = allm[allm["w"] > 0].copy()
    if allm.empty:
        return pd.DataFrame(columns=cols)
    allm["currency"] = allm["Currency"].astype(str).str.strip().str.lower()
    allm["bank"] = allm["BIN"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    allm["gateway"] = allm["gateway"].astype(str).str.strip()
    # Normalise within each export sub-cell (rpgt, currency, BIN, pmp, Country) → share sums to 1.
    _sub = ["rpgt", "currency", "bank"] + [c for c in ["paymentMethodProvider", "Country"] if c in allm.columns]
    _tot = allm.groupby(_sub)["w"].transform("sum")
    allm["_s"] = (allm["w"] / _tot).where(_tot > 0, 0.0)
    # Pool pmp / Country to the BIN grain by MEAN share, then re-normalise per (rpgt, currency, bank).
    out = (allm.groupby(["rpgt", "currency", "bank", "gateway"], as_index=False)["_s"].mean()
           .rename(columns={"_s": "share"}))
    _t2 = out.groupby(["rpgt", "currency", "bank"])["share"].transform("sum")
    out["share"] = (out["share"] / _t2).where(_t2 > 0, 0.0)
    return out[cols]


def count_pools_for_split(split_long, brand_name, go_live, *, wallet_incapable=frozenset(),
                          fid2vamp=None, mid_list_path=None, usa_only=frozenset(),
                          country_pres=None, max_share=0.97, brand_key="tav",
                          date_tag="000000", scheme="vi", mode="sales",
                          extra_priority_amount=200000, emit_generic=False):
    """Number of ConnectorPool configs build_split_exports -> generate_configs would
    produce for a given split. Used by the pool-count-targeting compression so it can
    ask 'how many pools does this cell budget yield?' at each search step. Every arg
    that affects the pool count (brand, wallet/country context, mode, caps) is threaded
    through so the count matches what the real export/config-gen will output.
    """
    from routing_optimiser.connector_pool_configs import generate_configs
    _exp = build_split_exports(
        split_long, brand_name, str(go_live),
        wallet_incapable=wallet_incapable, fid2vamp=fid2vamp, mid_list_path=mid_list_path,
        usa_only=usa_only, country_pres=country_pres, max_share=max_share)
    _pools, _counts = generate_configs(
        _exp, brand_key, date_tag, scheme=scheme, mode=mode,
        extra_priority_amount=int(extra_priority_amount), emit_generic=bool(emit_generic))
    return len(_pools)


def pool_targeted_core(split_ideal, *, target_pools, wallet_ctx, brand_name, brand_key,
                       go_live, mid_list_path, date_tag="000000", mode="sales", scheme="vi",
                       emit_generic=False):
    """Pure (NO session_state) pool-count-targeting compression for `split_ideal`.

    Returns (compressed_long, stats). Because it takes only picklable arguments and touches
    no Streamlit state, it is safe to run in a worker process (joblib/loky) — the dial
    positions are independent and deterministic, so parallelising them gives identical
    output. `pool_targeted_compression` wraps this with the ss cache.
    """
    from routing_optimiser.kmeans_compress import compress_to_pool_budget
    wc = wallet_ctx or {}
    _si = split_ideal.copy()
    if "cell_volume" not in _si.columns:
        _si["cell_volume"] = (_si.groupby(["rpgt", "currency", "bank"])["volume"].transform("sum")
                              if "volume" in _si.columns else 1.0)

    def _count(_cl):
        return count_pools_for_split(
            _cl, brand_name, go_live,
            wallet_incapable=set(wc.get("incapable", set())),
            fid2vamp=wc.get("fid2vamp"),
            mid_list_path=mid_list_path,
            usa_only=set(wc.get("usa_only", set())),
            country_pres=wc.get("country_pres", {}),
            max_share=float(wc.get("max_share", 0.97)),
            brand_key=brand_key, date_tag=date_tag, scheme=scheme, mode=mode,
            emit_generic=emit_generic)

    return compress_to_pool_budget(_si, int(target_pools), _count,
                                   max_gateway_cap=float(wc.get("max_share", 0.97)))


def _pool_disk_key(split_ideal, *, target_pools, wallet_ctx, brand_name, brand_key,
                   go_live, mid_list_path, date_tag, mode, scheme, emit_generic):
    """CONTENT hash of everything the compression output depends on: the split's own values
    (not object identity), all params, and the MID-list mtime. Because it hashes CONTENT, a
    changed split or setting yields a different key — so a disk hit can NEVER be stale."""
    import hashlib as _hl
    import json as _json
    wc = wallet_ctx or {}
    _cols = [c for c in ["rpgt", "currency", "bank", "gateway", "share", "volume",
                         "cell_volume", "baseline_share", "rate"] if c in split_ideal.columns]
    try:
        _h = pd.util.hash_pandas_object(split_ideal[_cols], index=False)
        _split_hash = _hl.sha256(np.ascontiguousarray(_h.to_numpy()).tobytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        _split_hash = _hl.sha256(split_ideal[_cols].to_csv(index=False).encode()).hexdigest()[:16]
    try:
        _midm = os.path.getmtime(mid_list_path)
    except Exception:  # noqa: BLE001
        _midm = 0
    _params = {
        "tp": int(target_pools), "bn": str(brand_name), "bk": str(brand_key),
        "gl": str(go_live), "dt": str(date_tag), "mode": str(mode), "scheme": str(scheme),
        "eg": bool(emit_generic), "midm": _midm,
        "ms": round(float(wc.get("max_share", 0.97)), 6),
        "inc": sorted(str(x) for x in (wc.get("incapable") or set())),
        "uo": sorted(str(x) for x in (wc.get("usa_only") or set())),
        "f2v": _hl.sha256(_json.dumps({str(k): str(v) for k, v in
                          sorted((wc.get("fid2vamp") or {}).items())}).encode()).hexdigest()[:12],
        "cp": _hl.sha256(_json.dumps((wc.get("country_pres") or {}), sort_keys=True,
                          default=str).encode()).hexdigest()[:12],
    }
    return _hl.sha256((_split_hash + _json.dumps(_params, sort_keys=True)).encode()).hexdigest()[:24]


def pool_targeted_compression(ss, split_ideal, *, target_pools, sig, wallet_ctx,
                              brand_name, brand_key, go_live, mid_list_path,
                              date_tag="000000", mode="sales", scheme="vi",
                              emit_generic=False):
    """Run (and cache in ss) the pool-count-targeting compression for `split_ideal`.

    Returns (compressed_long, stats) where the split is trimmed so the GENERATED pool
    count is <= target_pools (or the ideal split unchanged if target<=0 or it already
    fits). The result is cached in ss['_pool_comp'] keyed by `sig`, so the (expensive,
    multi-pass) search only runs when a build/generate button is clicked with settings
    not seen before. `stats` carries raw_cells/raw_pools/cells/pools/global_accuracy/
    feasible for the cards.
    """
    _cache = ss.get("_pool_comp") or {}
    if sig in _cache:
        _e = _cache[sig]
        return _e["long"], _e["stats"]

    # DISK CACHE (content-hash keyed → survives ss clears / restarts / code tweaks, and can NEVER
    # go stale because the key hashes the split's values + all params). A re-run with an unchanged
    # split skips the whole (multi-pass k-means) search.
    _dpath = None
    try:
        _cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
        _dk = _pool_disk_key(split_ideal, target_pools=target_pools, wallet_ctx=wallet_ctx,
                             brand_name=brand_name, brand_key=brand_key, go_live=go_live,
                             mid_list_path=mid_list_path, date_tag=date_tag, mode=mode,
                             scheme=scheme, emit_generic=emit_generic)
        _dpath = os.path.join(_cdir, f"pool_comp_{_dk}.pkl")
        if os.path.exists(_dpath):
            _obj = pd.read_pickle(_dpath)
            _cl, _st = _obj["long"], _obj["stats"]
            _cache[sig] = {"long": _cl, "stats": _st}
            ss["_pool_comp"] = _cache
            return _cl, _st
    except Exception:  # noqa: BLE001
        _dpath = None

    _cl, _st = pool_targeted_core(
        split_ideal, target_pools=target_pools, wallet_ctx=wallet_ctx,
        brand_name=brand_name, brand_key=brand_key, go_live=go_live,
        mid_list_path=mid_list_path, date_tag=date_tag, mode=mode, scheme=scheme,
        emit_generic=emit_generic)
    if _dpath:                                   # persist for future runs (best-effort)
        try:
            import glob as _glob
            os.makedirs(os.path.dirname(_dpath), exist_ok=True)
            pd.to_pickle({"long": _cl, "stats": _st}, _dpath)
            _existing = sorted(_glob.glob(os.path.join(os.path.dirname(_dpath), "pool_comp_*.pkl")),
                               key=os.path.getmtime)
            for _old in _existing[:-60]:         # keep the 60 most-recent compression caches
                try:
                    os.remove(_old)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    _cache[sig] = {"long": _cl, "stats": _st}
    # Keep enough signatures to hold a full 21-position dial sweep (precomputed at
    # variation-generation) plus a few tab-6 (mode/emit_generic) variants.
    if len(_cache) > 64:                      # keep only the most recent signatures
        for _k in list(_cache.keys())[:-64]:
            _cache.pop(_k, None)
    ss["_pool_comp"] = _cache
    return _cl, _st


def rpgt_avg_ticket(cell_agg):
    """RPGT-level average ticket from the 30D actuals (the window ending just before
    Month 0): Σ succ_amount / Σ successes per RPGT. Returns {rpgt_lower: ticket}."""
    if cell_agg is None or getattr(cell_agg, "empty", True):
        return {}
    g = cell_agg.groupby("rpgt_join").agg(rev=("cell_rev", "sum"), succ=("cell_succ", "sum"))
    return {str(rp).strip().lower(): (float(r["rev"]) / float(r["succ"]) if float(r["succ"]) > 0 else 0.0)
            for rp, r in g.iterrows()}


def mid_revenue_month_table(granular, rpgt_ticket, months=range(6)):
    """Per-vampMid × month VI Txn + $Revenue (pre/post) from the pro-rata granular.

    $Revenue = Σ_RPGT ticket[RPGT] × VI_Txn[vampMid, RPGT, month] (RPGT-level ticket
    from the actuals). VI Txn is the origination (t=0) volume, matching the VAMP
    table's 'VI Txn M{m}'. Returns a wide DataFrame: vampMid + per-month
    VI Txn / $Revenue / VI Txn Post / $Revenue Post."""
    g = granular.copy()
    g["period"] = pd.to_numeric(g["period"], errors="coerce").fillna(-1).astype(int)
    _tk = g["RPGT"].astype(str).str.strip().str.lower().map(lambda r: rpgt_ticket.get(r, 0.0)).astype(float)
    _vp = pd.to_numeric(g["VI_Txn_Pre"], errors="coerce").fillna(0.0)
    _vq = pd.to_numeric(g["VI_Txn_Post"], errors="coerce").fillna(0.0)
    g["_rev_pre"] = _vp * _tk
    g["_rev_post"] = _vq * _tk
    g["_vi_pre"] = _vp
    g["_vi_post"] = _vq
    agg = g.groupby(["vampMid", "period"], as_index=False).agg(
        vi_pre=("_vi_pre", "sum"), vi_post=("_vi_post", "sum"),
        rev_pre=("_rev_pre", "sum"), rev_post=("_rev_post", "sum"))
    out = pd.DataFrame({"vampMid": sorted(g["vampMid"].unique())}).set_index("vampMid")
    for m in months:
        _s = agg[agg["period"] == m].set_index("vampMid")
        out[f"VI Txn M{m}"] = _s["vi_pre"] if not _s.empty else 0.0
        out[f"$Revenue M{m}"] = _s["rev_pre"] if not _s.empty else 0.0
        out[f"VI Txn Post M{m}"] = _s["vi_post"] if not _s.empty else 0.0
        out[f"$Revenue Post M{m}"] = _s["rev_post"] if not _s.empty else 0.0
    return out.fillna(0.0).reset_index()
