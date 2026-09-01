"""Impact-tab calculations originally split out of streamlit_app.py (since evolved):
VAMP pre/post projection from the saved exports, wallet-capability lookup, the
production split-template builder, and the small Streamlit cache wrappers that keep
the Impact tab fast. Kept here to keep streamlit_app.py smaller and more organised.

ANALOGY: this module answers "if we deploy the proposed split, what actually changes?" It
replays the proposed routing through the SAME rules production applies (caps, wallet / country
capability, back-fill) and reports the BEFORE→AFTER on VAMP risk and revenue — the "impact".
The cache wrappers just memoise the heavy replays so moving a slider stays snappy."""
from __future__ import annotations

import csv
import io
import os

import time as _time
import numpy as np
import pandas as pd
import streamlit as st

from app_common import load_mid_list, _norm_cols, _renorm_share, _fid2vamp_from  # memoised MID reader + helpers


def _wide_by_mid(mids, vamp_pre, txn_pre, vamp_post, txn_post):
    """Assemble the per-vampMid VAMP/VI-Txn M0–5 (pre & post) wide table from the four
    (vampMid × period) frames. Shared, byte-identical builder for the three projection paths."""
    out = pd.DataFrame({"vampMid": sorted(mids)}).set_index("vampMid")
    for m in range(6):
        out[f"VAMP M{m}"] = vamp_pre[m] if m in vamp_pre.columns else 0.0
        out[f"VI Txn M{m}"] = txn_pre[m] if m in txn_pre.columns else 0.0
        out[f"VAMP Post M{m}"] = vamp_post[m] if m in vamp_post.columns else 0.0
        out[f"VI Txn Post M{m}"] = txn_post[m] if m in txn_post.columns else 0.0
    return out.fillna(0.0).reset_index()


def _apply_keep(t0, excluded_mids, kill_eff, month_0):
    """Set t0['_keep'] IN PLACE = per-row switch-off retention fraction: 0.0 for a binary-excluded
    MID (in excluded_mids and NOT date-gated), else its effective-date keep fraction. Shared,
    byte-identical across the three projection paths. Returns t0."""
    _keep = _mid_keep_fraction(t0["vampMid"], t0["period"], kill_eff, month_0)
    _dated = {m for m, _ in (kill_eff or ())}
    _binary = t0["vampMid"].isin(excluded_mids) & ~t0["vampMid"].isin(_dated)
    t0["_keep"] = np.where(_binary, 0.0, _keep)
    return t0

__build__ = ("2026-08-17b-count-only-pool-search+subcell-exporter+staged-enforcement"
             "+projection-mode-no-round+lt2-backfill-DELETED+no-coarse-prop-fallback+fid-grain-capability+txn-term-stash+denom-stash+t0-presence-backfill+ca-zerocell+vamp-term-stash")


# [FN-246b]
def build_vamp_off_mids(fid2vamp, overrides):
    """vampMids barred from HOLDING OR RECEIVING VAMP — every gatewayFid mapping to them carries
    `apply_to: "vamp"` (or "both") with target 0.

    ONE DEFINITION, called by both tabs. The rule has two halves that are easy to get subtly
    different: gateway ids must be canonicalised the same way, and a vampMid is off only when
    EVERY fid mapping to it is off (the same all-or-nothing rule `excluded_mids` uses). A private
    copy in each caller is how the brand filter ended up comparing "Total AV" against "TotalAV"
    and keeping nothing, twice.

    Returns a frozenset, so it is hashable and stable as an st.cache_data key component.
    """
    from app_common import _vamp_off_gateways
    from routing_optimiser.s2_forecast.vamp_forecast_pipeline import _canonical_gateway

    def _nf(x):
        return str(_canonical_gateway(x)).strip().lower()

    _off = {_nf(_f) for _f in _vamp_off_gateways(overrides if isinstance(overrides, dict) else {})}
    if not _off:
        return frozenset()
    _v2f = {}
    for _f, _v in (fid2vamp or {}).items():
        _v2f.setdefault(_v, set()).add(_nf(_f))
    return frozenset(_v for _v, _fids in _v2f.items() if _fids and _fids <= _off)


# [FN-247]
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


# [FN-248]
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


# [FN-249]
@st.cache_data(show_spinner=False)
def compute_vamp_post_by_mid(tp_path, prop_items, month_0, go_live, excluded_mids=frozenset(),
                             kill_eff=(), mtime: float = 0.0):
    # `mtime` is a cache-key-only argument (unused in the body): pass the tp_path file
    # mtime so a regenerated CSV at the same path busts this @st.cache_data. It must be a
    # PLAIN (non-underscore) name — st.cache_data excludes underscore-prefixed args from
    # the hash, so an `_mtime` name would silently NOT participate in the key.
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

    # [FN-250]
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
    _apply_keep(t0, excluded_mids, kill_eff, month_0)
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
    # VAMP follows the volume: the moved VAMP pool is redistributed by the SAME post-volume
    # share as the moved transactions (prop_eff), so grown MIDs pick up VAMP at the cell's
    # blended rate and Σ VAMP_Post == Σ VAMP_Pre (the cell's VAMP total is conserved).
    t0["_vprop"] = t0["prop_eff"]
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

    return _wide_by_mid(tp["vampMid"].unique(), vamp_pre, txn_pre, vamp_post, txn_post)


# [FN-251]
# [FN-clean-uniq]
def _clean_col(sr, lower=False, strip=True):
    """`sr.astype(str)[.str.strip()][.str.lower()]`, done once per DISTINCT VALUE instead of once
    per row. Returns an object ndarray, character-for-character identical to the per-row chain.

    WHY (19fx). The key-normalise step in compute_vamp_prepost_granular was 14.0s of the 172.9s
    [cvp-timing] measured, and it runs AFTER inject_capable_rows has grown the frame to 6,477,850
    rows. Measured on the real AUG/TotalAV/visa export at that width, those six columns hold
    10,251 distinct values between them:

        Currency                  5        Country                  2
        paymentMethodProvider     3        RPGT                     8
        vampMid                  21        BIN                 10,212

    So the per-row chain lowercases the string "usd" about 1.3 million times to learn one fact.
    Trim-and-lowercase is deterministic -- same input, same output, always -- so doing it once per
    distinct value CANNOT change an answer. That is what makes this the rare speedup with no
    bit-identity risk to weigh at all.

    NOT AS BIG AS THE RATIO LOOKS, and the honest number is 1.84x, not 1000x. Two full-width
    passes survive: pd.factorize still hashes all 6.5M strings, and the take at the end still
    builds a 6.5M object array. Only the CLEANING between them becomes free. Measured 6.60s ->
    3.59s at the run's real width; all six columns verified exactly identical by factorize
    fingerprint (uniques + codes) against the live export.

    NULLS FALL BACK TO THE SLOW CHAIN, and this is not caution -- it is a bug the edge-case test
    caught before this shipped. pd.factorize treats None and NaN as the SAME missing value and
    collapses them into one category, so a column holding both rendered BOTH as "nan", where the
    per-row chain renders None as "none" and NaN as "nan":

        pd.Series([None, nan]).astype(str).str.lower()  ->  ['none', 'nan']    the truth
        _clean_col without this guard                   ->  ['nan',  'nan']    WRONG

    The live export has no nulls in these six columns, so it would not have bitten today -- but
    `_vamp_post_core` takes an arbitrary pre-loaded frame. `isna().any()` is one cheap vectorised
    pass (~10 ms at 6.5M rows) and buys exactness for the case the fast path cannot represent.
    """
    sr = pd.Series(sr) if not isinstance(sr, pd.Series) else sr
    if bool(sr.isna().any()):
        s = sr.astype(str)
        if strip:
            s = s.str.strip()
        if lower:
            s = s.str.lower()
        return s.to_numpy()
    codes, uniq = pd.factorize(sr, use_na_sentinel=False)
    u = pd.Series(uniq).astype(str)
    if strip:
        u = u.str.strip()
    if lower:
        u = u.str.lower()
    return u.to_numpy()[codes]


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


# [FN-252]
def _vamp_post_core(pp, prop_items, excluded_mids=frozenset(), kill_eff=(), month_0=None,
                    scoped_rpgts=()):
    """Core projection on a PRE-LOADED pro-rata dataframe, so the per-MID cap
    feedback loop can re-project candidate splits without re-reading the CSV."""
    pp = pp.copy()
    # 19fx: cleaned once per DISTINCT VALUE, not once per row -- see _clean_col. Identical output.
    pp["Currency"] = _clean_col(pp["Currency"], lower=True)
    pp["BIN"] = _clean_col(pp["BIN"])
    pp["vampMid"] = _clean_col(pp["vampMid"])
    rpgt_col = "RPGT" if "RPGT" in pp.columns else "rpgt"
    pp["RPGT"] = _clean_col(pp[rpgt_col], strip=False)
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
    _apply_keep(t0, excluded_mids, kill_eff, month_0)
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
    # VAMP follows the volume: the moved VAMP pool is redistributed by the SAME post-volume
    # share as the moved transactions (prop_raw, renormalised below to prop_share), so grown
    # MIDs pick up VAMP at the cell's blended rate and the cell's VAMP total is conserved.
    t0["_vprop"] = t0["prop_raw"]
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
    return _wide_by_mid(pp["vampMid"].unique(), vamp_pre, txn_pre, vamp_post, txn_post)


# [FN-253]
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

                # [FN-254]
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

                # [FN-255]
                def _kv(_s):
                    return _s.astype(str).str.strip().str.lower()

                # [FN-256]
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
                    _mdf = load_mid_list(_mid_path)
                    _mdf.columns = [c.strip() for c in _mdf.columns]
                    _f2v = _fid2vamp_from(_mdf, "gatewayFid", "vampMid")
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

                            # [FN-257]
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
                        else:
                            _cl.append("\n=== (1) INPUT-SPLIT DIFF: prop_items not 7-tuple (enforced grain); skipped ===")
                    else:
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

                            # [FN-258]
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


# [FN-259]
def _inject_backfill_rows(pp, prop, prop_name_map=None):
    """#3 ZERO-BASELINE ROW INJECTION. Nothing to do with the deleted <2-gateway share
    back-fill — this adds ROWS, never share.

    The optimiser can route volume to a gateway that has NO baseline row in a cell (it has never
    served that sub-cell, so the pro-rata export has nothing there). The LEFT merge drops it, and
    its routed volume then wrongly redistributes to the MIDs that DO have rows. Re-inject those
    recipients into `pp` as zero-baseline t=0 rows (vampCount=0, VI=0) so they RECEIVE the routed
    volume; VAMP stays 0 for them (no historical VAMP to redistribute).

    Scoped to the enforced (7-tuple) path, which is the only one carrying per-sub-cell shares.
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
    # PRESENCE IS JUDGED ON THE t == 0 SLICE — the frame the caller's LEFT merge actually
    # consumes (`_t0 = pp[pp["t"] == 0]`). Judging it over ALL t (as this did until
    # 2026-08-18h) made a MID with an AGED row but no t0 row read as "present": back-fill
    # skipped it, then the t0 merge found nothing and its enforced share was dropped and
    # renormalised onto the survivors. Measured on the Aug baseline: 3,170 enforced items
    # (156.92 of 14,807 prop mass) vanished across 1,280 sub-cells, and in the worst cases
    # the ENTIRE sub-cell's routing decision was discarded (ghost 1.0000, surviving 0.0000).
    # `valid_sub` moves with it so we still never invent a (pmp, Country) the t0 baseline
    # lacks — injecting into a t>0-only sub-cell would create a t0 cell with cell_tot = 0
    # that contributes nothing but rows.
    _b0 = b[pd.to_numeric(b["t"], errors="coerce").fillna(0).astype(int) == 0]
    present = set(map(tuple, _b0[subk + ["_vml"]].drop_duplicates().to_numpy()))
    valid_sub = set(map(tuple, _b0[subk].drop_duplicates().to_numpy()))
    # Global _vml -> proper-case vampMid, so a MID that exists elsewhere in the export keeps its
    # display name and merges cleanly (no lower-case twin) on the final collapse.
    name_map = b.drop_duplicates("_vml").set_index("_vml")["vampMid"].to_dict()
    # Truly zero-baseline recipients have NO row anywhere in the export, so `b` can't supply a
    # proper-case name and they'd otherwise fall back to the lower-case merge key as their display
    # name. Fill those gaps from the proposed items' proper-case vampMid (sourced from the Master
    # MID list, captured by the caller before `vampMid` is dropped); baseline names keep priority
    # via setdefault so no present MID is renamed.
    if prop_name_map:
        for _k, _v in prop_name_map.items():
            name_map.setdefault(_k, _v)
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


# [FN-260]
# ============================ 19da — CAPABILITY + FRAME INJECTION ============================
# WHY THIS EXISTS. The aged VAMP frame carries a row only where a MID actually had fraud
# originating in that month. Redistribution then shares the movable pool across the rows that
# exist — so in a sparse layer the pool comes OUT of the two MIDs that had fraud and goes straight
# BACK to them, and volume routed to a MID with no row at that age has nowhere to land. Measured
# on the live export: 9.14 CAPABLE MIDs per cell against 2.03 PRESENT per layer.
#
# Completing the movable layers fixes that at the source. It also makes the two projectors agree
# WITHOUT the age renormalise: on the 19da fixture, injection alone takes
# Sigma|delivered - in-search| to 0.000000 with ROUTING_AGE_RENORM=0, and turning the renormalise
# back on leaves every delivered value identical — it is a genuine no-op on a complete frame, so
# it stays in as a guard rather than as the mechanism.
#
# ONLY MOVABLE AGES (t <= period). A frozen layer's pro_rata is 0, so its pool is 0 and an injected
# row there would receive exactly nothing — pure cost. Restricting to movable ages is also what
# keeps injection affordable: 622,592 of 671,364 groups.
#
# THIS CHANGES THE FORECAST, deliberately. Fraud follows the routed volume instead of piling onto
# whichever MID happened to have a row at that age.


def _brand_key(s):
    """Brand comparison key: strip ALL whitespace, then lower-case.

    ONE helper, used by `build_capability` and by `enforced_prop_items`'s gatewayFid->vampMid
    filter, because these two must never disagree about what "the same brand" means. The Master
    MID List spells it "Total AV"; the run's company is "TotalAV" (tab_3_split_outputs_impact:4178). A plain
    strip().lower() matches NOTHING between those two, and BOTH call sites fail SILENTLY when the
    match is empty — build_capability returned 0 capable gateways and injection did nothing on the
    2026-08-28 20:44 run, and an empty fid2vamp would make every proposed share unmatchable and
    report post == pre. Whitespace is not information here, so remove it before comparing.
    """
    return "".join(str(s or "").split()).lower()


_MID_TRUTHY = {"1", "true", "t", "yes", "y"}


def _mid_row_fid(fid, brand_val, is_active, brand_key_wanted):
    """The THREE rules that decide whether a Master MID List row may be a routing RECIPIENT.

    Returns the normalised gatewayFid, or None if the row is not admissible:

      1. PAYPAL is never a recipient — the one `paypal-gbp-tav` row is excluded outright.
      2. INACTIVE gateways are not recipients. This is NOT the same as deleting their history:
         a decommissioned MID like Merrick (1,543 VAMPs) still has rows in the pro-rata export
         and keeps every one of them. This rule only stops it being INJECTED as a zero-baseline
         recipient it can never actually receive into.
      3. The row must belong to the run's BRAND, compared on `_brand_key` (whitespace-stripped),
         because the MID list says "Total AV" and the run says "TotalAV".

    ONE function, used by `build_capability` AND by `enforced_prop_items`'s gatewayFid->vampMid
    map. They had three separate copies of these rules and only build_capability had all three:
    the map was brand-blind, kept inactive gateways, and let PayPal through, which is how 109
    other-brand vampMids plus Paypal, Adyen_TotalCleaner, Braintree - Total AV and
    TrustPayments - Total AV became phantom rows on tab 3. On this MID list the three rules
    together select exactly the 38 gateways / 15 vampMids the run already calls the capable set.
    """
    _f = str(fid or "").strip().lower()
    if not _f or _f in ("", "nan", "none") or "paypal" in _f:
        return None
    if str(is_active or "").strip().lower() not in _MID_TRUTHY:
        return None
    if brand_key_wanted and _brand_key(brand_val) != brand_key_wanted:
        return None
    return _f


def build_capability(mid_list_path=None, restrictions=None, brand=None):
    """Return `capable(cur, bin, rpgt, pmp, ctry) -> frozenset[vampMid]`, memoised.

    A vampMid is CAPABLE in a cell when it has at least one gatewayFid that is
      * IsActive in the Master MID List, of this brand, and not PayPal,
      * of the cell's CURRENCY,
      * not hit by a routing_restrictions `rules` entry for that rpgt / currency / bin,
      * wallet-capable (processWallet) when the cell's pmp is googlepay / applepay,
      * not in `usa_only_gateways` unless the cell is USA.
    That reproduces the engine's own capable-set line exactly (34 gateways -> 15 vampMids on the
    2026-08-28 run), which is the check that this is the SAME definition the router uses and not a
    second one that will drift away from it.
    """
    # impact_calcs has no PROJECT_ROOT of its own (app_common owns it, and importing it here
    # would be circular), so derive it from this file's location the same way app_common does.
    _root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    _p = mid_list_path or os.path.join(_root, "data", "mappings", "Master_MID_List.csv")
    _res = restrictions if isinstance(restrictions, dict) else {}
    _usa_only = {str(x).strip().lower() for x in (_res.get("usa_only_gateways") or [])}
    _rules = _res.get("rules") or []
    # 19de — NORMALISE OUT WHITESPACE. The Master MID List spells the brand "Total AV"; the run
    # spells the company "TotalAV". A plain lower()/strip() comparison therefore matched NOTHING,
    # build_capability returned 0 gateways, and injection silently did nothing on the 2026-08-28
    # 20:44 run — the log read "0 CAPABLE MID(s) (0 gateway(s))" and the frame came back +0 rows.
    # The Σ vampCount assertion did not catch it because injecting nothing conserves trivially.
    _norm = _brand_key          # 19dj: the SHARED key, so the two call sites cannot drift
    _brand = _norm(brand)
    _truthy = {"1", "true", "t", "yes", "y"}

    _gw = []
    with io.open(_p, encoding="utf-8-sig", newline="") as _fh:
        for _r in csv.DictReader(_fh):
            # 19dk: the three admissibility rules now live in ONE place (`_mid_row_fid`), shared
            # with enforced_prop_items' fid->vampMid map. Three copies is how the map ended up
            # brand-blind, inactive-blind and PayPal-blind while this function had all three.
            _fid = _mid_row_fid(_r.get("gatewayFid"), _r.get("brand"),
                                _r.get("IsActive"), _brand)
            if _fid is None:
                continue
            _gw.append((_fid, str(_r.get("vampMid") or "").strip(),
                        str(_r.get("currency") or "").strip().lower(),
                        str(_r.get("processWallet") or "").strip().lower() in _truthy,
                        _fid in _usa_only))

    def _banned(_mid, _fid, _rpgt, _cur, _bin):
        for _ru in _rules:
            _t = str(_ru.get("target") or "").strip().lower()
            if _t not in (_mid.lower(), _fid):
                continue
            _m = _ru.get("match") or {}
            _hit = True
            for _k, _v in _m.items():
                _vals = {str(x).strip().lower()
                         for x in (_v if isinstance(_v, list) else [_v])}
                _got = {"rpgt": _rpgt, "currency": _cur, "bin": _bin}.get(_k)
                # `country` is not part of the routing grain — the config says so and the engine
                # ignores such rules. A rule keyed on anything unrecognised must NOT silently
                # match everything, so an unknown field fails the rule closed.
                if _got is None or _got not in _vals:
                    _hit = False
                    break
            if _hit:
                return True
        return False

    _cache = {}

    def capable(cur, bin_, rpgt, pmp, ctry):
        _k = (cur, bin_, rpgt, pmp, ctry)
        _hit = _cache.get(_k)
        if _hit is not None:
            return _hit
        _wallet = pmp in ("googlepay", "applepay")
        _out = set()
        for _fid, _mid, _cur, _wok, _uonly in _gw:
            if _cur != cur:
                continue
            if _wallet and not _wok:
                continue
            if _uonly and ctry != "usa":
                continue
            if _banned(_mid, _fid, rpgt, cur, bin_):
                continue
            _out.add(_mid)
        _out = frozenset(_out)
        _cache[_k] = _out
        return _out

    capable.n_gateways = len(_gw)
    capable.n_mids = len({_g[1] for _g in _gw})
    # An EMPTY capable set is never a legitimate answer — it means the brand did not match, or the
    # MID list is not the file we think it is. Silently returning it makes injection a no-op that
    # LOOKS like success, which is exactly what happened on 2026-08-28 20:44. Raise instead, so the
    # caller's except-branch logs a SKIPPED line naming the cause.
    if not _gw:
        _brands = sorted({str(_r.get("brand") or "").strip()
                          for _r in csv.DictReader(io.open(_p, encoding="utf-8-sig", newline=""))
                          if str(_r.get("brand") or "").strip()})[:8]
        raise ValueError(
            f"build_capability found NO capable gateway for brand {brand!r} in {_p} — nothing "
            f"would be injected. Brands present: {_brands}. Check the brand spelling (the MID "
            "list uses a space, the run's company name may not).")
    return capable


def inject_capable_rows(pp, capable, cell_cols, mid_col="vampMid",
                        period_col="period", t_col="t"):
    """Complete every MOVABLE (t <= period) group of `pp` with a zero-VAMP row per absent capable
    MID. Returns (frame, n_added). VECTORISED — a per-group Python loop over the live export's
    622,592 movable groups is not viable.

    The injected row carries vampCount 0 and VI_Txn_Count 0, and takes `pro_rata` / `fcp1_frac`
    from its own layer (both are properties of the layer, not of the MID, so every row of a group
    already shares them — asserted below rather than assumed). A zero-VAMP row therefore adds
    NOTHING to the pool and changes no held term; its only effect is to give the layer a recipient
    the split can route to.
    """
    _need = list(cell_cols) + [mid_col, period_col, t_col]
    for _c in _need:
        if _c not in pp.columns:
            raise KeyError(f"inject_capable_rows: frame has no column {_c!r}")
    # 19dd — the test is MOVABILITY, not age. A layer is worth a recipient only if it has a POOL
    # to hand out: pool = Σ vampCount × fcp1_frac × pro_rata over the layer. That subsumes the age
    # test (t > period carries pro_rata 0, so its pool is 0 and it drops out on its own) and it
    # additionally excludes two cases the age test let through:
    #   * fcp1_frac == 0 — the layer's fraud is all FCP 2+ (retries). The split does not route
    #     retries, so none of it is movable. 32,537 movable layers on the live export.
    #   * every MID has vampCount 0 — the layer exists because those MIDs had TRANSACTIONS that
    #     month, but there is no fraud at that age to move. 15,930 layers.
    # An injected row in either case receives pool × share = 0 × share = 0, for every candidate,
    # forever. Measured: 48,761 layers, 424,674 rows that could only ever be zero.
    _per = pd.to_numeric(pp[period_col], errors="coerce")
    _t = pd.to_numeric(pp[t_col], errors="coerce")
    _vc = pd.to_numeric(pp.get("vampCount"), errors="coerce").fillna(0.0)
    _fc = pd.to_numeric(pp.get("fcp1_frac"), errors="coerce").fillna(0.0)
    _prr = pd.to_numeric(pp.get("pro_rata"), errors="coerce").fillna(0.0)
    _gkc = list(cell_cols) + [period_col, t_col]
    _poolg = (pp.assign(_mvbl=_vc * _fc * _prr)
              .groupby(_gkc, observed=True)["_mvbl"].transform("sum"))
    _mov = pp[(_poolg > 0) & _per.notna() & _t.notna()]
    if not len(_mov):
        return pp, 0

    _gk = list(cell_cols) + [period_col, t_col]
    _layer = _mov.groupby(_gk, as_index=False, observed=True).agg(
        pro_rata=("pro_rata", "first"), fcp1_frac=("fcp1_frac", "first"))

    _cells = _mov[list(cell_cols)].drop_duplicates()
    _cells["_cap"] = [sorted(capable(*[str(v).strip().lower() if i != 1 else str(v).strip()
                                       for i, v in enumerate(row)]))
                      for row in _cells.itertuples(index=False, name=None)]
    _cellcap = _cells.explode("_cap").rename(columns={"_cap": mid_col})
    _cellcap = _cellcap[_cellcap[mid_col].notna()]

    _full = _layer.merge(_cellcap, on=list(cell_cols), how="inner")
    # The anti-join key MUST include the MID. Keyed on the group alone every candidate row finds
    # a match, `_seen` is never NaN and the function silently injects nothing — which looks exactly
    # like "the frame was already complete" and would have shipped as a no-op.
    _hk = _gk + [mid_col]
    _have = _mov[_hk].drop_duplicates().assign(_seen=1)
    _new = _full.merge(_have, on=_hk, how="left")
    _new = _new[_new["_seen"].isna()].drop(columns=["_seen"])
    if not len(_new):
        return pp, 0
    _new["vampCount"] = 0.0
    _new["VI_Txn_Count"] = 0.0
    for _c in pp.columns:
        if _c not in _new.columns:
            _new[_c] = pp[_c].iloc[0] if len(pp) else None
    _new = _new[list(pp.columns)]
    _out = pd.concat([pp, _new], ignore_index=True)
    # INVARIANT: injection adds RECIPIENTS, never VAMP. Every injected row carries vampCount 0, so
    # the frame total must be untouched — measured 94,265.00 -> 94,265.00 on the live export. If
    # this ever moves, a row picked up a non-zero count from the column back-fill below and the
    # forecast has silently gained fraud that nobody booked.
    _b = pd.to_numeric(pp.get("vampCount"), errors="coerce").fillna(0.0).sum()
    _a = pd.to_numeric(_out.get("vampCount"), errors="coerce").fillna(0.0).sum()
    if abs(float(_a) - float(_b)) > 1e-6 * max(1.0, abs(float(_b))):
        raise AssertionError(
            f"inject_capable_rows CREATED VAMP: {_b:,.4f} -> {_a:,.4f}. Injected rows must carry "
            "vampCount 0; check the column back-fill.")
    return _out, int(len(_new))


def _max_share_waterfill(shares, t0, grp, cap, live):
    """Port of the search's per-cell max-share water-fill (`band_projection.py:317-347`).

    Line-for-line the same algorithm, vectorised: everything over `cap` is cut back to it, and
    the excess is handed to the under-cap rows of the SAME cell in proportion to the room each
    has left. Repeated up to 50 sweeps, because handing excess out can push a recipient over the
    cap in turn.

    THE THREE THINGS THAT MUST MATCH THE KERNEL, and each of which silently changes the answer:

      1. `_nzc` — the ">= 2 routed gateways" test — is computed ONCE, before the first sweep, and
         is NOT refreshed as rows are capped. A cell with a single routed gateway is left alone
         entirely (capping it would have nowhere to put the excess, so the kernel skips it).
      2. The excess is measured BEFORE the rows are cut to the cap, and the room is measured
         AFTER. Swapping either order changes the redistribution.
      3. Accumulation is in row order within a cell, which is what `np.bincount` does, so the
         floating-point summation order is the kernel's too.

    `live` is the kernel's `_psum[c] > 0` — rows in an unrouted cell take no part.
    """
    _sh = np.asarray(shares, dtype=float).copy()
    if _sh.size == 0:
        return _sh
    _g = t0.groupby(grp, sort=False, observed=True).ngroup().to_numpy()
    _ng = int(_g.max()) + 1
    _live = np.asarray(live, dtype=bool)
    EPS = 1e-12
    # (1) computed ONCE — the kernel does not refresh it between sweeps.
    _nz = np.bincount(_g, weights=(_live & (_sh > EPS)).astype(float), minlength=_ng)
    _multi = (_nz >= 2.0)[_g] & _live
    for _ in range(50):
        _over = _multi & (_sh > cap + EPS)
        if not _over.any():
            break
        # (2) excess BEFORE the cut ...
        _exc = np.bincount(_g, weights=np.where(_over, _sh - cap, 0.0), minlength=_ng)
        _sh = np.where(_over, cap, _sh)
        # ... room AFTER it.
        _room = _multi & (_sh > EPS) & (_sh < cap - EPS)
        _rsum = np.bincount(_g, weights=np.where(_room, cap - _sh, 0.0), minlength=_ng)
        _ok = _room & (_rsum[_g] > EPS)
        _sh = np.where(_ok,
                       _sh + (cap - _sh) / np.where(_ok, _rsum[_g], 1.0) * _exc[_g],
                       _sh)
    return _sh


def compute_vamp_prepost_granular(pp_path, prop_items, excluded_mids=frozenset(),
                                  kill_eff=(), month_0=None, scoped_rpgts=(),
                                  wallet_incapable=frozenset(), usa_only=frozenset(),
                                  exploration_floor=0.0, vamp_off_mids=frozenset(),
                                  capability=None, max_share=1.0):
    """Per-ROW baseline vs proposed VAMP / VI-Txn from the pro-rata export.

    Routes at the (vampMid, RPGT, BIN, Currency, pmp, Country) sub-cell grain when the
    export carries paymentMethodProvider / Country, applying the pipeline's static
    enforcement (wallet-incapable gateways can't serve wallet pmp; USA-only gateways can't
    serve Non-USA) so the projection tracks the pipeline more closely. Result is collapsed
    back to (vampMid, RPGT, BIN, Currency, period, t) for the filterable table.
    """
    # ── [cvp-timing] 19fn: WHERE THE PROJECTION'S TIME GOES ────────────────────────────
    # This function IS the cost of a delivery projection: 185.1s of the 187.6s [never-worse]
    # spent on one candidate on the 2026-09-01 12:23 run, against 2.5s for build_split_exports.
    # Offline I could only account for ~25s of it, so ~160s sat inside one function with no
    # breakdown at all. Marks are recorded in execution order and stashed on the module as
    # `_LAST_CVP_TIMING` for the caller to print — the same stash pattern as the other `_LAST_*`
    # by-products, so it survives [proj-memo]'s snapshot/restore.
    import time as _cv_time
    _cv = {"t": _cv_time.perf_counter(), "rows": []}

    def _cv_mark(_label):
        _n = _cv_time.perf_counter()
        _cv["rows"].append((_label, _n - _cv["t"]))
        _cv["t"] = _n

    pp = pd.read_csv(pp_path)
    _cv_mark("read_csv (the pro-rata export off disk)")
    # 19da — COMPLETE THE MOVABLE LAYERS. Without this the pool of a sparse layer is shared only
    # across the MIDs that happened to have fraud originating in that month, so volume routed to
    # anyone else has no row to land in and the pool circulates back to the incumbents. `capability`
    # is the SAME callable the search is given, so both sides project the identical frame; passing
    # None leaves the frame untouched (every pre-19da caller).
    if capability is not None:
        _cellk = [c for c in ("Currency", "BIN", "RPGT", "paymentMethodProvider", "Country")
                  if c in pp.columns]
        if len(_cellk) >= 3:
            pp, _n_inj = inject_capable_rows(pp, capability, _cellk)
            if _n_inj:
                globals()["_LAST_INJECTED"] = int(_n_inj)
    _cv_mark("inject_capable_rows (zero rows for capable doors)")
    # 19fx: cleaned once per DISTINCT VALUE, not once per row -- see _clean_col. Identical output.
    pp["Currency"] = _clean_col(pp["Currency"], lower=True)
    pp["BIN"] = _clean_col(pp["BIN"])
    pp["vampMid"] = _clean_col(pp["vampMid"])
    rpgt_col = "RPGT" if "RPGT" in pp.columns else "rpgt"
    pp["RPGT"] = _clean_col(pp[rpgt_col], strip=False)
    pp["pro_rata"] = pd.to_numeric(pp.get("pro_rata", 0.0), errors="coerce").fillna(0.0)
    pp["fcp1_frac"] = pd.to_numeric(pp.get("fcp1_frac", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    # Keep pmp / Country sub-cells (default '_all_' when the export lacks them) so the
    # projection can apply the pipeline's per-sub-cell wallet / USA-only enforcement.
    pp["_pmp"] = (_clean_col(pp["paymentMethodProvider"], lower=True)
                  if "paymentMethodProvider" in pp.columns else "_all_")
    pp["_ctry"] = (_clean_col(pp["Country"], lower=True)
                   if "Country" in pp.columns else "_all_")
    _cv_mark("key normalise (string strip/lower on 6 columns)")
    pp = pp.groupby(["vampMid", "RPGT", "BIN", "Currency", "_pmp", "_ctry", "period", "t"],
                    as_index=False).agg(
        vampCount=("vampCount", "sum"), VI_Txn_Count=("VI_Txn_Count", "sum"),
        pro_rata=("pro_rata", "first"), fcp1_frac=("fcp1_frac", "first"))
    _cv_mark("collapse groupby (8 string keys)")

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
    # Proper-case display name per _vml, captured BEFORE `vampMid` is dropped, so back-fill
    # injection can name truly zero-baseline recipients (which have no row in the export).
    _prop_name_map = (prop.dropna(subset=["vampMid"]).drop_duplicates("_vml")
                      .set_index("_vml")["vampMid"].to_dict())
    prop = prop.drop(columns=["vampMid"])
    if _by_rpgt:
        prop["_rpgtl"] = prop["RPGT"].astype(str).str.strip().str.lower()
        prop = prop.drop(columns=["RPGT"])
    if _enforced:
        prop["_pmp"] = prop["_pmp"].astype(str).str.strip().str.lower()
        prop["_ctry"] = prop["_ctry"].astype(str).str.strip().str.lower()
        _cv_mark("prop frame build (prop_items -> DataFrame + keys)")
        pp = _inject_backfill_rows(pp, prop, _prop_name_map)   # #3 add zero-baseline back-fill target rows
        _cv_mark("_inject_backfill_rows (zero-baseline recipients)")

    grp = ["Currency", "BIN", "RPGT", "_pmp", "_ctry", "period"]
    if _enforced:
        _t0 = pp[pp["t"] == 0].copy()
        _t0["_rpgtl"] = _t0["RPGT"].astype(str).str.strip().str.lower()
        _t0["_vml"] = _t0["vampMid"].astype(str).str.strip().str.lower()
        t0 = _t0.merge(prop, on=["Currency", "BIN", "_rpgtl", "_pmp", "_ctry", "_vml"], how="left")
        t0["_prop_from_coarse"] = 0.0   # DIAGNOSTIC flag
        # HIERARCHICAL coarse (pmp, Country) MEAN fallback — REMOVED 2026-08-17.
        # It filled sub-cells whose exact 6-key merge missed with a MEAN of the enforced
        # share over the surviving sub-cells. A mean is not a routing decision: for a
        # Country/pmp-concentrated gateway it HALVES the share (WoodForest, named in the
        # original comment), and it has NO in-search analogue at all — so every row it
        # touched was pure scored-vs-delivered drift the GA could never model.
        # An unmatched sub-cell now keeps prop_raw = NaN → 0 just below, so prop_sum = 0
        # there, _move = 0, and the cell is HELD AT BASELINE — exactly what the band
        # scaffold does for a cell it cannot represent.
        # Kill-switch: ROUTING_COARSE_PROP_FALLBACK=1 restores it.
        if os.environ.get("ROUTING_COARSE_PROP_FALLBACK", "0") == "1":
            for _fk in (["Currency", "BIN", "_rpgtl", "_ctry", "_vml"],
                        ["Currency", "BIN", "_rpgtl", "_pmp", "_vml"],
                        ["Currency", "BIN", "_rpgtl", "_vml"]):
                if not t0["prop_raw"].isna().any():
                    break
                _cm = prop.groupby(_fk, as_index=False)["prop_raw"].mean().rename(
                    columns={"prop_raw": "_pc"})
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
    _cv_mark("t0 slice + the prop merge")
    # Effective-date-gated switch-off (see _vamp_post_core / _mid_keep_fraction).
    _apply_keep(t0, excluded_mids, kill_eff, month_0)
    # _mid_keep_fraction inside this runs a PYTHON for-loop over every t0 row whenever
    # kill_eff is non-empty, which is why it is marked on its own.
    _cv_mark("_apply_keep (switch-off gating; per-row Python loop)")
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
    _cv_mark("per-cell transforms (cell_tot / _at / base_share / prop_sum)")
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
    # PER-MID movable fraction + VAMP-follows-the-volume redistribution share (see _vamp_post_core).
    t0["_move"] = np.where(t0["prop_sum"] > 0, t0["pro_rata"] * t0["fcp1_frac"], 0.0)
    # 19cv — VAMP-ELIGIBILITY. `apply_to:"vamp"` (target 0) IS honoured by the baseline pipeline
    # (those MIDs carry vampPre 0 against real VI_Txn) but was NOT honoured here: the recipient
    # share came straight from `prop_raw`, so an overridden MID held no VAMP of its own and was
    # then handed a slice of the moved pool. WoodForest 690 and Authorize 227 on the 14:39 run,
    # both from PRE 0. Zeroing `_vprop` removes them from the numerator AND the denominator, so
    # the remaining recipients absorb the whole pool and the cell VAMP total still conserves.
    #
    # NOT the same as `_keep` above: `_keep` zeroes prop_raw itself, which would also remove the
    # MID's TRANSACTIONS. apply_to:"vamp" must leave the transactions alone — that is the whole
    # point of the value — so the mask belongs here, on the VAMP share only, and post_txn above
    # is deliberately computed from the UNMASKED prop_share.
    #
    # The Bank x Currency `compute_vamp_prepost` above carries the SAME three lines unmasked. It
    # is not on the shipped delivery path (tab 3 calls the granular one), so it is left alone
    # rather than given a parameter no caller passes -- but if it is ever revived it needs this.
    # 19df — THE MAX-SHARE CAP, WHICH THE SEARCH HAD AND DELIVERY DID NOT.
    #
    # `band_projection.py:299` is commented `vshare from the (capped) ROUTED share`: the search
    # renormalises prop_raw to a cell share, runs the max-share water-fill, and THEN builds
    # vshare from the capped value. Delivery built vshare from raw `prop_raw`, which carries no
    # cap at all. Same shares in, two different recipients — and the difference favours exactly
    # the MIDs the cap exists to restrain, so a capped incumbent is SCORED at the cap and
    # DELIVERED above it. That is the sign pattern on the 2026-08-28 21:25 run: adyen_totalav
    # +186, braintree usa +156, worldpay +87 (all capped incumbents, delivered high) against
    # paysafe -156 and checkout -51 (small recipients, delivered low).
    #
    # Measured on the _19df fixture (one cell, adyen 99 vs 0.5 / 0.5, cap 0.97): Σ|delivered −
    # in-search| is 0.000000 at max_share 1.0 and 8.384 at 0.97, with DELIVERY returning the
    # identical numbers in both regimes — it never saw the cap. One variable, both shipped
    # functions, no code patched to produce it.
    #
    # THIS CHANGES WHAT SHIPS. The delivered M5 is the authoritative number and this moves it.
    # ROUTING_DELIV_MAXSHARE=0 reverts.
    #
    # NO-OP WHEN THE CAP DOES NOT BIND, by construction: with no row over the cap the capped
    # share is prop_raw / prop_sum, so vshare = (prop_raw/prop_sum) / Σ(prop_raw/prop_sum) =
    # prop_raw / Σ prop_raw — the previous line exactly. `test_19df` asserts that bit-identically
    # rather than trusting the algebra.
    #
    # DELIBERATELY NOT `prop_share`: that column also carries the 0.01 exploration floor (applied
    # at 1405-1408), which the search's water-fill does not apply. The `cf_ps` counterfactual uses
    # it and therefore OVERSTATES this effect 4-6x — it flips two things at once. That is why
    # cf_ps read 5,128.8 against an 829 reconciliation error and could not be taken at face value.
    #
    # `_LAST_DELIV_MAXSHARE` RECORDS WHAT ACTUALLY HAPPENED, not what was configured. The 19df
    # log line first keyed off ROUTING_DELIV_MAXSHARE alone and so announced "the CAP is now
    # applied on BOTH sides" on the 2026-08-29 07:45 run, where it was FALSE: three call sites in
    # tab_2_routing_engine still passed no max_share, so `max_share` defaulted to 1.0, the guard below took
    # the raw branch, and the run reproduced 829 exactly. The env var is INTENT; this global is
    # FACT, and the log must only ever report the second. None = the cap was not applied.
    _cap_ms = float(max_share) if max_share else 1.0
    if os.environ.get("ROUTING_DELIV_MAXSHARE", "1") == "0" or not (0.0 < _cap_ms < 1.0):
        globals()["_LAST_DELIV_MAXSHARE"] = None
        t0["_vprop"] = t0["prop_raw"]
    else:
        globals()["_LAST_DELIV_MAXSHARE"] = _cap_ms
        _psum_c = t0.groupby(grp)["prop_raw"].transform("sum")
        _live = (_psum_c > 0).to_numpy()
        _sh = np.where(_live, t0["prop_raw"].to_numpy(float)
                       / np.where(_live, _psum_c.to_numpy(float), 1.0), 0.0)
        t0["_vprop"] = _max_share_waterfill(_sh, t0, grp, _cap_ms, _live)
    if len(vamp_off_mids):
        _voff = {str(_m).strip().lower() for _m in vamp_off_mids}
        _vmask = ~t0["vampMid"].astype(str).str.strip().str.lower().isin(_voff)
        t0["_vprop"] = t0["_vprop"] * _vmask.astype(float)
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

    # ── TXN TERM STASH (read-only) ────────────────────────────────────────────────
    # post = cell_tot·(base_share·(1−move) + moved_tot·prop_share). These columns exist only
    # inside this function and are dropped on return, so the reconcile can never compare terms
    # with the in-search projector. Stash the per-(vampMid, period) sums here — nothing is
    # modified and nothing downstream reads this global.
    try:
        _tt_ct = pd.to_numeric(t0["cell_tot"], errors="coerce").fillna(0.0)
        _tt_bs = pd.to_numeric(t0["base_share"], errors="coerce").fillna(0.0)
        _tt_mv = pd.to_numeric(t0["_move"], errors="coerce").fillna(0.0)
        _tt_mt = pd.to_numeric(t0["_moved_tot"], errors="coerce").fillna(0.0)
        _tt_ps = pd.to_numeric(t0["prop_share"], errors="coerce").fillna(0.0)
        # ── DENOMINATOR STASH (read-only, opt-in) ──────────────────────────────────
        # prop_share = prop_raw / prop_sum, and the residual is now known to live here.
        # Stash the per-row numerator and the per-cell denominator so the reconcile can
        # compare them against the in-search prop_raw / psum on the SAME prop vector.
        # Gated on _RECON_MIDS (a set of lower-cased vampMids the reconcile sets just
        # before this call) and restricted to the cells those MIDs actually occupy, so
        # a normal run computes nothing and stores nothing.
        _rmD = globals().get("_RECON_MIDS")
        if _rmD:
            try:
                _ckD = (t0["Currency"].astype(str).str.strip().str.lower() + "|"
                        + t0["BIN"].astype(str).str.strip() + "|"
                        + t0["RPGT"].astype(str).str.strip().str.lower() + "|"
                        + t0["_pmp"].astype(str).str.strip().str.lower() + "|"
                        + t0["_ctry"].astype(str).str.strip().str.lower())
                _mlD = t0["vampMid"].astype(str).str.strip().str.lower()
                _perD = pd.to_numeric(t0["period"], errors="coerce").fillna(-1).astype(int)
                _cpkD = _ckD + "|" + _perD.astype(str)
                _wantD = _mlD.isin({str(x).strip().lower() for x in _rmD}) & (
                    (_tt_ps > 0) | (_tt_bs > 0))
                _selD = _cpkD.isin(set(_cpkD[_wantD].unique().tolist()))
                _keepD = (pd.to_numeric(t0["_keep"], errors="coerce").fillna(1.0)
                          if "_keep" in t0.columns else pd.Series(1.0, index=t0.index))
                _bfD = (pd.to_numeric(t0["_bf_inj"], errors="coerce").fillna(0.0)
                        if "_bf_inj" in t0.columns else pd.Series(0.0, index=t0.index))
                globals()["_LAST_TXN_DENOM"] = pd.DataFrame({
                    "ck": _ckD[_selD], "per": _perD[_selD], "midl": _mlD[_selD],
                    "praw": pd.to_numeric(t0["prop_raw"], errors="coerce").fillna(0.0)[_selD],
                    "psum": pd.to_numeric(t0["prop_sum"], errors="coerce").fillna(0.0)[_selD],
                    "pshare": _tt_ps[_selD], "base": _tt_bs[_selD], "ctot": _tt_ct[_selD],
                    "mvt": _tt_mt[_selD], "mv": _tt_mv[_selD],
                    "keep": _keepD[_selD], "bf": _bfD[_selD],
                }).reset_index(drop=True)
            except Exception:  # noqa: BLE001
                globals()["_LAST_TXN_DENOM"] = None
        globals()["_LAST_TXN_TERMS"] = pd.DataFrame({
            "midl": t0["vampMid"].astype(str).str.strip().str.lower(),
            "per": pd.to_numeric(t0["period"], errors="coerce").fillna(-1).astype(int),
            "pre": _tt_ct * _tt_bs,
            "held": _tt_ct * _tt_bs * (1.0 - _tt_mv),
            "out": _tt_ct * _tt_bs * _tt_mv,
            "inn": _tt_ct * _tt_mt * _tt_ps,
            "pool": _tt_ct * _tt_mt,
        }).groupby(["midl", "per"], as_index=False).sum()
    except Exception:  # noqa: BLE001 — a diagnostic must never break the projection
        globals()["_LAST_TXN_TERMS"] = None

    _sub = ["Currency", "BIN", "RPGT", "_pmp", "_ctry"]
    # #2 GO-LIVE TIMING: the pipeline applies the go-live weight by the APPEARANCE month
    # (target month m), not origination. So take the rule×cohort factor (_gf = fcp1 × has-rule
    # × scope, WITHOUT pro_rata) at the ORIGINATION cell, and multiply by the go-live pro_rata
    # of the APPEARANCE month (the t0 pro_rata at that period). VI-txn (t=0) is unchanged
    # because appearance == origination there.
    # BUGFIX: gate the VAMP move on the MID being ACTIVE (_keep>0), not just the cell being routed
    # (prop_sum>0). A switched-off vampMid (target=0 → _keep=0) has no transactions to re-route, so
    # its residual/baseline VAMP must pass through unchanged (VAMP_Post == VAMP_Pre). Previously the
    # cell-level go-live ramp "moved out" fraud from 0-transaction MIDs (e.g. EPX), draining it into
    # the pool and breaking the monthly Σ VAMP_Post == Σ VAMP_Pre conservation.
    t0["_gf"] = np.where((t0["prop_sum"] > 0) & (t0["_keep"] > 0), t0["fcp1_frac"], 0.0)
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
    # CONSERVATION: the moved VAMP pool must fully redistribute, so the recipient shares must sum to
    # 1 within each redistribution group. Merge misses / source differences can leave Σ_pshare < 1,
    # which leaks fraud out of the monthly total. Renormalise per group; and where a group has NO
    # valid VAMP recipient (Σ_pshare = 0) keep the VAMP in place (no move) rather than vanish it.
    _gk = _sub + ["period", "t"]
    _psum = pp.groupby(_gk)["_pshare"].transform("sum")
    # 19di — WHICH GATE MAKES DELIVERY'S MOVABLE FRACTION SMALLER THAN THE SEARCH'S?
    #
    # [vterms-is] on the 2026-08-29 09:06 run showed Δ HELD and Δ MOVED-OUT EQUAL AND OPPOSITE on
    # all twelve VAMP bands, always the same sign: identical PRE, and delivery holding +243 that
    # the search moves. Same total, different line. `_gf` above carries THREE gates the search's
    # `pc_heldfac` (band_projection:1435 = fcp[origin] x pro_rata[appearance]) does not have:
    #
    #     1. `_keep > 0`      switched-off vampMid (volume override target=0)
    #     2. `_oos`           unscoped RPGT zeroing
    #     3. the passthrough  Σ_pshare == 0 in the aged group  (below)
    #
    # Reading the code cannot rank them and neither can a whole-book counterfactual — that is
    # exactly how three runs went into the max-share cap, which turned out to be worth 1.2 units.
    # So MEASURE each gate on its own, one variable at a time, on the live data. Each variant
    # below re-derives the movable VAMP with ONE gate lifted and nothing else changed; whichever
    # closes the +243 IS the cause, and if none does, the cause is not a gate at all.
    #
    # Cost is four vectorised passes over a frame already in memory — no extra projection.
    try:
        _gk_v = _gk
        _pra = pd.to_numeric(pp["_pr_app"], errors="coerce").fillna(0.0)
        _om = pd.to_numeric(pp["orig_m"], errors="coerce").fillna(-1)
        _vcv = pd.to_numeric(pp["vampCount"], errors="coerce").fillna(0.0)
        _t0v = t0[_sub + ["vampMid", "period"]].copy()
        _t0v["_ps"] = pd.to_numeric(t0["prop_sum"], errors="coerce").fillna(0.0)
        _t0v["_kp"] = pd.to_numeric(t0["_keep"], errors="coerce").fillna(0.0)
        _t0v["_fc"] = pd.to_numeric(t0["fcp1_frac"], errors="coerce").fillna(0.0)
        _t0v["_oosf"] = 0.0
        if scoped_rpgts:
            _t0v.loc[_oos, "_oosf"] = 1.0
        _t0v = _t0v.rename(columns={"period": "orig_m"})
        _ppv = pp[_sub + ["vampMid", "orig_m"]].merge(
            _t0v, on=_sub + ["vampMid", "orig_m"], how="left")
        _ps = _ppv["_ps"].fillna(0.0).to_numpy()
        _kp = _ppv["_kp"].fillna(0.0).to_numpy()
        _fc = _ppv["_fc"].fillna(0.0).to_numpy()
        _oo = _ppv["_oosf"].fillna(0.0).to_numpy() > 0.5
        _live = (_om.to_numpy() >= 0) & (_ps > 0)
        _pt = (_psum.to_numpy() > 1e-12)          # the passthrough test, as shipped

        def _mv_of(keep_gate, scope_gate, pass_gate):
            _g = np.where(_live & ((_kp > 0) if keep_gate else True), _fc, 0.0)
            if scope_gate:
                _g = np.where(_oo, 0.0, _g)
            _m = _g * _pra.to_numpy()
            if pass_gate:
                _m = np.where(_pt, _m, 0.0)
            return _vcv.to_numpy() * _m

        _vg = pd.DataFrame({
            "midl": pp["vampMid"].astype(str).str.strip().str.lower().to_numpy(),
            "per": pd.to_numeric(pp["period"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            "shipped": _mv_of(True, True, True),
            "no_keep": _mv_of(False, True, True),
            "no_scope": _mv_of(True, False, True),
            "no_pass": _mv_of(True, True, False),
            "no_gates": _mv_of(False, False, False),
        })
        globals()["_LAST_MOVE_GATES"] = _vg.groupby(["midl", "per"], as_index=False).sum()
    except Exception:  # noqa: BLE001
        globals()["_LAST_MOVE_GATES"] = None

    # 19dl — WHICH GROUPS DOES THE PASSTHROUGH FIRE ON? The gate attribution proved this line is
    # the cause of Part A (delivery holds 251 VAMP the search moves), but not WHY the two sides'
    # "is there a recipient?" test disagrees. Both sum a recipient share per aged group and hold
    # when it is zero; they reach different totals. Two candidates were on the table and one is
    # already dead: measured on the export, ALL 573,831 movable groups have an origin t0 row for
    # every row, so "the origin does not exist" is NOT it. What survives is that the two sides sum
    # over different ROWS, or attach different SHARES to the same rows — and only the live run has
    # both sides, so the comparison has to happen here.
    #
    # Stash the FIRED SET only (not every group): the disagreement is set difference in both
    # directions, and delivery-fired plus the search's own fired set is enough for both. Detail
    # rows are capped so a diagnostic can never dominate the run.
    try:
        _pt_gf = pd.to_numeric(pp["_gf"], errors="coerce").fillna(0.0).to_numpy()
        _pt_pra = pd.to_numeric(pp["_pr_app"], errors="coerce").fillna(0.0).to_numpy()
        _pt_vc = pd.to_numeric(pp["vampCount"], errors="coerce").fillna(0.0).to_numpy()
        _pt_om = pd.to_numeric(pp["orig_m"], errors="coerce").fillna(-1).to_numpy()
        # movable BEFORE the passthrough zeroes it — otherwise the test is circular
        _pt_mv = np.where(_pt_om >= 0, _pt_vc * _pt_gf * _pt_pra, 0.0)
        _pt_key = (pp["Currency"].astype(str) + "|" + pp["BIN"].astype(str) + "|"
                   + pp["RPGT"].astype(str) + "|" + pp["_pmp"].astype(str) + "|"
                   + pp["_ctry"].astype(str) + "|" + pp["period"].astype(str) + "|"
                   + pp["t"].astype(str))
        _pt = pd.DataFrame({"k": _pt_key.to_numpy(), "mv": _pt_mv,
                            "ps": pd.to_numeric(pp["_pshare"], errors="coerce")
                            .fillna(0.0).to_numpy()})
        _pg = _pt.groupby("k", observed=True).agg(mv=("mv", "sum"), ps=("ps", "sum"),
                                                  n=("ps", "size"))
        _pg = _pg[_pg["mv"] > 0.0]                       # movable groups only
        _fired = _pg[_pg["ps"] <= 1e-12]
        # 19do — 400, not 40. The sample is ordered by movable VAMP and the OUT-OF-SCOPE
        # months dominate that ordering, so at 40 every group the 12:36 run printed was a
        # month the search does not carry — the half that cannot be evidence. The block
        # now filters to the common population and needs some to survive the filter, or
        # it can only report a count and ask for another run.
        _samp = _fired.sort_values("mv", ascending=False).head(400).index.tolist()
        _det = {}
        if _samp:
            _sub_d = _pt[_pt["k"].isin(set(_samp))]
            _mid_d = pp["vampMid"].astype(str).str.strip().to_numpy()
            _sub_d = _sub_d.assign(mid=_mid_d[_sub_d.index.to_numpy()])
            for _k, _grp in _sub_d.groupby("k", observed=True):
                _det[str(_k)] = [(str(r.mid), float(r.ps), float(r.mv))
                                 for r in _grp.head(16).itertuples()]
        globals()["_LAST_PASSTHRU"] = {
            "fired": set(_fired.index.astype(str)),
            # 19dn — the MOVABLE UNIVERSE, not only the fired subset. Two fired sets that share
            # no key are either a key-format bug or a total disagreement, and the counts cannot
            # tell those apart; the universes can. `sample` keeps two RAW keys so the exact
            # spelling is visible in the log rather than inferred from the code.
            "all": set(_pg.index.astype(str)),
            "sample": [str(_ks9) for _ks9 in _pg.index[:2]],
            "n_movable": int(len(_pg)),
            "mv_held": float(_fired["mv"].sum()),
            "detail": _det}
    except Exception:  # noqa: BLE001
        globals()["_LAST_PASSTHRU"] = None

    pp["_move"] = np.where(_psum > 1e-12, pp["_move"], 0.0)               # no recipient → passthrough
    pp["_pshare"] = np.where(_psum > 1e-12, pp["_pshare"] / _psum, 0.0)   # recipients sum to exactly 1
    pp["VAMP_Pre"] = pp["vampCount"]
    pp["_moved_v"] = pp["vampCount"] * pp["_move"]
    pp["_moved_vpool"] = pp.groupby(_gk)["_moved_v"].transform("sum")
    pp["VAMP_Post"] = pp["vampCount"] * (1.0 - pp["_move"]) + pp["_moved_vpool"] * pp["_pshare"]

    # ── WHY IS Σ_pshare < 1? (19cr, read-only) ─────────────────────────────────────────────
    # The renormalise above repairs a shortfall. Whether it is the RIGHT repair depends entirely on
    # why the shortfall exists, and Σ_pshare cannot tell you: an intended recipient missing from a
    # group is either a MID that genuinely has no fraud of that age (STRUCTURAL — re-basing is
    # correct, and the in-search projector should get the same pass) or a MID the aged frame never
    # carries for that cell at all (ABSENT — the frame is short and both sides are wrong). This
    # splits the missing share mass across those two classes and names who carries each.
    _cv_mark("recipient share + move fractions (_move / _pshare / VAMP_Post)")
    # 19fq: ROUTING_PSHARE_WHY DELETED. This stash is not optional instrumentation — tab 2 reads
    # it at two sites to explain a recipient-share drift, and a switch that can turn off the only
    # explanation of a number you are driving to zero is a switch that will be found off on the
    # run where it mattered. Unconditional now.
    if True:
        _pw_t_0 = _time.perf_counter()
        try:
            # INTENDED recipients, at origination: the rows whose vshare made up the 1.0. This is
            # keyed on `grp = _sub + ["period"]` (see above), which is exactly the key `_mv` is
            # merged back on, so these shares sum to 1 over the cell-month by construction.
            _pw_t0 = t0[_sub + ["vampMid", "period", "_vshare"]].copy()
            _pw_t0["_vshare"] = pd.to_numeric(_pw_t0["_vshare"], errors="coerce").fillna(0.0)
            _pw_t0 = _pw_t0[_pw_t0["_vshare"] > 0.0].rename(columns={"period": "orig_m"})

            # THE GROUP IS `_gk = _sub + ["period", "t"]`, NOT `orig_m`. 19cs: the previous version
            # tested membership at (cell, vampMid, orig_m), and orig_m = period - t is MANY-TO-ONE
            # over _gk — every (period, t) on one diagonal shares an orig_m, so a MID present at any
            # single age was scored present at every age. It found 0 missing on a frame with 168,945
            # short groups. Membership must be asked of the group that has to sum to 1.
            _pw_grp = pp.assign(_pw_s=_psum)[_gk + ["orig_m", "_pw_s"]].drop_duplicates(_gk)
            _pw_grp = _pw_grp[_pw_grp["_pw_s"] > 1e-12]        # passthrough groups are not repaired
            _pw_short = float((1.0 - _pw_grp["_pw_s"]).clip(lower=0.0).sum())

            # COUNTED, NOT EXPANDED. Expanding groups x intended recipients cost 13.1 s at the live
            # shape and was OOM-killed at 4x it. The expansion was only counting: an intended
            # (cell, orig_m, mid) with share v misses  v x (live groups there - live groups there
            # that carry a row for mid). Both are groupby sizes, so this is the same sum
            # reassociated onto the 700k-row INTENDED table. The reconciliation guard below is what
            # holds the reassociation honest.
            _pw_ng = (_pw_grp.groupby(_sub + ["orig_m"], as_index=False)
                      .size().rename(columns={"size": "_pw_nlive"}))
            _pw_seen = (_pw_grp[_gk + ["orig_m"]]
                        .merge(pp[_gk + ["vampMid"]].drop_duplicates(), on=_gk, how="inner")
                        .groupby(_sub + ["orig_m", "vampMid"], as_index=False)
                        .size().rename(columns={"size": "_pw_nseen"}))
            _pw_miss = _pw_t0.merge(_pw_ng, on=_sub + ["orig_m"], how="inner")
            _pw_miss = _pw_miss.merge(_pw_seen, on=_sub + ["orig_m", "vampMid"], how="left")
            _pw_miss["_pw_nseen"] = _pw_miss["_pw_nseen"].fillna(0.0)
            _pw_miss["_pw_ngap"] = (_pw_miss["_pw_nlive"] - _pw_miss["_pw_nseen"]).clip(lower=0.0)
            _pw_miss = _pw_miss[_pw_miss["_pw_ngap"] > 0].copy()
            # `_vshare` is now the SHARE MASS MISSED, not the per-group share: one intended row can
            # be missing from many groups. The class totals below sum this, so they stay comparable
            # with `_pw_short`, which is also summed over groups.
            _pw_slots = float(_pw_miss["_pw_ngap"].sum())
            _pw_miss["_vshare"] = _pw_miss["_vshare"] * _pw_miss["_pw_ngap"]

            # Of the MISSING, which appear ANYWHERE in the aged frame for that cell? A MID that does
            # appear elsewhere is known to the frame and merely has no fraud of this age; one that
            # never appears cannot be represented at all.
            _pw_any = pp[_sub + ["vampMid"]].drop_duplicates().assign(_pw_any=1)
            _pw_miss = _pw_miss.merge(_pw_any, on=_sub + ["vampMid"], how="left")
            _pw_struct = _pw_miss[_pw_miss["_pw_any"] == 1]
            _pw_absent = _pw_miss[_pw_miss["_pw_any"].isna()]
            _pw_found = float(_pw_miss["_vshare"].sum())
            _pw_tot = _pw_found or 1.0

            def _pw_top(_d):
                if _d.empty:
                    return []
                _g = (_d.assign(_m=_d["vampMid"].astype(str).str.strip().str.lower())
                      .groupby("_m")["_vshare"].sum().sort_values(ascending=False))
                return [(str(_k), float(_v)) for _k, _v in _g.head(6).items()]

            # THE GUARD. `1 - Σ_pshare` on a live group IS the intended share that had no row, so
            # the two must agree. While they do not, this block has no standing to say anything
            # about where the renormalise belongs, and `reconciles` is what stops it saying it.
            _pw_gap = abs(_pw_found - _pw_short)
            globals()["_LAST_PSHARE_WHY"] = {
                "intended_share": float(_pw_t0["_vshare"].sum()),
                "intended_rows": int(len(_pw_t0)),
                "missing_rows": int(len(_pw_miss)),
                "missing_slots": int(_pw_slots),
                "live_groups": int(len(_pw_grp)),
                "shortfall": _pw_short,
                "found": _pw_found,
                "gap": _pw_gap,
                "reconciles": bool(_pw_gap <= 1e-6 * max(1.0, _pw_short)),
                "structural_share": float(_pw_struct["_vshare"].sum()),
                "structural_rows": int(len(_pw_struct)),
                "absent_share": float(_pw_absent["_vshare"].sum()),
                "absent_rows": int(len(_pw_absent)),
                "structural_pct": 100.0 * float(_pw_struct["_vshare"].sum()) / _pw_tot,
                "absent_pct": 100.0 * float(_pw_absent["_vshare"].sum()) / _pw_tot,
                "structural_top": _pw_top(_pw_struct),
                "absent_top": _pw_top(_pw_absent),
                "secs": _time.perf_counter() - _pw_t_0,
            }
        except Exception as _pw_e:  # noqa: BLE001 — a diagnostic must never break the projection
            globals()["_LAST_PSHARE_WHY"] = {"error": f"{type(_pw_e).__name__}: {_pw_e}"}
    else:
        globals()["_LAST_PSHARE_WHY"] = None

    # ── VAMP TERM STASH (read-only) ────────────────────────────────────────────────────────
    # VAMP_Post = vampCount*(1-move) + moved_vpool*_pshare, i.e. HELD + MOVED-IN, exactly the
    # shape the TXN `[terms]` block decomposes. `[terms]` is TXN-only, so no VAMP band has ever
    # been decomposed — and the entire remaining DELIVERY DRIFT (+5, all of it worldpay [vamp],
    # all of it in the ROUTED leg, 79% of it in the PROJECTOR-swap step) lives on VAMP. Stash the
    # four terms plus three single-variable counterfactuals so the reconcile can attribute it.
    # Nothing here is consumed downstream; see patch note for why these three counterfactuals.
    _cv_mark("[pshare-why] recipient-share stash")
    # 19fq: ROUTING_VTERMS DELETED, same reason. _LAST_VAMP_TERMS / _LAST_VAMP_PSUM feed the
    # Search-vs-Delivery Reconciliation Breakdown, the [nw-attrib] table and the move-gate ladder.
    if True:
        try:
            _vt_vc = pd.to_numeric(pp["vampCount"], errors="coerce").fillna(0.0)
            _vt_mv = pd.to_numeric(pp["_move"], errors="coerce").fillna(0.0)
            _vt_pl = pd.to_numeric(pp["_moved_vpool"], errors="coerce").fillna(0.0)
            _vt_ps = pd.to_numeric(pp["_pshare"], errors="coerce").fillna(0.0)
            _vt_ml = pp["vampMid"].astype(str).str.strip().str.lower()
            _vt_pr = pd.to_numeric(pp["period"], errors="coerce").fillna(-1).astype(int)
            _vt_sum = pd.to_numeric(_psum, errors="coerce").fillna(0.0)

            # ── 19fq: THE FOUR TERMS ALWAYS; A COUNTERFACTUAL ONLY WHEN IT CAN DIFFER ──────
            # held / out / inn / pool are elementwise products of columns that already exist —
            # they cost nothing and they are what the Breakdown table prints every run.
            # The three COUNTERFACTUALS are the expensive half (a groupby-transform and a full
            # merge over the whole aged frame) and each answers "is THIS mechanism the cause of
            # the drift?". Each one has an exact precondition for being able to differ from
            # `post` at all. When the precondition is false the counterfactual EQUALS post by
            # construction — so short-circuiting it is not an approximation, it is the answer.
            # `_vt_skipped` records which were short-circuited, so a reader is never left to
            # assume a number was computed when it was inferred.
            _vt_skipped = []
            # 19fr: EACH COUNTERFACTUAL TIMED SEPARATELY. [cvp-timing]'s [vterms] mark covers the
            # four terms, all three counterfactuals and the txn merge in one figure, which cannot
            # answer "is cf_ps 2 seconds or 60?" — the question that decides whether its merge is
            # worth restructuring. These three marks are appended to the same _LAST_CVP_TIMING
            # list, so they appear in the [cvp-timing] table alongside everything else.
            _cv_mark("[vterms] the 4 terms (held/out/inn/pool)")
            _vt_post = _vt_vc * (1.0 - _vt_mv) + _vt_pl * _vt_ps

            # (2) cf_norenorm — undo ONLY the renormalise-to-1. The shipped line divided by
            #     _psum, so multiplying it back recovers the pre-renorm share exactly.
            #     CAN ONLY DIFFER if some cell's _psum is not exactly 1: if every live cell
            #     already sums to 1, multiplying by it is the identity.
            _vt_nr_live = _vt_sum[_vt_sum > 1e-12]
            if len(_vt_nr_live) and bool((_vt_nr_live.sub(1.0).abs() > 1e-9).any()):
                _vt_ps_raw = _vt_ps * _vt_sum
                _vt_cf_nr = _vt_vc * (1.0 - _vt_mv) + _vt_pl * _vt_ps_raw
            else:
                _vt_cf_nr = _vt_post
                _vt_skipped.append("cf_norenorm (every live cell's prop sums to 1, so undoing "
                                   "the renormalise is the identity)")
            _cv_mark("[vterms] counterfactual cf_norenorm (elementwise; skippable)")

            # (3) cf_nopass — undo ONLY the "no recipient -> passthrough" override, so `move`
            #     reverts to gf x pr_app and the pool is rebuilt from that.
            #     CAN ONLY DIFFER if that override actually fired, i.e. some cell had _psum == 0.
            #     This is the expensive one: a groupby-transform over the whole aged frame.
            if bool((_vt_sum <= 1e-12).any()):
                _vt_mv_raw = np.where(pd.to_numeric(pp["orig_m"], errors="coerce").fillna(-1) >= 0,
                                      pd.to_numeric(pp["_gf"], errors="coerce").fillna(0.0)
                                      * pd.to_numeric(pp["_pr_app"], errors="coerce").fillna(0.0), 0.0)
                _vt_pool_raw = (pp.assign(_mvr=_vt_vc * _vt_mv_raw)
                                .groupby(_gk)["_mvr"].transform("sum"))
                _vt_cf_np = _vt_vc * (1.0 - _vt_mv_raw) + _vt_pool_raw * _vt_ps
            else:
                _vt_cf_np = _vt_post
                _vt_skipped.append("cf_nopass (the no-recipient passthrough never fired, so "
                                   "undoing it is the identity)")
            _cv_mark("[vterms] counterfactual cf_nopass (1 groupby-transform; skippable)")

            # (1) cf_ps — rebuild vshare from `prop_share` (which carries the 0.97 max-share cap
            #     AND the 0.01 exploration floor) instead of raw `prop_raw` (which carries
            #     neither), then push it through the SAME merge / fillna / renormalise pipeline so
            #     the numerator object is the only thing that differs. Own copies throughout —
            #     `t0` and `pp` are not mutated.
            _vt_cf_psh = None
            try:
                _t0v = t0[grp + ["vampMid", "prop_share", "vampCount"]].copy()
                _t0v["_vp2"] = (pd.to_numeric(_t0v["prop_share"], errors="coerce").fillna(0.0)
                                * (pd.to_numeric(_t0v["vampCount"], errors="coerce").fillna(0.0)
                                   > 0).astype(float))
                _t0v["_vs2"] = np.where(_t0v.groupby(grp)["_vp2"].transform("sum") > 0,
                                        _t0v["_vp2"] / _t0v.groupby(grp)["_vp2"]
                                        .transform("sum").replace(0.0, 1.0), 0.0)
                _mv2 = (_t0v[_sub + ["vampMid", "period", "_vs2"]]
                        .rename(columns={"period": "orig_m"}))
                _pp2 = pp[_sub + ["vampMid", "orig_m", "period", "t"]].copy()
                _pp2 = _pp2.merge(_mv2, on=_sub + ["vampMid", "orig_m"], how="left")
                _ps2 = pd.to_numeric(_pp2["_vs2"], errors="coerce").fillna(0.0)
                _sm2 = _ps2.groupby([_pp2[c] for c in _gk]).transform("sum")
                _ps2 = np.where(_sm2 > 1e-12, _ps2 / np.where(_sm2 > 1e-12, _sm2, 1.0), 0.0)
                if len(_ps2) == len(_vt_vc):
                    _vt_cf_psh = _vt_vc * (1.0 - _vt_mv) + _vt_pl * _ps2
            except Exception:  # noqa: BLE001
                _vt_cf_psh = None
            # THE ONE THAT NEEDS A MERGE, and the only one with no cheap precondition — the 0.97
            # cap makes prop_share differ from the raw vshare on essentially every run, so it
            # cannot be short-circuited. If this mark is large, restructuring the merge is the
            # next move; if it is small, there is nothing here.
            _cv_mark("[vterms] counterfactual cf_ps (FULL MERGE + 2 groupbys; NOT skippable)")

            _vt_df = pd.DataFrame({
                "midl": _vt_ml, "per": _vt_pr,
                "pre": _vt_vc,
                "held": _vt_vc * (1.0 - _vt_mv),
                "out": _vt_vc * _vt_mv,
                "inn": _vt_pl * _vt_ps,
                "pool": _vt_pl,
                "post": _vt_vc * (1.0 - _vt_mv) + _vt_pl * _vt_ps,
                "cf_norenorm": _vt_cf_nr,
                "cf_nopass": _vt_cf_np,
                "cf_ps": (_vt_cf_psh if _vt_cf_psh is not None
                          else _vt_vc * (1.0 - _vt_mv) + _vt_pl * _vt_ps),
            })
            _vt_df["cfpsok"] = 1.0 if _vt_cf_psh is not None else 0.0
            globals()["_LAST_VAMP_CF_SKIPPED"] = list(_vt_skipped)
            globals()["_LAST_VAMP_TERMS"] = _vt_df.groupby(["midl", "per"], as_index=False).sum()
            # Is the renormalisation even doing anything? If _psum is 1.0 everywhere then
            # counterfactual (2) is a no-op and dead before it is read.
            _vt_g = pd.DataFrame({"s": _vt_sum}).assign(
                **{c: pp[c].astype(str) for c in _gk}).groupby(_gk, as_index=False)["s"].first()
            _vt_live = _vt_g[_vt_g["s"] > 1e-12]["s"]
            globals()["_LAST_VAMP_PSUM"] = {
                "groups": int(len(_vt_g)),
                "passthrough": int((_vt_g["s"] <= 1e-12).sum()),
                "off_one": int((_vt_live.sub(1.0).abs() > 1e-9).sum()),
                "sum_abs_dev": float(_vt_live.sub(1.0).abs().sum()),
                "min": float(_vt_live.min()) if len(_vt_live) else float("nan"),
                "p50": float(_vt_live.median()) if len(_vt_live) else float("nan"),
                "max": float(_vt_live.max()) if len(_vt_live) else float("nan"),
            }
        except Exception:  # noqa: BLE001 — a diagnostic must never break the projection
            globals()["_LAST_VAMP_TERMS"] = None
            globals()["_LAST_VAMP_PSUM"] = None
            globals()["_LAST_VAMP_CF_SKIPPED"] = None
    else:
        globals()["_LAST_VAMP_TERMS"] = None
        globals()["_LAST_VAMP_PSUM"] = None

    _tp = t0[_sub + ["vampMid", "period", "post_txn"]]
    pp = pp.merge(_tp, on=_sub + ["vampMid", "period"], how="left")
    pp["VI_Txn_Pre"] = np.where(pp["t"] == 0, pp["VI_Txn_Count"], 0.0)
    pp["VI_Txn_Post"] = np.where(pp["t"] == 0, pp["post_txn"].fillna(0.0), 0.0)
    _cv_mark("[vterms] VAMP-terms stash (4 terms + up to 3 counterfactuals) + the txn merge")
    _dump_projection_diag(t0, pp_path, prop_items, _enforced, _by_rpgt)   # heavy diagnostics
    # SELF-CHECK: VAMP (and VI-Txn) must conserve per period — Σ Post == Σ Pre. Warn (non-fatally) if
    # either drifts, so any future regression of the redistribution / passthrough logic is caught.
    try:
        _chk = pp.groupby("period")[["VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]].sum()
        for _pre_c, _post_c, _lbl in (("VAMP_Pre", "VAMP_Post", "VAMP"),
                                      ("VI_Txn_Pre", "VI_Txn_Post", "VI-Txn")):
            _rel = ((_chk[_post_c] - _chk[_pre_c]).abs()
                    / _chk[_pre_c].abs().clip(lower=1.0)).max()
            if float(_rel) > 1e-6:
                import warnings as _w
                _w.warn(f"compute_vamp_prepost_granular: {_lbl} not conserved per period "
                        f"(max rel drift {float(_rel):.2e}) — redistribution/passthrough regression?",
                        stacklevel=2)
    except Exception:  # noqa: BLE001
        pass
    _cv_mark("conservation check")
    # Collapse the pmp / Country sub-cells back to the reported grain (sums are exact).
    _cv_out = (pp.groupby(["vampMid", "RPGT", "BIN", "Currency", "period", "t"], as_index=False)
               [["VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]].sum())
    _cv_mark("final collapse back to the reported grain")
    globals()["_LAST_CVP_TIMING"] = list(_cv["rows"])
    return _cv_out


# [FN-261]
def brand_vamp_mids(mid_list_path=None, brand=None):
    """vampMids belonging to `brand`, ACTIVE OR NOT.

    Deliberately NOT `build_capability`'s set. That one answers "may this gateway RECEIVE routed
    volume?" and so excludes inactive gateways and PayPal. This one answers "does this vampMid
    belong to this company's book?", and an inactive gateway still does: Cardworks, EPX, Merrick
    and Bancard take no transactions any more but carry historic VAMP that is real, is in the
    baseline, and must keep its row.

    Uses the shared `_brand_key`, so "Total AV" and "TotalAV" resolve together — the spelling that
    has twice produced a filter matching nothing.

    An empty result means the brand matched nothing, which is a FAILURE, not an empty book. The
    caller must treat it as "filter unavailable" rather than "drop everything".
    """
    _bk = _brand_key(brand)
    if not _bk:
        return frozenset()
    _mm = load_mid_list(mid_list_path)
    _cc = _norm_cols(_mm)
    _vx, _bx = _cc.get("vampmid"), _cc.get("brand")
    if not _vx or not _bx:
        return frozenset()
    return frozenset(
        str(_v).strip() for _v, _b in zip(_mm[_vx].astype(str), _mm[_bx].astype(str))
        if _brand_key(_b) == _bk and str(_v).strip())


# [FN-261b]
def mid_table_from_granular(gran, keep_mids=None):
    """Per-vampMid VAMP / VI-Txn M0–5 (pre & post) table, derived by AGGREGATING the
    granular pre/post frame from compute_vamp_prepost_granular. This granular projection is the
    AUTHORITATIVE one — it adds go-live timing, zero-baseline back-fill and the exploration floor
    that the coarser compute_vamp_post_from_prorata does NOT, so the two are no longer identical.
    The Impact tab runs this ONE projection and reuses it for both the filterable detail AND this
    per-MID table."""
    if gran is None or getattr(gran, "empty", True):
        cols = ["vampMid"] + [f"{p} M{m}" for m in range(6)
                              for p in ("VAMP", "VI Txn", "VAMP Post", "VI Txn Post")]
        return pd.DataFrame(columns=cols)
    _vp = gran.groupby(["vampMid", "period"])["VAMP_Pre"].sum().unstack(fill_value=0.0)
    _vq = gran.groupby(["vampMid", "period"])["VAMP_Post"].sum().unstack(fill_value=0.0)
    _tp = gran.groupby(["vampMid", "period"])["VI_Txn_Pre"].sum().unstack(fill_value=0.0)
    _tq = gran.groupby(["vampMid", "period"])["VI_Txn_Post"].sum().unstack(fill_value=0.0)
    _mids = gran["vampMid"].unique()
    # 19dx — DROP OFF-BRAND ROWS THAT ARE ENTIRELY ZERO. The row universe is every vampMid in the
    # pro-rata export, which carries other brands (Stripe - VPN360, PaySafe - Total Cleaner,
    # WorldPay - Total Drive, …). 19dj stopped enforced_prop_items INVENTING such rows; these come
    # from the export itself, so they survived it.
    #
    # THE RULE IS DELIBERATELY TWO-PART: off-brand AND all-zero. A row with real numbers is never
    # hidden, whatever brand it claims — if an off-brand MID has actual VAMP in this scope, that is
    # a fact about the book and hiding it would understate it. So the worst this filter can do is
    # leave a phantom row in, never remove a real one.
    #
    # An EMPTY keep set means the brand matched nothing (the "Total AV" vs "TotalAV" failure), and
    # is treated as "filter unavailable" — every row is kept. Inverted, this would blank the table.
    if keep_mids:
        _keep = {str(_m).strip() for _m in keep_mids}
        _zero = {}
        for _m in _mids:
            _z = True
            for _fr in (_vp, _tp, _vq, _tq):
                if _m in _fr.index and float(np.abs(_fr.loc[_m].to_numpy(float)).sum()) > 1e-9:
                    _z = False
                    break
            _zero[_m] = _z
        _mids = [_m for _m in _mids if str(_m).strip() in _keep or not _zero[_m]]
    return _wide_by_mid(_mids, _vp, _tp, _vq, _tq)


# [FN-262]
def process_wallet_incapable(mid_list_path):
    """Set of gatewayFids (lowercased) that CANNOT process wallet (GOOGLEPAY /
    APPLEPAY), read from a processWallet-style column in Master_MID_List.

    ANALOGY: a guest list of which gateways can't accept Apple/Google Pay. We only strike a
    gateway off when the sheet EXPLICITLY says no (FALSE/0/NO); a blank is treated as "can",
    so we never wrongly ban a gateway on missing data.

    Robust to column-name variants (any column whose normalised name contains
    'wallet'). Only EXPLICIT false-like values (FALSE/F/0/NO/N) mark a gateway
    incapable; blanks/unknown default to capable (so we never over-restrict).
    """
    if not mid_list_path or not os.path.exists(mid_list_path):
        return set()
    try:
        _m = load_mid_list(mid_list_path)
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
# [FN-263]
def _cache_data(**kw):
    _dec = getattr(st, "cache_data", None) or getattr(st, "experimental_memo", None)
    return _dec(**kw) if _dec is not None else (lambda f: f)


# [FN-264]
def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# [FN-265]
@_cache_data(show_spinner=False)
def _c_read_parquet(path, m):
    # `m` is the file mtime — it MUST be a plain (non-underscore) name so st.cache_data
    # includes it in the cache key; a leading-underscore arg is excluded from the hash,
    # which would silently return a stale parquet after the file is regenerated.
    return pd.read_parquet(path)


# [FN-266]
@_cache_data(show_spinner=False)
def _c_vamp_post_prorata(pp_path, m, prop_items, excluded_mids, kill_eff=(), month_0=None,
                         scoped_rpgts=()):
    # `m` = file mtime; kept as a PLAIN (non-underscore) name so it actually participates
    # in the st.cache_data key (underscore-prefixed args are excluded from the hash), so a
    # regenerated pp_path busts this cache. Unused in the body — cache-key only.
    return compute_vamp_post_from_prorata(pp_path, prop_items, excluded_mids, kill_eff,
                                          month_0, scoped_rpgts)


# [FN-266b]
_PROJ_CODE_VER = "2026-08-26a-vamp-follows-volume+backfill-name"  # bump on ANY projection-logic
# change so the in-memory st.cache_data entries bust on the next rerun (the data signature alone
# can't see code edits: a re-used outputs folder + unchanged split => identical key => stale result).


def projection_cache_sig(pp_path, prop_items, exploration_floor=0.0, extra=""):
    """Stable cache-key SIGNATURE for the granular projection.

    The projection's `@st.cache_data` key used to lean on the pipeline file's mtime (`m`), which is
    frozen for a re-used outputs folder, plus Streamlit's implicit hashing of a ~51k-tuple `prop_items`
    list — unreliable enough that a changed SPLIT could hit a stale entry (tab-3 froze at one value
    across many engine runs). This returns a compact content hash of the ACTUAL projected split +
    exploration floor (+ any extra, e.g. blend flag), so the key busts whenever the deployed split or
    the floor changes, regardless of mtime. Cheap: one blake2b over the tuples (~tens of ms at 51k)."""
    import hashlib as _hl
    try:
        mt = os.path.getmtime(pp_path) if (pp_path and os.path.exists(pp_path)) else 0.0
    except OSError:
        mt = 0.0
    h = _hl.blake2b(digest_size=16)
    for t in (prop_items or ()):
        h.update(repr(t).encode("utf-8"))
    h.update(f"|floor={float(exploration_floor or 0.0):.8g}|{extra}|cv={_PROJ_CODE_VER}".encode("utf-8"))
    return f"{mt:.0f}:{len(prop_items or ()):d}:{h.hexdigest()}"


# [FN-267]
@_cache_data(show_spinner=False)
def _c_prepost_granular(pp_path, m, prop_items, excluded_mids, kill_eff=(), month_0=None,
                        scoped_rpgts=(), wallet_incapable=frozenset(), usa_only=frozenset(),
                        exploration_floor=0.0, vamp_off_mids=frozenset(),
                        cap_sig="", _capability=None, max_share=1.0):
    # `m` = cache-key SIGNATURE (callers now pass projection_cache_sig(): mtime + a content hash of
    # the actual split + floor). PLAIN (non-underscore) name so it participates in the st.cache_data
    # key (underscore args are excluded from the hash) — so the cache busts whenever the deployed
    # split OR the exploration floor changes, not just when the pipeline file's mtime changes (which
    # is frozen for a re-used outputs folder). Unused in the body — cache-key only.
    # 19cv: `vamp_off_mids` is a PLAIN (non-underscore) arg on purpose, so it participates in the
    # st.cache_data key — editing gateway_volume_overrides.json must bust this cache, and a
    # frozenset of strings hashes stably.
    # 19df: `max_share` is PLAIN for the same reason. It now CHANGES THE ANSWER (delivery applies
    # the cap to the vshare numerator), so a cached frame computed at a different cap must not be
    # served — changing max_gateway_share in the UI has to bust this cache, not survive it.
    return compute_vamp_prepost_granular(pp_path, prop_items, excluded_mids, kill_eff,
                                         month_0, scoped_rpgts, wallet_incapable, usa_only,
                                         exploration_floor=exploration_floor,
                                         vamp_off_mids=vamp_off_mids,
                                         # `_capability` is UNDERSCORED so st.cache_data leaves it
                                         # out of the hash (a function object is not hashable);
                                         # `cap_sig` above carries its identity into the key.
                                         capability=_capability,
                                         max_share=max_share)


# [FN-268]
def build_split_exports(split, brand, go_live, wallet_incapable=frozenset(), fid2vamp=None,
                        mid_list_path=None, usa_only=frozenset(), country_pres=None,
                        max_share=0.97, _stage=None, projection_mode=False):
    """Build the production template (one DataFrame per Brand×RPGT) from a split.

    ``_stage`` (diagnostic only; default None == final/shipped, byte-identical) stops the
    per-row enforcement pipeline early so a caller can measure how much each mechanism moves
    the delivered split. Order: "base" (normalised, no enforcement) → "zeroing" (+USA-only/
    wallet-incapable zero+renorm) → "backfill" (+<2-gateway back-fill) → "waterfill"
    (+max-share cap) → "final"/None (+2dp round & residual-push == shipped). For non-final
    stages the 2dp round/push is skipped and the raw stage shares (×100) are emitted, so the
    projection sees the exact intermediate split.

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
    # Stage gate (diagnostic). >=1 zeroing, >=3 water-fill, >=4 round+push.
    # THERE IS NO STAGE-2 MECHANISM ANY MORE. The <2-gateway back-fill it used to gate was
    # deleted from this function (see the note further down), so stage 2 is IDENTICAL to stage 1.
    # The "backfill" key is kept only so an old caller gets a DEFINED answer instead of silently
    # falling through the `.get(..., 4)` default to "final"; nothing in the app passes it.
    _slvl = {"base": 0, "zeroing": 1, "backfill": 2, "waterfill": 3,
             "final": 4}.get(str(_stage).lower(), 4) if _stage is not None else 4
    # Source of truth: read processWallet straight from Master_MID_List so the
    # export enforces it even if the routing run didn't populate the set.
    for _f in process_wallet_incapable(mid_list_path):
        wallet_incapable.add(_f)
        _vm = fid2vamp.get(_f)
        if _vm:
            wallet_incapable.add(_vm)
    # The fid → currency / fid → active lookups that used to be built here are GONE with the
    # <2-gateway back-fill that was their only consumer. They cost a Master_MID_List load plus a
    # per-row Python loop on EVERY projection call, so this is also the cheapest thing about the
    # deletion.
    df = split.copy()
    df["RPGT"] = df["rpgt"].astype(str)
    df["Currency"] = df["currency"].astype(str).str.upper()
    df["BIN"] = df["bin"].astype(str)
    df["gateway"] = df["gateway"].astype(str)
    df["share"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0)
    gateways = sorted(df["gateway"].unique().tolist())
    _pmps = ["GOOGLEPAY", "APPLEPAY", "non_gp_ap"]
    # SUB-CELL split: if the incoming split already carries pmp/Country per row (from a sub-cell
    # grain GA run), the rows ARE the sub-cells — build the template DIRECTLY from them instead of
    # expanding each cell into country×pmp. Cell-grain splits (no pmp/ctry, or all "_all_") take the
    # existing expansion path byte-for-byte. Map the split's pmp/ctry to the template's format.
    _has_subcell = ("pmp" in df.columns and "ctry" in df.columns
                    and not df["pmp"].astype(str).str.strip().str.lower().isin(["_all_", "", "nan"]).all())
    if _has_subcell:
        _pmpmap = {"googlepay": "GOOGLEPAY", "applepay": "APPLEPAY", "non_gp_ap": "non_gp_ap"}
        df["_PMP"] = (df["pmp"].astype(str).str.strip().str.lower().map(_pmpmap).fillna("non_gp_ap"))
        df["_CTRY"] = np.where(df["ctry"].astype(str).str.strip().str.lower().isin(["usa", "us"]),
                               "USA", "Non-USA")

    # FID-GRAIN capability (2026-08-17). `wallet_incapable` / `usa_only` hold BOTH
    # gatewayFids and their rolled-up vampMids. Template columns ARE fids, so `g in set`
    # is already exact; the extra `fid2vamp.get(g) in set` term rolled the vampMid's
    # capability onto every sibling fid and over-blocked the ones that CAN serve — e.g.
    # PaySafe - Total AV is wallet-capable on paysafe-usd-tav but not on paysafe-eur-tav /
    # -gbp-tav, and the roll-up zeroed the USD fid in wallet rows too. Removed. A column
    # that is a vampMid rather than a fid still matches, since the set holds both.
    # [FN-269]
    def _incap(gw):
        g = gw.strip().lower()
        return g in wallet_incapable

    # [FN-270]
    def _is_usa_only(gw):
        g = gw.strip().lower()
        return g in usa_only

    # [FN-272]
    def _cap_rows(V):
        """VECTORISED per-row cap + water-fill, applied to a whole (rows×gw)
        array at once. Each row (already normalised to sum 1) is capped at `_cap`, water-filling
        excess into the OTHER gateways already present. No-op for a row with <2 non-zero gateways
        or cap 1.0. Byte-identical to the scalar version — same 50-sweep water-fill, same order."""
        V = V.copy()
        m = (_cap < 1.0) & ((V > 1e-12).sum(1) >= 2)
        if m.any():
            W = V[m]
            for _ in range(50):
                over = W > _cap + 1e-12
                if not over.any():
                    break
                excess = np.where(over, W - _cap, 0.0).sum(1, keepdims=True)
                W = np.where(over, _cap, W)
                recip = (W > 1e-12) & (~over) & (W < _cap - 1e-12)
                room = np.where(recip, _cap - W, 0.0)
                rs = room.sum(1, keepdims=True)
                W = W + np.where(rs > 1e-12, room / np.where(rs > 1e-12, rs, 1.0) * excess, 0.0)
            V[m] = W
        return V

    # [FN-273]
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

    # VECTORISED per-row engine. Replaces the old per-(cell × country × pmp) Python loop, which
    # ran a groupby + Series ops + a 50-sweep cap PER ROW and scaled super-linearly with the cell
    # count (measured: tens of minutes at ~17k cells). This applies the SAME transforms — base
    # normalisation, Non-USA USA-only zeroing, wallet-incapable zeroing, max-share water-fill,
    # 2dp residual-push rounding and BIN-GROUP condition codes — across all rows with array ops.
    # It was proven byte-identical to the previous implementation on random splits, including
    # with the <2-gateway back-fill active; that back-fill has since been deleted outright, so
    # there is no longer any per-row Python path left in here at all.
    ng = len(gateways)
    incap_col = np.array([_incap(g) for g in gateways], dtype=bool)
    usa_col = np.array([_is_usa_only(g) for g in gateways], dtype=bool)
    _cols = (["GO LIVE", "BIN GROUP", "Brand", "RPGT", "Currency", "BIN",
              "paymentMethodProvider", "STICKY", "Country", "Check"] + gateways + ["DUP CHECK"])
    out = {}
    for rpgt, g_rpgt in df.groupby("RPGT"):
        if _has_subcell:
            # rows ARE the sub-cells: base per (Currency, BIN, pmp, Country), no expansion.
            base = (g_rpgt.groupby(["Currency", "BIN", "_PMP", "_CTRY", "gateway"])["share"].sum()
                    .unstack("gateway").reindex(columns=gateways).fillna(0.0))
            base = base.div(base.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
            _keys = list(base.index)
            if not _keys:
                out[(brand, rpgt)] = pd.DataFrame().reindex(columns=_cols)
                continue
            R = base.to_numpy(float)
            _cur = [k[0] for k in _keys]; _bin = [k[1] for k in _keys]
            _pmp = [k[2] for k in _keys]; _ctry = [k[3] for k in _keys]
        else:
            # per-cell normalised base (cells sorted by Currency,BIN — the same order the old
            # groupby(["Currency","BIN"]) iterated, so BIN-GROUP condition codes match).
            base = (g_rpgt.groupby(["Currency", "BIN", "gateway"])["share"].sum()
                    .unstack("gateway").reindex(columns=gateways).fillna(0.0))
            base = base.div(base.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
            cells = list(base.index)
            Bm = base.to_numpy(float)
            # expand rows in the SAME order as before: cell → country → pmp
            _idx, _cur, _bin, _ctry, _pmp = [], [], [], [], []
            for _ci, (cur, bin_) in enumerate(cells):
                for country in _countries_for(cur, bin_):
                    for pmp in _pmps:
                        _idx.append(_ci); _cur.append(cur); _bin.append(bin_)
                        _ctry.append(country); _pmp.append(pmp)
            if not _idx:
                out[(brand, rpgt)] = pd.DataFrame().reindex(columns=_cols)
                continue
            R = Bm[np.array(_idx)]
        ctry = np.array(_ctry); pmp = np.array(_pmp)
        # Non-USA rows: zero USA-only gateways, renorm   [stage >=1: zeroing]
        _nonusa = ctry == "Non-USA"
        if _slvl >= 1 and _nonusa.any() and usa_col.any():
            R[np.ix_(_nonusa, usa_col)] = 0.0
            _s = R[_nonusa].sum(1, keepdims=True)
            R[_nonusa] = np.where(_s > 0, R[_nonusa] / np.where(_s > 0, _s, 1.0), R[_nonusa])
        # Wallet rows (GOOGLEPAY/APPLEPAY): zero wallet-incapable gateways, renorm  [stage >=1]
        _wal = np.isin(pmp, ["GOOGLEPAY", "APPLEPAY"])
        if _slvl >= 1 and _wal.any() and wallet_incapable and incap_col.any():
            R[np.ix_(_wal, incap_col)] = 0.0
            _s = R[_wal].sum(1, keepdims=True)
            R[_wal] = np.where(_s > 0, R[_wal] / np.where(_s > 0, _s, 1.0), R[_wal] * 0.0)
        # ── <2-GATEWAY BACK-FILL: DELETED. NOT A SWITCH. ─────────────────────────────────
        # Removed from the delivered path 2026-08-17, and the ROUTING_LT2_BACKFILL escape hatch
        # that could turn it back on was deleted 2026-09-01. It is not configuration: it was a
        # defect, and leaving a variable that reinstates it only invites someone to reinstate it.
        #
        # WHAT IT DID. A row left with fewer than 2 live gateways got share INVENTED for
        # gateways the optimiser never assigned: an empty row was given 1/n across every valid
        # candidate, and a single-100% row gave each zero-share candidate a hard-coded 5% floor
        # (max(1 - have, 0.05)) before renormalising — with recipients drawn from the GLOBAL
        # template column set, not the cell's own doors.
        #
        # WHY THAT IS INDEFENSIBLE. It is share the GA cannot see at ANY grain, so it could never
        # be optimised against, never be predicted by the search, and never be attributed. It is
        # the mechanism that dumped volume onto Authorize / WoodForest.
        #
        # WHAT HAPPENS INSTEAD. A row left with <2 live gateways passes through UNTOUCHED and is
        # flagged by the `Check` column — the same treatment a genuinely single-gateway cell has
        # always had. The fix for such a row is data (open a second door in the MID list), not a
        # projection that pretends a door exists.
        #
        # Stage 2 ("backfill") therefore does nothing; see the stage-gate note above.
        if _slvl >= 3:                                   # [stage >=3: max-share water-fill]
            R = _cap_rows(R)
        # 2dp rounding + residual-push. Two fixes 2026-08-17:
        #  (1) the rounding was UN-GATED — it ran at EVERY _stage, so the breach-attribution
        #      "base" stage was never a true pre-rounding baseline. It is now inside the gate.
        #  (2) `projection_mode` (set by enforced_prop_items) skips BOTH. They exist to make an
        #      EXPORTED config file valid (2dp, sums to exactly 100.00) and have no place in a
        #      value projection: the residual is pushed onto the largest UNDER-cap gateway,
        #      which systematically moves mass from thin doors to fat incumbents and surfaces
        #      as scored-vs-delivered drift the GA cannot model. Exported templates are
        #      UNAFFECTED (projection_mode defaults to False).
        # 19ef: DEFAULT OFF. The 2dp round + residual-push made the exported sheet a different
        # split from the one tab 3 projected, for no gain: the ConnectorPool generator's own
        # `normalize_weights` rounds to integer tenths (0.1%) and re-balances to exactly 1000
        # from whatever share it is handed, so a 0.01% pre-round is finer than the thing that
        # consumes it and is discarded immediately. Rounding twice only loses information, and
        # the push "systematically moves mass from thin doors to fat incumbents" — which is why
        # `projection_mode` has skipped both since 2026-08-17. The export now takes that same
        # path. `Check` below is still rounded: it is the column that must READ as 100.00.
        # ROUTING_EXPORT_ROUND=1 restores the pre-19ef sheet exactly.
        _do_round = ((_slvl >= 4) and not projection_mode
                     and os.environ.get("ROUTING_EXPORT_ROUND", "0") == "1")
        RND = np.round(R * 100.0, 2) if _do_round else (R * 100.0)
        if _do_round:
            _rsum = np.round(RND.sum(1), 2)
            _cappct = round(_cap * 100.0, 2)
            for r in np.where((_rsum > 1e-9) & (np.abs(_rsum - 100.0) > 1e-9))[0]:
                _row = RND[r]
                _cand = [j for j in range(ng) if _row[j] > 0 and _row[j] < _cappct - 1e-9]
                _jmax = max(_cand, key=lambda j: _row[j]) if _cand else max(range(ng), key=lambda j: _row[j])
                RND[r, _jmax] = round(RND[r, _jmax] + (100.0 - _rsum[r]), 2)
        rdf = pd.DataFrame({"GO LIVE": go_live, "Brand": brand, "RPGT": rpgt,
                            "Currency": _cur, "BIN": _bin, "paymentMethodProvider": _pmp,
                            "STICKY": "Both", "Country": _ctry})
        for j, gw in enumerate(gateways):
            rdf[gw] = RND[:, j]
        rdf["Check"] = np.round(RND.sum(1), 2)
        if projection_mode:
            # BIN GROUP condition codes are an EXPORT artefact, and near-unique on unrounded
            # shares — building them would be pure cost. enforced_prop_items drops the column.
            rdf["BIN GROUP"] = ""
        else:
            _key = rdf[gateways].round(2).astype(str).agg("|".join, axis=1)
            _codes = {k: f"condition_{i+1}" for i, k in enumerate(dict.fromkeys(_key))}
            rdf["BIN GROUP"] = _key.map(_codes)
        rdf["DUP CHECK"] = 1
        out[(brand, rpgt)] = rdf.reindex(columns=_cols)
    return out


# [FN-274]
def enforced_prop_items(split, brand, go_live, wallet_incapable=frozenset(), fid2vamp=None,
                        mid_list_path=None, usa_only=frozenset(), country_pres=None,
                        max_share=0.97, _stage=None, projection_mode=True):
    """Proposed shares AFTER the pipeline's enforcement — cap, wallet-incapable zeroing and
    USA/Non-USA split — taken straight from build_split_exports' output, at
    (Currency, BIN, RPGT, pmp, Country, vampMid) grain.

    Enforcement here only ever REMOVES or REDISTRIBUTES share the optimiser assigned. It never
    invents share for a gateway the optimiser left at zero: the <2-gateway back-fill that used to
    do exactly that is deleted (see the note in build_split_exports).

    ANALOGY: what the split looks like once it's passed through production's "rulebook" — the
    same caps and capability filters the deployed config would apply — so the impact projection
    scores what will REALLY be routed, not the raw optimiser output.

    Feeding these into the projection reproduces the pipeline's back-fill gateways
    (WoodForest/Authorize) that the raw optimiser split never assigned. Returns a tuple of
    7-tuples (hashable for caching)."""
    fid2vamp = dict(fid2vamp or {})
    # Ensure a gatewayFid -> vampMid map (build from Master_MID_List if the caller didn't pass
    # one) — otherwise the gateway columns stay as raw FIDs and never match the export's vampMid,
    # so the projection sees no proposed shares and shows post == pre.
    # 19dj — BRAND-FILTER THIS MAP. It was built from EVERY row of the Master MID List: 539
    # gatewayFids across 27 brands, of which only 145 are Total AV. The other 394 (Total Adblock
    # 103, Total VPN 46, Total Password 36, Total Webshield 34, Total Drive 33, Cleaner 26, PC
    # Protect 26, Hotspot Shield 13, BetterNet 11, VPN360 3, …) entered the map, became prop
    # items, were back-filled as zero-baseline t0 rows by `_inject_backfill_rows`, and then
    # appeared as rows in tab 3's per-MID table — which builds its row list from
    # `gran["vampMid"].unique()` (mid_table_from_granular:1929). `brand` was already passed to
    # build_split_exports on the next line; only this map was blind to it.
    #
    # NOT a mis-attribution: no gatewayFid in the list maps to more than one vampMid (measured,
    # 0 of 539), and every injected row read 0 across VAMP / VI-Txn / both Post columns for
    # M0-M5. So this removes PHANTOM ROWS, not numbers — but it is still a change to what
    # enforced_prop_items returns. ROUTING_FID2VAMP_BRAND=0 reverts.
    #
    # THE SPELLING TRAP, which has already cost a day on this codebase: the MID list spells the
    # brand "Total AV" and the run's company is "TotalAV" (tab_3_split_outputs_impact:4178). A plain
    # strip().lower() matches NOTHING and would silently drop every gateway, exactly as the
    # 2026-08-28 20:44 run's `build_capability` did — injection reported success while doing
    # nothing. Compare on whitespace-stripped keys, and RAISE rather than return an empty map.
    _bkey = _brand_key(brand)
    _brand_on = os.environ.get("ROUTING_FID2VAMP_BRAND", "1") != "0" and bool(_bkey)
    if mid_list_path and os.path.exists(mid_list_path):
        try:
            _mm = load_mid_list(mid_list_path)
            _cc = _norm_cols(_mm)
            _gx, _vx = _cc.get("gatewayfid"), _cc.get("vampmid")
            _bx = _cc.get("brand")
            if _gx and _vx:
                _ax = _cc.get("isactive")
                _bs = (_mm[_bx].astype(str) if _bx
                       else pd.Series([""] * len(_mm), index=_mm.index))
                _as = (_mm[_ax].astype(str) if _ax
                       else pd.Series(["true"] * len(_mm), index=_mm.index))
                _kept = _drop = 0
                for _f0, _v, _b, _a in zip(_mm[_gx].astype(str),
                                           _mm[_vx].astype(str).str.strip(), _bs, _as):
                    if _brand_on:
                        # 19dk: PayPal, inactive and other-brand rows are all rejected here, by
                        # the SAME predicate build_capability uses. An inactive gateway keeps
                        # every row it already has in the export — this only stops it being
                        # injected as a recipient it can never receive into.
                        _f = _mid_row_fid(_f0, _b, _a, _bkey)
                        if _f is None:
                            _drop += 1
                            continue
                    else:
                        _f = str(_f0 or "").strip().lower()
                        if not _f or _f in ("", "nan", "none"):
                            continue
                    _kept += 1
                    fid2vamp.setdefault(_f, _v)
                if _brand_on and not _kept:
                    _bl = sorted({str(_x).strip() for _x in _mm[_bx].astype(str)
                                  if str(_x).strip()})[:8]
                    raise ValueError(
                        f"enforced_prop_items: filtering gatewayFid->vampMid on brand {brand!r} "
                        f"(+ active, non-PayPal) kept NO gateway out of {len(_mm):,} row(s) in "
                        f"{mid_list_path}. Brands "
                        f"present: {_bl}. An empty map makes every proposed share unmatchable and "
                        "the projection reports post == pre — refusing to return it silently.")
                globals()["_LAST_FID2VAMP_BRAND"] = {
                    "brand": str(brand), "kept": int(_kept), "dropped_other_brand": int(_drop),
                    "enabled": bool(_brand_on and _bx)}
        except ValueError:
            raise
        except Exception:  # noqa: BLE001
            pass
    templates = build_split_exports(
        split, brand, go_live, wallet_incapable=wallet_incapable, fid2vamp=fid2vamp,
        mid_list_path=mid_list_path, usa_only=usa_only, country_pres=country_pres,
        max_share=max_share, _stage=_stage, projection_mode=projection_mode)
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
    # Key normalisation moved ABOVE the positive filter (2026-08-18p) so an all-zero sub-cell can
    # still be identified at sub-cell grain before it is discarded. Order of operations only —
    # the normalisation itself is unchanged.
    allm["Currency"] = allm["Currency"].astype(str).str.strip().str.lower()
    allm["BIN"] = allm["BIN"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    allm["_pmp"] = (allm["paymentMethodProvider"].astype(str).str.strip().str.lower()
                    if "paymentMethodProvider" in allm.columns else "_all_")
    allm["_ctry"] = (allm["Country"].astype(str).str.strip().str.lower()
                     if "Country" in allm.columns else "_all_")
    allm["vampMid"] = (allm["_gw"].astype(str).str.strip().str.lower().map(fid2vamp)
                       .fillna(allm["_gw"].astype(str).str.strip()))
    # Canonicalise vampMid casing back to Master_MID_List (fixes lowercased back-fill labels, e.g. woodforest/adyen-na/paysafe).
    allm["vampMid"] = allm["vampMid"].map({str(v).strip().lower(): v for v in fid2vamp.values()}).fillna(allm["vampMid"])
    _subk = ["Currency", "BIN", "RPGT", "_pmp", "_ctry"]
    _pos = allm[allm["prop_raw"] > 0].copy()
    # ── CASE A: ZERO-CELL PLACEHOLDERS ────────────────────────────────────────────────────
    # A sub-cell whose gateways ALL land on zero used to be dropped outright here, so it never
    # reached prop_items — and `blend_prop_items` only loops over the cells prop_items contains.
    # `blend_cell_shares` therefore never ran for exactly the cells its own docstring is about
    # ("No specific share in the cell → undefined profile → fall back to the catch-all alone"),
    # which is why the tab-3 parity line reports "0 new key(s)" while the in-search twin injects
    # the catch-all into 145 such cells. That asymmetry is 20 of the 27 remaining reconciliation
    # units, confirmed per-MID by the [blend-cells] counterfactual.
    # Keeping ONE zero-prop row per such sub-cell is enough for the blend to see the cell:
    # blend_cell_shares filters `> 0`, gets an empty `spec`, and takes the catch-all branch. With
    # no catch-all configured it returns dict(spec) == {} and the cell emits nothing, i.e. exactly
    # today's behaviour; and with no blend at all a prop_raw of 0.0 adds nothing to any per-sub-cell
    # sum and moves no volume. Kill-switch: ROUTING_CA_ZEROCELL=0.
    _ph_n = 0
    if os.environ.get("ROUTING_CA_ZEROCELL", "1") != "0":
        try:
            _allc = allm[_subk].drop_duplicates()
            _posc = _pos[_subk].drop_duplicates()
            _zc = _allc.merge(_posc.assign(_p=1), on=_subk, how="left")
            _zc = _zc[_zc["_p"].isna()][_subk]
            if len(_zc):
                _ph = (allm.merge(_zc, on=_subk, how="inner")
                       .drop_duplicates(_subk).copy())
                _ph["prop_raw"] = 0.0
                _ph_n = len(_ph)
                _pos = pd.concat([_pos, _ph], ignore_index=True)
        except Exception:  # noqa: BLE001 — never break the projection over a diagnostic aid
            _ph_n = 0
    if _ph_n:
        # SAFETY RAIL. The whole fix depends on `split` already being scope-restricted. Measured
        # offline on the Aug baseline: 176 uncovered sub-cells among the SCOPED RPGTs (0.2% of t0
        # volume) versus 16,592 among the UNSCOPED ones (67%), which are frozen by design
        # (hold_unselected_at_baseline). If unscoped RPGTs ever reach here the placeholder count
        # explodes and the catch-all would reroute two thirds of the book — so say so loudly
        # rather than let it pass as a routine number.
        try:
            _rb = (_ph.groupby("RPGT").size().sort_values(ascending=False)
                   if "RPGT" in _ph.columns else None)
            _msg = (f"[ca-zerocell] {_ph_n:,} zero-share sub-cell(s) kept as placeholders so the "
                    f"backup catch-all can fire in profiles with NO specific rule "
                    f"(was: dropped at `prop_raw > 0`, so the catch-all never reached them)")
            if _rb is not None:
                _msg += " · by RPGT: " + " · ".join(f"{_k} {int(_v):,}" for _k, _v in _rb.items())
            if _ph_n > 2000:
                _msg += ("   ⚠ FAR more than the ~176 measured on the scoped Aug baseline — this "
                         "looks like UNSCOPED (baseline-frozen) RPGTs leaking into the split. "
                         "Those must NOT receive catch-all traffic; set ROUTING_CA_ZEROCELL=0 and "
                         "check the RPGT scope before trusting any delivered number from this run.")
            print("   " + _msg)
        except Exception:  # noqa: BLE001
            pass
    # STASH for the caller to LOG. print() lands in the terminal, not in the run log — and the run
    # log is the artefact that actually gets read, so a guard that only prints is not a guard.
    # tab_2_routing_engine re-emits this through log() as [ca-zerocell], including the unscoped-RPGT check.
    try:
        globals()["_LAST_CA_ZEROCELL"] = {
            "n": int(_ph_n),
            "by_rpgt": ({str(_k): int(_v) for _k, _v in
                         _ph.groupby("RPGT").size().items()}
                        if _ph_n and "RPGT" in getattr(_ph, "columns", []) else {}),
        }
    except Exception:  # noqa: BLE001
        globals()["_LAST_CA_ZEROCELL"] = {"n": int(_ph_n), "by_rpgt": {}}
    allm = _pos
    agg = allm.groupby(["Currency", "BIN", "RPGT", "_pmp", "_ctry", "vampMid"],
                       as_index=False)["prop_raw"].sum()
    return tuple(agg.itertuples(index=False, name=None))


# [FN-275]
def enforced_split_frame(split, brand, go_live, wallet_incapable=frozenset(), fid2vamp=None,
                         mid_list_path=None, usa_only=frozenset(), country_pres=None,
                         max_share=0.97):
    """Gateway-grain version of :func:`enforced_prop_items`.

    Returns the proposed split AFTER the pipeline's enforcement (cap, wallet-incapable
    zeroing, USA/Non-USA split) as a ``[rpgt, currency, bank, gateway, share]`` DataFrame — the
    SAME enforcement the VAMP projection uses, but keeping the gatewayFid so the revenue /
    success-rate views reflect the ACTUAL routed gateways.

    Every gateway in the result carries share the OPTIMISER gave it. It used to be able to
    carry gateways the optimiser never assigned at all — WoodForest / Authorize, via the
    <2-gateway back-fill — which is deleted.

    ``bank`` holds the BIN from the export (collapsed to a parent bank downstream via
    bin_to_bank, exactly like the raw split). pmp / Country variants are pooled by MEAN share
    per BIN cell (each variant already sums to 1, so the pooled shares sum to ~1). Share is
    re-normalised per (rpgt, currency, bank) cell. Empty frame if the split yields no rows.
    """
    cols = ["rpgt", "currency", "bin", "gateway", "share"]
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
    allm["bin"] = allm["BIN"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    allm["gateway"] = allm["gateway"].astype(str).str.strip()
    # Normalise within each export sub-cell (rpgt, currency, BIN, pmp, Country) → share sums to 1.
    _sub = ["rpgt", "currency", "bin"] + [c for c in ["paymentMethodProvider", "Country"] if c in allm.columns]
    _tot = allm.groupby(_sub)["w"].transform("sum")
    allm["_s"] = (allm["w"] / _tot).where(_tot > 0, 0.0)
    # Pool pmp / Country to the BIN grain by MEAN share, then re-normalise per (rpgt, currency, bank).
    out = (allm.groupby(["rpgt", "currency", "bin", "gateway"], as_index=False)["_s"].mean()
           .rename(columns={"_s": "share"}))
    _renorm_share(out, ["rpgt", "currency", "bin"])
    return out[cols]


# [FN-276]
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
    from routing_optimiser.s5_deliver.connector_pool_configs import generate_configs
    _exp = build_split_exports(
        split_long, brand_name, str(go_live),
        wallet_incapable=wallet_incapable, fid2vamp=fid2vamp, mid_list_path=mid_list_path,
        usa_only=usa_only, country_pres=country_pres, max_share=max_share)
    _pools, _counts = generate_configs(
        _exp, brand_key, date_tag, scheme=scheme, mode=mode,
        extra_priority_amount=int(extra_priority_amount), emit_generic=bool(emit_generic),
        count_only=True)   # search reads only len(_pools) → skip the make_pool payload build
    return len(_pools)


# [FN-277]
def pool_targeted_core(split_ideal, *, target_pools, wallet_ctx, brand_name, brand_key,
                       go_live, mid_list_path, date_tag="000000", mode="sales", scheme="vi",
                       emit_generic=False, method="kmeans", allocation="greedy", parallel=1):
    """Pure (NO session_state) pool-count-targeting compression for `split_ideal`.

    Returns (compressed_long, stats). Because it takes only picklable arguments and touches
    no Streamlit state, it is safe to run in a worker process (joblib/loky) — the dial
    positions are independent and deterministic, so parallelising them gives identical
    output. `pool_targeted_compression` wraps this with the ss cache.
    """
    from functools import partial as _partial
    from routing_optimiser.s5_deliver.kmeans_compress import compress_to_pool_budget
    wc = wallet_ctx or {}
    _si = split_ideal.copy()
    if "cell_volume" not in _si.columns:
        _si["cell_volume"] = (_si.groupby(["rpgt", "currency", "bin"])["volume"].transform("sum")
                              if "volume" in _si.columns else 1.0)

    # PICKLABLE count function (functools.partial of a MODULE-LEVEL fn) so the pool-budget
    # search can run the config-generation probes in loky/process workers — a plain closure
    # can't be pickled and would force joblib back to threading. Identical result to the old
    # closure (same args, just bound via partial; `_cl` fills the first positional at call).
    _count = _partial(
        count_pools_for_split, brand_name=brand_name, go_live=go_live,
        wallet_incapable=set(wc.get("incapable", set())), fid2vamp=wc.get("fid2vamp"),
        mid_list_path=mid_list_path, usa_only=set(wc.get("usa_only", set())),
        country_pres=wc.get("country_pres", {}), max_share=float(wc.get("max_share", 0.97)),
        brand_key=brand_key, date_tag=date_tag, scheme=scheme, mode=mode,
        emit_generic=emit_generic)

    return compress_to_pool_budget(_si, int(target_pools), _count,
                                   max_gateway_cap=float(wc.get("max_share", 0.97)),
                                   method=method, allocation=allocation, parallel=int(parallel))


# [FN-278]
def _pool_disk_key(split_ideal, *, target_pools, wallet_ctx, brand_name, brand_key,
                   go_live, mid_list_path, date_tag, mode, scheme, emit_generic,
                   method="kmeans", allocation="greedy"):
    """CONTENT hash of everything the compression output depends on: the split's own values
    (not object identity), all params, and the MID-list mtime. Because it hashes CONTENT, a
    changed split or setting yields a different key — so a disk hit can NEVER be stale."""
    import hashlib as _hl
    import json as _json
    wc = wallet_ctx or {}
    _cols = [c for c in ["rpgt", "currency", "bin", "gateway", "share", "volume",
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
        "cmeth": str(method), "calloc": str(allocation),
        "ms": round(float(wc.get("max_share", 0.97)), 6),
        "inc": sorted(str(x) for x in (wc.get("incapable") or set())),
        "uo": sorted(str(x) for x in (wc.get("usa_only") or set())),
        "f2v": _hl.sha256(_json.dumps({str(k): str(v) for k, v in
                          sorted((wc.get("fid2vamp") or {}).items())}).encode()).hexdigest()[:12],
        "cp": _hl.sha256(_json.dumps((wc.get("country_pres") or {}), sort_keys=True,
                          default=str).encode()).hexdigest()[:12],
    }
    return _hl.sha256((_split_hash + _json.dumps(_params, sort_keys=True)).encode()).hexdigest()[:24]


# [FN-279]
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
    # Opt-in compression clustering / budget-allocation (default kmeans/greedy = existing
    # behaviour). Read from ss and fold into the cache key so a change never returns a stale hit.
    _method = str(ss.get("compress_method", "ward"))
    _alloc = str(ss.get("compress_allocation", "knapsack"))
    sig = f"{sig}|cmp={_method}:{_alloc}"
    if sig in _cache:
        _e = _cache[sig]
        return _e["long"], _e["stats"]

    # DISK CACHE (content-hash keyed → survives ss clears / restarts / code tweaks, and can NEVER
    # go stale because the key hashes the split's values + all params). A re-run with an unchanged
    # split skips the whole (multi-pass k-means) search.
    _dpath = None
    try:
        # 19fk: renamed from ".cache" — see app_common.CACHE_DIR.
        _cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "data", "routing_engine_cached_input_data")
        _dk = _pool_disk_key(split_ideal, target_pools=target_pools, wallet_ctx=wallet_ctx,
                             brand_name=brand_name, brand_key=brand_key, go_live=go_live,
                             mid_list_path=mid_list_path, date_tag=date_tag, mode=mode,
                             scheme=scheme, emit_generic=emit_generic,
                             method=_method, allocation=_alloc)
        _dpath = os.path.join(_cdir, f"pool_comp_{_dk}.pkl")
        if os.path.exists(_dpath):
            _obj = pd.read_pickle(_dpath)
            _cl, _st = _obj["long"], _obj["stats"]
            _cache[sig] = {"long": _cl, "stats": _st}
            ss["_pool_comp"] = _cache
            return _cl, _st
    except Exception:  # noqa: BLE001
        _dpath = None

    # Parallel k-ary budget search: probe several cell budgets per round across the cores so
    # the (expensive) config-generation counts overlap. Same result as the serial binary search
    # (verified budget ≤ target). Bounded to ≤8 workers; ROUTING_COMPRESS_PARALLEL=1 disables it.
    _par = int(os.environ.get("ROUTING_COMPRESS_PARALLEL", "0") or 0)
    if _par <= 0:
        _par = min(max(2, os.cpu_count() or 2), 8)
    _cl, _st = pool_targeted_core(
        split_ideal, target_pools=target_pools, wallet_ctx=wallet_ctx,
        brand_name=brand_name, brand_key=brand_key, go_live=go_live,
        mid_list_path=mid_list_path, date_tag=date_tag, mode=mode, scheme=scheme,
        emit_generic=emit_generic, method=_method, allocation=_alloc, parallel=_par)
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


# [FN-280]
def rpgt_avg_ticket(cell_agg):
    """RPGT-level average ticket from the 30D actuals (the window ending just before
    Month 0): Σ succ_amount / Σ successes per RPGT. Returns {rpgt_lower: ticket}.
    Kept as the FALLBACK for (rpgt, currency) combos with no per-currency actuals."""
    if cell_agg is None or getattr(cell_agg, "empty", True):
        return {}
    g = cell_agg.groupby("rpgt_join").agg(rev=("cell_rev", "sum"), succ=("cell_succ", "sum"))
    return {str(rp).strip().lower(): (float(r["rev"]) / float(r["succ"]) if float(r["succ"]) > 0 else 0.0)
            for rp, r in g.iterrows()}


def rpgt_currency_avg_ticket(cell_agg):
    """RPGT × Currency average ticket from the 30D actuals: Σ succ_amount / Σ successes
    per (rpgt, currency). Returns {(rpgt_lower, currency_lower): ticket}. Finer grain
    than ``rpgt_avg_ticket`` so a given RPGT no longer shares one blended ticket across
    currencies. Combos with no actuals are simply absent (caller falls back to the
    RPGT-level ticket)."""
    if cell_agg is None or getattr(cell_agg, "empty", True):
        return {}
    if "currency_join" not in getattr(cell_agg, "columns", []):
        return {}
    g = cell_agg.groupby(["rpgt_join", "currency_join"]).agg(
        rev=("cell_rev", "sum"), succ=("cell_succ", "sum"))
    return {(str(rp).strip().lower(), str(cur).strip().lower()):
            (float(r["rev"]) / float(r["succ"]) if float(r["succ"]) > 0 else 0.0)
            for (rp, cur), r in g.iterrows()}


# [FN-281]
def mid_revenue_month_table(granular, rpgt_ticket, months=range(6), rc_ticket=None):
    """Per-vampMid × month VI Txn + $Revenue (pre/post) from the pro-rata granular.

    $Revenue = Σ ticket × VI_Txn[vampMid, RPGT, currency, month]. The ticket is the
    RPGT × Currency average from the actuals when ``rc_ticket`` is supplied — with NO
    fallback: every (RPGT, currency) in the granular MUST have a 30D-actuals ticket, and
    a missing one raises (a data problem to surface, not mask). Without ``rc_ticket`` the
    RPGT-level ticket is used (legacy behaviour). VI Txn is the origination (t=0) volume,
    matching the VAMP table's 'VI Txn M{m}'. Returns a wide DataFrame: vampMid + per-month
    VI Txn / $Revenue / VI Txn Post / $Revenue Post."""
    g = granular.copy()
    g["period"] = pd.to_numeric(g["period"], errors="coerce").fillna(-1).astype(int)
    _rp = g["RPGT"].astype(str).str.strip().str.lower()
    if rc_ticket is not None:
        if "Currency" not in g.columns:
            raise ValueError("mid_revenue_month_table: rc_ticket supplied but the granular has "
                             "no 'Currency' column — cannot price at RPGT × Currency.")
        _cur = g["Currency"].astype(str).str.strip().str.lower()
        _keys = list(zip(_rp.tolist(), _cur.tolist()))
        # Fallback for (RPGT, currency) combos with no 30D-actuals ticket — e.g. HELD / unscoped
        # RPGTs (renewals) that carry no SCORED attempts, so they legitimately have no ticket.
        # Priceable combos keep their EXACT RPGT×Currency ticket; only the gaps fall back to that
        # currency's average ticket, then the RPGT-level ticket, then a global average, then 0.
        # (This used to raise; held RPGTs in the granular made that too strict — a routine data
        # shape, not a data error.)
        _by_cur = {}
        for (_rk, _ck), _tv in rc_ticket.items():
            _by_cur.setdefault(_ck, []).append(float(_tv))
        _cur_avg = {c: (sum(v) / len(v)) for c, v in _by_cur.items() if v}
        _glob_avg = (sum(float(t) for t in rc_ticket.values()) / len(rc_ticket)) if rc_ticket else 0.0

        def _tk_for(_k):
            if _k in rc_ticket:
                return float(rc_ticket[_k])
            if _k[1] in _cur_avg:
                return _cur_avg[_k[1]]
            _rt = rpgt_ticket.get(_k[0]) if isinstance(rpgt_ticket, dict) else None
            return float(_rt) if _rt else _glob_avg
        _tk = pd.Series([_tk_for(k) for k in _keys], index=g.index, dtype=float)
    else:
        _tk = _rp.map(lambda r: rpgt_ticket.get(r, 0.0)).astype(float)
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
