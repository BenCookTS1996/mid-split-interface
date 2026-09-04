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

# ── 19kg: SETTINGS THAT USED TO BE ENVIRONMENT SWITCHES ──────────────────
# No environment variable changes a run any more. Each name below is frozen at the
# value the shipped run already used - the defaults, because no routing.env exists and
# run.command exports nothing - so what shipped is what these say. They stay NAMES, not
# literals inlined at the use site, for two reasons: a test can still A/B a whole search
# by rebinding one, and a reader can see in one place every decision this module makes.
# Changing behaviour now means editing this block and saying so in a commit.
_SW_BLOCK_NOFILL = False   # was ROUTING_BLOCK_NOFILL, default '0'
_SW_CA_ZEROPROFILE = True   # was ROUTING_CA_ZEROPROFILE, default '1'
_SW_COARSE_PROP_FALLBACK = False   # was ROUTING_COARSE_PROP_FALLBACK, default '0'
_SW_COMPRESS_PARALLEL = 0   # was ROUTING_COMPRESS_PARALLEL, default '0'
_SW_DELIV_MAXSHARE = True   # was ROUTING_DELIV_MAXSHARE, default '1'
_SW_EXPORT_ROUND = False   # was ROUTING_EXPORT_ROUND, default '0'
_SW_FID2VAMP_BRAND = True   # was ROUTING_FID2VAMP_BRAND, default '1'

try:    # 19iq: THE BLOCKED-ROW WATER-FILL RULE. See routing_optimiser/s4_search/blocked_fill.py.
    from routing_optimiser.s4_search import blocked_fill as _BFM
    _BFM.register("_cap_rows")
    _BFM.register("_max_share_waterfill")
except Exception:  # noqa: BLE001 - no rule available is a refusal, never a broken import
    _BFM = None


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

# ── [forensic] 19gt: THE FOUR ATTRIBUTION STASHES ARE COMPUTED ON DEMAND ────────────────────
# `compute_vamp_prepost_granular` builds four read-only stashes that exist for ONE purpose:
# explaining a non-zero reconciliation error. On the 2026-09-01 22:09 run they were 77s of a
# 162s projection — 47% — explaining an error of 0.
#
#   [passthru]    18.8s   which aged groups the no-recipient override fired on
#   [pshare-why]  18.7s   why Σ_pshare came out under 1
#   [vterms]      29.6s   the four VAMP terms + three single-variable counterfactuals
#   [move-gate]    9.8s   five one-gate-lifted movable-fraction variants
#
# 19fq deleted their env switches, and that reasoning stands: "a switch that can turn off the
# only explanation of a number you are driving to zero is a switch that will be found off on the
# run where it mattered". THIS IS NOT THAT SWITCH. The caller sets it False, projects, reads the
# per-band drift off that projection, and — if the drift is real — sets it True and projects
# AGAIN before anything downstream reads a stash. The explanation is never unavailable; it is
# computed exactly when there is something to explain, and the run log says which happened.
#
# A skipped stash is set to the string "skipped", NOT None. None already means "this failed",
# and a reader must be able to tell those apart.
FORENSIC = True


class _Skip(Exception):
    """Raised to leave a forensic stash block early when FORENSIC is off.

    A dedicated type, not a bare `return`: these blocks sit inside broad
    `except Exception` handlers that set the stash to None (= "this failed"), and a skip
    must not be recorded as a failure. Each handler re-checks the flag."""

__build__ = ("2026-08-17b-count-only-pool-search+profile-exporter+staged-enforcement"
             "+projection-mode-no-round+lt2-backfill-DELETED+no-coarse-prop-fallback+fid-grain-capability+txn-term-stash+denom-stash+t0-presence-backfill+ca-zeroprofile+vamp-term-stash+2026-09-01-19gq-gk-int-key+cvp-submarks+19gt-forensic-on-demand+2026-09-03-19ih-sentinel-unclobbered+2026-09-03-19je-cvp-submarks+2026-09-03-19jh-txnterms-on-demand+2026-09-03-19ji-backfill-cleancol")


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
    from the Split Go Live date. Transactions are conserved per profile; VAMPs move
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
    ct = t0.groupby(["Currency", "BIN", "period"], as_index=False).agg(profile_tot=("pre_txn", "sum"))
    t0 = t0.merge(ct, on=["Currency", "BIN", "period"]).merge(prop, on=["Currency", "BIN", "vampMid"], how="left")
    t0["f"] = t0["period"].map(frac)
    # vampMids switched off via gateway_volume_overrides are removed from BOTH the
    # pre-go-live retention and the proposed split; their volume redistributes to
    # the active gateways in the profile (transactions still conserved). The removal is
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
    t0["post_txn"] = t0["profile_tot"] * ((1 - t0["f"]) * t0["base_share"] + t0["f"] * t0["prop_eff"])
    t0["r"] = np.where(t0["pre_txn"] > 0, t0["post_txn"] / t0["pre_txn"], 1.0)

    # VAMP conserved & redistributed by the volume share (pipeline-faithful). This legacy
    # non-prorata fallback has no fcp data, so the movable slice is the go-live fraction only.
    t0["_move"] = np.where(t0["_prop_sum"] > 0, t0["f"], 0.0)
    # VAMP follows the volume: the moved VAMP pool is redistributed by the SAME post-volume
    # share as the moved transactions (prop_eff), so grown MIDs pick up VAMP at the profile's
    # blended rate and Σ VAMP_Post == Σ VAMP_Pre (the profile's VAMP total is conserved).
    t0["_vprop"] = t0["prop_eff"]
    t0["_vpsum"] = t0.groupby(["Currency", "BIN", "period"])["_vprop"].transform("sum")
    t0["_vshare"] = np.where(t0["_vpsum"] > 0, t0["_vprop"] / t0["_vpsum"], 0.0)
    tp["orig_m"] = tp["period"] - tp["t"]
    _mv = t0[["Currency", "BIN", "vampMid", "period", "_move", "_vshare"]].rename(
        columns={"period": "orig_m", "_vshare": "_pshare"})
    tp["_profile_vamp"] = tp.groupby(["Currency", "BIN", "period", "t"])["VAMP_Pre"].transform("sum")
    tp = tp.merge(_mv, on=["Currency", "BIN", "vampMid", "orig_m"], how="left")
    tp["_move"] = tp["_move"].fillna(0.0)
    tp["_pshare"] = tp["_pshare"].fillna(0.0)
    tp["VAMP_Post_c"] = tp["VAMP_Pre"] * (1.0 - tp["_move"]) + tp["_profile_vamp"] * tp["_move"] * tp["_pshare"]

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
    # fcp1_frac: fraction of the profile the pipeline actually reroutes (fcpNumber==1 /
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
    # pre-go-live retention and the proposed split; the profile total is unchanged so
    # their volume is redistributed to the active gateways (transactions conserved).
    # The removal is gated by each switch-off's effective_date (kill_eff): a
    # switched-off vampMid keeps its volume until its effective month, then drops
    # (mid-month pro-rated). vampMids in excluded_mids with no effective date are
    # removed for all periods (binary fallback).
    _apply_keep(t0, excluded_mids, kill_eff, month_0)
    t0["prop_raw"] = t0["prop_raw"] * t0["_keep"]
    t0["_active_vi"] = t0["VI_Txn_Count"] * t0["_keep"]
    # Group the profile keys ONCE and reuse for all three per-profile sums (the 5-col key is
    # otherwise re-factorised per groupby). Bit-identical to grouping separately.
    _g = t0.groupby(grp)
    t0["profile_tot"] = _g["VI_Txn_Count"].transform("sum")
    t0["_active_tot"] = _g["_active_vi"].transform("sum")
    t0["base_share"] = np.where(t0["_active_tot"] > 0, t0["_active_vi"] / t0["_active_tot"], 0.0)
    # Renormalise proposed shares over the gateways present in each profile so the
    # redistribution conserves the profile's transactions; if no proposed shares map
    # to this profile, fall back to the current (baseline) split.
    t0["prop_sum"] = _g["prop_raw"].transform("sum")
    t0["prop_share"] = np.where(t0["prop_sum"] > 0, t0["prop_raw"] / t0["prop_sum"], t0["base_share"])
    # Movable fraction = go-live pro-rata × fcp1 cohort fraction: only that slice of the
    # profile takes the proposed share; the rest (pre-go-live + FCP2+/retries) stays baseline.
    # PER-MID movable fraction (fcp1_frac is per-vampMid now): move_mid = pro_rata × fcp1_frac.
    t0["_move"] = np.where(t0["prop_sum"] > 0, t0["pro_rata"] * t0["fcp1_frac"], 0.0)
    # VAMP follows the volume: the moved VAMP pool is redistributed by the SAME post-volume
    # share as the moved transactions (prop_raw, renormalised below to prop_share), so grown
    # MIDs pick up VAMP at the profile's blended rate and the profile's VAMP total is conserved.
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
    t0["post_txn"] = t0["profile_tot"] * (t0["base_share"] * (1 - t0["_move"])
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
# 19kg: `_dump_projection_diag` DELETED (617 lines) with the twelve switches that
# only ever appeared inside it: ROUTING_PROJ_DIAG, _TRACE, _BASIS, _CMP_MID,
# _RULES_DIR, _RULESCMP, _TAB5CMP, _TAB5_EXPORT, _SAMPLE_MID, _SAMPLE_N,
# _SAMPLE_RPGT and _FINEGRAIN.
#
# It early-returned unless ROUTING_PROJ_DIAG=1, and when armed it wrote a ~170MB
# CSV next to the pro-rata export. One call site, return value unused. It was also
# the fifth most complex function in the codebase (cyclomatic 135). What it was
# built to investigate - the tab-3 vs tab-5 back-fill gap - is now answered every
# run by [rung], [profiles] and the RECONCILIATION ERROR line, on the split that
# actually ships rather than on a dump someone had to remember to switch on.


# [FN-259]
def _inject_backfill_rows(pp, prop, prop_name_map=None, mark=None):
    """#3 ZERO-BASELINE ROW INJECTION. Nothing to do with the deleted <2-gateway share
    back-fill — this adds ROWS, never share.

    The optimiser can route volume to a gateway that has NO baseline row in a profile (it has never
    served that profile, so the pro-rata export has nothing there). The LEFT merge drops it, and
    its routed volume then wrongly redistributes to the MIDs that DO have rows. Re-inject those
    recipients into `pp` as zero-baseline t=0 rows (vampCount=0, VI=0) so they RECEIVE the routed
    volume; VAMP stays 0 for them (no historical VAMP to redistribute).

    Scoped to the enforced (7-tuple) path, which is the only one carrying per-profile shares.
    """
    # Presence is judged at the pmp/Country PROFILE grain (Currency, BIN, RPGT, pmp, Country),
    # NOT the coarse profile — because the enforced table routes per profile, and a MID present in
    # ONE profile but routed volume in ANOTHER (e.g. WoodForest has baseline only in
    # non_gp_ap/non-usa but the template gives it 97% in non_gp_ap/usa) has no row there to
    # receive it, so its routed profile volume wrongly redistributes to the present MIDs.
    # GUARD: only inject into (pmp, Country) profiles that actually EXIST in the baseline for
    # the coarse profile — never invent a profile from a pmp/Country label the baseline lacks (a
    # pure label mismatch is handled by the hierarchical coarse fallback downstream, not here),
    # which is what previously twinned MIDs across mismatched profiles.
    subk = ["Currency", "BIN", "_rpgtl", "_pmp", "_ctry"]
    b = pp.copy()
    if mark is not None:
        mark("  backfill: copy the export frame")
    # 19ji: `_clean_col` instead of the per-row chain. Same function this file's own key-normalise
    # step uses (19fx): it cleans once per DISTINCT VALUE and returns an object ndarray that is
    # character-for-character identical, so there is no bit-identity question to weigh - only 21
    # vampMids and 8 RPGTs exist across 6.5M rows, and the per-row chain lower-cases each of them
    # about 300,000 times to learn one fact. It falls back to the slow chain on a column holding
    # nulls, which is a real trap and not caution: pd.factorize collapses None and NaN into one
    # missing value where the per-row chain renders them "none" and "nan".
    b["_rpgtl"] = _clean_col(b["RPGT"], lower=True)
    b["_vml"] = _clean_col(b["vampMid"], lower=True)
    if mark is not None:
        mark("  backfill: lower-case the RPGT and vampMid keys")
    # PRESENCE IS JUDGED ON THE t == 0 SLICE — the frame the caller's LEFT merge actually
    # consumes (`_t0 = pp[pp["t"] == 0]`). Judging it over ALL t (as this did until
    # 2026-08-18h) made a MID with an AGED row but no t0 row read as "present": back-fill
    # skipped it, then the t0 merge found nothing and its enforced share was dropped and
    # renormalised onto the survivors. Measured on the Aug baseline: 3,170 enforced items
    # (156.92 of 14,807 prop mass) vanished across 1,280 profiles, and in the worst cases
    # the ENTIRE profile's routing decision was discarded (ghost 1.0000, surviving 0.0000).
    # `valid_sub` moves with it so we still never invent a (pmp, Country) the t0 baseline
    # lacks — injecting into a t>0-only profile would create a t0 profile with profile_tot = 0
    # that contributes nothing but rows.
    _b0 = b[pd.to_numeric(b["t"], errors="coerce").fillna(0).astype(int) == 0]
    present = set(map(tuple, _b0[subk + ["_vml"]].drop_duplicates().to_numpy()))
    valid_sub = set(map(tuple, _b0[subk].drop_duplicates().to_numpy()))
    # Global _vml -> proper-case vampMid, so a MID that exists elsewhere in the export keeps its
    # display name and merges cleanly (no lower-case twin) on the final collapse.
    if mark is not None:
        mark("  backfill: the two presence sets (t0 uniques -> python tuples)")
    name_map = b.drop_duplicates("_vml").set_index("_vml")["vampMid"].to_dict()
    # Truly zero-baseline recipients have NO row anywhere in the export, so `b` can't supply a
    # proper-case name and they'd otherwise fall back to the lower-case merge key as their display
    # name. Fill those gaps from the proposed items' proper-case vampMid (sourced from the Master
    # MID list, captured by the caller before `vampMid` is dropped); baseline names keep priority
    # via setdefault so no present MID is renamed.
    if prop_name_map:
        for _k, _v in prop_name_map.items():
            name_map.setdefault(_k, _v)
    # Representative RPGT / go-live pro_rata / fcp1 per (profile, period), lowest-t row.
    reps = (b.sort_values("t").drop_duplicates(subk + ["period"])
            [subk + ["RPGT", "period", "pro_rata", "fcp1_frac"]])
    if mark is not None:
        mark("  backfill: the per-profile representative t0 rows")
    pc = prop[subk + ["_vml"]].drop_duplicates()
    # missing = enforced (profile, MID) with no baseline row in that profile, AND the profile
    # itself exists in the baseline (so we don't fabricate profiles from label mismatches).
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
# on the live export: 9.14 CAPABLE MIDs per profile against 2.03 PRESENT per layer.
#
# Completing the movable layers fixes that at the source. It also makes the two projectors agree
# WITHOUT the age renormalise: on the 19da fixture, injection alone takes
# Sigma|delivered - in-search| to 0.000000 with `_SW_AGE_RENORM = False`, and turning the renormalise
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

    A vampMid is CAPABLE in a profile when it has at least one gatewayFid that is
      * IsActive in the Master MID List, of this brand, and not PayPal,
      * of the profile's CURRENCY,
      * not hit by a routing_restrictions `rules` entry for that rpgt / currency / bin,
      * wallet-capable (processWallet) when the profile's pmp is googlepay / applepay,
      * not in `usa_only_gateways` unless the profile is USA.
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


def inject_capable_rows(pp, capable, profile_cols, mid_col="vampMid",
                        period_col="period", t_col="t", mark=None):
    """Complete every MOVABLE (t <= period) group of `pp` with a zero-VAMP row per absent capable
    MID. Returns (frame, n_added). VECTORISED — a per-group Python loop over the live export's
    622,592 movable groups is not viable.

    The injected row carries vampCount 0 and VI_Txn_Count 0, and takes `pro_rata` / `fcp1_frac`
    from its own layer (both are properties of the layer, not of the MID, so every row of a group
    already shares them — asserted below rather than assumed). A zero-VAMP row therefore adds
    NOTHING to the pool and changes no held term; its only effect is to give the layer a recipient
    the split can route to.
    """
    _need = list(profile_cols) + [mid_col, period_col, t_col]
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
    _gkc = list(profile_cols) + [period_col, t_col]
    _poolg = (pp.assign(_mvbl=_vc * _fc * _prr)
              .groupby(_gkc, observed=True)["_mvbl"].transform("sum"))
    _mov = pp[(_poolg > 0) & _per.notna() & _t.notna()]
    if not len(_mov):
        return pp, 0

    _gk = list(profile_cols) + [period_col, t_col]
    _layer = _mov.groupby(_gk, as_index=False, observed=True).agg(
        pro_rata=("pro_rata", "first"), fcp1_frac=("fcp1_frac", "first"))

    if mark is not None:
        mark("  capable: the movable-pool groupby-transform over the whole export")
    _profiles = _mov[list(profile_cols)].drop_duplicates()
    _profiles["_cap"] = [sorted(capable(*[str(v).strip().lower() if i != 1 else str(v).strip()
                                       for i, v in enumerate(row)]))
                      for row in _profiles.itertuples(index=False, name=None)]
    _profilecap = _profiles.explode("_cap").rename(columns={"_cap": mid_col})
    _profilecap = _profilecap[_profilecap[mid_col].notna()]

    _full = _layer.merge(_profilecap, on=list(profile_cols), how="inner")
    # The anti-join key MUST include the MID. Keyed on the group alone every candidate row finds
    # a match, `_seen` is never NaN and the function silently injects nothing — which looks exactly
    # like "the frame was already complete" and would have shipped as a no-op.
    _hk = _gk + [mid_col]
    _have = _mov[_hk].drop_duplicates().assign(_seen=1)
    _new = _full.merge(_have, on=_hk, how="left")
    if mark is not None:
        mark("  capable: the capability lookup per profile + the two anti-join merges")
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


def blocked_keys_for(blocked_pairs, mid_list_path):
    """(bin, gatewayFid) pairs -> the canonical (bin, vampMid, currency) keys, memoised.

    19iq stamps `_blocked` on the split, which serves the two sites that can express a
    gatewayFid. The VAMP forecast and the band projector cannot: their row identity is
    (vampMid, currency), and blocking a whole vampMid would block five currencies because one
    was flagged (31 of 37 active TotalAV fids sit under a multi-fid vampMid). (vampMid,
    currency) pins down a unique ACTIVE fid within a brand-scoped run, which is what makes the
    coarse sites able to reproduce the fine ones - see blocked_fill's canonical-key note and
    `equivalence_report`, which proves it per run rather than trusting that paragraph.

    Returns frozenset(); an empty set on any failure, because a mask must never break a
    projection - a site with no mask applies no rule and says so through `saw_mask`.
    """
    if not blocked_pairs or _BFM is None:
        return frozenset()
    _key = (tuple(sorted((str(a), str(b)) for a, b in blocked_pairs)), str(mid_list_path or ""))
    _hit = _BLK_KEY_MEMO.get(_key)
    if _hit is not None:
        return _hit
    try:
        _rows = load_mid_list(mid_list_path).to_dict("records")
        _out = frozenset(_BFM.canonical_keys(blocked_pairs, _rows))
    except Exception:  # noqa: BLE001 - no keys is a refusal, never a broken projection
        _out = frozenset()
    _BLK_KEY_MEMO[_key] = _out
    return _out


_BLK_KEY_MEMO = {}


def _max_share_waterfill(shares, t0, grp, cap, live, blocked=None):
    """Port of the search's per-profile max-share water-fill (`band_projection.py:317-347`).

    Line-for-line the same algorithm, vectorised: everything over `cap` is cut back to it, and
    the excess is handed to the under-cap rows of the SAME profile in proportion to the room each
    has left. Repeated up to 50 sweeps, because handing excess out can push a recipient over the
    cap in turn.

    THE THREE THINGS THAT MUST MATCH THE KERNEL, and each of which silently changes the answer:

      1. `_nzc` — the ">= 2 routed gateways" test — is computed ONCE, before the first sweep, and
         is NOT refreshed as rows are capped. A profile with a single routed gateway is left alone
         entirely (capping it would have nowhere to put the excess, so the kernel skips it).
      2. The excess is measured BEFORE the rows are cut to the cap, and the room is measured
         AFTER. Swapping either order changes the redistribution.
      3. Accumulation is in row order within a profile, which is what `np.bincount` does, so the
         floating-point summation order is the kernel's too.

    `live` is the kernel's `_psum[c] > 0` — rows in an unrouted profile take no part.

    `blocked` (19ir) is the per-row bank-blocked mask. Same rule, same arithmetic, as the two
    fine-grained sites: a blocked row is a recipient of LAST RESORT, and the rule is applied
    only when every site is wired AND armed. It is PRICED either way, because "the rule would
    move nothing here" is a measurement, not an assumption.
    """
    _sh = np.asarray(shares, dtype=float).copy()
    if _sh.size == 0:
        return _sh
    _g = t0.groupby(grp, sort=False, observed=True).ngroup().to_numpy()
    _ng = int(_g.max()) + 1
    _live = np.asarray(live, dtype=bool)
    EPS = 1e-12
    _blk = None
    if blocked is not None and _BFM is not None:
        _blk = np.asarray(blocked, bool)
        if _blk.shape != _sh.shape or not _blk.any():
            _blk = None
    _armed = False
    if _blk is not None:
        _armed, _amsg = _BFM.arming_verdict(
            _SW_BLOCK_NOFILL)
        _meas = {"on_blocked": 0.0, "unavoidable": 0.0, "avoidable": 0.0, "sweeps": 0,
                 "rows": int(_blk.sum()), "armed": bool(_armed), "msg": str(_amsg)}
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
        _add = np.where(_ok, (cap - _sh) / np.where(_ok, _rsum[_g], 1.0) * _exc[_g], 0.0)
        if _blk is not None:
            _roomv = np.where(_room, cap - _sh, 0.0)
            _onb = float(_add[_blk].sum())
            _nb = np.bincount(_g, weights=np.where(_blk, 0.0, _roomv), minlength=_ng)
            _need = float(np.minimum(np.maximum(_exc - _nb, 0.0),
                                     np.maximum(_exc, 0.0)).sum())
            _meas["on_blocked"] += _onb
            _meas["unavoidable"] += _need
            _meas["avoidable"] += max(0.0, _onb - _need)
            _meas["sweeps"] += 1
            if _armed:
                _add = _BFM.two_stage_add_grouped(_roomv, _blk, _exc, _g, _ng, _add)
        _sh = _sh + _add
    if _blk is not None:
        globals()["_LAST_BLK_FILL_VAMP"] = dict(_meas)
        _BFM.saw_mask("_max_share_waterfill", True, f"{int(_blk.sum()):,} row(s) blocked")
    return _sh


def compute_vamp_prepost_granular(pp_path, prop_items, excluded_mids=frozenset(),
                                  kill_eff=(), month_0=None, scoped_rpgts=(),
                                  wallet_incapable=frozenset(), usa_only=frozenset(),
                                  exploration_floor=0.0, vamp_off_mids=frozenset(),
                                  capability=None, max_share=1.0,
                                  wallet_incapable_pairs=frozenset(),
                                  usa_only_pairs=frozenset(),
                                  blocked_keys=frozenset()):
    """Per-ROW baseline vs proposed VAMP / VI-Txn from the pro-rata export.

    Routes at the (vampMid, RPGT, BIN, Currency, pmp, Country) profile grain when the
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
        _profilek = [c for c in ("Currency", "BIN", "RPGT", "paymentMethodProvider", "Country")
                  if c in pp.columns]
        if len(_profilek) >= 3:
            pp, _n_inj = inject_capable_rows(pp, capability, _profilek, mark=_cv_mark)
            if _n_inj:
                globals()["_LAST_INJECTED"] = int(_n_inj)
    _cv_mark("  capable: build and concat the injected rows")
    # 19fx: cleaned once per DISTINCT VALUE, not once per row -- see _clean_col. Identical output.
    pp["Currency"] = _clean_col(pp["Currency"], lower=True)
    pp["BIN"] = _clean_col(pp["BIN"])
    pp["vampMid"] = _clean_col(pp["vampMid"])
    rpgt_col = "RPGT" if "RPGT" in pp.columns else "rpgt"
    pp["RPGT"] = _clean_col(pp[rpgt_col], strip=False)
    pp["pro_rata"] = pd.to_numeric(pp.get("pro_rata", 0.0), errors="coerce").fillna(0.0)
    pp["fcp1_frac"] = pd.to_numeric(pp.get("fcp1_frac", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    # Keep pmp / Country profiles (default '_all_' when the export lacks them) so the
    # projection can apply the pipeline's per-profile wallet / USA-only enforcement.
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
        pp = _inject_backfill_rows(pp, prop, _prop_name_map, mark=_cv_mark)
        _cv_mark("  backfill: the anti-join + build and concat the new rows")

    grp = ["Currency", "BIN", "RPGT", "_pmp", "_ctry", "period"]
    if _enforced:
        _t0 = pp[pp["t"] == 0].copy()
        _t0["_rpgtl"] = _t0["RPGT"].astype(str).str.strip().str.lower()
        _t0["_vml"] = _t0["vampMid"].astype(str).str.strip().str.lower()
        t0 = _t0.merge(prop, on=["Currency", "BIN", "_rpgtl", "_pmp", "_ctry", "_vml"], how="left")
        t0["_prop_from_coarse"] = 0.0   # DIAGNOSTIC flag
        # HIERARCHICAL coarse (pmp, Country) MEAN fallback — REMOVED 2026-08-17.
        # It filled profiles whose exact 6-key merge missed with a MEAN of the enforced
        # share over the surviving profiles. A mean is not a routing decision: for a
        # Country/pmp-concentrated gateway it HALVES the share (WoodForest, named in the
        # original comment), and it has NO in-search analogue at all — so every row it
        # touched was pure scored-vs-delivered drift the GA could never model.
        # An unmatched profile now keeps prop_raw = NaN → 0 just below, so prop_sum = 0
        # there, _move = 0, and the profile is HELD AT BASELINE — exactly what the band
        # scaffold does for a profile it cannot represent.
        # Kill-switch: `_SW_COARSE_PROP_FALLBACK = True` restores it.
        if _SW_COARSE_PROP_FALLBACK:
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
    # profiles; USA-only gateways can't serve Non-USA profiles — zero their proposed share
    # there, so the renormalised split matches what the pipeline actually routes.
    # Static masks only for RAW prop_items — enforced shares already have them baked in.
    _wc_s = {str(x).strip().lower() for x in (wallet_incapable or set())}
    _uo_s = {str(x).strip().lower() for x in (usa_only or set())}
    # ── 19hh: ONE mask builder for BOTH sites in this function. ────────────────────────────
    # This function had the wallet/USA test written out TWICE — here for the `prop_raw` zeroing
    # and again inside the exploration-floor block as `_emask_f`. Two copies of one rule is how
    # they drift; there is now one.
    #
    # AND IT CAN RUN AT TWO GRAINS. The pro-rata export carries vampMid, NOT gatewayFid, so a
    # name-set test is only well defined when every fid of a vampMid agrees — and they do not:
    # PaySafe - Total AV is wallet-capable on paysafe-usd-tav but not on paysafe-eur-tav, so the
    # name-set mask bars PaySafe from wallet profiles in USD too. The SEARCH resolved this in
    # 2026-08-17 by keying on (vampMid, CURRENCY); `build_split_exports` resolved it the same
    # day by going FID-grain (its template columns ARE fids). This function is the one consumer
    # left on the coarse test, which is the asymmetry §14 of
    # docs/scope_exploration_floor_in_search.md is about.
    #
    # 19ht: DEFAULT ON. Ben's call - both sides do the fine version. There are THREE grains in
    # this codebase, not two, and only the coarsest is wrong:
    #
    #   fid                    build_split_exports, since 2026-08-17. Its template columns ARE
    #                          fids, so it is exact by construction.
    #   (vampMid, currency)    the SEARCH, since 2026-08-17, and this function under the switch.
    #                          A pair is incapable only when every ACTIVE fid for it is.
    #   vampMid name sets      what this function used to do. OVER-BLOCKS: PaySafe - Total AV is
    #                          wallet-capable on paysafe-usd-tav but not on paysafe-eur-tav, so
    #                          the name-set test barred PaySafe from wallet profiles in USD too.
    #
    # MEASURED on the live Master_MID_List (2026-09-02): 296 (vampMid, currency) groups, 111 with
    # more than one fid, and **0** where the ACTIVE fids disagree on wallet capability. So the
    # pair grain and the fid grain give the SAME answer today - the two fine grains agree, and it
    # is only the coarse one that ever differed. Five groups disagree among ALL fids, every one of
    # them an inactive `-test` sibling; app_common.capability_pairs now flags it if an active fid
    # ever joins them, because that is the day pairs and fids stop agreeing.
    #
    # 19ip: ROUTING_EMASK_PAIRS IS GONE. It was a way back to the coarse vampMid-only test,
    # and it could not be taken safely: `projection_cache_sig` read the same switch with
    # DEFAULT "0" while this function read it with default "1", so an UNSET run (pair grain,
    # hashed emp=0) and an explicit ROUTING_EMASK_PAIRS=0 run (coarse grain, hashed emp=0)
    # produced the SAME cache key and DIFFERENT answers - the exact stale-projection class the
    # hash was added to prevent. The coarse test also over-blocks any vampMid whose fids differ
    # in capability by currency, which is why nothing had asked for it back. The pair grain is
    # now unconditional wherever pair data exists; with none supplied the name-set fallback
    # below still runs, so a caller that has only vampMid names is unaffected.
    _wc_p = {(str(a).strip().lower(), str(b).strip().lower())
             for a, b in (wallet_incapable_pairs or ())}
    _uo_p = {(str(a).strip().lower(), str(b).strip().lower())
             for a, b in (usa_only_pairs or ())}
    _use_pairs = bool(_wc_p or _uo_p)

    def _cap_emask():
        """The wallet/USA capability mask over `t0`, at whichever grain is armed."""
        _wallet = t0["_pmp"].isin(["googlepay", "applepay"])
        _nonusa = ~t0["_ctry"].isin(["usa", "us", "_all_", ""])
        _ml = t0["vampMid"].astype(str).str.strip().str.lower()
        if _use_pairs:
            _cu = t0["Currency"].astype(str).str.strip().str.lower()
            _pr_key = list(zip(_ml.tolist(), _cu.tolist()))
            _wc_hit = pd.Series(np.array([_k in _wc_p for _k in _pr_key], dtype=bool),
                                index=t0.index)
            _uo_hit = pd.Series(np.array([_k in _uo_p for _k in _pr_key], dtype=bool),
                                index=t0.index)
            return (_wallet & _wc_hit) | (_nonusa & _uo_hit)
        return (_wallet & _ml.isin(_wc_s)) | (_nonusa & _ml.isin(_uo_s))

    # 19hr: this global is FACT (19df), and until now it recorded only which grain was ARMED —
    # which is intent wearing a fact's clothes. The mask has exactly TWO consumers in this
    # function, and BOTH can be closed at once: the `prop_raw` zeroing is skipped for an ENFORCED
    # 7-tuple frame (the shares already have the masks baked in), and the exploration-floor block
    # is skipped when `exploration_floor` is 0. tab_2's reconcile path hits both, so the grain
    # this function chose changed NOTHING on that path - which is what the 2026-09-02 16:19 run
    # was, and nothing in the log could say so. Record the consumers that actually ran.
    _emask_grain = ("(vampMid, currency) pairs" if _use_pairs else
                    "vampMid name sets" if (_wc_s or _uo_s) else
                    "none - no capability data supplied")
    _emask_applied = []
    if (_wc_s or _uo_s or _use_pairs) and not _enforced:
        _emask = _cap_emask()
        t0["prop_raw"] = np.where(_emask, 0.0, t0["prop_raw"])
        _emask_applied.append("prop_raw zeroing")
    t0["_av"] = t0["VI_Txn_Count"] * t0["_keep"]
    _g = t0.groupby(grp)   # group profile keys ONCE, reuse for all three sums (bit-identical)
    t0["profile_tot"] = _g["VI_Txn_Count"].transform("sum")
    t0["_at"] = _g["_av"].transform("sum")
    t0["base_share"] = np.where(t0["_at"] > 0, t0["_av"] / t0["_at"], 0.0)
    # Renormalise each profile's proposed shares back to a clean 100 budget after the coarse
    # pmp/Country fill and _keep zeroing may have pushed the per-profile sum off 100 (diagnostic:
    # _psum_pre keeps the pre-renorm sum). NOTE: prop_share below is prop_raw/prop_sum, which is
    # scale-invariant, so this does not change the projected split — it only keeps prop_sum ≈ 100
    # so the shares read as a clean percentage and no downstream code can assume a stale budget.
    t0["_psum_pre"] = _g["prop_raw"].transform("sum")
    t0["prop_raw"] = np.where(t0["_psum_pre"] > 0, t0["prop_raw"] * 100.0 / t0["_psum_pre"], t0["prop_raw"])
    t0["prop_sum"] = t0.groupby(grp)["prop_raw"].transform("sum")
    t0["prop_share"] = np.where(t0["prop_sum"] > 0, t0["prop_raw"] / t0["prop_sum"], t0["base_share"])
    _cv_mark("per-profile transforms (profile_tot / _at / base_share / prop_sum)")
    # EXPLORATION FLOOR (replicate the AllocationEngine): every ELIGIBLE gateway in a routed profile
    # keeps >= floor of the redistributed share, then renormalise. This is the primary reason a
    # 0%-rule incumbent (e.g. Braintree in a restricted RPGT) still retains volume in tab 5 — the
    # flat exported rule drives it to ~0, but the engine floors it. Eligible = present in the profile
    # (base_share>0 or prop_raw>0), not switched-off (_keep>0), and NOT wallet/USA-masked (so the
    # floor never un-masks an ineligible gateway). floor=0 → unchanged (backward-compatible).
    _efloor = float(exploration_floor or 0.0)
    if _efloor > 0.0:
        # 19hh: was a second, independent copy of the wallet/USA test. Now the SAME builder
        # the `prop_raw` zeroing above uses, so the floor's eligibility set and the zeroing's
        # can no longer disagree — both read the same builder at the same grain.
        _emask_f = _cap_emask()
        _emask_applied.append("exploration-floor eligibility")
        _elig_f = (((t0["base_share"] > 0) | (t0["prop_raw"] > 0)) & (t0["_keep"] > 0)
                   & (~_emask_f) & (t0["prop_sum"] > 0))
        _nef = t0.assign(_ef=_elig_f.astype(float)).groupby(grp)["_ef"].transform("sum")
        _flc = np.where(_nef > 0, np.minimum(_efloor, 1.0 / np.maximum(_nef, 1.0)), 0.0)
        t0["prop_share"] = np.where(_elig_f, np.maximum(t0["prop_share"], _flc), t0["prop_share"])
        _psh_sum = t0.groupby(grp)["prop_share"].transform("sum")   # renormalise profiles we floored
        _do_renorm = (t0["prop_sum"] > 0) & (_psh_sum > 0)
        t0["prop_share"] = np.where(_do_renorm, t0["prop_share"] / _psh_sum, t0["prop_share"])
    if _emask_applied:
        globals()["_LAST_EMASK_GRAIN"] = (
            _emask_grain + " - applied at: " + ", ".join(_emask_applied))
    else:
        _why = ("frame is ENFORCED (7-tuple), so the prop_raw zeroing is skipped" if _enforced
                else "no wallet/USA capability data was supplied")
        globals()["_LAST_EMASK_GRAIN"] = (
            _emask_grain + " - NOT APPLIED, no consumer ran: " + _why
            + f"; exploration_floor={_efloor:g} so the floor block is skipped.")
    _cv_mark("  the exploration floor (a no-op when the floor is 0, which delivery's is)")
    # Movable fraction = go-live pro-rata × fcp1 cohort fraction (see _vamp_post_core).
    _p = t0["pro_rata"] * t0["fcp1_frac"]
    t0["post_txn"] = t0["profile_tot"] * ((1 - _p) * t0["base_share"] + _p * t0["prop_share"])
    # PER-MID movable fraction + VAMP-follows-the-volume redistribution share (see _vamp_post_core).
    t0["_move"] = np.where(t0["prop_sum"] > 0, t0["pro_rata"] * t0["fcp1_frac"], 0.0)
    # 19cv — VAMP-ELIGIBILITY. `apply_to:"vamp"` (target 0) IS honoured by the baseline pipeline
    # (those MIDs carry vampPre 0 against real VI_Txn) but was NOT honoured here: the recipient
    # share came straight from `prop_raw`, so an overridden MID held no VAMP of its own and was
    # then handed a slice of the moved pool. WoodForest 690 and Authorize 227 on the 14:39 run,
    # both from PRE 0. Zeroing `_vprop` removes them from the numerator AND the denominator, so
    # the remaining recipients absorb the whole pool and the profile VAMP total still conserves.
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
    # renormalises prop_raw to a profile share, runs the max-share water-fill, and THEN builds
    # vshare from the capped value. Delivery built vshare from raw `prop_raw`, which carries no
    # cap at all. Same shares in, two different recipients — and the difference favours exactly
    # the MIDs the cap exists to restrain, so a capped incumbent is SCORED at the cap and
    # DELIVERED above it. That is the sign pattern on the 2026-08-28 21:25 run: adyen_totalav
    # +186, braintree usa +156, worldpay +87 (all capped incumbents, delivered high) against
    # paysafe -156 and checkout -51 (small recipients, delivered low).
    #
    # Measured on the _19df fixture (one profile, adyen 99 vs 0.5 / 0.5, cap 0.97): Σ|delivered −
    # in-search| is 0.000000 at max_share 1.0 and 8.384 at 0.97, with DELIVERY returning the
    # identical numbers in both regimes — it never saw the cap. One variable, both shipped
    # functions, no code patched to produce it.
    #
    # THIS CHANGES WHAT SHIPS. The delivered M5 is the authoritative number and this moves it.
    # `_SW_DELIV_MAXSHARE = False` reverts.
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
    # log line first keyed off `_SW_DELIV_MAXSHARE` alone and so announced "the CAP is now
    # applied on BOTH sides" on the 2026-08-29 07:45 run, where it was FALSE: three call sites in
    # tab_2_routing_engine still passed no max_share, so `max_share` defaulted to 1.0, the guard below took
    # the raw branch, and the run reproduced 829 exactly. The env var is INTENT; this global is
    # FACT, and the log must only ever report the second. None = the cap was not applied.
    _cap_ms = float(max_share) if max_share else 1.0
    if (not _SW_DELIV_MAXSHARE) or not (0.0 < _cap_ms < 1.0):
        globals()["_LAST_DELIV_MAXSHARE"] = None
        t0["_vprop"] = t0["prop_raw"]
    else:
        globals()["_LAST_DELIV_MAXSHARE"] = _cap_ms
        _psum_c = t0.groupby(grp)["prop_raw"].transform("sum")
        _live = (_psum_c > 0).to_numpy()
        _sh = np.where(_live, t0["prop_raw"].to_numpy(float)
                       / np.where(_live, _psum_c.to_numpy(float), 1.0), 0.0)
        # ── 19ir: THE BANK-BLOCKED MASK AT (BIN, vampMid, currency) ──────────────────────
        # This site cannot express a gatewayFid, so it reads the canonical key instead - see
        # `blocked_keys_for`. No keys = no mask = no rule, recorded rather than assumed.
        _blk_ms = None
        if blocked_keys:
            try:
                _bk_b = t0["BIN"].astype(str).str.strip().to_numpy()
                _bk_v = t0["vampMid"].astype(str).str.strip().str.lower().to_numpy()
                _bk_c = t0["Currency"].astype(str).str.strip().str.lower().to_numpy()
                _bks = set(blocked_keys)
                _blk_ms = np.array([(_b, _v, _c) in _bks
                                    for _b, _v, _c in zip(_bk_b, _bk_v, _bk_c)], dtype=bool)
            except Exception:  # noqa: BLE001 - a mask must never break a projection
                _blk_ms = None
        if _BFM is not None and not blocked_keys:
            _BFM.saw_mask("_max_share_waterfill", False,
                          "no blocked_keys were passed to compute_vamp_prepost_granular")
        t0["_vprop"] = _max_share_waterfill(_sh, t0, grp, _cap_ms, _live, blocked=_blk_ms)
    if len(vamp_off_mids):
        _voff = {str(_m).strip().lower() for _m in vamp_off_mids}
        _vmask = ~t0["vampMid"].astype(str).str.strip().str.lower().isin(_voff)
        t0["_vprop"] = t0["_vprop"] * _vmask.astype(float)
    t0["_vpsum"] = t0.groupby(grp)["_vprop"].transform("sum")
    t0["_vshare"] = np.where(t0["_vpsum"] > 0, t0["_vprop"] / t0["_vpsum"], 0.0)
    _cv_mark("  the max-share cap water-fill (_max_share_waterfill) + the VAMP share")

    # RPGT scope: hold non-scoped RPGTs at their current baseline split (post == pre).
    if scoped_rpgts:
        _scope = {str(r).strip().lower() for r in scoped_rpgts}
        _oos = ~t0["RPGT"].astype(str).str.strip().str.lower().isin(_scope)
        t0.loc[_oos, "_move"] = 0.0

    # TWO-COHORT volume (per-MID held on own gateway; pooled movable slice redistributed).
    t0["_bm"] = t0["base_share"] * t0["_move"]
    t0["_moved_tot"] = t0.groupby(grp)["_bm"].transform("sum")
    t0["post_txn"] = t0["profile_tot"] * (t0["base_share"] * (1 - t0["_move"])
                                       + t0["_moved_tot"] * t0["prop_share"])

    # ── TXN TERM STASH (read-only) ────────────────────────────────────────────────
    # post = profile_tot·(base_share·(1−move) + moved_tot·prop_share). These columns exist only
    # inside this function and are dropped on return, so the reconcile can never compare terms
    # with the in-search projector. Stash the per-(vampMid, period) sums here — nothing is
    # modified and nothing downstream reads this global.
    #
    # 19jh: GATED ON `FORENSIC`, like the other four. 19gt made [passthru], [pshare-why],
    # [vterms] and [move-gate] on-demand and this one was left behind — nobody noticed,
    # because until 19je split the row it was hidden inside a step named after something
    # else. [cvp-timing] then put it at 10.0s of an 80.7s projection, the LARGEST single
    # step, on a run where [forensic] skipped every other stash because the reconciliation
    # error was 1 unit, inside the float32 noise floor. Ten seconds explaining a number
    # there was nothing to explain about.
    #
    # This is not the switch 19fq deleted. The caller projects with FORENSIC False, reads
    # the drift off that projection, and projects AGAIN with it True if the drift is real -
    # so the explanation is never unavailable, it is computed exactly when there is
    # something to explain. A skipped stash reads "skipped", not None: None means it FAILED
    # and a reader has to be able to tell those apart.
    if not FORENSIC:
        globals()["_LAST_TXN_TERMS"] = "skipped"
        globals()["_LAST_TXN_DENOM"] = "skipped"
    try:
        if not FORENSIC:
            raise _Skip()
        _tt_ct = pd.to_numeric(t0["profile_tot"], errors="coerce").fillna(0.0)
        _tt_bs = pd.to_numeric(t0["base_share"], errors="coerce").fillna(0.0)
        _tt_mv = pd.to_numeric(t0["_move"], errors="coerce").fillna(0.0)
        _tt_mt = pd.to_numeric(t0["_moved_tot"], errors="coerce").fillna(0.0)
        _tt_ps = pd.to_numeric(t0["prop_share"], errors="coerce").fillna(0.0)
        # ── DENOMINATOR STASH (read-only, opt-in) ──────────────────────────────────
        # prop_share = prop_raw / prop_sum, and the residual is now known to live here.
        # Stash the per-row numerator and the per-profile denominator so the reconcile can
        # compare them against the in-search prop_raw / psum on the SAME prop vector.
        # Gated on _RECON_MIDS (a set of lower-cased vampMids the reconcile sets just
        # before this call) and restricted to the profiles those MIDs actually occupy, so
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
    except _Skip:
        pass                       # [forensic] 19jh: skipped, not failed — stash already set
    except Exception:  # noqa: BLE001 — a diagnostic must never break the projection
        globals()["_LAST_TXN_TERMS"] = None

    # 19gq: the mark below used to cover EVERYTHING from "per-profile transforms" to
    # `VAMP_Post` — ~330 lines and 44.7s (32.7%) on the 2026-09-01 20:21 run, the single
    # largest line in [cvp-timing] and the one that could say least about itself. Six marks
    # now split it, so the next run names the expensive half instead of leaving it to be
    # guessed at. Marks only; nothing computed here moved.
    _cv_mark("  the [txn-terms] / [txn-denom] diagnostic stash")

    _sub = ["Currency", "BIN", "RPGT", "_pmp", "_ctry"]
    # #2 GO-LIVE TIMING: the pipeline applies the go-live weight by the APPEARANCE month
    # (target month m), not origination. So take the rule×cohort factor (_gf = fcp1 × has-rule
    # × scope, WITHOUT pro_rata) at the ORIGINATION profile, and multiply by the go-live pro_rata
    # of the APPEARANCE month (the t0 pro_rata at that period). VI-txn (t=0) is unchanged
    # because appearance == origination there.
    # BUGFIX: gate the VAMP move on the MID being ACTIVE (_keep>0), not just the profile being routed
    # (prop_sum>0). A switched-off vampMid (target=0 → _keep=0) has no transactions to re-route, so
    # its residual/baseline VAMP must pass through unchanged (VAMP_Post == VAMP_Pre). Previously the
    # profile-level go-live ramp "moved out" fraud from 0-transaction MIDs (e.g. EPX), draining it into
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
    _cv_mark("aged-frame merges (_gf at origination + _pr_app at appearance)")
    _gk = _sub + ["period", "t"]
    # ── [gk-code] 19gq: ONE INTEGER GROUP KEY, FIVE GROUPBYS ──────────────────────────
    # `_gk` is 7 columns, five of them STRINGS, and the aged frame is millions of rows.
    # Grouping on it costs a full hash of those strings, and this function does it FIVE
    # separate times: Σ_pshare here, the `_moved_vpool` pool below, cf_nopass's pool, and
    # cf_ps's two. Factorise the columns ONCE into a single int64 code and hand pandas that
    # instead — the same trick 19fy used on the projector's `_static`.
    #
    # BIT-IDENTICAL BY CONSTRUCTION, not by measurement. Only the group LABELS change; the
    # groupby kernel, the row order it accumulates in and the per-row scatter back are all
    # untouched, so every sum is the same additions in the same order. That matters because
    # these are float sums and float addition is not associative — a re-labelling that
    # merged or split even one group would move the last bits.
    #
    # IT DECLINES, and falls back to the string key, in two cases:
    #   * any `_gk` column holds a null. `groupby` DROPS null keys (dropna=True) and an
    #     integer code cannot express "dropped"; those rows would silently join a group.
    #   * `_grp_codes` itself declines — a value containing "|", or mixed-radix overflow.
    # Either way `_gk_by` stays the column list and nothing below can tell the difference.
    _gk_by, _gk_note = _gk, ""
    try:
        if any(bool(pp[_c].isna().any()) for _c in _gk):
            _gk_note = "a _gk column holds nulls, and groupby DROPS null keys"
        else:
            from routing_optimiser.s4_search.band_projection import _grp_codes as _gk_codes
            _gk_c = _gk_codes(pp, _gk)
            if _gk_c is None:
                _gk_note = "_grp_codes declined (a value contains '|', or radix overflow)"
            else:
                _gk_by = pd.Series(np.asarray(_gk_c, np.int64), index=pp.index, name="_gkc")
    except Exception as _gke:  # noqa: BLE001
        _gk_by, _gk_note = _gk, f"{type(_gke).__name__}: {_gke}"
    globals()["_LAST_GK_CODE"] = {
        "used": _gk_by is not _gk, "why": _gk_note,
        "groups": (int(pd.Series(_gk_by).nunique()) if _gk_by is not _gk else -1),
        "rows": int(len(pp)), "verified": None, "verify_secs": 0.0}

    _psum = pp.groupby(_gk_by)["_pshare"].transform("sum")
    # ONE-SHOT PROOF, on the first of the five. Costs one string groupby — the very thing
    # being removed — so it is worth paying exactly once and then turning off. It compares
    # the int64 BIT PATTERNS, not `==`: two float arrays that differ in the last ulp compare
    # equal under `allclose` and would let a re-labelling through.
    # 19ju: DEFAULT FLIPPED TO OFF, which is the lifecycle this comment always described -
    # "worth paying exactly once and then turning off". It has printed VERIFIED on every run
    # since 19gq, so the 1.7s is now paid for a line nobody reads. ROUTING_GKCODE_VERIFY=1
    # brings it back, and a FAILURE still ships the reference and shouts. The identity is
    # structural anyway - same kernel, same row order, only the group LABELS differ.
    # 19kg: ROUTING_GKCODE_VERIFY deleted - it was already default OFF, so this is the
    # behaviour every run has had. The reference path stays for the same reason [cap-key]'s
    # does; only the way to arm it from a shell is gone.
    if False:
        try:
            _gv_t = _cv_time.perf_counter()
            _gv_ref = pp.groupby(_gk)["_pshare"].transform("sum")
            _gv_ok = bool(np.array_equal(
                np.asarray(_psum, float).view(np.int64),
                np.asarray(_gv_ref, float).view(np.int64)))
            _LAST_GK_CODE["verified"] = _gv_ok
            _LAST_GK_CODE["verify_secs"] = float(_cv_time.perf_counter() - _gv_t)
            if not _gv_ok:
                # NOT a silent fallback: ship the reference and say so. A wrong group key
                # changes which VAMP is redistributed where, which is the number this whole
                # function exists to produce.
                _psum = _gv_ref
                _LAST_GK_CODE["used"] = False
                _LAST_GK_CODE["why"] = ("VERIFY FAILED — the int64 key did not reproduce the "
                                        "string key's Σ_pshare bit for bit; reverted to the "
                                        "string key for THIS groupby only")
                _gk_by = _gk
        except Exception as _gve:  # noqa: BLE001
            _LAST_GK_CODE["verified"] = None
            _LAST_GK_CODE["why"] = f"verify skipped ({type(_gve).__name__}: {_gve})"
    _cv_mark("Σ_pshare per aged group (groupby-transform" 
             + (" on the [gk-code] int key)" if _gk_by is not _gk else " on the string key)"))
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
    if not FORENSIC:
        globals()["_LAST_MOVE_GATES"] = "skipped"
    try:
        if not FORENSIC:
            raise _Skip()
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
    except _Skip:
        pass                       # [forensic] 19gt: skipped, not failed — stash already set
    except Exception:  # noqa: BLE001
        globals()["_LAST_MOVE_GATES"] = None
    _cv_mark("[move-gate] five one-gate-lifted variants + stash")

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
        if not FORENSIC:
            globals()["_LAST_PASSTHRU"] = "skipped"
            raise _Skip()
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
    except _Skip:
        pass                       # [forensic] 19gt: skipped, not failed — stash already set
    except Exception:  # noqa: BLE001
        globals()["_LAST_PASSTHRU"] = None
    _cv_mark("[passthru] fired-set stash (7-column string key + 2 groupbys)")

    pp["_move"] = np.where(_psum > 1e-12, pp["_move"], 0.0)               # no recipient → passthrough
    pp["_pshare"] = np.where(_psum > 1e-12, pp["_pshare"] / _psum, 0.0)   # recipients sum to exactly 1
    pp["VAMP_Pre"] = pp["vampCount"]
    pp["_moved_v"] = pp["vampCount"] * pp["_move"]
    pp["_moved_vpool"] = pp.groupby(_gk_by)["_moved_v"].transform("sum")   # [gk-code] 19gq
    pp["VAMP_Post"] = pp["vampCount"] * (1.0 - pp["_move"]) + pp["_moved_vpool"] * pp["_pshare"]

    # ── WHY IS Σ_pshare < 1? (19cr, read-only) ─────────────────────────────────────────────
    # The renormalise above repairs a shortfall. Whether it is the RIGHT repair depends entirely on
    # why the shortfall exists, and Σ_pshare cannot tell you: an intended recipient missing from a
    # group is either a MID that genuinely has no fraud of that age (STRUCTURAL — re-basing is
    # correct, and the in-search projector should get the same pass) or a MID the aged frame never
    # carries for that profile at all (ABSENT — the frame is short and both sides are wrong). This
    # splits the missing share mass across those two classes and names who carries each.
    _cv_mark("recipient share + move fractions (_move / _pshare / VAMP_Post)")
    # 19fq: ROUTING_PSHARE_WHY DELETED. This stash is not optional instrumentation — tab 2 reads
    # it at two sites to explain a recipient-share drift, and a switch that can turn off the only
    # explanation of a number you are driving to zero is a switch that will be found off on the
    # run where it mattered. Unconditional now.
    if not FORENSIC:
        globals()["_LAST_PSHARE_WHY"] = "skipped"
    if FORENSIC:
        _pw_t_0 = _time.perf_counter()
        try:
            # INTENDED recipients, at origination: the rows whose vshare made up the 1.0. This is
            # keyed on `grp = _sub + ["period"]` (see above), which is exactly the key `_mv` is
            # merged back on, so these shares sum to 1 over the profile-month by construction.
            _pw_t0 = t0[_sub + ["vampMid", "period", "_vshare"]].copy()
            _pw_t0["_vshare"] = pd.to_numeric(_pw_t0["_vshare"], errors="coerce").fillna(0.0)
            _pw_t0 = _pw_t0[_pw_t0["_vshare"] > 0.0].rename(columns={"period": "orig_m"})

            # THE GROUP IS `_gk = _sub + ["period", "t"]`, NOT `orig_m`. 19cs: the previous version
            # tested membership at (profile, vampMid, orig_m), and orig_m = period - t is MANY-TO-ONE
            # over _gk — every (period, t) on one diagonal shares an orig_m, so a MID present at any
            # single age was scored present at every age. It found 0 missing on a frame with 168,945
            # short groups. Membership must be asked of the group that has to sum to 1.
            _pw_grp = pp.assign(_pw_s=_psum)[_gk + ["orig_m", "_pw_s"]].drop_duplicates(_gk)
            _pw_grp = _pw_grp[_pw_grp["_pw_s"] > 1e-12]        # passthrough groups are not repaired
            _pw_short = float((1.0 - _pw_grp["_pw_s"]).clip(lower=0.0).sum())

            # COUNTED, NOT EXPANDED. Expanding groups x intended recipients cost 13.1 s at the live
            # shape and was OOM-killed at 4x it. The expansion was only counting: an intended
            # (profile, orig_m, mid) with share v misses  v x (live groups there - live groups there
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

            # Of the MISSING, which appear ANYWHERE in the aged frame for that profile? A MID that does
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
    # 19ih: THE SENTINEL WAS WRITTEN AND THEN CLOBBERED THREE LINES LATER.
    # 19gt added the "skipped" sentinel so [recon-breakdown] could tell a DELIBERATE skip from a
    # genuine absence, and 19hw taught it to read the sentinel. But the `if FORENSIC: ... else:`
    # below still carried its pre-19gt `else` branch, which set the same globals back to None on
    # exactly the runs the sentinel was for. So the sentinel never survived to a reader, and
    # every non-forensic run printed "\u26a0 [recon-breakdown] UNAVAILABLE, and NOT BY THE
    # FORENSIC GATE: the delivered VAMP-terms stash is missing" - when the forensic gate is
    # precisely what skipped it. The message was inverted, not merely unhelpful: it sent the
    # reader looking for a defect on every clean run.
    # One if/else now, so there is no second writer.
    if not FORENSIC:
        globals()["_LAST_VAMP_TERMS"] = "skipped"
        globals()["_LAST_VAMP_PSUM"] = "skipped"
        globals()["_LAST_VAMP_CF_SKIPPED"] = "skipped"
    else:
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
            #     CAN ONLY DIFFER if some profile's _psum is not exactly 1: if every live profile
            #     already sums to 1, multiplying by it is the identity.
            _vt_nr_live = _vt_sum[_vt_sum > 1e-12]
            if len(_vt_nr_live) and bool((_vt_nr_live.sub(1.0).abs() > 1e-9).any()):
                _vt_ps_raw = _vt_ps * _vt_sum
                _vt_cf_nr = _vt_vc * (1.0 - _vt_mv) + _vt_pl * _vt_ps_raw
            else:
                _vt_cf_nr = _vt_post
                _vt_skipped.append("cf_norenorm (every live profile's prop sums to 1, so undoing "
                                   "the renormalise is the identity)")
            _cv_mark("[vterms] counterfactual cf_norenorm (elementwise; skippable)")

            # (3) cf_nopass — undo ONLY the "no recipient -> passthrough" override, so `move`
            #     reverts to gf x pr_app and the pool is rebuilt from that.
            #     CAN ONLY DIFFER if that override actually fired, i.e. some profile had _psum == 0.
            #     This is the expensive one: a groupby-transform over the whole aged frame.
            if bool((_vt_sum <= 1e-12).any()):
                _vt_mv_raw = np.where(pd.to_numeric(pp["orig_m"], errors="coerce").fillna(-1) >= 0,
                                      pd.to_numeric(pp["_gf"], errors="coerce").fillna(0.0)
                                      * pd.to_numeric(pp["_pr_app"], errors="coerce").fillna(0.0), 0.0)
                _vt_pool_raw = (pp.assign(_mvr=_vt_vc * _vt_mv_raw)
                                .groupby(_gk_by)["_mvr"].transform("sum"))   # [gk-code] 19gq
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
                # [gk-code] 19gq: POSITIONAL, and only when the merge left the frame the same
                # height as `pp`. `_pp2` comes back from a left merge with a fresh RangeIndex, so
                # the code Series cannot be aligned by label here — it is handed over as a raw
                # array, which pandas groups by position. A merge that MULTIPLIED rows (a
                # non-unique `_mv2` key) breaks that correspondence, and the length test is what
                # catches it; the string key is exact either way.
                _sm2 = (_ps2.groupby(np.asarray(_gk_by)).transform("sum")
                        if (_gk_by is not _gk and len(_pp2) == len(pp))
                        else _ps2.groupby([_pp2[c] for c in _gk]).transform("sum"))
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

    _tp = t0[_sub + ["vampMid", "period", "post_txn"]]
    pp = pp.merge(_tp, on=_sub + ["vampMid", "period"], how="left")
    pp["VI_Txn_Pre"] = np.where(pp["t"] == 0, pp["VI_Txn_Count"], 0.0)
    pp["VI_Txn_Post"] = np.where(pp["t"] == 0, pp["post_txn"].fillna(0.0), 0.0)
    _cv_mark("[vterms] VAMP-terms stash (4 terms + up to 3 counterfactuals) + the txn merge")
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
    # Collapse the pmp / Country profiles back to the reported grain (sums are exact).
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
_PROJ_CODE_VER = "2026-09-03-19ip-pair-grain-ONLY"  # bump on ANY projection-logic
# change so the in-memory st.cache_data entries bust on the next rerun (the data signature alone
# can't see code edits: a re-used outputs folder + unchanged split => identical key => stale result).


def projection_cache_sig(pp_path, prop_items, exploration_floor=0.0, extra="",
                         wallet_incapable_pairs=frozenset(), usa_only_pairs=frozenset(),
                         blocked_keys=frozenset()):
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
    # 19hh: the capability GRAIN is part of the answer, so the pair SETS are part of the key.
    # 19ip: the switch that selected the grain is gone, and so is the `|emp=` term that hashed
    # it. That term was itself the bug it was meant to prevent - it read ROUTING_EMASK_PAIRS
    # with default "0" while the projection read it with default "1", so an unset run and an
    # explicit =0 run hashed IDENTICALLY and computed at DIFFERENT grains. With one grain there
    # is nothing left to disambiguate. Dropping the term changes every key once: that is a
    # recompute, not a different answer.
    h.update(("|wcp=" + ",".join(sorted(f"{a}~{b}" for a, b in (wallet_incapable_pairs or ())))
              + "|uop=" + ",".join(sorted(f"{a}~{b}" for a, b in (usa_only_pairs or ())))
              # 19ir: the blocked keys change the water-filled share once the rule is armed, so
              # they belong in the key. Hashed even while the rule is refused, so an armed run
              # cannot be served an unarmed run's projection.
              + "|blk=" + ",".join(sorted("~".join(str(_x) for _x in _k)
                                          for _k in (blocked_keys or ())))
              ).encode("utf-8"))
    h.update(f"|floor={float(exploration_floor or 0.0):.8g}|{extra}|cv={_PROJ_CODE_VER}".encode("utf-8"))
    return f"{mt:.0f}:{len(prop_items or ()):d}:{h.hexdigest()}"


# [FN-267]
@_cache_data(show_spinner=False)
def _c_prepost_granular(pp_path, m, prop_items, excluded_mids, kill_eff=(), month_0=None,
                        scoped_rpgts=(), wallet_incapable=frozenset(), usa_only=frozenset(),
                        exploration_floor=0.0, vamp_off_mids=frozenset(),
                        cap_sig="", _capability=None, max_share=1.0,
                        wallet_incapable_pairs=frozenset(), usa_only_pairs=frozenset(),
                        blocked_keys=frozenset()):
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
                                         max_share=max_share,
                                         # 19ir: PLAIN, so it participates in the cache key -
                                         # the keys change the water-filled share once the rule
                                         # is armed, and a frozenset of tuples hashes stably.
                                         blocked_keys=blocked_keys,
                                         # 19hh: PLAIN (non-underscore) names so they
                                         # participate in the st.cache_data key — the grain
                                         # changes the answer, so a frame computed at the other
                                         # grain must not be served.
                                         wallet_incapable_pairs=wallet_incapable_pairs,
                                         usa_only_pairs=usa_only_pairs)


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
      * Country: each profile is split into USA and/or Non-USA rows from the attempts
        `country` field (country_pres). USA-only gateways (usa_only) appear in USA
        rows ONLY — zeroed and renormalised in Non-USA rows.
      * Max share: no gateway exceeds `max_share`; the excess is redistributed to the
        OTHER gateways ALREADY in the split (never activates a new gateway). Only
        applied when ≥2 gateways are present — a genuinely single-gateway profile can't
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
    # ── 19iq: THE BANK-BLOCKED MASK, FROM THE DATA ───────────────────────────────────────
    # `_apply_blocked_caps` stamps `_blocked` on the split it floors, so this water-fill knows
    # which rows are blocked without any caller having to remember to say so. No column = no
    # information = no rule (and the pricing does not run either, rather than reading all-False
    # as "nothing is blocked", which is a different claim).
    _blk_pairs = set()
    if "_blocked" in df.columns:
        try:
            _bm = df["_blocked"].fillna(False).astype(bool).to_numpy()
            if _bm.any():
                _blk_pairs = set(zip(df["BIN"].to_numpy()[_bm], df["gateway"].to_numpy()[_bm]))
        except Exception:  # noqa: BLE001 - a mask must never break an export
            _blk_pairs = set()
    _BLK_ARMED, _BLK_MSG = ((False, "[blk-fill] rule unavailable (blocked_fill did not import)")
                            if _BFM is None else
                            _BFM.arming_verdict(
                                _SW_BLOCK_NOFILL))
    if _BFM is not None:
        _BFM.saw_mask("_cap_rows", bool(_blk_pairs),
                      f"{len(_blk_pairs):,} (BIN, gateway) pair(s) from the split's _blocked "
                      f"column" if _blk_pairs else
                      ("the split carries _blocked but nothing is flagged"
                       if "_blocked" in df.columns else
                       "the split carries NO _blocked column, so this export has no blocked-row "
                       "information at all"))
    _BLK_MEAS = {"on_blocked": 0.0, "unavoidable": 0.0, "avoidable": 0.0, "sweeps": 0,
                 "pairs": len(_blk_pairs), "armed": bool(_BLK_ARMED), "msg": str(_BLK_MSG)}
    _pmps = ["GOOGLEPAY", "APPLEPAY", "non_gp_ap"]
    # PROFILE split: if the incoming split already carries pmp/Country per row (from a profile
    # grain GA run), the rows ARE the profiles — build the template DIRECTLY from them instead of
    # expanding each profile into country×pmp. Profile-grain splits (no pmp/ctry, or all "_all_") take the
    # existing expansion path byte-for-byte. Map the split's pmp/ctry to the template's format.
    _has_profile = ("pmp" in df.columns and "ctry" in df.columns
                    and not df["pmp"].astype(str).str.strip().str.lower().isin(["_all_", "", "nan"]).all())
    if _has_profile:
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
    def _cap_rows(V, _blk=None):
        """VECTORISED per-row cap + water-fill, applied to a whole (rows×gw)
        array at once. Each row (already normalised to sum 1) is capped at `_cap`, water-filling
        excess into the OTHER gateways already present. No-op for a row with <2 non-zero gateways
        or cap 1.0. Byte-identical to the scalar version — same 50-sweep water-fill, same order.

        `_blk` (19iq) is the (rows × gw) bank-blocked mask, from the split's own `_blocked`
        column. THIS IS THE WATER-FILL THAT SHIPS, so it is the one the blocked-row rule is
        ultimately about: the recipient test below (`share > 0`, not over, under the cap) is
        satisfied by a row that the auto-block pass has just pinned to the exploration floor, so
        the excess lifts it straight back off the floor. Under the rule a blocked row is a
        recipient of LAST RESORT. The rule only applies when every site is wired AND armed; the
        pricing (what the current behaviour puts on blocked rows, and how much of that a
        non-blocked sibling had room for) is measured either way."""
        V = V.copy()
        m = (_cap < 1.0) & ((V > 1e-12).sum(1) >= 2)
        if m.any():
            W = V[m]
            _Wb = None
            if _blk is not None and _BFM is not None:
                _Wb = np.asarray(_blk, bool)[m]
                if not _Wb.any():
                    _Wb = None
            for _ in range(50):
                over = W > _cap + 1e-12
                if not over.any():
                    break
                excess = np.where(over, W - _cap, 0.0).sum(1, keepdims=True)
                W = np.where(over, _cap, W)
                recip = (W > 1e-12) & (~over) & (W < _cap - 1e-12)
                room = np.where(recip, _cap - W, 0.0)
                rs = room.sum(1, keepdims=True)
                _add = np.where(rs > 1e-12, room / np.where(rs > 1e-12, rs, 1.0) * excess, 0.0)
                if _Wb is not None:
                    # PRICE IT (always). `_add[_Wb]` is what the unmodified rule hands to blocked
                    # rows; `unavoidable_excess` is the part of the excess no non-blocked sibling
                    # had room for. The difference is share the rule would move.
                    _onblk = float(_add[_Wb].sum())
                    _need = float(_BFM.unavoidable_excess_rowwise(excess, room, _Wb).sum())
                    _BLK_MEAS["on_blocked"] += _onblk
                    _BLK_MEAS["unavoidable"] += _need
                    _BLK_MEAS["avoidable"] += max(0.0, _onblk - _need)
                    _BLK_MEAS["sweeps"] += 1
                    if _BLK_ARMED:
                        _add = _BFM.two_stage_add_rowwise(room, _Wb, excess, _add)
                W = W + _add
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

    # VECTORISED per-row engine. Replaces the old per-(profile × country × pmp) Python loop, which
    # ran a groupby + Series ops + a 50-sweep cap PER ROW and scaled super-linearly with the profile
    # count (measured: tens of minutes at ~17k profiles). This applies the SAME transforms — base
    # normalisation, Non-USA USA-only zeroing, wallet-incapable zeroing, max-share water-fill,
    # 2dp residual-push rounding and BIN-GROUP condition codes — across all rows with array ops.
    # It was proven byte-identical to the previous implementation on random splits, including
    # with the <2-gateway back-fill active; that back-fill has since been deleted outright, so
    # there is no longer any per-row Python path left in here at all.
    globals()["_LAST_BLK_FILL"] = None      # 19iq: set at the end of the row engine below
    ng = len(gateways)
    incap_col = np.array([_incap(g) for g in gateways], dtype=bool)
    usa_col = np.array([_is_usa_only(g) for g in gateways], dtype=bool)
    _cols = (["GO LIVE", "BIN GROUP", "Brand", "RPGT", "Currency", "BIN",
              "paymentMethodProvider", "STICKY", "Country", "Check"] + gateways + ["DUP CHECK"])
    out = {}
    for rpgt, g_rpgt in df.groupby("RPGT"):
        if _has_profile:
            # rows ARE the profiles: base per (Currency, BIN, pmp, Country), no expansion.
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
            # per-profile normalised base (profiles sorted by Currency,BIN — the same order the old
            # groupby(["Currency","BIN"]) iterated, so BIN-GROUP condition codes match).
            base = (g_rpgt.groupby(["Currency", "BIN", "gateway"])["share"].sum()
                    .unstack("gateway").reindex(columns=gateways).fillna(0.0))
            base = base.div(base.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
            profiles = list(base.index)
            Bm = base.to_numpy(float)
            # expand rows in the SAME order as before: profile → country → pmp
            _idx, _cur, _bin, _ctry, _pmp = [], [], [], [], []
            for _ci, (cur, bin_) in enumerate(profiles):
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
        # template column set, not the profile's own doors.
        #
        # WHY THAT IS INDEFENSIBLE. It is share the GA cannot see at ANY grain, so it could never
        # be optimised against, never be predicted by the search, and never be attributed. It is
        # the mechanism that dumped volume onto Authorize / WoodForest.
        #
        # WHAT HAPPENS INSTEAD. A row left with <2 live gateways passes through UNTOUCHED and is
        # flagged by the `Check` column — the same treatment a genuinely single-gateway profile has
        # always had. The fix for such a row is data (open a second door in the MID list), not a
        # projection that pretends a door exists.
        #
        # Stage 2 ("backfill") therefore does nothing; see the stage-gate note above.
        if _slvl >= 3:                                   # [stage >=3: max-share water-fill]
            # 19iq: the mask in THIS block's row order - `_bin` per row, `gateways` per column.
            _Rblk = None
            if _blk_pairs:
                _Rblk = np.array([[(str(_b), _g) in _blk_pairs for _g in gateways]
                                  for _b in _bin], dtype=bool)
            R = _cap_rows(R, _Rblk)
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
        # `_SW_EXPORT_ROUND = True` restores the pre-19ef sheet exactly.
        _do_round = ((_slvl >= 4) and not projection_mode
                     and _SW_EXPORT_ROUND)
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
    # 19iq: PRICE THE RULE ON THE SHIPPING PATH. `on_blocked` is what the water-fill handed to
    # bank-blocked rows; `unavoidable` is the part of the excess no non-blocked sibling had room
    # for - the rule's own exception. `avoidable` is the difference, i.e. share that would move
    # if the rule were armed. tab_2's [blk-fill] prints the search-side twin of these numbers;
    # this is the delivered side, and the two have to agree once the rule is on.
    globals()["_LAST_BLK_FILL"] = dict(_BLK_MEAS)
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
    # enforced_prop_items returns. `_SW_FID2VAMP_BRAND = False` reverts.
    #
    # THE SPELLING TRAP, which has already cost a day on this codebase: the MID list spells the
    # brand "Total AV" and the run's company is "TotalAV" (tab_3_split_outputs_impact:4178). A plain
    # strip().lower() matches NOTHING and would silently drop every gateway, exactly as the
    # 2026-08-28 20:44 run's `build_capability` did — injection reported success while doing
    # nothing. Compare on whitespace-stripped keys, and RAISE rather than return an empty map.
    _bkey = _brand_key(brand)
    _brand_on = _SW_FID2VAMP_BRAND and bool(_bkey)
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
    # Key normalisation moved ABOVE the positive filter (2026-08-18p) so an all-zero profile can
    # still be identified at profile grain before it is discarded. Order of operations only —
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
    # ── CASE A: ZERO-PROFILE PLACEHOLDERS ────────────────────────────────────────────────────
    # A profile whose gateways ALL land on zero used to be dropped outright here, so it never
    # reached prop_items — and `blend_prop_items` only loops over the profiles prop_items contains.
    # `blend_profile_shares` therefore never ran for exactly the profiles its own docstring is about
    # ("No specific share in the profile → undefined profile → fall back to the catch-all alone"),
    # which is why the tab-3 parity line reports "0 new key(s)" while the in-search twin injects
    # the catch-all into 145 such profiles. That asymmetry is 20 of the 27 remaining reconciliation
    # units, confirmed per-MID by the [blend-profiles] counterfactual.
    # Keeping ONE zero-prop row per such profile is enough for the blend to see the profile:
    # blend_profile_shares filters `> 0`, gets an empty `spec`, and takes the catch-all branch. With
    # no catch-all configured it returns dict(spec) == {} and the profile emits nothing, i.e. exactly
    # today's behaviour; and with no blend at all a prop_raw of 0.0 adds nothing to any per-profile
    # sum and moves no volume. Revert with `_SW_CA_ZEROPROFILE = False` at the
    # top of this module.
    _ph_n = 0
    if _SW_CA_ZEROPROFILE:
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
        # offline on the Aug baseline: 176 uncovered profiles among the SCOPED RPGTs (0.2% of t0
        # volume) versus 16,592 among the UNSCOPED ones (67%), which are frozen by design
        # (hold_unselected_at_baseline). If unscoped RPGTs ever reach here the placeholder count
        # explodes and the catch-all would reroute two thirds of the book — so say so loudly
        # rather than let it pass as a routine number.
        try:
            _rb = (_ph.groupby("RPGT").size().sort_values(ascending=False)
                   if "RPGT" in _ph.columns else None)
            _msg = (f"[ca-zeroprofile] {_ph_n:,} zero-share profile(s) kept as placeholders so the "
                    f"backup catch-all can fire in profiles with NO specific rule "
                    f"(was: dropped at `prop_raw > 0`, so the catch-all never reached them)")
            if _rb is not None:
                _msg += " · by RPGT: " + " · ".join(f"{_k} {int(_v):,}" for _k, _v in _rb.items())
            if _ph_n > 2000:
                _msg += ("   ⚠ FAR more than the ~176 measured on the scoped Aug baseline — this "
                         "looks like UNSCOPED (baseline-frozen) RPGTs leaking into the split. "
                         "Those must NOT receive catch-all traffic; set `_SW_CA_ZEROPROFILE = False` and "
                         "check the RPGT scope before trusting any delivered number from this run.")
            print("   " + _msg)
        except Exception:  # noqa: BLE001
            pass
    # STASH for the caller to LOG. print() lands in the terminal, not in the run log — and the run
    # log is the artefact that actually gets read, so a guard that only prints is not a guard.
    # tab_2_routing_engine re-emits this through log() as [ca-zeroprofile], including the unscoped-RPGT check.
    try:
        globals()["_LAST_CA_ZEROPROFILE"] = {
            "n": int(_ph_n),
            "by_rpgt": ({str(_k): int(_v) for _k, _v in
                         _ph.groupby("RPGT").size().items()}
                        if _ph_n and "RPGT" in getattr(_ph, "columns", []) else {}),
        }
    except Exception:  # noqa: BLE001
        globals()["_LAST_CA_ZEROPROFILE"] = {"n": int(_ph_n), "by_rpgt": {}}
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
    per BIN profile (each variant already sums to 1, so the pooled shares sum to ~1). Share is
    re-normalised per (rpgt, currency, bank) profile. Empty frame if the split yields no rows.
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
    # Normalise within each export profile (rpgt, currency, BIN, pmp, Country) → share sums to 1.
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
    ask 'how many pools does this profile budget yield?' at each search step. Every arg
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
    if "profile_volume" not in _si.columns:
        _si["profile_volume"] = (_si.groupby(["rpgt", "currency", "bin"])["volume"].transform("sum")
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
                         "profile_volume", "baseline_share", "rate"] if c in split_ideal.columns]
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
    not seen before. `stats` carries raw_profiles/raw_pools/profiles/pools/global_accuracy/
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

    # Parallel k-ary budget search: probe several profile budgets per round across the cores so
    # the (expensive) config-generation counts overlap. Same result as the serial binary search
    # (verified budget ≤ target). Bounded to ≤8 workers; `_SW_COMPRESS_PARALLEL = 1` disables it.
    _par = _SW_COMPRESS_PARALLEL
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
def rpgt_avg_ticket(profile_agg):
    """RPGT-level average ticket from the 30D actuals (the window ending just before
    Month 0): Σ succ_amount / Σ successes per RPGT. Returns {rpgt_lower: ticket}.
    Kept as the FALLBACK for (rpgt, currency) combos with no per-currency actuals."""
    if profile_agg is None or getattr(profile_agg, "empty", True):
        return {}
    g = profile_agg.groupby("rpgt_join").agg(rev=("profile_rev", "sum"), succ=("profile_succ", "sum"))
    return {str(rp).strip().lower(): (float(r["rev"]) / float(r["succ"]) if float(r["succ"]) > 0 else 0.0)
            for rp, r in g.iterrows()}


def rpgt_currency_avg_ticket(profile_agg):
    """RPGT × Currency average ticket from the 30D actuals: Σ succ_amount / Σ successes
    per (rpgt, currency). Returns {(rpgt_lower, currency_lower): ticket}. Finer grain
    than ``rpgt_avg_ticket`` so a given RPGT no longer shares one blended ticket across
    currencies. Combos with no actuals are simply absent (caller falls back to the
    RPGT-level ticket)."""
    if profile_agg is None or getattr(profile_agg, "empty", True):
        return {}
    if "currency_join" not in getattr(profile_agg, "columns", []):
        return {}
    g = profile_agg.groupby(["rpgt_join", "currency_join"]).agg(
        rev=("profile_rev", "sum"), succ=("profile_succ", "sum"))
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
