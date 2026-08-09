"""Shared constants, the log handler, and small helpers used across the app's tabs.

Pulled out of the (very large) ``streamlit_app.py`` so each tab can live in its own file and
import what it needs from here — instead of every tab sharing one giant module scope. This
module has no side effects worth worrying about: it just defines names (and reads the
per-session ``st.session_state`` singleton, which is the SAME object in every module).

As more tabs are moved into their own files, the helpers they share move here too.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

# session_state is a per-session singleton — accessing it here gives the SAME object every
# other module sees, so tabs in separate files still share one state.
ss = st.session_state

# --- paths & project constants ---------------------------------------------------------
_HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
SQL_DIR = os.path.join(_HERE, "..", "queries")
CACHE_DIR = os.path.join(_HERE, "..", ".cache")
INPUTS_DIR = os.path.join(_HERE, "..", "config", "inputs")
GCP_PROJECT = "sapient-tangent-172609"  # matches the VAMP repo's BigQuery project

# RPGTs (transaction types) used across the pipeline and templates.
RPGT_LIST = [
    "Monthly Initial", "Annual Sub Sale", "Addon Sale", "Upgrades",
    "Monthly Renewal", "Annual Sub Renewal", "P6M Renewals", "Addon Renewal",
]
COMPANIES = ["TotalAV", "Total Drive", "Total Adblock", "Total Cleaner", "Total VPN"]


class StreamlitLogHandler(logging.Handler):
    """Streams log records live by calling a sink with each formatted line."""

    # [FN-233]
    def __init__(self, sink):
        super().__init__(logging.INFO)
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    # [FN-234]
    def emit(self, record):
        try:
            self.sink(self.format(record))
        except Exception:  # noqa: BLE001
            pass


# [FN-235]
def _switched_off_gateways(ov: dict) -> set:
    """Canonicalised, lower-cased gateway ids that are SWITCHED OFF in an already-loaded
    gateway_volume_overrides dict — i.e. target == 0 with apply_to in ("trx", "both").
    Centralises the identical set-building duplicated across the app."""
    from routing_optimiser.forecast_pipeline import _canonical_gateway
    out = set()
    if not isinstance(ov, dict):
        return out
    for _gw, _cfg in ov.items():
        if isinstance(_cfg, dict) \
                and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
            out.add(str(_canonical_gateway(_gw)).strip().lower())
    return out



# ============================ moved from streamlit_app.py ============================

# [FN-236]
def _variance_gap_temp(agg_sr, anchor=0.17, t_ceiling=0.30, n_cap=500.0):
    """Per-Bank×Currency softmax temperature from the STATISTICAL SIGNIFICANCE of
    the best-vs-second-best success-rate gap (variance-of-the-gap method).

    For each cell: z = (p1 - p2) / sqrt(se1^2 + se2^2), where p1/p2 are the top two
    gateways' success rates and se_i = sqrt(p_i(1-p_i)/n_i) on effective attempts.
    Big z (a confidently-real gap) → sharpen; z≈0 (overlapping error bars) → flat.
    Auto-calibrated: scale so the MEDIAN cell's temperature == `anchor` (the current
    0.17 default), so overall aggressiveness is unchanged and only the distribution
    across cells is data-driven. No user input. Returns (temps_by_cell, median_z, scale).
    """
    g = agg_sr.copy()
    g["_c"] = g["currency"].astype(str).str.strip().str.lower()
    g["_b"] = g["bank"].astype(str).str.strip().str.lower()
    g["_n"] = pd.to_numeric(g["attempts"], errors="coerce").fillna(0.0)
    g["_p"] = pd.to_numeric(g["success_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    zmap = {}
    for (c, b), grp in g.groupby(["_c", "_b"]):
        sub = grp[grp["_n"] > 0].sort_values("_p", ascending=False)
        if len(sub) < 2:
            zmap[(c, b)] = np.nan            # single gateway -> temperature irrelevant
            continue
        p1, p2 = float(sub["_p"].iloc[0]), float(sub["_p"].iloc[1])
        # Cap effective attempts: beyond n_cap, more data shouldn't keep inflating
        # the t-stat (otherwise every high-volume cell saturates the ceiling). The
        # dial then reflects WHETHER the gap is real, not how many millions prove it.
        n1 = min(float(sub["_n"].iloc[0]), n_cap)
        n2 = min(float(sub["_n"].iloc[1]), n_cap)
        se = np.sqrt(max(p1 * (1 - p1), 1e-9) / max(n1, 1e-9) +
                     max(p2 * (1 - p2), 1e-9) / max(n2, 1e-9))
        z = (p1 - p2) / se if se > 1e-12 else 50.0
        zmap[(c, b)] = max(float(z), 0.0)
    vals = [v for v in zmap.values() if v == v]
    med = float(np.median(vals)) if vals else 0.0
    scale = (anchor / med) if med > 1e-9 else None
    temps = {}
    for k, z in zmap.items():
        if (z != z) or (scale is None):      # nan gap or no calibration -> anchor
            temps[k] = float(anchor)
        else:
            temps[k] = float(min(max(z * scale, 0.0), t_ceiling))
    return temps, med, scale


# [FN-237]
def _ink_caption(md: str):
    """Render a caption in ink (near-black) rather than Streamlit's default grey."""
    import re as _re
    html = _re.sub(r"`([^`]+)`", r"<code>\1</code>", md)
    st.markdown(
        f"<div style='color:var(--tav-ink); font-size:0.82rem; line-height:1.35;'>{html}</div>",
        unsafe_allow_html=True)


# [FN-238]
def _fmt_secs(s):
    """Human-friendly duration, e.g. 45s, 2m 05s, 1h 12m."""
    s = max(int(round(s)), 0)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# [FN-239]
def _load_ga_perf():
    """Load the last GA timing calibration from disk (survives restarts)."""
    try:
        _p = os.path.join(CACHE_DIR, "ga_perf.json")
        if os.path.exists(_p):
            import json as _json
            with open(_p) as _f:
                return _json.load(_f)
    except Exception:
        pass
    return None


# [FN-240]
def _save_ga_perf(d):
    """Persist the GA timing calibration so the estimate survives Streamlit restarts."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        import json as _json
        with open(os.path.join(CACHE_DIR, "ga_perf.json"), "w") as _f:
            _json.dump(d, _f)
    except Exception:
        pass


_GA_N_SEED = 4   # parallel CMA-ES seeds per endpoint (also read by the settings-aware ETA scaling)


# [FN-241]
def _physical_cpu_count(default=4):
    """Number of PHYSICAL CPU cores (excludes hyperthreads). The seed default uses this so the
    parallel wave matches REAL cores: one CMA-ES seed per logical core oversubscribes an
    8-physical / 16-logical machine (common on macOS) and the workers then thrash the shared
    memory bandwidth, so more seeds stop buying throughput. Order: psutil (installed) →
    `sysctl hw.physicalcpu` on macOS → logical os.cpu_count() → `default`. Never raises."""
    try:
        import psutil  # a pinned project dependency
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess
        import sys
        if sys.platform == "darwin":
            _o = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                capture_output=True, text=True, timeout=2)
            if _o.returncode == 0 and _o.stdout.strip().isdigit():
                return int(_o.stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    return int(os.cpu_count() or default)


# [FN-242]
def _apply_blocked_caps(split, blocked_pairs, floor, bin_to_bank=None):
    """Cap the share of any BANK-BLOCKED (bank, gateway) to the exploration floor and redistribute
    the freed share to the OTHER (non-blocked) gateways in the same cell, proportionally. Cells with
    no non-blocked recipient are left unchanged (nowhere to move the volume). Matches on
    lower/stripped (bank, gateway) — and, when `bin_to_bank` is given, ALSO on the parent-bank grain,
    so a BIN-vs-parent grain mismatch can't silently cap nothing here while the pre-GA auto-block
    excludes the same rows. Returns (new_split, n_rows_capped). Deterministic; no-op when
    `blocked_pairs` is empty or nothing matches."""
    if not blocked_pairs or split is None or getattr(split, "empty", True) \
            or not {"bank", "gateway", "share"}.issubset(split.columns):
        return split, 0
    d = split.copy()
    _bk = d["bank"].astype(str).str.strip().str.lower()
    _gw = d["gateway"].astype(str).str.strip().str.lower()
    # Grain-robust match. The split's `bank` is BIN-level, but `blocked_pairs` may be keyed at
    # PARENT-bank grain (detect_blocked_gateways runs on bin_to_bank-mapped attempts). Treat a row
    # as blocked if EITHER its BIN bank OR its parent bank pairs with the blocked gateway. With no
    # map supplied this is byte-identical to the old BIN-only match (no behaviour change).
    if bin_to_bank:
        _bb = {str(k).strip().lower(): str(v).strip().lower() for k, v in bin_to_bank.items()}
        _pk = _bk.map(lambda b: _bb.get(b, b))
        _isb = np.array([((_b, _g) in blocked_pairs) or ((_p, _g) in blocked_pairs)
                         for _b, _p, _g in zip(_bk, _pk, _gw)])
    else:
        _isb = np.array([(_b, _g) in blocked_pairs for _b, _g in zip(_bk, _gw)])
    if not _isb.any():
        return d, 0
    _key = [c for c in ("rpgt", "currency", "bank", "pmp") if c in d.columns] or ["bank"]
    _sh = pd.to_numeric(d["share"], errors="coerce").fillna(0.0).to_numpy()
    _cap = np.where(_isb, np.minimum(_sh, float(floor)), _sh)          # blocked -> <= floor
    d["_freed"] = _sh - _cap                                            # >= 0, only blocked rows
    d["_rw"] = np.where(_isb, 0.0, _cap)                                # recipients = non-blocked
    _grp = d.groupby(_key)
    _freed_cell = _grp["_freed"].transform("sum").to_numpy()
    _rw_cell = _grp["_rw"].transform("sum").to_numpy()
    _has_recip = _rw_cell > 1e-12
    _add = np.where(_has_recip, d["_rw"].to_numpy() * _freed_cell / np.where(_has_recip, _rw_cell, 1.0), 0.0)
    # Apply the cap+redistribute only in cells that HAVE a non-blocked recipient; a cell whose
    # gateways are ALL blocked is left untouched (nowhere to move the freed volume).
    _new = np.where(_has_recip, _cap + _add, _sh)
    d["share"] = _new
    d = d.drop(columns=["_freed", "_rw"])
    return d, int((_isb & _has_recip & (_new < _sh - 1e-12)).sum())


_TAV_FIDS = [
    'adyen-aud-tav','adyen-aud-tav-emea','adyen-cad-tav','adyen-cad-tav-emea',
    'adyen-eur-tav','adyen-gbp-tav','adyen-gbp-tav-prem','adyen-gbp-tav-pro',
    'adyen-gbp-tav-ultimate','adyen-usd-tav','adyen-usd-tav-avonline',
    'adyen-usd-tav-emea','adyen-usd-tav-prem','adyen-usd-tav-pro',
    'adyen-usd-tav-secure','adyen-usd-tav-ultimate','adyen-usd-tsc-x-tav',
    'authorize-usd-tav','bancard-usd-tav','braintree-aud-tav','braintree-cad-tav',
    'braintree-eur-tav','braintree-gbp-tav','braintree-usd-tav',
    'checkout-aud-tav-new','checkout-cad-tav-new','checkout-eur-tav-new',
    'checkout-gbp-tav-new','checkout-usd-tav-new','cwams-usd-tav',
    'merrick-usd-tav','paysafe-aud-tav','paysafe-cad-tav','paysafe-eur-tav',
    'paysafe-gbp-tav','paysafe-usd-tav','woodforest-usd-tav',
    'worldpay-aud-tav-nt','worldpay-cad-tav-nt','worldpay-usd-tav-nt','adyen-usd-tav-na',
]


_TDR_FIDS = [
    'adyen-aud-tdr','adyen-cad-tdr','adyen-eur-tdr','adyen-gbp-tdr','adyen-usd-tdr',
    'authorize-usd-tdr','braintree-aud-tdr','braintree-cad-tdr','braintree-eur-tdr',
    'braintree-gbp-tdr','braintree-usd-tdr','worldpay-usd-tdr-nt','worldpay-usd-tdr',
    'adyen-usd-tdr-backup','adyen-usd-tdr-secure','woodforest-usd-tdr',
    'adyen-gbp-tdr-backup','adyen-gbp-tdr-secure','adyen-usd-tdr-na',
]


_TAB_FIDS = [
    'adyen-aud-tab','adyen-cad-tab','adyen-eur-tab','adyen-gbp-tab',
    'adyen-gbp-tab-online','adyen-gbp-tab-pro','adyen-usd-tab',
    'adyen-usd-tab-blockerpro','adyen-usd-tab-emea','adyen-usd-tab-mobile',
    'adyen-usd-tab-online','adyen-usd-tab-pro','adyen-usd-tsc-x-tab',
    'authorize-usd-tab','bancard-usd-tab','braintree-aud-tab',
    'braintree-cad-tab','braintree-eur-tab','braintree-gbp-tab',
    'braintree-usd-tab','checkout-aud-tab-new','checkout-cad-tab-new',
    'checkout-eur-tab-new','checkout-gbp-tab-new','checkout-usd-tab-new',
    'cwams-usd-tab','paysafe-aud-tab','paysafe-cad-tab','paysafe-eur-tab',
    'paysafe-gbp-tab','paysafe-usd-tab','woodforest-usd-tab','worldpay-usd-tab-nt','adyen-usd-tab-na',
]


DEFAULT_GATEWAY_FIDS = "(" + ",".join(f"'{f}'" for f in _TAV_FIDS + _TDR_FIDS + _TAB_FIDS) + ")"


APP_BUILD = "2026-07-24l"  # Post Volume now = cell_volume × routed share (same basis as $ Impact/Δ Share) — no more 0s


# [FN-243]
def _ensure_base_30d_metrics():
    """Compute & cache the 30-day baseline metrics (cell/gateway success rates,
    avg ticket, base totals) that the impact views rely on. Idempotent and shared
    by the Routing-engine tab (pre/post visuals) and the Impact tab, so both report
    identical pre/post revenue. Returns the cache dict, or None if no attempts data."""
    if "cached_base_30d_metrics" in ss:
        return ss["cached_base_30d_metrics"]
    if "adf" not in ss:
        return None
    adf_raw = ss["adf"].copy()
    date_col = "date" if "date" in adf_raw.columns else ("Date" if "Date" in adf_raw.columns else None)
    adf_30d = adf_raw.copy()
    if date_col:
        df_dates = pd.to_datetime(adf_raw[date_col], errors="coerce")
        valid_dates = df_dates.dropna()
        if not valid_dates.empty:
            max_dt = valid_dates.max()
            mask = (df_dates > (max_dt - pd.Timedelta(days=30))) & (df_dates <= max_dt)
            if mask.sum() > 0:
                adf_30d = adf_raw[mask].copy()

    # Collapse BINs into their parent Bank so the whole tab operates at the
    # Bank x Currency grain (matching the engine's scoring grain).
    _b2b = ss.get("bin_to_bank", {})
    if _b2b and "bank" in adf_30d.columns:
        adf_30d["bank"] = adf_30d["bank"].map(
            lambda b: _b2b.get(b, _b2b.get(str(b).strip().lower(), b))).astype(str)

    if "amount" in adf_30d.columns:
        adf_30d["amount"] = pd.to_numeric(adf_30d["amount"], errors="coerce").fillna(25.0)
    else:
        adf_30d["amount"] = 25.0
    if "success" in adf_30d.columns:
        adf_30d["success"] = pd.to_numeric(adf_30d["success"], errors="coerce").fillna(0)
    else:
        adf_30d["success"] = 0
    if "attempts" in adf_30d.columns:
        adf_30d["attempts"] = pd.to_numeric(adf_30d["attempts"], errors="coerce").fillna(0)
    else:
        adf_30d["attempts"] = 0
    adf_30d["succ_amount"] = adf_30d["amount"] * adf_30d["success"]

    cell_agg = adf_30d.groupby([adf_30d["rpgt"].astype(str).str.strip().str.lower(),
                                adf_30d["currency"].astype(str).str.strip().str.lower(),
                                adf_30d["bank"].astype(str).str.strip().str.lower()]).agg(
        cell_att=("attempts", "sum"), cell_succ=("success", "sum"), cell_rev=("succ_amount", "sum")
    ).reset_index().rename(columns={"rpgt": "rpgt_join", "currency": "currency_join", "bank": "bank_join"})
    cell_agg["cell_sr"] = np.where(cell_agg["cell_att"] > 0, cell_agg["cell_succ"] / cell_agg["cell_att"], 0)

    # Average value per successful transaction at the Bank x Currency level (ONE
    # value per bank x currency), used consistently for every revenue figure so the
    # impact tables reconcile. Falls back to $25 if a cell has no successes.
    bc_val = adf_30d.groupby([adf_30d["currency"].astype(str).str.strip().str.lower(),
                              adf_30d["bank"].astype(str).str.strip().str.lower()]).agg(
        bc_rev=("succ_amount", "sum"), bc_succ=("success", "sum"), bc_att=("attempts", "sum")
    ).reset_index()
    bc_val.columns = ["currency_join", "bank_join", "bc_rev", "bc_succ", "bc_att"]
    bc_val["avg_txn_value"] = np.where(bc_val["bc_succ"] > 0, bc_val["bc_rev"] / bc_val["bc_succ"], 25.0)
    cell_agg = cell_agg.merge(bc_val[["currency_join", "bank_join", "avg_txn_value"]],
                              on=["currency_join", "bank_join"], how="left")
    cell_agg["avg_ticket"] = cell_agg["avg_txn_value"].fillna(25.0)
    cell_agg = cell_agg.drop(columns=["avg_txn_value"])
    # Per-RPGT ticket (Bank×Currency×RPGT grain): used for revenue when the optimisation
    # grain is per-RPGT, so revenue tracks the RPGT mix (e.g. Annual Sub tickets ≫ Addon
    # tickets) instead of one blended value. Falls back to the Bank×Currency ticket where
    # an RPGT has no successes. At Bank×Currency grain the BC ticket (avg_ticket) is used.
    cell_agg["rpgt_ticket"] = np.where(cell_agg["cell_succ"] > 0,
                                       cell_agg["cell_rev"] / cell_agg["cell_succ"],
                                       cell_agg["avg_ticket"])

    gw_agg = adf_30d.groupby([adf_30d["rpgt"].astype(str).str.strip().str.lower(),
                              adf_30d["currency"].astype(str).str.strip().str.lower(),
                              adf_30d["bank"].astype(str).str.strip().str.lower(),
                              adf_30d["gateway"].astype(str).str.strip().str.lower()]).agg(
        gw_att=("attempts", "sum"), gw_succ=("success", "sum")
    ).reset_index().rename(columns={"rpgt": "rpgt_join", "currency": "currency_join", "bank": "bank_join", "gateway": "gateway_join"})
    gw_agg["gw_sr"] = np.where(gw_agg["gw_att"] > 0, gw_agg["gw_succ"] / gw_agg["gw_att"], np.nan)

    ss["cached_base_30d_metrics"] = {
        "base_att": adf_30d["attempts"].sum(),
        "base_succ": adf_30d["success"].sum(),
        "base_rev": adf_30d["succ_amount"].sum(),
        "cell_agg": cell_agg,
        "gw_agg": gw_agg,
        "adf_30d_raw": adf_30d,
        "date_col": date_col,
        "bc_val": bc_val,
    }
    return ss["cached_base_30d_metrics"]


# [FN-244]
def _impact_eval_frame(split, cache, by_rpgt=False):
    """Per-(rpgt, currency, bank, gateway) pre/post frame for a proposed split,
    using the SAME revenue basis as the Impact tab (cell_att × share × gw_sr ×
    avg_ticket). Adds pre/post/delta for volume, revenue and share. BINs are
    collapsed to parent bank. Returns a DataFrame.

    by_rpgt: when True (optimisation grain = Bank×Currency×RPGT) revenue uses the
    per-RPGT ticket so it tracks the RPGT mix; otherwise the Bank×Currency ticket."""
    b2b = ss.get("bin_to_bank", {})
    cell_agg, gw_agg = cache["cell_agg"], cache["gw_agg"]
    sv = split.copy()
    if "bank" in sv.columns:
        sv["bank"] = sv["bank"].map(
            lambda b: b2b.get(b, b2b.get(str(b).strip().lower(), b))).astype(str)
    for c in ["rpgt", "currency", "bank", "gateway"]:
        if c in sv.columns:
            sv[f"{c}_join"] = sv[c].astype(str).str.strip().str.lower()
    gcols = ["rpgt_join", "currency_join", "bank_join", "gateway_join"]
    amap = {c: (c, "first") for c in ["rpgt", "currency", "bank", "gateway"] if c in sv.columns}
    if "share" in sv.columns: amap["share"] = ("share", "mean")
    if "baseline_share" in sv.columns: amap["baseline_share"] = ("baseline_share", "mean")
    if "volume" in sv.columns: amap["volume"] = ("volume", "sum")
    if "cell_volume" in sv.columns: amap["cell_volume"] = ("cell_volume", "sum")
    sv = sv.groupby(gcols, as_index=False).agg(**amap)

    ev = sv.merge(cell_agg, on=["rpgt_join", "currency_join", "bank_join"], how="left")
    ev = ev.merge(gw_agg[["rpgt_join", "currency_join", "bank_join", "gateway_join", "gw_sr"]],
                  on=gcols, how="left")
    # Guarantee every column the calc below reads exists as a real Series. A split fed in without
    # cell_volume / avg_ticket / etc. (e.g. the enforced-split revenue view) would otherwise make
    # `ev.get(col, scalar)` return a scalar and crash on `.fillna`.
    for _c, _d in (("cell_att", 0.0), ("cell_sr", 0.0), ("avg_ticket", 25.0),
                   ("rpgt_ticket", np.nan), ("cell_volume", 0.0), ("volume", np.nan),
                   ("share", 0.0), ("baseline_share", 0.0)):
        if _c not in ev.columns:
            ev[_c] = _d
    ev["gw_sr"] = ev["gw_sr"].fillna(ev.get("cell_sr")).fillna(0.0)
    ev["cell_att"] = pd.to_numeric(ev.get("cell_att", 0), errors="coerce").fillna(0.0)
    # Ticket grain follows the optimisation grain (per-RPGT when the split is per-RPGT).
    _bc_ticket = pd.to_numeric(ev.get("avg_ticket", 25.0), errors="coerce")
    if by_rpgt and "rpgt_ticket" in ev.columns:
        ev["avg_ticket"] = pd.to_numeric(ev["rpgt_ticket"], errors="coerce").fillna(_bc_ticket).fillna(25.0)
    else:
        ev["avg_ticket"] = _bc_ticket.fillna(25.0)
    ev["share"] = pd.to_numeric(ev.get("share", 0), errors="coerce").fillna(0.0)
    ev["baseline_share"] = pd.to_numeric(ev.get("baseline_share", 0), errors="coerce").fillna(0.0)
    # ROOT FIX (dilution): the proposed `share` can arrive summing to ≪1 per cell — the
    # optimiser runs at parent-bank grain but the split is exploded to BINs, so each row's
    # share is a slice of the parent's total spread across its ~N BINs (≈ 1/N per cell),
    # which understates post volume/revenue by ≈N. Renormalise share (and baseline_share) to a
    # proper per-(rpgt,currency,bank) distribution HERE, at the shared source, so post_att /
    # post_succ / post_rev are correct for every downstream table (Bank Analysis, Financial
    # Impact, per-RPGT breakdown) with no per-table rescaling. Idempotent — a no-op when the
    # shares already sum to 1.
    for _sc in ("share", "baseline_share"):
        _tsum = ev.groupby(["rpgt_join", "currency_join", "bank_join"])[_sc].transform("sum").to_numpy()
        ev[_sc] = np.where(_tsum > 0, ev[_sc].to_numpy() / _tsum, ev[_sc].to_numpy())
    _cv = pd.to_numeric(ev.get("cell_volume", 0), errors="coerce").fillna(0.0)

    # Volume basis = forecast routed volume (cell_volume × share). Post uses the
    # summed 'volume' when present (identical), pre derives from baseline_share.
    # Post/Pre volume = forecast routed volume (cell_volume × share), computed from the
    # SAME routed share the revenue uses — so Volume and $ Impact always move together.
    # (The old code read a separate summed 'volume' column that could arrive as 0 and
    # zero out Post Volume while $ Impact / Δ Share still showed a gain.)
    ev["post_vol"] = _cv * ev["share"]
    ev["pre_vol"] = _cv * ev["baseline_share"]
    ev["vol_delta"] = ev["post_vol"] - ev["pre_vol"]
    
    # New base attempt/success calculations for the SR charts
    ev["post_succ"] = ev["cell_att"] * ev["share"] * ev["gw_sr"]
    ev["pre_succ"] = ev["cell_att"] * ev["baseline_share"] * ev["gw_sr"]
    ev["post_att"] = ev["cell_att"] * ev["share"]
    ev["pre_att"] = ev["cell_att"] * ev["baseline_share"]

    # Revenue basis identical to the Impact tab.
    ev["post_rev"] = ev["post_succ"] * ev["avg_ticket"]
    ev["pre_rev"] = ev["pre_succ"] * ev["avg_ticket"]
    ev["rev_delta"] = ev["post_rev"] - ev["pre_rev"]
    
    # Share change in percentage points.
    ev["share_delta_pp"] = (ev["share"] - ev["baseline_share"]) * 100.0
    return ev



# ---- HAS_PLOTLY probe (is plotly available?) ----
try:
    import plotly.express  # noqa: F401 — availability probe only
    HAS_PLOTLY = True
except Exception:  # noqa: BLE001
    HAS_PLOTLY = False


# ---- more helpers moved from streamlit_app.py ----

# [FN-245]
def _locked_panel(step_html):
    """Calm centered placeholder for a results tab that has no run behind it yet."""
    st.markdown(
        "<div style='text-align:center; padding:3.6rem 1rem; color:var(--tav-muted);'>"
        "<div style='font-size:2.4rem; line-height:1; margin-bottom:0.6rem;'>🔒</div>"
        "<div style='font-size:1.05rem; font-weight:700; color:var(--tav-ink); "
        "margin-bottom:0.3rem;'>Nothing to show yet</div>"
        f"<div style='font-size:0.9rem;'>{step_html}</div></div>",
        unsafe_allow_html=True)


# [FN-246]
def _split_df_to_xlsx_bytes(rdf):
    """Serialise one split DataFrame to .xlsx bytes for the export ZIP. Primary path uses
    xlsxwriter, which writes large sheets fast and applies the GO LIVE date format ONCE at the
    workbook level (no per-cell number_format loop). Falls back to openpyxl if xlsxwriter is
    unavailable. Module-level so joblib/loky can pickle it for the parallel export writes."""
    import io as _io
    _rdf = rdf.copy()
    if "GO LIVE" in _rdf.columns:
        _rdf["GO LIVE"] = pd.to_datetime(_rdf["GO LIVE"], errors="coerce")
    try:
        _xb = _io.BytesIO()
        with pd.ExcelWriter(_xb, engine="xlsxwriter",
                            datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as _w:
            _rdf.to_excel(_w, index=False, sheet_name="Sheet1")
        return _xb.getvalue()
    except Exception:  # noqa: BLE001
        pass
    # Fallback: openpyxl (retains the per-column date format; only used if xlsxwriter is missing).
    _xb = _io.BytesIO()
    with pd.ExcelWriter(_xb, engine="openpyxl") as _w:
        _rdf.to_excel(_w, index=False, sheet_name="Sheet1")
        _ws = _w.sheets["Sheet1"]
        _hdr = [c.value for c in _ws[1]]
        if "GO LIVE" in _hdr:
            _gi = _hdr.index("GO LIVE") + 1
            for _row in _ws.iter_rows(min_row=2, min_col=_gi, max_col=_gi):
                for _cell in _row:
                    _cell.number_format = "yyyy-mm-dd"
    return _xb.getvalue()
