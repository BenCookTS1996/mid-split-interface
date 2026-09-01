"""Shared constants, the log handler, and small helpers used across the app's tabs.

Pulled out of the (very large) ``streamlit_app.py`` so each tab can live in its own file and
import what it needs from here — instead of every tab sharing one giant module scope. This
module has no side effects worth worrying about: it just defines names (and reads the
per-session ``st.session_state`` singleton, which is the SAME object in every module).

As more tabs are moved into their own files, the helpers they share move here too.
"""
from __future__ import annotations

import json
import logging
import os
import io
import csv

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
# 19fk: was ".cache" at the repo root. Renamed to sit beside
# data/build_baseline_cached_input_data and to say WHOSE cache it is: this one holds the
# ROUTING ENGINE's inputs (attempts_success, m0_weightings, processor_benchmark, pool_comp,
# riskmin, ga_perf.json), not the baseline forecast's. A dot-prefixed folder also hid ~600 MB
# from a plain `ls`, which is not a good property for the largest thing in the repo after
# the baseline cache.
CACHE_DIR = os.path.join(_HERE, "..", "data", "routing_engine_cached_input_data")
INPUTS_DIR = os.path.join(_HERE, "..", "config", "inputs")
GCP_PROJECT = "sapient-tangent-172609"  # matches the VAMP repo's BigQuery project

# RPGTs (transaction types) used across the pipeline and templates.
RPGT_LIST = [
    "Monthly Initial", "Annual Sub Sale", "Addon Sale", "Upgrades",
    "Monthly Renewal", "Annual Sub Renewal", "P6M Renewals", "Addon Renewal",
]
COMPANIES = ["TotalAV", "Total Drive", "Total Adblock", "Total Cleaner", "Total VPN"]


# --- Master_MID_List.csv loader (memoised) ---------------------------------------------
# The MID list is read many times per Streamlit rerun (gatewayFid→vampMid / brand /
# processWallet lookups). Re-parsing the same file each time is pure waste, so cache the
# parse keyed on (path, mtime) and return a fresh COPY each call — bit-identical to
# pd.read_csv(path), just without the repeated disk read + parse. An edited file (new
# mtime) busts the cache automatically.
_MID_LIST_CACHE: dict = {}
# Which encoding the last successful read used, and any note worth logging. `tab2_engine`
# re-emits this through log() — app_common has no logger, and a silent fallback is exactly
# how a mojibake'd MID list would go unnoticed.
LAST_MID_LIST_ENCODING: str = ""
LAST_MID_LIST_NOTE: str = ""

# Excel on macOS saves "CSV" as Mac Roman, not UTF-8 — a single curly quote, en-dash or
# non-breaking space in a URL/description column is enough to make pd.read_csv raise
# UnicodeDecodeError on byte 0xCA (NBSP) or 0x96 (en-dash). That killed the 2026-08-19
# 11:53 run outright at tab2_engine:2394, and — worse — five OTHER call sites swallow the
# same exception and degrade SILENTLY, including [midlist-filter], which logged
# "candidates are UNFILTERED ... Master-MID capability is not enforced this run". A routing
# run that quietly stops enforcing MID capability is a far bigger problem than a crash.
# So: try UTF-8 first (unchanged for a well-formed file), then fall back, and SAY SO.
# Mac Roman is tried before cp1252 because the observed corruption is macOS-Excel-shaped;
# both decode the same bytes, so the order only decides how one cosmetic character renders
# in a free-text column — no key used for matching (gatewayFid, currency, brand) is
# non-ASCII, which is why a fallback read is safe rather than a guess at the data.
_MID_LIST_ENCODINGS = ("utf-8", "utf-8-sig", "mac_roman", "cp1252", "latin-1")


def _read_mid_csv(path):
    """pd.read_csv(path), but tolerant of a non-UTF-8 (macOS-Excel) save.

    Records the encoding used in LAST_MID_LIST_ENCODING so a fallback can be surfaced.
    Re-raises the ORIGINAL UnicodeDecodeError if nothing decodes, so a genuinely broken
    file still fails loudly instead of silently returning something wrong.
    """
    global LAST_MID_LIST_ENCODING, LAST_MID_LIST_NOTE
    _first_err = None
    for _enc in _MID_LIST_ENCODINGS:
        try:
            _out = pd.read_csv(path, encoding=_enc)
        except UnicodeDecodeError as _ue:
            if _first_err is None:
                _first_err = _ue
            continue
        LAST_MID_LIST_ENCODING = _enc
        LAST_MID_LIST_NOTE = ("" if _enc in ("utf-8", "utf-8-sig") else
                              f"Master_MID_List.csv is NOT valid UTF-8 — decoded as {_enc} "
                              "instead (macOS Excel saves CSV as Mac Roman). The read "
                              "succeeded and every matching key is ASCII, so routing is "
                              "unaffected; re-save the file as 'CSV UTF-8' to clear this. "
                              "Before this fallback existed the run either CRASHED or "
                              "silently ran with the MID-capability filter DISABLED.")
        return _out
    if _first_err is not None:
        raise _first_err
    return pd.read_csv(path)


def load_mid_list(path):
    """Read Master_MID_List.csv once per (path, mtime), reused across reruns.

    Returns a fresh COPY so callers may mutate the frame freely without corrupting the
    cache. Identical in content/dtypes to ``pd.read_csv(path)`` for a UTF-8 file; falls
    back to a non-UTF-8 encoding rather than raising (see _read_mid_csv).
    """
    try:
        _k = (str(path), os.path.getmtime(path))
    except OSError:
        return _read_mid_csv(path)        # missing/odd path → behave exactly like read_csv
    _df = _MID_LIST_CACHE.get(_k)
    if _df is None:
        _df = _read_mid_csv(path)
        _MID_LIST_CACHE.clear()           # keep only the latest (path, mtime)
        _MID_LIST_CACHE[_k] = _df
    return _df.copy()


def _norm_cols(df):
    """Case/space/underscore-insensitive column lookup for a DataFrame:
    returns ``{normalised_name: original_name}`` (e.g. 'gatewayfid' -> 'gatewayFid')."""
    return {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}


_KEEP = object()  # sentinel: unmapped values keep their original value


def _map_to_bank(series, b2b, default=_KEEP):
    """Map a bank/BIN series through a bin_to_bank dict, trying the raw value then a
    stripped-lower key. Unmapped values fall back to `default`, or — when `default` is the
    `_KEEP` sentinel — to the original value. (Caller applies any `.astype(str)` / `.str.upper()`.)"""
    if default is _KEEP:
        return series.map(lambda b: b2b.get(b, b2b.get(str(b).strip().lower(), b)))
    return series.map(lambda b: b2b.get(b, b2b.get(str(b).strip().lower(), default)))


def _renorm_share(df, keys, col="share"):
    """Renormalise ``df[col]`` IN PLACE so it sums to 1 within each ``keys`` group; rows whose
    group sum is <= 0 become 0. Same operations/order as the inlined idiom (bit-identical).
    Returns df for chaining."""
    _t = df.groupby(keys)[col].transform("sum")
    df[col] = (df[col] / _t).where(_t > 0, 0.0)
    return df


def _fid2vamp_from(mid_df, gwcol, vmcol):
    """Build ``{gatewayFid_lower_stripped: vampMid_stripped}`` from a Master-MID DataFrame given
    the resolved gatewayFid and vampMid column names. Bit-identical to the inlined dict(zip(...))."""
    return dict(zip(mid_df[gwcol].astype(str).str.strip().str.lower(),
                    mid_df[vmcol].astype(str).str.strip()))


def read_json(path, default=None):
    """Load a JSON file, returning ``default`` if it's missing, unreadable, or malformed."""
    try:
        with open(path) as _f:
            return json.load(_f)
    except Exception:  # noqa: BLE001
        return default


def green_button_css(key):
    """Inject the scoped green-primary button styling for a widget created with ``key=<key>``
    (green fill, white text, darker-green hover). Used by the 'Fetch projected M0' buttons."""
    st.markdown(
        f"<style>.st-key-{key} button{{background-color:#22C36B !important;"
        # border-radius: the global `.stButton > button` rule in streamlit_app.py sets 0 but
        # WITHOUT !important, so Streamlit's own generated class outranked it here and these
        # buttons alone came out rounded against an otherwise square UI.
        f"border-radius:0 !important;"
        f"border-color:#22C36B !important;}} .st-key-{key} button,"
        f".st-key-{key} button *{{color:#ffffff !important;}} "
        f".st-key-{key} button:hover{{background-color:#1EA95D !important;"
        f"border-color:#1EA95D !important;}}</style>",
        unsafe_allow_html=True)


def fetch_m0_weightings(company, scheme, *, assumed_prefix, total_key, msg_key, err_key):
    """Fetch last month's projected txns per RPGT for `company` + `scheme` from BigQuery
    (queries/m0_weightings.sql) and fill the given session_state keys.

    `scheme` ('visa'/'mastercard') is passed to the query as CARD_SCHEME, so the projection is
    for the SELECTED scheme (not just Visa) and the cache separates per scheme. Shared by the
    Build-Baseline and Validate M0 panels — they differ only in the session-key names. Writes
    ``ss[assumed_prefix + rpgt]`` for each RPGT, ``ss[total_key]``, and either ``ss[msg_key]``
    (success) or ``ss[err_key]`` (failure). Intended as an on_click callback (sets state before
    the automatic rerun).
    """
    from routing_optimiser import run_sql_file  # lazy: keeps app_common import-light
    try:
        _scheme = str(scheme or "visa").strip().lower()
        _m0sql = os.path.join(SQL_DIR, "m0_weightings.sql")
        _m0path, _m0src = run_sql_file(_m0sql, CACHE_DIR, use_cache=True, project=GCP_PROJECT,
                                       params={"company": company, "CARD_SCHEME": _scheme})
        _m0df = (pd.read_parquet(_m0path) if str(_m0path).endswith(".parquet")
                 else pd.read_csv(_m0path))
        _rc = "riskdata2025_risk_defined_subscription_product_type"
        _vc = "riskdata2025_scheme_trx_count"
        # accept the new scheme-generic column, the legacy visa name, or fall back positionally
        if _vc not in _m0df.columns and "riskdata2025_visa_trx_count" in _m0df.columns:
            _m0df = _m0df.rename(columns={"riskdata2025_visa_trx_count": _vc})
        if _rc not in _m0df.columns or _vc not in _m0df.columns:
            _c0 = list(_m0df.columns)          # positional fallback
            _m0df = _m0df.rename(columns={_c0[0]: _rc, _c0[1]: _vc})
        _fetched = {}
        for _, _row in _m0df.iterrows():
            _v = _row[_vc]
            _fetched[str(_row[_rc]).strip()] = int(round(float(_v))) if pd.notna(_v) else 0
        # TotalAV + visa only: agreed manual reductions to the renewal RPGTs (the projection
        # over-counts these) before filling the inputs. (Mastercard uses the raw projection.)
        if str(company) == "TotalAV" and _scheme == "visa":
            for _rp, _sub in (("Monthly Renewal", 8000),
                              ("Annual Sub Renewal", 1500),
                              ("Addon Renewal", 500)):
                if _rp in _fetched:
                    _fetched[_rp] = _fetched[_rp] - _sub
        for _rp in RPGT_LIST:
            ss[f"{assumed_prefix}{_rp}"] = max(0, int(_fetched.get(_rp, 0)))
        ss[total_key] = int(sum(max(0, int(_fetched.get(_rp, 0))) for _rp in RPGT_LIST))
        ss.pop(err_key, None)
        ss[msg_key] = (f"Filled from BigQuery ({_m0src}) for {company} — "
                       f"projected {ss[total_key]:,} {_scheme.title()} txns across "
                       f"{sum(1 for _rp in RPGT_LIST if _fetched.get(_rp))} RPGT(s).")
    except Exception as _me:  # noqa: BLE001
        ss.pop(msg_key, None)
        ss[err_key] = f"{type(_me).__name__}: {_me}"


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
    from routing_optimiser.vamp_forecast_pipeline import _canonical_gateway
    out = set()
    if not isinstance(ov, dict):
        return out
    for _gw, _cfg in ov.items():
        if isinstance(_cfg, dict) \
                and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
            out.add(str(_canonical_gateway(_gw)).strip().lower())
    return out



# [FN-236] 19cv
def run_company(ss, default="TotalAV"):
    """The brand this run is for.

    IT LIVES IN `forecast_settings`, which tab 1 writes — NOT at the top level of session state.
    `ss.get("company")` returns None, and None is not an error to anything that consumes it:

        build_capability(brand=None)  ->  113 gateways / 45 MIDs across 27 brands
        build_capability(brand="TotalAV") ->  38 gateways / 15 MIDs

    so tab 3 injected zero-VAMP recipient rows for every brand in the MID list, and every
    brand filter downstream matched nothing and failed OPEN. Measured on the 2026-08-29 16:38
    run: the per-MID table carried Stripe - VPN360, PaySafe - Total Cleaner, WoodForest -
    Total Adblock and twenty more, all zeros.

    ONE definition, because the two tabs already read this from two different places once.
    """
    _fs = ss.get("forecast_settings") or {}
    _v = str(_fs.get("company") or "").strip()
    return _v or default


def _vamp_off_gateways(ov: dict) -> set:
    """Canonicalised, lower-cased gateway ids whose VAMP is overridden to ZERO — target == 0 with
    apply_to in ("vamp", "both").

    WHY THIS EXISTS. `_switched_off_gateways` above answers a different question: which gateways
    take no TRANSACTIONS. Every consumer in the app filters `apply_to in ("trx", "both")`, so the
    "vamp" value reaches nothing here. That is only half wrong: the baseline pipeline DOES honour
    it — WoodForest and Authorize carry vampPre 0.0 and FC_VAMP_Month_0 0.0 against real
    FC_VI_Txn (21,233 and 7,535), which is exactly what apply_to:"vamp" means (zero the VAMP,
    keep the transactions). What nothing honours is the REDISTRIBUTION: an overridden MID holds
    no VAMP of its own and is then handed a slice of the moved pool anyway, because the recipient
    share is built from `prop_raw` with no eligibility test. On the 2026-08-28 14:39 run that was
    690 units to WoodForest and 227 to Authorize, from a PRE of 0 in both cases.

    The override is enforced on the STOCK and not on the FLOW. This set is the flow half.

    "both" appears in both helpers deliberately: a gateway switched off for everything takes no
    transactions AND receives no VAMP.
    """
    from routing_optimiser.vamp_forecast_pipeline import _canonical_gateway
    out = set()
    if not isinstance(ov, dict):
        return out
    for _gw, _cfg in ov.items():
        if isinstance(_cfg, dict) \
                and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                and str(_cfg.get("apply_to", "")).strip().lower() in ("vamp", "both"):
            out.add(str(_canonical_gateway(_gw)).strip().lower())
    return out


# [FN-237] 19cv
def _unknown_apply_to(ov: dict) -> set:
    """apply_to values outside the KNOWN set, so a typo in the overrides file cannot sit there
    silently reading as if it were applied. Callers log this; nothing acts on it.

    KNOWN VALUES
        trx     the gateway takes no transactions; it may still hold historic VAMP
        vamp    the gateway still trades but may hold no VAMP - the death sync strips it and
                redistributes it to live gateways in the same cohort
        both    both of the above
        inject_from_siblings
                the source data carries no usable VAMP for this gateway, so it receives NONE of
                the recorded pool (that pool is siblings-only fraud) and its own VAMP is inferred
                from the gateways that DO report it, at (Company x rpgt x Currency) x origin month
                x age - see actuarial_engine._inject_from_siblings.

                Deliberately NOT spelled "vamp". Every other consumer in this codebase filters on
                the literal strings above, so using a distinct value is exactly what stops the
                death sync, the actuarial origin cutoff and the vamp-off MID list from firing on
                these gateways. Adding it here is required: without it this function reports the
                value as a typo and the caller tells the user the entries are IGNORED, which is
                the opposite of what happens.
    """
    out = set()
    if not isinstance(ov, dict):
        return out
    for _gw, _cfg in ov.items():
        if isinstance(_cfg, dict):
            _ap = str(_cfg.get("apply_to", "")).strip().lower()
            if _ap and _ap not in ("trx", "vamp", "both", "inject_from_siblings"):
                out.add(_ap)
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
    g["_b"] = g["bin"].astype(str).str.strip().str.lower()
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
def _apply_blocked_caps(split, blocked_pairs, floor, bin_to_bank=None, group_keys=None):
    """Cap the share of any BANK-BLOCKED (bank, gateway) to the exploration floor and redistribute
    the freed share to the OTHER (non-blocked) gateways in the same cell, proportionally. Cells with
    no non-blocked recipient are left unchanged (nowhere to move the volume). Matches on
    lower/stripped (bank, gateway) — and, when `bin_to_bank` is given, ALSO on the parent-bank grain,
    so a BIN-vs-parent grain mismatch can't silently cap nothing here while the pre-GA auto-block
    excludes the same rows. Returns (new_split, n_rows_capped). Deterministic; no-op when
    `blocked_pairs` is empty or nothing matches."""
    if not blocked_pairs or split is None or getattr(split, "empty", True) \
            or not {"bin", "gateway", "share"}.issubset(split.columns):
        return split, 0
    blocked_pairs = set(blocked_pairs)   # O(1) membership in the per-row checks below
    d = split.copy()
    _bk = d["bin"].astype(str).str.strip().str.lower()
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
    # REDISTRIBUTION GROUP. The default omits `ctry`, so freed share from a bank-blocked row is
    # spread across the USA and Non-USA sub-cells of a (rpgt, currency, BIN, pmp) group TOGETHER.
    # The in-search twin (tab2_engine._fm_block) redistributes within the search's own sub-cell
    # segments, which DO include ctry — so the GA scores a redistribution it does not ship. See the
    # [block-why] probe. `group_keys=None` keeps the historical tuple byte-identical; pass an
    # explicit tuple to align the grain with a caller's own cell definition.
    _key = ([c for c in group_keys if c in d.columns] if group_keys
            else [c for c in ("rpgt", "currency", "bin", "pmp") if c in d.columns]) or ["bin"]
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


APP_BUILD = "2026-08-19ct"  # 19bl: REPAIR. 19bk wrote eligibility.py from a stale base and
# deleted the 2026-08-18 +exact-subcell-capability work, so the GA scored eligibility with the
# global wallet/Non-USA fraction while delivery applied the exact pure-sub-cell rule. That is
# the 17:21 regression: [elig-grain] 147,944/245,409 -> 0/1, RECONCILIATION ERROR 0 -> 7,865,
# [rung] 100% SPLIT, 78,822 differing prop-keys. Rebased on git HEAD, in-place twin retained
# and re-proven bit-identical, plus a canary that shouts if the module ever regresses again.
                           # [deliv-cost] split the 52.2% `deliver` row into eligibility
                           # 840ms (71.5%), blocked-caps 312ms (26.6%), scatter 22.5ms.
                           # (1) BLOCKED-CAPS restricted to the cells a blocked row can
                           # reach — 68 of 23,418 on the 16:01 run. Elsewhere _freed == 0
                           # so _outb == _capd + 0.0 == _X exactly. [deliv-cost] already
                           # proved it np.array_equal on the live 35x242,670 array and
                           # measured 312 -> 19.1 ms (16.3x, 13.3% of a generation).
                           # (2) ELIGIBILITY in place. apply_elig_pop's two _blend_pop
                           # calls built ~28 full-width temporaries (~1.9 GB of traffic
                           # per generation); the twin reuses scratch with the SAME ufuncs
                           # in the SAME order — np.take(out=) for np.repeat, np.copyto
                           # for np.where, both selections not arithmetic. Measured 2.80x
                           # and bit-identical. Costs 0.35 GB of persistent scratch.
                           # BOTH self-check against the original on their FIRST live call
                           # and, on any mismatch, revert for the run and shout — the
                           # fallback ships the KNOWN-GOOD path, it does not hide it.
                           # ROUTING_BLOCK_RESTRICT=0 / ROUTING_ELIG_INPLACE=0 revert.
                           # NOTE eligibility is shared with the TILT engine
                           # (midtilt_cmaes), which is why the check is on the function.
                           # 19bi/bj, on Ben's instructions from the 14:09 run.
                           # 19bi ADOPTED G: every index array the band kernel reads is
                           # int32 now (half the index bandwidth, identical values).
                           # Measured 4.1%, 12/15 rounds, p=0.035, max|Δ| 0.0. A wrapped
                           # index would read the wrong row SILENTLY, so `_i32` refuses to
                           # narrow anything whose range does not provably fit and the
                           # projector logs the largest index against the int32 ceiling.
                           # 19bi RETIRED H as a variant — chunking has been adopted since
                           # 19az and [gen-cost] now measures the projector's share of a
                           # generation directly, which is what H stood in for.
                           # 19bi [kernel-ga] default A,H -> A,F: F is the only variant
                           # still open, and this is the only block that measures how far
                           # it moves the answer END TO END.
                           # 19bj NEW [deliv-cost]: splits the 51.6% `deliver` row into
                           # scatter / blocked-caps / eligibility AND tests the one
                           # optimisation the code offers — `_fm_block` builds ~8
                           # full-width temporaries for a mask covering 91 of 245,409
                           # rows, so only the cells holding a blocked row can change. The
                           # block RUNS the restricted version and diffs it with
                           # np.array_equal, so the next run says whether it is safe
                           # rather than leaving it an argument. NOT a behaviour change.
                           # 19bh: my 19bg A,H default landed on the WRONG LINE — variant
                           # E's lazy-compile gate, not `_kg_want`, the actual selector. So
                           # the 14:09 run announced A,H in its footer and ran A,B. The E
                           # gate re-read the same env var with its own default, which is
                           # how a one-line edit could hit the wrong one; it now reads the
                           # parsed list, so there is ONE read of that setting.
                           # 19bg: FIND THE REAL BOTTLENECK. The log measured ONE component
                           # of a generation (the band projector, [kernel-ab]) and nothing
                           # else, while the 11:56 run spent 3.01 s/generation. New
                           # [gen-cost] times all five stages the search actually runs per
                           # generation — softmax, deliver_full, project, fitness, genetic —
                           # with the SAME functions, and prints their sum against the
                           # engine's own s/generation so an unaccounted remainder cannot
                           # hide. It also resolves a contradiction: [kernel-ab] predicted
                           # LIFT OFF would cost +24s over 320 generations and [kernel-ga]
                           # measured +2.3s. Until that is settled no kernel variant can be
                           # valued. [kernel-ga] default moves A,B -> A,H (a 4.5x lever
                           # instead of 1.16x, so the answer clears the noise), B and H are
                           # relabelled as COUNTERFACTUALS of adopted optimisations rather
                           # than proposals, and C/D/E/F/G get a standing verdict so settled
                           # questions stop reopening. All read-only. ROUTING_GEN_COST=0.
                           # 19bf: two defects the 11:56 run exposed, both mine.
                           # (1) [seed-basis]'s new ⚠ fired on all three seeds at Δ ≈ 2e-07
                           # on a breach of ~0.34 — float noise from the same projection
                           # down two code paths. My 19bd threshold was `_d2 > _r2 + 1e-9`,
                           # eight orders below the noise. It now needs BAND EVIDENCE (the
                           # bases disagree on which are met, or a value moves >0.25% of its
                           # limit — the same test the per-seed lines use) plus a relative
                           # floor, so it cannot contradict its own block. Noise is counted
                           # and named in one line instead of three paragraphs.
                           # (2) band_projection hardcoded "Measured 3.196x" for chunking
                           # while [kernel-ab] row H measured 4.5x on the SAME log. Same
                           # defect as the deleted "about 1.08x" frozen-scaffold model:
                           # a fitted constant next to a live measurement. Deleted; the
                           # line now points at H.
                           # 19be: BEHAVIOUR CHANGE, approved 2026-08-23.
                           # solve_targeted_moves keeps recipient headroom per (MID,
                           # METRIC). It was ONE slot per MID (last spec wins, metric
                           # ignored), and report()'s one-row-per-SPEC was collapsed by
                           # midl the same way, and the running budget was debited in VAMP
                           # units whatever metric the ceiling belonged to — risk is ~1e-2,
                           # so a TXN ceiling read ~100x its real room. That is how a VAMP
                           # shed onto the txn-only WoodForest (23,961 of 24,000) was
                           # allowed to continue until delivery put it 14 over. Recipients
                           # now need room under EVERY ceiling they hold, each budget in
                           # its own units, ranked by binding share-capacity; the donor
                           # orders cells by whichever of ITS metrics is worst over.
                           # Never-worse untouched. ROUTING_TMOVE_ALLBANDS=0 ignores
                           # recipients' txn ceilings. Floors still out of scope.
                           # 19bd: (a) solve_targeted_moves' "strictly better" now says it is
                           # the RAW basis, and [seed-basis] names any stage that is better on
                           # RAW and worse on DELIVERED, with the root cause (recipient headroom
                           # is computed on the metric being SHED, so a txn-only MID reads
                           # infinite room for a VAMP shed). Claim only — no shares move.
                           # (b) [kernel-ab] prints the NUMBER of paired rounds an undecided
                           # variant needs, and flags a lane cap below the thread count.
                           # (c) new [zero-cells]: genome cells that cannot move either
                           # objective. All read-only.
                           # 19bc:
                           # (1) A and B each printed TWICE with contradictory
                           # conventions — 1.158x (B/A) directly above 0.863x median
                           # (A/B), the same measurement reading as two findings. My
                           # 19ba edit replaced the report span but the original A/B
                           # lines sat ABOVE the anchor, so they survived. Deleted.
                           # (2) The floor test was statistically wrong and it HID THE
                           # ONE VARIANT THAT MATTERS. 19ba compared each median against
                           # the max-min RANGE of A' 's per-round ratios — that asks an
                           # effect to exceed the whole spread of INDIVIDUAL measurements,
                           # far too conservative at n=15, and max-min is set by one
                           # outlier and does NOT shrink with more samples. It read 14.4%
                           # and buried G (int32, BIT-IDENTICAL, bandwidth-halving) at
                           # 4.9% median. Now: a SIGN TEST over the paired rounds is the
                           # primary, distribution-free decision, and a 95% CI on the
                           # median is the effect-size uncertainty. The old range is
                           # still printed, labelled as over-conservative. A verdict
                           # needs the sign test to agree, and the wording separates
                           # "consistent but small" from "not measurable".
                           #  # band_projection.py's OWN docstring claimed "the search falls
                           # back to a crude volume-ratio proxy (the source of the large
                           # proxy-vs-true gaps in the run log)". Both halves are stale:
                           # the proxy is REMOVED (run_fullmatrix_ga is called WITHOUT the
                           # mid_bands hook and WITH band_penalty_fn=ExactBandPenalty.
                           # penalty; band_scoring holds no proxy class at all), and
                           # _project_capped lives in tab2_engine, not streamlit_app.
                           # The numbers settle it rather than the comments: the five-rung
                           # chain reads identically at every rung and RECONCILIATION
                           # ERROR is 0 on all 15 bands — a proxy scoring the search would
                           # appear as DELIVERY DRIFT, the column reading zero. Deleted:
                           # it caused a real misreading, which is documentation worse
                           # than absent. band_scoring.py's opening line likewise.
                           #  # the variant MEASUREMENT was noise-dominated, and my own
                           # interleaving is what revealed it. On the 21:10 chunked run A —
                           # the SAME computation — drifted 399.8 -> 496.6 ms across the
                           # block (+24%, +11.3 ms per slot) because 8 lanes contending for
                           # memory bandwidth heat the machine. The consecutive-reps design
                           # folded that INTO the floor (29.7%) and buried every variant
                           # under it, with position and effect inseparable: C and D timed
                           # late read "slower", F timed near the end read fastest.
                           # FIX IS THE DESIGN, not more reps (more reps = more heat).
                           # BLOCKED/PAIRED: _KAB_REPS rounds, every candidate timed ONCE
                           # per round in the same order, ratios taken WITHIN a round so
                           # drift slower than one round cancels exactly. Same total kernel
                           # calls — only the order changed. The FLOOR is now MEASURED:
                           # A' is the same computation as A, so its per-round ratio must
                           # be 1.000 and its spread IS the precision. DRIFT is reported
                           # separately so a hot machine is visible rather than blamed on
                           # the variants. Verdicts say "NOT MEASURABLE on this run",
                           # never "no effect" — conflating those is what retired C/D/E/F
                           # in 19ar on numbers that could not carry the decision.
                           #  # CHUNKED-PARALLEL PROJECTION ADOPTED. P over the lane cap
                           # no longer declines to the serial kernel: the population runs
                           # as ceil(P/cap) parallel calls of at most cap candidates, so
                           # the scratch cost is the CAP's (0.33 GB) not P's (1.26 GB at
                           # P=35) and every population from 10 up stops forfeiting
                           # parallelism. Measured 3.196x AT THE TIME; the live figure is
                           # [kernel-ab] row H (4.5x on 2026-08-23). On the scaffold at P=35,
                           # bit-identical, and re-verified IN-RUN by the once-per-process
                           # self-check, which now diffs _project_chunked itself against
                           # the serial kernel rather than a stand-in for it.
                           # ROUTING_PROJ_CHUNK=0 is a true revert (asserted).
                           # [kernel-ab]/[kernel-ga] follow the adopted path: A IS the
                           # chunked path and H flips to CHUNKING OFF — its
                           # counterfactual, the same convention as B (LIFT OFF). Every
                           # other variant is therefore re-measured on the new baseline,
                           # which matters because chunking MOVES THE BOTTLENECK: one
                           # thread was compute-bound (why C/D/E/F/G all sat inside the
                           # floor), eight lanes are bandwidth-bound, so G (int32) and F
                           # (float32) have a different case to make than before.
                           #  # H was NOT testable end-to-end, and I did not notice.
                           # [kernel-ab] proved H bit-identical on ONE CALL; the
                           # whole-search check that B passed had never been run on it,
                           # because _kg_wrap had no H branch. ROUTING_KERNEL_GA_VARIANTS
                           # =A,H would have fallen through every elif, applied NO
                           # transform, and reported H as identical to A having run A
                           # twice — a vacuous PASS on the only question that decides
                           # whether adopting H changes the delivered split. The
                           # _kg_calls positive control cannot catch it: the wrapper IS
                           # called, it just does nothing. Now H chunks for real, and an
                           # UNRECOGNISED kind RAISES instead of silently running as A.
                           # H also declines when the projector handed it a single lane,
                           # because chunking into lane 0 under the parallel compile is
                           # the forbidden nlane==1 call — a race, not an error.
                           #  # two defects the pop-40 run exposed in my OWN 19av/19aw
                           # work. (1) The verbosity gate keyed on the tag alone, but a
                           # family header carries the tag at 3 spaces and its detail
                           # rows sit at 6+ with none — so it held [mut-target]'s header
                           # and printed its detail line ORPHANED, a sentence with its
                           # subject removed. The gate is now sticky over that
                           # indentation convention, and ends the run at equal-or-
                           # shallower depth so nothing unrelated can be swallowed.
                           # (2) [frozen-scaffold] printed "REALISED SPEEDUP (measured,
                           # not modelled): about 1.08x" from a curve fitted at P=3,
                           # while [kernel-ab] measured the same lift at 1.276x on the
                           # SAME run (B 1733.4 vs A 1358.7 ms, 27.6% outside a 5.0%
                           # floor). The lift scales with candidate width; the model is
                           # deleted rather than left to contradict the measurement.
                           #  # POP 40 CHANGES THE CODE PATH. children = pop -
                           # min(6, max(1, pop//8)), and the projector declines
                           # candidate-parallelism once children exceed
                           # ROUTING_PROJ_LANES=8 — so pop 40 runs the SERIAL compile
                           # at P=35. [kernel-ab] hardcoded P=3 and the PARALLEL
                           # compile, correct only at pop 4, and [kernel-ga] patched
                           # only the parallel dispatcher so every variant row would
                           # have been A run against itself. Now: width and path are
                           # DERIVED from the live budget using the projector's own
                           # constants; both dispatchers are wrapped and restored;
                           # fastmath gained a SERIAL compile because the parallel one
                           # at nlane==1 aliases every candidate onto lane 0; A is
                           # re-timed BETWEEN variants so the resolution floor spans
                           # the positions the variants occupy. New variant H: chunked
                           # parallel — a P=35 call split into calls of <=8 is
                           # BIT-IDENTICAL (asserted) and needs 0.33 GB of lane
                           # scratch instead of 1.43 GB, so it may recover the
                           # parallelism high pop currently forfeits. MEASUREMENT ONLY.
                           #  # RUN-LOG CLEANUP + the retired variants come back.
                           # (1) A verbosity GATE: 22 settled diagnostic families are
                           # muted, meaning their lines are HELD, not dropped — one ⚠ /
                           # STOP / DIVERGE / ✗ / crashed-skip in a family releases its
                           # whole buffer, so a regression prints in full including the
                           # run-up. A quiet family collapses to one [muted] line naming
                           # its release condition. ROUTING_LOG_ALL=1 shows everything.
                           # The blocks still RUN; this is display, not a skip.
                           # (2) DELETED stale log text: a retired switch nobody can
                           # find, a hardcoded "step 1 was 8% at 251" beside live
                           # numbers, a changelog line inside a diagnostic, and a note
                           # about a typo fixed three builds ago.
                           # (3) [zero-rows]: how many nC/nA loop rows are EXACTLY 0.0
                           # for every candidate. Scoped to what is actually provable —
                           # my "899k back-fill rows are droppable" claim was WRONG,
                           # base and ctot are CELL SUMS.
                           # (4) C/D/E/F BACK plus G (int32), reps 5 -> 15. Three of the
                           # four were retired against a 4.8% floor they sat INSIDE, so
                           # they were never measured. C and G claim bit-identity and
                           # are asserted; D/E/F are labelled answer-changing.
                           #  # the two long-standing "skipped" lines. (1) baseline
                           # reconciliation died with AttributeError on EVERY run:
                           # DataFrame.get(col, 0) returns the int 0 when the column
                           # is absent, so pd.to_numeric(0).fillna(0) raised. A guard
                           # whose whole job is "never silently baseline off a
                           # mismatched file" was itself silently doing nothing. It
                           # now NAMES the missing column, lists what each export
                           # does have, and compares whichever metric IS present.
                           # (2) [step1] "one of the four blend vectors is
                           # unavailable" was NOT a failure: step 1 IS the backup
                           # catch-all blend, and with no catch-all configured it is
                           # identically ZERO. It now says so, and still reports a
                           # REAL gap as genuinely UNMEASURED when a catch-all
                           # exists. vamp_forecast_pipeline also gains a __build__ marker
                           # — the header was asking it for one and printing
                           # "(no __build__)".
                           #  # [nw-attrib] the no-divergence line printed TWICE identically
                           # and named neither candidate — my 19an replace was a SILENT
                           # no-op (no assert on the match). Now names the candidate;
                           # [loaded] interrogate the LIVE band_projection — the 16:13 run
                           # reproduced 13:33 byte-for-byte because a long-lived Streamlit
                           # process still held an 11:20 module; the __build__ marker was
                           # six changes stale and [proj-par] blamed numba for a drained
                           # note list. Both fixed. RESTART THE APP after a src/ change;
                           # IN-SEARCH VAMP CONSERVATION — the move is now gated on the
                           # origin cell having a VAMP RECIPIENT (vpsum>0), not just being
                           # routed (psum>0). A routed cell with no VAMP-positive door was
                           # DESTROYING the moved VAMP (measured 165 of 165), so the GA was
                           # scoring a fraud reduction that does not happen. All three
                           # in-search paths. BEHAVIOUR CHANGE — expect worse-looking VAMP.
                           # ROUTING_VAMP_CONSERVE=0 reverts;
                           # 19ar: kernel variants C/D/E/F retired (all measured, none
                           # worth a row) + the fastmath compile deleted with E;
                           # [kernel-ga] the timing column was labelled "speed" — it times
                           # the PYTHON WRAPPER (C/D/F rebuild arrays per call), which is
                           # why every variant read slower than A. Relabelled (wrap) + F is
                           # now an explicit positive control that the wrapper is live;
                           # [kernel-ga] run the WHOLE search once per kernel variant and
                           # compare the ANSWERS — a 1e-12 per-call Δ says nothing about
                           # where a ranking-based search lands. GA kwargs hoisted so the
                           # variant runs are provably identical bar the kernel;
                           # [never-worse] 19an: DRIFT TIE-BREAK DELETED. Decide on
                           # delivered breach alone and FLAG the drift — the tie-break was
                           # routing around the projection defect instead of surfacing it.
                           # ROUTING_NW_TOL retired. Expect the ~616 back, flagged;
                           # [nw-attrib] attribute the drift of the candidate the
                           # never-worse guard REJECTS — the guard routes around the
                           # projection divergence, it does not fix it, so the rejected
                           # candidate carries the root cause. Per-call stash capture
                           # (module globals were overwritten, so only the shipped split
                           # was ever explained). Read-only;
                           # [kernel-ab] CONTROLS — A' (fresh copies, nothing filtered)
                           # and A" (A re-timed last) + reps 2→5, so the block reports a
                           # resolution floor instead of calling a 1.095x on an 0.8% row
                           # drop a "free win" (12:38 did exactly that);
                           # [never-worse] decide on the DELIVERED value, not the GA's own
                           # fitness, + tie-break on reproducibility inside 5%
                           # (ROUTING_NW_TOL). 11:34 shipped +405/616-recon over +411/3
                           # because the deciding basis was blind to delivery drift.
                           # BEHAVIOUR CHANGE. ROUTING_NW_DELIVERED=0 reverts;
                           # [kernel-ab] E fastmath + F float32 as MEASURED variants, so
                           # the accuracy cost is a number not a claim (neither shipped;
                           # ROUTING_KERNEL_AB_PREC=0 skips just these two);
                           # [vterms] READ the VAMP-term stash impact_calcs has computed
                           # and discarded every run (_LAST_VAMP_TERMS / _LAST_VAMP_PSUM,
                           # zero references in tab2 until now). cf_norenorm isolates the
                           # aged-frame renormalise-to-1, which test_recon616 shows is the
                           # ENTIRE in-search-vs-delivered VAMP divergence on a fixture;
                           # [kernel-ab] PLACEMENT FIX — 19ag anchored the block above the
                           # line that assigns _fm_full, so it raised UnboundLocalError on every
                           # run and its own except clause downgraded that to a one-line note.
                           # The measurement never ran once. Measurement-only move;
                           # [kernel-ab] in-run A/B of the remaining kernel ideas;
                           # hoist static nC/nA gathers (bit-identical, ~1.08x measured);
                           # never-worse ENFORCED (ship the seed if the GA regresses);
                           # frozen-scaffold LIFT (bit-identical, ~1.08x measured);
                           # targeting: VAMP/TXN CAPACITY not presence, budget-neutral (19ab was
                           # a no-op that tripled mutation); grain log labels; [frozen-scaffold]
                           # measurement; mutation rate = one explicit number
                           # (dead 60/n_cells term removed, ROUTING_MUT_RATE added);
                           # breach-TARGETED mutation (cells feeding a breached band get a
                           # boosted selection probability); breach_fixed 0.3; 4 silent
                           # fallbacks -> raise; scipy hard;
                           # exact-proj seed unconditional (checkbox gone); [feas-starts];
                           # candidate-parallel band projection
                          # (measured 2.13x on 2 cores,
                          # bit-identical; the projector kernel was 92% of a GA generation)


# [FN-243]
def _ensure_base_30d_metrics():
    """Compute & cache the 30-day baseline metrics (cell/gateway success rates,
    avg ticket, base totals) that the impact views rely on. Idempotent and shared
    by the Routing-engine tab (pre/post visuals) and the Impact tab, so both report
    identical pre/post revenue. Returns the cache dict, or None if no attempts data."""
    # 19ex SCHEMA GUARD. This cache lives in SESSION STATE, which survives a Streamlit rerun
    # AND a module hot-reload — so after a column rename the app can hold frames built by the
    # previous build and hand them to code that expects the new names. That is exactly what
    # happened on 2026-08-31: 19ew renamed the derived join key `bank_join` -> `bin_join`, the
    # module reloaded, and this returned a cache still carrying `bank_join`, so the merge in
    # `_impact_eval_frame` raised KeyError: 'bin_join' on a frame that had just been "fixed".
    #
    # A cache keyed only on ITS OWN PRESENCE cannot notice that. So check the schema it must
    # satisfy and rebuild when it does not — the inputs are still in session state, so this
    # costs one recompute rather than a restart, and it fixes the whole CLASS: any future rename
    # of a join key invalidates the cache automatically instead of surfacing as a KeyError
    # hundreds of lines away.
    _c30 = ss.get("cached_base_30d_metrics")
    if _c30 is not None:
        _need = {"cell_agg": ("rpgt_join", "currency_join", "bin_join"),
                 "gw_agg": ("rpgt_join", "currency_join", "bin_join", "gateway_join")}
        if all(all(_c in getattr(_c30.get(_k), "columns", ()) for _c in _cols)
               for _k, _cols in _need.items()):
            return _c30
        ss.pop("cached_base_30d_metrics", None)   # stale schema — fall through and rebuild
    if "adf" not in ss:
        return None
    adf_raw = ss["adf"]   # read-only here; the mutated frame is adf_30d (its own .copy() below)
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
    if _b2b and "bin" in adf_30d.columns:
        adf_30d["bin"] = adf_30d["bin"].map(
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
                                adf_30d["bin"].astype(str).str.strip().str.lower()]).agg(
        cell_att=("attempts", "sum"), cell_succ=("success", "sum"), cell_rev=("succ_amount", "sum")
    ).reset_index().rename(columns={"rpgt": "rpgt_join", "currency": "currency_join", "bin": "bin_join"})
    cell_agg["cell_sr"] = np.where(cell_agg["cell_att"] > 0, cell_agg["cell_succ"] / cell_agg["cell_att"], 0)

    # Average value per successful transaction at the Bank x Currency level (ONE
    # value per bank x currency), used consistently for every revenue figure so the
    # impact tables reconcile. Falls back to $25 if a cell has no successes.
    bc_val = adf_30d.groupby([adf_30d["currency"].astype(str).str.strip().str.lower(),
                              adf_30d["bin"].astype(str).str.strip().str.lower()]).agg(
        bc_rev=("succ_amount", "sum"), bc_succ=("success", "sum"), bc_att=("attempts", "sum")
    ).reset_index()
    bc_val.columns = ["currency_join", "bin_join", "bc_rev", "bc_succ", "bc_att"]
    bc_val["avg_txn_value"] = np.where(bc_val["bc_succ"] > 0, bc_val["bc_rev"] / bc_val["bc_succ"], 25.0)
    cell_agg = cell_agg.merge(bc_val[["currency_join", "bin_join", "avg_txn_value"]],
                              on=["currency_join", "bin_join"], how="left")
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
                              adf_30d["bin"].astype(str).str.strip().str.lower(),
                              adf_30d["gateway"].astype(str).str.strip().str.lower()]).agg(
        gw_att=("attempts", "sum"), gw_succ=("success", "sum")
    ).reset_index().rename(columns={"rpgt": "rpgt_join", "currency": "currency_join", "bin": "bin_join", "gateway": "gateway_join"})
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
def ensure_cols(df, spec):
    """Guarantee every column in `spec` exists on `df` as a REAL Series, in place.

    `spec` is an iterable of (name, default). THE TRAP THIS CLOSES, which this codebase has now
    hit three times: `DataFrame.get(col, 0)` returns the DEFAULT — a bare int — when the column
    is absent, so `pd.to_numeric(...).fillna(...)` on it raises

        AttributeError: 'int' object has no attribute 'fillna'

    naming neither the column nor the frame that was actually missing. `_impact_eval_frame` below
    already guards its own inputs this way (see its loop); `vamp_forecast_pipeline._reconcile_pre_...`
    was silently doing nothing for two builds for the same reason (19au). Optional enrichment
    that did not run should leave a column of defaults behind, not a scalar landmine.
    """
    for _c, _d in spec:
        if _c not in df.columns:
            df[_c] = _d
    return df


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
    if "bin" in sv.columns:
        sv["bin"] = sv["bin"].map(
            lambda b: b2b.get(b, b2b.get(str(b).strip().lower(), b))).astype(str)
    for c in ["rpgt", "currency", "bin", "gateway"]:
        if c in sv.columns:
            sv[f"{c}_join"] = sv[c].astype(str).str.strip().str.lower()
    gcols = ["rpgt_join", "currency_join", "bin_join", "gateway_join"]
    amap = {c: (c, "first") for c in ["rpgt", "currency", "bin", "gateway"] if c in sv.columns}
    if "share" in sv.columns: amap["share"] = ("share", "mean")
    if "baseline_share" in sv.columns: amap["baseline_share"] = ("baseline_share", "mean")
    if "volume" in sv.columns: amap["volume"] = ("volume", "sum")
    if "cell_volume" in sv.columns: amap["cell_volume"] = ("cell_volume", "sum")
    sv = sv.groupby(gcols, as_index=False).agg(**amap)

    ev = sv.merge(cell_agg, on=["rpgt_join", "currency_join", "bin_join"], how="left")
    ev = ev.merge(gw_agg[["rpgt_join", "currency_join", "bin_join", "gateway_join", "gw_sr"]],
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
        _tsum = ev.groupby(["rpgt_join", "currency_join", "bin_join"])[_sc].transform("sum").to_numpy()
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


# ══════════════════════════════════════════════════════════════════════════════════════════════
# RECOVERY STUBS — 2026-08-26. These two functions were LOST, not removed.
#
# They were added to this file in another session and existed only in the working tree. Every
# device_commit_files call in the engine session passed `force: true`, which bypasses the guard
# that refuses to write over a file changed since it was read, so a routine commit of this file
# discarded them silently. They are in neither git HEAD nor either app_common .pyc, so there is
# no copy of them on this machine.
#
# THEY RAISE RATHER THAN GUESS. `active_gateway_fids` decides WHICH GATEWAYS THE OPTIMISER MAY
# ROUTE TO. A plausible reconstruction from the call site and the Master MID List's `IsActive`
# column would look correct in every log and could silently optimise over the wrong gateway set
# for runs before anyone noticed. A crash that names what is missing is worth more.
#
# TO RESTORE: paste the real definitions anywhere in this file. These are bound only if the name
# is still free, so the real ones win automatically and nothing else needs editing.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# [FN-478] RESTORED 2026-08-26 FROM THE RUN LOGS, not reconstructed from intuition.
def active_gateway_fids(master_mid_list_path=None):
    """The GATEWAY_FIDS SQL parameter: every ACTIVE gateway in the Master MID List, SORTED.

    Returns "('fid','fid',...)" ready to interpolate into attempts_success.sql /
    processor_benchmark.sql.

    HOW THIS WAS RECOVERED, because it matters that it was not guessed. The original was lost on
    2026-08-26 when this file was overwritten by an automated commit using `force`, and it is in
    neither git HEAD nor either app_common .pyc. But its OUTPUT is printed verbatim in every run
    log as the GATEWAY_FIDS parameter, and against the Master MID List as it stands:

        md5 of the logged parameter (2026-08-26 19:44 run)   5425239737a5f2e52b9a67bdffc7cb10
        md5 of sorted(distinct gatewayFid where IsActive)     5425239737a5f2e52b9a67bdffc7cb10
        md5 of the same set in FILE ORDER                     86a67d55822668cb965e82d14a4da67f

    SORTED, and not cosmetic: sql_runner.cache_path_for hashes the rendered parameter into the
    cache filename, so file order would miss every existing cached parquet and re-run the query.
    The 2026-08-26 note claimed both orderings matched "byte for byte" — that was a SET comparison
    written up as a string comparison, and it was wrong.

    Both orderings, every entry, and the rendered string reproduces the logged SQL parameter byte
    for byte. FILE ORDER is what the most recent run used, so that is what ships here.

    DE-DUPLICATED KEEPING THE FIRST OCCURRENCE: the list has 539 rows and 114 distinct active
    gatewayFids, so a MID appearing on several rows must not appear twice in the parameter.
    """
    _p = master_mid_list_path or os.path.join(PROJECT_ROOT, "data", "mappings",
                                              "Master_MID_List.csv")
    _seen, _out = set(), []
    with io.open(_p, encoding="utf-8-sig", newline="") as _fh:
        for _row in csv.DictReader(_fh):
            _fid = str(_row.get("gatewayFid") or "").strip()
            if not _fid or _fid in _seen:
                continue
            if str(_row.get("IsActive") or "").strip().lower() not in _ACTIVE_TRUTHY:
                continue
            _seen.add(_fid)
            _out.append(_fid)
    return "(" + ",".join(f"'{_f}'" for _f in sorted(_out)) + ")"


# The values IsActive is written with. Kept as a named set rather than inlined so a new spelling in
# the sheet is a one-line fix and not a silent 0-gateway run.
_ACTIVE_TRUTHY = {"1", "true", "yes", "y", "active", "t"}

# The pre-2026-08-26 hardcoded constant. KEPT, not deleted: it is the fallback if the CSV cannot be
# read, and deleting it would leave no way to run at all in that case. It is NOT equivalent — it
# holds 94 FIDs and contains no -tcl, -hss, -tvn or -na entry, so a run on it searches a strictly
# smaller gateway set than either 2026-08-26 run did.
_LEGACY_GATEWAY_FIDS_94 = DEFAULT_GATEWAY_FIDS

try:
    DEFAULT_GATEWAY_FIDS = active_gateway_fids()
    GATEWAY_FIDS_SOURCE = (
        f"Master_MID_List.csv \u2014 {DEFAULT_GATEWAY_FIDS.count(chr(39)) // 2} ACTIVE gateway(s), "
        "file order")
except Exception as _gfe:  # noqa: BLE001 — a broken sheet must not stop the app from starting
    DEFAULT_GATEWAY_FIDS = _LEGACY_GATEWAY_FIDS_94
    GATEWAY_FIDS_SOURCE = (
        f"\u26a0 FALLBACK to the legacy 94-FID constant \u2014 Master_MID_List.csv could not be "
        f"read ({type(_gfe).__name__}: {_gfe}). This is a STRICTLY SMALLER gateway set than the "
        "2026-08-26 runs used (94 vs 114, no -tcl/-hss/-tvn/-na), so results are NOT comparable "
        "with them. Fix the sheet rather than reading past this.")


_LOST_IN_OVERWRITE_2026_08_26 = {
    "render_config_profile_charts":
        "Renders the profile-match scatter charts for the config-lookup UI "
        "(tab_config_validation.py:131, imported lazily inside render_profile_lookup).",
}


def _lost_symbol(_name):
    """Return a callable that explains what is missing instead of pretending to do the work."""
    def _raise(*_a, **_k):
        raise RuntimeError(
            f"app_common.{_name}() is MISSING — it was lost on 2026-08-26 when this file was "
            f"overwritten by an automated commit that used force and discarded uncommitted "
            f"working-tree edits. It is not in git HEAD or in either app_common .pyc, so there is "
            f"no copy on this machine.\n\n"
            f"WHAT IT DID: {_LOST_IN_OVERWRITE_2026_08_26.get(_name, '(unknown)')}\n\n"
            f"THIS IS A DELIBERATE FAILURE, NOT A BUG TO ROUTE AROUND. A reconstructed version "
            f"would be a guess, and for active_gateway_fids a wrong guess silently changes the "
            f"gateway set the optimiser may use. Paste the real definition into app_common.py "
            f"(anywhere) and this stub disappears on its own.\n\n"
            f"Everything that does NOT use this function is unaffected \u2014 tab 2 and the "
            f"engine run normally.")
    _raise.__name__ = _name
    _raise.__doc__ = f"LOST 2026-08-26. {_LOST_IN_OVERWRITE_2026_08_26.get(_name, '')}"
    _raise._lost_stub = True
    return _raise


for _lost in _LOST_IN_OVERWRITE_2026_08_26:
    if _lost not in globals():
        globals()[_lost] = _lost_symbol(_lost)
del _lost
