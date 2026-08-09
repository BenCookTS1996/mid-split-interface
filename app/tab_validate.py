"""Tab: Validate — run the REAL VAMP pipeline with the exported rules applied.

Lets you point at a folder of exported split templates, auto-merges them with your
renewal/backup rules, runs the vendored VAMP pipeline end-to-end (BigQuery), and
renders the resulting per-vampMid pre/post table in the SAME format as tab 3 — so
you can compare the pipeline's actual forecast against tab 3's projection.

Call render(ss, PROJECT_ROOT, GCP_PROJECT) from inside `with tab_val:`.
"""
from __future__ import annotations

import datetime
import glob
import logging
import os
import shutil
import traceback

import pandas as pd
import streamlit as st

__build__ = "2026-07-22-validate-layout6-attempts-window"

_RULE_GLOBS = ("*.xlsx", "*.xls", "*.csv")


# [FN-389]
def _covered_rpgts(merged_dir: str) -> set:
    """Lower-cased set of RPGTs that HAVE a rule (from the RPGT column of each rule
    file in the merged dir). Used to auto-force-actuals for the RPGTs with no rule."""
    covered = set()
    for pat in _RULE_GLOBS:
        for f in glob.glob(os.path.join(merged_dir, pat)):
            try:
                is_x = f.lower().endswith((".xlsx", ".xls"))
                hdr = (pd.read_excel(f, nrows=0) if is_x else pd.read_csv(f, nrows=0))
                rc = next((c for c in hdr.columns if str(c).strip().lower() == "rpgt"), None)
                if rc is None:
                    continue
                vals = (pd.read_excel(f, usecols=[rc]) if is_x else pd.read_csv(f, usecols=[rc]))[rc]
                covered |= {str(v).strip().lower() for v in vals.dropna().unique() if str(v).strip()}
            except Exception:  # noqa: BLE001
                continue
    return covered


# [FN-390]
def _to_prepost(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the pipeline's mid_level.csv columns onto the tab-3 table names."""
    ren = {}
    for m in range(6):
        ren[f"FC_VAMP_Month_{m}"] = f"VAMP M{m}"
        ren[f"FC_VAMP_Month_{m}_Post"] = f"VAMP Post M{m}"
        ren[f"FC_VI_Txn_Month_{m}"] = f"VI Txn M{m}"
        ren[f"FC_VI_Txn_Month_{m}_Post"] = f"VI Txn Post M{m}"
    vp = df.rename(columns=ren).copy()
    cols = ["vampMid"]
    for m in range(6):
        cols += [f"VAMP M{m}", f"VI Txn M{m}", f"VAMP Post M{m}", f"VI Txn Post M{m}"]
    cols = [c for c in cols if c in vp.columns]
    vp = vp[cols].copy()
    for c in cols:
        if c != "vampMid":
            vp[c] = pd.to_numeric(vp[c], errors="coerce").fillna(0.0)
    if "VAMP M0" in vp.columns:
        vp = vp.sort_values("VAMP M0", ascending=False)
    return vp


# [FN-391]
def _render_prepost_table(vp: pd.DataFrame, fit_content: bool = False, bold: bool = True) -> None:
    """Same red-header / month-spacer / TOTAL-row table tab 3 uses.

    fit_content=True sizes the table to its content (width:auto) instead of stretching
    to 100% — used for the tab-1 PRE-only baseline table, where the 100% stretch left a
    large gap between the vampMid column and the (fewer) month columns.
    """
    _tw = "auto" if fit_content else "100%"
    _dw = "max-content" if fit_content else "100%"
    _disp = "display:inline-block; max-width:100%;" if fit_content else ""
    col_groups, cols = [], ["vampMid"]
    for m in range(6):
        grp = [f"VAMP M{m}", f"VI Txn M{m}", f"VAMP Post M{m}", f"VI Txn Post M{m}"]
        grp = [c for c in grp if c in vp.columns]
        if grp:
            cols.extend(grp)
            col_groups.append(grp)

    total = {"vampMid": "TOTAL"}
    for c in cols:
        if c != "vampMid":
            total[c] = vp[c].sum()
    vp_view = pd.concat([vp[cols], pd.DataFrame([total])], ignore_index=True)

    html = [f'<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow-x:auto; '
            f'width:{_dw}; {_disp} background-color:var(--tav-card); border:1px solid var(--tav-line);">']
    html.append(f'<table style="width:{_tw}; border-collapse:collapse; font-family:inherit; '
                'font-size:0.68rem; line-height:1.1;"><tr>')
    html.append('<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; '
                'text-align:left; position:sticky; left:0; width:1%; white-space:nowrap;">vampMid</th>')
    for gi, grp in enumerate(col_groups):
        for c in grp:
            html.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; '
                        f'text-align:right; white-space:nowrap;">{c}</th>')
        if gi < len(col_groups) - 1:
            html.append('<th style="background-color:var(--tav-card); border:none; width:8px; '
                        'min-width:8px; padding:0;"></th>')
    html.append('</tr>')

    for _, r in vp_view.iterrows():
        is_total = str(r["vampMid"]) == "TOTAL"
        tb = "border-top:2px solid var(--tav-ink);" if is_total else "border-bottom:1px solid var(--tav-line);"
        wt = ("800" if is_total else "600") if bold else ("600" if is_total else "400")
        # Conditional formatting (matches tab 3): a VAMP cell (and its paired VI Txn) is
        # RED when its VAMP rate > 1.5% and VAMP > 1500, AMBER when rate > 1.2% and VAMP > 1200.
        _bgmap = {}
        if not is_total:
            for c in cols:
                if c != "vampMid" and c.startswith("VAMP"):
                    _txn = c.replace("VAMP", "VI Txn")
                    _vv = float(r[c]); _tt = float(r[_txn]) if _txn in r.index else 0.0
                    _rt = (_vv / _tt) if _tt > 0 else 0.0
                    if _rt > 0.015 and _vv > 1500:
                        _bgmap[c] = _bgmap[_txn] = "background-color:rgba(230,55,72,0.30);"
                    elif _rt > 0.012 and _vv > 1200:
                        _bgmap[c] = _bgmap[_txn] = "background-color:rgba(245,158,11,0.38);"
        html.append('<tr>')
        html.append(f'<td style="padding:2px 8px; text-align:left; color:#000000; font-weight:{wt}; '
                    f'{tb} position:sticky; left:0; background-color:var(--tav-card); width:1%; '
                    f'white-space:nowrap;">{r["vampMid"]}</td>')
        for gi, grp in enumerate(col_groups):
            for c in grp:
                ital = "font-style:italic;" if "Post" in c else ""
                _bg = _bgmap.get(c, "")
                html.append(f'<td style="padding:2px 6px; text-align:right; color:#000000; '
                            f'font-weight:{wt}; {ital} {_bg} {tb} white-space:nowrap; width:1%;">{r[c]:,.0f}</td>')
            if gi < len(col_groups) - 1:
                html.append(f'<td style="width:8px; min-width:8px; padding:0; {tb}"></td>')
        html.append('</tr>')
    html.append('</table></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# [FN-392]
def _read_export_manifest(rules_dir: str) -> dict:
    """Read _export_manifest.json (the drift-guard stamp) from an exported rules folder.
    Returns {} if absent/unreadable. Looks in the folder and one level up (the zip root)."""
    import json
    if not rules_dir or not os.path.isdir(rules_dir):
        return {}
    for cand in (rules_dir, os.path.dirname(os.path.normpath(rules_dir))):
        p = os.path.join(cand, "_export_manifest.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return {}
    return {}


# [FN-393]
def _drift_check(ss, rules_dir: str) -> None:
    """Warn if the rule files in `rules_dir` were exported for a DIFFERENT split than the latest
    export in this session — i.e. tab 5 would run rules that no longer match tab 3's projection."""
    man = _read_export_manifest(rules_dir)
    if not man:
        return
    st.markdown(
        f"<span style='color:#808080; font-size:0.8rem;'>Rules built for: dial <b>{man.get('dial')}</b> · "
        f"pools≤<b>{man.get('max_pools')}</b> · engine <b>{man.get('engine')}</b> · "
        f"go-live <b>{man.get('go_live')}</b> · max-share <b>{man.get('max_share')}</b> · "
        f"built {man.get('built_at')}</span>", unsafe_allow_html=True)
    _cur = ss.get("_split_export_sig")
    if _cur is not None and list(_cur) != list(man.get("exp_sig", [])):
        st.warning("⚠ **Split drift:** these rule files were exported for a different split than your "
                   "latest in-session export (dial / pool-target / engine / go-live / max-share "
                   "differ). tab 5 will run them as-is, but they may **not** match tab 3's current "
                   "projection. Re-export + re-download and point this folder at the fresh files to sync.")


# [FN-394]
def _stage_rules(rules_dir: str, merged_dir: str) -> int:
    """Copy every rule file from the exported rules folder into a clean staging dir the
    pipeline reads. RPGTs with no rule file here are auto-routed on actuals (see render)."""
    shutil.rmtree(merged_dir, ignore_errors=True)
    os.makedirs(merged_dir, exist_ok=True)
    _n = 0
    if rules_dir and os.path.isdir(rules_dir):
        for pat in _RULE_GLOBS:
            for f in glob.glob(os.path.join(rules_dir, pat)):
                shutil.copy2(f, os.path.join(merged_dir, os.path.basename(f)))
                _n += 1
    return _n


# [FN-395]
def render(ss, PROJECT_ROOT, GCP_PROJECT):
    # No hard gate: this sub-tab builds its OWN forecast via the pipeline. forecast_settings is
    # kept in sync every rerun by the Build Baseline sub-tab (from its widgets), so it's available
    # here even on a fresh reopen without running/loading a baseline first. Fall back to {} (all
    # fields below have sensible defaults) if it's somehow absent.
    from app_common import SQL_DIR, CACHE_DIR, RPGT_LIST  # shared constants (fn-level import avoids any load-order cycle)
    from routing_optimiser import run_sql_file
    fs = ss.get("forecast_settings") or {}

    _company = str(fs.get("company", "TotalAV"))
    _month = str(fs.get("month_var", "") or "")

    # [FN-396]
    def _d(key, fallback):
        try:
            return pd.to_datetime(fs.get(key)).date() if fs.get(key) else fallback
        except Exception:  # noqa: BLE001
            return fallback

    _today = datetime.date.today()
    _m0 = _d("month_0", _today.replace(day=1))
    _rpgts = list((fs.get("m0_transaction_weightings") or {}).keys())
    _prev_force = {str(r).strip().lower() for r in (fs.get("force_actuals_for") or [])}

    # Everything sits in a FORM: changing any input does NOT rerun/reload the app — the
    # pipeline (and the whole tab) only re-evaluates when you click the submit button.
    st.markdown("""<style>
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
            background:#22C36B !important; border-color:#22C36B !important; border-radius:0 !important; }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] * { color:#fff !important; }
    </style>""", unsafe_allow_html=True)

    # --- M0 Transaction Weightings (mirrors Build Baseline · independent, pre-filled) ----------
    # OUTSIDE the form on purpose: the fetch button uses an on_click callback, and st.form forbids
    # a plain st.button. These inputs pre-fill ONCE from the Build Baseline forecast_settings, then
    # are edited independently here (validate a split against its own assumed M0 without touching
    # the baseline). The form's submit handler reads them back from session_state.
    _scheme = str(fs.get("card_scheme", "visa") or "visa")

    def _v_fetch_m0(_co, _sch):
        try:
            _m0sql = os.path.join(SQL_DIR, "m0_weightings.sql")
            _m0path, _m0src = run_sql_file(_m0sql, CACHE_DIR, use_cache=True,
                                           project=GCP_PROJECT, params={"company": _co})
            _m0df = (pd.read_parquet(_m0path) if str(_m0path).endswith(".parquet")
                     else pd.read_csv(_m0path))
            _rc = "riskdata2025_risk_defined_subscription_product_type"
            _vc = "riskdata2025_visa_trx_count"
            if _rc not in _m0df.columns or _vc not in _m0df.columns:
                _c0 = list(_m0df.columns)          # positional fallback
                _m0df = _m0df.rename(columns={_c0[0]: _rc, _c0[1]: _vc})
            _fetched = {}
            for _, _row in _m0df.iterrows():
                _v = _row[_vc]
                _fetched[str(_row[_rc]).strip()] = int(round(float(_v))) if pd.notna(_v) else 0
            # TotalAV + visa only: same agreed renewal reductions as Build Baseline so the fetched
            # numbers match across the two tabs.
            if str(_co) == "TotalAV" and str(_sch).strip().lower() == "visa":
                for _rp, _sub in (("Monthly Renewal", 8000),
                                  ("Annual Sub Renewal", 1500),
                                  ("Addon Renewal", 500)):
                    if _rp in _fetched:
                        _fetched[_rp] = _fetched[_rp] - _sub
            for _rp in RPGT_LIST:
                ss[f"validate_assumed_{_rp}"] = max(0, int(_fetched.get(_rp, 0)))
            ss["validate_m0_total_key"] = int(sum(max(0, int(_fetched.get(_rp, 0))) for _rp in RPGT_LIST))
            ss.pop("_v_m0_fetch_err", None)
            ss["_v_m0_fetch_msg"] = (f"Filled from BigQuery ({_m0src}) for {_co} — "
                                     f"projected {ss['validate_m0_total_key']:,} Visa txns across "
                                     f"{sum(1 for _rp in RPGT_LIST if _fetched.get(_rp))} RPGT(s).")
        except Exception as _me:  # noqa: BLE001
            ss.pop("_v_m0_fetch_msg", None)
            ss["_v_m0_fetch_err"] = f"{type(_me).__name__}: {_me}"

    with st.container(border=True):
        st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>M0 Transaction Weightings</h5>",
                    unsafe_allow_html=True)
        # Green button, white text (scoped to this button's key).
        st.markdown(
            "<style>.st-key-validate_fetch_m0_btn button{background-color:#22C36B !important;"
            "border-color:#22C36B !important;} .st-key-validate_fetch_m0_btn button,"
            ".st-key-validate_fetch_m0_btn button *{color:#ffffff !important;} "
            ".st-key-validate_fetch_m0_btn button:hover{background-color:#1EA95D !important;"
            "border-color:#1EA95D !important;}</style>",
            unsafe_allow_html=True)
        _vfb1, _vfb2 = st.columns([1, 1.5], vertical_alignment="center")
        _vfb1.button("Fetch projected M0 from BigQuery", key="validate_fetch_m0_btn",
                     on_click=_v_fetch_m0, args=(_company, _scheme),
                     help=f"Query last month's projected Visa transactions per RPGT for "
                          f"{_company} and fill the weightings below.")
        if ss.get("_v_m0_fetch_err"):
            _vfb2.markdown("<span style='color:#e63748; font-size:0.8rem;'>✗ M0 fetch failed: "
                           f"{ss.get('_v_m0_fetch_err')}</span>", unsafe_allow_html=True)
        else:
            _vmsg = ss.get("_v_m0_fetch_msg", "")
            _vfb2.markdown("<span style='color:var(--tav-muted); font-size:0.8rem;'>"
                           f"{('✓ ' + _vmsg) if _vmsg else ''}</span>", unsafe_allow_html=True)
        # Pre-fill ONCE from Build Baseline's current M0 values, then edit independently here.
        _w0 = fs.get("m0_transaction_weightings") or {}
        ss.setdefault("validate_m0_total_key", int(fs.get("m0_total_transactions", 0) or 0))
        for _rp in RPGT_LIST:
            ss.setdefault(f"validate_assumed_{_rp}", int(_w0.get(_rp, 0) or 0))
        _v_alloc = sum(int(ss.get(f"validate_assumed_{_rp}", 0) or 0) for _rp in RPGT_LIST)
        _v_total = int(ss.get("validate_m0_total_key", 0) or 0)
        _vmt1, _vmt2 = st.columns([3, 2], vertical_alignment="center")
        _vmt1.number_input(f"M0 {_company} - {_scheme} - Total", 0, 50_000_000,
                           step=1000, key="validate_m0_total_key",
                           help="Total starting transactions for month 0.")
        if _v_total == _v_alloc:
            _vmt2.markdown("<div style='color:#1D9E75; font-size:0.8rem; font-weight:700;'>"
                           "✓ matches RPGT sum</div>", unsafe_allow_html=True)
        else:
            _vd = _v_alloc - _v_total
            _vmt2.markdown("<div style='color:#e63748; font-size:0.8rem; font-weight:700;'>"
                           f"⚠ RPGT sum {_v_alloc:,} ≠ Total "
                           f"({'+' if _vd > 0 else ''}{_vd:,})</div>", unsafe_allow_html=True)
        _vw_cols = st.columns(2)
        for _i, _rpgt in enumerate(RPGT_LIST):
            _vw_cols[_i % 2].number_input(
                _rpgt, 0, 50_000_000, step=500, key=f"validate_assumed_{_rpgt}",
                help="Assumed month-0 volume for this type.")

    with st.form("validate_form", border=False):
        # Left: '1. Rules Import' then '2. Live actuals' below it.
        # Right: '2. Inputs & Assumptions' then 'Target volumes' below it.
        _L, _R = st.columns(2)
        with _L:
            st.markdown("**1. Rules Import**")
            _ri1, _ri2 = st.columns(2)   # narrow the folder box to ≈ the Start Date width
            rules_dir = _ri1.text_input(
                "Exported rules folder", value=ss.get("validate_rules_dir", ""),
                key="validate_rules_dir",
                help="Folder containing ALL the rule files for this run (your exported split "
                     "templates). Any RPGT with NO rule file here is automatically routed on "
                     "ACTUALS (force-actuals).")
            _drift_check(ss, ss.get("validate_rules_dir", ""))   # flag if these rules ≠ tab-3's split

            st.markdown("**2. Live actuals**")
            v_use_live = st.checkbox(
                "Use Live Actuals",
                value=bool(ss.get("validate_use_live", fs.get("use_live_actuals", False))),
                key="validate_use_live", help="Blend in real recent results. Uses the dates below.")
            _d1, _d2 = st.columns(2)
            v_start = _d1.date_input("Start Date", value=ss.get("validate_actuals_start", _d("start_date", _today)),
                                     key="validate_actuals_start")
            v_end = _d2.date_input("End Date", value=ss.get("validate_actuals_end", _d("end_date", _today)),
                                   key="validate_actuals_end")
            st.markdown("**Force Actuals for**")
            # Two side-by-side columns of tickboxes (alternating left / right).
            _fa1, _fa2 = st.columns(2)
            v_force_manual = []
            for _i, r in enumerate(_rpgts):
                _fac = _fa1 if (_i % 2 == 0) else _fa2
                if _fac.checkbox(r, value=(str(r).strip().lower() in _prev_force),
                                 key=f"validate_fa_{r}"):
                    v_force_manual.append(r)

            # Attempts & success data window (same inputs as the Routing engine tab). Used to
            # pull the success-rate data that populates tab 3's impact views for this split.
            st.markdown("**3. Attempts & success data**")
            _yday = _today - datetime.timedelta(days=1)
            _as1, _as2 = st.columns(2)
            v_att_start = _as1.date_input(
                "Start date",
                value=ss.get("validate_attempts_start", _yday - datetime.timedelta(days=14)),
                key="validate_attempts_start",
                help="Success-rate data window (attempts & successes) used to populate tab 3's "
                     "impact views for the validated split.")
            v_att_end = _as2.date_input(
                "End date", value=ss.get("validate_attempts_end", _yday),
                key="validate_attempts_end")
        with _R:
            st.markdown("**2.  Inputs & Assumptions**")
            # Future Anchor beneath Split Go Live (left); Thermometer beneath t0 lookback (right).
            _gc, _tc = st.columns(2)
            with _gc:
                v_go_live = st.date_input(
                    "Split Go Live date", value=ss.get("validate_go_live", _d("split_go_live_date", _m0)),
                    key="validate_go_live", help="Date the proposed split goes live (drives mid-month pro-rata).")
                v_anchor = st.date_input(
                    "Future Anchor Date", value=ss.get("validate_anchor", _d("future_anchor_date", _today)),
                    key="validate_anchor", help="Date the forecast is anchored to.")
            with _tc:
                v_lookback = st.number_input(
                    "t0 lookback (months)", min_value=1, max_value=12,
                    value=int(ss.get("validate_lookback", int(fs.get("t0_lookback_months", 1) or 1))),
                    key="validate_lookback", help="Actuarial t0_lookback_months.")
                v_thermo = st.number_input(
                    "Thermometer sample (months)", min_value=1, max_value=12,
                    value=int(ss.get("validate_thermo", int(fs.get("thermometer_sample_months", 2) or 2))),
                    key="validate_thermo", help="Actuarial thermometer_sample_months.")
            # M0 target volumes / weightings now live in the 'M0 Transaction Weightings' card ABOVE
            # the form (so the fetch button can use an on_click callback, which st.form forbids).
            # The submit handler reads company_target_volume + per-RPGT from session_state.
            st.markdown("**Target volumes**")
            st.caption("Set in the **M0 Transaction Weightings** card above "
                       "(pre-filled from Build Baseline, editable here, with a BigQuery fetch).")

        run = st.form_submit_button("Run VAMP pipeline with these rules", type="primary")

    if run:
        if v_use_live and v_start > v_end:
            st.error("Start Date must be on or before End Date.")
            return
        if not rules_dir or not os.path.isdir(rules_dir):
            st.error(f"Exported rules folder not found: {rules_dir or '(empty)'}")
            return
        _found = sum(len(glob.glob(os.path.join(rules_dir, p))) for p in _RULE_GLOBS)
        if _found == 0:
            st.error(f"No rule files (.xlsx/.csv) found in: {rules_dir}")
            return

        merged_dir = os.path.join(PROJECT_ROOT, "data", "rules", "_validate", _month, _company)
        nr = _stage_rules(rules_dir, merged_dir)

        try:
            from routing_optimiser.forecast_pipeline import (build_pipeline_config,
                                                             run_vamp_pipeline)
        except Exception as _ie:  # noqa: BLE001
            st.error(f"Could not import the pipeline runner: {type(_ie).__name__}: {_ie}")
            return

        cfg = build_pipeline_config(fs)
        cfg.setdefault("paths", {})
        cfg["paths"]["chunked_files_dir"] = merged_dir            # absolute -> used as-is
        # Separate output dir so this never clobbers tab 1/tab 3's live outputs.
        cfg["paths"]["output_dir"] = os.path.join("data", "outputs", "_validate",
                                                  "{month_var}", "{company}") + os.sep
        cfg["run_settings"]["use_chunked_csv_files"] = True

        # Live actuals from the '4 · Live actuals' inputs above.
        cfg.setdefault("actuarial_settings", {})
        cfg["run_settings"]["use_live_actuals"] = bool(v_use_live)
        if v_use_live:
            cfg["run_settings"]["actuals_start_date"] = str(v_start)
            cfg["run_settings"]["actuals_end_date"] = str(v_end)

        # Force-actuals = the RPGTs ticked above PLUS any RPGT with NO rule file in the folder.
        _universe = {str(r).strip(): str(r).strip().lower() for r in _rpgts}
        _covered = _covered_rpgts(merged_dir)
        _auto_force = [orig for orig, low in _universe.items() if low not in _covered]
        _force = sorted(set(v_force_manual) | set(_auto_force), key=str.lower)
        cfg.setdefault("filters", {})
        cfg["filters"]["force_actuals_for_rpgts"] = _force

        # Forecast-shaping inputs (same as tab 1) so the pipeline forecast is fully specified.
        cfg["run_settings"]["split_go_live_date"] = str(v_go_live)
        cfg["run_settings"]["future_anchor_date"] = str(v_anchor)
        cfg["run_settings"]["blend_future_sheet_rules"] = bool(v_anchor)
        cfg.setdefault("targets", {})
        try:
            # Company M0 total + per-RPGT weightings from the M0 card's session_state keys.
            _allt = {str(_rp).strip(): int(ss.get(f"validate_assumed_{_rp}", 0) or 0) for _rp in RPGT_LIST}
            _ctv = int(ss.get("validate_m0_total_key", int(fs.get("m0_total_transactions", 0) or 0)) or 0)
            cfg["targets"]["company_target_volume"] = int(_ctv)
            if _allt:
                cfg["targets"]["company_rpgt_target_volumes"] = _allt
        except Exception:  # noqa: BLE001
            cfg["targets"]["company_target_volume"] = int(fs.get("m0_total_transactions", 0) or 0)
        cfg["actuarial_settings"]["t0_lookback_months"] = int(v_lookback)
        cfg["actuarial_settings"]["thermometer_sample_months"] = int(v_thermo)

        status = st.status(f"Running the VAMP pipeline with {nr} rule file(s)… "
                           f"(BigQuery; uses cache where available)", expanded=True)
        with status:
            _area = st.empty()
            _lines: list[str] = []

            # [FN-397]
            def _log(msg):
                _lines.append(str(msg))
                _area.code("\n".join(_lines[-500:]), language="log")

            class _H(logging.Handler):
                # [FN-398]
                def emit(self, rec):
                    try:
                        _log(self.format(rec))
                    except Exception:  # noqa: BLE001
                        pass

            _h = _H()
            _h.setFormatter(logging.Formatter("%(message)s"))
            _root = logging.getLogger()
            _prev = _root.level
            _root.addHandler(_h)
            _root.setLevel(logging.INFO)
            try:
                _log(f"Merged rules -> {merged_dir}")
                _log(f"Live actuals: {v_start} → {v_end}" if v_use_live else "Live actuals: OFF")
                if _auto_force:
                    _log(f"No rule file for {len(_auto_force)} RPGT(s) → forcing ACTUALS: "
                         + ", ".join(_auto_force))
                else:
                    _log("Every RPGT has a rule file — no auto force-actuals needed.")
                out = run_vamp_pipeline(cfg, PROJECT_ROOT, gcp_project=GCP_PROJECT)
                _mid = os.path.join(out, "mid_level.csv")
                if not os.path.exists(_mid):
                    raise FileNotFoundError(f"mid_level.csv not found in pipeline output: {out}")
                _df = pd.read_csv(_mid)
                ss["validate_result"] = _df
                ss["validate_out_dir"] = out
                # Feed tab 3 (impact): point its VAMP exports at THIS run's outputs, and request
                # the "impact-from-validated-split" populate (parse rules -> split, pull the
                # attempts/success window, build the eval frames) — done on the impact tab so its
                # status shows there. The split engine (tab 2) is NOT run.
                ss["pipeline_out_dir"] = out
                ss["validate_populate_req"] = {
                    "rules_dir": rules_dir,
                    "attempts_start": str(v_att_start),
                    "attempts_end": str(v_att_end),
                    "company": _company,
                    "scheme": str(fs.get("card_scheme", "visa") or "visa"),
                }
                status.update(label=f"Pipeline complete — {len(_df)} vampMids.",
                              state="complete", expanded=False)
            except Exception as _e:  # noqa: BLE001
                status.update(label="Pipeline FAILED", state="error", expanded=True)
                st.error(f"{type(_e).__name__}: {_e}")
                st.code(traceback.format_exc())
            finally:
                _root.removeHandler(_h)
                _root.setLevel(_prev)

    _df = ss.get("validate_result")
    if _df is not None and not getattr(_df, "empty", True):
        st.markdown(f"**Pipeline pre vs post** · output: `{ss.get('validate_out_dir', '')}`")
        _render_prepost_table(_to_prepost(_df))
    elif not run:
        st.info("Point at your exported rules folder and run the pipeline to see its pre/post table.")
