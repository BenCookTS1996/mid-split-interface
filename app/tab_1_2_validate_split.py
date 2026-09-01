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


def _prepost_idcol(df: pd.DataFrame) -> str:
    """The MID identifier column name: 'vampMid' (Visa) or 'mastercardMid' (Mastercard).
    Falls back to the first column so the table never crashes on an unexpected shape."""
    for c in ("vampMid", "mastercardMid"):
        if c in df.columns:
            return c
    return str(df.columns[0]) if len(df.columns) else "vampMid"


# [FN-390]
def _to_prepost(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the pipeline's mid_level.csv columns onto the tab-3 table names.

    Handles BOTH schemes: Visa mid_level.csv uses vampMid / FC_VAMP_Month_* / FC_VI_Txn_Month_*;
    Mastercard uses mastercardMid / FC_CB_Month_* (chargebacks) / FC_MC_Txn_Month_* (transactions).
    The Mastercard columns are mapped onto the SAME shared table (chargebacks under the VAMP
    columns, MC txns under the VI Txn columns) so the pre/post view renders for either scheme.
    """
    ren = {}
    for m in range(6):
        ren[f"FC_VAMP_Month_{m}"] = f"VAMP M{m}"
        ren[f"FC_VAMP_Month_{m}_Post"] = f"VAMP Post M{m}"
        ren[f"FC_VI_Txn_Month_{m}"] = f"VI Txn M{m}"
        ren[f"FC_VI_Txn_Month_{m}_Post"] = f"VI Txn Post M{m}"
    # Mastercard mid_level.csv uses different VALUE names — map them onto the same columns (only
    # when the Visa names are absent, so a Visa file is never touched). The ID column keeps its real
    # name ('mastercardMid') so the table labels it correctly.
    if "vampMid" not in df.columns and "mastercardMid" in df.columns:
        for m in range(6):
            ren[f"FC_CB_Month_{m}"] = f"VAMP M{m}"
            ren[f"FC_CB_Month_{m}_Post"] = f"VAMP Post M{m}"
            ren[f"FC_MC_Txn_Month_{m}"] = f"VI Txn M{m}"
            ren[f"FC_MC_Txn_Month_{m}_Post"] = f"VI Txn Post M{m}"
    vp = df.rename(columns=ren)   # rename returns a fresh frame; no extra .copy() needed
    _id = _prepost_idcol(vp)
    cols = [_id]
    for m in range(6):
        cols += [f"VAMP M{m}", f"VI Txn M{m}", f"VAMP Post M{m}", f"VI Txn Post M{m}"]
    cols = [c for c in cols if c in vp.columns]
    vp = vp[cols].copy()
    for c in cols:
        if c != _id:
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
    _id = _prepost_idcol(vp)                       # 'vampMid' (Visa) or 'mastercardMid' (Mastercard)
    # Conditional-formatting thresholds. Visa keys off VAMP counts (1500/1200); Mastercard keys off
    # chargeback counts (100/70) with a 1.5% / 0.9% ratio. (count, rate) — both must be exceeded.
    _is_mc = (_id == "mastercardMid")
    _red_cnt, _red_rate = (100.0, 0.015) if _is_mc else (1500.0, 0.015)
    _amb_cnt, _amb_rate = (70.0, 0.009) if _is_mc else (1200.0, 0.012)
    col_groups, cols = [], [_id]
    for m in range(6):
        grp = [f"VAMP M{m}", f"VI Txn M{m}", f"VAMP Post M{m}", f"VI Txn Post M{m}"]
        grp = [c for c in grp if c in vp.columns]
        if grp:
            cols.extend(grp)
            col_groups.append(grp)

    total = {_id: "TOTAL"}
    for c in cols:
        if c != _id:
            total[c] = vp[c].sum()
    vp_view = pd.concat([vp[cols], pd.DataFrame([total])], ignore_index=True)

    html = [f'<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow-x:auto; '
            f'width:{_dw}; {_disp} background-color:var(--tav-card); border:1px solid var(--tav-line);">']
    html.append(f'<table style="width:{_tw}; border-collapse:collapse; font-family:inherit; '
                'font-size:0.68rem; line-height:1.1;"><tr>')
    html.append('<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; '
                f'text-align:left; position:sticky; left:0; width:1%; white-space:nowrap;">{_id}</th>')
    for gi, grp in enumerate(col_groups):
        for c in grp:
            # Mastercard uses chargebacks (CB) and mastercard txn (MC), so relabel the headers
            # for display only — the underlying column names (VAMP/VI Txn) are unchanged, so all
            # the totals / conditional-formatting logic below still keys off them.
            # Mastercard relabels the metric names (CB / MC Txn) but KEEPS the M0..M5 period axis —
            # the M0->M1 rename applies only to the input widgets/buttons, never the output table.
            _hdr = (c.replace("VAMP", "CB").replace("VI Txn", "MC Txn")) if _is_mc else c
            html.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; '
                        f'text-align:right; white-space:nowrap;">{_hdr}</th>')
        if gi < len(col_groups) - 1:
            html.append('<th style="background-color:var(--tav-card); border:none; width:8px; '
                        'min-width:8px; padding:0;"></th>')
    html.append('</tr>')

    for _, r in vp_view.iterrows():
        is_total = str(r[_id]) == "TOTAL"
        # Mastercard: drop the per-row bottom gridlines so it reads like the Visa/VAMP table.
        # (TOTAL keeps its heavy top rule either way.)
        if is_total:
            tb = "border-top:2px solid var(--tav-ink);"
        else:
            tb = "" if _is_mc else "border-bottom:1px solid var(--tav-line);"
        wt = ("800" if is_total else "600") if bold else ("600" if is_total else "400")
        # Conditional formatting: a VAMP/CB cell (and its paired txn) is RED / AMBER when BOTH its
        # count and its rate exceed the scheme thresholds set above (Visa 1500·1.5% / 1200·1.2%;
        # Mastercard 100·1.5% / 70·0.9%).
        _bgmap = {}
        if not is_total:
            for c in cols:
                if c != _id and c.startswith("VAMP"):
                    _txn = c.replace("VAMP", "VI Txn")
                    _vv = float(r[c]); _tt = float(r[_txn]) if _txn in r.index else 0.0
                    _rt = (_vv / _tt) if _tt > 0 else 0.0
                    if _rt > _red_rate and _vv > _red_cnt:
                        _bgmap[c] = _bgmap[_txn] = "background-color:rgba(230,55,72,0.30);"
                    elif _rt > _amb_rate and _vv > _amb_cnt:
                        _bgmap[c] = _bgmap[_txn] = "background-color:rgba(245,158,11,0.38);"
        html.append('<tr>')
        html.append(f'<td style="padding:2px 8px; text-align:left; color:#000000; font-weight:{wt}; '
                    f'{tb} position:sticky; left:0; background-color:var(--tav-card); width:1%; '
                    f'white-space:nowrap;">{r[_id]}</td>')
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
    from app_common import read_json
    if not rules_dir or not os.path.isdir(rules_dir):
        return {}
    for cand in (rules_dir, os.path.dirname(os.path.normpath(rules_dir))):
        p = os.path.join(cand, "_export_manifest.json")
        if os.path.exists(p):
            return read_json(p, default={})
    return {}


# [FN-393]
def _drift_check(ss, rules_dir: str) -> None:
    """Warn if the rule files in `rules_dir` were exported for a DIFFERENT split than the latest
    export in this session — i.e. tab 5 would run rules that no longer match tab 3's projection."""
    man = _read_export_manifest(rules_dir)
    if not man:
        return
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
    from app_common import RPGT_LIST, COMPANIES, fetch_m0_weightings, green_button_css  # shared constants + helpers
    fs = ss.get("forecast_settings") or {}

    # Company + Month 0 are settable in Section 1 (widgets keyed 'validate_company' / 'validate_month0').
    # Read them here from session_state (same "read at top, render the widget below" pattern as the
    # scheme selector), falling back to the inherited Build Baseline values on first render.
    _company_default = str(fs.get("company", "TotalAV"))
    if _company_default not in COMPANIES:
        _company_default = COMPANIES[0]
    ss.setdefault("validate_company", _company_default)
    # Guard: a stale/invalid stored value (e.g. an old free-text entry) would break the Company
    # selectbox below, so snap it back to a valid option.
    if ss.get("validate_company") not in COMPANIES:
        ss["validate_company"] = _company_default
    _company = str(ss.get("validate_company") or _company_default)

    # [FN-396]
    def _d(key, fallback):
        try:
            return pd.to_datetime(fs.get(key)).date() if fs.get(key) else fallback
        except Exception:  # noqa: BLE001
            return fallback

    _today = datetime.date.today()
    _m0_default = _d("month_0", _today.replace(day=1))
    ss.setdefault("validate_month0", _m0_default)
    _m0 = ss.get("validate_month0") or _m0_default
    if not isinstance(_m0, datetime.date):
        try:
            _m0 = pd.to_datetime(_m0).date()
        except Exception:  # noqa: BLE001
            _m0 = _m0_default
    # month_var (the output month-folder label, e.g. "AUG") is DERIVED from Month 0 so outputs land in
    # the right folder when the user changes it here.
    _month = _m0.strftime("%b").upper()
    _rpgts = list((fs.get("m0_transaction_weightings") or {}).keys())

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
    # NOTE: the Card Scheme selector lives inside the '1. Rules Import' card below (on the same row as
    # the Exported rules folder) so it can drive that folder's scheme-specific default. It is read
    # BEFORE the M0 fetch button that needs `_scheme`, and it overrides a LOCAL copy of fs so the whole
    # validate run uses the selected scheme without touching the saved baseline.
    def _v_fetch_m0(_co, _sch):
        fetch_m0_weightings(_co, _sch, assumed_prefix="validate_assumed_",
                            total_key="validate_m0_total_key", msg_key="_v_m0_fetch_msg",
                            err_key="_v_m0_fetch_err")

    # ---- 2×2 grid, OUTSIDE the form: the M0 fetch button can't live in st.form, so Rules /
    #      Live / Attempts sit beside it and read from session_state at submit (like the M0 card).
    #      LEFT col: 1. Rules Import → 2. M0 Transaction Weightings · RIGHT col: 3. Live actuals → 4. Attempts.
    _gL, _gR = st.columns(2)
    with _gL:
        with st.container(border=True):   # bordered card so it aligns with the M0 card below
            st.markdown("**1. Rules Import**")
            # Folder input + Card Scheme on ONE row. Scheme is read first so the folder can default to
            # its scheme-specific subfolder (data/exported_rules/<scheme>). Selecting a scheme drives the
            # Visa-vs-Mastercard forecast pipeline + attempts/success SQL; we override a LOCAL copy of fs
            # so build_pipeline_config and every fs.get('card_scheme') use it (baseline untouched).
            # Fill the columns LEFT→RIGHT (folder then scheme). The scheme VALUE is available from
            # session_state before either widget renders, so we don't need to render the selectbox first
            # to know it. Filling out of order (scheme in _ri2 before folder in _ri1) plus a state-
            # mutating callback in this nested-column block is what tripped Streamlit's "Bad setIn index"
            # delta on the next callback-driven rerun (e.g. the M0 fetch).
            _ri1, _ri2 = st.columns([3, 1])
            _scheme_default = str(fs.get("card_scheme", "visa") or "visa").strip().lower()
            ss.setdefault("validate_card_scheme", _scheme_default)
            _scheme = str(ss.get("validate_card_scheme") or _scheme_default).strip().lower()
            fs = {**fs, "card_scheme": _scheme, "company": _company,
                  "month_0": str(_m0), "month_var": _month}
            # Mastercard's pipeline is M1-anchored (month 0 = injected historical baseline), so every
            # "M0" header/input in this tab reads "M1" for mastercard. Visa stays "M0".
            _mlabel = "M1" if _scheme == "mastercard" else "M0"
            # 19fh: the default is now data/exported_rules/<Company>/<scheme>, matching the
            # outputs layout (data/outputs/<MONTH>/<Company>/<scheme>/). Scheme alone put every
            # brand's rule files in one folder, so exporting a second brand overwrote the first's
            # templates with no warning — the folder name is the only thing that separated them.
            def _v_rules_default(_co, _sc):
                return os.path.join("data", "exported_rules",
                                    str(_co or "").replace(" ", ""),
                                    str(_sc or "visa").strip().lower())

            ss.setdefault("validate_rules_dir", _v_rules_default(_company, _scheme))

            # Rules folder follows the scheme via an on_change CALLBACK (a programmatic write in the
            # render body would make st.tabs jump back to 'Build Baseline').
            def _v_scheme_changed():
                _sc = str(ss.get("validate_card_scheme", "visa") or "visa").strip().lower()
                ss["validate_rules_dir"] = _v_rules_default(_company, _sc)

            rules_dir = _ri1.text_input(
                "Exported rules folder", key="validate_rules_dir",
                help="Folder containing ALL the rule files for this run (your exported split "
                     "templates). Defaults to data/exported_rules/<Company>/<scheme>. Any RPGT with NO rule file "
                     "here is automatically routed on ACTUALS (force-actuals).")
            _ri2.selectbox(
                "Card Scheme", ["visa", "mastercard"], key="validate_card_scheme",
                on_change=_v_scheme_changed,
                help="Which card scheme's pipeline to validate. Selects the Visa vs Mastercard forecast "
                     "pipeline and the attempts/success SQL, and points the rules folder at that "
                     "scheme's subfolder. Defaults to the Build Baseline scheme.")
            # Company + Month 0 (settable here; default to the inherited Build Baseline values). Both
            # are read back from session_state at the top of render(); no value= kwarg, so there is no
            # "default value + Session State API" warning.
            _ri3, _ri4 = st.columns(2)
            # 19fh: the rules folder now carries the COMPANY as well as the scheme, so changing
            # the company has to move the folder with it — otherwise the field keeps pointing at
            # the previous brand's templates, which is the exact mix-up the per-company folder
            # was added to prevent. Same callback shape as the scheme selector, for the same
            # reason (a programmatic write in the render body makes st.tabs jump back to
            # 'Build Baseline').
            def _v_company_changed():
                ss["validate_rules_dir"] = _v_rules_default(
                    ss.get("validate_company", ""),
                    ss.get("validate_card_scheme", "visa"))

            _ri3.selectbox(
                "Company", COMPANIES, key="validate_company",
                on_change=_v_company_changed,
                help="Company to forecast/validate. Defaults to the Build Baseline company. "
                     "Changing it repoints the Exported rules folder at that company.")
            _ri4.date_input(
                "M0 start date", key="validate_month0",
                help="Month 0 start date (the 1st of the base month). Sets the forecast anchor and the "
                     "output month folder (month_var). Defaults to the Build Baseline Month 0.")
            _drift_check(ss, ss.get("validate_rules_dir", ""))   # flag if these rules ≠ tab-3's split

        with st.container(border=True):
            st.markdown(f"**2. {_mlabel} Transaction Weightings**")   # same bold body size as headers 1 & 3
            # Green button, white text (scoped to this button's key).
            green_button_css("validate_fetch_m0_btn")
            _vfb1, _vfb2 = st.columns([1, 1.5], vertical_alignment="center")
            _vfb1.button(f"Fetch {_mlabel} Weightings", key="validate_fetch_m0_btn",
                         on_click=_v_fetch_m0, args=(_company, _scheme),
                         help=f"Query last month's projected {_scheme.title()} transactions per RPGT "
                              f"for {_company} and fill the weightings below.")
            if ss.get("_v_m0_fetch_err"):
                _vfb2.markdown(f"<span style='color:#e63748; font-size:0.8rem;'>✗ {_mlabel} fetch failed: "
                               f"{ss.get('_v_m0_fetch_err')}</span>", unsafe_allow_html=True)
            else:
                _vfb2.markdown("<span></span>", unsafe_allow_html=True)  # fetch success message suppressed
            # Auto-populate the M0 weightings from BigQuery — the SAME projection the 'Fetch M0
            # Weightings' button runs — so the DEFAULTS are the fetched values. Runs once per
            # (company, scheme); manual edits then persist until company/scheme changes. Falls back to
            # the Build Baseline M0 values if the fetch fails or leaves any field unset.
            _w0 = fs.get("m0_transaction_weightings") or {}
            _m0_sig = (_company, _scheme)
            if ss.get("_validate_m0_autofetch_sig") != _m0_sig:
                ss["_validate_m0_autofetch_sig"] = _m0_sig
                # Clear the previous (company, scheme) values so a re-fetch replaces them and a failed
                # fetch falls back to the Build Baseline defaults set below.
                ss.pop("validate_m0_total_key", None)
                for _rp in RPGT_LIST:
                    ss.pop(f"validate_assumed_{_rp}", None)
                try:
                    _v_fetch_m0(_company, _scheme)
                except Exception:  # noqa: BLE001
                    pass
            ss.setdefault("validate_m0_total_key", int(fs.get("m0_total_transactions", 0) or 0))
            for _rp in RPGT_LIST:
                ss.setdefault(f"validate_assumed_{_rp}", int(_w0.get(_rp, 0) or 0))
            _v_alloc = sum(int(ss.get(f"validate_assumed_{_rp}", 0) or 0) for _rp in RPGT_LIST)
            _v_total = int(ss.get("validate_m0_total_key", 0) or 0)
            _vmt1, _vmt2 = st.columns([3, 2], vertical_alignment="center")
            _vmt1.number_input(f"{_mlabel} {_company} - {_scheme} - Total", 0, 50_000_000,
                               step=1000, key="validate_m0_total_key",
                               help=f"Total starting transactions for {_mlabel}.")
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
        # Load-previous toggle lives in the LEFT column, directly below section 2 (≈ under the
        # P6M Renewals weighting). OUTSIDE the form so ticking it reruns immediately and shows/
        # hides the Forecast outputs folder in the form below (a form would defer that to submit).
        v_use_prev = st.checkbox(
            "Load a previously-run forecast (skip the live pipeline)",
            value=bool(ss.get("validate_use_prev", False)), key="validate_use_prev",
            help="Reuse an existing data/outputs/<MONTH>/<COMPANY>/ folder from a prior run "
                 "instead of re-running the pipeline. The rules folder above is still parsed "
                 "for the impact split.")
    with _gR:
        v_use_live = True   # always on — the 'Use Live Actuals' toggle was removed
        # Header + date pair share ONE bordered container (mirrors section 2 and the section-4 form).
        # The shared border/padding (a) lines the "3. Live actuals" header up horizontally with the
        # "4. Inputs & Assumptions" header below, and (b) makes the combined Start+End width match the
        # 'Force Actuals for' input inside that form.
        with st.container(border=True):
            st.markdown("**3. Live actuals**")
            _d1, _d2 = st.columns(2)
            v_start = _d1.date_input("Start Date", value=ss.get("validate_actuals_start", _d("start_date", _today)),
                                     key="validate_actuals_start")
            v_end = _d2.date_input("End Date", value=ss.get("validate_actuals_end", _d("end_date", _today)),
                                   key="validate_actuals_end")

        # Spacer: Section 1 (left) carries an extra input row (Company + M0 start date) that Section 3
        # (right) does not, so Section 2's header sits one row lower than Section 4's. Drop Section 4
        # by the same amount so its "4. Inputs & Assumptions" header lines up horizontally with the
        # "2. … Transaction Weightings" header. Tune the px if it's slightly off on your screen
        # (≈ one date-input row incl. its label + the inter-element gap).
        st.markdown("<div style='height:76px'></div>", unsafe_allow_html=True)

        # 4. Inputs & Assumptions — moved directly below Live actuals (same right column). Its
        # submit button drives the run log rendered full-width below the grid.
        with st.form("validate_form", border=True):
            st.markdown("**4. Inputs & Assumptions**")
            _gc, _tc = st.columns(2)   # go-live / anchor (left) · lookback / thermometer (right)
            with _gc:
                v_go_live = st.date_input(
                    "Split Go Live date", value=ss.get("validate_go_live", _d("split_go_live_date", _m0)),
                    key="validate_go_live", help="Date the proposed split goes live (drives mid-month pro-rata).")
                v_anchor = st.date_input(
                    "Future Anchor Date", value=ss.get("validate_anchor", _d("future_anchor_date", _today)),
                    key="validate_anchor", help="Date the forecast is anchored to.")
            with _tc:
                v_lookback = st.number_input(
                    "T0 lookback (months)", min_value=1, max_value=12,
                    value=int(ss.get("validate_lookback", int(fs.get("t0_lookback_months", 1) or 1))),
                    key="validate_lookback", help="Actuarial t0_lookback_months.")
                v_thermo = st.number_input(
                    "Thermometer sample (months)", min_value=1, max_value=12,
                    value=int(ss.get("validate_thermo", int(fs.get("thermometer_sample_months", 2) or 2))),
                    key="validate_thermo", help="Actuarial thermometer_sample_months.")
            # Inside the form so changing it does NOT rerun the tab — it only takes effect when
            # the green submit button is pressed. Defaults to ALL RPGTs.
            # 'Error' is never a routable RPGT (removed from attempts_success.sql), so it must
            # never be an option here. Filter it from the options; and on repeat renders scrub any
            # stale 'Error' out of the existing selection. Pass `default` ONLY on the FIRST render
            # (no session_state yet) — passing `default` AND writing session_state triggers
            # Streamlit's "default value + Session State API" warning.
            _force_opts = [str(r) for r in _rpgts if str(r).strip().lower() != "error"]
            _ms_kw = {}
            if isinstance(ss.get("validate_force_actuals"), list):
                ss["validate_force_actuals"] = [
                    _x for _x in ss["validate_force_actuals"] if str(_x).strip().lower() != "error"]
            else:
                _ms_kw["default"] = list(_force_opts)
            v_force_manual = st.multiselect(
                "Force Actuals for", options=_force_opts, key="validate_force_actuals",
                help="These transaction types use live actuals instead of the forecast for "
                     "month 0. Leave empty to force none.", **_ms_kw)
            # Forecast outputs folder only appears when the load-previous box (above, outside
            # the form) is ticked. Default to the stored path when hidden so `if run:` is safe.
            v_prev_dir = ss.get("validate_prev_dir", "")
            if v_use_prev:
                v_prev_dir = st.text_input(
                    "Forecast outputs folder", value=ss.get("validate_prev_dir", ""),
                    key="validate_prev_dir",
                    help="The data/outputs/<MONTH>/<COMPANY>/ folder containing mid_level.csv "
                         "(and the other export CSVs) from a previous run.")
            run = st.form_submit_button("Run Validation", type="primary")

    # 5. Attempts & success data — its own row, below Inputs & Assumptions + the run log
    # (swapped up from where it used to sit beside Live actuals). Same inputs as the Routing
    # engine tab; used to pull the success-rate data that populates tab 3's impact views.
    _as_col, _as_sp = st.columns(2)
    with _as_col:
        # Bordered container mirrors section 1 (Rules Import): the same border/padding inset makes the
        # combined Start+End width match the 'Exported rules folder' + 'Card Scheme' combined width.
        with st.container(border=True):
            st.markdown("**5. Attempts & success data**")
            _yday = _today - datetime.timedelta(days=1)
            # Default Start date = the 1st of the month 3 months ago (e.g. Aug -> 1 May).
            _mi3 = (_today.year * 12 + _today.month - 1) - 3
            _att_start_default = datetime.date(_mi3 // 12, _mi3 % 12 + 1, 1)
            _as1, _as2 = st.columns(2)
            v_att_start = _as1.date_input(
                "Start date",
                value=ss.get("validate_attempts_start", _att_start_default),
                key="validate_attempts_start",
                help="Success-rate data window (attempts & successes) used to populate tab 3's "
                     "impact views for the validated split.")
            v_att_end = _as2.date_input(
                "End date", value=ss.get("validate_attempts_end", _yday),
                key="validate_attempts_end")

    # Run log — full width below the sections (the Inputs & Assumptions form moved into the right
    # column, so the log no longer sits beside it). Target container; populated on submit.
    _log_col = st.container()

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

        if v_use_prev:
            # ---- Load a previously-run forecast: reuse a prior run's outputs, no BigQuery. ----
            if not v_prev_dir or not os.path.isdir(v_prev_dir):
                st.error(f"Forecast outputs folder not found: {v_prev_dir or '(empty)'}")
                return
            _mid = os.path.join(v_prev_dir, "mid_level.csv")
            if not os.path.exists(_mid):
                st.error(f"mid_level.csv not found in: {v_prev_dir} — is this a completed "
                         "data/outputs/<MONTH>/<COMPANY>/ run folder?")
                return
            try:
                _df = pd.read_csv(_mid)
            except Exception as _e:  # noqa: BLE001
                st.error(f"Could not read {_mid}: {type(_e).__name__}: {_e}")
                return
            ss["validate_result"] = _df
            ss["validate_out_dir"] = v_prev_dir
            # Point tab 3 (impact) at this loaded forecast and request the populate-from-split
            # (parse rules -> split, pull the attempts/success window, build eval frames).
            ss["pipeline_out_dir"] = v_prev_dir
            ss["validate_populate_req"] = {
                "rules_dir": rules_dir,
                "attempts_start": str(v_att_start),
                "attempts_end": str(v_att_end),
                "company": _company,
                "scheme": str(fs.get("card_scheme", "visa") or "visa"),
            }
            _log_col.success(f"Loaded previously-run forecast — {len(_df)} vampMids from "
                             f"`{v_prev_dir}`. Impact populates on tab 3.")
            _df = ss.get("validate_result")
            if _df is not None and not getattr(_df, "empty", True):
                _render_prepost_table(_to_prepost(_df), bold=False)
            return

        merged_dir = os.path.join(PROJECT_ROOT, "data", "rules", "_validate", _month, _company)
        nr = _stage_rules(rules_dir, merged_dir)

        _scheme = str(fs.get("card_scheme", "visa") or "visa").strip().lower()
        _is_mc = (_scheme == "mastercard")
        try:
            if _is_mc:
                from routing_optimiser.s2_forecast.mastercard_forecast_pipeline import (
                    build_mc_pipeline_config as build_pipeline_config,
                    run_mastercard_pipeline as run_vamp_pipeline)
            else:
                from routing_optimiser.s2_forecast.vamp_forecast_pipeline import (build_pipeline_config,
                                                                 run_vamp_pipeline)
        except Exception as _ie:  # noqa: BLE001
            st.error(f"Could not import the pipeline runner: {type(_ie).__name__}: {_ie}")
            return

        cfg = build_pipeline_config(fs)
        cfg.setdefault("paths", {})
        cfg["paths"]["chunked_files_dir"] = merged_dir            # absolute -> used as-is
        # Separate output dir so this never clobbers tab 1/tab 3's live outputs.
        # Mastercard runs land in their own subfolder so the two schemes never collide.
        _validate_out = os.path.join("data", "outputs", "_validate", "{month_var}", "{company}")
        # Each scheme lands in its own subfolder so the two never collide (visa used to
        # sit bare in <company>/; now symmetric with mastercard).
        _validate_out = os.path.join(_validate_out, "mastercard" if _is_mc else "visa")
        cfg["paths"]["output_dir"] = _validate_out + os.sep
        cfg["run_settings"]["use_chunked_csv_files"] = True

        # Live actuals from the '2. Live actuals' inputs above.
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

        # Run log renders into the RIGHT column beside '5. Inputs & Assumptions' (created above).
        status = _log_col.status(f"Running the {'MASTERCARD' if _is_mc else 'VAMP'} pipeline with "
                                 f"{nr} rule file(s)… (BigQuery; uses cache where available)", expanded=True)
        with status:
            # Fixed-height scroll box so the run log stays bounded to ~the Inputs & Assumptions
            # form's height (its bottom lines up ~with the green submit button) instead of
            # sprawling down the page. Tune _LOG_H (px) to nudge the bottom to match on your screen.
            _LOG_H = 460
            _area = st.container(height=_LOG_H).empty()
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
        _render_prepost_table(_to_prepost(_df), bold=False)
    elif not run:
        st.info("Point at your exported rules folder and run the pipeline to see its pre/post table.")
