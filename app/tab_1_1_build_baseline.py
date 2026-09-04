"""Tab 1 · sub-tab 1 — Build Baseline.  (also the host of tab 1's sub-tab bar)

Originally split out of streamlit_app.py into its own file (since evolved) so the main
script stays small. streamlit_app.py calls `render()` from inside `with tab_fc:`.

TWO JOBS, which is why the filename says 1_1 and this docstring says more than that:
  1. `render()` creates tab 1's TWO sub-tabs — Build Baseline / Validate Split — and delegates
     1·2 to tab_1_2_validate_split.render() at the bottom of this file.
  2. Everything ABOVE that call is sub-tab 1's own body: run identity, data sources,
     assumptions, configs & overrides, the green Calculate/Load Forecast button and the run log.

19jz: there WAS a third sub-tab, 'Config Validation'. Its left column embedded tab 4's config
generator and its right column held a profile lookup — i.e. tab 4's two jobs, in tab 1, with a
second set of widget keys. Ben removed it as duplication and tab 4 (now '4 · Config Files')
took the layout whole. It is also what fixed the whiteout: Streamlit builds tab panels in
script order and the generator BLOCKS that run, so while it worked from in here, tabs 2, 3 and
4 had not been built yet and showed blank.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from datetime import date

import pandas as pd
import streamlit as st

from app_common import fetch_m0_weightings, green_button_css, read_json
import yaml

from routing_optimiser import build_pipeline_config, run_vamp_pipeline
from routing_optimiser import build_mc_pipeline_config, run_mastercard_pipeline

from app_common import (ss, PROJECT_ROOT, GCP_PROJECT, input_json_path,
                        RPGT_LIST, COMPANIES, StreamlitLogHandler, _switched_off_gateways)


# ── 19kg: SETTINGS THAT USED TO BE ENVIRONMENT SWITCHES ──────────────────
# No environment variable changes a run any more. Each name below is frozen at the
# value the shipped run already used - the defaults, because no routing.env exists and
# run.command exports nothing - so what shipped is what these say. They stay NAMES, not
# literals inlined at the use site, for two reasons: a test can still A/B a whole search
# by rebinding one, and a reader can see in one place every decision this module makes.
# Changing behaviour now means editing this block and saying so in a commit.
_SW_TAB1_PATHPARSE = True   # was ROUTING_TAB1_PATHPARSE, default '1'

# [FN-293]
def render():
    _bb, _vs = st.tabs(["Build Baseline", "Validate Split"])
    with _bb:
        # --- Highly targeted CSS: applies card shadow ONLY to the 6 section containers ---
        # --- Also aggressively squashes vertical margins specifically inside these cards ---
        st.markdown("""<style>
            div[data-testid="column"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
                box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
                background-color: #FFFFFF !important;
                border-radius: 0 !important;
                border: 1px solid var(--tav-line) !important;
                padding: 1rem 1rem 0.5rem 1rem !important;
            }
            div[data-testid="column"] > div > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
                gap: 0.15rem !important;
            }
        </style>""", unsafe_allow_html=True)

        # ---- defaults (used when settings are hidden while loading a forecast) --
        # [FN-294]
        def _load_default_json(name):
            # Scheme-aware (19ft): the visa/ or mastercard/ copy, else the shared root file.
            # NOTE this runs BEFORE the Card Scheme selectbox renders, which is exactly why
            # that widget carries key="ident_card_scheme" -- see run_scheme()'s docstring.
            return read_json(input_json_path(name))

        company = COMPANIES[0]
        scheme = "visa"
        m0_date = date.today().replace(day=1)
        month_var = m0_date.strftime("%b").upper()
        future_anchor_date = date.today()
        m0_total = 318_077
        rpgt_assumed = {r: 0 for r in RPGT_LIST}
        use_live_actuals = False
        actuals_start_date = actuals_end_date = None
        actuals_valid = True
        force_actuals_for_rpgts = []
        t0_lookback_months, decay_factor, thermometer_sample_months = 1, 0.5, 1
        shrink = 12.0
        # 19ka: `use_yaml_asis` deleted with its checkbox (see the Data Sources row below).
        run_live, reuse_cached_curves = True, True
        use_cached_inputs, load_baseline, cached_inputs_path = False, False, ""
        test_gateways = _load_default_json("test_gateways.json") or {}
        thermometer_config = _load_default_json("thermometer_config.json")
        gateway_volume_overrides = _load_default_json("gateway_volume_overrides.json")

        # ---- use a previously created forecast --------------------------------
        use_prev = st.checkbox(
            "Use a previously created forecast", value=False,
            help="Load a finished VAMP forecast from disk instead of running the "
                 "pipeline. Give the folder the outputs were saved to; if valid, the "
                 "other inputs are hidden and the forecast is loaded for tab 2.")
        # 19et: how tall the two scrollable code boxes are — the settings.yaml preview and the
        # run log. Streamlit renders `st.code` at roughly 21 px per line, so ~10 lines plus the
        # block's own padding is ~225 px. THIS IS THE WHOLE KNOB for both: raise it to show more
        # lines before scrolling, lower it to show fewer. Kept as one constant so the two boxes
        # cannot drift apart.
        _CODE_BOX_PX = 225

        # 19ev: BUILD MODE ONLY (19fc). The height RESERVED in row 2's right column for the
        # settings.yaml preview and the run log together. Both live inside one container of this
        # height, so opening either scrolls INSIDE it and the container never changes size.
        #
        # WHY THAT IS THE FIX. The pre/post table is not in a column — it renders after the whole
        # row-2 block, and a Streamlit row is as tall as its tallest column. So anything that
        # grows a column grows the row and pushes everything below it down. Nothing can be
        # "beside" the table, because the table starts where the row ends. Reserving the space up
        # front is the only way to make expanding free.
        #
        # COSTS NOTHING IN PRACTICE while it stays under the height of the LEFT column (the
        # "4 · Assumptions" box), because the row is already that tall — the reserve is spent on
        # whitespace the row had anyway. Raise it past that and the row grows by the difference,
        # permanently. That is the trade being made here: a little reserved space always, instead
        # of a jump every time something is opened.
        _GROW_BOX_PX = 300

        # 19fa: DECLARED HERE, above the previous-forecast block, because that block now reserves
        # the green button's position inside its OWN LEFT column (_pv1). It used to be declared
        # below the block, which would have overwritten the reserved slot with None the instant
        # the block finished — the slot has to outlive the `with`-less column it is created in.
        _calc_btn_slot = None   # green "Calculate Forecast" / "Load forecast" button
        _pv_msg_slot = None     # previous-forecast folder status line (green ✓ / red error)
        settings_hidden = False
        if use_prev:
            # Forecast-outputs folder on the LEFT; Split Go Live date on the RIGHT (the other
            # inputs are hidden in this mode, so it's shown here so it's always available).
            # Split Go Live keeps its reduced width (0.175 of the row); the folder input is widened
            # to 0.5 so its long label fits on one line. The trailing column is an empty spacer.
            # folder + Split-Go-Live combined width ≈ 0.5 of the row, matching the "Month 0" +
            # "Future Anchor Date" pair (which fill one half of an st.columns(2) split). Ratio kept
            # ~folder:date = 0.37:0.13 (≈ the original 0.5:0.175).
            _pv1, _pv2, _pv_sp = st.columns([0.37, 0.13, 0.5])
            prev_dir = _pv1.text_input(
                "data/outputs/<MONTH>/<COMPANY>/<SCHEME>/ ", "")
            # 19fa: THE GREEN BUTTON AND THE FOLDER STATUS LINE RENDER HERE, IN THE LEFT COLUMN.
            # Both used to render full-width BELOW this row. A Streamlit row is as tall as its
            # tallest column, so everything under the row moves whenever the run-log column
            # (`_pv_sp`, on the right) changes height — which is exactly what opening the log
            # does. Reserving their positions inside `_pv1` anchors them to the TOP of the left
            # column, where the right column's height cannot reach them. This is the same
            # slot-reservation pattern 19eq uses for row 2 in build mode, and for the same reason:
            # a widget renders where it is CREATED, and neither of these can be created here (the
            # button needs `settings_hidden`, the message needs the parsed folder).
            _pv_msg_slot = _pv1.empty()
            _calc_btn_slot = _pv1.empty()
            split_go_live = _pv2.date_input(
                "Split Go Live date", value=ss.get("split_go_live_date", m0_date), key="sgl_hidden",
                help="Date the proposed split goes live. Drives the mid-month pro-rata "
                     "element in the forecast export and the tab-4 VAMP Post projection.")
            # 19fc: THE RUN LOG NO LONGER LIVES IN THIS ROW. It used to render into `_pv_sp`,
            # the trailing spacer column, inside a height-capped box (19ev) so that expanding it
            # would scroll rather than grow the row. The cap was not enough: the pre-vs-post
            # baseline table renders BELOW the whole row, and a Streamlit row is as tall as its
            # tallest column, so any height this column takes — reserved OR grown — lands on the
            # table. Height-capping only fixes the growth, not the reserve, and there is no cap
            # value that makes a column's height free to the content underneath it.
            # The log is now created LAST, below the table, where nothing sits beneath it and it
            # can expand into empty space. `_pv_sp` is a plain spacer again, which is what keeps
            # the folder + Split-Go-Live pair at half the row's width.
            # SEE the "19fc" block further down, next to `_pre_table_ctx`.
            ss["split_go_live_date"] = split_go_live
            if prev_dir:
                need_all = ["mid_level.csv", "vamp_t_period_export.csv"]
                need_any = ["bin_rpgt_impact_export.csv", "effective_rate_impact.csv"]
                miss = [f for f in need_all if not os.path.exists(os.path.join(prev_dir, f))]
                if not any(os.path.exists(os.path.join(prev_dir, f)) for f in need_any):
                    miss.append("bin_rpgt_impact_export.csv or effective_rate_impact.csv")
                if os.path.isdir(prev_dir) and not miss:
                    settings_hidden = True
                    run_live, load_baseline, use_cached_inputs = False, True, True
                    cached_inputs_path = prev_dir
                    parts = [p for p in os.path.normpath(prev_dir).split(os.sep) if p]
                    # ── IDENTIFY BY CONTENT, NOT POSITION (2026-08-19h) ──────────────────
                    # This used to read `company, month_var = parts[-1], parts[-2]`, i.e. it
                    # assumed the layout `.../<MONTH>/<COMPANY>/`. When the outputs gained a
                    # per-scheme subfolder the path became `.../AUG/TotalAV/visa/`, so the
                    # SCHEME was read as the company and the COMPANY as the month. tab 2 then
                    # queried BigQuery with COMPANY='visa', got 0 rows, and the run died at
                    # "RPGT scope removed all rows". The month parse below raises ValueError on
                    # 'Totalav' and is swallowed, so nothing pointed at the real cause, and this
                    # branch hides the Company selectbox so there was no way to correct it.
                    # Match each component on what it IS instead. Scan forward and keep the LAST
                    # match, so the deepest directory wins and a stray 'May' or a company-shaped
                    # folder higher up an absolute path cannot outrank the real one.
                    _pp_sch = _pp_co = None
                    _pp_mo = None                      # (MONTH_LABEL, month_number)
                    if _SW_TAB1_PATHPARSE:
                        _pp_known = {str(_c).replace(" ", "").lower(): str(_c) for _c in COMPANIES}
                        for _pp in parts:
                            _ppl = str(_pp).strip().lower()
                            if _ppl in ("visa", "mastercard"):
                                _pp_sch = _ppl
                            _ppk = _ppl.replace(" ", "")
                            if _ppk in _pp_known:
                                _pp_co = _pp_known[_ppk]
                            try:
                                _pp_mo = (str(_pp).upper(),
                                          datetime.datetime.strptime(
                                              str(_pp).capitalize(), "%b").month)
                            except ValueError:
                                pass
                    if _pp_co is not None and _pp_mo is not None:
                        company, month_var = _pp_co, _pp_mo[0]
                        m0_date = m0_date.replace(month=_pp_mo[1])
                        if _pp_sch is not None:
                            # The old parser never read the scheme, so a mastercard forecast
                            # silently inherited the 'visa' default — and with it the WRONG
                            # MID list and BIN prefix. Take it from the folder that names it.
                            scheme = _pp_sch
                    elif len(parts) >= 2:
                        # Not positively identified ⇒ fall through to the ORIGINAL positional
                        # behaviour rather than refusing to load a non-standard folder.
                        company, month_var = parts[-1], parts[-2]
                        try:
                            m_int = datetime.datetime.strptime(month_var.capitalize(), "%b").month
                            m0_date = m0_date.replace(month=m_int)
                        except ValueError:
                            pass
                    # 12px to match the widget-label text size (e.g. the 'Forecast outputs folder'
                    # label), instead of the larger default st.success alert.
                    # 19fb: `margin-bottom: 14px` is the GAP between this line and the green
                    # button directly below it. Both live in the same left column now (19fa), and
                    # a 12px line sitting hard against a button reads as one block. The gap is on
                    # THIS element rather than a separate spacer because `_pv_msg_slot` is an
                    # st.empty(), which holds exactly one element — a spacer would need a
                    # container and would then also have to be cleared on the error branch.
                    (_pv_msg_slot or st).markdown(
                        f"<div style='font-size:12px; color:#1D9E75; font-weight:600; "
                        f"margin:2px 0 14px 0;'>"
                        f"✓ Valid forecast found — {company} ({month_var}). Other inputs hidden. "
                        f"Click <b>Load forecast</b>, then open tab 2.</div>",
                        unsafe_allow_html=True)
                else:
                    (_pv_msg_slot or st).error("Missing/invalid outputs: "
                             + (", ".join(miss) if miss else "not a folder")
                             + f".  Looked in: {prev_dir}")

        _fc_log_slot = None   # forecast run-log slot (assigned in ROW 2 right column)
        # 19eq: a position RESERVED in the row-2 right column and written to further down. A
        # widget renders where it is created, and the log cannot be created there.
        # 19jz: `_yaml_slot` is GONE with it. It reserved a spot for the settings.yaml preview
        # inside that same row; the preview now renders at the bottom of the tab, where it
        # needs no slot because nothing follows it.
        # `_calc_btn_slot` is DELIBERATELY NOT RESET HERE (19fa): it is declared above the
        # previous-forecast block, and in that mode it already holds the left-column slot the
        # block reserved. Resetting it to None here would send the green button back to
        # full-width below the row, which is the bug 19fa fixed.

        if not use_prev:
            # --- ROW 1: Run Identity & Data Sources ---
            # Hidden as soon as 'Use a previously created forecast' is ticked (not only once a valid
            # folder is found) — forecast_settings below uses the pre-initialised defaults / the folder
            # parse, so skipping these widgets is safe.
            r1_c1, r1_c2 = st.columns(2, gap="large")
        
            with r1_c1:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>1 · Run identity</h5>", unsafe_allow_html=True)
                    id_c1, id_c2 = st.columns(2)
                    company = id_c1.selectbox("Company", COMPANIES,
                                              help="Brand this forecast is for.")
                    # 19ft: `key` is load-bearing, not cosmetic. The config/inputs loader runs
                    # ~180 lines ABOVE this widget in the same script pass, and a keyed widget's
                    # value is in session_state BEFORE the script re-runs -- so this is how
                    # app_common.run_scheme() sees the NEW scheme on the rerun a change triggers,
                    # instead of the previous one out of forecast_settings.
                    scheme = id_c2.selectbox("Card Scheme", ["visa", "mastercard"],
                                             key="ident_card_scheme",
                                             help="Card network to model.")

                    id_c3, id_c4 = st.columns(2)
                    picked = id_c3.date_input("Month 0", value=date.today().replace(day=1),
                                              help="First month of the forecast.")
                    m0_date = picked.replace(day=1)
                    month_var = m0_date.strftime("%b").upper()
                    future_anchor_date = id_c4.date_input("Future Anchor Date", value=date.today(),
                                                          help="Date the forecast is anchored to.")
                    # Split Go Live date, directly beneath Month 0 (same column).
                    split_go_live = id_c3.date_input(
                        "Split Go Live date",
                        value=ss.get("split_go_live_date", date.today() + datetime.timedelta(days=1)),
                        key="sgl_ident",
                        help="Date the proposed split goes live. Drives the mid-month pro-rata "
                             "element in the forecast export and the tab-4 VAMP Post projection.")
                    ss["split_go_live_date"] = split_go_live
                    # Live actuals (moved here from its own section) — blend in real recent results.
                    # Nudge the checkbox down so it vertically centres on the Split Go Live INPUT
                    # box (left column), not up level with its label.
                    st.markdown("<style>.st-key-use_live_actuals_cb{margin-top:1.75rem;}</style>",
                                unsafe_allow_html=True)
                    use_live_actuals = id_c4.checkbox("Use Live Actuals", value=True,
                                                      key="use_live_actuals_cb",
                                                      help="Blend in real recent results.")
                    if use_live_actuals:
                        _la1, _la2 = st.columns(2)
                        # End first so Start can default to End − 14 days. Keys persist edits.
                        actuals_end_date = _la2.date_input("Actuals End Date", value=date.today(),
                                                           key="actuals_end_date_inp")
                        actuals_start_date = _la1.date_input(
                            "Actuals Start Date",
                            value=actuals_end_date - datetime.timedelta(days=14),
                            key="actuals_start_date_inp")
                        if actuals_start_date >= actuals_end_date:
                            actuals_valid = False
                            st.error("Actuals Start Date must be before the End Date.")

                    # --- Backup rules folder: the deployed pipeline blends the exported split with
                    #     the backup files' catch-all (BIN=Other) rows, which re-add gateways the split
                    #     zeroed/omitted. Point tab 2 (optimiser) + tab 3 (impact) at those backups so
                    #     they optimise/project against what is ACTUALLY routed (matches tab 5). ---
                    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
                    _bk_dir = st.text_input(
                        "Catch-All Folder",
                        value=ss.get("backup_rules_dir", "data/backup_rules/"), key="backup_rules_dir_input",
                        help="Folder holding the backup split files (e.g. mid_split_*_Eff_Backup_*). Their "
                             "catch-all 'BIN=Other' rows re-add gateways your exported split zeroed/omitted, "
                             "exactly as the deployed pipeline (tab 5) does. Tabs 2 & 3 blend these in so their "
                             "numbers match tab 5. Leave blank to use the raw exported split only.")
                    ss["backup_rules_dir"] = _bk_dir
                    try:
                        from routing_optimiser.s5_deliver.backup_blend import parse_backup_catchall as _pbc
                        if _bk_dir and os.path.isdir(_bk_dir):
                            _bk_sig = (_bk_dir, os.path.getmtime(_bk_dir))
                            if ss.get("_backup_catchall_sig") != _bk_sig:
                                _bc_parsed = _pbc(_bk_dir)
                                # Exclude switched-off gateways (gateway_volume_overrides target=0,
                                # trx/both) from the catch-all pool, so the backup blend never re-adds
                                # a gateway that was turned off. This pool feeds the tab-2/3 eval view,
                                # the GA fitness AND the exports, so one filter here keeps them all
                                # consistent (fixes bancard/cwams reappearing with share).
                                try:
                                    import json as _jbc
                                    from routing_optimiser.s2_forecast.vamp_forecast_pipeline import _canonical_gateway as _cgbc
                                    _ovp_bc = input_json_path("gateway_volume_overrides.json", scheme)
                                    _off_bc = set()
                                    if os.path.exists(_ovp_bc):
                                        with open(_ovp_bc) as _fbc:
                                            _off_bc = _switched_off_gateways(_jbc.load(_fbc) or {})
                                    if _off_bc and isinstance(_bc_parsed, dict):
                                        _nrm = 0
                                        _clean = {}
                                        for _k, _gw in _bc_parsed.items():
                                            _kept = {g: v for g, v in _gw.items()
                                                     if str(_cgbc(g)).strip().lower() not in _off_bc}
                                            _nrm += len(_gw) - len(_kept)
                                            _clean[_k] = _kept
                                        _bc_parsed = _clean
                                        if _nrm:
                                            st.caption(f"(backup catch-all: dropped {_nrm} switched-off gateway "
                                                       "entr(y/ies) — target=0, trx/both)")
                                except Exception:  # noqa: BLE001
                                    pass
                                ss["backup_catchall"] = _bc_parsed
                                ss["_backup_catchall_sig"] = _bk_sig
                        else:
                            ss["backup_catchall"] = {}
                            ss["_backup_catchall_sig"] = None
                    except Exception as _be:  # noqa: BLE001
                        ss["backup_catchall"] = {}
                        st.caption(f"(backup catch-all parse failed: {type(_be).__name__}: {_be})")

                    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
                    # 19ka - 'Use config/vamp_settings.yaml as-is (repo parity)' IS DELETED, on
                    # Ben's instruction, and so is the code path behind it. Ticking it made the
                    # pipeline read config/vamp_settings.yaml (or mastercard_settings.yaml)
                    # straight off disk and IGNORE every widget in sections 1-4 of this tab - the
                    # company, the scheme, month 0, the weightings, the lot. The run log said so
                    # in a NOTE, which is the only reason a run under it was interpretable at all.
                    # Its default was OFF, so removing it changes nothing about a normal run; what
                    # goes is the ability to produce a forecast whose inputs are not the ones on
                    # screen. `build_pipeline_config(forecast_settings)` is now the only way this
                    # tab assembles a config, which is what the settings.yaml PREVIEW at the
                    # bottom of the tab has always shown.
                    _ds1, _ds2 = st.columns(2)
                    reuse_cached_curves = _ds1.checkbox(
                        "Reuse cached actuarial curves", value=True,
                        help="Reuse cached reference_curves (load_curves_from_cache).")

            with r1_c2:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>2 · M0 Transaction Weightings</h5>", unsafe_allow_html=True)
                    # Auto-fill from BigQuery: last month's PROJECTED Visa txn count per RPGT for the
                    # selected company (queries/m0_weightings.sql, {company} templated). Uses an on_click
                    # CALLBACK (not st.rerun) — a callback runs BEFORE the automatic rerun and can safely
                    # set the input widgets' session_state, avoiding the "Bad setIn index" delta error
                    # that st.rerun() from inside nested containers triggered. Result is cached by sql_runner.
                    # [FN-295]
                    def _fetch_m0(_co, _sch):
                        fetch_m0_weightings(_co, _sch, assumed_prefix="assumed_",
                                            total_key="m0_total_key", msg_key="_m0_fetch_msg",
                                            err_key="_m0_fetch_err")

                    # Green button, white text (scoped to this button's key).
                    green_button_css("fetch_m0_btn")
                    # Button + last-fetch status side by side. Status is GREY and persists next to the
                    # button until the next fetch (an error shows red). Always renders exactly one
                    # element in the status column so the layout tree stays stable across reruns.
                    _fb1, _fb2 = st.columns([1, 1.5], vertical_alignment="center")
                    _fb1.button("Fetch M0 Weightings", key="fetch_m0_btn",
                                on_click=_fetch_m0, args=(company, scheme),
                                help=f"Query last month's projected {str(scheme).title()} transactions "
                                     f"per RPGT for {company} and fill the weightings below.")
                    if ss.get("_m0_fetch_err"):
                        _fb2.markdown("<span style='color:#e63748; font-size:0.8rem;'>✗ M0 fetch failed: "
                                      f"{ss.get('_m0_fetch_err')}</span>", unsafe_allow_html=True)
                    else:
                        _fb2.markdown("<span></span>", unsafe_allow_html=True)  # fetch success message suppressed
                    # Inputs read their default from session_state (so the fetch can set them); no
                    # `value=` arg avoids the "default + session_state" warning. The validator renders
                    # INLINE from the committed session_state sum (no deferred st.empty() across sibling
                    # columns — that deferred write was what tripped Streamlit's "Bad setIn index" delta).
                    ss.setdefault("m0_total_key", 318_077)
                    for _rp in RPGT_LIST:
                        ss.setdefault(f"assumed_{_rp}", 0)
                    _alloc_now = sum(int(ss.get(f"assumed_{_rp}", 0) or 0) for _rp in RPGT_LIST)
                    _total_now = int(ss.get("m0_total_key", 0) or 0)
                    _mtc1, _mtc2 = st.columns([3, 2], vertical_alignment="center")
                    m0_total = _mtc1.number_input(f"M0 {company} - {scheme} - Total", 0, 50_000_000,
                                                  step=1000, key="m0_total_key",
                                                  help="Total starting transactions for month 0.")
                    if _total_now == _alloc_now:
                        _mtc2.markdown("<div style='color:#1D9E75; font-size:0.8rem; font-weight:700;'>"
                                       "✓ matches RPGT sum</div>", unsafe_allow_html=True)
                    else:
                        _diff = _alloc_now - _total_now
                        _mtc2.markdown("<div style='color:#e63748; font-size:0.8rem; font-weight:700;'>"
                                       f"⚠ RPGT sum {_alloc_now:,} ≠ Total "
                                       f"({'+' if _diff > 0 else ''}{_diff:,})</div>", unsafe_allow_html=True)
                    rpgt_assumed = {}
                    w_cols = st.columns(2)
                    for i, rpgt in enumerate(RPGT_LIST):
                        _ak = f"assumed_{rpgt}"
                        rpgt_assumed[rpgt] = w_cols[i % 2].number_input(
                            rpgt, 0, 50_000_000, step=500, key=_ak,
                            help="Assumed month-0 volume for this type.")

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

            # --- ROW 2: Configs & Overrides (left) · Assumptions (right) ---
            r2_c1, r2_c2 = st.columns(2, gap="large")

            with r2_c1:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>4 · Assumptions</h5>", unsafe_allow_html=True)
                    p1, p2 = st.columns(2)
                    t0_lookback_months = p1.number_input("T0 Lookback Months", 0, 36, 1, step=1,
                                                         help="Months of history to learn from. 0 = the last completed month.")
                    decay_factor = p2.number_input("Decay Factor", 0.0, 1.0, 0.5, step=0.01,
                                                   format="%.2f",
                                                   help="How fast old months lose weight.")
                    thermometer_sample_months = st.number_input("Thermometer Sample Months",
                                                                0, 36, 1, step=1,
                                                                help="Months used to shape the ramp-up. 0 = the last completed month.")

                # 19es: the green "Calculate Forecast" button renders HERE — bottom of the LEFT
                # column, below the Assumptions box, so it lines up across the page with the
                # config captions that end the RIGHT column. 19eq put it in a sub-column INSIDE
                # the right column, which put it on the right-hand half of the page; "in-line
                # with the Test Gateways caption" meant the same horizontal band, not the same
                # column. The slot is reserved here and written to further down, because the
                # button needs `actuals_valid` and a persisted `forecast_settings` that do not
                # exist yet at this point in the script.
                _calc_btn_slot = st.empty()

            with r2_c2:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>5 · Configs & Overrides</h5>", unsafe_allow_html=True)

                    # [FN-296]
                    def _read_json(upload, default_path, label):
                        if upload is not None:
                            try:
                                return json.load(upload)
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"Could not parse {label}: {exc}")
                                return None
                        return read_json(default_path)

                    # Shrink the file-uploader "Browse files" button text to match the input text size
                    # (the default is larger than the surrounding inputs).
                    st.markdown(
                        "<style>[data-testid='stFileUploader'] button,"
                        " [data-testid='stFileUploader'] button *,"
                        " [data-testid='stFileUploaderDropzoneInstructions'],"
                        " [data-testid='stFileUploaderDropzoneInstructions'] * "
                        "{ font-size: 0.8rem !important; }</style>",
                        unsafe_allow_html=True)
                    g1, g2, g3 = st.columns(3)
                    test_gw_file = g1.file_uploader("Test Gateways (JSON)", type=["json"])
                    thermo_file = g2.file_uploader("Thermometer Config", type=["json"])
                    override_file = g3.file_uploader("Gateway Volume Overrides", type=["json"])

                    # 19ft: `scheme` is in hand here (the selectbox is above this point), so it
                    # is passed explicitly -- no session-state read, no ordering question.
                    test_gateways = _read_json(test_gw_file,
                                               input_json_path("test_gateways.json", scheme),
                                               "Test Gateways") or {}
                    thermometer_config = _read_json(
                        thermo_file, input_json_path("thermometer_config.json", scheme),
                        "thermometer config")
                    gateway_volume_overrides = _read_json(
                        override_file, input_json_path("gateway_volume_overrides.json", scheme),
                        "gateway volume overrides")
                    # Confirm each JSON actually loaded — from an uploaded file OR the default on disk.
                    # 19es: full width of this column again. 19eq split it to seat the button
                    # beside the first caption; the button now lives in the LEFT column instead.
                    _cfg_txt_col = st.container()
                    # 19gg: the "Gateway Volume Overrides" line shares a ROW with the settings.yaml
                    # preview, so the preview's header sits horizontally in-line with it.
                    #
                    # WHY THE WHOLE _grow_box MOVES UP HERE rather than just the preview. The
                    # preview and the run log share ONE fixed-height reserve (_GROW_BOX_PX), which
                    # is what makes opening either one FREE — it scrolls inside the reserve instead
                    # of growing row 2 and shoving the pre/post table down (19ev/19fc). Giving the
                    # preview its own box beside the caption and leaving the log its own would
                    # DOUBLE the reserve to ~560 px, past the height of the left "4 · Assumptions"
                    # column, and the row would then grow permanently — paying every run for a
                    # tidier line. Moving the pair up costs nothing: the reserve is unchanged, it
                    # just sits beside the caption instead of below it, so the column gets one
                    # line SHORTER.
                    #
                    # THE ONE COST is width: the preview and the log now occupy this fraction of
                    # the right column instead of all of it. Raise it to widen them at the caption's
                    # expense; the caption wraps to two lines below roughly 0.45.
                    _CFG_ROW_SPLIT = [0.52, 0.48]
                    _gvo_col, _grow_col = st.columns(_CFG_ROW_SPLIT)
                    for _cfg_lbl, _cfg_val, _cfg_up in (
                            ("Test Gateways", test_gateways, test_gw_file),
                            ("Thermometer Config", thermometer_config, thermo_file),
                            ("Gateway Volume Overrides", gateway_volume_overrides, override_file)):
                        if _cfg_val:
                            _cfg_src = "uploaded" if _cfg_up is not None else "default file"
                            _cfg_n = len(_cfg_val)
                            # the last one renders in the ROW's left column; the first two stay
                            # full width, above the row.
                            (_gvo_col if _cfg_lbl == "Gateway Volume Overrides"
                             else _cfg_txt_col).markdown(
                                "<span style='color:#1D9E75; font-size:0.8rem; font-weight:700;'>"
                                f"✓ {_cfg_lbl} loaded</span>"
                                "<span style='color:var(--tav-muted); font-size:0.78rem;'> "
                                f"({_cfg_src}, {_cfg_n} entr{'y' if _cfg_n == 1 else 'ies'})</span>",
                                unsafe_allow_html=True)

                    # 19eq asks 2 + 3: both bodies live HERE, in the right column, so each is one
                    # column wide instead of the full page. The run log's fixed height keeps it
                    # scrolling inside its own box rather than growing the page.
                    # 19ev: BOTH live inside ONE fixed-height container, so opening either one
                    # scrolls within it and the row-2 height never changes — the pre/post table
                    # below stays exactly where it is. `_CODE_BOX_PX` still caps each code block
                    # inside; this caps the region they share.
                    # 19gg: created in `_grow_col` — the RIGHT half of the Gateway-Volume-Overrides
                    # row — instead of full width below it. Same container, same reserve, one row up.
                    # 19jz: the settings.yaml preview LEFT this reserve. It now renders below
                    # the pre/post table at the bottom of the tab (Ben's instruction: column
                    # width, beneath that table), so the box holds the run log alone.
                    #
                    # THE RESERVE STILL EARNS ITS KEEP, for the same reason as before: the log
                    # scrolls inside a fixed height instead of growing row 2 and shoving the
                    # pre/post table down. Nothing needs re-tuning — one occupant in a reserve
                    # sized for two just means the log gets all of it, which is more log.
                    _grow_box = _grow_col.container(height=_GROW_BOX_PX)
                    _fc_log_slot = _grow_box.container()

                # The forecast run log used to render HERE, in the row-2 right column - i.e.
                # ABOVE the "Calculate Forecast" button - so expanding it pushed the button down
                # the page. It is now created at the very BOTTOM of the tab instead - search for
                # the "RUN LOG, LAST" banner below.



            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Split Go Live date is shown in '1 · Run identity' (normal mode) or beside the
        # Forecast-outputs folder (previous-forecast mode) — always defined by here.

        # ---- assemble settings (always) ---------------------------------------
        forecast_settings = {
            "split_go_live_date": str(split_go_live),
            "company": company,
            "card_scheme": scheme,
            "month_0": str(m0_date),
            "month_var": month_var,
            "future_anchor_date": str(future_anchor_date),
            "use_cached_inputs": bool(use_cached_inputs),
            "cached_inputs_path": cached_inputs_path or None,
            "reuse_cached_curves": bool(reuse_cached_curves),
            "mid_list_file": ("data/mappings/Master_MID_List_Mastercard.csv"
                              if scheme == "mastercard"
                              else "data/mappings/Master_MID_List.csv"),
            "m0_total_transactions": int(m0_total),
            "m0_transaction_weightings": {k: int(v) for k, v in rpgt_assumed.items()},
            "use_live_actuals": bool(use_live_actuals),
            "start_date": str(actuals_start_date) if use_live_actuals else None,
            "end_date": str(actuals_end_date) if use_live_actuals else None,
            "force_actuals_for": force_actuals_for_rpgts,
            "t0_lookback_months": int(t0_lookback_months),
            "decay_factor": float(decay_factor),
            "thermometer_sample_months": int(thermometer_sample_months),
            "shrink_strength": float(shrink),
            "test_gateways": test_gateways,
            "thermometer_config_loaded": thermometer_config is not None,
            "gateway_volume_overrides_loaded": gateway_volume_overrides is not None,
            # The ACTUAL override dict must reach build_pipeline_config -> the pipeline's
            # AllocationEngine, so target:0 gateways (e.g. Cardworks, effective in June)
            # are killed from their effective_date in the BASELINE forecast (mid-month
            # pro-rated). Previously only the *_loaded flag was passed, so overrides = {}.
            "gateway_volume_overrides": gateway_volume_overrides or {},
        }
        # Persist forecast_settings every rerun (from the current Build Baseline widgets) so the
        # 'Validate Split' sub-tab is usable WITHOUT first running/loading a baseline — it builds
        # its own forecast via the pipeline using these settings.
        ss["forecast_settings"] = forecast_settings

        # White label text on this green primary button (default primary text is ink).
        st.markdown("""<style>
            .st-key-calc_cache_btn button, .st-key-calc_cache_btn button * { color: #ffffff !important; }
        </style>""", unsafe_allow_html=True)
        # 19eq: rendered into the slot reserved beside the config captions. Built here because
        # this is the first point where `forecast_settings` has been persisted and
        # `actuals_valid` is known; `st` is the fallback for previous-forecast mode, where ROW 2
        # (and therefore the slot) never rendered.
        _btn_ctx = _calc_btn_slot if _calc_btn_slot is not None else st
        _load_clicked = _btn_ctx.button(
            "Load forecast" if use_prev else "Calculate Forecast",
            type="primary", key="calc_cache_btn",
            disabled=(not settings_hidden) if use_prev else (not actuals_valid))
        # 19jz: the settings.yaml preview USED TO RENDER HERE, just below the Calculate
        # Forecast button and inside row 2's shared reserve — so it was ~48% of the right
        # column, about a quarter of the page, and the YAML wrapped badly in it. It now renders
        # at the very BOTTOM of this sub-tab, one COLUMN wide and beneath the pre/post table
        # (Ben's instruction). Search for "SETTINGS.YAML PREVIEW, LAST" below.
        #
        # Moving it DOWN is what makes the width free: 19fc's rule is that a thing which can
        # grow must come after everything it could push, and nothing follows the preview now.
        # ── RUN LOG ───────────────────────────────────────────────────────────────────────────
        # 19eq: the container is created UP IN THE ROW-2 RIGHT COLUMN, next to the config
        # captions — that is what makes it one column wide rather than full page, and it takes
        # its st.status header ("Forecast ready for …") with it, since header and body are one
        # widget. 19gg moved it one row further up, into the RIGHT HALF of the Gateway-Volume-
        # Overrides caption row, so the settings.yaml preview above it lines up with that caption.
        # It is therefore now _CFG_ROW_SPLIT[1] of the right column wide, not all of it.
        #
        # BUILD MODE ONLY. That arrangement holds because row 2's right column reserves a fixed
        # `_GROW_BOX_PX` for the preview + log pair, so opening either scrolls inside the reserve
        # instead of growing the row.
        # PREVIOUS-FORECAST MODE went back to "create it LAST" in 19fc — see the block below. The
        # reserve trick cannot work there: that mode's row is the folder input itself, so any
        # height the log column holds is height the baseline table below the row has to move for.
        #
        # REMAINS None in previous-forecast mode; the branch below builds that mode's slots.
        _pre_table_ctx = None
        if settings_hidden:
            # 19fc: PREVIOUS-FORECAST MODE — ORDER IS THE FIX.
            # The pre-vs-post baseline table's position is RESERVED FIRST, then the run log is
            # created BELOW it. Streamlit lays out in script order, so from here on the log can
            # only ever push things that come after it — and nothing does. Expanding it is free.
            # (Reserving the table is what makes this work: the table is written ~150 lines
            # further down, after the forecast has loaded, so without a reserved slot it would
            # land after the log and be the thing that gets pushed.)
            # No outer height cap on the log. It does not need one any more — the body already
            # scrolls inside its own `_CODE_BOX_PX` box — and a cap would only reintroduce
            # reserved whitespace for no benefit.
            _pre_table_ctx = st.container()
            # Half width, so the log stays one column wide as it is in build mode rather than
            # stretching across the page.
            _fc_log_slot = st.columns([0.5, 0.5])[0].container()
        if _load_clicked:
            # Render the run log into the BOTTOM slot (normal mode) or the previous-forecast
            # right column; fall back to full-width only if neither exists.
            _log_ctx = _fc_log_slot if _fc_log_slot is not None else st
            # 19eu: starts COLLAPSED. The trade-off is real and worth knowing: the log no longer
            # unrolls live while the forecast runs, so watching progress means clicking it open.
            # What is NOT lost is a failure — the error path below re-opens it with
            # `expanded=True`, so a run that breaks still shows its own log without being asked.
            # The completion path already collapsed it, so a finished run looks the same as before.
            with _log_ctx.status("Calculating & caching forecast...", expanded=False) as status:
                # 19et: the log scrolls inside its OWN fixed-height box (~10 lines) rather than
                # growing with the run. `log_area` is an st.empty() INSIDE that box, so every
                # rewrite lands in the same scroll region and the status header above it stays
                # put. Streamlit keeps a height-capped container scrolled where the user left
                # it, so this does not fight a user who has scrolled back to read something.
                log_area = st.container(height=_CODE_BOX_PX).empty()
                log_lines: list[str] = []

                # [FN-297]
                def log(msg):
                    log_lines.append(msg)
                    log_area.code("\n".join(log_lines[-300:]), language="log")

                handler = StreamlitLogHandler(log)
                root_logger = logging.getLogger()
                root_logger.addHandler(handler)
                prev_level = root_logger.level
                root_logger.setLevel(logging.INFO)
                pipeline_out_dir = None
                try:
                    # Echo the EXACT inputs this run used, so the log is self-documenting.
                    _fs = forecast_settings
                    _mode = ("live BigQuery pipeline" if run_live
                             else "load previously-run outputs" if load_baseline
                             else "synthesised from attempts (no BigQuery)")
                    log("── Input settings used ──")
                    log(f"   mode: {_mode}")
                    log(f"   company={_fs['company']} · scheme={_fs['card_scheme']} · month={_fs['month_var']} · "
                        f"month_0={_fs['month_0']}")
                    log(f"   split_go_live={_fs['split_go_live_date']} · future_anchor={_fs['future_anchor_date']}")
                    log(f"   use_live_actuals={_fs['use_live_actuals']} · "
                        f"actuals_window={_fs['start_date']} → {_fs['end_date']}")
                    log(f"   force_actuals_for={_fs['force_actuals_for'] or '(none)'}")
                    log(f"   reuse_cached_curves={_fs['reuse_cached_curves']}")
                    log(f"   use_cached_inputs={_fs['use_cached_inputs']} · "
                        f"cached_inputs_path={_fs['cached_inputs_path'] or '(none)'}")
                    log(f"   shrink_strength={_fs['shrink_strength']} · t0_lookback_months={_fs['t0_lookback_months']} · "
                        f"decay_factor={_fs['decay_factor']} · thermometer_sample_months={_fs['thermometer_sample_months']}")
                    log(f"   m0_total_transactions={_fs['m0_total_transactions']:,} · "
                        f"m0_weightings={_fs['m0_transaction_weightings']}")
                    log(f"   test_gateways set={bool(_fs.get('test_gateways'))} · "
                        f"gateway_volume_overrides set={bool(_fs.get('gateway_volume_overrides'))} · "
                        f"thermometer_config_loaded={_fs['thermometer_config_loaded']}")
                    log(f"   MID list: {_fs['mid_list_file']}")

                    log("── Forecast (risk / VAMP baseline) ──")
                    _, fc_src = None, "synthesised from attempts"
                    if run_live:
                        _is_mc = (scheme == "mastercard")
                        _scheme_lbl = "MASTERCARD" if _is_mc else "VAMP"
                        log(f"• Running {_scheme_lbl} pipeline (BigQuery); pipeline logs:")
                        try:
                            # 19ka: the `use_yaml_asis` arm that read config/<scheme>_settings.yaml
                            # off disk is gone with its checkbox. The config is BUILT from this
                            # tab's own settings, always - which is what the preview shows.
                            if _is_mc:
                                pipeline_config = build_mc_pipeline_config(forecast_settings)
                            else:
                                pipeline_config = build_pipeline_config(forecast_settings)
                            if _is_mc:
                                pipeline_out_dir = run_mastercard_pipeline(
                                    pipeline_config, PROJECT_ROOT, gcp_project=GCP_PROJECT)
                            else:
                                pipeline_out_dir = run_vamp_pipeline(
                                    pipeline_config, PROJECT_ROOT, gcp_project=GCP_PROJECT)
                            _, fc_src = pipeline_out_dir, f"{_scheme_lbl} pipeline (live)"
                            log(f"  pipeline outputs: {pipeline_out_dir}")
                        except Exception as exc:  # noqa: BLE001
                            import traceback as _tb
                            tb = _tb.format_exc()
                            status.update(label="VAMP pipeline FAILED", state="error",
                                          expanded=True)
                            st.error(f"VAMP pipeline failed: {type(exc).__name__}: "
                                     f"{exc}. No synthesised baseline was used.")
                            st.markdown("**Full pipeline log** (last line = step it "
                                        "reached before failing):")
                            st.code("\n".join(log_lines) or "(no logs captured)")
                            st.markdown("**Traceback** (bottom `file:line` = exact "
                                        "failure point):")
                            st.code(tb)
                            root_logger.removeHandler(handler)
                            root_logger.setLevel(prev_level)
                            st.stop()
                    elif load_baseline and cached_inputs_path:
                        _, fc_src = cached_inputs_path, "VAMP pipeline pre (cached)"
                        pipeline_out_dir = (cached_inputs_path
                                            if os.path.isdir(cached_inputs_path)
                                            else os.path.dirname(cached_inputs_path))
                        log(f"• Using cached pipeline baseline: {cached_inputs_path}")
                    else:
                        log("• No pipeline output — baseline will not be built here; "
                            "tab 3 will fetch success data and build routing profiles.")

                finally:
                    root_logger.removeHandler(handler)
                    root_logger.setLevel(prev_level)
                    # Persist the full run log to a folder so it survives tab switches / app restarts
                    # and can be shared when diagnosing a run (success OR failure both land here).
                    try:
                        import datetime as _dt
                        # 19fh: logs/<Company>/<scheme>/, and the SCHEME is in the filename too.
                        # Flat logs/ mixed every brand and both card schemes into one 286-file
                        # directory, and nothing in a filename said which scheme a run was — so a
                        # visa and a mastercard run of the same brand and month were
                        # indistinguishable except by opening them. The folders make the set
                        # navigable; the filename keeps a log self-describing once it is copied
                        # out of its folder, which is how they actually get shared.
                        _co = str(forecast_settings.get("company", "run")).replace(" ", "")
                        _sc = str(forecast_settings.get("card_scheme", "visa") or "visa").strip().lower()
                        _mo = str(forecast_settings.get("month_var", ""))
                        # 19ge: logs/ and runs/ consolidated under ONE logs/ tree —
                        # logs/forecast_logs/ (this) and logs/engine_logs/ (the run bundles).
                        _logs_dir = os.path.join(PROJECT_ROOT, "logs", "forecast_logs", _co, _sc)
                        os.makedirs(_logs_dir, exist_ok=True)
                        _log_path = os.path.join(
                            _logs_dir,
                            f"forecast_{_co}_{_sc}_{_mo}_{_dt.datetime.now():%Y%m%d_%H%M%S}.log")
                        with open(_log_path, "w", encoding="utf-8") as _lf:
                            _lf.write("\n".join(log_lines) + "\n")
                        ss["last_forecast_log_path"] = _log_path
                        log(f"── run log saved: {_log_path}")
                    except Exception as _lge:  # noqa: BLE001
                        log(f"[warning] could not save run log to disk ({type(_lge).__name__}: {_lge})")

                status.update(
                    label=f"Forecast ready for {company} ({month_var}); baseline: {fc_src}",
                    state="complete", expanded=False)

            # Clear downstream artifacts so old runs don't linger. Tab 3 owns success
            # rates and routing profiles now, so we clear them too — the user runs tab 3
            # to pull attempts data, pre-process (Bayesian smoothing + time decay),
            # and generate split variations.
            for k in ("problems", "sr", "forecast", "variations",
                      "selected_variation_weight", "split", "settings", "frontier",
                      "compressed", "elbow", "stats", "configs"):
                ss.pop(k, None)

            ss["forecast_settings"] = forecast_settings
            ss["thermometer_config"] = thermometer_config
            ss["gateway_volume_overrides"] = gateway_volume_overrides
            ss["pipeline_out_dir"] = pipeline_out_dir

        # Baseline forecast — VI Txn & VAMP by month × vampMid (PRE months only), shown once a forecast
        # has been calculated/cached/loaded, in the SAME tab-3 table format + conditional formatting
        # (reuses tab_1_2_validate_split's renderer). Reads mid_level.csv from the cached forecast output dir.
        # In previous-forecast mode the table renders into the LEFT column created above (log sits in
        # the right column); otherwise it renders full-width via a throwaway container.
        _tbl_ctx = _pre_table_ctx if _pre_table_ctx is not None else st.container()
        with _tbl_ctx:
            try:
                _fc_out = ss.get("pipeline_out_dir")
                _mid_csv = os.path.join(_fc_out, "mid_level.csv") if _fc_out else None
                if _mid_csv and os.path.isfile(_mid_csv):
                    from tab_1_2_validate_split import _to_prepost as _tpp, _render_prepost_table as _rpt
                    _pre = _tpp(pd.read_csv(_mid_csv))
                    _pre = _pre[[c for c in _pre.columns if "Post" not in c]]   # PRE months only
                    _rpt(_pre, fit_content=True, bold=False)   # hug content; values not bold (tab-1 baseline)
            except Exception as _e:  # noqa: BLE001
                st.caption(f"(baseline VI/VAMP table unavailable: {type(_e).__name__}: {_e})")

        # ── SETTINGS.YAML PREVIEW, LAST ──────────────────────────────────────────────────
        # 19jz, on Ben's instruction: BENEATH the pre/post table, and one COLUMN wide rather
        # than the quarter-page slice it had inside row 2's reserve.
        #
        # Build mode only. In previous-forecast mode there is no assembled settings.yaml to
        # preview — that mode runs from a folder of outputs someone else's run produced — so
        # the expander is absent rather than empty.
        #
        # Being last is deliberate and is the whole reason it can be this wide: expanding it
        # can only push things BELOW it, and there is nothing below it. That is 19fc's ordering
        # rule, which is also why the pre/post table's slot is reserved above.
        if not use_prev:
            _yaml_col = st.columns(2)[0]
            # 19eu: `expanded=False` stated rather than relied on. It is st.expander's default,
            # but a default is not a decision — this one is, so it is written down.
            with _yaml_col.expander("Preview assembled settings.yaml (VAMP pipeline schema)",
                                    expanded=False):
                pipeline_config = build_pipeline_config(forecast_settings)
                # 19et: the YAML scrolls INSIDE a fixed-height box (~10 lines) instead of
                # rendering its full length. The download button stays OUTSIDE the box so it is
                # always reachable without scrolling to the bottom of the config.
                st.container(height=_CODE_BOX_PX).code(
                    yaml.safe_dump(pipeline_config, sort_keys=False), language="yaml")
                st.download_button("Download settings.yaml",
                                   yaml.safe_dump(pipeline_config, sort_keys=False),
                                   file_name="vamp_settings.yaml", mime="text/yaml")


    with _vs:
        import tab_1_2_validate_split
        tab_1_2_validate_split.render(ss, PROJECT_ROOT, GCP_PROJECT)
