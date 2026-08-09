"""Tab 2 — Routing engine.

Extracted verbatim from streamlit_app.py (behaviour unchanged) into its own file so the main
script stays small. streamlit_app.py calls `render()` from inside `with tab_eng:`.
"""
from __future__ import annotations

import datetime
import logging
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from routing_optimiser import (HardConstraints, OptimiserSettings, SoftConstraints,
                               build_cell_problems, detect_blocked_gateways, engine_choices,
                               gateway_success_rates, load_forecast, load_success_data,
                               optimise_split, portfolio_summary, run_sql_file)
from impact_calcs import _mtime, pool_targeted_compression, process_wallet_incapable

from app_common import (ss, PROJECT_ROOT, SQL_DIR, CACHE_DIR, GCP_PROJECT, StreamlitLogHandler,
                        _switched_off_gateways, APP_BUILD, DEFAULT_GATEWAY_FIDS, _GA_N_SEED,
                        _apply_blocked_caps, _ensure_base_30d_metrics, _fmt_secs,
                        _impact_eval_frame, _ink_caption, _load_ga_perf, _physical_cpu_count,
                        _save_ga_perf, _variance_gap_temp)


# [FN-298]
def render():
    out_dir = ss.get("pipeline_out_dir")

    # Mid-run STOP control (file-based cooperative stop). Writing the signal makes a running genetic
    # search halt at the next generation and keep the best split so far. Streamlit runs the compute
    # synchronously in the ACTIVE tab, so trigger this from a SECOND browser tab/window (a separate
    # session) — or run `touch <project>/runs/_stop` — to stop a run already in progress.
    with st.sidebar:
        if st.button("⛔ Stop genetic run", key="ga_stop_btn",
                     help="Halt a running genetic search at the next generation, keeping the best "
                          "split so far. Because the compute runs synchronously in the active tab, "
                          "trigger this from a SECOND browser tab (or `touch runs/_stop`) to stop a "
                          "run already in progress."):
            try:
                from routing_optimiser.run_bundle import request_stop as _rq_stop
                _rd_stop = os.path.join(PROJECT_ROOT, "runs")
                os.makedirs(_rd_stop, exist_ok=True)     # so the signal path is always runs/_stop
                _rq_stop(_rd_stop)
                st.sidebar.warning("Stop sent — the genetic search will halt at the next generation.")
            except Exception as _e:  # noqa: BLE001
                st.sidebar.error(f"Stop failed: {type(_e).__name__}: {_e}")

    if "pipeline_out_dir" not in ss:
        st.info("Cache a forecast in tab 1 first.")
    elif not out_dir:
        st.info("Cache a forecast in tab 1 first. (Forecast path is empty).")
    elif not os.path.isdir(out_dir):
        st.error(f"**Path Error:** Tab 1 saved a forecast path but Python cannot find it as a valid folder on disk.\n\n"
                 f"**Path checked:** `{out_dir}`\n\n"
                 f"**Current Working Directory:** `{os.getcwd()}`")
    else:
        st.markdown("""<style>
            .stSlider [data-testid="stTickBar"] > div,
            .stSlider [data-testid="stThumbValue"],
            .stSlider div[role="slider"] > div { color: var(--tav-ink) !important; }
        </style>""", unsafe_allow_html=True)

        # --- Engine-form polish: square corners, checkbox↔input centring, no chip
        #     truncation, run-log box sizing. Scoped to keyed widgets / the form. ---
        st.markdown("""<style>
            /* Square the number inputs (incl. their +/- stepper buttons) */
            .st-key-decay_half_inp [data-baseweb="input"],
            .st-key-vamp_cap_inp [data-baseweb="input"],
            .st-key-xborder_inp [data-baseweb="input"],
            .st-key-max_configs_inp [data-baseweb="input"],
            .st-key-decay_half_inp [data-baseweb="input"] *,
            .st-key-vamp_cap_inp [data-baseweb="input"] *,
            .st-key-xborder_inp [data-baseweb="input"] *,
            .st-key-max_configs_inp [data-baseweb="input"] * { border-radius: 0 !important; }
            /* Square the MID-constraints grid */
            .st-key-mid_constraints_editor,
            .st-key-mid_constraints_editor [data-testid="stDataFrameResizable"],
            .st-key-mid_constraints_editor [data-testid="stDataFrame"] { border-radius: 0 !important; }
            /* RPGT multiselect: never truncate a tag's text */
            .st-key-eng_rpgt_scope [data-baseweb="tag"],
            .st-key-eng_rpgt_scope [data-baseweb="tag"] span { max-width: none !important;
                overflow: visible !important; text-overflow: clip !important; }
            /* Run-log dark box: square + capped height with its own scroll (matches the
               MID-constraints box); it sits in the form's Run-Log column. */
            div[data-testid="stForm"] [data-testid="stCode"] { border-radius: 0 !important; }
            div[data-testid="stForm"] [data-testid="stCode"] > pre { border-radius: 0 !important;
                max-height: 440px; overflow: auto !important; }
            /* Vertically centre the checkbox against its sibling INPUT box (not the label
               above it) — push it down by roughly one label's height. */
            .st-key-apply_decay_cb { margin-top: 1.8rem !important; }
            .st-key-vamp_on_cb { margin-top: 1.9rem !important; }
        </style>""", unsafe_allow_html=True)

        fs = ss.get("forecast_settings", {}) or {}

        # Entropy is retired from the UI; 'genetic_ref' (the revenue reference) is not offered as a
        # standalone engine — it stays in the backend as the genetic engine's internal reference.
        choices = [(k, lbl) for k, lbl in engine_choices() if k not in ("entropy", "genetic_ref")]
        # The Genetic algorithm is the cross-cell per-vampMid tilt GA (genetic_global.run_midtilt_ga),
        # dispatched directly by the app (not a registry engine). CONSOLIDATED: the former separate
        # 'genetic' (NumPy) and 'genetic_numba' options are gone — there is now ONE genetic engine:
        # the Numba fused kernel (verify-or-fallback to NumPy, so it can never produce a different
        # split) run on a rule-safe PRE-CLUSTERED problem (near-lossless; see routing_optimiser.
        # precluster). It keys as 'genetic_numba' so all engine logic is unchanged; pre-clustering is
        # turned on for it below (ss['ga_precluster']; set ROUTING_GA_PRECLUSTER=0 to disable as an
        # escape hatch). If Numba is unavailable it self-falls-back to the NumPy path.
        if "genetic_numba" not in {k for k, _ in choices}:
            choices.append(("genetic_numba", "Genetic algorithm (Numba + pre-clustering)"))
        labels = {k: lbl for k, lbl in choices}
        keys = [k for k, _ in choices]
        # Default to the Genetic algorithm (the production engine); fall back to softmax/first.
        default_idx = (keys.index("genetic_numba") if "genetic_numba" in keys
                       else (keys.index("softmax") if "softmax" in keys else 0))
        # Sections 1 & 2 sit side by side (Engine Type 1/3 | Data & Pre-Processing 2/3),
        # BOTH outside the form so the engine / method selectors show/hide inputs live.
        _ga_auto = True
        _ga_pop, _ga_gen = None, None   # auto-sized from the problem at compute time
        sr_company = fs.get("company", "TotalAV")   # company & scheme come from tab 1
        sr_scheme = fs.get("card_scheme", "visa")
        trace_gateway = ""  # gateway trace disabled
        today = date.today()
        yesterday = today - datetime.timedelta(days=1)

        # Everything the engine needs sits in ONE form: changing any input / dropdown /
        # checkbox no longer reruns the tab — it only re-evaluates when you click the
        # "Compute split variations" submit button. The two column rows are created INSIDE
        # the form so every widget in them (sections 1-4) is a form member; they're then
        # filled in the `with _c_*:` blocks below (Streamlit binds by container lineage,
        # not code nesting). Trade-off: engine-specific settings (e.g. the softmax
        # temperature slider) now reveal on submit rather than instantly — a non-issue for
        # the default Genetic engine, which has no extra settings.
        # --- LIVE genetic search-budget panel (OUTSIDE the form) --------------------------
        # The candidate-split count / ETA depend only on seeds × restarts × generations ×
        # population (λ). Those four inputs are rendered here, ABOVE the form, so editing any of
        # them reruns the tab and refreshes the readout IMMEDIATELY — widgets inside a form don't
        # rerun until submit, which froze the old in-form readout at its last-submitted values.
        # They write to the same session_state keys the compute path reads via ss.get, so nothing
        # downstream changes. Only shown for the genetic engine; engine is read from session_state
        # (default = the genetic engine 'genetic_numba') since the engine selector lives in the form.
        # Must default to a key that EXISTS in the dropdown — after consolidation the genetic key is
        # 'genetic_numba' (the old 'genetic' is gone), so defaulting to it keeps this panel visible
        # on a fresh session where engine_key_select isn't set yet.
        _pre_keys = [k for k, _ in choices]
        _pre_default = ("genetic_numba" if "genetic_numba" in _pre_keys
                        else ("softmax" if "softmax" in _pre_keys else (_pre_keys[0] if _pre_keys else "")))
        _pre_engine = str(ss.get("engine_key_select", _pre_default) or _pre_default)
        if _pre_engine not in _pre_keys:      # stale session value (e.g. a removed key) → default
            _pre_engine = _pre_default
        if _pre_engine in ("genetic", "genetic_numba"):
            _cpu_seeds_default = max(1, min(_physical_cpu_count(), 16))
            # [FN-299]
            def _budget_panel():
                with st.container(border=True):
                    st.markdown("##### Genetic search budget")
                    _bc1, _bc2 = st.columns(2)
                    _bc1.number_input(
                        "GA generations", min_value=20, max_value=400, value=150, step=10,
                        key="ga_generations",
                        help="How many evolution rounds the search runs. More = potentially better plans, "
                             "but slower. 150 = default.")
                    _bc2.number_input(
                        "No Candidate Splits", min_value=0, max_value=300, value=64, step=10,
                        key="ga_pop_override",
                        help="Candidate plans per generation (λ). 64 = default. 0 = auto-size from the "
                             "problem. Larger = more thorough, slower.")
                    _bc3, _bc4 = st.columns(2)
                    _bc3.number_input(
                        "Number of seeds", min_value=1, max_value=16, value=_cpu_seeds_default, step=1,
                        key="ga_n_seeds",
                        help="Independent CMA-ES searches from different random starts; the fittest is kept. "
                             "Defaults to your PHYSICAL core count so seeds run one-per-real-core in a "
                             "single parallel wave (avoids hyperthread oversubscription that thrashes "
                             "memory bandwidth). Applies to both Genetic and GA - Numba.")
                    _bc4.number_input(
                        "Restarts per seed", min_value=1, max_value=10, value=2, step=1,
                        key="ga_restarts",
                        help="Each seed re-launches from a fresh spread this many times, keeping the best. "
                             "More = better odds of the best split, longer. 2 = default.")
                    st.checkbox(
                        "Run all generations (no early-stopping)", value=False, key="ga_no_early_stop",
                        help="OFF (default): each seed stops early once it converges (no improvement for a "
                             "while, or the search collapses to a point) — saves time and rarely changes the "
                             "result. ON: every seed runs the FULL generation cap regardless, so the candidate "
                             "count becomes exact (= seeds × restarts × generations × λ) and the run is longer, "
                             "usually with no better split. Use for a deterministic, maximum-budget search.")
                    with st.expander("Advanced — step size (CMA-ES σ)", expanded=False):
                        st.caption("CMA-ES already auto-adapts the step size every generation; these only "
                                   "nudge its starting point and limits. σ₀×1.5 and damping×1.5 are the "
                                   "tuned defaults for this problem (σ floor stays a no-op at 0) — leave them "
                                   "unless you're deliberately experimenting, and compare runs on the "
                                   "EFFICIENCY (score/min) figure the log prints.")
                        _sc1, _sc2, _sc3 = st.columns(3)
                        _sc1.number_input(
                            "σ₀ multiplier", min_value=0.25, max_value=4.0, value=1.5, step=0.25,
                            key="ga_sigma0_mult",
                            help="Scales the INITIAL step size. >1 = wider first strides (more early "
                                 "exploration, can help escape a plateau but wastes some early gens); "
                                 "<1 = tighter start (faster local convergence, risks settling early). "
                                 "1.0 = default (no change).")
                        _sc2.number_input(
                            "Min step-size (σ floor)", min_value=0.0, max_value=0.5, value=0.0, step=0.01,
                            format="%.2f", key="ga_sigma_floor",
                            help="Stops the step size collapsing below this value. >0 keeps the search "
                                 "moving (more exploration, avoids premature fine-convergence) but prevents "
                                 "the final polishing. Typical σ_final on this problem is ~0.05, so try "
                                 "0.02–0.05 to probe the plateau. 0 = default (let σ collapse naturally).")
                        _sc3.number_input(
                            "σ damping ×", min_value=0.5, max_value=3.0, value=1.5, step=0.25,
                            key="ga_damps_mult",
                            help="How SLOWLY the step size is allowed to change. >1 = steadier, more "
                                 "cautious σ moves (smoother, slower to react); <1 = faster, more "
                                 "aggressive adaptation. 1.0 = default (no change).")
                    # Candidate-split RANGE + ETA — same maths as before, just recomputed live here.
                    # Restart strategy stays in the form, so its last-submitted value (Lean default) is
                    # read from session_state; Lean keeps λ constant → floor == ceiling (single value).
                    _lean_ui = str(ss.get("ga_restart_mode", "Lean")).startswith("Lean")
                    _bud_seeds = max(1, int(ss.get("ga_n_seeds", _cpu_seeds_default) or _cpu_seeds_default))
                    _bud_rst = max(1, int(ss.get("ga_restarts", 4) or 4))
                    _bud_gen = int(ss.get("ga_generations", 80) or 80)
                    _bud_pop = int(ss.get("ga_pop_override", 0) or 0)
                    _bud_lam = _bud_pop if _bud_pop > 0 else 64
                    #   floor   = seeds × restarts × generations × λ  (fixed λ)
                    #   ceiling = seeds × generations × Σ min(λ·2^e, 4λ)  — IPOP doubles λ per restart (cap 4λ)
                    _bud_floor = _bud_seeds * _bud_rst * _bud_gen * _bud_lam
                    _ceil_mult = (int(_bud_rst) if _lean_ui
                                  else sum(min(2 ** _e, 4) for _e in range(max(int(_bud_rst), 1))))
                    _bud_ceil = _bud_seeds * _bud_gen * _bud_lam * _ceil_mult
                    # Once calibrated, narrow to last run's realization ratio scaled to this floor, ±15%.
                    _ratio = float(ss.get("last_ga_ratio", 0.0) or 0.0)
                    if _ratio > 0:
                        _exp = _bud_floor * _ratio
                        _lo_c, _hi_c = _exp * 0.85, _exp * 1.15
                    else:
                        _lo_c, _hi_c = float(_bud_floor), float(_bud_ceil)
                    # End-to-end ETA, calibrated from the last run (fixed overhead + candidates ÷ rate).
                    _lc = int(ss.get("last_ga_cands", 0) or 0)
                    _lsecs = float(ss.get("last_ga_secs", 0.0) or 0.0)
                    _ltot = float(ss.get("last_total_secs", 0.0) or 0.0)
                    _rate = (_lc / _lsecs) if (_lc > 0 and _lsecs > 0) else 0.0
                    _fixed = max(0.0, _ltot - _lsecs) if _ltot > 0 else 0.0

                    # [FN-300]
                    def _fmt_eta(_secs):
                        _m = _secs / 60.0
                        return f"{_m:.0f} min" if _m >= 1.5 else f"{max(1, int(_secs))}s"
                    if _rate > 0:
                        _hi = _fixed + _hi_c / _rate
                        _eta_sub = (f"est. end-to-end up to ~{_fmt_eta(_hi)}"
                                    + (f" (last run {_fmt_eta(_ltot)} total)" if _ltot > 0 else ""))
                    else:
                        _eta_sub = "est. time: run once to calibrate the throughput"
                    # Show a single UPPER BOUND ("up to X") rather than a floor–ceiling band: early-stop
                    # only ever lands the run BELOW this, so it's the honest one-number cap. _hi_c is the
                    # calibrated high end once a run exists, else the theoretical ceiling.
                    _rng_lbl = ("Candidate splits (est. max)" if _ratio > 0 else "Candidate splits (max)")
                    st.markdown(
                        "<div style='font-size:0.78rem; color:var(--tav-muted); line-height:1.2;'>"
                        f"{_rng_lbl}<br>"
                        f"<span style='font-size:1.1rem; font-weight:800; color:var(--tav-ink);'>"
                        f"up to {int(_hi_c):,}</span>"
                        f"<div style='font-size:0.72rem;'>{_eta_sub}</div></div>", unsafe_allow_html=True)
            # Fragment: editing gen-cap/λ/seeds/restarts re-runs ONLY this panel (live count + ETA),
            # not the whole app. The panel is now RENDERED BELOW the engine form (the call sits just
            # after the form) so it sits beneath '1. Engine Type & Settings' while staying live —
            # kept outside the form, so its readouts refresh immediately rather than only on submit.

        _engine_form = st.form("engine_master_form", border=False)
        with _engine_form:
            _c_eng, _c_data = st.columns([1, 1])
            _c_rc, _c_log = st.columns([1, 1])
        with _c_eng:
            with st.container(border=True):
                st.markdown("##### 1. Engine Type and Settings")
                engine_key = st.selectbox("Split engine", keys, index=default_idx,
                                          format_func=lambda k: labels[k], key="engine_key_select",
                                          help="Method used to choose the split.")
                # The single genetic engine ('genetic_numba') ALWAYS runs Numba + pre-clustering.
                # Record the intent in ss — the GA block reads ss['ga_precluster'] to swap in the
                # pre-clustering wrapper at the call. Escape hatch: ROUTING_GA_PRECLUSTER=0 disables
                # pre-clustering (falls back to the plain Numba search) for debugging / A/B.
                ss["ga_precluster"] = (engine_key == "genetic_numba"
                                       and os.environ.get("ROUTING_GA_PRECLUSTER", "1") != "0")
                # These two flags keep the rest of the tab from special-casing the genetic key.
                # ('genetic' is retained here only so any legacy session/state value still resolves.)
                _is_genetic = engine_key in ("genetic", "genetic_numba")
                _use_numba = (engine_key == "genetic_numba")
                # (Bypass-enforcement checkbox removed — the enforcement layer no longer gates the
                # delivered split; the GA search output is delivered directly. See the delivery site.)

                st.divider()
                # ---- Split scope & grain (RPGTs, grain, hold-others, auto-explore) ----
                # RPGT options: prefer a previous pro-rata export, then the current split, then a
                # canonical fallback list. Computed here so the RPGT widgets AND the per-MID
                # constraint editor (further down, in the form) both see `_rpgt_opts`.
                _rpgt_opts = []
                _pp_c = os.path.join(out_dir, "vamp_t_period_prorata_export.csv")
                try:
                    if os.path.exists(_pp_c):
                        _ppc = pd.read_csv(_pp_c, usecols=lambda c: c.strip().lower() == "rpgt")
                        _rc = next((c for c in _ppc.columns if c.strip().lower() == "rpgt"), None)
                        if _rc:
                            _rpgt_opts = sorted(_ppc[_rc].dropna().astype(str).unique().tolist())
                except Exception:
                    _rpgt_opts = []
                if not _rpgt_opts:
                    _spl0 = ss.get("split")
                    if _spl0 is not None and "rpgt" in getattr(_spl0, "columns", []):
                        _rpgt_opts = sorted(_spl0["rpgt"].dropna().astype(str).unique().tolist())
                if not _rpgt_opts:
                    _rpgt_opts = ["Monthly Initial", "Annual Sub Sale", "Addon Sale", "Upgrade",
                                  "Monthly Renewal", "Annual Sub Renewal", "P6M Renewals", "Addon Renewal"]
                _rpgt_selected = st.multiselect(
                    "RPGTs to include in this split", options=_rpgt_opts, default=_rpgt_opts,
                    key="eng_rpgt_scope",
                    help="Only these transaction types feed the engine and appear in the proposed "
                         "split, VAMP and impact. Leave all selected to route every RPGT.")
                # These two are ALWAYS ON now (checkboxes removed). Keep both the local vars and
                # the session_state keys True so every downstream reader (locals + ss.get) agrees.
                _rpgt_hold_others = True   # hold unselected RPGTs at their current split (pre = post)
                _auto_explore = True       # auto-explore capable-but-untested gateways
                ss["eng_rpgt_hold_others"] = True
                ss["eng_auto_explore"] = True
                # Exploration min cell volume gate REMOVED (input deleted) and pinned to 0 = explore
                # all. Non-zero left single-gateway cells with an UNSATISFIABLE 97% max-share cap
                # (a lone gateway is forced to 100%), which floored the GA at a large structural
                # violation and made every run infeasible. 0 injects a fallback gateway into every
                # single-gateway cell, so the search stays feasible.
                ss["explore_min_cell_vol"] = 0
                _grc1, _grc2 = st.columns(2)
                _score_grain = _grc1.selectbox(
                    "Engine Score grain",
                    ["Bank × Currency", "Bank × Currency × RPGT"], index=0, key="eng_score_grain",
                    help="How the gateway SUCCESS RATE is pooled. Bank × Currency: one blended rate per "
                         "gateway per bank×currency (pools all RPGTs — more data, stabler). Bank × Currency "
                         "× RPGT: a separate rate per RPGT (more specific, thinner data → noisier).")
                _opt_grain = _grc2.selectbox(
                    "Optimisation grain",
                    ["Bank × Currency", "Bank × Currency × RPGT"], index=1, key="eng_opt_grain",
                    help="The cell grain at which the split is MADE and traffic is MOVED to meet risk "
                         "constraints. Bank × Currency: ONE split per bank×currency applied across RPGTs. "
                         "Bank × Currency × RPGT: a separate split per RPGT, with the VAMP cap enforced at "
                         "the per-RPGT level. Can differ from the score grain (e.g. score Bank×Currency for "
                         "a stable rate, optimise per RPGT). A score finer than the optimisation is pooled "
                         "up; a score coarser is broadcast to each RPGT cell.")

                # --- Engine settings / temperature (moved here from Data & Pre-Processing) ---
                params = {}
                temp_method, softmax_temperature = "Manual", 0.17
                # Genetic and Thompson have no engine settings — show nothing (no header, no caption).
                if engine_key not in ("genetic", "genetic_numba", "thompson"):
                    st.divider()
                    st.markdown("**Engine settings**")
                if engine_key == "softmax":
                    temp_method = st.selectbox(
                        "Temperature Method", ["Variance-Scaled (auto)", "Manual"],
                        help="Variance-Scaled (auto, recommended): per-cell temperature from the "
                             "significance of the best-vs-2nd-best gateway gap. Manual: one fixed "
                             "temperature for every cell.")
                    if temp_method == "Manual":
                        softmax_temperature = st.slider(
                            "Softmax temperature", 0.005, 0.3, 0.17, 0.005,
                            help="One fixed temperature applied to every cell.")
                elif engine_key == "thompson":
                    pass   # no dials / caption for Thompson
                elif engine_key == "portfolio":
                    _ink_caption("No dials. Reference maximises conversion minus the DOWNSIDE (CVaR) of "
                                 "each gateway's VAMP spiking; risk aversion auto-calibrated. Softmax "
                                 "temperature does not apply.")
                elif _is_genetic:
                    # (Optimisation objective dropdown removed — the GA ALWAYS maximises the
                    # VOLUME-WEIGHTED SUCCESS RATE, i.e. maximise Σ share·volume·SR.)
                    # Compliance target is FIXED to the GA / CMA-ES risk-minimised endpoint — the dropdown
                    # was removed because its only other option (Max approvals / greedy+LP) didn't use the
                    # GA. It maximises approvals subject to compliance (no extra risk-minimisation).
                    # Risk-aversion dial REMOVED — this engine ALWAYS runs the revenue-shaped risk-min
                    # endpoint (equivalent to ga_risk_aversion = 0: no EXTRA risk-minimisation beyond
                    # meeting the caps, closest to Max approvals). The value is forced to 0.0 at the GA
                    # call site, so removing the widget doesn't fall back to a non-zero default.
                    # (Cap-breach penalty input removed — FIXED at 0: the GA no longer penalises a
                    # split going over a VAMP/volume cap, consistent with enforcement being removed.
                    # Band penalty strength input removed — FIXED at 1.0.)
                    st.selectbox(
                        "Restart strategy",
                        ["Lean (constant-λ, spread-out)", "IPOP (thorough)"],
                        index=0, key="ga_restart_mode",
                        help="How each seed's restarts work. Lean (default): keep the population the SAME "
                             "size every restart and send each one to the least-searched region of the "
                             "space (coordinated coverage) — keeps the 'don't get stuck' benefit at "
                             "roughly linear cost. IPOP: reseed from the best-so-far and DOUBLE the "
                             "population (λ) each restart — more thorough on bumpy/multimodal problems, "
                             "but the doubling makes each restart cost more than the last. A/B them using "
                             "the ④ efficiency readout in the run log. Applies to both Genetic and "
                             "GA - Numba.")
                    # Re-projection budget input + the post-GA correction removed entirely — band scoring
                    # is EXACT in-search, so the search satisfies the true bands directly. The delivered
                    # split still gets a READ-ONLY true-band readout (+ incidence self-check) in the log.
                    # (GA parallelism dropdown removed — the multi-seed GA ALWAYS runs on loky
                    # worker processes. If loky can't run, the run fails loudly; there is no
                    # threading / sequential fallback.)
                    # EXACT band scoring is now the PERMANENT default (no toggle) — the search scores the
                    # true pro-rata projection per generation, pre-clustering off, with an incidence
                    # self-check on the delivered split and automatic fall-back to the proxy on any setup
                    # failure. See the ctx build (_ga_bands present ⇒ exact bands on).
                    # Search-BUDGET inputs (GA generations, population λ, seeds, restarts) plus the
                    # candidate-split range / ETA readout now live ABOVE this form (the "Genetic search
                    # budget" panel) so editing them refreshes the count LIVE — form members don't rerun
                    # until submit, which is why the old in-form readout stayed frozen until you ran.
                    # (Diverse-seed search checkbox removed — it is now ALWAYS on: the parallel
                    # seeds are always spread across the revenue↔risk axis with varied
                    # explore/exploit settings. Same seeds/generations/restarts/population, so the
                    # one-wave wall time is unchanged.)
                else:
                    _ink_caption(f"No settings for the {labels.get(engine_key, engine_key)} engine.")
                # The risk↔conversion tradeoff is expressed by the dial variations, so the reference
                # stays pure-conversion (no separate γ risk-aversion knob).

                # --- Split-shaping sliders (moved here from Data & Pre-Processing) ---
                st.markdown("""<style>
                .st-key-max_share_sld [data-testid*="TickBar"],
                .st-key-floor_sld [data-testid*="TickBar"] { display: none !important; }
                </style>""", unsafe_allow_html=True)
                _ms1, _ms2 = st.columns(2)
                max_share = _ms1.slider(
                    "Max share per gateway", 0.5, 1.0, 0.97, 0.01, key="max_share_sld",
                    help="No single gateway may take more than this share of a cell.")
                floor = _ms2.slider(
                    "Exploration floor (%)", 0.0, 5.0, 1.0, 0.25, key="floor_sld",
                    help="Every eligible gateway keeps at least this share, so none goes dark.") / 100.0

                # --- Compression shaping (drives the pool compression in tab 3 / configs) ---
                _cm1, _cm2 = st.columns(2)
                _cm1.selectbox(
                    "Compression clustering", ["kmeans", "ward"], index=1, key="compress_method",
                    help="Ward hierarchical (default) — builds one tree, so any distinct-split count "
                         "is an instant tree-cut with no re-fit; or k-means.")
                _cm2.selectbox(
                    "Budget allocation", ["greedy", "knapsack"], index=1, key="compress_allocation",
                    help="Knapsack (default) — exactly maximises retained fidelity for the pool "
                         "budget; or greedy (faster, one-step-ahead). Knapsack is a bit slower.")

        with _c_data:
            with st.container(border=True):
                st.markdown("##### 2. Data & Pre-Processing")
                # Plain container so the inputs stack vertically in this half-width column while
                # one-level column rows inside (dates / cross-border+pools / decay) are allowed —
                # columns can't nest more than one level deep.
                _dpp_l = st.container()
                with _dpp_l:
                    _ds1, _ds2 = st.columns(2)
                    attempts_start = _ds1.date_input(
                        "Start date", value=yesterday - datetime.timedelta(days=14),
                        help="First day of results to analyse.")
                    attempts_end = _ds2.date_input("End date", value=yesterday,
                                                   help="Last day of results to analyse.")
                    if attempts_start >= attempts_end:
                        st.error("⚠ Start date must be before the End date.")
                    # Cross-border penalty + Max pools, directly beneath the date range.
                    _xb1, _xb2 = st.columns(2)
                    xborder_penalty = _xb1.number_input(
                        "Cross-border penalty (%)", min_value=0.0, max_value=100.0, value=60.0, step=5.0,
                        key="xborder_inp",
                        help="Gateways flagged isCrossBorder = TRUE in Master_MID_List have their Engine "
                             "Score multiplied by this %, lowering their proposed share. 60% turns 60% into 36%.") / 100.0
                    max_configs = _xb2.number_input(
                        "Max pools (0 = no compression)", min_value=0, max_value=20000, value=500, step=50,
                        key="max_configs_inp",
                        help="Target MAX number of ConnectorPool config files (pools) to deploy. A "
                             "volume-weighted k-means trims the split, and the pool target is met by "
                             "searching the cell budget so the GENERATED pool count stays at or below "
                             "this number (never above), keeping fidelity as high as possible. High-volume "
                             "RPGTs keep detail first. 0 = no compression (a pool per distinct BIN rule). "
                             "The pool-targeting runs when you click Build & Export / Generate configs.")
                    ss["max_configs"] = int(max_configs)
                    _dc1, _dc2 = st.columns(2)
                    apply_decay = _dc1.checkbox(
                        "Apply time decay", value=True, key="apply_decay_cb",
                        help="Weight recent attempts more heavily when estimating success rates.")
                    decay_half = _dc2.number_input(
                        "Half-life (days)", min_value=1, max_value=365, value=15, step=1, key="decay_half_inp",
                        help="Attempts this many days old count half as much. Used only when time decay is on.")
                    # --- Auto-block dead gateways (bank appears to have blocked the merchant) ---
                    # vertical_alignment="center" so the checkbox sits mid-height against the
                    # (taller, labelled) number input beside it instead of pinning to the top.
                    _bg1, _bg2 = st.columns(2, vertical_alignment="center")
                    _bg1.checkbox(
                        "Auto-block dead gateways", value=True, key="block_gw_cb",
                        help="Flag a gateway as BANK-BLOCKED when its MOST-RECENT attempts have all "
                             "failed (0 approvals) for at least the count on the right, then cap that "
                             "gateway's share (for that bank) to the exploration floor so the engine "
                             "stops routing real volume to a door the bank has closed.")
                    _bg2.slider(
                        "Min consecutive failed attempts", min_value=10, max_value=1000,
                        value=100, step=10, key="block_min_inp",
                        help="How many of the most-recent, all-failed attempts (from the latest "
                             "attempt backward) before a gateway counts as blocked by that bank. "
                             "e.g. 100 = the last 100 attempts for that gateway+bank all failed. Only "
                             "used when 'Auto-block dead gateways' is on.")
                    # --- Bayesian smoothing (moved back here from Engine Type & Settings) ---
                    if engine_key == "thompson":
                        # Thompson uses its OWN self-contained Beta posteriors (raw time-decayed
                        # counts), not the pipeline's Bayesian smoothing — so hide these inputs.
                        bayes_method, use_eb, shrink = "Empirical Bayes", True, 300
                    else:
                        bayes_method = st.selectbox(
                            "Low Volume Method", ["Empirical Bayes", "Fixed Number"],
                            help="Empirical Bayes (default): estimate the smoothing strength per Bank x Currency "
                                 "from how much the gateways' success rates vary. Fixed Number: one set value for all.")
                        use_eb = (bayes_method == "Empirical Bayes")
                        # Show the smoothing-volume input ONLY for Fixed Number (it's meaningless under
                        # Empirical Bayes, which estimates the volume per Bank×Currency). These live in
                        # the compute form, so the selectbox commits on submit — the input therefore
                        # reveals/hides when you pick the method and click "Compute split variations",
                        # exactly like the engine-specific settings. Default keeps `shrink` defined
                        # while the input is hidden.
                        _shrink_in = 300
                        if not use_eb:
                            _shrink_in = st.slider(
                                "Bayesian Smoothing Volume", min_value=0, max_value=2000, value=300, step=50,
                                help="Pseudo-attempts applied to every gateway under Fixed Number smoothing. "
                                     "Higher = more shrinkage toward the prior.")
                        shrink = 300 if use_eb else int(_shrink_in)

        # Section 3 (Risk constraints + per-MID editor) on the LEFT, Section 4 (Run Log)
        # on the RIGHT. Both columns were created inside the form above, so every widget
        # placed here is a form member and only takes effect on submit.
        with _c_log:
            with st.container(border=True):
                st.markdown("##### 4. Run Log")
                _run_prog_slot = st.container(key="run_prog_slot")   # % complete + ETA bar (filled during a run)
                # Lift the status row (spinner + "Running…" label) up so its text lines up with
                # the "Enforce VAMP cap" checkbox at the top of the Risk Constraints panel beside
                # it — the st.status box carries a top margin that otherwise sits it lower. Same
                # .st-key-<key> technique used for the checkbox alignment elsewhere in this app.
                st.markdown("<style>.st-key-run_prog_slot{margin-top:-0.5rem;}</style>",
                            unsafe_allow_html=True)
                _run_log_slot = st.container()
        with _c_rc:
            with st.container(border=True):
                st.markdown("##### 3. Risk Constraints")

                # Narrow VAMP-cap % input on the LEFT (≈20% width), the enable checkbox to its right.
                # number_input can't render a literal '%' in its format string, so the '(%)'
                # label above the box signals the value is a percentage (6.00 = 6%).
                _v1, _v2 = st.columns([2, 8])
                vamp_on = _v2.checkbox("Enforce VAMP cap", value=True, key="vamp_on_cb")
                if vamp_on:
                    vamp_cap_pct = _v1.number_input("VAMP cap (%)", min_value=0.01, max_value=20.0,
                                                    value=6.0, step=0.1, format="%.2f", key="vamp_cap_inp")
                    vamp_cap = vamp_cap_pct / 100.0
                else:
                    vamp_cap = None

                mid_path = os.path.join(out_dir, "mid_level.csv")
                fs_cfg = ss.get("forecast_settings", {})
                try:
                    base_dt = pd.to_datetime(fs_cfg.get("month_0", date.today().replace(day=1)))
                except Exception:
                    base_dt = pd.to_datetime(date.today().replace(day=1))

                # Baseline M0–M3 totals per vampMid (used to turn an All/All target into
                # a volume scale for the existing enforcement).
                _base_totals = {}
                mids = []
                if os.path.exists(mid_path):
                    mid_data = pd.read_csv(mid_path)
                    if "vampMid" in mid_data.columns:
                        mid_data = mid_data[mid_data["vampMid"].astype(str).str.upper() != "TOTAL"]
                        mids = sorted([str(m) for m in mid_data["vampMid"].dropna().unique() if str(m).strip() != ""])
                        for mid in mids:
                            r = mid_data[mid_data["vampMid"] == mid]
                            _bt = _bv = 0.0
                            if not r.empty:
                                for m in range(4):
                                    _bt += float(pd.to_numeric(r.iloc[0].get(f"FC_VI_Txn_Month_{m}", 0), errors="coerce") or 0)
                                    # FC_VAMP_Month is ALREADY calendar-day (the pipeline's actuarial
                                    # carryover applied days/30.4167). Sum it raw — the SAME basis as the
                                    # transaction baseline (_bt) and as _scope_base (which reads calendar
                                    # vampCount from the pro-rata export). Re-multiplying by days/30.4167
                                    # here double-flexed the per-MID VAMP baseline and skewed the per-MID
                                    # VAMP / VAMP% caps (and disagreed with the scoped-rule baseline).
                                    _bv += float(pd.to_numeric(r.iloc[0].get(f"FC_VAMP_Month_{m}", 0), errors="coerce") or 0)
                            _base_totals[str(mid).strip().lower()] = (_bt, _bv)

                # RPGT scope, grain, hold-others and auto-explore now live in the "Engine Type
                # and Settings" section above (their widgets set _rpgt_opts / _rpgt_selected /
                # _rpgt_hold_others / _score_grain / _opt_grain / _auto_explore, used below).

                _rules_cols = ["vampMid", "RPGT", "Month", "Metric", "Type", "Target", "Tol %", "Priority"]
                _rules_seed = pd.DataFrame({c: pd.Series(dtype="object") for c in _rules_cols})
                _rules_seed["Target"] = pd.Series(dtype="float")
                _rules_seed["Tol %"] = pd.Series(dtype="float")

                _vm_cfg = (st.column_config.SelectboxColumn("vampMid", options=mids, required=True, width="medium")
                           if mids else st.column_config.TextColumn("vampMid", required=True, width="medium"))
                col_cfg = {
                    # Only vampMid keeps a wide column (long MID names); every other column is
                    # narrowed to 'small' (st.data_editor supports small/medium/large, not exact
                    # auto-fit or a % width, so 'small' is the closest to fit-to-content here).
                    "vampMid": _vm_cfg,
                    "RPGT": st.column_config.SelectboxColumn("RPGT", options=["All"] + _rpgt_opts, width="small",
                                                             help="'All' applies across every RPGT."),
                    "Month": st.column_config.SelectboxColumn("Month", options=["All", "M0", "M1", "M2", "M3", "M4", "M5"], width="small",
                                                              help="A specific month (M0–M5) is enforced on that month's projection. 'All' applies across M0–M3."),
                    "Metric": st.column_config.SelectboxColumn("Metric", options=["Txn", "VAMP", "VAMP %"], width="small"),
                    "Type": st.column_config.SelectboxColumn(
                        "Type", options=["range", "ceiling", "floor"], width="small", default="range",
                        help="range = within ±Tol of Target (two-sided). ceiling = at most Target(+Tol) "
                             "(upper bound only). floor = at least Target(−Tol) (lower bound only)."),
                    # Whole numbers for Txn/VAMP counts. (Per-row format isn't possible in st.data_editor,
                    # so VAMP % rate targets also display as whole numbers — enter them as a fraction.)
                    "Target": st.column_config.NumberColumn("Target", min_value=0.0, format="%.0f", width="small",
                                                            help="Txn / VAMP: a whole-number count cap. VAMP %: the aggregate "
                                                                 "VAMP-rate cap (a fraction, e.g. 0.90 — but this column displays whole numbers)."),
                    "Tol %": st.column_config.NumberColumn("Tol %", min_value=0.0, max_value=500.0, format="%.0f%%", width="small",
                                                           help="Headroom on the target (ignored for VAMP %)."),
                    "Priority": st.column_config.NumberColumn(
                        "Priority", min_value=1, max_value=99, step=1, format="%d", width="small", default=1,
                        help="1 = highest priority. When constraints conflict (can't all be met), the engine "
                             "keeps low-number priorities and lets higher numbers yield first."),
                }
                st.markdown("""<style>
                .st-key-mid_constraints_editor {
                    --gdg-base-font-style: 11px;
                    --gdg-header-font-style: 600 11px;
                }
                </style>""", unsafe_allow_html=True)

                # The editor fills this column (Section 3 is now itself half the page width).
                edited_mids = st.data_editor(
                    _rules_seed, column_config=col_cfg, hide_index=True, num_rows="dynamic",
                    use_container_width=True, height=440, key="mid_constraints_editor")

                _metric_key = {"Txn": "txn", "VAMP": "vamp", "VAMP %": "vamp_pct"}
                clean_records = []
                for row in edited_mids.to_dict("records"):
                    _mid = row.get("vampMid")
                    if _mid is None or (isinstance(_mid, float) and pd.isna(_mid)) or str(_mid).strip() == "":
                        continue
                    _tgt = row.get("Target")
                    if pd.isna(_tgt):
                        continue
                    _rp = row.get("RPGT")
                    _rp = None if (pd.isna(_rp) or str(_rp).strip() in ("", "All")) else str(_rp)
                    _mo = row.get("Month")
                    _mo = None if (pd.isna(_mo) or str(_mo).strip() in ("", "All")) else int(str(_mo).replace("M", ""))
                    _tl = row.get("Tol %")
                    clean_records.append({
                        "vampMid": str(_mid).strip(),
                        "rpgt": _rp,                 # None = all RPGTs
                        "month": _mo,                # None = all of M0–M3
                        "metric": _metric_key.get(str(row.get("Metric") or "Txn"), "txn"),
                        "target": float(_tgt),       # count (txn/vamp) or rate % (vamp_pct)
                        "tol": (float(_tl) / 100.0 if pd.notna(_tl) else None),
                        # constraint TYPE: range = two-sided ±tol; ceiling = upper bound only;
                        # floor = lower bound only. Default range (matches prior behaviour).
                        "direction": (lambda _t: _t if _t in ("range", "ceiling", "floor") else "range")(
                            str(row.get("Type") or "range").strip().lower()),
                        # priority: 1 = highest. Lower-priority (higher-number) constraints yield
                        # first when the set is jointly infeasible.
                        "priority": (lambda _p: int(_p) if (pd.notna(_p) and int(_p) >= 1) else 1)(
                            row.get("Priority") if pd.notna(row.get("Priority")) else 1),
                    })
            params["mid_constraints"] = clean_records
            params["mid_base_totals"] = _base_totals

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("""<style>
                button[kind="primaryFormSubmit"] { background-color: #22C36B !important; border-color: #22C36B !important; border-radius: 0 !important; }
                button[kind="primaryFormSubmit"]:hover { background-color: #1BA85B !important; border-color: #1BA85B !important; }
                button[kind="primaryFormSubmit"] p { color: #FFFFFF !important; }
                button[kind="primaryFormSubmit"] div { color: #FFFFFF !important; }
            </style>""", unsafe_allow_html=True)
            submit_engine = st.form_submit_button("Compute split variations", type="primary")

        # Genetic search budget — rendered BELOW the engine form (moved down from the top of the tab),
        # so it sits beneath '1. Engine Type & Settings'. Kept OUTSIDE the form as a live fragment so
        # editing generations/λ/seeds/restarts refreshes the candidate-count + ETA immediately.
        if _pre_engine in ("genetic", "genetic_numba"):
            (st.fragment(_budget_panel) if hasattr(st, "fragment") else _budget_panel)()

        if submit_engine:
            ss["exploration_floor"] = float(floor)   # tab 3 uses this to replicate the engine's floor
            # Mid-run STOP: clear any stale signal, then make a poller the GA checks each generation
            # (halts + keeps best-so-far when the sidebar 'Stop' writes the signal).
            try:
                from routing_optimiser.run_bundle import clear_stop as _clr_stop, make_stop_check as _mk_stop
                _runs_dir_stop = os.path.join(PROJECT_ROOT, "runs")
                os.makedirs(_runs_dir_stop, exist_ok=True)   # signal path is always runs/_stop
                _clr_stop(_runs_dir_stop)
                _ga_stop = _mk_stop(_runs_dir_stop)
            except Exception:  # noqa: BLE001
                _ga_stop = None
            base_settings = OptimiserSettings(
                risk_conversion_weight=0.5, engine=engine_key, engine_params=params,
                hard=HardConstraints(max_gateway_share=max_share, vamp_cap=vamp_cap),
                soft=SoftConstraints(exploration_floor=floor))

            import time as _pt
            _run_t0 = _pt.time()
            # CALM PROGRESS VIEW: a prominent "~N remaining" headline + current stage on top,
            # a thin % bar under it, and the full technical log tucked inside a collapsed
            # "Show technical details" expander (the st.status widget) so a non-expert sees a
            # quiet progress screen while the raw diagnostics stay one click away for experts.
            _eta_slot = _run_prog_slot.empty()                 # big remaining-time headline
            _pbar = _run_prog_slot.progress(0.0, text="starting…")
            status = _run_prog_slot.status("Show technical details", expanded=False)

            # ADAPTIVE ETA: derive the stage-boundary fractions from the LAST run's measured
            # engine (④) + compression (⑥) wall-times (persisted in ga_perf.json), so the % and
            # ETA self-calibrate to this machine + settings — multi-seed, dial count and pool
            # target all shift the engine/compression split, which broke the old fixed fractions.
            _perf0 = _load_ga_perf() or {}
            _E_est = float(_perf0.get("secs", 715.0) or 715.0)           # engine ④ secs (last run)
            _C_est = float(_perf0.get("compress_secs", 527.0) or 527.0)  # compression ⑥ secs
            _PRE_est = 43.0                                              # ①②③ (roughly fixed)
            # SETTINGS-AWARE scaling: the last run's times were measured under ITS settings. Rescale
            # them to the CURRENT settings (generations × population × seeds for the engine, pool
            # target for compression) so the FIRST estimate reflects a bigger/smaller run instead of
            # waiting a full run to self-calibrate. Any missing stored field → ratio 1 (no-op), so an
            # old ga_perf.json still works; ratios are clamped so a wild value can't blow up the ETA.
            # [FN-301]
            def _ratio(cur, prev, lo=0.2, hi=5.0):
                try:
                    cur, prev = float(cur), float(prev)
                    return min(hi, max(lo, cur / prev)) if (prev > 0 and cur > 0) else 1.0
                except Exception:  # noqa: BLE001
                    return 1.0
            _cur_gen = int(ss.get("ga_generations", 80) or 80)
            _cur_pop_ovr = int(ss.get("ga_pop_override", 0) or 0)
            # pop auto-sizes to n_mid (unknown pre-processing) → assume the same data ⇒ same auto pop
            # as last run; only an explicit override changes the pop ratio.
            _cur_pop = _cur_pop_ovr if _cur_pop_ovr > 0 else int(_perf0.get("pop", 0) or 0)
            # Seeds run in parallel up to the core count, so wall time scales with the number of
            # sequential WAVES = ceil(seeds / cores), not the raw seed count (adding seeds within
            # the core budget is ~free; only overflow beyond the cores adds a wave).
            _cpu_now = max(1, int(os.cpu_count() or 4))
            _cur_seeds = max(1, int(ss.get("ga_n_seeds", _GA_N_SEED) or _GA_N_SEED))
            _prev_seeds = max(1, int(_perf0.get("seeds", _GA_N_SEED) or _GA_N_SEED))
            _cur_waves = -(-_cur_seeds // _cpu_now)      # ceil division
            _prev_waves = -(-_prev_seeds // _cpu_now)
            # Restarts multiply each seed's total generations (gens_max = generations × restarts),
            # so they scale engine time roughly linearly — fold them into the estimate too.
            _cur_restarts = max(1, int(ss.get("ga_restarts", 4) or 4))
            _eng_scale = (_ratio(_cur_gen, _perf0.get("gen"))
                          * _ratio(_cur_pop, _perf0.get("pop"))
                          * _ratio(_cur_waves, _prev_waves)
                          * _ratio(_cur_restarts, _perf0.get("restarts", 4)))
            _E_est *= min(6.0, max(0.15, _eng_scale))
            _C_est *= _ratio(int(ss.get("max_configs", 0) or 0), _perf0.get("pool_target"),
                             lo=0.3, hi=3.0)
            _T_est = max(_PRE_est + _E_est + _C_est, 1.0)
            _f_cells = 26.0 / _T_est                    # assembling-cells checkpoint
            _f_eng = _PRE_est / _T_est                  # engine start
            _f_eng_end = (_PRE_est + _E_est) / _T_est   # engine done → compression start
            # [FN-302]
            def _eng(fr):                               # within-engine fraction → global fraction
                return _f_eng + float(fr) * (_f_eng_end - _f_eng)
            _f_rmin, _f_enf1, _f_enf2, _f_var = _eng(0.37), _eng(0.80), _eng(0.90), _eng(0.97)
            _t6_0 = None   # compression-stage start (set at stage ⑥; used to persist compress_secs)

            # [FN-303]
            def _progress(frac, label=""):
                frac = max(0.0, min(1.0, float(frac)))
                _pct = int(round(frac * 100))
                _el = _pt.time() - _run_t0
                if frac >= 0.999:
                    _rem = ""
                elif _t6_0 is not None:
                    # FINAL stage (pool compression) is one long blocking call that never ticks the
                    # bar, so a fraction-based linear ETA freezes the fraction while time keeps
                    # elapsing and balloons (the old "~4386s left" bug). Anchor it to elapsed time in
                    # the stage vs the calibrated compression time instead.
                    _eta = int(max(1, min(_C_est - (_pt.time() - _t6_0), _T_est)))
                    _rem = f" · ~{_eta}s left (est.)"
                elif frac <= 0.02:
                    _eta = int(max(1, min(_T_est - _el, _T_est)))
                    _rem = f" · ~{_eta}s left (est.)"
                else:
                    # Blend the calibrated remaining with a live linear extrapolation, then HARD-
                    # CLAMP to [1, total estimate]. The clamp is the safety net: a frozen fraction
                    # during a long stage can no longer push the ETA past the whole estimated run.
                    _eta_model = _T_est - _el
                    _eta_lin = _el * (1.0 - frac) / max(frac, 1e-6)
                    _eta = frac * _eta_lin + (1.0 - frac) * _eta_model
                    _eta = int(max(1, min(_eta, _T_est)))
                    _rem = f" · ~{_eta}s left (est.)"
                _txt = f"{_pct}%" + (f" · {label}" if label else "")
                try:
                    _pbar.progress(frac, text=_txt)
                except Exception:  # noqa: BLE001
                    pass
                # PROMINENT ETA headline (the calm view's focal point): humanise the estimate
                # to minutes when it's long, and pair it with the current stage in muted text.
                try:
                    if frac >= 0.999:
                        _head = "✓ Finishing up…"
                    elif int(_eta) >= 90:
                        _head = f"~{int(round(int(_eta) / 60))} min remaining"
                    else:
                        _head = f"~{max(1, int(_eta))}s remaining"
                    _eta_slot.markdown(
                        "<div style='display:flex; align-items:baseline; gap:0.6rem; "
                        "padding:0.1rem 0 0.35rem 0;'>"
                        f"<span style='font-size:1.5rem; font-weight:800; line-height:1.1; "
                        f"color:var(--tav-ink);'>{_head}</span>"
                        f"<span style='font-size:0.9rem; color:var(--tav-muted);'>"
                        f"{label or 'working…'}</span></div>",
                        unsafe_allow_html=True)
                except Exception:  # noqa: BLE001
                    pass

            with status:
                # Render the live log INSIDE the "Show technical details" expander (the status
                # widget) so the default view stays calm — the raw diagnostics only appear when
                # the user expands it. `st.empty()` here places the log within the status.
                log_area = st.empty()
                log_lines: list[str] = []
                # [FN-304]
                def log(msg):
                    # Keep the FULL log in log_lines (shown in the expandable panel and copyable);
                    # render the tail live so the panel stays responsive during long runs. Every
                    # line is stamped with wall-clock time + seconds elapsed since the run started,
                    # so you can see when each stage/action started and finished.
                    _ts = datetime.datetime.now().strftime("%H:%M:%S")
                    log_lines.append(f"[{_ts} +{_pt.time() - _run_t0:6.1f}s] {msg}")
                    log_area.code("\n".join(log_lines[-1200:]), language="log")

                # Stage timer: logs "▶ … started" and, when the next stage begins (or the run
                # ends), "✓ … finished in Ns" for the previous stage.
                _stage_state = {"name": None, "t": None}
                # [FN-305]
                def _stage(name):
                    _now = _pt.time()
                    if _stage_state["name"] is not None:
                        log(f"   ✓ {_stage_state['name']} — finished in {_now - _stage_state['t']:.1f}s")
                    _stage_state["name"] = name
                    _stage_state["t"] = _now
                    log(f"▶ {name} — started")
                # [FN-306]
                def _stage_end():
                    if _stage_state["name"] is not None:
                        log(f"   ✓ {_stage_state['name']} — finished in {_pt.time() - _stage_state['t']:.1f}s")
                        _stage_state["name"] = None

                # [FN-307]
                def _diag(msg):
                    """Verbose diagnostic line (same sink as log). Wrapped so a diagnostics
                    failure can NEVER crash a run — diagnostics are best-effort."""
                    try:
                        log(msg)
                    except Exception:  # noqa: BLE001
                        pass

                handler = StreamlitLogHandler(log)
                root_logger = logging.getLogger()
                prev_level = root_logger.level
                root_logger.addHandler(handler)
                root_logger.setLevel(logging.INFO)
                try:
                    # ═══════════════════ RUN DIAGNOSTICS HEADER (verbose) ═══════════════════
                    # Everything needed to reproduce/inspect a run: environment, code builds,
                    # full config, and the input files (path + mtime + size). Best-effort — any
                    # failure here is swallowed so it never affects the run.
                    try:
                        import datetime as _dt, platform as _plat, importlib as _il
                        _L = locals()
                        # [FN-308]
                        def _gv(name, default="?"):
                            return _L.get(name, default)
                        # [FN-309]
                        def _bmark(modpath):
                            try:
                                return getattr(_il.import_module(modpath), "__build__", "(no __build__)")
                            except Exception as _e:  # noqa: BLE001
                                return f"(import failed: {_e})"
                        # [FN-310]
                        def _finfo(p):
                            try:
                                _s = os.stat(p)
                                return f"{p}  [{_s.st_size/1e6:.2f} MB, mtime {_dt.datetime.fromtimestamp(_s.st_mtime):%Y-%m-%d %H:%M:%S}]"
                            except Exception:  # noqa: BLE001
                                return f"{p}  [missing]"
                        _diag("═════════════════════════ RUN DIAGNOSTICS ═════════════════════════")
                        _diag(f"   started {_dt.datetime.now():%Y-%m-%d %H:%M:%S} · python {_plat.python_version()} · "
                              f"pandas {pd.__version__} · numpy {np.__version__}")
                        _diag(f"   APP_BUILD: {APP_BUILD.split(' · ')[0]}")
                        _diag("   backend build markers (if any ≠ expected → stale bytecode; clear __pycache__):")
                        for _m in ["routing_optimiser.optimiser", "routing_optimiser.eligibility",
                                   "routing_optimiser.success_rates", "routing_optimiser.forecast_pipeline",
                                   "routing_optimiser.data_loader", "routing_optimiser.sql_runner",
                                   "routing_optimiser.constraints", "routing_optimiser.engines.base",
                                   "routing_optimiser.engines.softmax", "routing_optimiser.engines.thompson",
                                   "routing_optimiser.genetic_global", "routing_optimiser.engines.portfolio"]:
                            _diag(f"      {_m.split('.')[-1]:16s} {_bmark(_m)}")
                        # impact_calcs is imported by-name (from impact_calcs import ...), so the module
                        # object isn't in scope — import it explicitly so its build marker is ALWAYS
                        # shown (confirms the _project_capped vectorise/memoise speedups are loaded).
                        _diag(f"      {'impact_calcs':16s} {_bmark('impact_calcs')}")
                        _diag("   RUN CONFIG:")
                        _diag(f"      company={_gv('sr_company')} · scheme={_gv('sr_scheme')} · "
                              f"attempts_window={_gv('attempts_start')} → {_gv('attempts_end')}")
                        _diag(f"      engine={_gv('engine_key')} · score_grain={_gv('_score_grain')} · opt_grain={_gv('_opt_grain')}")
                        _diag(f"      vamp_cap={_gv('vamp_cap')} · exploration_floor={_gv('floor')} · max_gateway_share={_gv('max_share')}")
                        _diag(f"      bayes_method={_gv('bayes_method')} · shrink κ={_gv('shrink')} · "
                              f"time_decay={('on ' + str(_gv('decay_half')) + 'd') if _gv('apply_decay') else 'off'} · "
                              f"xborder_penalty={_gv('xborder_penalty')}")
                        _diag(f"      max_pools_target={_gv('max_configs')}")
                        _diag("      band scoring=EXACT in-search (per-generation pro-rata projection; "
                              "no proxy, no post-hoc correction)")
                        _pk = {k: params.get(k) for k in ("temperature", "temp_method",
                                                          "explore_cap_total", "explore_cap_each", "n_variations")
                               if isinstance(params, dict) and k in params}
                        _diag(f"      engine_params={_pk}")
                        _diag(f"      auto_explore={_gv('_auto_explore')} · RPGT_scope={('ALL' if not _gv('_sel_rpgts', None) else _gv('_sel_rpgts'))} · "
                              f"hold_unselected_at_baseline={ss.get('eng_rpgt_hold_others')}")
                        _diag(f"      gateway auto-block={'ON' if ss.get('block_gw_cb', False) else 'off'}"
                              + (f" · >={int(ss.get('block_min_inp', 100) or 100)} consecutive failed attempts"
                                 if ss.get('block_gw_cb', False) else ""))
                        # Step-size (CMA-ES σ) overrides — echoed ONLY when dialled off their no-op defaults,
                        # so a tuning A/B is self-documenting in the saved run bundle (absent line ⇒ defaults).
                        if _gv('engine_key') in ("genetic", "genetic_numba"):
                            _s0m = float(ss.get("ga_sigma0_mult", 1.5) or 1.5)
                            _sfl = float(ss.get("ga_sigma_floor", 0.0) or 0.0)
                            _dmp = float(ss.get("ga_damps_mult", 1.5) or 1.5)
                            if abs(_s0m - 1.0) > 1e-9 or _sfl > 0.0 or abs(_dmp - 1.0) > 1e-9:
                                _diag(f"      step-size (CMA-ES σ) TUNED: σ₀×{_s0m:g} · σ_floor={_sfl:g} · damping×{_dmp:g} "
                                      "(defaults: σ₀×1 · σ_floor=0 · damping×1)")
                        _mcn = params.get("mid_constraints", []) if isinstance(params, dict) else []
                        _diag(f"      per-MID constraints configured: {len(_mcn or [])}")
                        for _r in (_mcn or [])[:40]:
                            try:
                                _rp = _r.get("rpgt") or "ALL-RPGT"
                                _mo = "ALL" if _r.get("month") is None else f"M{_r.get('month')}"
                                _tl = _r.get("tol")
                                _tls = "n/a" if _tl is None else f"{float(_tl) * 100:g}%"
                                _diag(f"         • {_r.get('vampMid')} | {_rp} | {_mo} | "
                                      f"{_r.get('metric')} [{_r.get('direction', 'range')}] "
                                      f"target={_r.get('target')} tol={_tls} prio={_r.get('priority', 1)}")
                            except Exception:  # noqa: BLE001
                                pass
                        _diag("   INPUT FILES:")
                        _mid_p = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                        for _label, _p in [("outputs dir", _gv("out_dir")), ("MID list", _mid_p)]:
                            if isinstance(_p, str):
                                _diag(f"      {_label}: {_finfo(_p)}")
                        _diag("════════════════════════════════════════════════════════════════════")
                    except Exception as _e:  # noqa: BLE001
                        _diag(f"   [diagnostics header partial/failed: {_e}]")

                    _progress(0.01, "Fetching attempts…")
                    _stage("① Fetch attempts/success data")
                    sql_params = {
                        "START_DATE": str(attempts_start),
                        "END_DATE": str(attempts_end),
                        "COMPANY": sr_company,
                        "CARD_SCHEME": sr_scheme,
                        "BIN_PREFIX": "4" if sr_scheme == "visa" else "5",
                        "GATEWAY_FIDS": DEFAULT_GATEWAY_FIDS,
                    }
                    sql_path = os.path.join(SQL_DIR, "attempts_success.sql")
                    if not os.path.exists(sql_path): raise FileNotFoundError(f"attempts_success.sql not found.")
                    attempts_path, src = run_sql_file(sql_path, CACHE_DIR, use_cache=True, fallback_csv=None, project=GCP_PROJECT, params=sql_params)
                    log(f"   attempts source: {src}")

                    # Optional CROSS-BRAND processor benchmark → layer-2 prior for untested MIDs.
                    # Runs only when auto-explore is on AND queries/processor_benchmark.sql exists, and
                    # is fully guarded: any failure logs a note and leaves the benchmark empty, so
                    # untested MIDs fall back to the same-brand sibling / bank×currency average. The
                    # SQL is a DRAFT — validate it on BigQuery before trusting the layer-2 rates.
                    ss["processor_benchmark"] = {}
                    if ss.get("eng_auto_explore"):
                        try:
                            _pb_sql = os.path.join(SQL_DIR, "processor_benchmark.sql")
                            if os.path.exists(_pb_sql):
                                _pb_path, _pb_src = run_sql_file(_pb_sql, CACHE_DIR, use_cache=True,
                                                                 fallback_csv=None, project=GCP_PROJECT, params=sql_params)
                                _pbdf = pd.read_parquet(_pb_path)
                                _pcol = {str(c).strip().lower(): c for c in _pbdf.columns}
                                _pp, _pc = _pcol.get("processor"), _pcol.get("currency")
                                _ps, _pa = _pcol.get("successes"), _pcol.get("attempts")
                                if _pp and _pc and _ps and _pa:
                                    _pg = _pbdf.groupby(
                                        [_pbdf[_pp].astype(str).str.strip().str.lower(),
                                         _pbdf[_pc].astype(str).str.strip().str.lower()]).agg(
                                        s=(_ps, "sum"), a=(_pa, "sum"))
                                    ss["processor_benchmark"] = {k: float(r["s"]) / float(r["a"])
                                                                 for k, r in _pg.iterrows() if float(r["a"]) > 0}
                                    log(f"   cross-brand processor benchmark: {len(ss['processor_benchmark'])} "
                                        f"(processor, currency) rates ({_pb_src}) → untested-MID layer-2 prior.")
                        except Exception as _e:  # noqa: BLE001
                            log(f"   [Note] cross-brand processor benchmark unavailable ({_e}); untested MIDs "
                                "use same-brand sibling / cell average.")

                    _progress(0.02, "Pre-processing…")
                    _stage("② Pre-processing (Bayesian smoothing)")
                    # Cache the parsed attempts in-memory (keyed on path + mtime) so switching
                    # engine / re-running doesn't re-parse the same ~700k-row file every time.
                    _adf_k = (attempts_path, _mtime(attempts_path))
                    if ss.get("_adf_cache_k") == _adf_k and ss.get("_adf_cache") is not None:
                        adf = ss["_adf_cache"].copy()
                        log("   (reused parsed attempts from in-memory cache)")
                    else:
                        adf = load_success_data(attempts_path)
                        ss["_adf_cache_k"] = _adf_k
                        ss["_adf_cache"] = adf.copy()
                    
                    # RPGT canonicalisation for the attempts data is handled upstream in
                    # load_success_data (schema.SCENARIO_TO_RPGT) + the fixed attempts_success.sql,
                    # and the engine joins are case-insensitive — so no per-tab remap is needed here.
                    # (The forecast below is a SEPARATE source with no such chokepoint — see there.)
                        
                    # Cache the parsed baseline forecast too (keyed on out_dir + baseline mtime +
                    # attempts identity — the only things it depends on). Re-running the FORECAST
                    # (tab 1) changes the baseline mtime and invalidates it.
                    _fc_bl = os.path.join(out_dir, "bin_rpgt_impact_export.csv")
                    _fc_k = (out_dir, _mtime(_fc_bl), attempts_path, _mtime(attempts_path))
                    if ss.get("_fc_cache_k") == _fc_k and ss.get("_fc_cache") is not None:
                        forecast_temp = ss["_fc_cache"].copy()
                        log("   (reused parsed baseline forecast from in-memory cache)")
                    else:
                        forecast_temp = load_forecast(out_dir, adf)
                        ss["_fc_cache_k"] = _fc_k
                        ss["_fc_cache"] = forecast_temp.copy()

                    if "bank" in forecast_temp.columns:
                        fc_banks = set(forecast_temp["bank"].dropna().astype(str).str.strip().str.upper())
                        adf_banks = set(adf["bank"].dropna().astype(str).str.strip().str.upper()) if "bank" in adf.columns else set()
                        best_col = "bank"
                        best_overlap = len(fc_banks.intersection(adf_banks))
                        
                        for alt_col in ["bin", "BIN", "bankName", "bank_name"]:
                            if alt_col in adf.columns:
                                alt_set = set(adf[alt_col].dropna().astype(str).str.strip().str.upper())
                                overlap = len(fc_banks.intersection(alt_set))
                                if overlap > best_overlap:
                                    best_overlap = overlap; best_col = alt_col
                        if best_col != "bank" and best_overlap > 0:
                            adf["original_bank_name"] = adf["bank"]
                            adf["bank"] = adf[best_col]
                        elif "original_bank_name" not in adf.columns:
                            adf["original_bank_name"] = adf["bank"]

                    mid_list_path = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    if os.path.exists(mid_list_path) and "gateway" in forecast_temp.columns:
                        try:
                            mid_df = pd.read_csv(mid_list_path)
                            clean_cols = {str(c).lower().replace(" ", "").replace("_", ""): c for c in mid_df.columns}
                            v_col, g_col = clean_cols.get("vampmid"), clean_cols.get("gatewayfid")
                            
                            if v_col and g_col:
                                v2f = dict(zip(mid_df[v_col].astype(str).str.strip().str.upper(), mid_df[g_col].astype(str).str.strip().str.lower()))
                                forecast_temp["gateway_mapped"] = forecast_temp["gateway"].astype(str).str.strip().str.upper().map(v2f)
                                forecast_temp["gateway"] = forecast_temp["gateway_mapped"].fillna(forecast_temp["gateway"])
                                # (Per-MID constraints stay keyed by vampMid — the volume-cap
                                # enforcement matches on vampMid, so no FID remap is applied.)
                        except Exception as e:
                            log(f"   [Warning] Failed to map MIDs: {e}")

                    for c in ["currency", "bank", "gateway"]:
                        if c in adf.columns: adf[c] = adf[c].astype(str).str.strip().str.lower()
                        if c in forecast_temp.columns: forecast_temp[c] = forecast_temp[c].astype(str).str.strip().str.lower()
                    # Forecast RPGT comes from the VAMP pipeline output (bin_rpgt_impact_export.csv),
                    # NOT from attempts_success.sql and NOT through load_success_data — it has no
                    # canonicalisation chokepoint, so it keeps its own local RPGT fix.
                    _fc_rpgt_fix = {
                        "MONTHLY INTIIAL": "Monthly Initial", "MONTHLY INITIAL": "Monthly Initial",
                        "ANNUAL SUB SALE": "Annual Sub Sale", "ADDON SALE": "Addon Sale",
                        "UPGRADE": "Upgrades", "UPGRADES": "Upgrades",
                        "MONTHLY RENEWAL": "Monthly Renewal", "ANNUAL SUB RENEWAL": "Annual Sub Renewal",
                        "P6M RENEWALS": "P6M Renewals", "ADDON RENEWAL": "Addon Renewal",
                    }
                    if "rpgt" in forecast_temp.columns:
                        forecast_temp["rpgt"] = (forecast_temp["rpgt"].astype(str).str.strip().str.upper()
                                                 .map(_fc_rpgt_fix).fillna(forecast_temp["rpgt"]))

                    # RPGT scope (tab 2 multiselect): restrict the WHOLE optimisation to the
                    # selected RPGTs — their attempts, volume, VAMP and risk. Only applied when
                    # the user has narrowed the selection (leaving all selected is a no-op, which
                    # also avoids dropping RPGTs whose names differ from the option list).
                    _sel_rpgts = {str(r).strip().lower() for r in (_rpgt_selected or [])}
                    _all_rpgts = {str(r).strip().lower() for r in _rpgt_opts}
                    _do_rpgt_filter = bool(_sel_rpgts and _sel_rpgts != _all_rpgts)
                    _score_by_rpgt = (_score_grain == "Bank × Currency × RPGT")
                    _opt_by_rpgt = (_opt_grain == "Bank × Currency × RPGT")
                    # The RPGT filter that narrows attempts/forecast to the SELECTED RPGTs is
                    # applied further down (after the currency / switch-off cleanups) — NOT here.
                    # Why: in Bank×Currency mode the ENGINE SCORE must pool ALL RPGTs for the cell
                    # (all transaction types inform the gateway's success rate), while only the
                    # volume routed, eligibility and the VAMP cap are restricted to the selected
                    # RPGTs. In Bank×Currency×RPGT mode each selected RPGT is scored on its own.

                    # Persist the RPGT scope so the Impact tab can hold unselected RPGTs at
                    # their current baseline split (pre == post) when the tickbox is ON. The
                    # engine decision is always informed by the selected RPGTs only; this
                    # controls whether the decision is APPLIED to the others too.
                    ss["rpgt_scope"] = {
                        "selected": tuple(sorted(_sel_rpgts)),
                        "all": tuple(sorted(_all_rpgts)),
                        "hold_others": bool(_rpgt_hold_others),
                    }

                    # Drop attempt rows whose currency disagrees with the
                    # gateway's designated currency in the Master MID list (e.g. a
                    # EUR-only gateway carrying USD-tagged rows). Only drops when
                    # the gateway IS in the list with real (non-EXCLUDED) currencies.
                    try:
                        if os.path.exists(mid_list_path) and {"gateway", "currency"}.issubset(adf.columns):
                            from routing_optimiser.forecast_pipeline import _canonical_gateway
                            _mm = pd.read_csv(mid_list_path)
                            _cc = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mm.columns}
                            _gcol, _curcol = _cc.get("gatewayfid"), _cc.get("currency")
                            if _gcol and _curcol:
                                _mm["_g"] = _mm[_gcol].map(_canonical_gateway).astype(str).str.strip().str.lower()
                                _mm["_c"] = _mm[_curcol].astype(str).str.strip().str.lower()
                                _mm = _mm[~_mm["_c"].isin(["", "excluded", "nan", "none"])]
                                allowed = _mm[["_g", "_c"]].drop_duplicates()
                                allowed["_ok"] = True
                                gw_in_map = set(_mm["_g"])
                                adf["_g"] = adf["gateway"].astype(str).str.strip().str.lower()
                                adf["_c"] = adf["currency"].astype(str).str.strip().str.lower()
                                adf = adf.merge(allowed, on=["_g", "_c"], how="left")
                                _mismatch = adf["_g"].isin(gw_in_map) & adf["_ok"].isna()
                                _ndrop = int(_mismatch.sum())
                                adf = adf[~_mismatch].drop(columns=["_g", "_c", "_ok"]).reset_index(drop=True)
                                log(f"   currency filter: dropped {_ndrop:,} attempt rows where currency disagreed with Master MID currency")
                    except Exception as e:
                        log(f"   [Warning] currency filter skipped: {e}")

                    # Remove gateways switched off in gateway_volume_overrides.json
                    # (target == 0 with apply_to "trx" or "both") - not eligible.
                    try:
                        _ovr_path = os.path.join(PROJECT_ROOT, "config", "inputs", "gateway_volume_overrides.json")
                        if os.path.exists(_ovr_path) and "gateway" in adf.columns:
                            import json as _json
                            from routing_optimiser.forecast_pipeline import _canonical_gateway
                            with open(_ovr_path) as _fh:
                                _ovr = _json.load(_fh)
                            _excl = _switched_off_gateways(_ovr)
                            if _excl:
                                _gg = adf["gateway"].map(_canonical_gateway).astype(str).str.strip().str.lower()
                                _drop = _gg.isin(_excl)
                                _nd = int(_drop.sum())
                                adf = adf[~_drop].reset_index(drop=True)
                                log(f"   volume-override filter: dropped {_nd:,} rows for {len(_excl)} switched-off gateways (target=0, trx/both)")
                    except Exception as e:
                        log(f"   [Warning] volume-override filter skipped: {e}")

                    # Keep an ALL-RPGT (cleaned) copy for the engine SCORE in Bank×Currency mode
                    # (the score pools every transaction type for the cell). Then narrow the
                    # attempts/forecast to the SELECTED RPGTs, which drive eligibility, the volume
                    # routed and the VAMP cap ('all for score, selected for volume/VAMP').
                    _adf_all_rpgts = adf.copy()
                    if _do_rpgt_filter:
                        _n0, _f0 = len(adf), len(forecast_temp)
                        if "rpgt" in adf.columns:
                            adf = adf[adf["rpgt"].astype(str).str.strip().str.lower().isin(_sel_rpgts)].copy()
                        if "rpgt" in forecast_temp.columns:
                            forecast_temp = forecast_temp[
                                forecast_temp["rpgt"].astype(str).str.strip().str.lower().isin(_sel_rpgts)].copy()
                        log(f"   RPGT scope: volume/eligibility/VAMP restricted to {len(_sel_rpgts)} RPGT(s) "
                            f"({len(adf):,}/{_n0:,} attempts, {len(forecast_temp):,}/{_f0:,} forecast rows); "
                            f"score {'per selected RPGT' if _score_by_rpgt else 'pooled over ALL RPGTs'}.")
                        if getattr(adf, "empty", True) or getattr(forecast_temp, "empty", True):
                            raise ValueError(
                                "RPGT scope removed all rows — the selected RPGTs in tab 2 don't "
                                "match the attempts/forecast data. Widen the RPGT selection.")

                    orig_adf = adf.copy()
                    orig_forecast = forecast_temp.copy()

                    agg_forecast = forecast_temp.copy()

                    # Eligibility + success rates come from the LAST 30 DAYS of
                    # attempts only (window ending at the attempts-end date), at
                    # Bank x Currency level. Gateways with no attempts in that
                    # window (stale/other-currency noise) are NOT eligible.
                    agg_adf = adf.copy()
                    _dc = "date" if "date" in agg_adf.columns else ("Date" if "Date" in agg_adf.columns else None)
                    if _dc:
                        _d = pd.to_datetime(agg_adf[_dc], errors="coerce")
                        if _d.notna().any():
                            _mx = pd.to_datetime(attempts_end)
                            if pd.isna(_mx):
                                _mx = _d.max()
                            _win = (_d > (_mx - pd.Timedelta(days=30))) & (_d <= _mx)
                            if _win.sum() > 0:
                                agg_adf = agg_adf[_win].copy()
                                log(f"   eligibility window: {_win.sum():,} attempt rows in 30D ending {_mx.date()}")
                    # ---- DATA-SHAPE DIAGNOSTICS (verbose) --------------------------------
                    try:
                        # [FN-311]
                        def _shape(df, name):
                            if df is None or getattr(df, "empty", True):
                                _diag(f"      {name}: (empty)"); return
                            _cols = {c.lower(): c for c in df.columns}
                            _cur = _cols.get("currency"); _gw = _cols.get("gateway"); _rp = _cols.get("rpgt")
                            _bits = [f"rows={len(df):,}"]
                            if _cur: _bits.append(f"currencies={df[_cur].nunique()}")
                            if _gw: _bits.append(f"gateways={df[_gw].nunique()}")
                            if _rp: _bits.append(f"rpgts={df[_rp].nunique()}")
                            for _amt in ("attempts", "successes", "amount", "volume"):
                                if _amt in _cols:
                                    _bits.append(f"Σ{_amt}={pd.to_numeric(df[_cols[_amt]], errors='coerce').sum():,.0f}")
                            _diag(f"      {name}: " + " · ".join(_bits))
                        _diag("②·diag DATA SHAPES after pre-processing/filters:")
                        _shape(locals().get("agg_adf"), "attempts (post-filter, eligibility window)")
                        _shape(locals().get("agg_forecast"), "forecast baseline (routing volume)")
                        _rpgts_all = locals().get("_all_rpgts"); _rpgts_sel = locals().get("_sel_rpgts")
                        if _rpgts_all is not None:
                            _diag(f"      RPGTs available={sorted(map(str, _rpgts_all))[:12]} · scoped={('ALL' if not _rpgts_sel else sorted(map(str, _rpgts_sel)))}")
                    except Exception as _e:  # noqa: BLE001
                        _diag(f"   [data-shape diagnostics failed: {_e}]")

                    if "original_bank_name" in agg_adf.columns:
                        valid_banks = agg_adf[agg_adf["original_bank_name"].str.strip() != ""]
                        bin_to_bank = valid_banks.groupby("bank")["original_bank_name"].agg(lambda x: x.mode()[0] if not x.mode().empty else "UNKNOWN").to_dict()
                    else:
                        bin_to_bank = {b: b for b in agg_adf["bank"].unique()}

                    agg_forecast["parent_bank"] = agg_forecast["bank"].map(bin_to_bank).fillna(agg_forecast["bank"])
                    agg_adf["parent_bank"] = agg_adf["bank"].map(bin_to_bank).fillna(agg_adf["bank"])

                    # Optimisation grain (the CELL grain — where the split is made & traffic moved).
                    # Bank×Currency collapses RPGT into ONE cell per (currency, parent_bank);
                    # Bank×Currency×RPGT keeps RPGT in the cell key (a split per RPGT). Cell keys
                    # use `_gk`. The Engine Score grain (_score_by_rpgt) is separate and is aligned
                    # to these cells below.
                    _gk = (["rpgt", "currency", "parent_bank"] if _opt_by_rpgt
                           else ["currency", "parent_bank"])
                    log(f"   Optimisation grain: {'Bank×Currency×RPGT (per-RPGT cells)' if _opt_by_rpgt else 'Bank×Currency (RPGT collapsed)'}; "
                        f"Engine Score grain: {'per-RPGT' if _score_by_rpgt else 'Bank×Currency (pooled)'}.")

                    # Total forecast volume to route per cell.
                    # The forecast is used ONLY for how much volume to route, not
                    # for which gateways are eligible.
                    fc_tot = (agg_forecast.groupby(_gk)["volume"].sum()
                              .rename("fc_volume").reset_index())

                    # Eligibility comes from the last 30D Raw Attempts: any gateway
                    # with attempts in the window is a candidate the engine may use.
                    agg_adf = agg_adf[agg_adf["attempts"] > 0]
                    agg_adf = agg_adf.groupby(_gk + ["gateway"]).sum(numeric_only=True).reset_index()
                    if not _opt_by_rpgt:
                        agg_adf["rpgt"] = "ALL_RPGTS"
                    agg_adf["bank"] = agg_adf["parent_bank"]

                    # Build the routing cells from those attempts-based gateways.
                    # Current split (baseline_share) = each gateway's 30D attempts
                    # share; per-gateway volume = forecast cell total x that share
                    # (falls back to 30D attempts volume if the forecast doesn't
                    # cover the bank), so the cell total still equals the forecast.
                    att = agg_adf[_gk + ["gateway", "attempts"]].copy()
                    att["cell_att"] = att.groupby(_gk)["attempts"].transform("sum")
                    att["baseline_share"] = np.where(att["cell_att"] > 0, att["attempts"] / att["cell_att"], 0.0)
                    att = att.merge(fc_tot, on=_gk, how="left")
                    att["fc_volume"] = att["fc_volume"].fillna(att["cell_att"])
                    att["volume"] = att["fc_volume"] * att["baseline_share"]

                    agg_forecast = att[_gk + ["gateway", "volume", "baseline_share"]].copy()
                    if not _opt_by_rpgt:
                        agg_forecast["rpgt"] = "ALL_RPGTS"
                    agg_forecast["bank"] = agg_forecast["parent_bank"]

                    # Attach period-0 risk (from bin_rpgt_impact_export via the
                    # forecast's risk_rate), volume-weighted across RPGTs to the
                    # Bank x Currency x gateway grain. Gateways with no forecast
                    # risk fall back to the default.
                    agg_forecast["risk_rate"] = np.nan
                    if "risk_rate" in orig_forecast.columns:
                        rf = orig_forecast.copy()
                        rf["parent_bank"] = rf["bank"].map(bin_to_bank).fillna(rf["bank"])
                        rf["_ck"] = rf["currency"].astype(str).str.strip().str.lower()
                        rf["_pk"] = rf["parent_bank"].astype(str).str.strip().str.lower()
                        rf["_gwk"] = rf["gateway"].astype(str).str.strip().str.lower()
                        rf["_rk"] = rf["rpgt"].astype(str).str.strip().str.lower()
                        rf["_vw"] = pd.to_numeric(rf["risk_rate"], errors="coerce").fillna(0.0) * pd.to_numeric(rf["volume"], errors="coerce").fillna(0.0)
                        # Risk rate follows the OPTIMISATION (cell) grain: per-RPGT when cells are
                        # per-RPGT, else pooled across RPGTs (classic behaviour).
                        _rkeys = (["_rk"] if _opt_by_rpgt else []) + ["_ck", "_pk", "_gwk"]
                        rr = rf.groupby(_rkeys).agg(_vw=("_vw", "sum"), _v=("volume", "sum")).reset_index()
                        rr["risk_cb"] = np.where(rr["_v"] > 0, rr["_vw"] / rr["_v"], np.nan)
                        agg_forecast["_ck"] = agg_forecast["currency"].astype(str).str.strip().str.lower()
                        agg_forecast["_pk"] = agg_forecast["parent_bank"].astype(str).str.strip().str.lower()
                        agg_forecast["_gwk"] = agg_forecast["gateway"].astype(str).str.strip().str.lower()
                        agg_forecast["_rk"] = agg_forecast["rpgt"].astype(str).str.strip().str.lower()
                        # Carry the risk-rate DENOMINATOR (_v = the Txn/sales count) as risk_n so the
                        # portfolio σ uses the VAMP-rate's own sample size, not auth attempts. (C1)
                        agg_forecast = agg_forecast.merge(rr[_rkeys + ["risk_cb", "_v"]], on=_rkeys, how="left")
                        agg_forecast["risk_rate"] = agg_forecast["risk_cb"]
                        agg_forecast["risk_n"] = pd.to_numeric(agg_forecast["_v"], errors="coerce").fillna(0.0)
                        agg_forecast = agg_forecast.drop(columns=["_ck", "_pk", "_gwk", "_rk", "risk_cb", "_v"])
                    # Seed the VAMP rate of NO-VAMP-DATA gateways (risk_rate 0/NaN or risk_n==0 — e.g.
                    # WoodForest, whose 0 is a data gap, NOT true zero-risk) with the risk_n-weighted
                    # average VAMP rate of gateways WITH data at the OPTIMISATION-grain cell, so they
                    # aren't treated as risk-free and over-favoured by the risk-min dial. Falls back to
                    # the currency-level, then global weighted rate, then the 0.006 default below.
                    try:
                        _rrv = pd.to_numeric(agg_forecast["risk_rate"], errors="coerce")
                        _rnv = pd.to_numeric(agg_forecast.get("risk_n", 0.0), errors="coerce").fillna(0.0)
                        _has_vamp = (_rrv > 0) & (_rnv > 0)
                        if _has_vamp.any():
                            _w = agg_forecast[list(dict.fromkeys(_gk + ["currency"]))].copy()
                            _w["_vc"] = np.where(_has_vamp, _rrv.fillna(0.0) * _rnv, 0.0)   # vampCount = rate × denom
                            _w["_rn"] = np.where(_has_vamp, _rnv, 0.0)
                            _cg = _w.groupby(_gk, as_index=False).agg(_vc=("_vc", "sum"), _rn=("_rn", "sum"))
                            _cg["_cellrate"] = np.where(_cg["_rn"] > 0, _cg["_vc"] / _cg["_rn"], np.nan)
                            agg_forecast = agg_forecast.merge(_cg[_gk + ["_cellrate"]], on=_gk, how="left")
                            _cc = _w.groupby("currency", as_index=False).agg(_vc=("_vc", "sum"), _rn=("_rn", "sum"))
                            _cc["_currate"] = np.where(_cc["_rn"] > 0, _cc["_vc"] / _cc["_rn"], np.nan)
                            agg_forecast = agg_forecast.merge(_cc[["currency", "_currate"]], on="currency", how="left")
                            _globrate = float(_w["_vc"].sum() / max(_w["_rn"].sum(), 1e-9))
                            _rrv2 = pd.to_numeric(agg_forecast["risk_rate"], errors="coerce")
                            _rnv2 = pd.to_numeric(agg_forecast["risk_n"], errors="coerce").fillna(0.0)
                            _nodata = ~((_rrv2 > 0) & (_rnv2 > 0))
                            _seedrate = agg_forecast["_cellrate"].fillna(agg_forecast["_currate"]).fillna(_globrate)
                            agg_forecast["risk_rate"] = np.where(_nodata.to_numpy(), _seedrate.to_numpy(), _rrv2.to_numpy())
                            agg_forecast = agg_forecast.drop(columns=["_cellrate", "_currate"])
                            log(f"   risk seeding: {int(_nodata.sum()):,} gateway-cell(s) with NO VAMP data seeded from "
                                f"the opt-grain weighted-avg VAMP rate (currency/global fallback {_globrate:.4f}) — "
                                "so 0-VAMP gateways aren't treated as risk-free.")
                    except Exception as _e:  # noqa: BLE001
                        log(f"   [Warning] no-VAMP-data risk seeding skipped: {_e}")
                    agg_forecast["risk_rate"] = pd.to_numeric(agg_forecast["risk_rate"], errors="coerce").fillna(0.006)

                    # ---- New-gateway EXPLORATION eligibility ------------------------------
                    # Eligibility above is built from OBSERVED 30D attempts, so a capable-but-
                    # untested gateway (no attempts for a bank) is never a candidate and gets ZERO
                    # volume under every engine/dial. We add capable-but-untested gateways as
                    # candidates (volume 0 / baseline 0) so the engine CAN explore them. Two sources:
                    #   • explore_untested_gateways list (manual, always on), and
                    #   • the auto toggle → every gateway approved for the cell's currency in
                    #     Master_MID_List (minus scrubbed / switched-off).
                    # The injected rows are SEEDED below (after the score is built) at the bank×
                    # currency AVERAGE rate as a WEAK prior — so Thompson keeps a wide posterior
                    # (natural exploration) and softmax scores them at the local average + the
                    # exploration floor. `_inj_fc_keys` tracks them for that seeding step.
                    _inj_fc_keys = []
                    try:
                        from routing_optimiser.eligibility import load_explore_gateways as _load_expl
                        from routing_optimiser.forecast_pipeline import _canonical_gateway as _cg_ex
                        _rr_p = os.path.join(PROJECT_ROOT, "config", "inputs", "routing_restrictions.json")
                        _explore = set(_load_expl(_rr_p))
                        _fid_cur = {}
                        _fid_brand, _fid_active, _fid_proc = {}, {}, {}
                        _norm_b = lambda s: str(s).strip().lower().replace(" ", "")
                        if os.path.exists(mid_list_path):
                            _mmx = pd.read_csv(mid_list_path)
                            _ccx = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmx.columns}
                            _gx, _cx = _ccx.get("gatewayfid"), _ccx.get("currency")
                            _bx, _ax, _px = _ccx.get("brand"), _ccx.get("isactive"), _ccx.get("gateway")
                            if _gx and _cx:
                                _gcol = _mmx[_gx].map(_cg_ex).astype(str).str.strip().str.lower()
                                _rng = range(len(_mmx))
                                for _i, _g, _c in zip(_rng, _gcol,
                                                      _mmx[_cx].astype(str).str.strip().str.lower()):
                                    if _c in ("", "excluded", "nan", "none"):
                                        continue
                                    _fid_cur.setdefault(_g, _c)
                                    if _bx and _g not in _fid_brand:
                                        _fid_brand[_g] = _norm_b(_mmx[_bx].iloc[_i])
                                    if _ax and _g not in _fid_active:
                                        _fid_active[_g] = str(_mmx[_ax].iloc[_i]).strip().lower() in ("true", "1", "yes", "t", "y")
                                    if _px and _g not in _fid_proc:
                                        _fid_proc[_g] = _norm_b(_mmx[_px].iloc[_i])
                        if _auto_explore:   # currency-capable gateways, filtered + minus scrubbed / switched-off
                            import json as _json
                            _skip = set()
                            for _pth, _key in [(os.path.join(PROJECT_ROOT, "config", "inputs", "test_gateways.json"), "scrub")]:
                                try:
                                    if os.path.exists(_pth):
                                        with open(_pth) as _fpth:
                                            _j = _json.load(_fpth)
                                        _skip |= {str(_cg_ex(g)).strip().lower() for g in (_j.get(_key, []) if isinstance(_j, dict) else [])}
                                except Exception:  # noqa: BLE001
                                    pass
                            try:
                                _ovp = os.path.join(PROJECT_ROOT, "config", "inputs", "gateway_volume_overrides.json")
                                if os.path.exists(_ovp):
                                    with open(_ovp) as _fov:
                                        _ov = _json.load(_fov)
                                    _skip |= _switched_off_gateways(_ov)
                            except Exception:  # noqa: BLE001
                                pass
                            # Master-MID-list guards for the AUTO capable set:
                            #   • same brand as the run's company (normalised, "TotalAV" == "Total AV"),
                            #   • IsActive = TRUE,
                            #   • processor is NOT PayPal.
                            # (The manual explore_untested_gateways list bypasses these — it's an
                            # explicit user opt-in.)
                            _run_brand = _norm_b(sr_company)
                            _n0 = len([g for g in _fid_cur if g not in _skip])
                            _cand = set()
                            _drop_brand = _drop_inact = _drop_pp = 0
                            for _g in _fid_cur:
                                if _g in _skip:
                                    continue
                                if _fid_active and not _fid_active.get(_g, True):
                                    _drop_inact += 1; continue
                                if _fid_proc.get(_g, "") == "paypal":
                                    _drop_pp += 1; continue
                                if _fid_brand and _run_brand and _fid_brand.get(_g, _run_brand) != _run_brand:
                                    _drop_brand += 1; continue
                                _cand.add(_g)
                            _explore |= _cand
                            log(f"   auto-explore capable set: {len(_cand)} gateway(s) after Master-MID guards "
                                f"(from {_n0}; dropped {_drop_inact} inactive, {_drop_pp} PayPal, "
                                f"{_drop_brand} other-brand vs '{sr_company}').")
                        if _explore:
                            # PER-CELL presence: a gateway present in ONE bank of a currency must still
                            # be injected into OTHER banks of that currency where it's absent (the old
                            # (currency, gateway) check skipped it everywhere → 0 injected → single-
                            # gateway cells never got a fallback → 100% in the export). We backfill
                            # ONLY cells that would otherwise be single-gateway (fewer than _MIN_GW
                            # eligible), so well-populated cells aren't bloated. The explore share cap
                            # keeps the injected fallbacks to ≤10% combined so the primary stays dominant.
                            _MIN_GW = 2
                            _cellkey_cols = _gk + ["bank"]
                            _af_keys = list(zip(*[agg_forecast[c].astype(str).str.strip().str.lower()
                                                  for c in _cellkey_cols]))
                            _af_gw = agg_forecast["gateway"].astype(str).str.strip().str.lower().tolist()
                            _have, _cell_gws = set(), {}
                            for _k, _gw in zip(_af_keys, _af_gw):
                                _have.add(_k + (_gw,))
                                _cell_gws.setdefault(_k, set()).add(_gw)
                            _cells = agg_forecast[_cellkey_cols].drop_duplicates()
                            _cells_cur = _cells["currency"].astype(str).str.strip().str.lower().to_numpy()
                            # OPT-IN volume gate: per-cell total forecast volume, used to skip
                            # exploration injection in near-empty cells. 0 → no gating (unchanged).
                            _expl_min_vol = float(ss.get("explore_min_cell_vol", 0) or 0)
                            _cell_vol_map = {}
                            if _expl_min_vol > 0:
                                _cv_keys = list(zip(*[agg_forecast[c].astype(str).str.strip().str.lower()
                                                      for c in _cellkey_cols]))
                                _cv_vol = pd.to_numeric(agg_forecast["volume"], errors="coerce").fillna(0.0).to_numpy()
                                for _k, _v in zip(_cv_keys, _cv_vol):
                                    _cell_vol_map[_k] = _cell_vol_map.get(_k, 0.0) + float(_v)
                            _pruned_cells = set()
                            _new_rows = []
                            for _g in sorted(_explore):
                                _gc = _fid_cur.get(_g)
                                if not _gc:
                                    continue
                                _sel = _cells[_cells_cur == _gc]
                                for _c in _sel.itertuples(index=False):
                                    _cd = _c._asdict()
                                    _ck = tuple(str(_cd[c]).strip().lower() for c in _cellkey_cols)
                                    if len(_cell_gws.get(_ck, ())) >= _MIN_GW:   # cell already has a fallback
                                        continue
                                    if _ck + (_g,) in _have:
                                        continue
                                    if _expl_min_vol > 0 and _cell_vol_map.get(_ck, 0.0) < _expl_min_vol:
                                        _pruned_cells.add(_ck)          # near-empty cell → skip exploration
                                        continue
                                    _rp = _cd.get("rpgt", "ALL_RPGTS")
                                    _nr = {k: _cd[k] for k in _gk}
                                    _nr.update({"bank": _cd["bank"], "gateway": _g, "volume": 0.0,
                                                "baseline_share": 0.0, "rpgt": _rp,
                                                "risk_rate": np.nan, "risk_n": 0.0,
                                                "is_explore": True})  # risk filled at seeding; capped in reference
                                    _new_rows.append(_nr)
                                    _inj_fc_keys.append((str(_cd["currency"]).strip().lower(),
                                                         str(_cd["bank"]).strip().lower(),
                                                         str(_rp).strip().lower(), _g))
                            if _new_rows:
                                agg_forecast = pd.concat([agg_forecast, pd.DataFrame(_new_rows)], ignore_index=True)
                                log(f"   exploration: injected {len(_new_rows)} fallback candidate row(s) into "
                                    f"single-gateway cells ({len(set(k[3] for k in _inj_fc_keys))} gateway(s), "
                                    f"{'auto: currency-capable' if _auto_explore else 'explore list'}); "
                                    "seeded at the bank×currency average (weak prior) + explore cap.")
                            if _expl_min_vol > 0:
                                log(f"   exploration volume gate ON (min cell volume {_expl_min_vol:,.0f}): "
                                    f"skipped {len(_pruned_cells)} near-empty cell(s) → fewer fallback rows "
                                    "injected, smaller GA matrix (A/B: compare total gateway-rows + dial-0 "
                                    "VAMP/revenue/MIDs-over-cap vs a run at 0).")
                    except Exception as _e:  # noqa: BLE001
                        log(f"   [Warning] new-gateway exploration injection skipped: {_e}")
                        _inj_fc_keys = []

                    # Engine score (success-rate smoothing) uses the FULL attempts
                    # window and the Bank x Currency prior, matching the All-Time
                    # columns. Eligibility above stays on the 30D window; the rate
                    # estimate uses all available history for a stabler number.
                    # SCORE source: Bank×Currency pools ALL RPGTs for the cell (use the all-RPGT
                    # copy); Bank×Currency×RPGT scores each selected RPGT from its own rows.
                    agg_adf_full = (adf.copy() if _score_by_rpgt else _adf_all_rpgts.copy())
                    if "date" not in agg_adf_full.columns and "Date" in agg_adf_full.columns:
                        agg_adf_full = agg_adf_full.rename(columns={"Date": "date"})
                    agg_adf_full["parent_bank"] = agg_adf_full["bank"].map(bin_to_bank).fillna(agg_adf_full["bank"])
                    agg_adf_full = agg_adf_full[agg_adf_full["attempts"] > 0].copy()
                    if not _score_by_rpgt:
                        agg_adf_full["rpgt"] = "ALL_RPGTS"   # per-RPGT keeps real rpgt -> per-RPGT rates
                    agg_adf_full["bank"] = agg_adf_full["parent_bank"]

                    # Pass the DATED rows (NOT pre-summed) so the time-decay half-life
                    # weights recent attempts before the rate is computed; gateway_
                    # success_rates does the grouping. This makes decay affect scores.
                    agg_sr = gateway_success_rates(
                        agg_adf_full, shrink_strength=float(shrink),
                        time_decay_half_life_days=(float(decay_half) if apply_decay else None),
                        prior_scope=("rpgt", "currency", "bank"), empirical_bayes=use_eb)

                    log(f"   {len(agg_sr):,} dense aggregated cell × gateway success rates (full-window rate, 30D eligibility)")

                    # Cross-border penalty: multiply the Engine Score (smoothed SR)
                    # by xborder_penalty for gateways flagged isCrossBorder=TRUE in the
                    # Master MID list, so they get a smaller proposed share.
                    xborder_fids = set()
                    try:
                        if os.path.exists(mid_list_path):
                            _mmx = pd.read_csv(mid_list_path)
                            _ccx = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmx.columns}
                            _gx, _xb = _ccx.get("gatewayfid"), _ccx.get("iscrossborder")
                            if _gx and _xb:
                                from routing_optimiser.forecast_pipeline import _canonical_gateway
                                _flag = _mmx[_xb].astype(str).str.strip().str.upper().isin(["TRUE", "T", "1", "YES", "Y"])
                                xborder_fids = set(_mmx.loc[_flag, _gx].map(_canonical_gateway).astype(str).str.strip().str.lower())
                    except Exception as e:
                        log(f"   [Warning] cross-border flag load skipped: {e}")
                    if xborder_fids and xborder_penalty is not None:
                        _xmask = agg_sr["gateway"].astype(str).str.strip().str.lower().isin(xborder_fids)
                        agg_sr.loc[_xmask, "success_rate"] = agg_sr.loc[_xmask, "success_rate"] * float(xborder_penalty)
                        log(f"   cross-border penalty {xborder_penalty:.0%} applied to {int(_xmask.sum())} gateway cells "
                            f"({len(xborder_fids)} cross-border FIDs)")
                    ss["xborder_fids"] = xborder_fids

                    # Align the Engine-Score grain to the Optimisation (cell) grain so the score
                    # attaches to every cell. build_cell_problems joins on (rpgt, currency, bank,
                    # gateway), so agg_sr's rpgt values must match agg_forecast's:
                    #   • score coarser than opt (score=Bank×Currency, opt=per-RPGT): BROADCAST the
                    #     pooled bank rate to each RPGT cell.
                    #   • score finer than opt (score=per-RPGT, opt=Bank×Currency): POOL the per-RPGT
                    #     rates up to one bank rate (attempt-weighted), tagged ALL_RPGTS.
                    if _opt_by_rpgt and not _score_by_rpgt:
                        _rpgts_opt = sorted(agg_forecast["rpgt"].astype(str).unique().tolist())
                        _base_sr = agg_sr.drop(columns=[c for c in ["rpgt"] if c in agg_sr.columns])
                        agg_sr = pd.concat([_base_sr.assign(rpgt=_rp) for _rp in _rpgts_opt], ignore_index=True)
                        log(f"   score→opt align: broadcast pooled Bank×Currency score to {len(_rpgts_opt)} RPGT cell-grain(s).")
                    elif (not _opt_by_rpgt) and _score_by_rpgt:
                        _s = agg_sr.copy()
                        _s["_w"] = pd.to_numeric(_s["attempts"], errors="coerce").fillna(0.0)
                        if "success" not in _s.columns:
                            _s["success"] = _s["success_rate"] * _s["_w"]
                        if "prior_rate" not in _s.columns:
                            _s["prior_rate"] = _s["success_rate"]
                        _s["_wr"] = _s["success_rate"] * _s["_w"]
                        _s["_wp"] = _s["prior_rate"] * _s["_w"]
                        _aggc = dict(attempts=("attempts", "sum"), success=("success", "sum"),
                                     _wr=("_wr", "sum"), _wp=("_wp", "sum"), _w=("_w", "sum"))
                        if "kappa" in _s.columns:
                            _aggc["kappa"] = ("kappa", "first")
                        _agg = _s.groupby(["currency", "bank", "gateway"], as_index=False).agg(**_aggc)
                        _agg["success_rate"] = np.where(_agg["_w"] > 0, _agg["_wr"] / _agg["_w"],
                                                        np.where(_agg["attempts"] > 0, _agg["success"] / _agg["attempts"], 0.0))
                        _agg["prior_rate"] = np.where(_agg["_w"] > 0, _agg["_wp"] / _agg["_w"], _agg["success_rate"])
                        _agg["rpgt"] = "ALL_RPGTS"
                        agg_sr = _agg.drop(columns=["_wr", "_wp", "_w"])
                        log("   score→opt align: pooled per-RPGT score up to Bank×Currency (attempt-weighted).")

                    # Seed the injected exploration candidates at the bank×currency AVERAGE rate
                    # (a WEAK prior). For each injected (currency, bank, rpgt) cell we take the mean
                    # success/prior rate over the cell's RATED gateways and add an agg_sr row for the
                    # untested gateway with attempts=0 — so Thompson keeps a WIDE posterior (weak
                    # pseudo-count → natural exploration, no dilution cap needed) and softmax scores
                    # it at the local average + the exploration floor. The injected forecast risk is
                    # filled with the cell-average risk (else the 0.006 default).
                    if _inj_fc_keys:
                        try:
                            _sr = agg_sr.copy()
                            _sr["_ck"] = _sr["currency"].astype(str).str.strip().str.lower()
                            _sr["_bk"] = _sr["bank"].astype(str).str.strip().str.lower()
                            _sr["_rk"] = (_sr["rpgt"].astype(str).str.strip().str.lower()
                                          if "rpgt" in _sr.columns else "all_rpgts")
                            if "prior_rate" not in _sr.columns:
                                _sr["prior_rate"] = _sr["success_rate"]
                            _cellavg = _sr.groupby(["_ck", "_bk", "_rk"]).agg(
                                _sr_m=("success_rate", "mean"), _pr_m=("prior_rate", "mean")).to_dict("index")
                            _glob_sr = float(pd.to_numeric(agg_sr["success_rate"], errors="coerce").mean()) if len(agg_sr) else 0.85
                            _af = agg_forecast.copy()
                            _af["_ck"] = _af["currency"].astype(str).str.strip().str.lower()
                            _af["_bk"] = _af["bank"].astype(str).str.strip().str.lower()
                            _af["_rk"] = _af["rpgt"].astype(str).str.strip().str.lower()
                            _riskavg = (_af[pd.to_numeric(_af["risk_rate"], errors="coerce").notna()]
                                        .groupby(["_ck", "_bk", "_rk"])["risk_rate"].mean().to_dict())
                            # ---- Sibling-processor prior (#9): if an untested gatewayFid's PROCESSOR
                            # (Master-MID 'gateway' col) + brand + currency has other gatewayFids WITH
                            # data, seed it from their volume-weighted average instead of the cell mean
                            # (a same-processor rate is a better prior than the bank×currency average).
                            _fid_pb = {}
                            try:
                                from routing_optimiser.forecast_pipeline import _canonical_gateway as _cg_sib
                                if os.path.exists(mid_list_path):
                                    _mms = pd.read_csv(mid_list_path)
                                    _ccs = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mms.columns}
                                    _gs, _ps, _bs = _ccs.get("gatewayfid"), _ccs.get("gateway"), _ccs.get("brand")
                                    if _gs and _ps:
                                        _brc = (_mms[_bs].astype(str).str.strip().str.lower() if _bs else pd.Series([""] * len(_mms)))
                                        for _f, _p, _br in zip(_mms[_gs].map(_cg_sib).astype(str).str.strip().str.lower(),
                                                               _mms[_ps].astype(str).str.strip().str.lower(), _brc):
                                            _fid_pb.setdefault(_f, (_p, _br))
                            except Exception:  # noqa: BLE001
                                _fid_pb = {}
                            _sib = {}
                            try:
                                _srr = _sr.copy()
                                _srr["_att"] = pd.to_numeric(_srr.get("attempts", 0.0), errors="coerce").fillna(0.0)
                                _srr = _srr[_srr["_att"] > 0]
                                _gwl = _srr["gateway"].astype(str).str.strip().str.lower()
                                _srr["_proc"] = _gwl.map(lambda g: _fid_pb.get(g, ("", ""))[0])
                                _srr["_brand"] = _gwl.map(lambda g: _fid_pb.get(g, ("", ""))[1])
                                _srr = _srr[_srr["_proc"] != ""]
                                if len(_srr):
                                    _srr["_wsr"] = pd.to_numeric(_srr["success_rate"], errors="coerce").fillna(0.0) * _srr["_att"]
                                    _srr["_wpr"] = pd.to_numeric(_srr["prior_rate"], errors="coerce").fillna(0.0) * _srr["_att"]
                                    _sg = _srr.groupby(["_proc", "_brand", "_ck"]).agg(
                                        _wsr=("_wsr", "sum"), _wpr=("_wpr", "sum"), _w=("_att", "sum"))
                                    _sib = {k: (float(r["_wsr"] / r["_w"]), float(r["_wpr"] / r["_w"]))
                                            for k, r in _sg.iterrows() if r["_w"] > 0}
                            except Exception:  # noqa: BLE001
                                _sib = {}
                            # Layer 2: CROSS-BRAND processor benchmark {(processor, currency): rate},
                            # populated from processor_benchmark.sql (all brands, same processor +
                            # Engine-Score grain). Empty unless that query has been run → then this
                            # layer is skipped and we fall through to the bank×currency average.
                            _proc_bench = ss.get("processor_benchmark") or {}
                            _sr_rows = []
                            _seen_sr = set()
                            _n_sib = _n_xbrand = 0
                            for (_c, _b, _rp, _g) in _inj_fc_keys:
                                if (_c, _b, _rp, _g) in _seen_sr:
                                    continue
                                _seen_sr.add((_c, _b, _rp, _g))
                                _pb = _fid_pb.get(_g)
                                _sa = _sib.get((_pb[0], _pb[1], _c)) if _pb else None
                                _xb = _proc_bench.get((_pb[0], _c)) if _pb else None
                                if _sa is not None:                       # L1: same processor+brand+currency
                                    _srv, _prv = float(_sa[0]), float(_sa[1])
                                    _n_sib += 1
                                elif _xb is not None:                     # L2: same processor+currency, ANY brand
                                    _srv = _prv = float(_xb)
                                    _n_xbrand += 1
                                else:                                     # L3: bank×currency average
                                    _a = _cellavg.get((_c, _b, _rp))
                                    _srv = float(_a["_sr_m"]) if _a else _glob_sr
                                    _prv = float(_a["_pr_m"]) if _a else _srv
                                _row = {"rpgt": (_rp if _rp != "all_rpgts" else "ALL_RPGTS"),
                                        "currency": _c, "bank": _b, "gateway": _g,
                                        "success_rate": _srv, "prior_rate": _prv,
                                        "attempts": 0.0, "success": 0.0}
                                if "kappa" in agg_sr.columns:
                                    _row["kappa"] = 8.0   # weak pseudo-count → wide posterior (Thompson explores)
                                _sr_rows.append(_row)
                            if _sr_rows:
                                _new_sr = pd.DataFrame(_sr_rows).reindex(columns=agg_sr.columns)
                                agg_sr = pd.concat([agg_sr, _new_sr], ignore_index=True)
                            _nar = pd.to_numeric(agg_forecast["risk_rate"], errors="coerce").isna()
                            if _nar.any():
                                _ck2 = agg_forecast["currency"].astype(str).str.strip().str.lower()
                                _bk2 = agg_forecast["bank"].astype(str).str.strip().str.lower()
                                _rk2 = agg_forecast["rpgt"].astype(str).str.strip().str.lower()
                                agg_forecast.loc[_nar, "risk_rate"] = [
                                    _riskavg.get((c, b, r), 0.006)
                                    for c, b, r in zip(_ck2[_nar], _bk2[_nar], _rk2[_nar])]
                            # Only the Thompson engine actually samples this posterior; for every
                            # other engine (incl. genetic/CMA-ES) it's just a weak, uncertain prior
                            # scored as a point rate — so don't imply Thompson sampling when unused.
                            _wp = ("wide Thompson posterior the Thompson engine samples to explore"
                                   if engine_key == "thompson"
                                   else "weak/uncertain prior, scored cautiously")
                            log(f"   exploration seeding: {len(_sr_rows)} untested gateway cell(s) seeded "
                                f"(attempts=0 → {_wp}): {_n_sib} from a same-"
                                f"processor+brand+currency sibling, {_n_xbrand} from a CROSS-BRAND processor "
                                f"benchmark, {len(_sr_rows) - _n_sib - _n_xbrand} from the bank×currency average.")
                        except Exception as _e:  # noqa: BLE001
                            log(f"   [Warning] exploration seeding skipped: {_e}")
                    agg_forecast["risk_rate"] = pd.to_numeric(agg_forecast["risk_rate"], errors="coerce").fillna(0.006)

                    # AUTHORITATIVE switched-off exclusion (root fix). Drop every gateway turned off in
                    # gateway_volume_overrides.json (target=0, apply_to trx/both) from the routing
                    # CANDIDATE frame, so it can NEVER receive proposed share — whatever path it entered
                    # by. The upstream attempts + auto-explore filters miss a gateway injected as an
                    # UNTESTED-EXPLORATION candidate (0 attempts, 0% score); the enforcement layer then
                    # loads it with "safe" (0-risk) volume to meet VAMP caps — exactly the bancard/cwams
                    # case (~46% of volume on switched-off gateways, dragging Expected SR/revenue down).
                    # This is the single choke through which every candidate reaches build_cell_problems.
                    try:
                        from routing_optimiser.forecast_pipeline import _canonical_gateway as _cg_off
                        _ovr_off = ss.get("gateway_volume_overrides") or {}
                        _off_route = _switched_off_gateways(_ovr_off)
                        if _off_route and "gateway" in agg_forecast.columns:
                            _gwc = agg_forecast["gateway"].map(_cg_off).astype(str).str.strip().str.lower()
                            _dropm = _gwc.isin(_off_route)
                            _noff = int(_dropm.sum())
                            if _noff:
                                _hit = sorted(set(agg_forecast.loc[_dropm, "gateway"].astype(str)))
                                agg_forecast = agg_forecast[~_dropm].reset_index(drop=True)
                                log(f"   switched-off exclusion: removed {_noff} candidate row(s) for "
                                    f"{len(_hit)} gateway(s) turned off in gateway_volume_overrides "
                                    f"(target=0, trx/both) — zero proposed share. Hit: {', '.join(_hit[:12])}"
                                    + (" …" if len(_hit) > 12 else ""))
                    except Exception as _e:  # noqa: BLE001
                        log(f"   [Warning] switched-off candidate exclusion skipped: {_e}")

                    _progress(_f_cells, "Assembling cells…")
                    _stage("③ Assemble routing cells from 30D attempts (forecast supplies volume only)")
                    agg_problems = build_cell_problems(agg_forecast, agg_sr)

                    # ---- CELL / GATEWAY DIAGNOSTICS (verbose) ----------------------------
                    try:
                        _ng = np.array([p.n() for p in agg_problems], dtype=float)
                        _nelig = np.array([int((np.asarray(p.risk_rates) >= 0).sum()) for p in agg_problems], dtype=float) if agg_problems else np.array([])
                        _vols = np.array([float(getattr(p, "volume", 0.0)) for p in agg_problems], dtype=float)
                        _npool = sum(int(np.asarray(getattr(p, "pooled_fallback", np.zeros(p.n(), bool))).sum()) for p in agg_problems)
                        _nexpl = sum(int(np.asarray(getattr(p, "is_explore", np.zeros(p.n(), bool))).sum()) for p in agg_problems)
                        # [FN-312]
                        def _q(a, x):
                            return float(np.quantile(a, x)) if len(a) else 0.0
                        _diag("④·diag ROUTING CELLS assembled:")
                        _diag(f"      cells={len(agg_problems):,} · total gateway-rows={int(_ng.sum()):,} · "
                              f"total forecast volume={_vols.sum():,.0f}")
                        if len(_ng):
                            _diag(f"      gateways/cell: min={int(_ng.min())} p50={int(_q(_ng,0.5))} "
                                  f"mean={_ng.mean():.1f} p95={int(_q(_ng,0.95))} max={int(_ng.max())}")
                            _diag(f"      cells with 1 gateway (cap unsatisfiable): {int((_ng <= 1).sum()):,} · "
                                  f"cells >50 gateways: {int((_ng > 50).sum()):,}")
                        _diag(f"      gateway-rows on POOLED prior (no per-cell attempts): {_npool:,} · "
                              f"auto-explore injected rows: {_nexpl:,}")
                        # currency / bank / RPGT spread
                        _curs = sorted({str(p.currency) for p in agg_problems})
                        _rpgts = sorted({str(p.rpgt) for p in agg_problems})
                        _banks = len({(str(p.currency), str(p.bank)) for p in agg_problems})
                        _diag(f"      currencies={_curs} · distinct banks(×cur)={_banks:,} · rpgt grain values={_rpgts[:8]}"
                              + (" …" if len(_rpgts) > 8 else ""))
                    except Exception as _e:  # noqa: BLE001
                        _diag(f"   [cell diagnostics failed: {_e}]")

                    # Temperature for the softmax-based reference. Softmax, Portfolio and Genetic
                    # ALL build their slider-100 (revenue) reference with the softmax engine, so
                    # they must share the SAME temperature — otherwise the revenue endpoint is
                    # flatter and earns less than the softmax benchmark. Softmax honours the user's
                    # temp method; Portfolio/Genetic have no temp control, so they always use the
                    # (parameter-free) variance-scaled temperature. Thompson uses its own engine.
                    cell_temp = {}
                    if engine_key == "softmax":
                        params["temperature"] = float(softmax_temperature)
                    # Variance-scaled temperature only shapes the SOFTMAX reference. Genetic now
                    # builds its OWN (waterfall) reference with no temperature, and Thompson/
                    # portfolio build their own references — so it applies to softmax only.
                    _do_vs = (engine_key == "softmax" and temp_method == "Variance-Scaled (auto)")
                    if _do_vs:
                        cell_temp, _medz, _scl = _variance_gap_temp(agg_sr)
                        _matched = 0
                        for p in agg_problems:
                            t = cell_temp.get((str(p.currency).strip().lower(), str(p.bank).strip().lower()))
                            if t is not None:
                                p.temperature = float(t)
                                _matched += 1
                        if cell_temp:
                            log(f"   variance-scaled temperature: set on {_matched} cell(s) "
                                f"(shared with the revenue reference); median gap t-stat={_medz:.2f}, "
                                f"range {min(cell_temp.values()):.3f}–{max(cell_temp.values()):.3f}.")
                        else:
                            log("   variance-scaled temperature: no valid cells; using fallback 0.170.")

                    # 5 variations (0, 25, 50, 75, 100) instead of 21: each non-reference
                    # weight re-runs the full granular enforcement (VAMP recap + per-MID /
                    # per-(MID×RPGT) cap scaling) on the exploded split, which is the slow
                    # part — so fewer stops ≈ proportionally faster with per-MID constraints.
                    # Only the risk-minimised compliant endpoint (dial 0) is produced now — the
                    # max-revenue ceiling (dial 100) was removed at the user's request. The greedy
                    # revenue split is STILL computed internally as a GA warm-start seed, so dial 0
                    # is unchanged; dropping dial 100 just skips its pool-compression pass. Every
                    # variation-builder below loops `weights` and only makes a dial-100 entry when
                    # w>=1.0, so a single [0.0] weight yields exactly the dial-0 split.
                    _N_VARIATIONS = 1   # dial 0 only (compliant); dial 100 ceiling removed
                    weights = [round(float(w), 2) for w in np.linspace(0.0, 0.0, _N_VARIATIONS)]
                    _progress(_f_eng, "Running engine…")   # adaptive ETA (see _f_* above)
                    _stage(f"④ Run {engine_key} engine across the Risk↔Conversion axis")
                    log(f"   {_N_VARIATIONS} dials: {', '.join(str(int(round(w * 100))) for w in weights)}")

                    mapping_df = orig_forecast[["rpgt", "currency", "bank"]].drop_duplicates()
                    mapping_df["parent_bank"] = mapping_df["bank"].map(bin_to_bank).fillna(mapping_df["bank"])
                    mapping_df = mapping_df.rename(columns={"rpgt": "orig_rpgt", "bank": "orig_bank"})

                    # [FN-313]
                    def _explode(agg_split):
                        # Per-RPGT optimisation: cells already carry the real RPGT — the split IS
                        # the exploded split, so map parent_bank back to the BIN-level bank(s) but
                        # keep each cell's own RPGT (no fan-out across RPGTs).
                        if _opt_by_rpgt:
                            sc = agg_split.copy()
                            sc["_rk"] = sc["rpgt"].astype(str).str.strip().str.lower()
                            if "bank" in sc.columns:
                                sc = sc.rename(columns={"bank": "parent_bank"})
                            _mp = orig_forecast[["rpgt", "currency", "bank"]].drop_duplicates().copy()
                            _mp["parent_bank"] = _mp["bank"].map(bin_to_bank).fillna(_mp["bank"])
                            _mp["_rk"] = _mp["rpgt"].astype(str).str.strip().str.lower()
                            _mp = _mp.rename(columns={"rpgt": "_orig_rpgt", "bank": "_orig_bank"}).drop(columns=["rpgt"], errors="ignore")
                            ex = _mp.merge(sc.drop(columns=["rpgt"], errors="ignore"),
                                           on=["currency", "parent_bank", "_rk"], how="inner")
                            return ex.rename(columns={"_orig_rpgt": "rpgt", "_orig_bank": "bank"}).drop(
                                columns=["parent_bank", "_rk"], errors="ignore")
                        sc = agg_split.copy()
                        if "rpgt" in sc.columns:
                            sc = sc.drop(columns=["rpgt"])
                        if "bank" in sc.columns:
                            sc = sc.rename(columns={"bank": "parent_bank"})
                        if "currency" not in sc.columns:
                            sc["currency"] = mapping_df["currency"].iloc[0] if not mapping_df.empty else "usd"
                        ex = mapping_df.merge(sc, on=["currency", "parent_bank"], how="inner")
                        return ex.rename(columns={"orig_rpgt": "rpgt", "orig_bank": "bank"}).drop(columns=["parent_bank"])

                    # Reference = conversion-optimal split (no per-cell cap). The risk
                    # constraint is applied CROSS-CELL, per vampMid, afterwards.
                    from routing_optimiser import optimiser as _optmod
                    from routing_optimiser.optimiser import (enforce_mid_vamp_caps, enforce_mid_volume_caps)
                    from routing_optimiser.genetic_global import run_midtilt_ga as _run_midtilt_ga
                    # The Genetic engine runs on a rule-safe PRE-CLUSTERED problem by default
                    # (ss['ga_precluster'], set above; ROUTING_GA_PRECLUSTER=0 disables it). The wrapper
                    # is a drop-in for run_midtilt_ga — it rule-safely clusters cells, runs the SAME GA on
                    # the reduced problem, and expands the result back (near-lossless; see
                    # routing_optimiser.precluster). Same call/return contract, so every call site (incl.
                    # the parallel loky workers and the numba pre-compile) is unchanged. If it's off,
                    # `_run_midtilt_ga` stays the plain Numba search.
                    if bool(ss.get("ga_precluster")):
                        from routing_optimiser.precluster import run_midtilt_ga_preclustered as _run_midtilt_ga
                        log("   pre-clustering ON → GA runs on rule-safe pre-clustered cells "
                            "(near-lossless; expand-back before enforcement).")
                    log(f"   optimiser build: {getattr(_optmod, '__build__', 'UNKNOWN — stale bytecode?')} "
                        "(expect 2026-07-16-vamp-frontier-lp — if not, clear __pycache__).")
                    # Softmax and Thompson are per-cell engines: the reference IS their
                    # slider=100 split, and the shared risk layer below (reference→compliant
                    # blend + hard-enforce) does the rest. For the genetic engine the
                    # reference is only cell STRUCTURE (gateways, rates, baseline) — the
                    # global GA overwrites the shares — so it falls back to the fast softmax.
                    # Softmax and Thompson: their slider-100 reference IS revenue/conversion-
                    # optimal, so use it directly. Portfolio's own reference prices CVaR at every
                    # dial (never revenue-optimal), which starves dial 100 — so it takes the
                    # softmax revenue reference here and gets its CVaR split as the dial-0 endpoint
                    # in a dedicated branch below.
                    # Each per-cell engine builds its OWN slider-100 reference so the engines
                    # diverge and can be compared: softmax = exp(success/temp), Thompson = bandit
                    # probability-of-best, portfolio = mean-CVaR. Genetic uses softmax only for cell
                    # STRUCTURE (its global GA overwrites the shares in its own branch).
                    # Each engine builds its OWN slider-100 reference — no borrowing. Softmax/
                    # Thompson/Portfolio are per-cell engines whose reference IS their split;
                    # genetic uses its OWN revenue-greedy waterfall reference (genetic_ref), NOT
                    # softmax, so it's genuinely standalone.
                    _ref_engine = engine_key if engine_key in ("softmax", "thompson", "portfolio") else "genetic_ref"
                    _ref_params = dict(params) if engine_key in ("softmax", "thompson", "portfolio") else {}
                    ref_settings = OptimiserSettings(
                        risk_conversion_weight=1.0, engine=_ref_engine,
                        engine_params=_ref_params,
                        hard=HardConstraints(max_gateway_share=max_share, vamp_cap=None),
                        soft=SoftConstraints(exploration_floor=floor))
                    ref_agg = optimise_split(agg_problems, ref_settings).reset_index(drop=True)

                    # Per-(Currency, parent-bank, vampMid) VAMP rate. Prefer the pro-rata
                    # export (full lifecycle, matches Post_Mx); fall back to the period-0
                    # gateway risk rate already on the split.
                    fid2vamp = {}
                    _mmp = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    if os.path.exists(_mmp):
                        _mmd = pd.read_csv(_mmp)
                        _cc = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmd.columns}
                        if _cc.get("gatewayfid") and _cc.get("vampmid"):
                            fid2vamp = dict(zip(_mmd[_cc["gatewayfid"]].astype(str).str.strip().str.lower(),
                                                _mmd[_cc["vampmid"]].astype(str).str.strip()))
                    mid_rate = {}
                    _ppf = os.path.join(out_dir, "vamp_t_period_prorata_export.csv")
                    if os.path.exists(_ppf):
                        try:
                            _pp = pd.read_csv(_ppf, usecols=["vampMid", "BIN", "Currency", "vampCount", "VI_Txn_Count"])
                            _pp["Currency"] = _pp["Currency"].astype(str).str.strip().str.lower()
                            _pp["parent"] = _pp["BIN"].astype(str).map(
                                lambda b: bin_to_bank.get(b, bin_to_bank.get(str(b).strip().lower(), b))).astype(str).str.strip().str.lower()
                            _pp["vampMid"] = _pp["vampMid"].astype(str).str.strip()
                            _g = _pp.groupby(["Currency", "parent", "vampMid"]).agg(vc=("vampCount", "sum"), vt=("VI_Txn_Count", "sum"))
                            _g["rate"] = _g["vc"] / _g["vt"].replace(0, np.nan)
                            mid_rate = _g["rate"].dropna().to_dict()
                            log(f"   MID VAMP rates from pro-rata export ({len(mid_rate):,} MID×cell rates).")
                        except Exception as e:
                            log(f"   [Warning] pro-rata rate load failed ({e}); using period-0 rates.")
                    else:
                        log("   pro-rata export not found — using period-0 risk rates for MID caps.")

                    # Build the cross-cell input and enforce per-vampMid caps. The cell key
                    # includes RPGT so that in per-RPGT mode traffic is moved within each
                    # (currency, bank, RPGT) cell; in Bank×Currency mode rpgt is a constant
                    # ("ALL_RPGTS"), so the key collapses to currency|bank (unchanged).
                    _mc = ref_agg.copy()
                    _rp_key = (_mc["rpgt"].astype(str).str.strip().str.lower()
                               if "rpgt" in _mc.columns else "all_rpgts")
                    _mc["cell"] = (_mc["currency"].astype(str).str.lower() + "|"
                                   + _mc["bank"].astype(str).str.lower() + "|" + _rp_key)
                    _mc["vampMid"] = _mc["gateway"].astype(str).str.strip().str.lower().map(fid2vamp).fillna(_mc["gateway"].astype(str))
                    _ck = _mc["currency"].astype(str).str.strip().str.lower()
                    _pk = _mc["bank"].astype(str).str.strip().str.lower()
                    _mc["rate"] = [mid_rate.get((c, b, v), np.nan) for c, b, v in zip(_ck, _pk, _mc["vampMid"])]
                    _mc["rate"] = pd.to_numeric(_mc["rate"], errors="coerce").fillna(_mc["gateway_risk_rate"])
                    _mc["cell_vol"] = _mc["cell_volume"]

                    if vamp_cap is not None:
                        _inp = _mc[["cell", "gateway", "vampMid", "cell_vol", "rate", "share"]].copy()
                        compliant, retired, still_over = enforce_mid_vamp_caps(
                            _inp, cap=float(vamp_cap), floor=float(floor), max_share=float(max_share))
                        comp_share = compliant["share"].to_numpy()
                    else:
                        comp_share = ref_agg["share"].to_numpy()
                        retired, still_over = set(), set()

                    # ---- Per-MID target ± tolerance caps (hard) --------------------
                    # Rules are (vampMid, RPGT, month, metric, target, tol):
                    #   * Aggregate (RPGT=All & month=All) -> MID-TOTAL scale, mid_level base.
                    #   * Month-only (RPGT=All, month set) -> MID-TOTAL scale, that month's
                    #     pro-rata base.
                    #   * RPGT-scoped (RPGT set) -> per-(MID, RPGT) scale on the exploded
                    #     split (base Bank×Currency split adjusted afterwards, like the
                    #     other risk constraints).
                    # Any RPGT/month-scoped rule needs the pro-rata export; we NEVER fall
                    # back to the mid_level aggregate for these — missing => hard error.
                    # For each rule the allowed volume ratio a_max = target×(1+tol)/baseline
                    # (Txn/VAMP); a VAMP % cap below the scope's baseline rate retires it
                    # (a_max = 0). Over-cap MIDs are scaled to a_max × baseline.
                    _all_rules = params.get("mid_constraints", []) or []
                    _scoped_rules = [r for r in _all_rules
                                     if r.get("rpgt") is not None or r.get("month") is not None]
                    _pp_tidy = None
                    if _scoped_rules:
                        if not os.path.exists(_ppf):
                            raise RuntimeError(
                                "Per-MID constraints scoped by RPGT/month need the pro-rata "
                                "export 'vamp_t_period_prorata_export.csv', which was not found "
                                f"in {out_dir}. Set a Split Go Live date on the Forecast tab and "
                                "re-run the forecast to generate it — no mid_level fallback is "
                                "used for scoped constraints. [build 2026-07-08-mid-rpgt-caps]")
                        _pps = pd.read_csv(_ppf, usecols=["vampMid", "RPGT", "period", "vampCount", "VI_Txn_Count"])
                        _pps["_mid"] = _pps["vampMid"].astype(str).str.strip().str.lower()
                        _pps["_rpgt"] = _pps["RPGT"].astype(str).str.strip().str.lower()
                        _pps["_per"] = pd.to_numeric(_pps["period"], errors="coerce").fillna(-1).astype(int)
                        _pp_tidy = _pps.groupby(["_mid", "_rpgt", "_per"], as_index=False).agg(
                            txn=("VI_Txn_Count", "sum"), vamp=("vampCount", "sum"))

                    # [FN-314]
                    def _scope_base(mid, rpgt, month):
                        d = _pp_tidy[_pp_tidy["_mid"] == str(mid).strip().lower()]
                        if rpgt is not None:
                            d = d[d["_rpgt"] == str(rpgt).strip().lower()]
                        if month is not None:
                            d = d[d["_per"] == int(month)]
                        else:
                            d = d[(d["_per"] >= 0) & (d["_per"] <= 3)]
                        return float(d["txn"].sum()), float(d["vamp"].sum())

                    # [FN-315]
                    def _rule_a(_tg, _tl, _mtr, _bt, _bv):
                        if _mtr == "txn" and _bt > 0:
                            return _tg * (1.0 + (float(_tl) if _tl is not None else 0.0)) / _bt
                        if _mtr == "vamp" and _bv > 0:
                            return _tg * (1.0 + (float(_tl) if _tl is not None else 0.0)) / _bv
                        if _mtr == "vamp_pct" and _bt > 0 and (_bv / _bt) > (_tg / 100.0) + 1e-9:
                            return 0.0
                        return np.inf

                    _bt_map = params.get("mid_base_totals", {}) or {}
                    _agg_a, a_max_by_key = {}, {}
                    for _rec in _all_rules:
                        _mk = str(_rec.get("vampMid", "")).strip().lower()
                        _tg = float(_rec.get("target") or 0.0)
                        _tl = _rec.get("tol")
                        _mtr = _rec.get("metric", "txn")
                        if str(_rec.get("direction", "range")) == "floor":
                            continue   # floor-type has NO ceiling → no routing-space a_max cap
                        _rp, _mo = _rec.get("rpgt"), _rec.get("month")
                        if _rp is None and _mo is None:                       # aggregate (mid_level base)
                            _bt, _bv = _bt_map.get(_mk, (0.0, 0.0))
                            _agg_a[_mk] = min(_agg_a.get(_mk, np.inf), _rule_a(_tg, _tl, _mtr, _bt, _bv))
                        elif _rp is None:                                     # month-only -> MID-total
                            _bt, _bv = _scope_base(_mk, None, _mo)
                            _agg_a[_mk] = min(_agg_a.get(_mk, np.inf), _rule_a(_tg, _tl, _mtr, _bt, _bv))
                        else:                                                 # RPGT-scoped -> per (mid, rpgt)
                            _bt, _bv = _scope_base(_mk, _rp, _mo)
                            _kk = (_mk, str(_rp).strip().lower())
                            a_max_by_key[_kk] = min(a_max_by_key.get(_kk, np.inf), _rule_a(_tg, _tl, _mtr, _bt, _bv))

                    a_max_by_mid, mid_vol_constrained = {}, set()
                    for _mk, _a in _agg_a.items():
                        if _a < np.inf:
                            a_max_by_mid[_mk] = max(_a, 0.0)
                    a_max_by_key = {k: max(v, 0.0) for k, v in a_max_by_key.items() if v < np.inf}

                    # ---- Projection-feedback inputs for per-MID month/aggregate caps ----
                    # These caps are on the PROJECTED VAMP/Txn (what tab 4 shows), whose
                    # baseline is the forecast pro-rata split — NOT the routing 30D split.
                    # Routing-space scaling misses because the two baselines differ, so we
                    # enforce by RE-PROJECTING each candidate split and scaling the MID until
                    # its projected value meets the cap.
                    _mid_month_rules = []   # (mid_lower, month|None, metric, target, tol, direction)
                    # PRIORITY lookups (1 = highest). _prio_lookup keyed per constraint; _prio_by_mid
                    # keeps the highest importance (lowest number) per MID for the greedy weighting.
                    # PRIORITY tiering with a per-tier gap: p1=1.0, p2=1/GAP, p3=1/GAP², …
                    # (higher number = lower priority). GAP=8 (restored — the 14:35-run value): the OLD
                    # gap of 5000 made prio-2 weight 0.0002, so a prio-2 VAMP-ceiling breach contributed
                    # ≈0 to the GA's ranked violation and the search treated those ceilings as decorative
                    # (it would push a MID far over a prio-2 cap and still call the split "feasible").
                    # GAP=8 keeps a clear hierarchy (prio-1 = 8× prio-2 = 64× prio-3) while giving prio-2
                    # a MATERIAL weight (0.125), so the search itself drives prio-2 ceilings down. VAMP-cap
                    # compliance is UNAFFECTED (it ranks far above every band via _VAMP_W). The GA "hang"
                    # this was briefly blamed for turned out to be the silent progress poller (FUSE),
                    # now fixed — so this is safe to keep. Retune here: bigger = more decorative low tiers.
                    _PRIORITY_GAP = 8.0
                    _prio_lookup, _prio_by_mid = {}, {}
                    for _rec in _all_rules:
                        _pmk = str(_rec.get("vampMid", "")).strip().lower()
                        _pp = int(_rec.get("priority", 1) or 1)
                        _prio_lookup[(_pmk, _rec.get("month"), _rec.get("metric", "txn"))] = _pp
                        _prio_by_mid[_pmk] = min(_prio_by_mid.get(_pmk, 99), _pp)
                    # [FN-316]
                    def _prio_mult(_p):
                        # p1 → 1, p2 → 1/GAP, p3 → 1/GAP², … (higher number = lower priority).
                        return float(_PRIORITY_GAP ** (1 - max(int(_p), 1)))
                    try:
                        _prios_used = sorted({int(_v) for _v in _prio_lookup.values()})
                        log("   priority→band weight (GAP={:.0f}): ".format(_PRIORITY_GAP)
                            + " · ".join("prio{}={:.4g}".format(_p, _prio_mult(_p)) for _p in _prios_used)
                            + "  (higher weight ⇒ the GA works harder to satisfy that tier)")
                    except Exception:  # noqa: BLE001 - logging must never break a run
                        pass
                    for _rec in _all_rules:
                        if _rec.get("rpgt") is None:                 # aggregate + month-only
                            _mid_month_rules.append((
                                str(_rec.get("vampMid", "")).strip().lower(),
                                _rec.get("month"), _rec.get("metric", "txn"),
                                float(_rec.get("target") or 0.0), _rec.get("tol"),
                                str(_rec.get("direction", "range"))))
                    _pp_full = pd.read_csv(_ppf) if (_mid_month_rules and os.path.exists(_ppf)) else None
                    # vampMids fully switched off in overrides — excluded from the projection,
                    # matching the tab-4 VAMP impact table. (Defined before the scaffold below,
                    # which references it.)
                    from routing_optimiser.forecast_pipeline import _canonical_gateway as _canon_gw
                    _ovr2 = ss.get("gateway_volume_overrides") or {}
                    _off2 = set()
                    for _gwid, _cfg in (_ovr2.items() if isinstance(_ovr2, dict) else []):
                        if isinstance(_cfg, dict) and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                           and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
                            _off2.add(str(_canon_gw(_gwid)).strip().lower())
                    _v2f = {}
                    for _f, _v in fid2vamp.items():
                        _v2f.setdefault(str(_v).strip(), set()).add(str(_canon_gw(_f)).strip().lower())
                    _excluded_mids = frozenset(v for v, fids in _v2f.items() if fids and fids <= _off2)
                    # Precompute the STATIC projection scaffold ONCE (restricted to the cells
                    # containing a capped MID — a capped MID's projected VAMP depends only on
                    # its own cells, so this is EXACT). Each feedback iteration then only
                    # recomputes the prop-dependent parts on this small frame, instead of
                    # re-projecting millions of pro-rata rows.
                    _capped_l = {_row[0] for _row in _mid_month_rules}
                    _T0 = _Pc = None
                    # Precomputed static structures for _project_capped (filled below when the
                    # scaffold is built) so it never re-hashes string keys per call.
                    _T0_pk = _T0_pk_rpgt = _T0_gcodes = _T0_excl_a = _T0_base_a = _T0_ctot_a = None
                    _T0_prr_a = _T0_vi_a = _T0_capidx = _Pc_to_t0 = _Pc_vc_a = _T0_fcp_a = None
                    _Pc_movedvpool_a = _T0_vc_a = _T0_emask_a = None
                    _pc_aggcodes = _pc_agg_labels = _t0cap_aggcodes = _t0cap_agg_labels = None
                    _T0_pk_codes = _T0_pk_uniq_ix = _T0_pkr_codes = _T0_pkr_uniq_ix = None
                    _n_gc = _n_pc_agg = _n_t0cap_agg = 0
                    _grpk = ["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per"]
                    if _pp_full is not None and _capped_l:
                        _P = _pp_full.copy()
                        _rpc = "RPGT" if "RPGT" in _P.columns else "rpgt"
                        _P["_cur"] = _P["Currency"].astype(str).str.strip().str.lower()
                        _P["_bin"] = _P["BIN"].astype(str).str.strip()
                        _P["_rpgt"] = _P[_rpc].astype(str)
                        _P["_mid"] = _P["vampMid"].astype(str).str.strip()
                        _P["_midl"] = _P["_mid"].str.lower()
                        _P["_per"] = pd.to_numeric(_P["period"], errors="coerce").fillna(-1).astype(int)
                        _P["_t"] = pd.to_numeric(_P["t"], errors="coerce").fillna(0).astype(int)
                        _P["_vi"] = pd.to_numeric(_P["VI_Txn_Count"], errors="coerce").fillna(0.0)
                        _P["_vc"] = pd.to_numeric(_P["vampCount"], errors="coerce").fillna(0.0)
                        _P["_pr"] = pd.to_numeric(_P.get("pro_rata", 0.0), errors="coerce").fillna(0.0)
                        # fcp1_frac: cohort the pipeline actually reroutes (missing -> 1.0).
                        _P["_fcp"] = pd.to_numeric(_P.get("fcp1_frac", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
                        # Keep pmp / Country sub-cells (default '_all_') so the enforcement can
                        # apply the pipeline's wallet-incapable / USA-only masks per sub-cell.
                        _P["_pmp"] = (_P["paymentMethodProvider"].astype(str).str.strip().str.lower()
                                      if "paymentMethodProvider" in _P.columns else "_all_")
                        _P["_ctry"] = (_P["Country"].astype(str).str.strip().str.lower()
                                       if "Country" in _P.columns else "_all_")
                        _P = _P.groupby(["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_mid", "_midl",
                                         "_per", "_t"], as_index=False).agg(
                            _vi=("_vi", "sum"), _vc=("_vc", "sum"), _pr=("_pr", "first"), _fcp=("_fcp", "first"))
                        _cellk = _P["_cur"] + "|" + _P["_bin"] + "|" + _P["_rpgt"]
                        _keep = set(_cellk[_P["_midl"].isin(_capped_l)].unique())
                        _P = _P[_cellk.isin(_keep)].copy()
                        _T0 = _P[_P["_t"] == 0].copy()
                        _T0["_bf"] = 0   # 0 = real baseline row, 1 = injected back-fill row
                        # ---- BACK-FILL sub-cell rows (mirror the tab-3 projection fix) --------
                        # A MID present in a cell but absent from one of its pmp/Country sub-cells
                        # gets no routed volume there, so that sub-cell's proposed shares
                        # renormalise onto the MIDs that ARE present — overstating their projected
                        # txn (the WoodForest-in-non-usa / routed-in-usa case). Give every MID a
                        # ZERO-baseline t0 row in every sub-cell of any cell it already appears in.
                        # The proposed share is broadcast by the coarse cur|bin|rpgt key, so this
                        # is SPLIT-INDEPENDENT → computed once here, no per-call cost in the loop.
                        # Only sibling sub-cells of an existing cell are targeted (never invents a
                        # sub-cell), and injected rows carry _vi=_vc=0 so they receive volume but
                        # hold none and add no VAMP — matching _inject_backfill_rows in tab 3.
                        if len(_T0):
                            _ck = _T0["_cur"] + "|" + _T0["_bin"] + "|" + _T0["_rpgt"]
                            _mids_in_cell = (_T0.assign(_ck=_ck)[["_ck", "_mid", "_midl"]]
                                             .drop_duplicates())
                            _subper = (_T0.assign(_ck=_ck)
                                       .drop_duplicates(["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per"])
                                       [["_ck", "_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per", "_pr", "_fcp"]])
                            _grid = _subper.merge(_mids_in_cell, on="_ck")
                            # Vectorised anti-join (replaces a Python membership loop over the grid):
                            # keep grid (sub-cell × MID) rows with NO existing t0 row — bit-identical
                            # to the `[k not in _have ...]` filter it replaces (a set-membership test,
                            # no arithmetic).
                            _bkey = ["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per", "_midl"]
                            _newbf = _grid.merge(_T0[_bkey].drop_duplicates(), on=_bkey,
                                                 how="left", indicator=True)
                            _newbf = _newbf[_newbf["_merge"] == "left_only"].drop(columns="_merge").copy()
                            if len(_newbf):
                                _newbf["_t"] = 0
                                _newbf["_vi"] = 0.0
                                _newbf["_vc"] = 0.0
                                _newbf["_bf"] = 1
                                _newbf = _newbf[["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_mid",
                                                 "_midl", "_per", "_t", "_vi", "_vc", "_pr", "_fcp", "_bf"]]
                                _T0 = pd.concat([_T0, _newbf], ignore_index=True, sort=False)
                                log(f"   back-fill sub-cell rows injected into cap scaffold: {len(_newbf):,}")
                        _T0["_excl"] = _T0["_mid"].isin(_excluded_mids)
                        _T0["_ctot"] = _T0.groupby(_grpk)["_vi"].transform("sum")
                        _T0["_av"] = np.where(_T0["_excl"], 0.0, _T0["_vi"])
                        _T0["_at"] = _T0.groupby(_grpk)["_av"].transform("sum")
                        _T0["_base"] = np.where(_T0["_at"] > 0, _T0["_av"] / _T0["_at"], 0.0)
                        # Static pipeline-enforcement mask per t0 row: wallet-incapable MID in a
                        # wallet-pmp sub-cell, or USA-only MID in a Non-USA sub-cell. Zeroes that
                        # MID's proposed share there (matches build_split_exports).
                        _wc_es = ss.get("wallet_ctx") or {}
                        _wc_set = {str(x).strip().lower() for x in (_wc_es.get("incapable") or set())}
                        _uo_set = {str(x).strip().lower() for x in (_wc_es.get("usa_only") or set())}
                        _T0_emask_a = (
                            (_T0["_pmp"].isin(["googlepay", "applepay"]).to_numpy()
                             & _T0["_midl"].isin(_wc_set).to_numpy())
                            | ((~_T0["_ctry"].isin(["usa", "us", "_all_", ""])).to_numpy()
                               & _T0["_midl"].isin(_uo_set).to_numpy()))
                        if not (_wc_set or _uo_set):
                            _T0_emask_a = None
                        _Pc = _P[_P["_midl"].isin(_capped_l)].copy()
                        _Pc["_om"] = _Pc["_per"] - _Pc["_t"]
                        log(f"   per-MID cap projection scaffold: {len(_T0):,} t0 rows, "
                            f"{len(_Pc):,} capped-MID rows ({len(_keep):,} cells).")

                        # ---- Precompute the STATIC structure once, so _project_capped never
                        # re-hashes string keys on the ~50 calls/pass the LP finite-diff + greedy
                        # make. All per-call ops below become an array map + integer-code group-bys
                        # + an index gather → bit-identical to the merge version (same pandas
                        # summation), minus the per-call string hashing that dominated the cost.
                        _T0_pk = (_T0["_cur"] + "|" + _T0["_bin"] + "|" + _T0["_mid"]).to_numpy()
                        # RPGT-keyed variant, used when the split is per-RPGT (Bank×Currency×RPGT
                        # grain) so a per-RPGT proposed share projects onto the matching RPGT rows.
                        _T0_pk_rpgt = (_T0["_cur"] + "|" + _T0["_bin"] + "|"
                                       + _T0["_rpgt"].astype(str).str.strip().str.lower() + "|" + _T0["_mid"]).to_numpy()
                        # Speedup 2: factorize the scaffold keys ONCE. _project_capped then aligns the
                        # per-call proposed shares with a C-level get_indexer + numpy gather (keep-last
                        # via last-write-wins fancy assignment), instead of building a pandas Series +
                        # de-duplicating + reindexing over ~600k rows on every call. Exact.
                        _T0_pk_codes, _u = pd.factorize(_T0_pk)
                        _T0_pk_uniq_ix = pd.Index(_u)
                        _T0_pkr_codes, _ur = pd.factorize(_T0_pk_rpgt)
                        _T0_pkr_uniq_ix = pd.Index(_ur)
                        _T0_gcodes = pd.factorize(
                            _T0["_cur"] + "|" + _T0["_bin"] + "|" + _T0["_rpgt"] + "|"
                            + _T0["_pmp"] + "|" + _T0["_ctry"] + "|" + _T0["_per"].astype(str))[0]
                        _T0_excl_a = _T0["_excl"].to_numpy(bool)
                        _T0_base_a = _T0["_base"].to_numpy(float)
                        _T0_ctot_a = _T0["_ctot"].to_numpy(float)
                        _T0_prr_a = _T0["_pr"].to_numpy(float)
                        _T0_fcp_a = _T0["_fcp"].to_numpy(float)   # movable-cohort fraction (fcp1)
                        _T0_vc_a = _T0["_vc"].to_numpy(float)     # baseline VAMP at t0 (for VAMP share)
                        _T0_vi_a = _T0["_vi"].to_numpy(float)
                        _T0_capidx = np.where(_T0["_midl"].isin(_capped_l).to_numpy())[0]
                        # _Pc → _T0 row index by (cur,bin,rpgt,midl, _Pc._om == _T0._per)
                        _t0_join = (_T0["_cur"] + "|" + _T0["_bin"] + "|" + _T0["_rpgt"] + "|"
                                    + _T0["_pmp"] + "|" + _T0["_ctry"] + "|"
                                    + _T0["_midl"] + "|" + _T0["_per"].astype(str)).to_numpy()
                        # Exclude injected back-fill rows as VAMP join targets: they carry zero
                        # baseline VAMP, so mapping a _Pc row onto one would move VAMP out of the
                        # cohort without redistributing any back. Keeping them out leaves the VAMP
                        # projection (and every VAMP-band decision) byte-identical to pre-back-fill;
                        # the injection only corrects the TXN share normalisation.
                        _bf_mask = (_T0["_bf"].to_numpy() > 0) if "_bf" in _T0.columns else np.zeros(len(_T0), bool)
                        _pc_join = (_Pc["_cur"] + "|" + _Pc["_bin"] + "|" + _Pc["_rpgt"] + "|"
                                    + _Pc["_pmp"] + "|" + _Pc["_ctry"] + "|"
                                    + _Pc["_midl"] + "|" + _Pc["_om"].astype(str)).to_numpy()
                        # Vectorised _Pc -> _T0 row-index map (replaces a per-row dict build + fromiter
                        # over ~1.3M rows). A Series indexed by the non-back-fill t0 keys, reindexed to
                        # the _Pc keys, gives each _Pc row its t0 position or -1 — identical to the
                        # {k:i ...}.get(k,-1) it replaces (keep-last on any duplicate key, no arithmetic).
                        _valid = ~_bf_mask
                        _t0_pos = pd.Series(np.where(_valid)[0], index=_t0_join[_valid])
                        _t0_pos = _t0_pos[~_t0_pos.index.duplicated(keep="last")]
                        _Pc_to_t0 = _t0_pos.reindex(_pc_join).fillna(-1).to_numpy().astype(np.int64)
                        _Pc_vc_a = _Pc["_vc"].to_numpy(float)
                        # Moved-VAMP pool per (cur,bin,rpgt,period,t) = Σ over ALL MIDs of
                        # vampCount × pro_rata × fcp1_frac (all static in the export), for the
                        # two-cohort VAMP projection. _P holds every MID in the kept cells, so
                        # the sum is complete; precomputed once (split-independent).
                        _P["_mvraw"] = _P["_vc"] * _P["_pr"] * _P["_fcp"]
                        _mvp_map = _P.groupby(["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per", "_t"],
                                              observed=True)["_mvraw"].sum().to_dict()
                        _Pc_movedvpool_a = np.fromiter(
                            (_mvp_map.get((_c, _b, _r, _pm, _ct, _p, _t), 0.0)
                             for _c, _b, _r, _pm, _ct, _p, _t in
                             zip(_Pc["_cur"], _Pc["_bin"], _Pc["_rpgt"], _Pc["_pmp"], _Pc["_ctry"],
                                 _Pc["_per"], _Pc["_t"])),
                            dtype=float, count=len(_Pc))
                        # Aggregation group codes + (midl, period) labels — VAMP over _Pc rows,
                        # TXN over capped _T0 rows. Same groups as the old (_midl,_per) group-by.
                        _SEP = ""
                        _pc_aggcodes, _pc_agguniq = pd.factorize(
                            _Pc["_midl"].astype(str) + _SEP + _Pc["_per"].astype(str))
                        _pc_agg_labels = [(_s.rsplit(_SEP, 1)[0], int(_s.rsplit(_SEP, 1)[1])) for _s in _pc_agguniq]
                        _t0cap_key = (_T0["_midl"].astype(str) + _SEP + _T0["_per"].astype(str)).to_numpy()[_T0_capidx]
                        _t0cap_aggcodes, _t0cap_agguniq = pd.factorize(_t0cap_key)
                        _t0cap_agg_labels = [(_s.rsplit(_SEP, 1)[0], int(_s.rsplit(_SEP, 1)[1])) for _s in _t0cap_agguniq]
                        # Group counts for np.bincount (speedup 1): factorize codes are contiguous
                        # 0..n-1, so minlength = max+1 = #unique. Precomputed once with the codes.
                        _n_gc = (int(_T0_gcodes.max()) + 1) if len(_T0_gcodes) else 0
                        _n_pc_agg = len(_pc_agg_labels)
                        _n_t0cap_agg = len(_t0cap_agg_labels)

                    _pc_cache = {}   # memoise _project_capped on identical prop_items (per run)

                    # [FN-317]
                    def _project_capped(prop_items, _use_cache=True):
                        # {(mid_lower, period): (vamp_post, txn_post)} for capped MIDs. Uses the
                        # precomputed static arrays/codes — array map + integer-code group-bys +
                        # index gather. Bit-identical to the original merge-based projection.
                        # _use_cache=False bypasses _pc_cache entirely (no reads/writes) so the LP
                        # Jacobian can call this from worker threads without racing the shared dict
                        # (speedup 4). All other state read here is static/read-only, so it's safe.
                        _pi = list(prop_items)
                        _by_rpgt = bool(_pi) and len(_pi[0]) == 5
                        # MEMOISE: band calibration + the true-breach re-projection re-project the
                        # SAME split repeatedly. Key on the split content (static arrays are fixed for
                        # the run, so prop_items is a complete key). Return a COPY so a caller can
                        # never mutate the cached dict.
                        # Cache key (speedup 6): prop rows are already tuples, so a shallow tuple()
                        # is an exact, hashable key at O(n) — avoids rebuilding ~18k inner tuples
                        # (tuple(map(tuple,…))) every call. Fall back to the deep build only if a row
                        # isn't already a tuple.
                        _ckey = None
                        if _use_cache:
                            _ckey = tuple(_pi) if (not _pi or isinstance(_pi[0], tuple)) else tuple(map(tuple, _pi))
                            _cached = _pc_cache.get(_ckey)
                            if _cached is not None:
                                return {k: list(v) for k, v in _cached.items()}
                        # Vectorised key->value map (replaces a ~600k-iteration Python dict.get loop
                        # plus the f-string dict build). Keys are built with the SAME strip/lower rule
                        # as _T0_pk / _T0_pk_rpgt; keep-last on duplicate keys (matches the dict);
                        # missing keys -> 0.0. This is a value COPY (no summation) so it is
                        # bit-identical to the loop it replaces.
                        if _by_rpgt:
                            _pdf = pd.DataFrame(_pi, columns=["_c", "_b", "_rp", "_m", "_v"])
                            _pkey = (_pdf["_c"].astype(str).str.strip().str.lower() + "|"
                                     + _pdf["_b"].astype(str).str.strip() + "|"
                                     + _pdf["_rp"].astype(str).str.strip().str.lower() + "|"
                                     + _pdf["_m"].astype(str).str.strip()).to_numpy()
                            _keys_codes, _uniq_ix = _T0_pkr_codes, _T0_pkr_uniq_ix
                        else:
                            _pdf = pd.DataFrame(_pi, columns=["_c", "_b", "_m", "_v"])
                            _pkey = (_pdf["_c"].astype(str).str.strip().str.lower() + "|"
                                     + _pdf["_b"].astype(str).str.strip() + "|"
                                     + _pdf["_m"].astype(str).str.strip()).to_numpy()
                            _keys_codes, _uniq_ix = _T0_pk_codes, _T0_pk_uniq_ix
                        # Speedup 2: align proposed shares onto the scaffold via the precomputed
                        # key factorization. get_indexer maps each prop key to its unique-key code
                        # (-1 if absent → dropped, matching reindex); last-write-wins fancy assignment
                        # reproduces de-dup keep="last"; NaN prop values → 0 (matches the old fillna).
                        # Then gather per scaffold row. Bit-for-bit equivalent to the Series reindex.
                        if len(_pi):
                            _vals = pd.to_numeric(_pdf["_v"], errors="coerce").to_numpy(dtype=float)
                            _vals[np.isnan(_vals)] = 0.0
                            _pcode = _uniq_ix.get_indexer(_pkey)
                            _valbycode = np.zeros(len(_uniq_ix), dtype=float)
                            _pres = _pcode >= 0
                            _valbycode[_pcode[_pres]] = _vals[_pres]
                            prop_raw = _valbycode[_keys_codes]
                        else:
                            prop_raw = np.zeros(len(_keys_codes), dtype=float)
                        prop_raw = np.where(_T0_excl_a, 0.0, prop_raw)
                        if _T0_emask_a is not None:      # wallet-incapable / USA-only enforcement
                            prop_raw = np.where(_T0_emask_a, 0.0, prop_raw)
                        # per-cell (_grpk) proposed-share sum. np.bincount+gather is the same
                        # group-sum-then-broadcast as groupby.transform("sum") at C speed, no pandas
                        # object overhead (speedup 1). Numerically identical (float accumulation order
                        # differs by ~1e-12, far below any band/VAMP threshold).
                        _psum = np.bincount(_T0_gcodes, weights=prop_raw, minlength=_n_gc)[_T0_gcodes]
                        # np.divide(where=…) divides ONLY where the denominator is non-zero and
                        # leaves the pre-filled fallback elsewhere — same result as np.where but
                        # without evaluating 0/0 (which triggers the invalid-value RuntimeWarning).
                        _pshare = np.array(_T0_base_a, dtype=float)          # fallback = baseline share
                        np.divide(prop_raw, _psum, out=_pshare, where=_psum > 0)
                        # PER-MID movable fraction = go-live pro-rata × fcp1 (per-vampMid). No
                        # movement where no rule applies. TWO-COHORT volume: each MID holds
                        # (1-move) of its OWN volume; the pooled movable slice (Σ base×move) is
                        # redistributed by the proposed share — matches the tab-3 projection.
                        _mv = np.where(_psum > 0, _T0_prr_a * _T0_fcp_a, 0.0)
                        _bm = _T0_base_a * _mv
                        _moved_tot = np.bincount(_T0_gcodes, weights=_bm, minlength=_n_gc)[_T0_gcodes]
                        _ptxn = _T0_ctot_a * (_T0_base_a * (1.0 - _mv) + _moved_tot * _pshare)
                        _ptxn = np.where(_T0_excl_a, 0.0, _ptxn)
                        # TWO-COHORT VAMP (pipeline-faithful): hold (1-move) of each capped MID's
                        # VAMP; the pooled moved VAMP is redistributed ONLY across VAMP-carrying
                        # MIDs (zero-VAMP MIDs stay 0), conserving the cell VAMP total.
                        _vprop = prop_raw * (_T0_vc_a > 0)
                        _vpsum = np.bincount(_T0_gcodes, weights=_vprop, minlength=_n_gc)[_T0_gcodes]
                        _vshare = np.zeros_like(_vprop, dtype=float)
                        np.divide(_vprop, _vpsum, out=_vshare, where=_vpsum > 0)
                        _gi = np.where(_Pc_to_t0 >= 0, _Pc_to_t0, 0)
                        _move_pc = np.where(_Pc_to_t0 >= 0, _mv[_gi], 0.0)
                        _psh_pc = np.where(_Pc_to_t0 >= 0, _vshare[_gi], 0.0)
                        _vp = _Pc_vc_a * (1.0 - _move_pc) + _Pc_movedvpool_a * _psh_pc
                        _out = {}
                        # VAMP over _Pc rows, TXN over capped _T0 rows: bincount aggregation (speedup
                        # 1). factorize codes cover every group, so iterating range(n) reproduces the
                        # SAME key set as groupby.sum() (including exact-zero groups).
                        _vsum = np.bincount(_pc_aggcodes, weights=_vp, minlength=_n_pc_agg)
                        for _code in range(_n_pc_agg):
                            _mk, _p = _pc_agg_labels[_code]
                            _out[(_mk, _p)] = [float(_vsum[_code]), 0.0]
                        _tsum = np.bincount(_t0cap_aggcodes, weights=_ptxn[_T0_capidx], minlength=_n_t0cap_agg)
                        for _code in range(_n_t0cap_agg):
                            _mk, _p = _t0cap_agg_labels[_code]
                            _out.setdefault((_mk, _p), [0.0, 0.0])[1] = float(_tsum[_code])
                        if _use_cache:
                            if len(_pc_cache) >= 64:       # bounded LRU-ish cache (evict oldest)
                                _pc_cache.pop(next(iter(_pc_cache)))
                            _pc_cache[_ckey] = _out
                        return {k: list(v) for k, v in _out.items()}
                    # Pretty "mid (RPGT)" labels + a running set of the groups actually
                    # scaled/retired, surfaced under the tab-4 tiles.
                    _rpgt_disp = {}
                    for _rec in _all_rules:
                        if _rec.get("rpgt") is not None:
                            _rpgt_disp[f"{str(_rec.get('vampMid','')).strip().lower()}||{str(_rec.get('rpgt')).strip().lower()}"] = \
                                f"{str(_rec.get('vampMid')).strip()} ({str(_rec.get('rpgt')).strip()})"
                    _rpgt_constrained = set()
                    _mid_gran_constrained = set()
                    if a_max_by_key:
                        log(f"   per-(MID×RPGT) scoped caps active: {len(a_max_by_key)} "
                            "(adjust-after on the exploded per-RPGT split).")
                    if a_max_by_mid:
                        _vc = _mc.copy()
                        _vc["vampMid"] = _vc["vampMid"].astype(str).str.strip().str.lower()
                        _vc["baseline_share"] = ref_agg["baseline_share"].to_numpy()
                        _vc["share"] = comp_share
                        _vc2, mid_vol_constrained = enforce_mid_volume_caps(
                            _vc[["cell", "gateway", "vampMid", "cell_vol", "baseline_share", "share", "rate"]],
                            a_max_by_mid, max_share=float(max_share))
                        comp_share = _vc2["share"].to_numpy()
                        log(f"   per-MID target±tolerance caps: {len(a_max_by_mid)} active; "
                            f"{len(mid_vol_constrained)} MID(s) scaled/retired.")
                    ss["mid_vol_constrained"] = sorted(str(m) for m in mid_vol_constrained)

                    ref_share = ref_agg["share"].to_numpy()
                    changed = not np.allclose(ref_share, comp_share, atol=1e-6)
                    ss["retired_mids"] = sorted(str(m) for m in retired)

                    # Count vampMids whose AGGREGATE rate is over the cap at a given split.
                    _vm = _mc["vampMid"].to_numpy()
                    _rt = pd.to_numeric(_mc["rate"], errors="coerce").fillna(0.0).to_numpy()
                    _cv = pd.to_numeric(_mc["cell_vol"], errors="coerce").fillna(0.0).to_numpy()

                    # [FN-318]
                    def _mids_over(shares):
                        if vamp_cap is None:
                            return 0
                        vol = _cv * np.asarray(shares, float)
                        t = pd.DataFrame({"m": _vm, "vol": vol, "vr": vol * _rt}).groupby("m").sum()
                        gr = t["vr"] / t["vol"].replace(0, np.nan)
                        return int((gr > float(vamp_cap) + 1e-9).sum())

                    # [FN-319]
                    def _summ_from_shares(shares):
                        a = ref_agg.copy()
                        a["share"] = shares
                        a["volume"] = a["cell_volume"] * a["share"]
                        return a, portfolio_summary(a)

                    # --- Eligibility restrictions (RPGT/currency bans + wallet capability) ---
                    # Applied to the exploded (per-RPGT) split: banned gateways are zeroed
                    # and volume redistributed to eligible ones; wallet-incapable gateways
                    # keep only their non-wallet share.
                    from routing_optimiser.eligibility import (
                        load_restrictions, load_usa_only, apply_restrictions, WALLET_VALUES, unenforceable_fields)
                    _rr_path = os.path.join(PROJECT_ROOT, "config", "inputs", "routing_restrictions.json")
                    _elig_rules = load_restrictions(_rr_path)
                    _unenf = unenforceable_fields(_elig_rules, ["rpgt", "currency", "bank"])
                    if _unenf:
                        log(f"   [Warning] restriction field(s) not enforceable at the routing grain "
                            f"(ignored — need finer routing): {', '.join(sorted(_unenf))}.")
                    _fid2vamp_l = {k: str(v).strip().lower() for k, v in fid2vamp.items()}
                    _wallet_incapable, _wallet_frac, _wallet_default = set(), {}, 0.0
                    for _f in process_wallet_incapable(_mmp):   # explicit processWallet=FALSE fids
                        _wallet_incapable.add(_f)
                        if _f in _fid2vamp_l:
                            _wallet_incapable.add(_fid2vamp_l[_f])
                    if _wallet_incapable and "paymentMethodProvider" in orig_adf.columns and "attempts" in orig_adf.columns:
                        _w = orig_adf.copy()
                        # attempts_success.sql maps non-wallet -> 'non_gp_ap' and leaves
                        # wallet (GOOGLEPAY/APPLEPAY) as NULL, so wallet = NOT non_gp_ap.
                        _pmpv = _w["paymentMethodProvider"].astype(str).str.strip().str.lower()
                        _w["_wal"] = ~_pmpv.isin(["non_gp_ap"])
                        _w["_att"] = pd.to_numeric(_w["attempts"], errors="coerce").fillna(0.0)
                        _w["_watt"] = np.where(_w["_wal"], _w["_att"], 0.0)
                        _w["_c"] = _w["currency"].astype(str).str.strip().str.lower()
                        _w["_b"] = _w["bank"].astype(str).str.strip().str.lower()
                        _wg = _w.groupby(["_c", "_b"]).agg(a=("_att", "sum"), wa=("_watt", "sum")).reset_index()
                        _wallet_frac = {(c, b): (float(wa) / float(a) if a > 0 else 0.0)
                                        for c, b, a, wa in zip(_wg["_c"], _wg["_b"], _wg["a"], _wg["wa"])}
                        _tot = float(_w["_att"].sum())
                        _wallet_default = float(_w["_watt"].sum() / _tot) if _tot > 0 else 0.0

                    # Country capability — USA-only gateways (explicit list in the JSON) can
                    # only serve country='USA'. Enforced like wallet: keep only the USA share
                    # of each cell, redistribute the Non-USA portion. Needs a per-(currency,
                    # bank) Non-USA fraction from the attempts data.
                    _usa_only, _nonusa_frac, _nonusa_default = set(), {}, 0.0
                    for _f in load_usa_only(_rr_path):
                        _usa_only.add(_f)
                        if _f in _fid2vamp_l:
                            _usa_only.add(_fid2vamp_l[_f])
                    if _usa_only:
                        if "country" in orig_adf.columns and "attempts" in orig_adf.columns:
                            _cy = orig_adf.copy()
                            _cyv = _cy["country"].astype(str).str.strip().str.upper()
                            _cy["_non"] = ~_cyv.isin(["USA", "US"])   # everything not USA
                            _cy["_att"] = pd.to_numeric(_cy["attempts"], errors="coerce").fillna(0.0)
                            _cy["_natt"] = np.where(_cy["_non"], _cy["_att"], 0.0)
                            _cy["_c"] = _cy["currency"].astype(str).str.strip().str.lower()
                            _cy["_b"] = _cy["bank"].astype(str).str.strip().str.lower()
                            _cg = _cy.groupby(["_c", "_b"]).agg(a=("_att", "sum"), na=("_natt", "sum")).reset_index()
                            _nonusa_frac = {(c, b): (float(na) / float(a) if a > 0 else 0.0)
                                            for c, b, a, na in zip(_cg["_c"], _cg["_b"], _cg["a"], _cg["na"])}
                            _tot_c = float(_cy["_att"].sum())
                            _nonusa_default = float(_cy["_natt"].sum() / _tot_c) if _tot_c > 0 else 0.0
                        else:
                            log("   [Warning] USA-only gateways configured but no 'country' column "
                                "in the attempts data — country restriction NOT enforced this run.")
                            _usa_only = set()
                    if _elig_rules or _wallet_incapable or _usa_only:
                        log(f"   eligibility: {len(_elig_rules)} ban rule(s), {len(_wallet_incapable)} wallet-incapable id(s), "
                            f"global wallet share {_wallet_default:.1%}; {len(_usa_only)} USA-only id(s), "
                            f"global Non-USA share {_nonusa_default:.1%}.")
                    # Country presence per (currency, BIN) from the attempts data — drives the
                    # export's USA / Non-USA row split. USA-only gateways appear in USA rows only.
                    _country_pres = {}
                    if "country" in orig_adf.columns and "attempts" in orig_adf.columns:
                        _cp = orig_adf.copy()
                        _cpv = _cp["country"].astype(str).str.strip().str.upper()
                        _isusa = _cpv.isin(["USA", "US"]).to_numpy()
                        _catt = pd.to_numeric(_cp["attempts"], errors="coerce").fillna(0.0).to_numpy()
                        _cp["_usa_att"] = np.where(_isusa, _catt, 0.0)
                        _cp["_non_att"] = np.where(~_isusa, _catt, 0.0)
                        _cp["_c"] = _cp["currency"].astype(str).str.strip().str.lower()
                        _cp["_b"] = _cp["bank"].astype(str).str.strip()
                        _cpg = _cp.groupby(["_c", "_b"], as_index=False).agg(usa=("_usa_att", "sum"), non=("_non_att", "sum"))
                        _country_pres = {(c, b): (float(u), float(n))
                                         for c, b, u, n in zip(_cpg["_c"], _cpg["_b"], _cpg["usa"], _cpg["non"])}
                    # The USA-only gatewayFids (+ their vampMids) — loaded regardless of whether the
                    # country restriction was enforced this run, so the export can always split rows.
                    _usa_only_export = set(load_usa_only(_rr_path))
                    for _f in list(_usa_only_export):
                        if _f in _fid2vamp_l:
                            _usa_only_export.add(_fid2vamp_l[_f])

                    # Wallet context for the k-means/config wallet dimension (tab 5) + the export.
                    ss["wallet_ctx"] = {"incapable": set(_wallet_incapable),
                                        "frac": dict(_wallet_frac),
                                        "default": float(_wallet_default),
                                        "fid2vamp": dict(_fid2vamp_l),
                                        "country_pres": _country_pres,
                                        "usa_only": _usa_only_export,
                                        "max_share": float(max_share)}

                    # [FN-320]
                    def _restrict(spl):
                        if not _elig_rules and not _wallet_incapable and not _usa_only:
                            return spl
                        return apply_restrictions(spl, _elig_rules, _fid2vamp_l,
                                                  wallet_incapable=frozenset(_wallet_incapable),
                                                  wallet_frac=_wallet_frac, wallet_default=_wallet_default,
                                                  usa_only=frozenset(_usa_only),
                                                  nonusa_frac=_nonusa_frac, nonusa_default=_nonusa_default)

                    # Fold eligibility INTO the solve: after eligibility redistributes
                    # volume, re-enforce the per-vampMid VAMP cap on the eligibility-
                    # respecting GRANULAR split (cell = rpgt|currency|bin), then re-apply
                    # eligibility, iterating to a consistent split. So the delivered split
                    # is both eligibility-respecting AND VAMP-compliant, and compliance is
                    # measured on what is actually routed.
                    # Everything _mid_cap_granular builds (_cur/_pb/_gw/_vm/cell/_key/rate/
                    # cell_vol) depends only on the row's currency/bank/gateway/rpgt/cell_volume —
                    # NOT on `share`. The VAMP loop runs it 2× per pass across both passes, so we
                    # memoise the static columns on a content hash of those key columns and only
                    # re-attach the current `share`. Bit-identical; self-invalidates (hash changes)
                    # if the row content or order ever differs.
                    _mcg_cache = {"key": None, "static": None}

                    # [FN-321]
                    def _mid_cap_granular(gran):
                        _kc = ["currency", "bank", "gateway", "rpgt"]
                        _key = None
                        try:
                            _h = int(pd.util.hash_pandas_object(gran[_kc].astype(str), index=False).sum() & ((1 << 63) - 1))
                            _cvser = pd.to_numeric(gran.get("cell_volume", pd.Series(0.0, index=gran.index)), errors="coerce").fillna(0.0)
                            _hv = int(pd.util.hash_pandas_object(_cvser, index=False).sum() & ((1 << 63) - 1))
                            _key = (len(gran), _h, _hv)
                        except Exception:  # noqa: BLE001 — hashing failed → compute directly
                            _key = None
                        if _key is not None and _mcg_cache["key"] == _key:
                            g = _mcg_cache["static"].copy()
                            g["share"] = gran["share"].to_numpy()
                            return g
                        g = gran.copy()
                        g["_cur"] = g["currency"].astype(str).str.strip().str.lower()
                        g["_pb"] = g["bank"].astype(str).map(
                            lambda b: bin_to_bank.get(b, bin_to_bank.get(str(b).strip().lower(), b))).astype(str).str.strip().str.lower()
                        g["_gw"] = g["gateway"].astype(str).str.strip().str.lower()
                        g["_vm"] = g["_gw"].map(_fid2vamp_l).fillna(g["_gw"])
                        g["cell"] = (g["rpgt"].astype(str).str.lower() + "|" + g["_cur"] + "|" + g["bank"].astype(str).str.lower())
                        g["_key"] = list(zip(g["_cur"], g["_pb"], g["_vm"]))
                        g["rate"] = pd.to_numeric(g["_key"].map(mid_rate), errors="coerce")
                        g["rate"] = g["rate"].fillna(pd.to_numeric(g.get("gateway_risk_rate", 0.006), errors="coerce")).fillna(0.006)
                        g["cell_vol"] = pd.to_numeric(g.get("cell_volume", 0.0), errors="coerce").fillna(0.0)
                        if _key is not None:
                            _mcg_cache["key"] = _key
                            _mcg_cache["static"] = g.copy()
                        return g

                    # STAGE 4 — feed the backup catch-all into the GA FITNESS so the engine optimises
                    # against the shares the pipeline will ACTUALLY route (tab 5), not the raw split.
                    # The catch-all is pooled (unweighted mean over pmp/Country) to the optimiser's
                    # coarser currency×bank×rpgt grain and mapped fid→vampMid ONCE here; the exact
                    # per-pmp/Country blend is applied in the tab-3 projection (Stage 3). Empty ⇒
                    # no blend. ROUTING_BACKUP_BLEND=0 disables. NOTE: this steers the GA's band/
                    # badness objective; the hard VAMP-cap enforcement still runs on the raw split.
                    # (Restored 2026-08-04: this setup block was dropped during the tab split, which
                    # left _bcs4/_bpool_rpgt/_bpool_all undefined → every per-MID band path raised a
                    # NameError and silently fell back to "post-enforcement only" / "no cap enforced".)
                    _bpool_rpgt, _bpool_all, _bcs4 = {}, {}, None
                    _bcatch_ga = ss.get("backup_catchall") or {}
                    if _bcatch_ga and os.environ.get("ROUTING_BACKUP_BLEND", "1") != "0":
                        try:
                            from routing_optimiser.backup_blend import blend_cell_shares as _bcs4
                            from collections import defaultdict as _dd4
                            _ar, _cr = _dd4(lambda: _dd4(float)), _dd4(int)
                            _aa, _ca4 = _dd4(lambda: _dd4(float)), _dd4(int)
                            for (_cur4, _rp4, _pmp4, _ct4), _gw4 in _bcatch_ga.items():
                                _cr[(_cur4, _rp4)] += 1; _ca4[_cur4] += 1
                                for _fid4, _pct4 in _gw4.items():
                                    _vm4 = fid2vamp.get(str(_fid4).strip().lower())
                                    if _vm4 is None:
                                        continue
                                    _ar[(_cur4, _rp4)][_vm4] += float(_pct4)
                                    _aa[_cur4][_vm4] += float(_pct4)
                            _bpool_rpgt = {k: {vm: v / _cr[k] for vm, v in d.items()} for k, d in _ar.items()}
                            _bpool_all = {k: {vm: v / _ca4[k] for vm, v in d.items()} for k, d in _aa.items()}
                            log(f"   backup catch-all blended into the GA fitness: "
                                f"{len(_bpool_rpgt)} (currency×rpgt) pool(s).")
                        except Exception as _b4e:  # noqa: BLE001
                            _bpool_rpgt, _bpool_all, _bcs4 = {}, {}, None
                            log(f"   [backup GA blend disabled: {type(_b4e).__name__}: {_b4e}]")

                    # [FN-322]
                    def _blend_ga(prop):
                        # Inject the pooled catch-all vampMids per cell (renormalising), reusing the
                        # tested blend_cell_shares. No-op when no backup is configured.
                        if not prop or _bcs4 is None or not (_bpool_rpgt or _bpool_all):
                            return prop
                        _byr = len(prop[0]) == 5
                        _cells, _order = {}, []
                        for _t in prop:
                            if _byr:
                                _c, _b, _rp, _vm, _s = _t; _ck = (_c, _b, _rp)
                            else:
                                _c, _b, _vm, _s = _t; _ck = (_c, _b)
                            if _ck not in _cells:
                                _cells[_ck] = {}; _order.append(_ck)
                            _cells[_ck][_vm] = _cells[_ck].get(_vm, 0.0) + _s
                        _out = []
                        for _ck in _order:
                            if _byr:
                                _c, _b, _rp = _ck; _ca = _bpool_rpgt.get((_c, _rp), {})
                            else:
                                _c, _b = _ck; _ca = _bpool_all.get(_c, {})
                            _eff = _bcs4(_cells[_ck], _ca) if _ca else _cells[_ck]
                            for _vm, _s in _eff.items():
                                _out.append((_c, _b, _rp, _vm, _s) if _byr else (_c, _b, _vm, _s))
                        return tuple(_out)

                    # [FN-323]
                    def _prop_items_from_gran(gran):
                        # Proposed shares the pro-rata projection consumes (matches the tab-4 VAMP
                        # impact table). At Bank×Currency grain → (Currency, BIN, vampMid, share).
                        # At Bank×Currency×RPGT grain → (Currency, BIN, RPGT, vampMid, share), so a
                        # per-RPGT split is projected/enforced per RPGT (e.g. move addon sales off a
                        # MID without touching its other RPGTs) instead of one share across RPGTs.
                        # The backup catch-all (Stage 4) is folded in so the GA scores actual routing.
                        g = gran.copy()
                        g["_vm"] = g["gateway"].astype(str).str.strip().str.lower().map(fid2vamp)
                        g = g.dropna(subset=["_vm"])
                        if _opt_by_rpgt:
                            pr = g.groupby(["currency", "bank", "rpgt", "_vm"], as_index=False)["share"].sum()
                            _prop = tuple((str(c).lower(), str(b), str(rp), str(v), float(s))
                                          for c, b, rp, v, s in
                                          pr[["currency", "bank", "rpgt", "_vm", "share"]].itertuples(index=False))
                        else:
                            pr = g.groupby(["currency", "bank", "_vm"], as_index=False)["share"].sum()
                            _prop = tuple((str(c).lower(), str(b), str(v), float(s))
                                          for c, b, v, s in pr[["currency", "bank", "_vm", "share"]].itertuples(index=False))
                        return _blend_ga(_prop)

                    # ---- GATE-1 diagnostic: collapsed BandProjector vs the TRUE _project_capped ----
                    # Proves the exact linear collapse (src/routing_optimiser/band_projection.py)
                    # reproduces _project_capped on THIS run's real scaffold, before we ever let it
                    # score the search. Toggle: Tab 3 "Band-projection collapse diagnostic". One-shot,
                    # fully guarded — a failure only skips the log line, never the run.
                    _band_diag_state = {"bp": None, "done": False, "cost_done": False,
                                        "frames": None, "pbp": None, "slope_done": False}

                    # [FN-324]
                    def _band_frames():
                        """Adapter (T0a, Pca, pool, sorted band-set, by_rpgt) for the collapse
                        projectors, cached per run. None if this run built no cap scaffold."""
                        if _band_diag_state["frames"] is not None:
                            return _band_diag_state["frames"]
                        if _T0 is None or _Pc is None or _Pc_movedvpool_a is None:
                            return None
                        _em = (_T0_emask_a if _T0_emask_a is not None else np.zeros(len(_T0), bool))
                        _T0a = pd.DataFrame({
                            "cur": _T0["_cur"].to_numpy(), "bin": _T0["_bin"].to_numpy(),
                            "rpgt": _T0["_rpgt"].to_numpy(), "pmp": _T0["_pmp"].to_numpy(),
                            "ctry": _T0["_ctry"].to_numpy(), "mid": _T0["_mid"].to_numpy(),
                            "midl": _T0["_midl"].to_numpy(), "per": _T0["_per"].to_numpy(),
                            "vi": _T0["_vi"].to_numpy(), "vc": _T0["_vc"].to_numpy(),
                            "pr": _T0["_pr"].to_numpy(), "fcp": _T0["_fcp"].to_numpy(),
                            "bf": _T0["_bf"].to_numpy(), "excl": _T0["_excl"].to_numpy(),
                            "emask": np.asarray(_em, bool),
                            "iscap": _T0["_midl"].isin(_capped_l).to_numpy(), "_av": _T0["_av"].to_numpy()})
                        _Pca = pd.DataFrame({
                            "cur": _Pc["_cur"].to_numpy(), "bin": _Pc["_bin"].to_numpy(),
                            "rpgt": _Pc["_rpgt"].to_numpy(), "pmp": _Pc["_pmp"].to_numpy(),
                            "ctry": _Pc["_ctry"].to_numpy(), "mid": _Pc["_mid"].to_numpy(),
                            "midl": _Pc["_midl"].to_numpy(), "per": _Pc["_per"].to_numpy(),
                            "t": _Pc["_t"].to_numpy(), "vc": _Pc["_vc"].to_numpy()})
                        # ONLY the actually-constrained (midl, period) pairs — restricting the band
                        # set here is what lets the reduced scaffold shrink (fewer banded rows →
                        # fewer relevant cells). Derived from the live per-MID month rules.
                        _bset = set()
                        for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                            if _mtr not in ("txn", "vamp"):
                                continue
                            _mkl = str(_mk).strip().lower()
                            if _mo is None:
                                for _m in range(6):
                                    _bset.add((_mkl, _m))
                            else:
                                _bset.add((_mkl, int(_mo)))
                        _band_diag_state["frames"] = (_T0a, _Pca, np.asarray(_Pc_movedvpool_a, float),
                                                      sorted(_bset), bool(_opt_by_rpgt))
                        return _band_diag_state["frames"]

                    # [FN-325]
                    def _band_collapse_diag(prop_items, label):
                        if not ss.get("ga_band_diag", False) or _band_diag_state["done"]:
                            return
                        _band_diag_state["done"] = True
                        try:
                            from routing_optimiser.band_projection import BandProjector as _BP
                            if _band_diag_state["bp"] is None:
                                _fr = _band_frames()
                                if _fr is None:
                                    log("   [gate-1 band diagnostic: no cap scaffold this run — skipped]")
                                    return
                                _T0a, _Pca, _poolarr, _bset, _byr = _fr
                                _band_diag_state["bp"] = _BP(_T0a, _Pca, _poolarr, _bset, by_rpgt=_byr)
                            _bp = _band_diag_state["bp"]
                            _prop = {tuple(t[:-1]): float(t[-1]) for t in prop_items}
                            _fast = _bp.project(_prop)
                            _true = _project_capped(list(prop_items), _use_cache=False)
                            log(f"   ── COLLAPSED vs TRUE band projection ({label}) · gate-1 diagnostic ──")
                            log(f"      {'band':<32} {'metric':<5} {'true':>13} {'collapsed':>13} {'|gap|':>11}")
                            _maxg = 0.0; _seen = set()
                            for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                if _mtr not in ("txn", "vamp"):
                                    continue
                                _rk = (_mk, _mo, _mtr)
                                if _rk in _seen:
                                    continue
                                _seen.add(_rk)
                                _months = [int(_mo)] if _mo is not None else list(range(4))
                                _ix = 1 if _mtr == "txn" else 0
                                _tv = float(sum(_true.get((_mk, m), (0.0, 0.0))[_ix] for m in _months))
                                _fv = float(sum(_fast.get((_mk, m), [0.0, 0.0])[_ix] for m in _months))
                                _g = abs(_tv - _fv); _maxg = max(_maxg, _g)
                                _mo_s = f"M{int(_mo)}" if _mo is not None else "M0-3"
                                log(f"      {(_mk + ' ' + _mo_s)[:32]:<32} {_mtr:<5} "
                                    f"{_tv:>13,.1f} {_fv:>13,.1f} {_g:>11,.2e}")
                            log(f"      → max |gap| = {_maxg:,.3e}  (≈0 ⇒ the collapse == the true "
                                "projection on THIS scaffold; ready to wire into the GA score)")
                        except Exception as _bde:  # noqa: BLE001 - diagnostic must never break a run
                            log(f"   [gate-1 band diagnostic skipped: {type(_bde).__name__}: {_bde}]")

                    # [FN-326]
                    def _band_cost_probe():
                        """Measure the REAL reduced-scaffold size + one population project_pop time,
                        so we know the per-generation cost of exact NumPy band scoring BEFORE wiring
                        it into the worker fitness. Toggle: Tab 3 'Band-score cost probe (gate 2)'.
                        One-shot, fully guarded — never breaks a run."""
                        if not ss.get("ga_band_cost", False) or _band_diag_state["cost_done"]:
                            return
                        _band_diag_state["cost_done"] = True
                        try:
                            import time as _time
                            from routing_optimiser.band_projection import PopulationBandProjector as _PBP
                            _fr = _band_frames()
                            if _fr is None:
                                log("   [gate-2 cost probe: no cap scaffold this run — skipped]")
                                return
                            _T0a, _Pca, _poolarr, _bset, _byr = _fr
                            _tb = _time.perf_counter()
                            _pbp = _PBP(_T0a, _Pca, _poolarr, _bset, by_rpgt=_byr)
                            _build_ms = (_time.perf_counter() - _tb) * 1000.0
                            _K = len(_pbp.prop_keys); _B = len(_pbp.band_order)
                            _ncell = int(_pbp._ngc); _nrel = int(len(_pbp._gcode))
                            _npc = int(len(_pbp._pc_bandcol)); _ncap = int(len(_pbp._t_rows))
                            log(f"   ── gate-2 band-score COST probe ── reduced scaffold: "
                                f"cells={_ncell:,} · t0_rows={_nrel:,} · prop_keys={_K:,} · "
                                f"banded_aged_rows={_npc:,} · cap_rows={_ncap:,} · bands={_B} · "
                                f"build={_build_ms:.0f}ms")
                            _rng = np.random.default_rng(0)
                            for _P in (25, 64):
                                _pr = _rng.random((_P, max(_K, 1)))
                                _pr /= _pr.sum(axis=1, keepdims=True)
                                _r0 = _time.perf_counter()
                                for _ in range(5):
                                    _pbp.project_pop(_pr)
                                _ms = (_time.perf_counter() - _r0) / 5 * 1000.0
                                log(f"      project_pop(P={_P}) = {_ms:.1f} ms/call "
                                    f"(≈ per-generation band-score cost at λ={_P})")
                            log("      → compare to the numba step (~9 ms/gen). If project_pop is a small "
                                "multiple, per-generation exact scoring is viable; if ≫, we cadence it.")
                        except Exception as _cpe:  # noqa: BLE001 - diagnostic must never break a run
                            log(f"   [gate-2 cost probe skipped: {type(_cpe).__name__}: {_cpe}]")

                    # [FN-327]
                    def _band_compress_probe():
                        """How far could the EXACT projection scaffold shrink losslessly? The candidate
                        only decides at (cur,bin,rpgt); the projection is carried ~15× finer (× pmp ×
                        ctry). Rows/cells that respond IDENTICALLY to the candidate merge losslessly.
                        The exact merge key is (cur,bin,rpgt,per, active-MID signature) — the signature =
                        the set of MIDs that are active (not excl / not emask) and which carry VAMP
                        (vc>0), because the wallet/USA mask is what varies across pmp/ctry. This probe
                        COUNTS those groups vs the raw scaffold, so we know the compression ceiling (and
                        thus whether exact-in-loop is revived) before building the compressed projector.
                        Toggle: Tab 3 'Band-score compression probe (gate 2)'. One-shot, guarded."""
                        if not ss.get("ga_band_compress", False) or _band_diag_state.get("compress_done"):
                            return
                        _band_diag_state["compress_done"] = True
                        try:
                            _fr = _band_frames()
                            if _fr is None:
                                log("   [gate-2 compression probe: no cap scaffold this run — skipped]")
                                return
                            _T0a = _fr[0]
                            _gk = ["cur", "bin", "rpgt", "pmp", "ctry", "per"]
                            _active = ~(_T0a["excl"].to_numpy(bool) | _T0a["emask"].to_numpy(bool))
                            _act = _T0a.loc[_active].copy()
                            # order-independent set signature per fine cell = Σ hash(midl|vc>0)
                            _item = (_act["midl"].astype(str) + "|"
                                     + (_act["vc"].to_numpy(float) > 0).astype(int).astype(str))
                            _ih = pd.util.hash_pandas_object(_item, index=False).to_numpy().astype("uint64")
                            _cc, _ = pd.factorize(_act[_gk].astype(str).agg("|".join, axis=1))
                            _act = _act.assign(_cc=_cc, _ih=_ih)
                            _g = _act.groupby("_cc")
                            _cellinfo = _g[_gk].first()
                            _cellinfo["sig"] = _g["_ih"].sum().to_numpy().astype("int64")
                            _cellinfo["coarse"] = (_cellinfo["cur"] + "|" + _cellinfo["bin"] + "|"
                                                   + _cellinfo["rpgt"] + "|" + _cellinfo["per"].astype(str))
                            _cur_rows = int(len(_T0a)); _cur_cells = int(_T0a.drop_duplicates(_gk).shape[0])
                            _coll_cells = int(_cellinfo["coarse"].nunique())              # collapse pmp/ctry only
                            _exact_cells = int(_cellinfo.drop_duplicates(["coarse", "sig"]).shape[0])
                            _act2 = _act.merge(_cellinfo.reset_index()[["_cc", "sig", "coarse"]], on="_cc")
                            _coll_rows = int(_act2.drop_duplicates(["coarse", "midl"]).shape[0])
                            _exact_rows = int(_act2.drop_duplicates(["coarse", "sig", "midl"]).shape[0])
                            _sig_per_coarse = _cellinfo.groupby("coarse")["sig"].nunique()
                            _mean_sig = float(_sig_per_coarse.mean()); _max_sig = int(_sig_per_coarse.max())
                            log("   ── gate-2 band-score COMPRESSION probe (lossless minimal grain) ──")
                            log(f"      raw scaffold            : t0_rows={_cur_rows:,} · fine_cells={_cur_cells:,}")
                            log(f"      collapse pmp/ctry only  : rows={_coll_rows:,} ({_cur_rows/max(_coll_rows,1):.1f}×) "
                                f"· cells={_coll_cells:,} ({_cur_cells/max(_coll_cells,1):.1f}×)")
                            log(f"      exact (mask-signature)  : rows={_exact_rows:,} ({_cur_rows/max(_exact_rows,1):.1f}×) "
                                f"· cells={_exact_cells:,} ({_cur_cells/max(_exact_cells,1):.1f}×)")
                            log(f"      distinct signatures per (cur,bin,rpgt,per): mean={_mean_sig:.2f} · max={_max_sig} "
                                "(≈1 ⇒ masking is uniform ⇒ near-full collapse; ≫1 ⇒ masking splits cells)")
                            log(f"      → exact projection cost scales with rows, so ≈{_cur_rows/max(_exact_rows,1):.0f}× "
                                "cheaper is the ceiling. Apply to the last project_pop time to see if exact-in-loop "
                                "(or even per-candidate) is revived.")
                        except Exception as _cme:  # noqa: BLE001 - diagnostic must never break a run
                            log(f"   [gate-2 compression probe skipped: {type(_cme).__name__}: {_cme}]")

                    # [FN-328]
                    def _get_pbp():
                        """Build & cache the population band projector on this run's real scaffold."""
                        if _band_diag_state["pbp"] is not None:
                            return _band_diag_state["pbp"]
                        _fr = _band_frames()
                        if _fr is None:
                            return None
                        from routing_optimiser.band_projection import PopulationBandProjector as _PBP
                        _T0a, _Pca, _poolarr, _bset, _byr = _fr
                        _band_diag_state["pbp"] = _PBP(_T0a, _Pca, _poolarr, _bset, by_rpgt=_byr)
                        return _band_diag_state["pbp"]

                    # [FN-329]
                    def _band_slope_probe(ref_prop_items, end_prop_items):
                        """Does a FIRST-ORDER (slope) band model stay accurate over the search's ACTUAL
                        move? Walk the straight line from the reference split to the delivered endpoint,
                        evaluate the EXACT band values at α=0,.25,.5,.75,1, and compare a slope
                        extrapolation (calibrated from α:0→.25) to the exact value at the endpoint. A big
                        gap ⇒ the tilts are too large for a pure slope model (needs the quadratic term).
                        Toggle: Tab 3 'Band-score slope-accuracy probe (gate 2)'. One-shot, guarded."""
                        if not ss.get("ga_band_slope", False) or _band_diag_state["slope_done"]:
                            return
                        _band_diag_state["slope_done"] = True
                        try:
                            _pbp = _get_pbp()
                            if _pbp is None:
                                log("   [gate-2 slope probe: no cap scaffold this run — skipped]")
                                return
                            _ref = {tuple(t[:-1]): float(t[-1]) for t in ref_prop_items}
                            _end = {tuple(t[:-1]): float(t[-1]) for t in end_prop_items}
                            _keys = set(_ref) | set(_end)
                            _alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                            _props = [{k: _ref.get(k, 0.0) + _a * (_end.get(k, 0.0) - _ref.get(k, 0.0))
                                       for k in _keys} for _a in _alphas]
                            _vamp, _txn = _pbp.project_pop_from_props(_props)      # (5, B) each
                            _bpos = {b: i for i, b in enumerate(_pbp.band_order)}
                            log("   ── gate-2 band-score SLOPE-ACCURACY probe (reference → delivered endpoint) ──")
                            log(f"      {'band':<30} {'mtr':<4} {'exact@ref':>11} {'exact@end':>11} "
                                f"{'slope@end':>11} {'|gap|':>10} {'rel%':>6}")
                            _maxrel = 0.0; _seen = set()
                            for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                if _mtr not in ("txn", "vamp"):
                                    continue
                                _mkl = str(_mk).strip().lower()
                                _rk = (_mkl, _mo, _mtr)
                                if _rk in _seen:
                                    continue
                                _seen.add(_rk)
                                _months = [int(_mo)] if _mo is not None else list(range(6))
                                _mat = _txn if _mtr == "txn" else _vamp

                                # [FN-330]
                                def _col(_ai):
                                    return float(sum(_mat[_ai, _bpos[(_mkl, m)]]
                                                     for m in _months if (_mkl, m) in _bpos))
                                _v0 = _col(0); _vend = _col(4)
                                _slope_end = _v0 + (_col(1) - _v0) / 0.25          # extrapolate slope to α=1
                                _gap = abs(_vend - _slope_end)
                                _rel = _gap / max(abs(_vend), 1.0); _maxrel = max(_maxrel, _rel)
                                _mo_s = f"M{int(_mo)}" if _mo is not None else "M*"
                                log(f"      {(_mk + ' ' + _mo_s)[:30]:<30} {_mtr:<4} {_v0:>11,.1f} "
                                    f"{_vend:>11,.1f} {_slope_end:>11,.1f} {_gap:>10,.1f} {_rel*100:>5.1f}%")
                            log(f"      → max first-order (slope) error = {_maxrel*100:.1f}% of the band value "
                                "at the delivered endpoint. Small ⇒ a slope model is accurate enough to wire "
                                "into the search; large ⇒ the tilts need the quadratic (curvature) term.")
                        except Exception as _spe:  # noqa: BLE001 - diagnostic must never break a run
                            log(f"   [gate-2 slope probe skipped: {type(_spe).__name__}: {_spe}]")

                    # [FN-331]
                    def _band_enabler_probes(ref_prop_items, end_prop_items):
                        """Measure the two enablers of the 'near-exact + incredibly fast' path:
                        (1) VOLUME PRUNING — drop the near-zero-volume cell tail, report the row shrink,
                            the project_pop speedup, and the per-band error vs the full exact projection.
                        (2) LOCAL-LINEAR — fit a linear model at the delivered endpoint from a tiny step
                            and test it over a few-generations-sized step; if it stays accurate LOCALLY
                            (unlike the 54% global slope error), a cheap local surrogate is viable.
                        Toggle: Tab 3 'Band-score enabler probes (gate 2)'. One-shot, guarded."""
                        if not ss.get("ga_band_enablers", False) or _band_diag_state.get("enab_done"):
                            return
                        _band_diag_state["enab_done"] = True
                        try:
                            import time as _time
                            from routing_optimiser.band_projection import PopulationBandProjector as _PBP
                            _fr = _band_frames()
                            if _fr is None:
                                log("   [gate-2 enabler probes: no cap scaffold this run — skipped]")
                                return
                            _T0a, _Pca, _poolarr, _bset, _byr = _fr
                            _end = {tuple(t[:-1]): float(t[-1]) for t in end_prop_items}
                            _ref = {tuple(t[:-1]): float(t[-1]) for t in ref_prop_items}
                            _rules = [(str(_mk).strip().lower(), _mo, _mtr)
                                      for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules
                                      if _mtr in ("txn", "vamp")]
                            _rules = list(dict.fromkeys(_rules))

                            # [FN-332]
                            def _val(vmat, tmat, border, row, mkl, mo, mtr):
                                _bp = {b: i for i, b in enumerate(border)}
                                _ms = [int(mo)] if mo is not None else list(range(6))
                                _m = tmat if mtr == "txn" else vmat
                                return float(sum(_m[row, _bp[(mkl, mm)]] for mm in _ms if (mkl, mm) in _bp))

                            # [FN-333]
                            def _timeit(proj, P=25, reps=3):
                                _K = max(len(proj.prop_keys), 1)
                                _pr = np.random.default_rng(0).random((P, _K)); _pr /= _pr.sum(1, keepdims=True)
                                proj.project_pop(_pr)                       # warm
                                _r0 = _time.perf_counter()
                                for _ in range(reps):
                                    proj.project_pop(_pr)
                                return (_time.perf_counter() - _r0) / reps * 1000.0

                            _full = _get_pbp()
                            _fv, _ft = _full.project_pop_from_props([_end])
                            _full_ms = _timeit(_full)

                            # ---- ENABLER 1: volume pruning ----
                            # [FN-334]
                            def _ck(df):
                                return (df["cur"] + "|" + df["bin"] + "|" + df["rpgt"] + "|"
                                        + df["pmp"] + "|" + df["ctry"] + "|" + df["per"].astype(str))
                            _cellvol = _T0a.assign(_k=_ck(_T0a)).groupby("_k")["vi"].sum().sort_values(ascending=False)
                            _cum = _cellvol.cumsum() / max(float(_cellvol.sum()), 1e-9)
                            _t0k = _ck(_T0a); _pck = _ck(_Pca)
                            log("   ── gate-2 ENABLER 1: volume pruning (drop near-zero-volume cell tail) ──")
                            log(f"      full: t0_rows={len(_T0a):,} · project_pop(P=25)={_full_ms:.0f}ms")
                            for _cov in (0.99, 0.999):
                                _keep = set(_cum.index[_cum.to_numpy() <= _cov]) | {_cum.index[0]}
                                _m0 = _t0k.isin(_keep).to_numpy(); _mp = _pck.isin(_keep).to_numpy()
                                _pp = _PBP(_T0a[_m0].reset_index(drop=True), _Pca[_mp].reset_index(drop=True),
                                           _poolarr[_mp], _bset, by_rpgt=_byr)
                                _pv, _pt = _pp.project_pop_from_props([_end])
                                _pms = _timeit(_pp)
                                _maxe = 0.0
                                for (_mkl, _mo, _mtr) in _rules:
                                    _tv = _val(_fv, _ft, _full.band_order, 0, _mkl, _mo, _mtr)
                                    _pv2 = _val(_pv, _pt, _pp.band_order, 0, _mkl, _mo, _mtr)
                                    _maxe = max(_maxe, abs(_tv - _pv2) / max(abs(_tv), 1.0))
                                log(f"      keep {_cov*100:.1f}% vol: rows={int(_m0.sum()):,} "
                                    f"({len(_T0a)/max(int(_m0.sum()),1):.1f}× smaller) · "
                                    f"project_pop={_pms:.0f}ms ({_full_ms/max(_pms,1e-9):.1f}× faster) · "
                                    f"max band error={_maxe*100:.2f}%")

                            # ---- ENABLER 2: local-linear accuracy over a few-generation step ----
                            _dir = {k: _ref.get(k, 0.0) - _end.get(k, 0.0) for k in set(_ref) | set(_end)}
                            _steps = [0.0, 0.02, 0.10]      # 0=endpoint, 0.02=tiny(slope), 0.10≈a few gens back
                            _lp = [{k: _end.get(k, 0.0) + _s * _dir[k] for k in _dir} for _s in _steps]
                            _lv, _lt = _full.project_pop_from_props(_lp)
                            log("   ── gate-2 ENABLER 2: LOCAL-linear accuracy (model refreshed at endpoint) ──")
                            log(f"      {'band':<30} {'mtr':<4} {'exact':>11} {'local-lin':>11} {'err%':>6}")
                            _maxle = 0.0
                            for (_mkl, _mo, _mtr) in _rules:
                                _v0 = _val(_lv, _lt, _full.band_order, 0, _mkl, _mo, _mtr)
                                _vs = _val(_lv, _lt, _full.band_order, 1, _mkl, _mo, _mtr)
                                _vt = _val(_lv, _lt, _full.band_order, 2, _mkl, _mo, _mtr)
                                _slope = (_vs - _v0) / 0.02
                                _pred = _v0 + _slope * 0.10
                                _err = abs(_pred - _vt) / max(abs(_vt), 1.0); _maxle = max(_maxle, _err)
                                _mo_s = f"M{int(_mo)}" if _mo is not None else "M*"
                                log(f"      {(_mkl + ' ' + _mo_s)[:30]:<30} {_mtr:<4} {_vt:>11,.1f} "
                                    f"{_pred:>11,.1f} {_err*100:>5.1f}%")
                            log(f"      → max LOCAL-linear error = {_maxle*100:.1f}% over a ~10%-of-move step "
                                "(≈4 generations). Small ⇒ a local surrogate refreshed each generation stays "
                                "near-exact ⇒ the 'near-exact + incredibly fast' path is on the table.")
                        except Exception as _epe:  # noqa: BLE001 - diagnostic must never break a run
                            log(f"   [gate-2 enabler probes skipped: {type(_epe).__name__}: {_epe}]")

                    # Per-(currency, bank, gateway) INCREMENTAL-REVENUE rate = success_rate × avg ticket.
                    # Used below to redistribute a MID's freed volume to the HIGHEST-revenue alternative
                    # (cost-aware cuts, #4) rather than proportionally. Built once from the engine's
                    # success rates (agg_sr) × the attempts' per-(currency,bank) ticket; {} ⇒ the scaler
                    # falls back to the prior proportional redistribution (no regression if unavailable).
                    _inc_rev_rate = {}
                    try:
                        if (isinstance(agg_sr, pd.DataFrame)
                                and {"currency", "bank", "gateway", "success_rate"}.issubset(agg_sr.columns)):
                            _tk = {}
                            if (isinstance(orig_adf, pd.DataFrame)
                                    and {"currency", "bank", "amount", "success"}.issubset(orig_adf.columns)):
                                _a = orig_adf
                                _sa = (pd.to_numeric(_a["amount"], errors="coerce").fillna(0.0)
                                       * pd.to_numeric(_a["success"], errors="coerce").fillna(0.0))
                                _sc = pd.to_numeric(_a["success"], errors="coerce").fillna(0.0)
                                _tg = pd.DataFrame({
                                    "cur": _a["currency"].astype(str).str.strip().str.lower(),
                                    "bk": _a["bank"].astype(str).str.strip().str.lower(),
                                    "sa": _sa.to_numpy(), "s": _sc.to_numpy()}).groupby(["cur", "bk"]).sum()
                                _tk = {k: (float(r["sa"]) / float(r["s"]) if float(r["s"]) > 0 else 25.0)
                                       for k, r in _tg.iterrows()}
                            _srv = pd.to_numeric(agg_sr["success_rate"], errors="coerce").fillna(0.0).to_numpy()
                            _cu = agg_sr["currency"].astype(str).str.strip().str.lower().to_numpy()
                            _bk = agg_sr["bank"].astype(str).str.strip().str.lower().to_numpy()
                            _gw = agg_sr["gateway"].astype(str).str.strip().str.lower().to_numpy()
                            for _i in range(len(_srv)):
                                _inc_rev_rate[(_cu[_i], _bk[_i], _gw[_i])] = float(_srv[_i]) * float(_tk.get((_cu[_i], _bk[_i]), 25.0))
                    except Exception:  # noqa: BLE001
                        _inc_rev_rate = {}

                    # Vectorised "cur|bank|gw" -> incremental-revenue-rate map (speedup 3), so the
                    # per-row _ir builds via a pandas .map instead of a ~78k-row Python dict.get loop.
                    _inc_rev_joined = ({f"{_k[0]}|{_k[1]}|{_k[2]}": _v for _k, _v in _inc_rev_rate.items()}
                                       if _inc_rev_rate else {})

                    # #4b toggle (default OFF = uniform scaling, the safe baseline). When ON,
                    # _scale_mids_in_gran's DOWN path cuts a MID from its CHEAPEST cells first
                    # (cost-ordered) instead of uniformly. _restrict_and_recap flips this ON only
                    # for a Pareto-guarded A/B, so it can never regress. A mutable holder so the
                    # closure can toggle it.
                    _cost_order = {"on": False}
                    _smig_cache = {}   # cached static row-structure keyed on the gran layout (speedup 3)

                    # [FN-335]
                    def _mids_over_granular(gran):
                        if vamp_cap is None or gran is None or getattr(gran, "empty", True):
                            return 0
                        g = _mid_cap_granular(gran)
                        _v = g["cell_vol"] * g["share"]
                        _t = pd.DataFrame({"m": g["_vm"], "vol": _v, "vr": _v * g["rate"]}).groupby("m").sum()
                        _gr = _t["vr"] / _t["vol"].replace(0, np.nan)
                        return int((_gr > float(vamp_cap) + 1e-9).sum())

                    # [FN-336]
                    def _mids_over_blended(gran):
                        # MIDs over the VAMP cap on the ACTUAL routed (backup-blended) projection —
                        # the same numbers tab 3/tab 5 show. Reuses the tested Stage-4 blend
                        # (_prop_items_from_gran) + pro-rata projection (_project_capped), so the
                        # enforcement can iterate until the ROUTED VAMP (not the raw split) is
                        # compliant. Falls back to the raw check when no backup is configured or
                        # the projection is unavailable.
                        if vamp_cap is None or gran is None or getattr(gran, "empty", True):
                            return 0
                        if not (_bpool_rpgt or _bpool_all):
                            return _mids_over_granular(gran)
                        try:
                            _pr = _project_capped(_prop_items_from_gran(gran))   # {(mid,period):(vamp,txn)}
                        except Exception:  # noqa: BLE001
                            return _mids_over_granular(gran)
                        from collections import defaultdict as _ddb
                        _vv, _tt2 = _ddb(float), _ddb(float)
                        for (_mk, _per), _val in _pr.items():
                            _vv[_mk] += float(_val[0]); _tt2[_mk] += float(_val[1])
                        return sum(1 for _mk in _tt2
                                   if _tt2[_mk] > 0 and _vv[_mk] / _tt2[_mk] > float(vamp_cap) + 1e-9)

                    if engine_key in ("genetic", "genetic_numba"):
                        # ---- Global GA: revenue − λ·risk, WARM-STARTED from softmax + HARD-
                        # ENFORCED on output. The GA population is seeded with softmax's
                        # revenue-optimal and compliant splits, so (with elitism) it can't
                        # end up worse than softmax on revenue; the output then runs through
                        # the exact enforcement, so it matches softmax on compliance. λ (from
                        # the slider) still shapes the GA's own search.
                        import routing_optimiser.genetic_global as _gg
                        log(f"   genetic build: {getattr(_gg, '__build__', '?')} — global GA, "
                            "own revenue-greedy reference (genetic_ref, not softmax) + exact hard enforcement.")
                        # From the 30D attempts, build the SAME quantities tab 4 uses for
                        # incremental revenue: avg ticket + cell attempts + raw gateway SR,
                        # keyed by (currency, parent-bank[, gateway]).
                        _at_map, _cellatt_map, _gwsr_map = {}, {}, {}
                        try:
                            _a = orig_adf.copy()
                            # Window to the LAST 30 days (identical to how tab 4 slices the
                            # attempts for its incremental-revenue figure), so magnitudes tie out.
                            _dc = "date" if "date" in _a.columns else ("Date" if "Date" in _a.columns else None)
                            if _dc:
                                _dts = pd.to_datetime(_a[_dc], errors="coerce")
                                _vd = _dts.dropna()
                                if not _vd.empty:
                                    _mx = _vd.max()
                                    _a = _a[(_dts > (_mx - pd.Timedelta(days=30))) & (_dts <= _mx)].copy()
                            _a["_pb"] = _a["bank"].map(lambda b: bin_to_bank.get(b, bin_to_bank.get(str(b).strip().lower(), b))).astype(str).str.strip().str.lower()
                            _a["_cur"] = _a["currency"].astype(str).str.strip().str.lower()
                            _a["_gw"] = _a["gateway"].astype(str).str.strip().str.lower()

                            # [FN-337]
                            def _colnum(_df, _name):   # numeric Series, or zeros if the column is absent
                                return (pd.to_numeric(_df[_name], errors="coerce").fillna(0.0)
                                        if _name in _df.columns else pd.Series(0.0, index=_df.index))
                            _a["_suc"] = _colnum(_a, "success")
                            _a["_att"] = _colnum(_a, "attempts")
                            # succ_amount isn't in the attempts frame — tab 4 derives it as
                            # amount × successes; do the same so the revenue basis matches.
                            _a["_amt"] = (_colnum(_a, "succ_amount") if "succ_amount" in _a.columns
                                          else _colnum(_a, "amount") * _a["_suc"])
                            _gt = _a.groupby(["_cur", "_pb"], as_index=False).agg(amt=("_amt", "sum"), suc=("_suc", "sum"), att=("_att", "sum"))
                            _at_map = {(r["_cur"], r["_pb"]): (r["amt"] / r["suc"] if r["suc"] > 0 else 25.0) for _, r in _gt.iterrows()}
                            _cellatt_map = {(r["_cur"], r["_pb"]): float(r["att"]) for _, r in _gt.iterrows()}
                            _gg2 = _a.groupby(["_cur", "_pb", "_gw"], as_index=False).agg(att=("_att", "sum"), suc=("_suc", "sum"))
                            _gwsr_map = {(r["_cur"], r["_pb"], r["_gw"]): (r["suc"] / r["att"] if r["att"] > 0 else 0.0) for _, r in _gg2.iterrows()}
                        except Exception as _e:
                            log(f"   [Warning] revenue basis (attempts/SR/ticket) failed ({_e}); using fallbacks.")
                        # Build the GA context on the aggregate split rows (sorted so each
                        # cell is a contiguous block). Carry the softmax reference share and
                        # the softmax compliant share through the sort so the reparameterised
                        # GA can use the reference as its decode base (θ=0) in GA row order.
                        G = _mc.copy()
                        G["_ref_share"] = pd.to_numeric(ref_agg["share"], errors="coerce").fillna(0.0).to_numpy()
                        G["_comp_share"] = np.asarray(comp_share, dtype=float)
                        G["_cellk"] = G["cell"].astype(str)
                        G = G.sort_values("_cellk", kind="stable").reset_index(drop=True)
                        _cellk = G["_cellk"].to_numpy()
                        _uc, _counts = np.unique(_cellk, return_counts=True)
                        # np.unique sorts; reorder counts to first-appearance (contiguous) order
                        _order_cells = list(dict.fromkeys(_cellk.tolist()))
                        _cnt_map = dict(zip(_uc.tolist(), _counts.tolist()))
                        _counts = np.array([_cnt_map[c] for c in _order_cells], dtype=int)
                        _cell_starts = np.concatenate([[0], np.cumsum(_counts)[:-1]]).astype(int)
                        _vm = G["vampMid"].astype(str).str.strip().str.lower().to_numpy()
                        _mids_u = list(dict.fromkeys(_vm.tolist()))
                        _mid_index = {m: k for k, m in enumerate(_mids_u)}
                        _mid_id = np.array([_mid_index[m] for m in _vm], dtype=int)
                        _mid_rows = [np.where(_mid_id == k)[0] for k in range(len(_mids_u))]
                        _cvol = pd.to_numeric(G["cell_vol"], errors="coerce").fillna(0.0).to_numpy()
                        _base = pd.to_numeric(G["baseline_share"], errors="coerce").fillna(0.0).to_numpy()
                        _ref_share_G = pd.to_numeric(G["_ref_share"], errors="coerce").fillna(0.0).to_numpy()   # softmax revenue-opt
                        _comp_share_G = pd.to_numeric(G["_comp_share"], errors="coerce").fillna(0.0).to_numpy()  # softmax compliant (greedy)
                        _srr = pd.to_numeric(G["gateway_success_rate"], errors="coerce").fillna(0.0).to_numpy()
                        _rkr = pd.to_numeric(G["rate"], errors="coerce").fillna(0.0).to_numpy()
                        _cur_l = G["currency"].astype(str).str.strip().str.lower().tolist()
                        _pb_l = G["bank"].astype(str).str.strip().str.lower().tolist()
                        _gw_l = G["gateway"].astype(str).str.strip().str.lower().tolist()
                        _tick = np.array([_at_map.get((c, b), 25.0) for c, b in zip(_cur_l, _pb_l)], dtype=float)
                        # Revenue basis = 30D cell attempts × SHRUNK gateway SR × avg ticket. Uses
                        # the SAME shrunk success rate the report and softmax trust (gateway_success_
                        # rate), NOT the raw 30D rate — so a noisy "100% on 2 attempts" gateway can't
                        # look revenue-optimal to the GA, and the GA optimises what's displayed. (E2)
                        _rev_vol = np.array([_cellatt_map.get((c, b), 0.0) for c, b in zip(_cur_l, _pb_l)], dtype=float)
                        _rev_sr = _srr   # shrunk gateway success rate (report-aligned)
                        _rev_coef = _rev_vol * _rev_sr * _tick
                        if float(_rev_coef.sum()) <= 0:
                            _rev_coef = _cvol * _srr * _tick
                            log("   [Warning] GA revenue basis empty; using forecast × smoothed-SR fallback.")
                        # OBJECTIVE (tab-2 dropdown): maximise REVENUE (default) or the VOLUME-WEIGHTED
                        # SUCCESS RATE. For success, drop the avg-ticket factor so the coefficient is
                        # volume × SR — maximising Σ share·vol·SR ≡ maximising the volume-weighted success
                        # rate. Everything downstream (GA fitness, penalty scaling, greedy-vs-GA adoption)
                        # then optimises the chosen objective while the SAME risk constraints still hold.
                        # Objective FIXED to volume-weighted success rate (dropdown removed): drop the
                        # avg-ticket factor so the coefficient is volume × SR, maximising Σ share·vol·SR.
                        _succ_coef = _rev_vol * _rev_sr
                        if float(_succ_coef.sum()) <= 0:
                            _succ_coef = _cvol * _srr
                        _rev_coef = _succ_coef
                        log("   GA objective: maximise VOLUME-WEIGHTED SUCCESS RATE (avg-ticket factor dropped; fixed).")
                        # per-MID stats (reference MID volume feeds the band proxy; ticket/SR for
                        # revenue). The standalone per-MID VOLUME cap was DROPPED — per-MID rules are
                        # enforced via the month bands (projection space), so no routing-volume ceiling.
                        _mid_bvol = np.array([float((_cvol[r] * _base[r]).sum()) for r in _mid_rows])
                        _mid_tick = np.array([float(_tick[r].mean()) if len(r) else 25.0 for r in _mid_rows])
                        _mid_srm = np.array([float(_rev_sr[r].mean()) if len(r) else 0.0 for r in _mid_rows])

                        # Fold the MONTH-SPECIFIC per-MID bands (tab-3 rules) into the GA fitness so
                        # the search actively seeks tilts that satisfy them — not just the aggregate
                        # VAMP cap. Volume-ratio proxy: projected metric ≈ baseline_projected ×
                        # (MID volume / baseline MID volume). Baseline projection is taken once at the
                        # revenue reference. vamp_pct rules are scale-invariant under a volume tilt, so
                        # they're excluded here and left to the exact post-GA enforcement.
                        # [FN-338]
                        def _build_ga_bands(_anchor):
                            """Month-specific per-MID bands for the GA fitness, CALIBRATED so the
                            volume-ratio proxy reproduces the TRUE pro-rata projection AT `_anchor`.
                            Anchoring at the revenue reference ≈ the old behaviour; re-anchoring at
                            the GA's own split (the re-project/correct loop below) removes the proxy's
                            error near the band edges. vamp_pct rules are scale-invariant under a
                            volume tilt → excluded (left to the exact post-GA enforcement)."""
                            _VAMP_VAR_MULT = 3.0   # VAMP-metric bands get 3× the quadratic (variable) weight
                            _agg = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                            _agg["share"] = np.asarray(_anchor, float)
                            _agg["volume"] = _agg["cell_volume"] * _agg["share"]
                            _proj = _project_capped(_prop_items_from_gran(_explode(_agg)))
                            _vol = _cvol * np.asarray(_anchor, float)
                            _midv = np.array([float(_vol[r].sum()) for r in _mid_rows])
                            _bands = []
                            for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                if _mtr == "vamp_pct" or _mk not in _mid_index:
                                    continue
                                _mi = _mid_index[_mk]
                                _months = [int(_mo)] if _mo is not None else list(range(4))
                                _bix = 1 if _mtr == "txn" else 0
                                _true = float(sum(_proj.get((_mk, m), (0.0, 0.0))[_bix] for m in _months))
                                if _true <= 0:
                                    continue
                                # fitness proxy: proj(x) = _bval × (MID_vol(x) / mid_base_vol). Back-solve
                                # _bval so the proxy EQUALS the true projection at this anchor (exact here,
                                # first-order accurate nearby) — this is the calibration.
                                _rat = (_midv[_mi] / _mid_bvol[_mi]) if _mid_bvol[_mi] > 1e-12 else 1.0
                                _bval = (_true / _rat) if _rat > 1e-9 else _true
                                _tolb = float(_tl) if _tl is not None else 0.0
                                # constraint TYPE: ceiling → no floor; floor → no ceiling; range → both.
                                _ceilb = (_tg * (1.0 + _tolb)) if _dir in ("range", "ceiling") else None
                                _floorb = (_tg * (1.0 - _tolb)) if (_dir in ("range", "floor") and _tolb < 1.0) else 0.0
                                _vmul = _VAMP_VAR_MULT if _mtr == "vamp" else 1.0   # VAMP harder
                                _pmul = _prio_mult(_prio_lookup.get((_mk, _mo, _mtr), 1))   # priority weight
                                # skip a rule that ends up with NO active edge (shouldn't happen)
                                if _ceilb is None and not (_floorb and _floorb > 0):
                                    continue
                                _bands.append((int(_mi), float(_bval),
                                               (float(_ceilb) if _ceilb is not None else None),
                                               (float(_floorb) if _floorb > 0 else None),
                                               float(_vmul), float(_pmul)))
                            return _bands

                        _ga_bands = []
                        if _mid_month_rules:
                            try:
                                _ga_bands = _build_ga_bands(_ref_share_G)
                                if _ga_bands:
                                    log(f"   GA fitness: {len(_ga_bands)} month-specific per-MID band(s) folded "
                                        "into the search (calibrated volume-ratio proxy + re-projection "
                                        "correction; vamp_pct left to post-enforcement).")
                            except Exception as _e:  # noqa: BLE001
                                log(f"   [Warning] could not fold per-MID bands into GA fitness ({_e}); "
                                    "post-enforcement only.")
                                _ga_bands = []
                        # Per-MID ROUTING-space VAMP floor for the dial-0 risk-min CLAMP: the risk-min
                        # term won't push a MID's VAMP below this, so the two-sided VAMP bands stay
                        # satisfiable at dial 0 (keeps the ranges, doesn't overshoot the lower edge).
                        # Derived from each VAMP band floor via reference proportionality. 0 = none.
                        _vfloor_route = np.zeros(len(_mids_u))
                        if _mid_month_rules:
                            try:
                                _agg_ref = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                                _agg_ref["share"] = _ref_share_G
                                _agg_ref["volume"] = _agg_ref["cell_volume"] * _ref_share_G
                                _bp_ref = _project_capped(_prop_items_from_gran(_explode(_agg_ref)))
                                _midvr_ref = np.array([float((_cvol[r] * _ref_share_G[r] * _rkr[r]).sum())
                                                       for r in _mid_rows])
                                for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                    # only VAMP constraints that HAVE a lower edge (range/floor) clamp
                                    # the dial-0 risk-min; a pure ceiling has no floor to protect.
                                    if _mtr != "vamp" or _mk not in _mid_index or _dir not in ("range", "floor"):
                                        continue
                                    _tolb = float(_tl) if _tl is not None else 0.0
                                    if _tolb >= 1.0:
                                        continue
                                    _mi = _mid_index[_mk]
                                    _months = [int(_mo)] if _mo is not None else list(range(4))
                                    _bref = float(sum(_bp_ref.get((_mk, m), (0.0, 0.0))[0] for m in _months))
                                    if _bref <= 0:
                                        continue
                                    _vfloor_route[_mi] = max(_vfloor_route[_mi],
                                                             _midvr_ref[_mi] * (_tg * (1.0 - _tolb) / _bref))
                            except Exception as _e:  # noqa: BLE001
                                log(f"   [Warning] VAMP floor-route calc failed ({_e}); dial-0 risk-min unclamped.")
                                _vfloor_route = np.zeros(len(_mids_u))
                        # --- ELIGIBILITY IN THE SEARCH (optional, ROUTING_GA_ELIG=0 to disable) ---
                        # Fold the SAME bans + wallet/USA-only capability enforcement applies
                        # (apply_restrictions) into a static per-row operator so the GA SCORES the
                        # actually-routable shares, instead of a split eligibility later perturbs.
                        # Built on G's exact (sorted, contiguous) row order, so its cell segments match
                        # cell_starts; it reproduces apply_restrictions row-for-row (proven to ~2e-16).
                        # Applied ONLY in the scoring path (_obj_viol/_mid_over) — the returned best stays
                        # the RAW decode so enforcement blends exactly once (wallet/USA blend isn't
                        # idempotent). No-op when no eligibility is configured or the build fails.
                        # DEFAULT ON (set ROUTING_GA_ELIG=0 to disable). The GA scores the ACTUALLY-
                        # routable shares every generation, so it optimises what actually gets deployed.
                        # The genetic_numba kernel implements the eligibility stage (numba_kernels.
                        # _fused_eval), so this KEEPS the fast Numba kernel — verify() still cross-checks
                        # kernel-vs-NumPy including eligibility (and, under the fail-loud policy, a
                        # kernel⇄NumPy eligibility mismatch now RAISES rather than silently reverting).
                        # The returned best stays the RAW decode, so end-of-run enforcement still blends
                        # eligibility exactly once.
                        _elig_op = None
                        if (os.environ.get("ROUTING_GA_ELIG", "1") != "0"
                                and (_elig_rules or _wallet_incapable or _usa_only)):
                            try:
                                from routing_optimiser.eligibility import build_elig_operator as _build_elig_op
                                _rpgt_col = (G["rpgt"].astype(str).to_numpy() if "rpgt" in G.columns
                                             else np.array([str(c).split("|")[-1] for c in _cellk]))
                                _cells_layout = pd.DataFrame({
                                    "cell": _cellk,
                                    "gateway": G["gateway"].astype(str).to_numpy(),
                                    "currency": G["currency"].astype(str).to_numpy(),
                                    "bank": G["bank"].astype(str).to_numpy(),
                                    "rpgt": _rpgt_col,
                                })
                                _elig_op = _build_elig_op(
                                    _cells_layout, _elig_rules, _fid2vamp_l,
                                    wallet_incapable=frozenset(_wallet_incapable), wallet_frac=_wallet_frac,
                                    wallet_default=_wallet_default, usa_only=frozenset(_usa_only),
                                    nonusa_frac=_nonusa_frac, nonusa_default=_nonusa_default)
                                log(f"   GA scores ELIGIBILITY-ADJUSTED shares: {int(_elig_op['ban'].sum())} banned "
                                    f"row(s), wallet={'on' if _elig_op['has_w'] else 'off'}, "
                                    f"USA-only={'on' if _elig_op['has_u'] else 'off'} (returned split is RAW; "
                                    "enforcement blends once). Set ROUTING_GA_ELIG=0 to disable.")
                            except Exception as _ee:  # noqa: BLE001
                                _elig_op = None
                                log(f"   [Warning] GA eligibility operator build failed ({type(_ee).__name__}: {_ee}) "
                                    "— GA scores unrestricted; eligibility still applied downstream in enforcement.")
                        # AUTO-BLOCK folded into the GA eligibility (QUALITY, not speed). A bank-blocked
                        # (bank, gateway) — ≥N most-recent consecutive failed attempts — is marked
                        # INELIGIBLE for the search, so the GA never routes real volume to a door the bank
                        # has closed and instead optimises the redistribution itself, rather than the
                        # post-hoc enforcement cap doing it afterward. Same detector + parent-bank keying
                        # as the enforcement cap. GUARD: a cell whose gateways are ALL blocked is left
                        # untouched (nowhere to move the volume — mirrors _apply_blocked_caps). Enforcement
                        # still caps to the floor as a safety net. Only ~0.1% of rows, so no material speed
                        # effect. NOTE: excluded rows get 0 share in the search (vs the exploration floor
                        # under the old post-hoc cap) — treating a dead door as unroutable, by design.
                        _ga_elig = np.ones(len(G))
                        if bool(ss.get("block_gw_cb", False)):
                            try:
                                _bapre_ga = orig_adf.copy()
                                # THIS run's BIN->parent map. ss["bin_to_bank"] is only written
                                # post-GA (below), so reading ss here would use LAST run's map (or
                                # empty on a session's first run) — the staleness that flipped the
                                # blocked set 32<->13 between runs. Use the current-run local.
                                _b2b_ga = bin_to_bank or {}
                                if _b2b_ga and "bank" in _bapre_ga.columns:
                                    _bapre_ga["bank"] = _bapre_ga["bank"].map(
                                        lambda _b: _b2b_ga.get(_b, _b2b_ga.get(str(_b), _b)))
                                _bdf_ga = detect_blocked_gateways(
                                    _bapre_ga, float(ss.get("block_min_inp", 100) or 100))
                                _bflag_ga = _bdf_ga[_bdf_ga["blocked"]] if not _bdf_ga.empty else _bdf_ga
                                _blk_ga = set(zip(
                                    _bflag_ga["bank"].astype(str).str.strip().str.lower(),
                                    _bflag_ga["gateway"].astype(str).str.strip().str.lower()))
                                if _blk_ga:
                                    _row_blk = np.array([(b, g) in _blk_ga
                                                         for b, g in zip(_pb_l, _gw_l)], dtype=bool)
                                    # per-cell counts: exclude blocked rows ONLY in cells that still keep a
                                    # non-blocked gateway (a fully-blocked cell is left for enforcement).
                                    _blk_per_cell = np.add.reduceat(_row_blk.astype(float), _cell_starts)
                                    _cell_mixed = (_blk_per_cell > 0) & (_blk_per_cell < _counts.astype(float))
                                    _excl = _row_blk & np.repeat(_cell_mixed, _counts)
                                    _n_excl = int(_excl.sum())
                                    # Do NOT hard-exclude these routes from the search. With EXACT in-search
                                    # band scoring, removing a blocked route can starve a cross-cell per-MID
                                    # band and force a hard infeasibility (observed 2026-08-04: 56 rows
                                    # excluded → all 8 seeds infeasible, violation 0.001913). Blocked
                                    # gateways are still capped to the exploration floor in the DELIVERED
                                    # split by the post-GA _apply_blocked_caps pass (feasibility-safe: caps
                                    # only where a non-blocked recipient exists), so the OUTPUT is ~unchanged
                                    # — a dead gateway lands at <=floor either way. (_ga_elig left all-eligible.)
                                    if _n_excl:
                                        log(f"   auto-block (pre-GA): {len(_blk_ga)} bank×gateway pair(s) → "
                                            f"{_n_excl} blocked row(s) KEPT eligible for the search, capped "
                                            "post-GA instead (hard-exclusion removed so exact MID bands can't "
                                            "be starved into infeasibility).")
                            except Exception as _bge:  # noqa: BLE001 — never break the run over this
                                log(f"   [Warning] pre-GA auto-block skipped ({type(_bge).__name__}: {_bge}); "
                                    "enforcement-time cap still applies.")
                        # HARD PRE-SEARCH BAN MASK: bake config bans (routing_restrictions) into the GA
                        # eligibility so the decode NEVER assigns a banned gateway any share — not merely
                        # scored-around and stripped at the end. Wallet/USA stay as the in-search fractional
                        # reroute (they are per-traffic-type capability rules, not whole-gateway exclusions,
                        # so a hard 0 would wrongly drop the capable traffic they ARE allowed to serve).
                        # Sourced from the eligibility operator's per-row ban mask; no-op if eligibility is
                        # disabled (ROUTING_GA_ELIG=0 — bans then still enforced at delivery via _restrict)
                        # or nothing is banned.
                        if _elig_op is not None and _elig_op.get("ban") is not None:
                            _ban_mask = np.asarray(_elig_op["ban"], float) > 0.5
                            if bool(_ban_mask.any()):
                                _ga_elig = np.asarray(_ga_elig, float) * (~_ban_mask).astype(float)
                                try:
                                    _seg_e = np.add.reduceat(_ga_elig, np.asarray(_cell_starts, np.intp))
                                    _dead = int((_seg_e <= 0.0).sum())
                                except Exception:  # noqa: BLE001
                                    _dead = 0
                                log(f"   hard pre-search ban mask: {int(_ban_mask.sum())} banned row(s) "
                                    "set elig=0 (the search never routes to them)" +
                                    (f"; ⚠ {_dead} cell(s) now have NO eligible gateway and will route "
                                     "nothing — check the ban config." if _dead else "."))
                        ctx = {
                            "n_row": len(G), "n_mid": len(_mids_u),
                            "cell_starts": _cell_starts, "cell_counts": _counts,
                            "elig": _ga_elig,   # config BANS hard-masked pre-search (share 0 throughout); bank-blocked rows kept eligible + capped post-GA; wallet/USA are the in-search fractional reroute (elig_op) + final enforcement
                            "elig_op": _elig_op,       # optional: GA scores eligibility-adjusted (routable) shares
                            "base": _base, "cell_vol": _cvol, "sr": _srr, "risk": _rkr, "ticket": _tick,
                            "rev_coef": _rev_coef,
                            "mid_id": _mid_id, "mid_rows": _mid_rows,
                            "vamp_cap": (float(vamp_cap) if vamp_cap is not None else None),
                            "mid_vol_cap": None,   # DROPPED — per-MID rules live in the month bands
                            "midband": (_ga_bands or None),   # month-specific per-MID bands in fitness
                            "vamp_floor_route": _vfloor_route,   # dial-0 risk-min clamp (per-MID VAMP floor)
                            "mid_base_vol": _mid_bvol,         # reference MID volume (for the ratio proxy)
                            "mid_ticket": _mid_tick, "mid_sr": _mid_srm,
                            "shape_mult": 10.0, "max_share": float(max_share), "floor": float(floor),
                            "breach_fixed": 0.0,   # cap-breach penalty removed — fixed at 0 (no cap penalty)
                            # CMA-ES engine: lean the θ=0 reference gently toward lower risk (γ,
                            # dimensionless) so freed volume redistributes to LOW-risk recipients and
                            # the base is already slightly compliant. 0 = no lean (unbiased reference).
                            "ref_gamma": float(ss.get("ga_ref_gamma", 0.25) or 0.0),
                        }
                        # CROSS-CELL per-MID tilt search — CMA-ES (2026-07-25 rebuild of the old GA).
                        # Genome per vampMid = [θr risk-tilt | θq revenue-tilt | g gain] (3·n_mid dims),
                        # shifting a MID toward its LOW-risk / HIGH-revenue cells and moving its overall
                        # presence — directly controlling the per-MID CROSS-cell VAMP rate (the actual
                        # constraint). CMA-ES ranks feasibility-FIRST (compliant always beats breaching;
                        # among compliant, higher revenue wins), with a smooth (no fixed-step) breach
                        # measure, reseeded restarts + a Nelder–Mead polish. Freed share redistributes
                        # to low-risk, revenue-efficient recipients (γ-leaned reference). Tiny/fast
                        # (seconds). dial 100 = softmax revenue reference; dial 0 = this search's best
                        # compliant split; blended, both endpoints hard-enforced (2 VAMP solves).
                        ctx["ref_share"] = _ref_share_G          # θ=0 decode base (revenue-optimal)
                        # "Run all generations" toggle: disables the CMA-ES convergence early-stop so
                        # every restart runs the full generation cap (exact candidate count, longer).
                        ctx["no_early_stop"] = bool(ss.get("ga_no_early_stop", False))
                        # UI step-size controls (default to no-ops so behaviour is unchanged unless dialled):
                        # σ₀ multiplier (starting stride), σ floor (don't let σ collapse), damping × (adapt σ slower).
                        ctx["sigma0_mult"] = float(ss.get("ga_sigma0_mult", 1.5) or 1.5)
                        ctx["sigma_floor"] = float(ss.get("ga_sigma_floor", 0.0) or 0.0)
                        ctx["damps_mult"] = float(ss.get("ga_damps_mult", 1.5) or 1.5)

                        # ---- EXACT per-generation band scoring (gate 2) ------------------------------
                        # Replace the volume-ratio band PROXY with the exact pro-rata projection, scored
                        # once per generation for the whole population (band_scoring.ExactBandPenalty on
                        # numba-projected values). Requires pre-clustering OFF so the search runs at
                        # parent-bank grain and the column→prop-key map is a clean parent→BIN replicate.
                        # PERMANENT default (no toggle); degrades to the proxy on ANY failure. The delivered
                        # split gets a one-shot self-check that the incidence reproduces the TRUE
                        # _prop_items_from_gran (which also folds in the backup catch-all blend) — if that
                        # gap is not ~0, exact bands are NOT trustworthy (blend/grain drift) and the log says so.
                        if _ga_bands and _mid_month_rules:
                            try:
                                import scipy.sparse as _spx
                                from routing_optimiser.genetic_global import run_midtilt_ga as _plain_ga
                                from routing_optimiser.band_scoring import ExactBandPenalty as _EBP, BandSpec as _BSpec
                                _pbp_x = _get_pbp()
                                if _pbp_x is None:
                                    raise RuntimeError("no cap scaffold — exact bands need the pro-rata projection")
                                _byr = bool(_opt_by_rpgt)
                                _rpgt_g = (G["rpgt"].astype(str).str.strip().str.lower().tolist()
                                           if "rpgt" in G.columns else [""] * len(G))
                                # parent(bank) → its BIN-level banks, per (currency, rpgt) [explode replicate]
                                _of = orig_forecast[["currency", "bank", "rpgt"]].drop_duplicates().copy()
                                _of["_cur"] = _of["currency"].astype(str).str.strip().str.lower()
                                _of["_bin"] = _of["bank"].astype(str).str.strip()
                                _of["_rk"] = _of["rpgt"].astype(str).str.strip().str.lower()
                                _of["_pb"] = _of["bank"].map(
                                    lambda b: bin_to_bank.get(b, bin_to_bank.get(str(b).strip().lower(), b))
                                ).astype(str).str.strip().str.lower()
                                _bins_by = {}
                                for _cx, _pbx, _rkx, _binx in zip(_of["_cur"], _of["_pb"],
                                                                  _of["_rk"], _of["_bin"]):
                                    _bins_by.setdefault((_cx, _pbx, _rkx), []).append(_binx)
                                _kpos = {str(k): i for i, k in enumerate(_pbp_x.prop_keys)}
                                _rows = []; _cols = []
                                for _j in range(len(G)):
                                    _vm = fid2vamp.get(_gw_l[_j])
                                    if _vm is None:
                                        continue
                                    _vm = str(_vm).strip()
                                    for _bin in _bins_by.get((_cur_l[_j], _pb_l[_j], _rpgt_g[_j]), ()):
                                        _pk = (f"{_cur_l[_j]}|{_bin}|{_rpgt_g[_j]}|{_vm}" if _byr
                                               else f"{_cur_l[_j]}|{_bin}|{_vm}")
                                        _i = _kpos.get(_pk)
                                        if _i is not None:
                                            _rows.append(_i); _cols.append(_j)
                                _inc = _spx.csr_matrix((np.ones(len(_rows)), (_rows, _cols)),
                                                       shape=(max(len(_pbp_x.prop_keys), 1), len(G)))
                                # specs from the rules (weight = pmul; wm≡1 since viol_vol_weight is off)
                                _specs = []
                                for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                    if _mtr not in ("txn", "vamp") or _mk not in _mid_index:
                                        continue
                                    _months = tuple(int(_mo) for _mo in ([int(_mo)] if _mo is not None else range(6)))
                                    _tolb = float(_tl) if _tl is not None else 0.0
                                    _ceilb = (_tg * (1.0 + _tolb)) if _dir in ("range", "ceiling") else None
                                    _floorb = (_tg * (1.0 - _tolb)) if (_dir in ("range", "floor") and _tolb < 1.0) else None
                                    if _ceilb is None and not (_floorb and _floorb > 0):
                                        continue
                                    _pmulx = _prio_mult(_prio_lookup.get((_mk, _mo, _mtr), 1))
                                    _specs.append(_BSpec(midl=str(_mk).strip().lower(), months=_months,
                                                         metric=_mtr, ceil=_ceilb, floor=_floorb, weight=float(_pmulx)))
                                _epx = _EBP(_pbp_x, _specs,
                                            breach_fixed=float(ctx.get("breach_fixed", 0.0) or 0.0),
                                            breach_quad=float(ctx.get("breach_quad", 1.0) or 1.0),
                                            breach_shape=str(ctx.get("breach_shape", "quadratic")))
                                ctx["exact_bands"] = _epx
                                ctx["band_incidence"] = _inc
                                ctx["_exact_bands_selfcheck"] = {"inc": _inc, "done": False}
                                _run_midtilt_ga = _plain_ga            # pre-clustering OFF for exact bands
                                log(f"   GA fitness: EXACT per-generation band scoring ON — {len(_specs)} band(s), "
                                    f"pre-clustering OFF, incidence {_inc.shape[0]}×{_inc.shape[1]} ({_inc.nnz} nnz). "
                                    "Delivered split gets an incidence self-check (must read ~0 to trust).")
                            except Exception as _ebe:  # noqa: BLE001
                                # NO proxy fallback (removed per config): exact band scoring is mandatory.
                                # Crash loudly so a broken setup is never silently downgraded to the proxy.
                                log(f"   ✗ EXACT band scoring setup FAILED ({type(_ebe).__name__}: {_ebe}). "
                                    "Proxy fallback is DISABLED — aborting the run. Clear __pycache__ + "
                                    "restart; if it persists, paste this line and the traceback.")
                                raise

                        ss["_ga_ctx"] = ctx                      # stash for the experimental NSGA-II / full-matrix explorer
                        _n_cells = int(len(_counts)); _n_mid = int(len(_mids_u))
                        # ── GA problem size (surfaced so the k-means pre-clustering decision is data-driven) ──
                        # `cells` = rpgt×currency×bank problems the search steers; genome D = 3·vampMids
                        # (a risk-tilt, a revenue-tilt and a gain per MID). CMA-ES per-generation cost grows
                        # with D² (rank-µ covariance update) and D³ at each periodic eigen-refresh — so D,
                        # NOT the cell count, is what k-means pre-clustering would actually shrink. Small D
                        # ⇒ clustering is a rounding error; large D ⇒ the cube term makes it a real win.
                        _ga_D = 3 * _n_mid
                        log(f"   GA problem size: {_n_cells} cells (rpgt×currency×bank), {int(len(G))} "
                            f"cell×gateway rows, {_n_mid} vampMids → genome D = {_ga_D}. "
                            f"CMA-ES per-generation cost ∝ D² (covariance update) and ∝ D³ (eigen-refresh); "
                            f"D is the dimension k-means pre-clustering would reduce.")
                        # ── Compressibility probe (READ-ONLY; changes nothing) — the k-means ceiling ──
                        # The 48-knob genome decodes each cell's shares from its gateways' (vampMid,
                        # reference share, risk) alone, so two cells whose eligible gateways carry the SAME
                        # signature decode IDENTICALLY under EVERY genome and could share one representative
                        # (carrying their summed volume) at ~no loss. Counting DISTINCT signatures gives the
                        # rough CEILING on how far pre-clustering could shrink the 58,745-row hot loop. This
                        # ONLY measures. Optimistic (a truly lossless merge also needs matching success /
                        # ticket rates); a real clusterer trades a controllable error for more compression.
                        try:
                            _mid_a = np.asarray(ctx["mid_id"])
                            _ref_a = np.asarray(ctx["ref_share"], float)
                            _risk_a = np.asarray(ctx["risk"], float)
                            _elig_a = np.asarray(ctx["elig"], float) > 0.5
                            _sig_all = set()
                            _single = 0
                            for _ci in range(len(_counts)):
                                _s0 = int(_cell_starts[_ci]); _s1 = _s0 + int(_counts[_ci])
                                if (_s1 - _s0) <= 1:
                                    _single += 1
                                _sig_all.add(tuple(sorted(
                                    (int(_mid_a[g]), round(float(_ref_a[g]), 5), round(float(_risk_a[g]), 6))
                                    for g in range(_s0, _s1) if _elig_a[g])))
                            _distinct = max(1, len(_sig_all))
                            _ratio = _n_cells / _distinct
                            log(f"   Compressibility probe: {_n_cells} cells → ~{_distinct} distinct "
                                f"decode-signatures (≈{_ratio:.1f}× cell-reduction CEILING); {_single} "
                                f"single-gateway cells are genome-invariant. Upper bound on what k-means "
                                f"pre-clustering could trim from the {int(len(G))}-row hot loop — optimistic "
                                f"(true lossless merge also needs matching success/ticket rates).")
                        except Exception as _e:  # noqa: BLE001 — a read-only probe must never break a run
                            log(f"   [Warning] compressibility probe skipped ({type(_e).__name__}: {_e}).")
                        _pop_ovr = int(ss.get("ga_pop_override", 0) or 0)   # 0 = auto-size
                        _ga_pop = _pop_ovr if _pop_ovr > 0 else int(np.clip(round(4 * _n_mid), 30, 80))
                        _ga_gen = int(ss.get("ga_generations", 80) or 80)
                        _ga_pat = 12
                        # multi-seed: keep the fittest of N parallel CMA-ES starts. N is the tab-2
                        # "Number of seeds" control (defaults to core count); _GA_N_SEED is the fallback.
                        _N_SEED = max(1, int(ss.get("ga_n_seeds", _GA_N_SEED) or _GA_N_SEED))
                                              # (module constant, also read by the settings-aware ETA)
                        _GA_GAIN_MAX = 3.5   # wider per-MID gain range (was 2.0) → more cross-MID reach
                        # CMA-ES self-adapts (covariance + step size) and ranks feasibility-first, so the
                        # legacy GA knobs (breach-targeted mutation, smart init, adaptive λ) no longer
                        # apply — run_midtilt_ga accepts them for compatibility and ignores them.
                        _rev_of = lambda _sh: float((np.asarray(_sh, float) * _rev_coef).sum())
                        log(f"   CMA-ES (cross-cell per-MID 3-axis tilt, {_n_mid} vampMids): λ={_ga_pop}, gen cap={_ga_gen} "
                            "(covariance-adapted + reseeded restarts + polish). dial 0 = risk-MINIMISED compliant; "
                            "dial 99 = max-revenue compliant; blended between (monotonic frontier); dial 100 = uncapped revenue ceiling.")
                        # [FN-339]
                        def _ga_true_breach(_sh):
                            """Total RELATIVE band breach of the TRUE pro-rata projection for `_sh`
                            (0 ⇒ every month band satisfied by the REAL projection, not the proxy)."""
                            if not _mid_month_rules:
                                return 0.0
                            _agg = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                            _agg["share"] = np.asarray(_sh, float)
                            _agg["volume"] = _agg["cell_volume"] * _agg["share"]
                            _tb_prop = _prop_items_from_gran(_explode(_agg))
                            # GATE-1 (one-shot): collapsed-vs-true band gap on the delivered split.
                            # Placed here so it fires even when per-MID enforcement is bypassed.
                            _band_collapse_diag(_tb_prop, "delivered split (re-projection)")
                            # EXACT-BANDS incidence self-check (one-shot): does the column→prop-key
                            # incidence used IN the search reproduce the TRUE _prop_items_from_gran
                            # (which also folds the backup catch-all blend)? Non-zero ⇒ don't trust it.
                            _scx = ctx.get("_exact_bands_selfcheck") if isinstance(ctx, dict) else None
                            _epx_sc = ctx.get("exact_bands") if isinstance(ctx, dict) else None
                            if _scx is not None and _epx_sc is not None and not _scx["done"]:
                                _scx["done"] = True
                                try:
                                    from routing_optimiser.band_scoring import shares_to_prop_raw as _s2pr_chk
                                    _projx = _epx_sc.projector
                                    _kpx = {str(k): i for i, k in enumerate(_projx.prop_keys)}
                                    _pr_inc = _s2pr_chk(np.asarray(_sh, float)[None, :], _scx["inc"])[0]
                                    _truth = np.zeros(len(_projx.prop_keys), dtype=float)
                                    for _t in _tb_prop:
                                        _k = ("|".join([str(_t[0]).strip().lower(), str(_t[1]).strip(),
                                                        str(_t[2]).strip().lower(), str(_t[3]).strip()])
                                              if _opt_by_rpgt else
                                              "|".join([str(_t[0]).strip().lower(), str(_t[1]).strip(),
                                                        str(_t[2]).strip()]))
                                        _ix = _kpx.get(_k)
                                        if _ix is not None:
                                            _truth[_ix] += float(_t[-1])
                                    _gap = float(np.abs(_pr_inc - _truth).max()) if len(_truth) else 0.0
                                    log(f"   ── exact-bands incidence self-check ── max|prop_raw(incidence) − "
                                        f"truth| = {_gap:.3e}  (Σtruth={_truth.sum():.1f}; ≈0 ⇒ the search's "
                                        "column→prop-key map matches _prop_items_from_gran incl. the blend)")
                                except Exception as _sce:  # noqa: BLE001
                                    log(f"   [exact-bands self-check skipped: {type(_sce).__name__}: {_sce}]")
                            _band_cost_probe()          # GATE-2 (one-shot): reduced-size + project_pop timing
                            _band_compress_probe()      # GATE-2 (one-shot): lossless compression ceiling
                            if ss.get("ga_band_slope", False) or ss.get("ga_band_enablers", False):
                                _agg_r = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                                _agg_r["share"] = G["_ref_share"].to_numpy(float)
                                _agg_r["volume"] = _agg_r["cell_volume"] * _agg_r["share"]
                                _ref_prop = _prop_items_from_gran(_explode(_agg_r))
                                _band_slope_probe(_ref_prop, _tb_prop)          # GATE-2 slope-accuracy
                                _band_enabler_probes(_ref_prop, _tb_prop)       # GATE-2 pruning + local-linear
                            _proj = _project_capped(_tb_prop)
                            _tot = 0.0
                            for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                if _mtr == "vamp_pct" or _mk not in _mid_index:
                                    continue
                                _months = [int(_mo)] if _mo is not None else list(range(4))
                                _bix = 1 if _mtr == "txn" else 0
                                _v = float(sum(_proj.get((_mk, m), (0.0, 0.0))[_bix] for m in _months))
                                _tolb = float(_tl) if _tl is not None else 0.0
                                _hi = _tg * (1.0 + _tolb) if _dir in ("range", "ceiling") else None
                                _lo = _tg * (1.0 - _tolb) if _dir in ("range", "floor") else 0.0
                                if _hi is not None and _v > _hi:
                                    _tot += _v / _hi - 1.0
                                elif _lo > 0 and _v < _lo:
                                    _tot += 1.0 - _v / _lo
                            return float(_tot)

                        # [FN-340]
                        def _ga_solve_with_correction(_risk_min_w, _seed=42, _rounds=1, _band_w=8.0,
                                                      _warm=None, _band_fix=20.0, _ref_gamma=None,
                                                      _n_fine=0, _n_restarts=None):
                            """Run the tilt GA, then RE-PROJECT & CORRECT (like the greedy): re-anchor
                            the band proxy at the GA's own split via the TRUE projection and re-run,
                            accepting a round ONLY if the true-projection band breach actually drops.
                            Bounded (≤ _rounds) and no-regression. `_band_w` scales the per-MID band
                            penalty (tougher at the dial-0 risk-min endpoint). `_warm` seeds a prior
                            run's genome into the population (free reach). `_ref_gamma` leans the θ=0
                            reference toward compliance PER ENDPOINT (≈0 at revenue-max so the revenue
                            ceiling isn't taxed; larger at risk-min). Returns (shares, info)."""
                            _rg_default = ctx.get("ref_gamma", 0.25)
                            ctx["risk_min_w"] = float(_risk_min_w)
                            ctx["band_weight"] = float(_band_w)
                            ctx["band_fixed"] = float(_band_fix)
                            if _ref_gamma is not None:
                                ctx["ref_gamma"] = float(_ref_gamma)   # #8 per-endpoint compliant lean
                            # Per-endpoint search knobs: richer per-cell genome (_n_fine) and extra
                            # restarts/budget — used at the risk-min end where the extra reach pays.
                            _extra_kw = {"n_fine": int(_n_fine)}
                            if _n_restarts is not None:
                                _extra_kw["n_restarts"] = int(_n_restarts)
                            if engine_key == "genetic_numba":
                                # opt-in Numba fused eval; the GA self-verifies vs NumPy and FAILS
                                # LOUDLY (raises with full diagnostics) on any mismatch/exception —
                                # it no longer falls back to NumPy. Forwarded to every run_midtilt_ga
                                # call below via **_extra_kw.
                                _extra_kw["numba"] = True
                            # Restart strategy (both engines): lean constant-λ + coordinated-coverage
                            # is the DEFAULT; IPOP (λ-doubling) only when explicitly chosen. Forwarded
                            # to every run_midtilt_ga call below.
                            _extra_kw["restart_mode"] = (
                                "ipop" if str(ss.get("ga_restart_mode", "")).startswith("IPOP")
                                else "lean")
                            try:
                                ctx["midband"] = (_build_ga_bands(_ref_share_G) or None)
                            except Exception as _e:  # noqa: BLE001
                                log(f"   [Warning] GA band build failed ({_e}); proxy bands off this run.")
                                ctx["midband"] = None
                            # MULTI-SEED: run a few random seeds and keep the fittest (elitism per
                            # seed + the warm-start make each cheap). Guards against an unlucky path.
                            # The seeds are INDEPENDENT and each fully DETERMINISTIC (seed=_seed+_s),
                            # so run them in parallel PROCESSES via joblib's loky backend (robust on
                            # macOS + Windows spawn, same as the compression stage). Results are
                            # consumed in seed order and the fittest kept with the SAME strictly-
                            # greater / first-wins tie-break as the sequential loop, so the outcome is
                            # byte-identical. ANY failure (or a single seed) → the sequential loop,
                            # also byte-identical. ctx is read-only inside the GA, so pickling a copy
                            # to each worker is safe.
                            _seed_results = None
                            # The multi-seed GA ALWAYS runs on loky worker processes (true multi-core).
                            # There is no backend selector and no threading/sequential fallback: if the
                            # loky pool can't run, the run fails loudly (see the `except` below).
                            if int(_N_SEED) > 1:
                                import time as _st_t
                                from joblib import Parallel, delayed
                                import inspect as _insp_jl
                                try:                        # older joblib lacks inner_max_num_threads
                                    _JL_INNER_OK = ("inner_max_num_threads" in
                                                    _insp_jl.signature(Parallel).parameters)
                                except Exception:  # noqa: BLE001
                                    _JL_INNER_OK = False
                                # Backend: loky (processes) gives TRUE multi-core parallelism because
                                # the CMA-ES holds the GIL a lot between numpy calls. Each seed is
                                # INDEPENDENT and fully DETERMINISTIC (seed=_seed+_s), and results are
                                # consumed in seed order. loky needs picklable args (ctx + the
                                # now-picklable stop_check); large ctx arrays are auto-memmapped by
                                # joblib, and inner_max_num_threads=1 stops the workers × BLAS-threads
                                # oversubscription. loky is the ONLY supported backend — if it can't
                                # run, the run fails loudly (no threading / sequential fallback).
                                _ga_backend = "loky"
                                _try_backends = ["loky"]
                                _njobs = min(int(_N_SEED), os.cpu_count() or 1)
                                # Can we stream results as each seed returns? (joblib >=1.3 has
                                # return_as). If so we log a per-seed convergence summary the moment
                                # each finishes; the generator is ORDERED so the fittest tie-break
                                # stays byte-identical to the blocking call. Older joblib → blocking.
                                try:
                                    _gen_ok = ("return_as" in _insp_jl.signature(Parallel).parameters)
                                except Exception:  # noqa: BLE001
                                    _gen_ok = False

                                # [FN-341]
                                def _log_seed(_idx, _infoc, _t0, _best_holder):
                                    """Verbose one-line summary for a finished seed (best-effort)."""
                                    try:
                                        _fit = float(_infoc.get("best_fit", float("nan")))
                                        _i0 = float(_infoc.get("init_fit", _fit))
                                        _feas = bool(_infoc.get("feasible", False))
                                        _viol = float(_infoc.get("violation", 0.0))
                                        _gns = int(_infoc.get("gens", 0))
                                        _gmax = int(_infoc.get("gens_max", 0))
                                        _sig = float(_infoc.get("sigma_final", 0.0))
                                        _es = bool(_infoc.get("early_stopped", False))
                                        _tag = ""
                                        if _best_holder[0] is None or _fit > _best_holder[0]:
                                            _best_holder[0] = _fit; _tag = "  ← best so far"
                                        log(f"      • seed {_idx}/{int(_N_SEED)} finished "
                                            f"(t+{_st_t.time() - _t0:.0f}s): fitness {_i0:,.4g}→{_fit:,.4g}, "
                                            f"feasible={_feas}, violation={_viol:.4g}, "
                                            f"{_gns}/{_gmax} gens{' (early-stop)' if _es else ''}, "
                                            f"σ_final={_sig:.3g}{_tag}")
                                    except Exception:  # noqa: BLE001
                                        pass

                                # LIVE PROGRESS: give each seed a picklable writer that records its
                                # running candidate-eval count to its own file; run the (blocking)
                                # Parallel in a BACKGROUND THREAD so THIS thread stays free to poll
                                # those files and log an aggregate "candidate splits evaluated so far"
                                # while the search runs. All best-effort — any failure cascades to the
                                # next backend / sequential path, byte-identical to before.
                                import concurrent.futures as _cf, shutil as _shutil
                                try:
                                    from routing_optimiser.run_bundle import _ProgressWriter as _PW
                                except Exception:  # noqa: BLE001
                                    _PW = None
                                # CRITICAL: keep the per-seed progress files on a LOCAL disk, NOT inside
                                # the project (which lives under a cloud-synced/FUSE mount). The loky
                                # WORKER processes write these files and the MAIN process (poller) reads
                                # them; on FUSE, cross-process writes don't propagate promptly, so the
                                # poller reads empty files, reports 0 candidates, and prints NO live
                                # progress — the run looks dead for 25 min even though the seeds are
                                # running. The OS temp dir is a real local fs with immediate visibility.
                                import tempfile as _tf_prog
                                _prog_dir = os.path.join(_tf_prog.gettempdir(), "routing_optimiser_gaprog")
                                try:
                                    _shutil.rmtree(_prog_dir, ignore_errors=True)
                                    os.makedirs(_prog_dir, exist_ok=True)
                                except Exception:  # noqa: BLE001
                                    _prog_dir = None
                                _writers = ([_PW(os.path.join(_prog_dir, f"seed_{_s}.txt"))
                                             for _s in range(int(_N_SEED))]
                                            if (_PW is not None and _prog_dir) else [None] * int(_N_SEED))
                                _restarts_est = int(_extra_kw.get("n_restarts", 2) or 2)
                                _total_cand = max(1, int(_N_SEED) * _restarts_est * int(_ga_gen) * int(_ga_pop))

                                # [FN-342]
                                def _poll_progress(_t0, _emit, _nseed):
                                    """Sum the per-seed progress files and surface the live count in BOTH
                                    the visible headline (so it's seen without expanding the technical log)
                                    and the technical log (rate-limited). The progress bar freezes during
                                    the GA — this headline is what tells you it's still working."""
                                    if not _prog_dir:
                                        return
                                    _done = 0
                                    _active = 0
                                    _best = None; _best_fit = None
                                    try:
                                        for _fn in os.listdir(_prog_dir):
                                            if _fn.endswith(".txt"):
                                                try:
                                                    with open(os.path.join(_prog_dir, _fn)) as _pf:
                                                        _parts = _pf.read().split("|")   # "total|best|fit"
                                                    _v = int(((_parts[0]).strip() or "0"))
                                                    _done += _v
                                                    if _v > 0:
                                                        _active += 1
                                                    if len(_parts) > 1 and _parts[1].strip():
                                                        _bs = float(_parts[1])
                                                        if _best is None or _bs > _best:
                                                            _best = _bs
                                                            _best_fit = (float(_parts[2]) if (len(_parts) > 2
                                                                         and _parts[2].strip()) else None)
                                                except Exception:  # noqa: BLE001
                                                    pass
                                    except Exception:  # noqa: BLE001
                                        return
                                    _now = _st_t.time()
                                    _rate = _done / max(_now - _t0, 1e-6)
                                    # VISIBLE headline (updated every poll) — no need to expand the log.
                                    try:
                                        _eta_slot.markdown(
                                            "<div style='padding:0.1rem 0 0.35rem 0; line-height:1.25;'>"
                                            "<span style='font-size:1.4rem; font-weight:800; color:var(--tav-ink);'>"
                                            f"Searching… ≈ {_done:,}</span>"
                                            "<span style='font-size:0.9rem; color:var(--tav-muted);'> candidate "
                                            f"splits evaluated</span><br>"
                                            "<span style='font-size:0.8rem; color:var(--tav-muted);'>"
                                            f"{_active}/{int(_nseed)} seeds reporting · {_rate:,.0f}/s · "
                                            f"t+{_now - _t0:.0f}s</span></div>", unsafe_allow_html=True)
                                    except Exception:  # noqa: BLE001
                                        pass
                                    # Technical-log line (rate-limited): raw count is exact; no denominator —
                                    # the true total exceeds seeds×restarts×gens×λ because CMA-ES doubles λ on
                                    # each IPOP restart, so a fixed "of Y" would read past 100%.
                                    if _done > _emit["n"] and (_now - _emit["t"]) >= 8.0:
                                        _emit["n"], _emit["t"] = _done, _now
                                        _bstr = (f" · best score {_best:,.0f}" if _best is not None else "")
                                        _fstr = (f" · fitness {_best_fit:,.0f}" if _best_fit is not None else "")
                                        log(f"   GA progress: ≈ {_done:,} candidate splits evaluated so far "
                                            f"({_active}/{int(_nseed)} seeds active, {_rate:,.0f}/s) · "
                                            f"t+{_now - _t0:.0f}s{_bstr}{_fstr}")

                                # ---- GA-Numba: PRE-COMPILE the kernel ONCE in the main process ----
                                # Otherwise all 16 loky workers cold-compile the fused kernel
                                # simultaneously (empty cache, no sharing) and fight for the cores —
                                # the ~9-minute stall before the first seed reports. Compiling once
                                # here first writes the persistent .numba_cache, so the workers then
                                # LOAD it in a fraction of a second. One tiny run (pop 4, 1 gen) is
                                # enough to trigger + verify the exact signature they'll use. Any
                                # failure here is now FATAL: it fails loudly with full diagnostics
                                # (no silent NumPy fallback) rather than letting workers limp on.
                                if engine_key == "genetic_numba":
                                    try:
                                        import time as _wt
                                        log("   GA-Numba: compiling the kernel once in the main "
                                            "process (first run after a code/version change only; "
                                            "cached to .numba_cache and reused by every later run "
                                            "and every split/settings combination)…")
                                        # Cache diagnostic: where Numba caches + how many kernel files
                                        # exist BEFORE we compile. Numba writes .nbi/.nbc into a SUBFOLDER
                                        # (…/_numba_cache/<pkg>_<pathhash>/), so count RECURSIVELY — a flat
                                        # listdir always reads 0 and falsely looks like the cache failed.
                                        try:
                                            import glob as _nbglob
                                            _nbcd = os.environ.get("NUMBA_CACHE_DIR", "(default __pycache__)")
                                            if os.path.isdir(_nbcd):
                                                _nbn = len(_nbglob.glob(os.path.join(_nbcd, "**", "*.nbi"),
                                                                        recursive=True)) + \
                                                       len(_nbglob.glob(os.path.join(_nbcd, "**", "*.nbc"),
                                                                        recursive=True))
                                            else:
                                                _nbn = -1
                                            log(f"      numba cache: dir={_nbcd} · exists="
                                                f"{os.path.isdir(_nbcd)} · {_nbn} kernel file(s) present "
                                                "before compile (>0 ⇒ a prior compile persisted; 0 on the "
                                                "FIRST run after a kernel/code change is normal)")
                                        except Exception:  # noqa: BLE001
                                            pass
                                        # Warmup does the FULL verify (no trust flag yet); the copy
                                        # is taken BEFORE we set numba_trust, so this call verifies.
                                        _wkw = dict(_extra_kw); _wkw["n_restarts"] = 1
                                        _wc0 = _wt.time()
                                        _wsh, _winfo = _run_midtilt_ga(
                                            ctx, lam=50.0, pop_size=4, generations=1,
                                            seed=_seed, polish=False, **_wkw)
                                        _wsecs = _wt.time() - _wc0
                                        _wn = (_winfo or {}).get("numba", {}) or {}
                                        if _wn.get("used"):
                                            # Verified once here → tell every worker to TRUST the cached
                                            # kernel and SKIP its own NumPy-vs-Numba re-check (was 16×
                                            # redundant, incl. 16 pointless NumPy evals).
                                            _extra_kw["numba_trust"] = True
                                            _verdict = ("loaded from cache" if _wsecs < 8
                                                        else "COMPILED (first run after a kernel/code change "
                                                             "— expected once; later runs with the SAME code "
                                                             "load this from .numba_cache in <1s)")
                                            log(f"   GA-Numba: kernel {_verdict} in "
                                                f"{_wsecs:.1f}s (max rel-obj diff "
                                                f"{_wn.get('max_rel_obj', float('nan')):.1e}, "
                                                f"max abs-viol {_wn.get('max_abs_viol', float('nan')):.1e}) "
                                                "— workers will load it from cache and skip re-verifying.")
                                        else:
                                            # FAIL-LOUD: the fast path was requested but the pre-compile
                                            # did not enable it. We do NOT silently drop to NumPy — surface
                                            # the full numba diagnostics and abort the run.
                                            log("   GA-Numba: pre-compile did NOT enable the fast path — "
                                                f"FAILING LOUDLY (no silent NumPy fallback). detail: {_wn}")
                                            raise RuntimeError(
                                                "GA-Numba pre-compile did not enable the fused kernel "
                                                f"({_wsecs:.1f}s). numba info = {_wn}. Refusing to fall "
                                                "back to NumPy silently — fix the kernel or run a "
                                                "non-numba engine.")
                                    except Exception as _we:  # noqa: BLE001
                                        # FAIL-LOUD: log the full traceback + build markers and re-raise so
                                        # the run stops with an easy-to-debug error (no silent NumPy fallback).
                                        import traceback as _wtb
                                        _wbuild = getattr(_gg, "__build__", "?")
                                        log("   GA-Numba pre-compile FAILED — failing loudly (no silent "
                                            f"NumPy fallback).\n      error: {type(_we).__name__}: {_we}\n"
                                            f"      genetic_global build: {_wbuild}\n"
                                            f"      traceback:\n{_wtb.format_exc()}")
                                        raise

                                for _bk in _try_backends:
                                    try:
                                        _pk = dict(n_jobs=_njobs, backend=_bk)
                                        if _bk in ("loky", "multiprocessing") and _JL_INNER_OK:
                                            _pk["inner_max_num_threads"] = 1   # avoid BLAS oversubscription
                                        _t_par0 = _st_t.time()
                                        log(f"   multi-seed GA (risk-min endpoint): launching "
                                            f"{int(_N_SEED)} seed(s) across {_njobs} {_bk} worker(s) "
                                            f"— gens≤{int(_ga_gen)}, pop={int(_ga_pop)} each; "
                                            f"≥{_total_cand:,} candidate splits (more with restart λ-growth), "
                                            "count logged live every few seconds…")
                                        # DIVERSE-SEED SEARCH (opt-in, free): give each worker a DISTINCT
                                        # start (blend of the revenue-greedy ↔ risk-greedy references +
                                        # light jitter) and a DISTINCT explore/exploit strategy (varying
                                        # gain_max). warm_shares stays length-1 per worker so the restart
                                        # count is UNCHANGED (n_r = max(n_restarts, #seeds)); same seeds,
                                        # generations, pop → same one-wave wall time. Off → byte-identical.
                                        _diverse = int(_N_SEED) > 1   # diverse-seed search ALWAYS on (checkbox removed)
                                        _seed_ctx = [ctx] * int(_N_SEED)
                                        _seed_gm = [_GA_GAIN_MAX] * int(_N_SEED)
                                        if _diverse:
                                            try:
                                                import numpy as _np_ds
                                                _ws0 = ctx.get("warm_shares")
                                                _anch = ([_np_ds.asarray(a, float) for a in _ws0]
                                                         if isinstance(_ws0, (list, tuple)) and len(_ws0) >= 2
                                                         else None)
                                                _cs_ds = _np_ds.asarray(ctx.get("cell_starts", []), int)
                                                _cc_ds = _np_ds.asarray(ctx.get("cell_counts", []), int)
                                                _seed_ctx, _seed_gm = [], []
                                                for _s in range(int(_N_SEED)):
                                                    _w = _s / max(int(_N_SEED) - 1, 1)   # 0=risk … 1=revenue
                                                    _seed_gm.append(float(_GA_GAIN_MAX * (0.7 + 0.6 * _w)))
                                                    if _anch is not None and _cs_ds.size and _anch[0].shape == _anch[1].shape:
                                                        _rng_ds = _np_ds.random.default_rng(int(_seed) + 7000 + _s)
                                                        _bl = _w * _anch[0] + (1.0 - _w) * _anch[1]
                                                        _bl = _np_ds.clip(_bl + _rng_ds.normal(0.0, 0.04, _bl.shape), 0.0, None)
                                                        _seg = _np_ds.add.reduceat(_bl, _cs_ds)
                                                        _bl = _bl / _np_ds.repeat(_np_ds.where(_seg > 1e-12, _seg, 1.0), _cc_ds)
                                                        _seed_ctx.append({**ctx, "warm_shares": [_bl]})
                                                    else:
                                                        _seed_ctx.append(ctx)
                                                log(f"   diverse-seed search ON: {int(_N_SEED)} workers spread across the "
                                                    f"revenue↔risk axis, explore/exploit gain_max "
                                                    f"{min(_seed_gm):.2f}–{max(_seed_gm):.2f} (same budget/time).")
                                            except Exception as _dse:  # noqa: BLE001
                                                log(f"   [Warning] diverse-seed setup skipped "
                                                    f"({type(_dse).__name__}: {_dse}); standard seeding used.")
                                                _seed_ctx = [ctx] * int(_N_SEED)
                                                _seed_gm = [_GA_GAIN_MAX] * int(_N_SEED)
                                        _tasks = [delayed(_run_midtilt_ga)(
                                            _seed_ctx[_s], lam=50.0, pop_size=_ga_pop, generations=_ga_gen,
                                            seed=_seed + _s, auto=True, patience=_ga_pat,
                                            warm_start=_warm, gain_max=_seed_gm[_s], stop_check=_ga_stop,
                                            progress_cb=_writers[_s], **_extra_kw)
                                            for _s in range(int(_N_SEED))]
                                        _box = {}
                                        # [FN-343]
                                        def _run_par(_pk=_pk, _tasks=_tasks, _box=_box):
                                            _box["r"] = list(Parallel(**_pk)(_tasks))
                                        _emit = {"n": 0, "t": 0.0}
                                        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                                            _fut = _ex.submit(_run_par)
                                            while not _fut.done():
                                                _st_t.sleep(3.0)
                                                _poll_progress(_t_par0, _emit, _N_SEED)
                                            _fut.result()          # re-raise any worker error → fallback
                                        _seed_results = _box.get("r")
                                        # Sum the per-seed progress files one last time → the EXACT
                                        # candidate count actually evaluated, stored so the tab-2 readout
                                        # can show the true number (not just the pre-run floor estimate).
                                        try:
                                            _fin_cands = 0
                                            if _prog_dir:
                                                for _fn in os.listdir(_prog_dir):
                                                    if _fn.endswith(".txt"):
                                                        try:
                                                            with open(os.path.join(_prog_dir, _fn)) as _pf:
                                                                # "total|best|fit" — take field 0 (the count).
                                                                _fin_cands += int(((_pf.read().split("|")[0]).strip() or "0"))
                                                        except Exception:  # noqa: BLE001
                                                            pass
                                            if _fin_cands > 0:
                                                ss["last_ga_cands"] = int(_fin_cands)
                                                ss["last_ga_secs"] = float(_st_t.time() - _t_par0)
                                                # Realization ratio = actual ÷ nominal floor. Captures how
                                                # much λ-growth (↑) and early-stops (↓) net out at THESE
                                                # settings, so the tab-2 readout can predict the next run's
                                                # count tightly (scale the floor by this) instead of showing
                                                # the wide theoretical floor–ceiling band.
                                                _nom = (int(_N_SEED)
                                                        * max(1, int(ss.get("ga_restarts", 4) or 4))
                                                        * int(_ga_gen) * int(_ga_pop))
                                                if _nom > 0:
                                                    ss["last_ga_ratio"] = float(_fin_cands) / float(_nom)
                                        except Exception:  # noqa: BLE001
                                            pass
                                        _bh = [None]
                                        for _si, (_shc, _infoc) in enumerate(_seed_results or [], 1):
                                            _log_seed(_si, _infoc, _t_par0, _bh)
                                        log(f"   multi-seed GA: {int(_N_SEED)} seeds in PARALLEL ({_bk}, "
                                            f"{_njobs} workers) in {_st_t.time() - _t_par0:.1f}s.")
                                        break
                                    except Exception as _pe:  # noqa: BLE001
                                        # loky is the ONLY supported backend — no threading/sequential
                                        # fallback. Fail loudly so a wedged/broken worker pool is never
                                        # silently downgraded to a slower path.
                                        log(f"   ✗ parallel multi-seed GA via {_bk} FAILED "
                                            f"({type(_pe).__name__}: {_pe}). No fallback — clear "
                                            "__pycache__ and fully restart Streamlit, then retry.")
                                        raise
                            _sh, _info = None, None
                            if _seed_results is not None:
                                for _shc, _infoc in _seed_results:            # seed order preserved
                                    if _info is None or _infoc["best_fit"] > _info["best_fit"]:
                                        _sh, _info = _shc, _infoc
                            else:
                                # Single-seed run (N_SEED == 1): no worker pool needed — run the one
                                # seed in-process. This is NOT a parallel fallback; multi-seed ALWAYS
                                # runs on loky and raises on failure (no sequential downgrade).
                                for _s in range(int(_N_SEED)):
                                    _shc, _infoc = _run_midtilt_ga(
                                        ctx, lam=50.0, pop_size=_ga_pop, generations=_ga_gen,
                                        seed=_seed + _s, auto=True, patience=_ga_pat,
                                        warm_start=_warm, gain_max=_GA_GAIN_MAX, stop_check=_ga_stop,
                                        **_extra_kw)
                                    if _info is None or _infoc["best_fit"] > _info["best_fit"]:
                                        _sh, _info = _shc, _infoc
                            # ---- GA - Numba: extensive cross-validation diagnostics --------------
                            # Printed whenever the Numba engine was requested, so a reviewer can
                            # confirm the compiled kernel produced the SAME (objective, violation) as
                            # the NumPy engine before trusting the fast path — or see exactly why it
                            # fell back. All seeds verify identically, so we log the winner's block.
                            try:
                                _nbi = (_info or {}).get("numba") if isinstance(_info, dict) else None
                                if _nbi and _nbi.get("requested"):
                                    log("   ── GA-Numba verification ──────────────────────────────")
                                    log(f"      kernel build   : {_nbi.get('build', '?')}")
                                    if _nbi.get("used") and _nbi.get("trusted"):
                                        log("      DECISION       : ✓ USING Numba fused float64 kernel "
                                            "(verified once at pre-compile; workers trusted the cached "
                                            "kernel — see the compile line above for the diff figures)")
                                    elif _nbi.get("used"):
                                        log("      DECISION       : ✓ USING Numba fused float64 kernel "
                                            "(verified identical to NumPy)")
                                    else:
                                        log(f"      DECISION       : ↩ FELL BACK to NumPy engine "
                                            f"— {_nbi.get('reason', 'unknown')}")
                                    _ns = _nbi.get("n_sample")
                                    if _ns:
                                        log(f"      verified on    : {int(_ns)} sample genomes "
                                            "(θ=0 base + warm-start + random draws)")
                                    # Per-worker cache diagnostic: how each seed's FIRST eval timed.
                                    # <1s = worker LOADED the cached kernel; tens of seconds = it
                                    # RE-COMPILED (cache miss in the worker) → the real cause of a
                                    # crawling multi-seed search. Aggregated across all seed results.
                                    try:
                                        _fe = [float((_ic.get("numba") or {}).get("first_eval_s"))
                                               for (_sc, _ic) in (_seed_results or [])
                                               if isinstance(_ic, dict)
                                               and (_ic.get("numba") or {}).get("first_eval_s") is not None]
                                        _ntot = len(_seed_results or [])
                                        _nused = sum(1 for (_sc, _ic) in (_seed_results or [])
                                                     if isinstance(_ic, dict)
                                                     and (_ic.get("numba") or {}).get("used"))
                                        if _ntot:
                                            log(f"      workers        : {_nused}/{_ntot} used Numba")
                                        if _fe:
                                            _slow = sum(1 for _x in _fe if _x > 8.0)
                                            log(f"      worker 1st-eval: min={min(_fe):.1f}s "
                                                f"max={max(_fe):.1f}s · {_slow}/{len(_fe)} RE-COMPILED "
                                                "(>8s ⇒ cache miss in worker → slow search)")
                                    except Exception:  # noqa: BLE001
                                        pass
                                    _mro = _nbi.get("max_rel_obj"); _mao = _nbi.get("max_abs_obj")
                                    _mav = _nbi.get("max_abs_viol"); _mrv = _nbi.get("max_rel_viol")
                                    if _mro is not None and _mro == _mro:   # not NaN
                                        log(f"      objective diff : max |rel|={_mro:.2e}  max |abs|={_mao:.2e}"
                                            "   (tolerance rel ≤ 1e-7)")
                                        log(f"      violation diff : max |abs|={_mav:.2e}  max |rel|={_mrv:.2e}"
                                            "   (tolerance abs ≤ 1e-7)")
                                    _tnp = _nbi.get("t_np"); _tnb = _nbi.get("t_nb")
                                    _cmp = _nbi.get("compile_s"); _spd = _nbi.get("speedup")
                                    if _tnp is not None and _tnp == _tnp:
                                        log(f"      sample timing  : numpy={_tnp*1e3:.2f} ms  "
                                            f"numba={_tnb*1e3:.2f} ms  (JIT compile {_cmp*1e3:.0f} ms, "
                                            f"one-off)  →  {_spd:.2f}× on the verify batch")
                                    log("      note           : per-run speed-up is larger than the tiny "
                                        "verify batch shows; compile is paid ONCE per worker process.")
                                    log("   ───────────────────────────────────────────────────────")
                            except Exception as _nle:  # noqa: BLE001 - logging must never break a run
                                log(f"   [Warning] GA-Numba diagnostics log skipped ({_nle}).")
                            # Preserve the MAIN search's full generation history so the convergence
                            # chart shows all N generations — the re-projection loop below may swap
                            # _info for a SHORT correction run (its history is only a handful of gens,
                            # which was why the chart showed e.g. 109 instead of the 300 that ran).
                            _main_hist = (list(_info.get("history"))
                                          if (isinstance(_info, dict) and _info.get("history")) else None)
                            if ctx["midband"]:
                                # Post-GA re-projection CORRECTION removed — band scoring is now EXACT
                                # in-search, so the search already satisfies the true pro-rata bands (no
                                # proxy→true gap to reconcile). This is a READ-ONLY compliance readout of
                                # the delivered split (and it runs the incidence self-check inside
                                # _ga_true_breach); it does NOT modify the split.
                                try:
                                    _br = _ga_true_breach(_sh)
                                    log(f"   delivered split — true-band breach {_br:.4g} "
                                        "(0 = all month bands satisfied by the real pro-rata projection; "
                                        "exact in-search scoring, no post-hoc correction).")
                                except Exception as _e:  # noqa: BLE001
                                    log(f"   [Warning] delivered-split band readout skipped ({_e}).")
                            ctx["risk_min_w"] = 0.0
                            ctx["band_weight"] = 8.0   # reset to defaults for anything downstream
                            ctx["band_fixed"] = 20.0
                            ctx["ref_gamma"] = _rg_default
                            # Restore the main search's full-length history for the convergence chart.
                            if _main_hist is not None and isinstance(_info, dict):
                                _info = {**_info, "history": _main_hist}
                            try:
                                log(f"   convergence history: {len(_info.get('history') or [])} "
                                    f"generation(s) recorded (main search; the re-projection run's own "
                                    f"short history is NOT used for the chart).")
                            except Exception:  # noqa: BLE001
                                pass
                            return _sh, _info

                        import time as _gatime
                        _ga_wall0 = _gatime.time()
                        # #10 greedy warm-start: hand the CMA-ES the known FEASIBLE greedy split so it
                        # begins INSIDE the feasible region (it fits a genome to it). The risk-min
                        # endpoint overrides this with the revenue endpoint's genome archive, so this
                        # only seeds the revenue-max endpoint.
                        ctx["warm_shares"] = np.asarray(_comp_share_G, float)
                        # Precompute the sparse per-MID incidence ONCE (speed #1) so every seed/worker
                        # reads it instead of rebuilding — identical results, hot-path saving.
                        try:
                            ctx["_mid_S"] = (_gg._build_mid_incidence(
                                np.asarray(ctx["mid_id"]), int(ctx["n_mid"]), int(ctx["n_row"]))
                                if int(ctx["n_mid"]) else None)
                        except Exception:  # noqa: BLE001
                            ctx["_mid_S"] = None
                        _progress(_f_eng, "Revenue-max endpoint (greedy)…")
                        # Aggregate per-vampMid VAMP-rate compliance guard (used for the dial-0 adoption).
                        # [FN-344]
                        def _agg_mid_ok(_sh):
                            _v = _cvol * np.asarray(_sh, float)
                            for _mi, _r in enumerate(_mid_rows):
                                _vol = float(_v[_r].sum())
                                if vamp_cap is not None and _vol > 1e-12:
                                    if float((_v[_r] * _rkr[_r]).sum()) / _vol > float(vamp_cap) + 1e-9:
                                        return False
                            return True
                        # Revenue-max (dial 99) endpoint = the greedy revenue reference already shaved to
                        # compliance by the LP (enforce_mid_vamp_caps), which is MINIMUM-MOVEMENT optimal
                        # for the per-cell revenue objective under the cross-cell VAMP cap. The CMA-ES
                        # smart-search that used to run here (~40 min) was consistently matched-or-beaten
                        # by that greedy+LP split and its output DISCARDED — so it's removed. Dial 99 takes
                        # the greedy compliant split directly; the CMA-ES still runs for the harder risk-min
                        # (dial 0) end below. Trade-off: in a rare case the search might beat greedy on
                        # revenue here — that upside is now forgone (greedy is the no-regression floor).
                        _comp_endpoint_G = _comp_share_G   # dial 99 = max-revenue compliant (greedy + LP)
                        log(f"   revenue-max (dial 99) endpoint: greedy revenue-compliant split "
                            f"${_rev_of(_comp_share_G):,.0f} (CMA-ES revenue search removed — greedy is "
                            f"LP-optimal here, so its result was always discarded).")
                        _progress(_f_rmin, "GA risk-min endpoint…")
                        # SECOND GA run for the SAFE (dial-0) endpoint: same setup + a risk-minimisation
                        # term so it tilts each MID further toward its low-risk cells while staying
                        # compliant. mu is auto-scaled from the reference so the risk term is a bounded
                        # fraction (~risk_aversion) of reference revenue — trades some revenue for lower
                        # aggregate VAMP without degenerating (the θ-tilt keeps it revenue-shaped per cell).
                        _rev_ref = max(_rev_of(_ref_share_G), 1.0)
                        _vamp_ref = max(float((_cvol * _ref_share_G * _rkr).sum()), 1.0)
                        _risk_aversion = 0.0   # FIXED: dial removed — always revenue-shaped (no extra risk-min beyond the caps)
                        # Reference lean OFF (#8): the θ=0 base is used as-is (no low-risk lean). A
                        # compliant starting point comes instead from the diverse warm-start seeds below
                        # (the greedy+LP compliant split and the risk-greedy split), which feasibility-first
                        # ranking keeps from generation 0. This is equivalent to ref_gamma = 0, so
                        # _leaned_ref is a no-op; set a positive value to re-enable the lean.
                        _ref_gamma_auto = 0.0
                        log("   reference lean: OFF (γ=0) — compliant start comes from the "
                            "compliant + risk-greedy warm-start seeds.")
                        _safe_wall0 = _gatime.time()
                        # risk-min endpoint (dial 0): same GA (EXACT in-search band scoring; read-only
                        # true-band readout, no correction), with the
                        # risk-min term AND a TOUGHER per-MID band penalty (4× the dial-99 weight) so
                        # dial 0 sits inside every band harder. Intermediate dials inherit this via the
                        # frontier blend between the dial-0 and dial-99 endpoints.
                        _warm_dial0 = None
                        _band_mult = 1.0   # band penalty strength fixed at 1.0 (input removed)
                        # #1 DIVERSE SEEDS: hand the risk-min search BOTH the revenue-greedy compliant
                        # split AND a risk-greedy split (each cell's share leaning to its lowest-risk
                        # gateways), so it starts inside the risk-min basin, not just the revenue corner.
                        try:
                            _zrc, _ = _gg._risk_z_per_cell(np.asarray(ctx["risk"], float),
                                                           np.asarray(ctx["cell_starts"], np.intp),
                                                           np.asarray(ctx["cell_counts"], np.intp),
                                                           int(ctx["n_row"]))
                            _rw = np.asarray(ctx["elig"], float) * np.exp(-6.0 * _zrc)     # lean low-risk
                            _seg = np.add.reduceat(_rw, np.asarray(ctx["cell_starts"], np.intp))
                            _risk_greedy_G = _rw / np.repeat(np.where(_seg > 1e-12, _seg, 1.0),
                                                             np.asarray(ctx["cell_counts"], np.intp))
                            ctx["warm_shares"] = [np.asarray(_comp_share_G, float), _risk_greedy_G]
                        except Exception:  # noqa: BLE001
                            _risk_greedy_G = np.asarray(_comp_share_G, float)
                            ctx["warm_shares"] = np.asarray(_comp_share_G, float)
                        _n_fine_rm = int(min(40, max(0, _n_cells)))   # #4 richer per-cell genome (bounded)
                        _rm_w = _risk_aversion * _rev_ref / _vamp_ref
                        # #6 DISK CACHE: the risk-min search is deterministic, so a re-run with identical
                        # inputs returns the SAME split instantly. The key hashes the engine build + every
                        # ctx array + all endpoint params, so a hit can NEVER be stale.
                        import hashlib as _hl_rm

                        # [FN-345]
                        def _riskmin_key():
                            _h = _hl_rm.md5(); _h.update(str(getattr(_gg, "__build__", "?")).encode())
                            for _k in ("ref_share", "risk", "rev_coef", "cell_vol", "mid_id",
                                       "cell_starts", "cell_counts", "elig", "mid_vol_cap",
                                       "mid_base_vol", "vamp_floor_route"):
                                _v = ctx.get(_k)
                                if _v is not None:
                                    _h.update(_k.encode())
                                    _h.update(np.ascontiguousarray(np.asarray(_v)).tobytes())
                            _h.update(np.ascontiguousarray(np.asarray(_comp_share_G, float)).tobytes())
                            _h.update(np.ascontiguousarray(np.asarray(_risk_greedy_G, float)).tobytes())
                            _h.update(repr((float(vamp_cap) if vamp_cap is not None else None,
                                            float(ctx.get("max_share", 1.0) or 1.0),
                                            float(ctx.get("floor", 0.0) or 0.0), repr(_mid_month_rules),
                                            round(float(_rm_w), 6), round(float(_band_mult), 4),
                                            round(float(_ref_gamma_auto), 6),
                                            int(_n_fine_rm), int(_ga_pop), int(_ga_gen), int(_N_SEED),
                                            float(_GA_GAIN_MAX),
                                            int(max(1, int(ss.get("ga_restarts", 4) or 4))),
                                            # σ step-size knobs + early-stop toggle MUST be in the key, else a
                                            # tuning change silently reloads a stale cached run (added 2026-08-04).
                                            round(float(ctx.get("sigma0_mult", 1.0) or 1.0), 6),
                                            round(float(ctx.get("sigma_floor", 0.0) or 0.0), 6),
                                            round(float(ctx.get("damps_mult", 1.0) or 1.0), 6),
                                            bool(ctx.get("no_early_stop", False)), 42)).encode())
                            return _h.hexdigest()

                        _rm_path = None; _safe_G = _inf2 = None
                        try:
                            _rm_path = os.path.join(PROJECT_ROOT, ".cache", f"riskmin_{_riskmin_key()}.pkl")
                            if os.path.exists(_rm_path):
                                import pickle as _pk_rm
                                with open(_rm_path, "rb") as _f_rm:
                                    _obj = _pk_rm.load(_f_rm)
                                _safe_G, _inf2 = _obj["safe_G"], _obj["inf2"]
                                log("   risk-min (dial 0): loaded from disk cache "
                                    "(deterministic — identical to re-searching).")
                        except Exception:  # noqa: BLE001
                            _safe_G = _inf2 = None; _rm_path = None
                        if _safe_G is None:
                            # dial-0: bands scaled by UI strength; reference lean OFF (γ=0, #8) — the
                            # compliant start comes from the warm-start seeds; richer per-cell genome (#4)
                            # + extra restarts (#3) at this harder end.
                            _safe_G, _inf2 = _ga_solve_with_correction(
                                _rm_w, _band_w=3375.0 * _band_mult, _band_fix=8100.0 * _band_mult,
                                _warm=_warm_dial0, _ref_gamma=_ref_gamma_auto, _n_fine=_n_fine_rm,
                                _n_restarts=max(1, int(ss.get("ga_restarts", 4) or 4)))
                            if _rm_path:
                                try:
                                    import pickle as _pk_rm, glob as _glob_rm
                                    os.makedirs(os.path.dirname(_rm_path), exist_ok=True)
                                    with open(_rm_path, "wb") as _f_rm:
                                        _pk_rm.dump({"safe_G": _safe_G, "inf2": _inf2}, _f_rm)
                                    _old_rm = sorted(_glob_rm.glob(os.path.join(
                                        os.path.dirname(_rm_path), "riskmin_*.pkl")), key=os.path.getmtime)
                                    for _o_rm in _old_rm[:-40]:
                                        try:
                                            os.remove(_o_rm)
                                        except Exception:  # noqa: BLE001
                                            pass
                                except Exception:  # noqa: BLE001
                                    pass
                        ss["ga_hist_rev"] = None                              # revenue-max CMA-ES removed
                        ss["ga_hist_safe"] = (_inf2.get("history") if _inf2 else None)   # convergence chart
                        # Engine-workings charts reflect the risk-min search (now the ONLY CMA-ES run).
                        ss["ga_genome"] = (_inf2.get("genome") if _inf2 else None)
                        ss["ga_mid_labels"] = [str(m) for m in _mids_u]
                        ss["ga_pop_obj"] = (_inf2.get("pop_obj") if _inf2 else None)
                        ss["ga_pop_viol"] = (_inf2.get("pop_viol") if _inf2 else None)
                        _ga_perf_secs = _gatime.time() - _safe_wall0          # the single CMA-ES run's wall time
                        ss["ga_perf"] = {"secs": float(_ga_perf_secs), "budget": int(_ga_pop * _ga_gen),
                                         "gen": int(_ga_gen), "pop": int(_ga_pop), "seeds": int(_N_SEED),
                                         "restarts": int(max(1, int(ss.get("ga_restarts", 4) or 4))),
                                         "nvar": 1, "n": int(len(G))}
                        _save_ga_perf(ss["ga_perf"])
                        # (_ga_solve_with_correction resets risk_min_w / band_weight on exit.)
                        _ga_wall_tot = _gatime.time() - _ga_wall0
                        # ---- ④ SETTINGS-EFFICIENCY SELF-REPORT --------------------------------
                        # So each run says how much search its settings bought and how efficiently.
                        # Compare these across runs ON THE SAME DATA/OBJECTIVE: a higher score/min for
                        # a similar (or better) best score = more efficient settings. Diversity from
                        # parallel seeds is ~free; restarts (IPOP λ-doubling) cost the most — this
                        # readout is how you see whether a knob earned its time.
                        try:
                            _eff_best = (float(_inf2.get("best_fit")) if (_inf2 and
                                         _inf2.get("best_fit") is not None) else float("nan"))
                            _eff_cands = int(ss.get("last_ga_cands", 0) or 0)
                            _eff_ssecs = float(ss.get("last_ga_secs", 0.0) or 0.0)   # search-only wall
                            _eff_rst = max(1, int(ss.get("ga_restarts", 4) or 4))
                            _eff_cps = (_eff_cands / _eff_ssecs) if (_eff_cands > 0 and _eff_ssecs > 0) else 0.0
                            _eff_spm = (_eff_best / (_eff_ssecs / 60.0)) if (_eff_ssecs > 0
                                        and _eff_best == _eff_best) else float("nan")   # nan-safe
                            _eff_mode = str((_inf2 or {}).get("restart_mode", "ipop"))
                            log("   ④ EFFICIENCY (how much search these settings bought, and how fast):")
                            log(f"      settings   : {int(_N_SEED)} seeds × {_eff_rst} restarts × "
                                f"{int(_ga_gen)} gens × λ{int(_ga_pop)} · restart-mode={_eff_mode}")
                            log(f"      best score : {_eff_best:,.0f}" if _eff_best == _eff_best
                                else "      best score : n/a")
                            if _eff_cands > 0 and _eff_ssecs > 0:
                                log(f"      search cost: {_eff_cands:,} candidate splits in "
                                    f"{_eff_ssecs:.0f}s ({_eff_cps:,.0f}/s throughput)")
                            else:
                                log("      search cost: candidate count unavailable this run "
                                    "(live counter reported 0)")
                            if _eff_spm == _eff_spm:
                                log(f"      EFFICIENCY : {_eff_spm:,.0f} score/min "
                                    "— compare across runs on the SAME data (higher = better settings)")
                        except Exception as _effe:  # noqa: BLE001 - a readout must never break a run
                            log(f"   [Warning] ④ efficiency self-report skipped ({_effe}).")
                        # Use the risk-min GA for dial 0 only if it is compliant; else fall back to the
                        # revenue-max endpoint (dial 0 == dial 99, frontier collapses but never regresses).
                        _safe_endpoint_G = _safe_G if _agg_mid_ok(_safe_G) else _comp_endpoint_G
                        _rate_of = lambda _sh: (float((_cvol * np.asarray(_sh, float) * _rkr).sum())
                                                / max(float((_cvol * np.asarray(_sh, float)).sum()), 1e-9))
                        log(f"   GA risk-min (dial 0) endpoint: aggregate VAMP rate {_rate_of(_safe_endpoint_G):.4f} "
                            f"vs revenue-max (dial 99) {_rate_of(_comp_endpoint_G):.4f}; revenue "
                            f"${_rev_of(_safe_endpoint_G):,.0f} vs ${_rev_of(_comp_endpoint_G):,.0f}.")
                        # [FN-346]
                        def _endpoint_agg(_shares):
                            _agg = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                            _agg["share"] = np.asarray(_shares, dtype=float)
                            _agg["volume"] = _agg["cell_volume"] * _agg["share"]
                            return _agg
                        # AUTO-BLOCK (pre-enforcement): detect bank-blocked (bank,gateway) pairs UP
                        # FRONT and cap them to the exploration floor INSIDE the enforcement input, so
                        # the freed volume is redistributed COMPLIANTLY (VAMP + per-MID bands) by the
                        # same enforcement pass that already runs — instead of the old post-hoc patch
                        # that capped AFTER enforcement and could perturb VAMP compliance. No extra
                        # solve, so ≈ no added time. Best-effort; empty set → unchanged behaviour.
                        _blk_pairs_pre = set()
                        if bool(ss.get("block_gw_cb", False)):
                            try:
                                _bapre = orig_adf.copy()
                                _b2bp = bin_to_bank or {}   # current-run map (not stale ss — see pre-GA note)
                                if _b2bp and "bank" in _bapre.columns:
                                    _bapre["bank"] = _bapre["bank"].map(
                                        lambda _b: _b2bp.get(_b, _b2bp.get(str(_b), _b)))
                                _bdfp = detect_blocked_gateways(
                                    _bapre, float(ss.get("block_min_inp", 100) or 100))
                                _bflagp = _bdfp[_bdfp["blocked"]] if not _bdfp.empty else _bdfp
                                _blk_pairs_pre = set(zip(
                                    _bflagp["bank"].astype(str).str.strip().str.lower(),
                                    _bflagp["gateway"].astype(str).str.strip().str.lower()))
                                if _blk_pairs_pre:
                                    log(f"   auto-block (pre-enforcement): {len(_blk_pairs_pre)} bank×gateway "
                                        "capped to the exploration floor INSIDE enforcement → the freed "
                                        "volume is redistributed compliantly (no post-hoc VAMP perturbation).")
                            except Exception as _bpe:  # noqa: BLE001
                                log(f"   [Warning] pre-enforcement auto-block detect skipped "
                                    f"({type(_bpe).__name__}: {_bpe}); post-hoc cap still applies.")

                        # ENFORCEMENT REMOVED — a single dial-0 variation only. The delivered split is
                        # the GA / CMA-ES risk-min endpoint (_safe_endpoint_G) with ONLY the eligibility
                        # projection kept (hard bans + wallet-incapable + USA-only via _restrict), so a
                        # production config never routes to an ineligible gateway. Dead (bank-blocked)
                        # gateways are still floored (data-driven). NO VAMP-cap / per-MID band projection,
                        # no revenue endpoint, no frontier blend.
                        _deliver_G = _safe_endpoint_G
                        _ga_gran = _explode(_endpoint_agg(_deliver_G))
                        if _blk_pairs_pre:                       # floor dead (bank-blocked) gateways
                            _ga_gran, _ = _apply_blocked_caps(_ga_gran, _blk_pairs_pre, float(floor),
                                                              bin_to_bank=bin_to_bank)
                        _comp_gran = _restrict(_ga_gran)         # eligibility only (bans / wallet / USA)
                        log("   Enforcement OFF: delivered split = GA search output + eligibility "
                            "(bans / wallet-incapable / USA-only); no VAMP-cap / per-MID band projection.")
                        _progress(_f_var, "Building variation…")
                        # SINGLE dial-0 variation = the eligibility-projected GA split (_comp_gran).
                        # No frontier / blend / revenue endpoint — the dial and its multi-position
                        # machinery are removed. "MIDs over cap" is logged for information only; no cap
                        # is enforced on the delivered split.
                        variations = []
                        if _comp_gran is not None and not getattr(_comp_gran, "empty", True):
                            _rg = _comp_gran.reset_index(drop=True).copy()
                            summ = portfolio_summary(_rg)
                            _mo = int(_mids_over_granular(_rg))
                            log(f"   ── GA single variation (dial 0): MIDs over cap={_mo} "
                                f"(informational — no cap enforced), succ={summ['expected_success_rate']:.4f}, "
                                f"risk={summ['expected_risk_rate']:.4f}")
                            variations.append({
                                "weight": 0.0, "split": _rg, "settings": ref_settings,
                                "mids_over_cap": _mo,
                                **{k: v for k, v in summ.items() if k != "volume"},
                                "volume": summ["volume"],
                            })
                        log(f"   GA total wall time: {_fmt_secs(_ga_wall_tot)} "
                            f"(1 GA run [risk-min] × {_n_mid} vampMid tilts; enforcement removed).")
                    else:
                        # NON-GENETIC engines (softmax / thompson / portfolio): a SINGLE dial-0
                        # variation = the engine reference split with ONLY the eligibility projection
                        # (bans / wallet-incapable / USA-only). Enforcement, dials and the frontier
                        # blend are removed, so `changed` / `comp_share` no longer matter here.
                        ref_gran = _restrict(_explode(ref_agg))
                        variations = []
                        if ref_gran is not None and not getattr(ref_gran, "empty", True):
                            ref_gran = ref_gran.reset_index(drop=True)
                            summ = portfolio_summary(ref_gran)
                            _mo = int(_mids_over_granular(ref_gran))
                            log(f"   ── {engine_key} single variation (dial 0): MIDs over cap={_mo} "
                                f"(informational — no cap enforced), succ={summ['expected_success_rate']:.4f}, "
                                f"risk={summ['expected_risk_rate']:.4f}")
                            variations.append({
                                "weight": 0.0, "split": ref_gran, "settings": ref_settings,
                                "mids_over_cap": _mo,
                                **{k: v for k, v in summ.items() if k != "volume"},
                                "volume": summ["volume"],
                            })

                    # --- GRANULAR PROFILE SAMPLES: dump a handful of representative engine
                    #     cells (currency × bank × rpgt) end-to-end — each gateway's baseline vs
                    #     proposed share, forecast volume, VAMP risk and vampMid — so every run
                    #     shows concrete profile-level decisions, not just aggregate counts.
                    #     Samples the biggest cells + the biggest reallocations. Best-effort. ---
                    try:
                        _samp_v = min(variations, key=lambda v: v["weight"]) if variations else None
                        _sdf = _samp_v["split"].copy() if _samp_v is not None else pd.DataFrame()
                        _ckeys = [c for c in ("currency", "bank", "rpgt") if c in _sdf.columns]
                        if _ckeys and "share" in _sdf.columns and not _sdf.empty:
                            _sdf["_bs"] = pd.to_numeric(_sdf.get("baseline_share", 0), errors="coerce").fillna(0.0)
                            _sdf["_sh"] = pd.to_numeric(_sdf["share"], errors="coerce").fillna(0.0)
                            _sdf["_vol"] = pd.to_numeric(_sdf.get("volume", 0), errors="coerce").fillna(0.0)
                            _cv = _sdf.groupby(_ckeys)["_vol"].sum()
                            _mv = (_sdf.assign(_d=(_sdf["_sh"] - _sdf["_bs"]).abs())
                                   .groupby(_ckeys)["_d"].sum())
                            _pick, _seen = [], set()
                            for _k in list(_cv.sort_values(ascending=False).head(3).index) + \
                                      list(_mv.sort_values(ascending=False).head(4).index):
                                _kk = _k if isinstance(_k, tuple) else (_k,)
                                if _kk not in _seen:
                                    _seen.add(_kk); _pick.append(_kk)
                                if len(_pick) >= 6:
                                    break
                            _grp = _sdf.groupby(_ckeys)
                            log(f"   ── GRANULAR PROFILE SAMPLES · dial {int(round(_samp_v['weight'] * 100))} · "
                                f"{len(_pick)} of {len(_cv):,} cells (currency × bank × rpgt) ──")
                            log("      each row: gateway · baseline% → proposed% (Δpp) · fc volume · VAMP risk · vampMid")
                            for _kk in _pick:
                                _rows = _grp.get_group(_kk if len(_kk) > 1 else _kk[0]).copy()
                                _rows = _rows[(_rows["_bs"] > 1e-6) | (_rows["_sh"] > 1e-6)]
                                _rows = _rows.sort_values("_sh", ascending=False).head(12)
                                _lbl = " / ".join(str(x) for x in _kk)
                                log(f"      • {_lbl}  ·  cell_vol={float(_rows['_vol'].sum()):,.0f}  ·  {len(_rows)} active gateway(s)")
                                for _, _r in _rows.iterrows():
                                    _gw = str(_r.get("gateway", "?"))
                                    _b, _p = float(_r["_bs"]) * 100.0, float(_r["_sh"]) * 100.0
                                    _rk = pd.to_numeric(_r.get("gateway_risk_rate", _r.get("rate", None)), errors="coerce")
                                    _rks = f"{float(_rk) * 100:.2f}%" if pd.notna(_rk) else "—"
                                    _vm = str(_r.get("vampMid", "") or "")
                                    log(f"          {_gw:<30s} {_b:5.1f}% → {_p:5.1f}% ({_p - _b:+5.1f}pp) · "
                                        f"vol {float(_r['_vol']):>8,.0f} · risk {_rks:>7s}" + (f" · {_vm}" if _vm else ""))
                    except Exception as _e:  # noqa: BLE001
                        _diag(f"   [granular profile samples failed: {_e}]")

                    granular_sr = gateway_success_rates(orig_adf, shrink_strength=float(shrink), time_decay_half_life_days=(float(decay_half) if apply_decay else None), prior_scope=("rpgt", "currency", "bank"), empirical_bayes=use_eb)
                    granular_problems = build_cell_problems(orig_forecast, granular_sr)

                    ss["mid_vol_constrained"] = sorted(
                        set(ss.get("mid_vol_constrained", [])) | {str(m) for m in _mid_gran_constrained})
                    ss["mid_rpgt_constrained"] = sorted({_rpgt_disp.get(k, k) for k in _rpgt_constrained})

                    # --- Auto-block dead gateways: cap bank-blocked (bank×gateway) shares to the floor
                    # across EVERY dial's split, BEFORE the impact frames are built, so impact + export
                    # both reflect it. (Applied post-engine, so it can slightly perturb VAMP compliance;
                    # blocked gateways are near-zero-volume by definition, so the effect is small.)
                    ss["blocked_pairs"] = set(); ss["blocked_gateways"] = None; ss["blocked_capped"] = 0
                    if bool(ss.get("block_gw_cb", False)):
                        try:
                            _badf = orig_adf.copy()
                            _b2b = bin_to_bank or {}   # current-run map (not stale ss — see pre-GA note)
                            if _b2b and "bank" in _badf.columns:   # match the split's (BIN→bank-mapped) bank
                                _badf["bank"] = _badf["bank"].map(
                                    lambda _b: _b2b.get(_b, _b2b.get(str(_b), _b)))
                            _bmin = float(ss.get("block_min_inp", 100) or 100)
                            _bdf = detect_blocked_gateways(_badf, _bmin)
                            ss["blocked_gateways"] = _bdf
                            _bflag = _bdf[_bdf["blocked"]] if not _bdf.empty else _bdf
                            _bpairs = set(zip(_bflag["bank"].astype(str).str.strip().str.lower(),
                                              _bflag["gateway"].astype(str).str.strip().str.lower()))
                            ss["blocked_pairs"] = _bpairs
                            if _bpairs:
                                _tot_cap = 0
                                for v in variations:
                                    v["split"], _nc = _apply_blocked_caps(v["split"], _bpairs, float(floor),
                                                                          bin_to_bank=bin_to_bank)
                                    _tot_cap += _nc
                                ss["blocked_capped"] = int(_tot_cap)
                                log(f"   auto-block: {len(_bpairs)} bank×gateway flagged as BANK-BLOCKED "
                                    f"(≥{int(_bmin)} most-recent consecutive failed attempts) → capped to the "
                                    f"exploration floor ({_tot_cap} split row(s) capped across dials).")
                                if _tot_cap == 0:
                                    log("   [Warning] auto-block: flagged pairs matched no split rows "
                                        "(bank/gateway naming mismatch?) — no shares were capped.")
                        except Exception as _be:  # noqa: BLE001
                            log(f"   [Warning] auto-block detection skipped ({type(_be).__name__}: {_be}).")

                    # --- CACHING UPGRADE: Pre-calculate the impact frames for instant sliders ---
                    _progress(_f_eng_end, "Pre-calculating impact…")
                    _stage("⑤ Pre-calculate impact frames for all variations")
                    _ensure_base_30d_metrics()
                    if "cached_base_30d_metrics" in ss:
                        _c30d = ss["cached_base_30d_metrics"]
                        for v in variations:
                            v["eval_df"] = _impact_eval_frame(v["split"], _c30d, by_rpgt=bool(_opt_by_rpgt))
                    # ----------------------------------------------------------------------------

                    ss["variations"] = variations
                    ss["_comp_eval_cache"] = {}   # invalidate cached compressed eval frames (new split)
                    ss["agg_problems"] = agg_problems
                    ss["score_by_rpgt"] = bool(_score_by_rpgt)
                    ss["opt_by_rpgt"] = bool(_opt_by_rpgt)
                    ss["mid_constraints"] = list(params.get("mid_constraints", []) or [])

                    ss["agg_sr"] = agg_sr            # ALL_RPGTS rates the engine actually uses
                    ss["agg_raw_att"] = agg_adf_full.groupby(  # UN-decayed parent attempts, for verification
                        ["currency", "parent_bank", "gateway"], as_index=False)["attempts"].sum()
                    ss["bin_to_bank"] = bin_to_bank  # BIN -> parent bank, for Engine Score lookup
                    ss["softmax_temperature"] = float(params.get("temperature", 0.05)) if engine_key == "softmax" else None
                    ss["temp_method"] = temp_method
                    ss["cell_temperature"] = {f"{c}|{b}": float(v) for (c, b), v in cell_temp.items()}
                    ss["shrink_kappa"] = float(shrink)
                    ss["apply_decay"] = bool(apply_decay)
                    ss["problems"] = granular_problems
                    ss["sr"] = granular_sr
                    ss["forecast"] = orig_forecast
                    ss["adf"] = orig_adf
                    ss["variations_engine"] = engine_key
                    ss["base_settings"] = base_settings
                    ss.pop("selected_variation", None)
                    ss.pop("split", None)
                    
                    ss.pop("tab4_cache", None)
                    ss.pop("cached_base_30d_metrics", None)

                    # -- Pre-compute the pool-targeted compression for EVERY dial position now,
                    #    so tab 3's Pools/Fidelity cards and the 'Compressed Rules' impact basis
                    #    are ready without clicking Export Templates. Keyed by the SAME signature
                    #    tab 3 reads (ss['_pool_comp']). Expensive (one search per variation) —
                    #    best-effort per position so one failure never kills the run; any missed
                    #    position simply falls back to on-Export computation in tab 3. --
                    _maxN_pc = int(ss.get("max_configs", 0) or 0)
                    if _maxN_pc > 0:
                        ss["_pool_comp"] = {}   # drop stale sigs from a previous variation set
                        _wc_pc = ss.get("wallet_ctx") or {}
                        _fs_pc = ss.get("forecast_settings", {}) or {}
                        _company_pc = str(_fs_pc.get("company", "TotalAV"))
                        _gl_pc = ss.get("split_go_live_date", date.today())
                        try:
                            from routing_optimiser.connector_pool_configs import (
                                BRANDS as _POOL_BRANDS_PC, company_to_brand_key as _co2brand_pc)
                            _bk_pc = _co2brand_pc(_company_pc)
                            _bn_pc = _POOL_BRANDS_PC.get(_bk_pc, {}).get("name", _company_pc)
                        except Exception:  # noqa: BLE001
                            _bk_pc, _bn_pc = "tav", _company_pc
                        _mid_list_pc = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                        _ms_pc = round(float(_wc_pc.get("max_share", 0.97)), 4)
                        # Precompute the pool-targeted compression for EVERY dial now (here on tab 2,
                        # during the run), so the tab-3 dial + impact views are instant with no
                        # on-demand wait. The dials are INDEPENDENT and DETERMINISTIC, so they run in
                        # PARALLEL below. (The tab-3 on-demand path is kept only as a fallback for any
                        # dial that somehow isn't precomputed here; results cache into ss['_pool_comp'].)
                        _stage(f"⑥ Pre-compute pool-targeted compression for all {len(variations)} "
                               f"dial(s) (target <= {_maxN_pc:,})")
                        _t6_0 = _pt.time()   # for the adaptive-ETA compression calibration
                        # Recalibrate the compression ETA to THIS run's observed speed: scale the
                        # last-run compression estimate by how the actual pre+engine wall time
                        # compared to its estimate, then rebase the total + start-fraction on the
                        # real elapsed so the countdown starts from a realistic remaining time
                        # (rather than a stale over-estimate from a slower prior run).
                        _el_now = _t6_0 - _run_t0
                        _spd = _el_now / max(_PRE_est + _E_est, 1.0)
                        _C_est = max(_C_est * _spd, 1.0)
                        _T_est = _el_now + _C_est
                        _f_eng_end = _el_now / _T_est
                        # Build one job per dial (sig + weight + its ideal split) — all of them now.
                        _jobs_pc = []
                        for _v in variations:
                            _w_pc = float(_v["weight"])
                            _sig_pc = (_w_pc, _maxN_pc, ss.get("variations_engine"), _bk_pc,
                                       str(_gl_pc), "sales", _ms_pc)
                            _jobs_pc.append((_sig_pc, _w_pc, _v["split"].copy()))
                        _nvar_pc = max(len(_jobs_pc), 1)

                        # [FN-347]
                        def _log_pc(_w, _sta):
                            log(f"      dial {int(round(_w * 100))}: "
                                f"{int(_sta.get('pools', 0)):,} pools "
                                f"(fidelity {_sta.get('global_accuracy', 0):.1f}%)")

                        # The dial compressions are INDEPENDENT and DETERMINISTIC, so run them in
                        # parallel PROCESSES via joblib's loky backend (robust on macOS + Windows
                        # spawn, and inside Streamlit). ANY failure → sequential fallback, which is
                        # byte-identical. Only worth the process overhead for >1 dial.
                        # Compression is the single biggest stage (~two-thirds of the run), so this
                        # spans the widest slice of the bar (0.35 → 0.95). Using joblib's ORDERED
                        # generator lets the bar tick up as each dial finishes, keeping the ETA live
                        # through the long stage.
                        #
                        # CORE-AWARE CHOICE (byte-identical either way): this "across-dials" branch
                        # gives each dial only ONE core (its inner k-ary budget search runs SERIALLY),
                        # so it caps at `_nvar_pc` cores no matter how many exist. When the machine has
                        # plenty of cores (>= 2·dials) the SEQUENTIAL fallback is FASTER — it runs one
                        # dial at a time but hands each dial's inner k-ary search ALL cores (probes many
                        # budgets per round → far fewer rounds) AND uses the disk cache. Crossover is
                        # ~4 cores for 2 dials (log_{C+1}(R)·dials < log2(R) ⇔ C>3). So only take the
                        # across-dials path when cores are scarce; otherwise fall through to sequential.
                        _pc_cpu = int(os.cpu_count() or 1)
                        _pc_results = None
                        if _nvar_pc > 1 and _pc_cpu < 2 * _nvar_pc:
                            try:
                                from joblib import Parallel, delayed
                                from impact_calcs import pool_targeted_core as _ptc
                                _progress(_f_eng_end, f"Compressing pools 0/{_nvar_pc} (parallel)…")
                                _t_par = _pt.time()
                                _gen_pc = Parallel(
                                    n_jobs=min(_nvar_pc, os.cpu_count() or 1), backend="loky",
                                    return_as="generator")(
                                    delayed(_ptc)(
                                        _spl, target_pools=_maxN_pc, wallet_ctx=_wc_pc,
                                        brand_name=_bn_pc, brand_key=_bk_pc, go_live=str(_gl_pc),
                                        mid_list_path=_mid_list_pc, mode="sales",
                                        method=str(ss.get("compress_method", "ward")),
                                        allocation=str(ss.get("compress_allocation", "knapsack")))
                                    for (_sig_pc, _w_pc, _spl) in _jobs_pc)
                                _pc_results = []
                                for _i_pc, _res_pc in enumerate(_gen_pc):
                                    _sig_pc, _w_pc, _ = _jobs_pc[_i_pc]   # generator preserves order
                                    _pc_results.append((_sig_pc, _w_pc, _res_pc))
                                    _progress(_f_eng_end + (1.0 - _f_eng_end) * (_i_pc + 1) / _nvar_pc,
                                              f"Compressing pools {_i_pc + 1}/{_nvar_pc} (parallel)…")
                                log(f"   parallel compression ({_nvar_pc} workers) finished in "
                                    f"{_pt.time() - _t_par:.1f}s")
                            except Exception as _pe:  # noqa: BLE001
                                log(f"   parallel compression unavailable ({type(_pe).__name__}: {_pe}); "
                                    "falling back to sequential.")
                                _pc_results = None

                        if _pc_results is not None:
                            # ss['_pool_comp'] was reset to {} above; store the parallel results.
                            _cache_pc = ss.get("_pool_comp") or {}
                            for _sig_pc, _w_pc, (_lng_pc, _sta_pc) in _pc_results:
                                _cache_pc[_sig_pc] = {"long": _lng_pc, "stats": _sta_pc}
                                _log_pc(_w_pc, _sta_pc)
                            ss["_pool_comp"] = _cache_pc
                        else:
                            # Sequential fallback (also the path when there is only one dial).
                            _t_seq_pc = _pt.time()   # compression-only timer (parallel path logs its own)
                            for _iv_pc, (_sig_pc, _w_pc, _spl) in enumerate(_jobs_pc, 1):
                                _progress(_f_eng_end + (1.0 - _f_eng_end) * (_iv_pc - 1) / _nvar_pc,
                                          f"Compressing pools {_iv_pc}/{_nvar_pc}…")
                                try:
                                    _lng_pc, _sta_pc = pool_targeted_compression(
                                        ss, _spl, target_pools=_maxN_pc, sig=_sig_pc,
                                        wallet_ctx=_wc_pc, brand_name=_bn_pc, brand_key=_bk_pc,
                                        go_live=str(_gl_pc), mid_list_path=_mid_list_pc, mode="sales")
                                    _log_pc(_w_pc, _sta_pc)
                                except Exception as _pce:  # noqa: BLE001
                                    log(f"      dial {int(round(_w_pc * 100))}: compression FAILED "
                                        f"({type(_pce).__name__}: {_pce}) — tab 3 will compute it on Export.")
                            log(f"   compression (sequential, {_nvar_pc} dial(s), "
                                f"method={ss.get('compress_method', 'ward')}/"
                                f"{ss.get('compress_allocation', 'knapsack')}) finished in "
                                f"{_pt.time() - _t_seq_pc:.1f}s")

                    _stage_end()
                    if _t6_0 is not None:   # persist compression secs → next run's adaptive ETA
                        try:
                            _gp = dict(ss.get("ga_perf") or {})
                            # EMA-smooth so a single slow/fast (cache-hit) run doesn't poison the
                            # next estimate — it converges to the typical compression time.
                            _new_cs = float(_pt.time() - _t6_0)
                            _old_cs = float(_gp.get("compress_secs", _new_cs) or _new_cs)
                            _gp["compress_secs"] = 0.5 * _old_cs + 0.5 * _new_cs
                            _gp["pool_target"] = int(ss.get("max_configs", 0) or 0)   # scales next run's compression ETA
                            ss["ga_perf"] = _gp
                            _save_ga_perf(_gp)
                        except Exception:  # noqa: BLE001
                            pass
                    _total_run_secs = _pt.time() - _run_t0
                    log(f"✅ Total run time {_total_run_secs:.1f}s")
                    # Persist the end-to-end wall time + the GA's share, so the tab-2 readout can
                    # estimate WHOLE-RUN time (data + engine + enforcement + compression), not just
                    # the candidate search. ga_secs may be 0 (non-genetic / cached) → guard downstream.
                    try:
                        ss["last_total_secs"] = float(_total_run_secs)
                        ss["last_ga_wall_secs"] = float(ss.get("last_ga_secs", 0.0) or 0.0)
                    except Exception:  # noqa: BLE001
                        pass

                    # Auto-save a reproducible run bundle (timestamped folder under runs/): the exact
                    # settings used + this run's full log. Best-effort, once per run; never breaks it.
                    try:
                        from routing_optimiser.run_bundle import write_run_bundle as _wrb
                        _rb_cfg = {
                            "engine": str(engine_key),
                            "dials": [v.get("weight") for v in (ss.get("variations") or [])],
                            "vamp_cap": (float(vamp_cap) if vamp_cap is not None else None),
                            "max_pools": int(ss.get("max_configs", 0) or 0),
                            "compress_method": str(ss.get("compress_method", "ward")),
                            "compress_allocation": str(ss.get("compress_allocation", "knapsack")),
                            "ga_search": "breach_targeted+smart_init+adaptive_lambda+diversity_archive (always on)",
                            "ga_risk_aversion": 0.0,   # dial removed — fixed at the revenue-shaped endpoint
                            "ga_breach_fixed": 0.0,   # input removed — fixed at 0
                            "ga_band_mult": 1.0,      # input removed — fixed at 1.0
                            "ga_generations": int(ss.get("ga_generations", 80) or 80),
                            "ga_pop_override": int(ss.get("ga_pop_override", 0) or 0),
                            "ga_perf": ss.get("ga_perf"),
                            "total_secs": round(_pt.time() - _run_t0, 1),
                        }
                        _rb_folder = _wrb(os.path.join(PROJECT_ROOT, "runs"), _rb_cfg,
                                          log=log_lines, name=str(engine_key), keep=30)
                        log(f"   run bundle saved: {_rb_folder}")
                    except Exception as _rbe:  # noqa: BLE001
                        log(f"   [Warning] run bundle save skipped ({type(_rbe).__name__}: {_rbe})")
                    _progress(1.0, "Done")
                    status.update(label="Show technical details", state="complete", expanded=False)
                    try:
                        _eta_slot.markdown(
                            "<div style='font-size:1.4rem; font-weight:800; line-height:1.1; "
                            "color:var(--tav-green-dark); padding:0.1rem 0 0.35rem 0;'>"
                            f"✓ Done in {int(_pt.time() - _run_t0)}s "
                            f"<span style='font-size:0.9rem; font-weight:600; color:var(--tav-muted);'>"
                            f"· {len(variations)} variations ready</span></div>",
                            unsafe_allow_html=True)
                    except Exception:  # noqa: BLE001
                        pass
                    st.success("Variations ready — open **3 · Split, outputs & impact**.")
                    # Trigger ONE rerun after this compute so the top-of-script readiness gate
                    # (_HAS_RUN) re-evaluates with the new variations — otherwise tabs 3 & 4 stay
                    # greyed/locked until the next click, because the gate is computed BEFORE this
                    # compute runs in the same pass. Flag is set here (success only) and fired after
                    # the finally block, so st.rerun()'s internal exception isn't caught as a failure.
                    ss["_engine_needs_rerun"] = True
                except Exception as exc:  # noqa: BLE001
                    import traceback as _tb
                    _fulltb = _tb.format_exc()
                    ss["_tab3_error"] = {"type": type(exc).__name__, "msg": str(exc), "tb": _fulltb}
                    # Also dump the failure + full traceback INTO the run log, so it's captured in
                    # the single copyable log you share (not just the separate error widget).
                    try:
                        _stage_end()
                        log(f"✗ RUN FAILED after {_pt.time() - _run_t0:.1f}s")
                        log("═══════════════════════════ RUN FAILED ═══════════════════════════")
                        log(f"   {type(exc).__name__}: {exc}")
                        for _ln in _fulltb.rstrip().splitlines():
                            log("   " + _ln)
                        log("═══════════════════════════════════════════════════════════════════")
                    except Exception:  # noqa: BLE001
                        pass
                    status.update(label="Run failed — technical details", state="error", expanded=True)
                    try:
                        _pbar.empty()   # clear the % bar so it doesn't look stuck mid-run
                        _eta_slot.markdown(
                            "<div style='font-size:1.3rem; font-weight:800; color:var(--tav-red); "
                            "padding:0.1rem 0 0.35rem 0;'>✕ Run failed — see details below</div>",
                            unsafe_allow_html=True)
                    except Exception:  # noqa: BLE001
                        pass
                finally:
                    root_logger.removeHandler(handler)
                    root_logger.setLevel(prev_level)
                    try:                        # persist the full log so it survives tab switches
                        ss["last_run_log"] = "\n".join(log_lines)
                    except Exception:  # noqa: BLE001
                        pass

        # Un-grey tabs 3 & 4 the instant the run finishes: rerun once so the readiness gate at the
        # top re-evaluates with the freshly-stored variations. Fires only after a successful compute
        # (flag set in the try above); popped so it can never loop. Safe here — outside the compute's
        # try/except, so st.rerun()'s control-flow exception isn't misread as a run failure.
        if ss.pop("_engine_needs_rerun", False):
            st.rerun()

        # Keep the last run log visible after switching tabs: Streamlit clears the live status
        # container on every rerun, so when we're NOT mid-run re-render the stored text.
        if not submit_engine and ss.get("last_run_log"):
            with _run_log_slot:
                with st.expander("Show technical details (last run)", expanded=False):
                    st.code(ss["last_run_log"], language="log")

        if ss.get("_tab3_error"):
            err = ss["_tab3_error"]
            st.error(f"{err['type']}: {err['msg']}")
            try:
                from routing_optimiser import sql_runner as _sr
                st.caption(f"sql_runner build: `{getattr(_sr, '__build__', 'UNKNOWN — stale bytecode?')}`")
            except Exception: pass
            st.markdown("**Full traceback:**")
            st.code(err["tb"])
            if st.button("Dismiss error"):
                ss.pop("_tab3_error", None)
                st.rerun()
            st.stop()
