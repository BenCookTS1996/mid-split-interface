"""Tab 3 — Split, outputs & impact.

Originally split out of streamlit_app.py into its own file (since evolved) so the main
script stays small. streamlit_app.py calls `render()` from inside `with tab_imp:`.
"""
from __future__ import annotations

import calendar
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from routing_optimiser import load_success_data, run_sql_file
from impact_calcs import (_c_prepost_granular, _c_read_parquet, _mtime, build_kill_eff,
                          build_split_exports, compute_vamp_post_by_mid, enforced_prop_items,
                          enforced_split_frame, mid_revenue_month_table, mid_table_from_granular,
                          pool_targeted_compression, projection_cache_sig, rpgt_avg_ticket,
                          rpgt_currency_avg_ticket)

# 19hh: ONE source for (vampMid, currency) wallet/USA capability — the three delivery
# projections below read it so tab 3 and the engine mask at the same grain. See §14 of
# docs/scope_exploration_floor_in_search.md.
from app_common import capability_pairs as _cap_pairs
from app_common import (ensure_cols, load_mid_list, _norm_cols, _map_to_bank, _renorm_share, run_company,
                        input_json_path,  # 19ft: ONE resolver for config/inputs
                        _fid2vamp_from)  # memoised MID reader + shared helpers
from app_common import (ss, PROJECT_ROOT, SQL_DIR, CACHE_DIR, GCP_PROJECT, DEFAULT_GATEWAY_FIDS,
                        HAS_PLOTLY, _ensure_base_30d_metrics, _impact_eval_frame, _ink_caption,
                        _switched_off_gateways, _vamp_off_gateways, _unknown_apply_to,
                        _locked_panel, _split_df_to_xlsx_bytes)


# [FN-347b] Scheme-normalisation shim. This whole tab is written against the VISA export schema
# (vampMid / VAMP_* / VI_Txn_* and 'vamp_'-prefixed filenames). A Mastercard run emits the SAME
# data under Mastercard names (mastercardMid / CB_* / MC_Txn_* and 'cb_'/'mc_cb_'-prefixed
# filenames). Rather than rename column references in ~2100 lines, we normalise at the load points:
# filenames resolve to their Mastercard twin, and columns are renamed to the Visa vocabulary on the
# way in. Both helpers are no-ops on a Visa run.
_MC_TWIN_FILES = {
    "vamp_t_period_export.csv": "cb_t_period_export.csv",
    "vamp_t_period_prorata_export.csv": "mc_cb_t_period_prorata_export.csv",
}


def _normalise_scheme_columns(df):
    """Rename Mastercard export columns to their Visa equivalents. No-op unless the frame is
    Mastercard-shaped (i.e. carries 'mastercardMid'). Covers bin_rpgt / mid_level / t-period /
    pro-rata / effective-rate schemas in one pass."""
    if df is None or getattr(df, "empty", True) or "mastercardMid" not in getattr(df, "columns", []):
        return df
    ren = {}
    for c in df.columns:
        nc = c
        if c == "mastercardMid":
            nc = "vampMid"
        elif c == "cbPre":
            nc = "vampPre"
        elif c == "cbRatioPre":
            nc = "vampRatioPre"
        elif c == "cbCount":
            nc = "vampCount"
        elif c == "MC_Txn_Count":
            nc = "VI_Txn_Count"
        else:
            # Order matters: strip the FC_ prefixes before the bare MC_Txn_/CB_ ones.
            nc = nc.replace("FC_MC_Txn_", "FC_VI_Txn_").replace("FC_CB_", "FC_VAMP_")
            nc = nc.replace("MC_Txn_Pre", "VI_Txn_Pre").replace("MC_Txn_Post", "VI_Txn_Post")
            nc = (nc.replace("CB_Pre", "VAMP_Pre").replace("CB_Post", "VAMP_Post")
                    .replace("CB_Diff", "VAMP_Diff"))
            nc = nc.replace("Forecast_CBs", "Forecast_VAMPs").replace("Sim_CBs", "Sim_VAMPs")
        if nc != c:
            ren[c] = nc
    return df.rename(columns=ren) if ren else df


def _scheme_norm_export(out_dir, visa_name):
    """Return a path to a VISA-schema CSV for `visa_name`.

    Visa run -> the file as-is. Mastercard run -> read the Mastercard-named twin, rename its
    columns to the Visa vocabulary, and cache a '_shim_'-prefixed copy next to it (regenerated only
    when the source is newer), so downstream Visa-hardcoded readers (incl. impact_calcs, which reads
    the path itself) work untouched. Returns the (absent) Visa path if neither file exists, so
    callers keep their existing 'No export found' behaviour."""
    _visa_p = os.path.join(out_dir or "", visa_name)
    if os.path.exists(_visa_p):
        return _visa_p
    _twin = _MC_TWIN_FILES.get(visa_name)
    if not _twin:
        return _visa_p
    _mc_p = os.path.join(out_dir or "", _twin)
    if not os.path.exists(_mc_p):
        return _visa_p
    _shim_p = os.path.join(out_dir or "", "_shim_" + visa_name)
    try:
        if (not os.path.exists(_shim_p)) or (os.path.getmtime(_shim_p) < os.path.getmtime(_mc_p)):
            _normalise_scheme_columns(pd.read_csv(_mc_p)).to_csv(_shim_p, index=False)
        return _shim_p
    except Exception:  # noqa: BLE001 - on any failure, fall back to the (absent) Visa path
        return _visa_p


# [FN-348]
def render():
    # `out_dir` used to be a module global set by the (now-extracted) engine tab; re-derive it
    # locally from session_state so this tab is self-contained — identical value (ss.get).
    out_dir = ss.get("pipeline_out_dir")

    # Mastercard runs land in a '/mastercard/' output folder; the scheme shim normalises their
    # exports to the Visa column vocabulary (vampMid / VAMP / VI Txn) so this tab's logic works
    # unchanged. That vocabulary was leaking into the DISPLAYED table headers — relabel just the
    # header text back to Mastercard (mastercardMid / CB / MC Txn). Data column NAMES are untouched
    # (the shim depends on them). No-op on Visa runs.
    _is_mc_disp = "mastercard" in os.path.normpath(str(out_dir or "")).lower().split(os.sep)

    def _mc_hdr(_t):
        if not _is_mc_disp:
            return _t
        return (str(_t).replace("VAMP", "CB").replace("VI Txn", "MC Txn")
                .replace("VI_Txn", "MC_Txn").replace("vampMid", "mastercardMid"))

    # VALIDATE-MODE single source of truth: the pipeline's granular bin_rpgt_impact_export.csv,
    # normalised to the granular frame shape mid_table_from_granular / mid_revenue_month_table expect
    # (vampMid, RPGT, [Currency], period, VAMP_Pre/Post, VI_Txn_Pre/Post). Scheme-agnostic (Visa
    # VAMP_/Txn_, Mastercard CB_/MC_Txn→Txn_). Used ONLY in validate mode so the Risk Impact +
    # Financial Impact tables tie exactly to the Validate Split table; the Tab 2 engine flow (a
    # PROPOSED split with no pipeline run) never calls this. Returns None if the file/cols are absent.
    def _validate_granular_from_bin_rpgt(_od):
        try:
            # ── 19ep: PREFER THE PRO-RATA EXPORT ──────────────────────────────────────────
            # Everything else in the pipeline baselines off the pro-rata export; this table
            # could not, because it is a PRE vs POST view and that file was baseline-only.
            # 19ep makes the pipeline emit `vampCount_Post` / `VI_Txn_Count_Post` (and the
            # Mastercard equivalents), so it can now — aggregated to bin_rpgt's own key set
            # (id, RPGT, BIN, Currency, period) so BOTH paths hand the code below an
            # identical shape and nothing downstream can tell which one it got.
            #
            # THE FALLBACK IS NOT OPTIONAL. Pro-rata exports written before 19ep have no Post
            # columns, and a missing Post must NEVER be read as a Post of zero — that renders
            # as "the split changed nothing", which is a wrong answer rather than a missing
            # one. So the Post columns must BOTH be present, or this reads bin_rpgt as before.
            _pp = os.path.join(_od or "", "vamp_t_period_prorata_export.csv")
            _mp_ = os.path.join(_od or "", "mc_cb_t_period_prorata_export.csv")
            _bd = None
            _src = None
            for _pth, _idc0, _pre, _post, _tpre, _tpost in (
                    (_pp, "vampMid", "vampCount", "vampCount_Post",
                     "VI_Txn_Count", "VI_Txn_Count_Post"),
                    (_mp_, "mastercardMid", "cbCount", "cbCount_Post",
                     "MC_Txn_Count", "MC_Txn_Count_Post")):
                if not os.path.exists(_pth):
                    continue
                _pd_ = pd.read_csv(_pth)
                if not {_post, _tpost}.issubset(_pd_.columns):
                    continue                      # pre-19ep export — no Post, use bin_rpgt
                _rpc0 = "RPGT" if "RPGT" in _pd_.columns else "rpgt"
                _k0 = [c for c in (_idc0, _rpc0, "BIN", "Currency", "period")
                       if c in _pd_.columns]
                _vals = [c for c in (_pre, _post, _tpre, _tpost) if c in _pd_.columns]
                # Sum over t / Country / paymentMethodProvider — the grain the pro-rata export
                # has and bin_rpgt does not.
                _bd = _pd_.groupby(_k0, as_index=False, observed=True)[_vals].sum()
                _bd = _bd.rename(columns={
                    _pre: ("VAMP_Pre" if _idc0 == "vampMid" else "CB_Pre"),
                    _post: ("VAMP_Post" if _idc0 == "vampMid" else "CB_Post"),
                    _tpre: "Txn_Pre", _tpost: "Txn_Post"})
                _src = os.path.basename(_pth)
                break
            if _bd is None:
                _bp = os.path.join(_od or "", "bin_rpgt_impact_export.csv")
                if not os.path.exists(_bp):
                    return None
                _bd = pd.read_csv(_bp)
                _src = "bin_rpgt_impact_export.csv"
            try:
                st.caption(f"Validate granular source: `{_src}`"
                           + ("" if _src.endswith("prorata_export.csv") else
                              " — the pro-rata export carries no POST columns yet, so this "
                              "table still reads the legacy impact export. Re-run the forecast "
                              "to move it onto the same file as everything else."))
            except Exception:  # noqa: BLE001 - a caption must never break the table
                pass
            _idc = ("vampMid" if "vampMid" in _bd.columns
                    else ("mastercardMid" if "mastercardMid" in _bd.columns else None))
            _vpre = "VAMP_Pre" if "VAMP_Pre" in _bd.columns else "CB_Pre"
            _vpost = "VAMP_Post" if "VAMP_Post" in _bd.columns else "CB_Post"
            _rpc = "RPGT" if "RPGT" in _bd.columns else ("rpgt" if "rpgt" in _bd.columns else None)
            if _idc is None or _rpc is None or not all(
                    c in _bd.columns for c in (_vpre, _vpost, "Txn_Pre", "Txn_Post", "period")):
                return None
            _ren = {_idc: "vampMid", _rpc: "RPGT", _vpre: "VAMP_Pre", _vpost: "VAMP_Post",
                    "Txn_Pre": "VI_Txn_Pre", "Txn_Post": "VI_Txn_Post"}
            _g = _bd.rename(columns=_ren)
            if "Currency" not in _g.columns and "currency" in _g.columns:
                _g = _g.rename(columns={"currency": "Currency"})
            _keep = ["vampMid", "RPGT", "period", "VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]
            if "Currency" in _g.columns:
                _keep.append("Currency")
            _g = _g[_keep].copy()
            _g["vampMid"] = _g["vampMid"].astype(str)
            _g["period"] = pd.to_numeric(_g["period"], errors="coerce").fillna(-1).astype(int)
            return _g
        except Exception:  # noqa: BLE001 - fall back to the projection on any error
            return None

    # --- Populate impact from a 'Validate Split' run: build a single "variation" from the
    #     VALIDATED split (parsed from the exported rule files) + the attempts/success window,
    #     so this tab's pre/post tables + bridges render WITHOUT running the routing engine.
    #     Fails loudly (this path can't be exercised offline — BigQuery + real rules). ---
    _vpr = ss.pop("validate_populate_req", None)
    if _vpr:
        with st.status("Populating impact from the validated split…", expanded=True) as _vst:
            try:
                from routing_optimiser.s5_deliver.backup_blend import parse_rules_to_split as _prs
                _split_v = _prs(_vpr.get("rules_dir", ""))
                if getattr(_split_v, "empty", True):
                    raise ValueError(f"No routing rules parsed from {_vpr.get('rules_dir')!r}.")
                _scheme_v = str(_vpr.get("scheme", "visa") or "visa")
                _sqlp = {"START_DATE": _vpr.get("attempts_start"), "END_DATE": _vpr.get("attempts_end"),
                         "COMPANY": _vpr.get("company"), "CARD_SCHEME": _scheme_v,
                         "BIN_PREFIX": "4" if _scheme_v == "visa" else "5",
                         "GATEWAY_FIDS": DEFAULT_GATEWAY_FIDS}
                _sqlf = os.path.join(SQL_DIR, "attempts_success.sql")
                if not os.path.exists(_sqlf):
                    raise FileNotFoundError("attempts_success.sql not found.")
                _ap, _ = run_sql_file(_sqlf, CACHE_DIR, use_cache=True, fallback_csv=None,
                                      project=GCP_PROJECT, params=_sqlp)
                _adf_v = load_success_data(_ap)
                # RPGT canonicalisation (incl. 'Upgrade'→'Upgrades' and the legacy 'Monthly Intiial'
                # typo) is handled upstream in load_success_data (schema.SCENARIO_TO_RPGT); the fixed
                # attempts_success.sql now emits canonical names and the impact join is case-
                # insensitive, so no per-tab RPGT remap is needed here.
                # Grain reconciliation. Exported rules mix two grains: some profiles name explicit
                # BINs (the annual sheet), the rest use a catch-all row (BIN == "Other"). Set each
                # attempt's bank to the explicit BIN when the split names it for that
                # (rpgt, currency); OTHERWISE keep the attempt's real issuing-bank name (from
                # load_success_data). Keeping the real bank name for fallback traffic — rather than
                # collapsing it to "Other" — means the Mid Detail / bank breakdowns show the actual
                # banks, while the numeric-vs-name split still cleanly separates BIN-specific
                # overrides (numeric) from fallback cells (bank names) everywhere downstream.
                if "bin" in _adf_v.columns and "bin" in _split_v.columns:
                    _sv_r = _split_v["rpgt"].astype(str).str.strip().str.lower()
                    _sv_c = _split_v["currency"].astype(str).str.strip().str.lower()
                    _sv_b = _split_v["bin"].astype(str).str.strip()
                    _explicit_bins: dict = {}   # (rpgt,currency) -> {explicit numeric BINs}
                    for _r, _c, _b in zip(_sv_r, _sv_c, _sv_b):
                        if _b.replace(".", "", 1).isdigit():
                            _explicit_bins.setdefault((_r, _c), set()).add(_b)
                    _a_r = _adf_v["rpgt"].astype(str).str.strip().str.lower()
                    _a_c = _adf_v["currency"].astype(str).str.strip().str.lower()
                    _a_b = _adf_v["bin"].astype(str).str.strip()
                    _orig_bank = _adf_v["bin"].astype(str).to_numpy()   # real issuing-bank name
                    _mapped = []
                    for _k, _b, _ob in zip(zip(_a_r, _a_c), _a_b, _orig_bank):
                        _mapped.append(_b if _b in _explicit_bins.get(_k, ()) else _ob)
                    _adf_v["bin"] = _mapped
                ss["adf"] = _adf_v
                ss.setdefault("bin_to_bank", {})   # v1: raw-bank alignment (BIN on both sides)
                ss["opt_by_rpgt"] = True
                ss.pop("cached_base_30d_metrics", None)
                _cache_v = _ensure_base_30d_metrics()
                if _cache_v is None:
                    raise ValueError("Base 30-day metrics could not be built from the attempts data.")
                # Volume basis + actual observed baseline routing. The parsed rules carry neither
                # cell_volume nor baseline_share. Build the split the impact/bridge is scored on
                # from two kinds of profile:
                #   • BIN-specific (explicit numeric BIN — here only Annual Sub Renewal) → the
                #     exported OVERRIDE routing (proposed share) measured against the OBSERVED
                #     per-BIN baseline. These are the only profiles that move the bridge.
                #   • catch-all ("Other") → NOT a routing change: for the ~96% of traffic with no
                #     BIN-specific rule the routing is dictated by the LIVE ACTUALS. Rebuild these
                #     profiles straight from the observed 30D attempts (ALL gateways, incl. any not in
                #     the sheet), with baseline == proposed → exactly 0 impact and a distribution
                #     that matches actuals.
                # (cell_att / gw_att / attempts arrive as pandas *nullable* Int64; convert to numpy
                # float64 throughout, else a downstream .to_numpy() is object and the share renorm
                # does Python division → ZeroDivisionError on zero-attempt cells.)
                def _lc(s):
                    return s.astype(str).str.strip().str.lower()

                def _is_num(s):
                    return _lc(s).str.replace(".", "", 1, regex=False).str.isdigit()
                _keep = ["rpgt", "currency", "bin", "gateway", "share", "baseline_share", "cell_volume"]

                # BIN-specific override rows: proposed share vs observed per-BIN baseline.
                _spec = _split_v[_is_num(_split_v["bin"]).to_numpy()].copy()
                if not _spec.empty:
                    _cagg = _cache_v["cell_agg"]
                    _gagg = _cache_v["gw_agg"]
                    _spec["rpgt_join"] = _lc(_spec["rpgt"])
                    _spec["currency_join"] = _lc(_spec["currency"])
                    _spec["bin_join"] = _lc(_spec["bin"])
                    _spec["gateway_join"] = _lc(_spec["gateway"])
                    _spec = _spec.merge(_cagg[["rpgt_join", "currency_join", "bin_join", "cell_att"]],
                                        on=["rpgt_join", "currency_join", "bin_join"], how="left")
                    _spec = _spec.merge(_gagg[["rpgt_join", "currency_join", "bin_join", "gateway_join", "gw_att"]],
                                        on=["rpgt_join", "currency_join", "bin_join", "gateway_join"], how="left")
                    _cv = pd.to_numeric(_spec["cell_att"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
                    _gv = pd.to_numeric(_spec["gw_att"], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
                    _cv = np.where(np.isnan(_cv), 0.0, _cv)
                    _gv = np.where(np.isnan(_gv), 0.0, _gv)
                    _spec["cell_volume"] = _cv
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _spec["baseline_share"] = np.where(_cv > 0, _gv / _cv, 0.0)
                    _spec["share"] = pd.to_numeric(_spec["share"], errors="coerce").fillna(0.0).astype("float64")
                    _spec = _spec[_keep]
                else:
                    _spec = pd.DataFrame(columns=_keep)

                # Catch-all profiles rebuilt from the observed live actuals (all gateways).
                _adf_o = _adf_v[~_is_num(_adf_v["bin"]).to_numpy()].copy()
                _adf_o["_a"] = pd.to_numeric(_adf_o.get("attempts", 0), errors="coerce").fillna(0.0).astype("float64")
                _obs = _adf_o.groupby(["rpgt", "currency", "bin", "gateway"], as_index=False)["_a"].sum()
                if not _obs.empty:
                    _ct = _obs.groupby(["rpgt", "currency", "bin"])["_a"].transform("sum").to_numpy(dtype="float64")
                    _av = _obs["_a"].to_numpy(dtype="float64")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _osh = np.where(_ct > 0, _av / _ct, 0.0)
                    _obs["share"] = _osh
                    _obs["baseline_share"] = _osh
                    _obs["cell_volume"] = _ct
                    _obs = _obs[_keep]
                else:
                    _obs = pd.DataFrame(columns=_keep)

                _split_v = pd.concat([_spec, _obs], ignore_index=True)
                _eval_v = _impact_eval_frame(_split_v, _cache_v, by_rpgt=True)
                ss["variations"] = [{"weight": 0.0, "split": _split_v, "settings": {}, "eval_df": _eval_v}]
                ss["split"] = _split_v
                ss["selected_variation_weight"] = 0.0
                ss["variations_engine"] = "validate"
                ss["_comp_eval_cache"] = {}
                ss.setdefault("sr", pd.DataFrame())
                ss.setdefault("problems", {})
                ss.setdefault("mid_constraints", [])
                # Count the ConnectorPools the folder's BIN-specific rules generate (sales mode,
                # read straight from the sheets — matches tab 4 · Generate configs) so the Pools
                # card reports the real deployed count, not the observed-routing compression.
                # count_only skips the payload build.
                ss["validate_folder_pool_count"] = None
                try:
                    from routing_optimiser.s5_deliver.connector_pool_configs import (
                        generate_configs as _gcfg, company_to_brand_key as _c2b, BRANDS as _BRANDS,
                        scheme_code as _scheme_code)
                    import glob as _glob_v
                    _rdir = _vpr.get("rules_dir", "")
                    _files = sorted(_glob_v.glob(os.path.join(_rdir, "*.xlsx"))
                                    + _glob_v.glob(os.path.join(_rdir, "*.xls"))
                                    + _glob_v.glob(os.path.join(_rdir, "*.csv")))
                    _frs = []
                    for _fp in _files:
                        try:
                            _frs.append(pd.read_excel(_fp) if _fp.lower().endswith((".xlsx", ".xls"))
                                        else pd.read_csv(_fp))
                        except Exception:  # noqa: BLE001
                            continue
                    if _frs:
                        _allr = pd.concat(_frs, ignore_index=True)
                        _bk = _c2b(_vpr.get("company", "TotalAV"))
                        _bnm = _BRANDS.get(_bk, {}).get("name", "TotalAV")
                        _exp = {}
                        if "RPGT" in _allr.columns:
                            for _rp, _sub in _allr.groupby("RPGT"):
                                _exp[(_bnm, str(_rp))] = _sub.reset_index(drop=True)
                        _, _fc = _gcfg(_exp, _bk, "000000",
                                       scheme=_scheme_code(_vpr.get("scheme", "visa")),
                                       mode="sales", count_only=True)
                        ss["validate_folder_pool_count"] = int(_fc.get("total", 0))
                except Exception:  # noqa: BLE001
                    ss["validate_folder_pool_count"] = None
                _vst.update(state="complete", expanded=False)
            except Exception as _ve:  # noqa: BLE001
                import traceback as _vtb
                _vst.update(label="Populate-impact FAILED (validated split)", state="error", expanded=True)
                st.error(f"{type(_ve).__name__}: {_ve}")
                st.code(_vtb.format_exc())

    # CSS override: Forces the text typed inside Selectboxes to be dark/visible
    st.markdown("""<style>
        div[data-testid="stSelectbox"] div[data-baseweb="select"] * { color: #0B1F3A !important; font-weight: 500; }
        div[data-testid="stSelectbox"] input { color: #0B1F3A !important; }
    </style>""", unsafe_allow_html=True)

    if "variations" not in ss or "adf" not in ss:
        _locked_panel("Nothing to show yet. Either head to <b>2 · Routing engine</b> and click "
                      "<b>Compute split variations</b>, or run <b>Validate Split</b> "
                      "(under <b>1 · Baseline &amp; Validate</b>) — your split, outputs and impact "
                      "analysis will appear here.")
    else:
        variations = ss["variations"]
        weights = [v["weight"] for v in variations]

        # 30D baseline metrics (profile/gateway success rates, avg ticket, base totals)
        # — computed once and shared with the Routing-engine tab visuals.
        _ensure_base_30d_metrics()
        cache = ss["cached_base_30d_metrics"]
        base_att, base_succ, base_rev = cache["base_att"], cache["base_succ"], cache["base_rev"]
        cell_agg, gw_agg, adf_30d = cache["cell_agg"], cache["gw_agg"], cache["adf_30d_raw"]
        date_col = cache["date_col"]
        base_sr = base_succ / base_att if base_att > 0 else 0
        # Baseline revenue on the SAME basis as the new impact calc: value baseline
        # successes at the Bank×Currency avg ticket (not the raw actual amount), so
        # the card's change matches the Pre/Post Revenue columns below.
        base_rev_adj = float((pd.to_numeric(cell_agg.get("avg_ticket", 0), errors="coerce").fillna(0)
                              * pd.to_numeric(cell_agg.get("cell_succ", 0), errors="coerce").fillna(0)).sum())


# -------------- Slider Selection ---------------------------
        with st.container(border=True):
            # Dial narrowed 40% (1.5→0.9); metric cards narrowed 30% (1→0.7) with a
            # trailing spacer absorbing the freed width so the cards don't stretch.
            # _con_col (feasibility report) widened so all 9 columns show; the Export
            # Templates column is narrowed to compensate (its button/label wrap is fine).
            _sld_col, _m1c, _m2c, _cc_col, _cf_col, _con_col, _exp_col = st.columns(
                [0.9, 0.7, 0.7, 0.7, 0.7, 2.4, 0.6])
            # Per-MID constraint table renders into this slot (between the last card and the
            # Export button); it's filled later from the Risk-Impact projection.
            _con_slot = _con_col.container()
            _prev_w = ss.get("selected_variation_weight")
            # Default the dial to 0 (risk-minimised compliant endpoint); keep the user's pick after.
            _def_w = _prev_w if _prev_w in weights else min(weights)
            if len(weights) > 1:
                picked_w = _sld_col.select_slider(
                    "**Risk  ↔  Conversion**", options=weights,
                    value=_def_w,
                    format_func=lambda w: f"{int(round(w * 100))}",
                    help="Dial: safer routing ↔ more revenue.")
            else:
                # Single variation (dial 100 removed) — no dial to pick, and no static label shown.
                picked_w = weights[0]
            ss["selected_variation_weight"] = picked_w
            # Impact basis: always the Compressed Rules — the split trimmed so the generated pool
            # count stays within your target, i.e. what the exported configs actually deliver. The
            # 'No Compression' view was removed. With no target set (_maxN == 0) the split is
            # uncompressed by definition, so there's nothing to switch to.
            _maxN = int(ss.get("max_configs", 0) or 0)
            _basis_compressed = _maxN > 0

            chosen = variations[weights.index(picked_w)]
            split_ideal = chosen["split"].copy()
            # ss["split"] stays the IDEAL split — the export compresses from it. The impact
            # tables use ss["impact_split"], which follows the basis toggle below.
            ss["split"] = split_ideal; ss["settings"] = chosen["settings"]

            # -- Pool-count-targeted compression. 'Max pools' (tab 2) is now a TARGET POOL
            #    COUNT: the k-means is driven so the GENERATED pool count stays <= target.
            #    The search re-runs config generation, so it is EXPENSIVE — it runs only when
            #    a build/generate button is clicked and is cached in ss['_pool_comp'] by
            #    signature. The cards/impact-basis read that cache; _comp_* stay None until a
            #    matching build has been run. --
            _maxN = int(ss.get("max_configs", 0) or 0)          # NOW a target POOL count
            _wc_e = ss.get("wallet_ctx") or {}
            _fs_e0 = ss.get("forecast_settings", {}) or {}
            _company_e0 = str(_fs_e0.get("company", "TotalAV"))
            _gl_e0 = ss.get("split_go_live_date", date.today())
            try:
                from routing_optimiser.s5_deliver.connector_pool_configs import (
                    BRANDS as _POOL_BRANDS0, company_to_brand_key as _co2brand0)
                _brand_key_e = _co2brand0(_company_e0)
                _brand_name_e = _POOL_BRANDS0.get(_brand_key_e, {}).get("name", _company_e0)
            except Exception:  # noqa: BLE001
                _brand_key_e, _brand_name_e = "tav", _company_e0
            _mid_list_e = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
            _csrc = split_ideal.copy()
            if "cell_volume" not in _csrc.columns:
                _csrc["cell_volume"] = (_csrc.groupby(["rpgt", "currency", "bin"])["volume"].transform("sum")
                                        if "volume" in _csrc.columns else 1.0)
            _raw_cells = int(_csrc.groupby(["rpgt", "currency", "bin"]).ngroups)
            # Signature of everything the pool-targeted result depends on (tab 4 uses the
            # default 'sales' mode for its cards/export; tab 6 keys its own mode separately).
            _pool_sig = (float(picked_w), _maxN, ss.get("variations_engine"), _brand_key_e,
                         str(_gl_e0), "sales", round(float(_wc_e.get("max_share", 0.97)), 4))
            _pool_cache = ss.get("_pool_comp") or {}
            _comp_long = None; _comp_stats = None
            if _maxN > 0 and _pool_sig in _pool_cache:
                _comp_long = _pool_cache[_pool_sig]["long"]
                _comp_stats = _pool_cache[_pool_sig]["stats"]

            # [FN-349]
            def _run_pool_compression():
                """Compute + cache the pool-targeted compression for the CURRENT settings."""
                _si = split_ideal
                if ss.get("variations_engine") == "validate":
                    # Validate: only the BIN-specific overrides are a real deployed change — every
                    # other profile is the live-actuals fallback. Count pools from the BIN-specific
                    # rows only, so the Pools card reflects the splits actually being validated.
                    _num = (_si["bin"].astype(str).str.strip()
                            .str.replace(".", "", 1, regex=False).str.isdigit())
                    _si = _si[_num.to_numpy()].copy()
                return pool_targeted_compression(
                    ss, _si, target_pools=_maxN, sig=_pool_sig, wallet_ctx=_wc_e,
                    brand_name=_brand_name_e, brand_key=_brand_key_e, go_live=str(_gl_e0),
                    mid_list_path=_mid_list_e, mode="sales")

            # LAZY on-demand: the run precomputes only the default dial (0). If the Compressed basis is
            # selected for a dial that wasn't precomputed, compute it NOW (once) and cache it into
            # ss['_pool_comp'] so later views are instant. Same output as the eager precompute, deferred.
            if _maxN > 0 and _basis_compressed and _comp_long is None:
                try:
                    with st.spinner("Compressing pools for this dial (first view — cached after)…"):
                        _comp_long, _comp_stats = _run_pool_compression()
                    _pc = ss.get("_pool_comp") or {}
                    _pc[_pool_sig] = {"long": _comp_long, "stats": _comp_stats}
                    ss["_pool_comp"] = _pc
                except Exception as _ce:  # noqa: BLE001
                    st.caption(f"Compression failed for this dial ({type(_ce).__name__}); showing uncompressed.")
                    _comp_long = None

            # Apply the impact basis chosen beneath the dial (_basis_compressed set above).
            _impact_split = split_ideal
            if _basis_compressed and _comp_long is not None:
                # Carry the baseline (pre) split + volume so the impact's pre/post is correct;
                # gateways new to a profile via the cluster centroid get baseline_share 0.
                _cl = _comp_long.copy()
                if "baseline_share" in split_ideal.columns:
                    _bl = split_ideal[["rpgt", "currency", "bin", "gateway", "baseline_share"]].drop_duplicates(
                        ["rpgt", "currency", "bin", "gateway"])
                    # The compressed frame may ALREADY carry a baseline_share; dropping it first stops the
                    # merge from producing suffixed baseline_share_x/_y (which broke the plain-name access
                    # below with a KeyError). split_ideal is the authoritative pre-split baseline.
                    if "baseline_share" in _cl.columns:
                        _cl = _cl.drop(columns=["baseline_share"])
                    _cl = _cl.merge(_bl, on=["rpgt", "currency", "bin", "gateway"], how="left")
                    _cl["baseline_share"] = _cl["baseline_share"].fillna(0.0)
                _cl["volume"] = _cl["cell_volume"] * _cl["share"]
                _impact_split = _cl
            elif _basis_compressed and _comp_long is None:
                _basis_compressed = False   # compression unavailable/failed → uncompressed view
            ss["impact_split"] = _impact_split
            split = _impact_split   # everything below (charts, tables) uses the chosen basis

            # eval_df drives EVERY revenue / success-rate / bank / gateway view. Build it from the
            # ENFORCED + backup-blended split (post cap / wallet / USA + the backup catch-all
            # re-adds) — the SAME routing basis the VAMP projection uses — so those
            # charts reconcile with the risk tables instead of showing the raw optimiser split.
            # A Validate-Split run already carries the routed shares in its parsed rules, so it uses
            # the eval frame built during populate; the enforcement is not re-applied.
            # [FN-350]
            def _enforced_blended_eval_split(_spl):
                """Enforced (build_split_exports) + backup-blended gateway-grain split, at the
                ideal split's (parent-bank) grain, carrying baseline_share/volume for pre/post."""
                _wc = ss.get("wallet_ctx") or {}
                _enf = enforced_split_frame(
                    _spl, _brand_name_e, str(_gl_e0),
                    wallet_incapable=set(_wc.get("incapable", set())),
                    fid2vamp=_wc.get("fid2vamp"), mid_list_path=_mid_list_e,
                    usa_only=set(_wc.get("usa_only", set())),
                    country_pres=_wc.get("country_pres", {}),
                    max_share=float(_wc.get("max_share", 0.97)))
                if _enf is None or getattr(_enf, "empty", True):
                    return _spl
                _b2b = ss.get("bin_to_bank", {})
                _enf = _enf.copy()
                _enf["bin"] = _map_to_bank(_enf["bin"], _b2b).astype(str)
                _enf = _enf.groupby(["rpgt", "currency", "bin", "gateway"], as_index=False)["share"].mean()
                _renorm_share(_enf, ["rpgt", "currency", "bin"])
                # Backup catch-all re-adds (e.g. Braintree) at gateway grain, per (rpgt,currency,bank).
                _bc = ss.get("backup_catchall") or {}
                if _bc and os.environ.get("ROUTING_BACKUP_BLEND", "1") != "0":
                    from collections import defaultdict as _dd
                    from routing_optimiser.s5_deliver.backup_blend import blend_cell_shares as _bcs
                    _acc, _cnt = _dd(lambda: _dd(float)), _dd(int)
                    for (_cur, _rp, _pmp, _ct), _gw in _bc.items():
                        _cnt[(_cur, _rp)] += 1
                        for _g, _v in _gw.items():
                            _acc[(_cur, _rp)][str(_g).strip().lower()] += float(_v)
                    _pooled = {k: {g: v / max(_cnt[k], 1) for g, v in gw.items()} for k, gw in _acc.items()}
                    _rows = []
                    for (_rp, _cur, _bnk), _grp in _enf.groupby(["rpgt", "currency", "bin"]):
                        _spec = {str(r["gateway"]): float(r["share"]) for _, r in _grp.iterrows()}
                        _ca = _pooled.get((str(_cur).strip().lower(), str(_rp).strip().lower()), {})
                        _eff = _bcs(_spec, _ca) if _ca else _spec
                        for _g, _s in _eff.items():
                            _rows.append({"rpgt": _rp, "currency": _cur, "bin": _bnk, "gateway": _g, "share": _s})
                    if _rows:
                        _enf = pd.DataFrame(_rows)
                if "baseline_share" in _spl.columns:
                    # OUTER-merge so gateways that were routed PRE but dropped to 0 POST still appear
                    # (post share 0), and back-fill gateways new POST get baseline 0 — both show in Δ.
                    _bl = _spl[["rpgt", "currency", "bin", "gateway", "baseline_share"]].drop_duplicates(
                        ["rpgt", "currency", "bin", "gateway"])
                    _enf = _enf.merge(_bl, on=["rpgt", "currency", "bin", "gateway"], how="outer")
                    _enf["share"] = _enf["share"].fillna(0.0)
                    _enf["baseline_share"] = _enf["baseline_share"].fillna(0.0)
                # Carry per-profile volume through so the eval frame can size pre/post volume + revenue.
                # (The ideal split has `cell_volume`; the enforced split from build_split_exports does
                # NOT — without this, _impact_eval_frame's cell_volume/volume would be missing.)
                _cvsrc = None
                if "cell_volume" in _spl.columns:
                    _cvsrc = _spl.groupby(["rpgt", "currency", "bin"], as_index=False)["cell_volume"].first()
                elif "volume" in _spl.columns:
                    _cvsrc = (_spl.groupby(["rpgt", "currency", "bin"], as_index=False)["volume"].sum()
                              .rename(columns={"volume": "cell_volume"}))
                if _cvsrc is not None:
                    _enf = _enf.merge(_cvsrc, on=["rpgt", "currency", "bin"], how="left")
                    _enf["cell_volume"] = pd.to_numeric(_enf["cell_volume"], errors="coerce").fillna(0.0)
                    _enf["volume"] = _enf["cell_volume"] * _enf["share"]
                # Belt-and-suspenders: drop switched-off gateways (target=0, trx/both) from the eval
                # split and renormalise each profile's post-share, so a turned-off gateway can NEVER show
                # routed share in the revenue view regardless of how it entered (engine candidate,
                # backup pool, or the baseline outer-merge).
                try:
                    import json as _je
                    from routing_optimiser.s2_forecast.vamp_forecast_pipeline import _canonical_gateway as _cge
                    _ovp_e = input_json_path("gateway_volume_overrides.json")
                    _off_e = set()
                    if os.path.exists(_ovp_e):
                        with open(_ovp_e) as _fe:
                            _off_e = _switched_off_gateways(_je.load(_fe) or {})
                    if _off_e and not _enf.empty and "gateway" in _enf.columns:
                        _gc = _enf["gateway"].map(_cge).astype(str).str.strip().str.lower()
                        _enf = _enf[~_gc.isin(_off_e)].copy()
                        if "share" in _enf.columns:
                            _renorm_share(_enf, ["rpgt", "currency", "bin"])
                            if "cell_volume" in _enf.columns:
                                _enf["volume"] = _enf["cell_volume"] * _enf["share"]
                except Exception as _e:  # noqa: BLE001
                    st.caption(f"Switched-off renormalisation skipped ({type(_e).__name__}: {_e}).")
                return _enf

            _is_validate = (ss.get("variations_engine") == "validate")
            _eval_cache = ss.setdefault("_comp_eval_cache", {})
            _ek = (_pool_sig, bool(_basis_compressed), bool(ss.get("opt_by_rpgt", False)), bool(_is_validate))
            if _is_validate and "eval_df" in chosen:
                eval_df = chosen["eval_df"].copy()   # parsed-rules split already = routed shares
            elif _ek in _eval_cache:
                eval_df = _eval_cache[_ek].copy()
            else:
                try:
                    with st.spinner("Applying enforcement + backup-blend to the revenue view…"):
                        _eval_split = _enforced_blended_eval_split(split)
                    eval_df = _impact_eval_frame(_eval_split, cache, by_rpgt=bool(ss.get("opt_by_rpgt", False)))
                except Exception as _ese:  # noqa: BLE001
                    st.warning(f"Enforced/blended revenue view unavailable ({type(_ese).__name__}: {_ese}); "
                               "revenue & SR charts fall back to the raw split. Risk/VAMP tables are unaffected.")
                    eval_df = _impact_eval_frame(split, cache, by_rpgt=bool(ss.get("opt_by_rpgt", False)))
                if len(_eval_cache) >= 12:          # bound memory (evict oldest)
                    _eval_cache.pop(next(iter(_eval_cache)))
                _eval_cache[_ek] = eval_df.copy()

            # Alias these so downstream charts still work perfectly
            eval_df["exp_succ"] = eval_df["post_succ"]
            eval_df["exp_rev"] = eval_df["post_rev"]

            # Validate mode isolates the BIN-specific overrides: catch-all ("Other") profiles were set
            # to pre == post (status quo) in the eval frame, so the baseline the cards compare
            # against is the eval frame's own PRE (Σ pre_succ / Σ pre_rev), NOT the raw 30D actual.
            # This makes the SR / Revenue cards agree with the bridge — only BIN-specific rules move.
            if ss.get("variations_engine") == "validate":
                if "pre_succ" in eval_df.columns:
                    _pre_succ_t = float(pd.to_numeric(eval_df["pre_succ"], errors="coerce").fillna(0.0).sum())
                    base_sr = _pre_succ_t / base_att if base_att > 0 else 0.0
                if "pre_rev" in eval_df.columns:
                    base_rev_adj = float(pd.to_numeric(eval_df["pre_rev"], errors="coerce").fillna(0.0).sum())

            new_succ, new_rev = eval_df["post_succ"].sum(), eval_df["post_rev"].sum()
            exp_sr = new_succ / base_att if base_att > 0 else 0
            rev_change = new_rev - base_rev_adj

            # Hand-rendered red cards: BIG colour-coded change on top, small pre→post beneath,
            # everything inside the card. (st.metric can't put the coloured change first with a
            # sub-line inside the same card.) Help text shows on card hover.
            # [FN-351]
            def _rcard(col, label, big, big_color, small, tip=""):
                _t = (' title="' + str(tip).replace('"', "'") + '"') if tip else ""
                col.markdown(
                    f"<div{_t} style='background:var(--tav-red);border:2px solid var(--tav-red);"
                    f"padding:10px 12px;min-height:112px;display:flex;flex-direction:column;"
                    f"justify-content:center;'>"
                    f"<div style='font-size:12px;font-weight:700;color:var(--tav-ink);line-height:1.15;'>{label}</div>"
                    f"<div style='font-size:22px;font-weight:800;color:{big_color};line-height:1.2;"
                    f"margin:2px 0;'>{big}</div>"
                    f"<div style='font-size:10px;color:var(--tav-ink);line-height:1.2;'>{small}</div>"
                    f"</div>", unsafe_allow_html=True)

            _GRN, _RED, _INK = "#22C36B", "#C21F2E", "var(--tav-ink)"

            _srd = (exp_sr - base_sr) * 100.0
            _rcard(_m1c, "Expected Success Rate (30D)",
                   f"{'▲' if _srd >= 0 else '▼'} {_srd:+.2f} pp", _GRN if _srd >= 0 else _INK,
                   f"{base_sr:.2%} → {exp_sr:.2%}",
                   tip="Card payments approved out of every 100 attempts.")
            # Revenue on the SAME attempts×SR×AOV basis as the Success Rate card and the by-vampMid
            # revenue bridge (waterfall): pre/post = Σ (cell_att × share × gw_sr × avg_ticket) =
            # Σ pre_rev / post_rev from the enforced+blended eval frame (the bridge, _ev == eval_df,
            # sums the very same columns). The baseline uses the MODELLED pre_rev (not actual
            # successes) so the card reconciles EXACTLY with the bridge and the delta is pure routing.
            _rev_pre = float(eval_df["pre_rev"].sum()) if "pre_rev" in eval_df.columns else float(base_rev_adj)
            # Validate flow has no separate baseline split (parse_rules_to_split carries no
            # baseline_share → pre_rev sums to 0). Fall back to the actual 30D baseline revenue so
            # this card matches the SR card, which likewise compares against the actual base_sr.
            if ss.get("variations_engine") == "validate" and _rev_pre <= 0.0:
                _rev_pre = float(base_rev_adj)
            _rev_post = float(eval_df["post_rev"].sum()) if "post_rev" in eval_df.columns else float(new_rev)
            _rev_chg = _rev_post - _rev_pre
            _rcard(_m2c, "Expected Revenue (30D)",
                   f"{'▲' if _rev_chg >= 0 else '▼'} ${_rev_chg:+,.0f}",
                   _GRN if _rev_chg >= 0 else _INK,
                   f"${_rev_pre:,.0f} → ${_rev_post:,.0f}",
                   tip="Expected successes × avg ticket (AOV) — same basis as the SR card and the "
                       "by-vampMid revenue bridge; pre uses the modelled baseline so they reconcile.")
            # --- Pools card: change vs ideal (fewer pools is good → green) ---
            if ss.get("variations_engine") == "validate":
                # Validate mode: show the ConnectorPools the folder's BIN-specific rules actually
                # generate — read straight from the exported sheets (sales mode), the SAME figure
                # tab 4 · Generate configs reports (computed in the populate step above).
                _p_folder = ss.get("validate_folder_pool_count")
                _p_txt = f"{int(_p_folder):,}" if _p_folder is not None else "—"
                _rcard(_cc_col, "Pools", _p_txt, _INK, "from BIN specific splits in this folder",
                       tip="ConnectorPool configs the BIN-specific rules in this folder generate "
                           "(sales mode; matches tab 4 · Generate configs).")
                _rcard(_cf_col, "Fidelity (30D)", "100.0%", _INK, "rules used as-is",
                       tip="No compression applied in Validate — the exported rules are generated as-is.")
            elif _comp_stats is not None:
                _p_before = int(_comp_stats.get("raw_pools", 0))
                _p_after = int(_comp_stats.get("pools", 0))
                _pdlt = _p_after - _p_before
                _psmall = f"{_p_before:,} → {_p_after:,}"
                if not _comp_stats.get("feasible", True):
                    _psmall += f"  (≤{_maxN:,} not reachable)"
                _rcard(_cc_col, "Pools", f"{'▼' if _pdlt <= 0 else '▲'} {_pdlt:+,}",
                       _GRN if _pdlt <= 0 else _INK, _psmall,
                       tip="ConnectorPool config files this will deploy (kept ≤ your target).")
                _rcard(_cf_col, "Fidelity (30D)", f"{_comp_stats.get('global_accuracy', 0):.1f}%",
                       _INK, "match to the full (uncompressed) set",
                       tip="How closely the trimmed rules match the full set.")
            elif _maxN > 0:
                _rcard(_cc_col, "Pools", "—", _INK, f"target ≤ {_maxN:,}",
                       tip="Click Export Templates to compute the compressed split.")
                _rcard(_cf_col, "Fidelity (30D)", "—", _INK, "computed on Export Templates",
                       tip="Computed when you Build & Export / Generate configs.")
            else:
                _rcard(_cc_col, "Pools", "—", _INK, "no compression",
                       tip="Set 'Max pools' in tab 2 to target a pool count.")
                _rcard(_cf_col, "Fidelity (30D)", "—", _INK, "no compression",
                       tip="How closely the trimmed rules match the full set.")
            
            # Export split templates — beside the cards (one .xlsx per Brand × RPGT).
            with _exp_col:
                # Add this CSS block to force primary button text to white
                st.markdown("""<style>
                    div[data-testid="stButton"] button[kind="primary"] * {
                        color: #FFFFFF !important;
                    }
                </style>""", unsafe_allow_html=True)
                
                _exp_split = ss.get("split")
                _fs_e = ss.get("forecast_settings", {}) or {}
                _brand_e = str(_fs_e.get("company", "TotalAV"))
                _gl_e = ss.get("split_go_live_date", date.today())
                # _maxN (target pools), _wc_e, _raw_cells, _comp_long/_comp_stats set above.
                if _exp_split is not None and not getattr(_exp_split, "empty", True):
                    # Isolate the export button + its heavy pool-compression / xlsx build in a
                    # FRAGMENT: clicking Export reruns ONLY this widget, so the charts and metric
                    # cards elsewhere on the page stay rendered and usable while the templates
                    # build (no full-page whiteout). The final st.rerun() (full app) fires only
                    # AFTER the build finishes, to refresh the Pools/Fidelity cards. Falls back
                    # to inline rendering on Streamlit builds without st.fragment.
                    # [FN-352]
                    def _export_ui():
                        # Signature of everything the export depends on. If it changes after a build,
                        # the ready-made zip is stale and the download is locked until a rebuild.
                        _exp_sig = (float(picked_w), _maxN, ss.get("variations_engine"),
                                    _brand_e, str(_gl_e))
                        if st.button("Export Templates", type="primary",
                                     key="export_splits_btn", use_container_width=True):
                            import io as _io
                            import zipfile as _zip
                            _ok = False
                            try:
                                # CACHE: if nothing the export depends on has changed since the last build,
                                # reuse the ready-made zip instead of rebuilding (instant re-click).
                                if ss.get("_split_export_zip") and ss.get("_split_export_sig") == _exp_sig:
                                    _ok = True
                                else:
                                    # Compute (+cache) the pool-targeted split now, if a target is set.
                                    _pt_long = None
                                    if _maxN > 0:
                                        with st.spinner("Finding the cell budget that hits your pool target…"):
                                            _pt_long, _ = _run_pool_compression()
                                    _gl_tag = pd.to_datetime(str(_gl_e)).strftime("%d_%m_%Y")

                                    # Gather every (arcname, split-DataFrame) job — build_split_exports runs
                                    # ONCE per basis — then serialise the xlsx bytes in PARALLEL (independent
                                    # + deterministic → joblib loky, the same cross-platform pattern as
                                    # compression). Byte-identical output; sequential fallback on any failure.
                                    # [FN-353]
                                    def _gather(_split_df, _subdir, _prefix):
                                        _ex = build_split_exports(
                                            _split_df, _brand_e, str(_gl_e),
                                            wallet_incapable=set(_wc_e.get("incapable", set())),
                                            fid2vamp=_wc_e.get("fid2vamp"),
                                            mid_list_path=os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                                            usa_only=set(_wc_e.get("usa_only", set())),
                                            country_pres=_wc_e.get("country_pres", {}),
                                            max_share=float(_wc_e.get("max_share", 0.97)))
                                        return [(f"{_subdir}/{_prefix}_{str(_rp).replace(' ', '_')}_{_br}_visa_{_gl_tag}.xlsx", _rdf)
                                                for (_br, _rp), _rdf in _ex.items()]
                                    _jobs = _gather(_exp_split, "ideal", "Rules")
                                    if _pt_long is not None:
                                        _jobs += _gather(_pt_long, "pool_targeted", "PoolTargeted_Rules")
                                    _n = len(_jobs)
                                    _written = None
                                    if len(_jobs) > 1:
                                        try:
                                            from joblib import Parallel, delayed
                                            _written = Parallel(n_jobs=min(len(_jobs), os.cpu_count() or 1),
                                                                backend="loky")(
                                                delayed(_split_df_to_xlsx_bytes)(_rdf) for _arc, _rdf in _jobs)
                                        except Exception:  # noqa: BLE001
                                            _written = None
                                    if _written is None:
                                        _written = [_split_df_to_xlsx_bytes(_rdf) for _arc, _rdf in _jobs]

                                    _buf = _io.BytesIO()
                                    with _zip.ZipFile(_buf, "w", _zip.ZIP_DEFLATED) as _z:
                                        for (_arc, _rdf), _bytes in zip(_jobs, _written):
                                            _z.writestr(_arc, _bytes)
                                        # DRIFT GUARD: stamp the split signature so tab 5 can detect when the
                                        # rule files it's running no longer match tab 3's split.
                                        try:
                                            import json as _json
                                            _manifest = {
                                                "exp_sig": list(_exp_sig),
                                                "dial": float(picked_w), "max_pools": int(_maxN),
                                                "engine": ss.get("variations_engine"), "brand": str(_brand_e),
                                                "go_live": str(_gl_e),
                                                "max_share": round(float(_wc_e.get("max_share", 0.97)), 4),
                                                "has_compressed": _pt_long is not None,
                                                "built_at": str(pd.Timestamp.now()),
                                            }
                                            _z.writestr("_export_manifest.json", _json.dumps(_manifest, indent=2))
                                        except Exception:  # noqa: BLE001
                                            pass
                                    ss["_split_export_zip"] = _buf.getvalue()
                                    ss["_split_export_n"] = _n
                                    ss["_split_export_has_comp"] = _pt_long is not None
                                    ss["_split_export_sig"] = _exp_sig
                                    _ok = True
                            except Exception as _e:
                                st.error(f"Export failed: {_e}")
                            if _ok:
                                st.rerun()   # refresh cards + impact basis with the fresh pool result
                        if ss.get("_split_export_zip"):
                            # Stale = the current settings no longer match what this zip was built from.
                            _stale = ss.get("_split_export_sig") != _exp_sig
                            if _stale:
                                st.download_button("⚠ Rebuild — settings changed",
                                                   ss["_split_export_zip"], file_name="split_templates.zip",
                                                   mime="application/zip", key="export_splits_dl",
                                                   disabled=True, use_container_width=True)
                                st.caption("The dial/engine/pool target changed since this was built. "
                                           "Click **Export Templates** to refresh the download.")
                            else:
                                st.download_button("⬇ Download",
                                                   ss["_split_export_zip"], file_name="split_templates.zip",
                                                   mime="application/zip", key="export_splits_dl",
                                                   use_container_width=True)

                    if hasattr(st, "fragment"):
                        st.fragment(_export_ui)()
                    else:
                        _export_ui()
            _retired = ss.get("retired_mids", [])
            if _retired:
                st.caption(f"⚠️ Retired to meet the VAMP cap ({len(_retired)}): "
                           + ", ".join(_retired[:25]) + ("…" if len(_retired) > 25 else ""))
            _mrc = ss.get("mid_rpgt_constrained", [])
            if _mrc:
                st.caption(f"🎯 Scaled/retired to meet per-(MID × RPGT) caps ({len(_mrc)}): "
                           + ", ".join(_mrc[:25]) + ("…" if len(_mrc) > 25 else ""))



        # RPGT scope for the impact projections: when the tab-2 tickbox holds unselected
        # RPGTs at baseline, pass the selected set so the projection forces post == pre for
        # every other RPGT. Empty tuple = apply the split to all RPGTs (tickbox OFF or all
        # RPGTs selected).
        _rscope = ss.get("rpgt_scope") or {}
        _scoped_rpgts = ()
        if _rscope.get("hold_others") and _rscope.get("selected") \
                and set(_rscope["selected"]) != set(_rscope.get("all", ())):
            _scoped_rpgts = tuple(_rscope["selected"])

        # [FN-354]
        def _fin_render_share_chart(_evframe, _target=None):
            """Before → after volume-share chart at vampMid grain, rendered inside Financial Impact.
            Three series per vampMid: current baseline (grey), proposed split (red) and a
            max-approval-rate hypothetical (green). The hypothetical routes each cell 100% to its
            highest-success ELIGIBLE gateway — it respects eligibility (wallet/USA/bans, i.e. the
            candidate set already present in the split) but ignores the VAMP and max-share caps."""
            _tc = _target or st
            if not HAS_PLOTLY:
                return
            try:
                import plotly.graph_objects as _go
                _sc = _evframe.copy()
                if _sc is None or getattr(_sc, "empty", True):
                    return
                # Profile key = (rpgt, currency, bank); cell_volume is the profile total on every row.
                _sc["_cellk"] = (_sc["rpgt_join"].astype(str) + "|"
                                 + _sc["currency_join"].astype(str) + "|"
                                 + _sc["bin_join"].astype(str))
                _sc["cell_volume"] = pd.to_numeric(_sc.get("cell_volume", 0), errors="coerce").fillna(0.0)
                _sc["gw_sr"] = pd.to_numeric(_sc.get("gw_sr", 0), errors="coerce").fillna(0.0)
                # Max-approval hypothetical: the single highest-gw_sr eligible gateway per profile takes
                # the whole profile's volume (ignores VAMP / share caps; eligibility already implied by
                # the candidate set present in the frame).
                _sc["_maxwin"] = 0.0
                _valid = _sc[_sc["cell_volume"] > 0]
                if not _valid.empty:
                    _win = _valid.groupby("_cellk")["gw_sr"].idxmax()
                    _sc.loc[_win, "_maxwin"] = _sc.loc[_win, "cell_volume"]
                _g = _sc.groupby("_vmid", as_index=False).agg(
                    cur=("pre_vol", "sum"), prop=("post_vol", "sum"), maxa=("_maxwin", "sum"))
                _tot = float(_g[["cur", "prop", "maxa"]].to_numpy().sum() / 3.0) or 1.0
                for _c in ("cur", "prop", "maxa"):
                    _g[_c] = 100.0 * _g[_c] / _tot
                _g = _g[(_g[["cur", "prop", "maxa"]].abs().sum(axis=1)) > 1e-9]
                if _g.empty:
                    return
                _g = _g.sort_values("prop", ascending=True)          # largest at top of h-chart
                _lbl = _g["_vmid"].astype(str).str.replace("_", " ").str.strip().str[:40].tolist()
                _tc.markdown(f"###### Before → after volume share (by {_mc_hdr('vampMid')})")
                _f1 = _go.Figure()
                # connector line spanning the three points per vampMid
                _xs, _ys = [], []
                for _i, _r in _g.iterrows():
                    _lo = min(_r["cur"], _r["prop"], _r["maxa"]); _hi = max(_r["cur"], _r["prop"], _r["maxa"])
                    _yl = str(_r["_vmid"]).replace("_", " ").strip()[:40]
                    _xs += [_lo, _hi, None]; _ys += [_yl, _yl, None]
                _f1.add_scatter(x=_xs, y=_ys, mode="lines", line=dict(color="#C7D0DE", width=2),
                                hoverinfo="skip", showlegend=False)
                _f1.add_scatter(x=_g["maxa"], y=_lbl, mode="markers",
                                marker=dict(color="#e63748", size=10), name="max-approval (ceiling)",
                                hovertemplate="%{y}<br>max-approval %{x:.1f}%<extra></extra>")
                _f1.add_scatter(x=_g["cur"], y=_lbl, mode="markers",
                                marker=dict(color="#8A93A6", size=10), name="current baseline",
                                hovertemplate="%{y}<br>current %{x:.1f}%<extra></extra>")
                _f1.add_scatter(x=_g["prop"], y=_lbl, mode="markers",
                                marker=dict(color="#22C36B", size=10), name="proposed split",
                                hovertemplate="%{y}<br>proposed %{x:.1f}%<extra></extra>")
                _f1.update_layout(height=max(260, 30 * len(_g) + 70),
                                  margin=dict(l=10, r=10, t=10, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#0B1F3A", size=9),
                                  xaxis=dict(title="", showgrid=False, ticksuffix="%",
                                             tickfont=dict(color="#0B1F3A", size=9)),
                                  yaxis=dict(showgrid=False, tickfont=dict(color="#0B1F3A", size=9)),
                                  legend=dict(orientation="h", y=1.08, font=dict(color="#0B1F3A", size=9)))
                _tc.plotly_chart(_f1, use_container_width=True)
            except Exception as _e:  # noqa: BLE001
                _tc.caption(f"Share chart unavailable: {type(_e).__name__}: {_e}")

        (_t_risk, _t_fin, _t_middet, _t_gwdet, _t_bank,
         _t_engwork) = st.tabs(
            ["Risk Impact", "Financial Impact", "Mid Detail",
             "Gateway Detail", "Bank Detail", "Engine Workings"])

        with _t_middet:
            _md_bank_slot = st.container()    # per-vampMid bank table + revenue bridge
            _md_charts_slot = st.container()  # vampMid-level SR + gateway-share charts

        with _t_fin:
            # Revenue-by-vampMid × month table renders into this slot, positioned
            # ABOVE the Bank x Currency Impact section (its code lives further down).
            _rev_slot = st.container(border=True)

            # Bridge charts (vampMid / RPGT revenue + success) are reserved HERE so they render
            # ABOVE the bank-blocked table and the before→after volume-share chart. The bridge
            # column-slots are created into this container further down (inside the `if date_col`
            # block) and filled from the pre/post section, keeping the fill logic unchanged.
            _finbridge_slot = st.container()

            # -------- Bank-blocked gateways + before/after share chart (same row) --------
            # Bank-blocked table narrowed 35% (0.5 → 0.325 of the row); share chart takes the rest.
            _bbc_col, _bsc_col = st.columns([0.65, 1.35])
            # Right column: before → after volume-share chart (filled once _evv is computed below).
            _finshare_slot = _bsc_col.container(border=True)
            _blk_df = ss.get("blocked_gateways")
            if (_blk_df is not None and not getattr(_blk_df, "empty", True)
                    and "blocked" in getattr(_blk_df, "columns", []) and bool(_blk_df["blocked"].any())):
                with _bbc_col.container(border=True):
                    st.markdown("##### Bank-blocked gateways")
                    _bf = _blk_df[_blk_df["blocked"]].copy()
                    # The 'bin' column holds the BIN; add the parent Bank Name beside it.
                    _b2b_blk = ss.get("bin_to_bank", {})
                    _bf["bank_name"] = _map_to_bank(_bf["bin"], _b2b_blk, default="").astype(str)
                    _cols_bf = [c for c in ["bin", "bank_name", "gateway", "consec_failed", "last_success_date"]
                                if c in _bf.columns]
                    if "consec_failed" in _bf.columns:
                        _bf = _bf.sort_values("consec_failed", ascending=False)
                    # Styled HTML table (red sticky header / card bg / ink text) so the format matches
                    # the app's other data tables.
                    _hdr_bf = {"bin": "BIN", "bank_name": "Bank Name", "gateway": "Gateway",
                               "consec_failed": "Failed Attempts",
                               "last_success_date": "Last success"}
                    _bkh = ['<div style="display:inline-block; max-width:100%; '
                            'box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; '
                            'overflow:auto; max-height:360px; background-color:var(--tav-card); '
                            'border:1px solid var(--tav-line);">'
                            '<table style="width:auto; border-collapse:collapse; font-family:inherit; '
                            'font-size:0.72rem; line-height:1.2; table-layout:auto;"><tr>']
                    for _c in _cols_bf:
                        _al = "left" if _c in ("bin", "bank_name", "gateway") else "right"
                        # Header wraps (white-space:normal) so a long label like "Consecutive failed
                        # attempts" no longer forces the column wider than its (short) values.
                        _bkh.append(f'<th style="background-color:var(--tav-red); color:#FFF; '
                                    f'font-weight:bold; padding:3px 6px; text-align:{_al}; '
                                    f'position:sticky; top:0; white-space:normal; '
                                    f'word-break:break-word; width:1%;">{_hdr_bf.get(_c, _c)}</th>')
                    _bkh.append('</tr>')
                    for _, _rbk in _bf.iterrows():
                        _bkh.append('<tr>')
                        for _c in _cols_bf:
                            _al = "left" if _c in ("bin", "bank_name", "gateway") else "right"
                            _cv = _rbk[_c]
                            if _c == "consec_failed":
                                _cn = pd.to_numeric(_cv, errors="coerce")
                                _txt = f"{_cn:,.0f}" if pd.notna(_cn) else "—"
                            elif _c == "last_success_date":
                                _dt = pd.to_datetime(_cv, errors="coerce")
                                _txt = _dt.strftime("%Y-%m-%d") if pd.notna(_dt) else "—"
                            else:
                                _txt = "—" if (_cv is None or (isinstance(_cv, float) and pd.isna(_cv))) else str(_cv)
                            _bkh.append(f'<td style="padding:2px 6px; text-align:{_al}; color:var(--tav-ink); '
                                        f'border-bottom:1px solid var(--tav-line); white-space:nowrap; '
                                        f'width:1%;">{_txt}</td>')
                        _bkh.append('</tr>')
                    _bkh.append('</table></div>')
                    st.markdown("".join(_bkh), unsafe_allow_html=True)

            # (RPGT routing-sensitivity priority chart removed per request.)

            # -------------- Bank Impact Table Layout --------
            # Renders into the "Bank Detail" sub-tab (its RPGT-table slot is filled later).
            with _t_bank.container(border=True):
                st.markdown("##### Bank x Currency Impact")

                # Shared revenue-bridge waterfall builder (used by the top-of-tab bridge AND the
                # per-vampMid / success bridges below), so every bridge has the SAME format. Defined
                # at this (outer) scope — the fragment below and the later pre/post section both call
                # it. X-axis min = running trough − pad; max = running peak + pad. Y labels show the
                # FULL name (automargin + a wide left margin), never truncated.
                # [FN-355]
                def _rev_bridge_waterfall(pre, post, names, deltas, money=True, pct=False, wide_min=False,
                                          height=560, tick_size=9, left_margin=None):
                    if not HAS_PLOTLY:
                        return None
                    import plotly.graph_objects as _gwf
                    pre, post = float(pre), float(post)
                    _dl = [float(d) for d in deltas]
                    _xs = ["Current"] + [str(n) for n in names] + ["Proposed"]
                    # Auto-size the left margin to the LONGEST y-label so short-name bridges (RPGT)
                    # don't get dead space and long-name ones (Bank×Currency) aren't clipped.
                    if left_margin is None:
                        _maxlab = max((len(str(s)) for s in _xs), default=6)
                        left_margin = int(min(240, max(30, _maxlab * tick_size * 0.62 + 14)))
                    _ys = [pre] + _dl + [0.0]
                    _labs = [pre] + _dl + [post]
                    _meas = ["absolute"] + ["relative"] * len(_dl) + ["total"]
                    # Tight x-range = the ACTUAL running-total envelope the waterfall traverses
                    # (Current → each cumulative step → Proposed), so there's no dead whitespace on
                    # the right from summing every increase. The pad scales to the VISIBLE movement
                    # (peak−trough), not the absolute magnitude, so a large-$ bridge with small moves
                    # doesn't get a huge margin; the right side gets extra room for outside labels.
                    _cum = pre
                    _peak = max(pre, post); _trough = min(pre, post)
                    for _d in _dl:
                        _cum += _d
                        _peak = max(_peak, _cum); _trough = min(_trough, _cum)
                    _span = max(abs(_peak - _trough), 1.0)
                    # wide_min: force the floor further left when a caller wants every bar fully shown.
                    _lo = ((min(pre, post) - sum(abs(d) for d in _dl) - 0.06 * _span) if wide_min
                           else (_trough - 0.06 * _span))
                    _hi = _peak + 0.16 * _span
                    if _hi <= _lo:
                        _hi = _lo + 1.0

                    # [FN-356]
                    def _fmt(_v):
                        if pct:
                            return f"{_v:,.2f}%"
                        if money:
                            return (f"${_v/1e6:,.2f}M" if abs(_v) >= 1e6 else f"${_v/1e3:,.1f}k")
                        return (f"{_v/1e6:,.2f}M" if abs(_v) >= 1e6 else f"{_v:,.0f}")
                    _text = [_fmt(_v) for _v in _labs]
                    # Right edge (value axis) of each bar, so EVERY value label sits to the RIGHT of
                    # its bar via annotations (rather than plotly's directional "outside", which puts
                    # decreasing-bar labels on the left).
                    _rights = [max(0.0, pre)]
                    _cum2 = pre
                    for _d in _dl:
                        _nxt = _cum2 + _d
                        _rights.append(max(_cum2, _nxt))
                        _cum2 = _nxt
                    _rights.append(max(0.0, post))
                    _r_margin = int(max(40, (max((len(t) for t in _text), default=6)) * tick_size * 0.62 + 12))
                    # Tooltip shows the FULL amount as $###,### (Current = pre, Proposed = post, each
                    # intermediate = its delta), instead of the abbreviated $x.xxM outside labels.
                    _hovfmt = ("%{y}<br>%{customdata:,.2f}%<extra></extra>" if pct
                               else ("%{y}<br>$%{customdata:,.0f}<extra></extra>" if money
                                     else "%{y}<br>%{customdata:,.0f}<extra></extra>"))
                    _wf = _gwf.Figure(_gwf.Waterfall(
                        orientation="h", measure=_meas, y=_xs, x=_ys,
                        text=_text, textposition="none",
                        cliponaxis=False,
                        customdata=_labs, hovertemplate=_hovfmt,
                        connector=dict(line=dict(color="#B9C6DA")),
                        increasing=dict(marker=dict(color="#22C36B")),
                        decreasing=dict(marker=dict(color="#e63748")),
                        totals=dict(marker=dict(color="#0B1F3A")), showlegend=False))
                    # Value labels pinned to the RIGHT of each bar.
                    for _yc, _xr, _tt in zip(_xs, _rights, _text):
                        _wf.add_annotation(x=_xr, y=_yc, text=_tt, showarrow=False,
                                           xanchor="left", xshift=3, yanchor="middle",
                                           font=dict(color="#0B1F3A", size=tick_size))
                    _wf.update_layout(
                        # automargin (below) grows the left margin to fit the full names; left_margin
                        # is the floor so short-name bridges still line up; r_margin fits the right labels.
                        height=height, margin=dict(l=left_margin, r=_r_margin, t=14, b=10),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#0B1F3A', family="inherit"))
                    _wf.update_xaxes(range=[_lo, _hi], showgrid=True, gridcolor='lightgrey',
                                     tickprefix=("$" if (money and not pct) else ""),
                                     ticksuffix=("%" if pct else ""),
                                     tickfont=dict(color='#0B1F3A', size=tick_size), title=None)
                    _wf.update_yaxes(type="category", autorange="reversed", showgrid=False,
                                     tickfont=dict(color='#0B1F3A', size=tick_size), title=None, automargin=True,
                                     tickmode="array", tickvals=_xs, ticktext=[str(s) for s in _xs])  # full names
                    return _wf

                # [FN-357]
                def _bridge_items(_df, _name_col, _pre_col, _post_col, _sort_mode, _other="Other", _max_n=7):
                    """Pick the movers for a bridge given a sort mode; roll the rest into one bar.
                    Returns (pre_total, post_total, names, deltas) or None."""
                    _d = _df[[_name_col, _pre_col, _post_col]].copy()
                    _d.columns = ["_n", "_p", "_q"]
                    _d["_p"] = pd.to_numeric(_d["_p"], errors="coerce").fillna(0.0)
                    _d["_q"] = pd.to_numeric(_d["_q"], errors="coerce").fillna(0.0)
                    _d["_dd"] = _d["_q"] - _d["_p"]
                    _d = _d[(_d["_p"].abs() + _d["_q"].abs()) > 0]
                    if _d.empty:
                        return None
                    if _sort_mode == "Top decreases":
                        _ranked = _d[_d["_dd"] < 0].sort_values("_dd", ascending=True)
                    elif _sort_mode == "Top absolute movers":
                        _ranked = _d.reindex(_d["_dd"].abs().sort_values(ascending=False).index)
                    else:  # "Top increases"
                        _ranked = _d[_d["_dd"] > 0].sort_values("_dd", ascending=False)
                    _sel = _ranked if _max_n is None else _ranked.head(_max_n)   # _max_n=None → show all, no 'Other'
                    _sel = _sel.sort_values("_dd", ascending=False)
                    _pre_t = float(_d["_p"].sum())
                    _post_t = float(_d["_q"].sum())
                    _has_other = (_max_n is not None) and (len(_d) > len(_sel))
                    _o = (_post_t - _pre_t) - float(_sel["_dd"].sum())
                    _names = _sel["_n"].astype(str).tolist() + ([_other] if _has_other else [])
                    _deltas = _sel["_dd"].tolist() + ([_o] if _has_other else [])
                    return _pre_t, _post_t, _names, _deltas

                # The filter row + bank table + revenue bridge live in a st.fragment, so changing the
                # RPGT / Sort / Order filters reruns ONLY this section — no full-page whiteout, other
                # sub-tabs stay put. The RPGT revenue table (_rpgt_tab_slot) is created OUTSIDE the
                # fragment and filled later from the pre/post section, so a filter change never blanks
                # it. Falls back to a plain call on Streamlit builds without st.fragment.
                # [FN-358]
                def _bank_detail_fragment():
                    # Filters — RPGT + sort column + order, all on one row.
                    _rpgt_opts_bk = (["(All)"] + sorted(eval_df["rpgt"].astype(str).dropna().unique().tolist())
                                     if "rpgt" in eval_df.columns else ["(All)"])
                    _sort_opts = ["30D $ Impact", "Attempts", "Baseline Success", "Expected Success",
                                  "Old Success Rate", "New Success Rate", "Bank"]
                    _rpf1, _fcol1, _fcol2, _rpf_sp = st.columns([1, 1, 1, 5])
                    _rpgt_sel_bk = _rpf1.selectbox("RPGT", _rpgt_opts_bk, index=0, key="bank_rpgt_filter")
                    _sort_by = _fcol1.selectbox("Sort by", _sort_opts, index=0, key="bank_sort_by")
                    _sort_dir = _fcol2.selectbox("Order", ["Descending", "Ascending"], index=0, key="bank_sort_dir")
                    _eval_bk, _cellagg_bk = eval_df, cell_agg
                    if _rpgt_sel_bk != "(All)":
                        _kbk = str(_rpgt_sel_bk).strip().lower()
                        if "rpgt_join" in eval_df.columns:
                            _eval_bk = eval_df[eval_df["rpgt_join"].astype(str).str.strip().str.lower() == _kbk]
                        if "rpgt_join" in cell_agg.columns:
                            _cellagg_bk = cell_agg[cell_agg["rpgt_join"].astype(str).str.strip().str.lower() == _kbk]
                    cell_impact = _eval_bk.groupby(["rpgt_join", "currency_join", "bin_join"]).agg(exp_succ=("exp_succ", "sum"), exp_rev=("exp_rev", "sum"), pre_rev=("pre_rev", "sum")).reset_index()
                    cell_full = _cellagg_bk.merge(cell_impact, on=["rpgt_join", "currency_join", "bin_join"], how="left").fillna(0)
                    bank_display_map = eval_df[["bin_join", "bin"]].drop_duplicates().set_index("bin_join")["bin"].to_dict()

                    bank_table = cell_full.groupby(["bin_join", "currency_join"]).agg(old_att=("cell_att", "sum"), old_succ=("cell_succ", "sum"), old_rev=("cell_rev", "sum"), old_rev_pre=("pre_rev", "sum"), new_succ=("exp_succ", "sum"), new_rev=("exp_rev", "sum"), avg_ticket=("avg_ticket", "first")).reset_index()
                    # Baseline revenue on the MODELLED pre_rev basis (same basis as the Expected
                    # Revenue card and every other bridge, so all revenue bridges reconcile).
                    bank_table["old_rev"] = bank_table["old_rev_pre"]
                    bank_table["Bank"] = (bank_table["bin_join"].map(bank_display_map).fillna(bank_table["bin_join"]).astype(str)
                                          + " - " + bank_table["currency_join"].astype(str).str.upper())
                    bank_table["Attempts"] = bank_table["old_att"]
                    bank_table["Baseline Success"] = bank_table["old_succ"]
                    bank_table["Expected Success"] = bank_table["new_succ"]
                    bank_table["Old Success Rate"] = np.where(bank_table["old_att"] > 0, (bank_table["old_succ"] / bank_table["old_att"]) * 100, 0)
                    bank_table["New Success Rate"] = np.where(bank_table["old_att"] > 0, (bank_table["new_succ"] / bank_table["old_att"]) * 100, 0)
                    bank_table["30D $ Impact"] = bank_table["new_rev"] - bank_table["old_rev"]

                    total_old_att = bank_table["old_att"].sum()
                    total_row = {
                        "Bank": "TOTAL", "Attempts": total_old_att, "Baseline Success": bank_table["old_succ"].sum(), "Expected Success": bank_table["new_succ"].sum(),
                        "Old Success Rate": (bank_table["old_succ"].sum() / total_old_att * 100) if total_old_att > 0 else 0,
                        "New Success Rate": (bank_table["new_succ"].sum() / total_old_att * 100) if total_old_att > 0 else 0,
                        "30D $ Impact": bank_table["new_rev"].sum() - bank_table["old_rev"].sum()
                    }

                    _rows = 20
                    bank_view = (bank_table.sort_values(_sort_by, ascending=(_sort_dir == "Ascending"))
                                 .head(int(_rows)).reset_index(drop=True))

                    _cols = ["Bank", "Attempts", "Baseline Success", "Expected Success",
                             "Old Success Rate", "New Success Rate", "30D $ Impact"]

                    # [FN-359]
                    def _fmt_cell(col, v):
                        if col == "Bank":
                            _s = str(v)
                            return (_s[:30] + "…") if len(_s) > 30 else _s
                        if col in ("Old Success Rate", "New Success Rate"):
                            return f"{float(v):.2f}%"
                        if col == "30D $ Impact":
                            return f"${float(v):+,.0f}"
                        return f"{float(v):,.0f}"

                    # [FN-360]
                    def _bcw(_c):
                        return ("padding:4px 8px; font-size:0.74rem;" if _c == "Bank"
                                else "padding:1px 2px; font-size:0.35rem;")
                    _h = ['<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; '
                          'overflow:auto; margin-bottom:1rem; '   # gap before the RPGT table beneath it
                          'background-color:var(--tav-card); border:1px solid var(--tav-line);">']
                    _h.append('<table style="width:100%; border-collapse:collapse; font-family:inherit; line-height:1.15;"><tr>')
                    for _c in _cols:
                        _al = "left" if _c == "Bank" else "right"
                        _ws = "nowrap" if _c == "Bank" else "normal"
                        _h.append(f'<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; '
                                  f'{_bcw(_c)} text-align:{_al}; white-space:{_ws};">{_c}</th>')
                    _h.append('</tr>')

                    # [FN-361]
                    def _bank_row_html(r, is_total=False):
                        _tb = "border-top:2px solid var(--tav-line);" if is_total else ""
                        _cells = []
                        for _c in _cols:
                            _al = "left" if _c == "Bank" else "right"
                            _fw = "800" if is_total else ("600" if _c == "Bank" else "normal")
                            _clr = "var(--tav-ink)"
                            if _c == "30D $ Impact" and not is_total:
                                _clr = "#22C36B" if float(r[_c]) >= 0 else "#e63748"
                            _cells.append(f'<td style="{_bcw(_c)} text-align:{_al}; color:{_clr}; '
                                          f'font-weight:{_fw}; {_tb} white-space:nowrap;">{_fmt_cell(_c, r[_c])}</td>')
                        return "<tr>" + "".join(_cells) + "</tr>"

                    for _, _r in bank_view.iterrows():
                        _h.append(_bank_row_html(_r))
                    _h.append(_bank_row_html(total_row, is_total=True))
                    _h.append("</table></div>")

                    # Revenue bridge across the top 10 most-impacted Bank×Currency cells.
                    _bank_wf = None
                    if HAS_PLOTLY and not bank_table.empty:
                        _bt = bank_table.copy()
                        _bt["delta"] = _bt["new_rev"] - _bt["old_rev"]
                        _bt = _bt[(_bt["old_rev"].abs() + _bt["new_rev"].abs()) > 0]
                        # Top 7 increases + top 7 decreases; everything else rolls into 'Other banks'.
                        _inc7 = _bt[_bt["delta"] > 0].sort_values("delta", ascending=False).head(7)
                        _dec7 = _bt[_bt["delta"] < 0].sort_values("delta", ascending=True).head(7)
                        _bt = pd.concat([_inc7, _dec7]).sort_values("delta", ascending=False)
                        if not _bt.empty:
                            # Bridge the FULL portfolio: total current -> top-10 deltas ->
                            # aggregate of all other banks -> total proposed, so it reconciles.
                            _pre = float(bank_table["old_rev"].sum())
                            _post = float(bank_table["new_rev"].sum())
                            _has_other = len(bank_table) > len(_bt)
                            _other = (_post - _pre) - float(_bt["delta"].sum())
                            _names = _bt["Bank"].tolist() + (["Other banks"] if _has_other else [])
                            _deltas = _bt["delta"].tolist() + ([_other] if _has_other else [])
                            # Same defaults as the Bank Analysis bridge (9px text, auto left margin),
                            # so the two bridges are visually consistent.
                            _bank_wf = _rev_bridge_waterfall(_pre, _post, _names, _deltas)

                    # Bank table (left third) + revenue bridge (right two-thirds) on one row.
                    _left_col, _right_col = st.columns([1, 2])
                    _left_col.markdown("".join(_h), unsafe_allow_html=True)
                    if _bank_wf is not None:
                        _right_col.plotly_chart(_bank_wf, use_container_width=True)

                (st.fragment(_bank_detail_fragment) if hasattr(st, "fragment") else _bank_detail_fragment)()

                # RPGT revenue table + SR-by-RPGT chart slot — created OUTSIDE the fragment so a
                # filter-only rerun never blanks it (it's populated later from the pre/post section).
                # Kept ~1/3 width + left-aligned, so it sits just under the bank table above.
                _rpgt_tab_slot = st.columns([1, 2])[0].container()


            # -------------- 30D revenue by vampMid × month (pre vs post) --------------
            # Same layout as the Risk-tab VAMP table, but VI Txn + $Revenue. Revenue =
            # RPGT-level avg ticket (from the actuals month before Month 0) × VI Txn for
            # that RPGT in that vampMid, summed to the vampMid. Renders into the slot
            # reserved above the Bank x Currency Impact section.
            with _rev_slot:
                _pp_r = _scheme_norm_export(out_dir, "vamp_t_period_prorata_export.csv")
                if not os.path.exists(_pp_r):
                    st.info("No pro-rata export found — revenue-by-vampMid table unavailable.")
                elif split is None or getattr(split, "empty", True):
                    st.info("No proposed split yet.")
                else:
                    _mm_r = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    _f2v_r = {}
                    if os.path.exists(_mm_r):
                        _mmd_r = load_mid_list(_mm_r)
                        _cc_r = _norm_cols(_mmd_r)
                        if _cc_r.get("gatewayfid") and _cc_r.get("vampmid"):
                            _f2v_r = _fid2vamp_from(_mmd_r, _cc_r["gatewayfid"], _cc_r["vampmid"])
                    _spr = split.copy()
                    _spr["_vm"] = _spr["gateway"].astype(str).str.strip().str.lower().map(_f2v_r)
                    _spr = _spr.dropna(subset=["_vm"])
                    if bool(ss.get("opt_by_rpgt", False)) and "rpgt" in _spr.columns:
                        _spr = _spr.drop_duplicates(["currency", "bin", "rpgt", "gateway"])
                        _pdf = _spr.groupby(["currency", "bin", "rpgt", "_vm"], as_index=False)["share"].sum()
                        _prop_r = tuple((str(c).lower(), str(b), str(rp), str(v), float(s))
                                        for c, b, rp, v, s in
                                        _pdf[["currency", "bin", "rpgt", "_vm", "share"]].itertuples(index=False))
                    else:
                        _spr = _spr.drop_duplicates(["currency", "bin", "gateway"])
                        _pdf = _spr.groupby(["currency", "bin", "_vm"], as_index=False)["share"].sum()
                        _prop_r = tuple((str(c).lower(), str(b), str(v), float(s))
                                        for c, b, v, s in _pdf[["currency", "bin", "_vm", "share"]].itertuples(index=False))
                    # Use ENFORCED shares (post cap / wallet / USA-Non-USA / back-fill) so revenue
                    # reflects the pipeline's actual routing — same source as the risk pre/post table
                    # (shared per-variation cache).
                    try:
                        _wc_rr = ss.get("wallet_ctx") or {}
                        _ep_key_r = (round(float(picked_w), 4), bool(_basis_compressed),
                                     str(ss.get("split_go_live_date", "")))
                        _ep_cache_r = ss.get("_enf_prop_cache") or {}
                        if _ep_cache_r.get("key") == _ep_key_r and _ep_cache_r.get("val"):
                            _prop_r = _ep_cache_r["val"]
                        else:
                            _ep_r = enforced_prop_items(
                                split, str((ss.get("forecast_settings", {}) or {}).get("company", "TotalAV")),
                                str(ss.get("split_go_live_date", "")),
                                wallet_incapable=set(_wc_rr.get("incapable", set())),
                                fid2vamp=_wc_rr.get("fid2vamp"), mid_list_path=_mm_r,
                                usa_only=set(_wc_rr.get("usa_only", set())),
                                country_pres=_wc_rr.get("country_pres", {}),
                                max_share=float(_wc_rr.get("max_share", 0.97)))
                            if _ep_r:
                                _prop_r = _ep_r
                                ss["_enf_prop_cache"] = {"key": _ep_key_r, "val": _ep_r}
                    except Exception as _e:  # noqa: BLE001
                        # keep the raw _prop_r on any failure, but surface why the enforced basis fell back
                        st.caption(f"Enforced-share revenue basis unavailable ({type(_e).__name__}: {_e}); using raw split shares.")
                    from routing_optimiser.s2_forecast.vamp_forecast_pipeline import _canonical_gateway as _cg_r
                    _ovr_r = ss.get("gateway_volume_overrides") or {}
                    _off_r = set()
                    _fid_eff_r = {}
                    for _gwid, _cfg in (_ovr_r.items() if isinstance(_ovr_r, dict) else []):
                        if isinstance(_cfg, dict):
                            _tgt = pd.to_numeric(_cfg.get("target"), errors="coerce")
                            if _tgt == 0 and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
                                _off_r.add(str(_cg_r(_gwid)).strip().lower())
                                if _cfg.get("effective_date"):
                                    _fid_eff_r[str(_cg_r(_gwid)).strip().lower()] = str(_cfg.get("effective_date"))
                    _v2f_r = {}
                    for _f, _v in _f2v_r.items():
                        _v2f_r.setdefault(_v, set()).add(str(_cg_r(_f)).strip().lower())
                    _excl_r = frozenset(v for v, fids in _v2f_r.items() if fids and fids <= _off_r)
                    _kill_r = build_kill_eff(_v2f_r, _fid_eff_r)
                    try:
                        _m0_r = str(pd.to_datetime(ss.get("forecast_settings", {}).get(
                            "month_0", date.today().replace(day=1))).date())
                    except Exception:
                        _m0_r = str(date.today().replace(day=1))

                    _wc_r = ss.get("wallet_ctx") or {}
                    _floor_r = (0.0 if os.environ.get("ROUTING_PROJ_FLOOR", "0") == "0"
                                else float(ss.get("exploration_floor", 0.0) or 0.0))
                    _wcp_r, _uop_r, _ = _cap_pairs(
                        os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                        input_json_path("routing_restrictions.json"))
                    _gran_r = _c_prepost_granular(
                        _pp_r, projection_cache_sig(_pp_r, _prop_r, _floor_r,
                                                    wallet_incapable_pairs=_wcp_r,
                                                    usa_only_pairs=_uop_r),
                        _prop_r, _excl_r, _kill_r, _m0_r, _scoped_rpgts,
                        frozenset(str(x).strip().lower() for x in (_wc_r.get("incapable") or set())),
                        frozenset(str(x).strip().lower() for x in (_wc_r.get("usa_only") or set())),
                        exploration_floor=_floor_r,
                        wallet_incapable_pairs=_wcp_r, usa_only_pairs=_uop_r,
                        # 19df — the same max-share cap the search applies, so this secondary
                        # table cannot disagree with the primary one at :4205 about what the
                        # delivered VAMP is. `_wc_rr` hands the identical value to
                        # enforced_prop_items at :1308.
                        max_share=float(_wc_r.get("max_share", 0.97)))
                    _tick_r = rpgt_avg_ticket(cache.get("cell_agg"))                 # RPGT fallback
                    _rc_tick_r = rpgt_currency_avg_ticket(cache.get("cell_agg"))     # RPGT × Currency
                    # VALIDATE MODE ONLY: source transactions from the pipeline's granular
                    # bin_rpgt_impact_export.csv so they tie EXACTLY to the Validate Split table, while
                    # KEEPING the RPGT×Currency avg-ticket → revenue = pipeline transactions ×
                    # ATV(RPGT×Currency). Only swapped when the granular carries Currency (the join key);
                    # otherwise the projection stands. Engine (tab 2) flow is untouched.
                    if ss.get("variations_engine") == "validate":
                        _gp_fin = _validate_granular_from_bin_rpgt(out_dir)
                        if _gp_fin is not None and not _gp_fin.empty and "Currency" in _gp_fin.columns:
                            _gran_r = _gp_fin
                    _rev_tbl = mid_revenue_month_table(_gran_r, _tick_r, months=range(6),
                                                       rc_ticket=_rc_tick_r)
                    _rev_tbl = _rev_tbl.sort_values("$Revenue M0", ascending=False)
                    _numc = [c for c in _rev_tbl.columns if c != "vampMid"]
                    _tot_r = {"vampMid": "TOTAL", **{c: float(_rev_tbl[c].sum()) for c in _numc}}
                    _rvv = pd.concat([_rev_tbl, pd.DataFrame([_tot_r])], ignore_index=True)
                    # Table shows M0–M2 only; the freed space holds the M1 revenue-bridge waterfall.
                    _grp6 = [[f"VI Txn M{m}", f"$Revenue M{m}", f"VI Txn Post M{m}", f"$Revenue Post M{m}"]
                             for m in range(3)]
                    # Colour scales for post-vs-pre change (same green↑/red↓ as other tables).
                    # VI Txn and $Revenue are coloured independently (different units/magnitudes).
                    _rev_maxabs = 0.0
                    _vi_maxabs = 0.0
                    for _m in range(3):
                        _d = (_rev_tbl[f"$Revenue Post M{_m}"] - _rev_tbl[f"$Revenue M{_m}"]).abs()
                        _rev_maxabs = max(_rev_maxabs, float(_d.max()) if not _d.empty else 0.0)
                        _dv = (_rev_tbl[f"VI Txn Post M{_m}"] - _rev_tbl[f"VI Txn M{_m}"]).abs()
                        _vi_maxabs = max(_vi_maxabs, float(_dv.max()) if not _dv.empty else 0.0)
                    _rev_maxabs = _rev_maxabs if _rev_maxabs > 1e-9 else 1.0
                    _vi_maxabs = _vi_maxabs if _vi_maxabs > 1e-9 else 1.0
                    _sp = '<th style="background-color:var(--tav-card); border:none; width:8px; min-width:8px; padding:0;"></th>'
                    _rh = ['<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow-x:auto; '
                           'width:100%; background-color:var(--tav-card); border:1px solid var(--tav-line);">']
                    _rh.append('<table style="width:100%; border-collapse:collapse; font-family:inherit; '
                               'font-size:0.68rem; line-height:1.1;"><tr>')
                    _rh.append('<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; text-align:left; '
                               f'position:sticky; left:0; width:1%; white-space:nowrap;">{_mc_hdr("vampMid")}</th>')
                    _rh.append(_sp)
                    for _grp in _grp6:
                        for _c in _grp:
                            _rh.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; '
                                       f'text-align:right; white-space:nowrap; width:1%;">{_mc_hdr(_c.replace("$Revenue", "$Amt"))}</th>')
                        _rh.append(_sp)
                    _rh.append('</tr>')
                    for _, _r in _rvv.iterrows():
                        _is_tot = (_r["vampMid"] == "TOTAL")
                        _tb = "border-top:2px solid var(--tav-line);" if _is_tot else ""
                        _wt = "800" if _is_tot else "normal"
                        _rh.append('<tr>')
                        _rh.append(f'<td style="padding:2px 8px; text-align:left; color:#000; '
                                   f'font-weight:{"800" if _is_tot else "600"}; {_tb} position:sticky; left:0; '
                                   f'background-color:var(--tav-card); width:1%; white-space:nowrap;">{_r["vampMid"]}</td>')
                        _rh.append(f'<td style="width:8px; min-width:8px; padding:0; {_tb}"></td>')
                        for _mi, _grp in enumerate(_grp6):
                            for _c in _grp:
                                _ital = "font-style:italic;" if "Post" in _c else ""
                                if "$Revenue" in _c:
                                    _rv = float(_r[_c])
                                    _txt = (f"${_rv/1e6:,.2f}M" if abs(_rv) >= 1e6 else f"${_rv/1e3:,.1f}k")
                                else:
                                    _txt = f"{_r[_c]:,.0f}"
                                _cbg = ""
                                _pcol = (f"$Revenue Post M{_mi}", f"$Revenue M{_mi}", _rev_maxabs) if _c == f"$Revenue Post M{_mi}" \
                                    else ((f"VI Txn Post M{_mi}", f"VI Txn M{_mi}", _vi_maxabs) if _c == f"VI Txn Post M{_mi}" else None)
                                if (not _is_tot) and _pcol is not None:
                                    _dl = float(_r[_pcol[0]]) - float(_r[_pcol[1]])
                                    _fr = max(-1.0, min(1.0, _dl / _pcol[2]))
                                    _cbg = (f"background-color: rgba(34,195,107,{0.75 * _fr:.3f});" if _fr >= 0
                                            else f"background-color: rgba(230,55,72,{0.75 * abs(_fr):.3f});")
                                _rh.append(f'<td style="padding:2px 6px; text-align:right; color:#000; font-weight:{_wt}; '
                                           f'{_ital} {_cbg} {_tb} white-space:nowrap; width:1%;">{_txt}</td>')
                            _rh.append(f'<td style="width:8px; min-width:8px; padding:0; {_tb}"></td>')
                        _rh.append('</tr>')
                    _rh.append('</table></div>')
                    # Table (M0–M2) on the LEFT; the freed space on the RIGHT holds an M1 revenue-bridge
                    # waterfall by vampMid/mastercardMid that reconciles $Amt M1 (pre) → $Amt Post M1 (post).
                    _tbl_col, _wf_col = st.columns([3, 2])
                    _tbl_col.markdown("".join(_rh), unsafe_allow_html=True)
                    # M1 revenue bridge — same _rev_bridge_waterfall format as the other bridges
                    # (horizontal, Current→movers→Proposed, green↑/red↓, right-pinned $ labels). Show
                    # every MID whose ABSOLUTE M1 revenue move exceeds $10k; THEN keep pulling the next-
                    # largest mover in until the pooled 'Other' bar is smaller (in absolute $) than the
                    # smallest displayed mover — so 'Other' never hides more than any single shown MID.
                    # Current = Σ $Amt M1, Proposed = Σ $Amt Post M1 → ties to the table's M1 columns.
                    _wfd = _rev_tbl[["vampMid", "$Revenue M1", "$Revenue Post M1"]].copy()
                    _wfd["_delta"] = _wfd["$Revenue Post M1"] - _wfd["$Revenue M1"]
                    _pre1 = float(_rev_tbl["$Revenue M1"].sum())
                    _post1 = float(_rev_tbl["$Revenue Post M1"].sum())
                    _wfd = _wfd[_wfd["_delta"].abs() > 1e-6]   # drop exactly-flat MIDs
                    _wfd = _wfd.reindex(
                        _wfd["_delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
                    _n_disp = int((_wfd["_delta"].abs() > 10_000).sum())
                    while _n_disp < len(_wfd):
                        _other_abs = abs(float(_wfd["_delta"].iloc[_n_disp:].sum()))
                        # 'Other' must be at least 40% smaller than the smallest shown MID (|Other| < 0.6×smallest).
                        if _n_disp >= 1 and _other_abs < 0.6 * float(_wfd["_delta"].iloc[:_n_disp].abs().min()):
                            break
                        _n_disp += 1
                    _disp = _wfd.iloc[:_n_disp].sort_values("_delta", ascending=False)
                    _names = _disp["vampMid"].astype(str).tolist()
                    _deltas = [float(_v) for _v in _disp["_delta"].tolist()]
                    if _n_disp < len(_wfd):   # roll the remaining movers into one reconciling 'Other'
                        _names.append("Other")
                        _deltas.append((_post1 - _pre1) - sum(_deltas))
                    _wf_fig = _rev_bridge_waterfall(_pre1, _post1, _names, _deltas, money=True, height=392)
                    if _wf_fig is not None:
                        _wf_col.plotly_chart(_wf_fig, use_container_width=True)
                    else:
                        _wf_col.caption("(M1 revenue bridge unavailable)")

            _gwshare_slot = None   # 'Current vs Proposed Gateway Share' table renders here (top-left)
            _gwwf_slot = None      # 'Revenue bridge by gateway' waterfall renders here (top-right)

            # -------------- Bank Analysis (own filters) --------
            # Renders into the "Bank Detail" sub-tab (its gateway-share / bridge slots filled later).
            with _t_bank.container(border=True):
                st.markdown("##### Bank Analysis")
                # Match the styling of the other Plotly charts on this tab.
                st.markdown("""<style>
                    [data-testid="stPlotlyChart"] { background-color: var(--tav-card) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important; border: 1px solid var(--tav-line) !important; border-radius: 0 !important; padding: 12px !important; margin-bottom: 1rem; overflow: hidden !important; }
                    [data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] .js-plotly-plot, [data-testid="stPlotlyChart"] .plot-container { max-width: 100% !important; }
                </style>""", unsafe_allow_html=True)

                _adf_raw = ss.get("adf")
                if _adf_raw is None or getattr(_adf_raw, "empty", True):
                    st.caption("No attempts data loaded — compute a split in tab 3 first.")
                elif not HAS_PLOTLY:
                    st.caption("Plotly is required to render these charts.")
                else:
                    a = _adf_raw.copy()   # px is re-imported later where it's first used
                    # Collapse BINs into their parent Bank (Bank x Currency grain).
                    _b2b_d = ss.get("bin_to_bank", {})
                    if _b2b_d and "bin" in a.columns:
                        a["bin"] = _map_to_bank(a["bin"], _b2b_d).astype(str)
                    _dc2 = "date" if "date" in a.columns else ("Date" if "Date" in a.columns else None)
                    a["bank_currency"] = a["bin"].astype(str).str.strip() + " - " + a["currency"].astype(str).str.strip().str.upper()
                    bc_opts = sorted(a["bank_currency"].dropna().unique().tolist())
                    gw_opts = sorted(a["gateway"].astype(str).str.strip().dropna().unique().tolist())

                    # [FN-362]
                    def _idx(opts, val):
                        return opts.index(val) if val in opts else 0

                    # Only the Bank / Currency filter remains (Gateway + date filters removed); the
                    # chart shows the bank×currency's most-recent-30-days daily totals across gateways.
                    fc1, _fcsp = st.columns([1, 5])
                    sel_bc = fc1.selectbox("Bank / Currency", bc_opts,
                                           index=_idx(bc_opts, "JPMORGAN CHASE BANK NA - USD"),
                                           key="raw_daily_bc")

                    if not _dc2:
                        st.caption("Attempts data has no date column.")
                    else:
                        a["_d"] = pd.to_datetime(a[_dc2], errors="coerce")
                        _bv = sel_bc.split(" - ")[0].strip().lower()
                        _cv = sel_bc.split(" - ")[1].strip().lower()
                        mask = ((a["bin"].astype(str).str.strip().str.lower() == _bv)
                                & (a["currency"].astype(str).str.strip().str.lower() == _cv))
                        d = a[mask].copy()
                        # No date filters now → default to the most recent 30 days in the data.
                        if not d.empty and d["_d"].notna().any():
                            d = d[d["_d"] >= (d["_d"].max() - pd.Timedelta(days=30))]
                        if d.empty:
                            st.caption("No rows match the selected bank / currency.")
                        else:
                            daily = (d.groupby(d["_d"].dt.date)
                                     .agg(attempts=("attempts", "sum"), success=("success", "sum"))
                                     .reset_index())
                            daily.columns = ["date", "attempts", "success"]
                            # Use a real datetime x-axis so go.Bar draws ONE bar per day
                            # (date objects can fall back to a category axis with fat bars).
                            daily["date"] = pd.to_datetime(daily["date"])
                            daily = daily.sort_values("date")
                            # Success rate can never exceed 100% — clip (the initial-attempt
                            # counting can otherwise push it slightly over on odd days).
                            daily["sr"] = np.where(daily["attempts"] > 0,
                                                   daily["success"] / daily["attempts"] * 100.0, 0.0)
                            daily["sr"] = daily["sr"].clip(upper=100.0)

                            import plotly.graph_objects as go
                            _afont = dict(color='#0B1F3A', size=9, family="inherit")

                            # [FN-363]
                            def _style(fig):
                                fig.update_layout(
                                    height=320, margin=dict(l=35, r=45, t=30, b=10), showlegend=False,
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#0B1F3A', family="inherit"))
                                fig.update_xaxes(tickformat="%d-%m", nticks=12, showgrid=False,
                                                 tickfont=_afont, automargin=True, title=None)
                                fig.update_yaxes(showgrid=True, gridcolor='lightgrey',
                                                 tickfont=_afont, automargin=True, title=None)

                            # Attempts (bars) + success-rate combo line on the right axis.
                            fig_a = go.Figure()
                            fig_a.add_bar(x=daily["date"], y=daily["attempts"], marker_color="#e63748",
                                          name="Attempts",
                                          text=daily["attempts"], texttemplate="%{text:,.0f}",
                                          textposition="outside", textfont=dict(size=9, color='#0B1F3A'),
                                          cliponaxis=False)
                            _srnz = daily.loc[daily["attempts"] > 0, "sr"]   # ignore zero-attempt days
                            _srmin = float(_srnz.min()) if not _srnz.empty else 0.0
                            _srmax = float(_srnz.max()) if not _srnz.empty else 100.0
                            # SR axis: min plotted −20% (floored at 0%); max = plotted max +10%
                            # but HARD-capped at 100% so the right axis can never exceed 100%.
                            _y2lo = max(0.0, _srmin * 0.8)
                            _y2hi = 100.0                       # success rate axis hard-capped at 100%
                            fig_a.add_scatter(x=daily["date"], y=daily["sr"], mode="lines+markers+text", yaxis="y2",
                                              name="Success rate",
                                              line=dict(color="#22C36B", width=2), marker=dict(size=4),
                                              text=[f"{_v:.0f}%" for _v in daily["sr"]], textposition="top center",
                                              textfont=dict(size=8, color="#22C36B"))
                            _style(fig_a)
                            fig_a.update_layout(
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                            font=dict(color="#0B1F3A", size=9)),
                                yaxis2=dict(overlaying="y", side="right", range=[_y2lo, _y2hi],
                                            showgrid=False, showticklabels=False))
                            _att_nz = daily.loc[daily["attempts"] > 0, "attempts"]     # left axis min = min bar − 20%
                            fig_a.update_yaxes(range=[float(_att_nz.min()) * 0.8 if not _att_nz.empty else 0.0,
                                                      float(daily["attempts"].max()) * 1.1 if daily["attempts"].max() > 0 else 1.0])

                            # Successes (bars).
                            fig_s = go.Figure()
                            fig_s.add_bar(x=daily["date"], y=daily["success"], marker_color="#22C36B",
                                          name="Successes",
                                          text=daily["success"], texttemplate="%{text:,.0f}",
                                          textposition="outside", textfont=dict(size=9, color='#0B1F3A'),
                                          cliponaxis=False)
                            _style(fig_s)
                            fig_s.update_layout(
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                            font=dict(color="#0B1F3A", size=9)))
                            _suc_nz = daily.loc[daily["success"] > 0, "success"]       # left axis min = min bar − 20%
                            fig_s.update_yaxes(range=[float(_suc_nz.min()) * 0.8 if not _suc_nz.empty else 0.0,
                                                      float(daily["success"].max()) * 1.1 if daily["success"].max() > 0 else 1.0])

                            # Row 1: gateway-share table (left ⅓, same width as the RPGT table) +
                            # revenue-bridge waterfall (right); both built in the Technical Impact section.
                            _gwtab_col, _gwwf_col = st.columns([1, 2])
                            _gwshare_slot = _gwtab_col.container()
                            _gwwf_slot = _gwwf_col.container()
                            # (Attempts & success-rate and Successes bar charts removed.)

    # -------------- Traffic movement (Sankey + delta bar) --------
            # -------------- Technical Impact Charts & Specific Details --------
            with st.container():
                _vmbr_slot = _rpgtbr_slot = None   # revenue-bridge slots (filled from the pre/post section)
                _vmsr_slot = _rpgtsr_slot = None   # approval-rate (success) bridge slots (Batch C)
                _finshare_slot = locals().get("_finshare_slot", None)  # before/after share chart slot
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

                st.markdown("""<style>
                    [data-testid="stPlotlyChart"] { background-color: var(--tav-card) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important; border: 1px solid var(--tav-line) !important; border-radius: 0 !important; padding: 12px !important; margin-bottom: 2rem; overflow: hidden !important; }
                    [data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] .js-plotly-plot, [data-testid="stPlotlyChart"] .plot-container { max-width: 100% !important; }
                </style>""", unsafe_allow_html=True)
            
                eval_df["bank_currency"] = eval_df["bin"] + " - " + eval_df["currency"].str.upper()
                bank_list = sorted(eval_df["bank_currency"].dropna().unique().tolist())
                # Follow the 'Raw daily attempts & successes' Bank/Currency selection instead of a
                # separate filter. Falls back to whole-portfolio if that selection isn't present in
                # the impact frame (naming mismatch), so the lookups below can't crash.
                _raw_bc = ss.get("raw_daily_bc")
                selected_bank = _raw_bc if (_raw_bc in bank_list) else "(All Portfolio)"
            
                if date_col and not adf_30d.empty:
                    adf_30d["date_clean"] = pd.to_datetime(adf_30d[date_col]).dt.date
                
                    if selected_bank == "(All Portfolio)":
                        plot_adf_sel = adf_30d.copy(); b_df = eval_df.copy()
                    else:
                        b_val = selected_bank.split(" - ")[0]
                        c_val = selected_bank.split(" - ")[1].lower()
                        _bj_tmp = eval_df.loc[(eval_df["bin"] == b_val) & (eval_df["currency_join"] == c_val), "bin_join"]
                        # A bank label containing " - " can split wrong, leaving an empty match — fall
                        # back to the bank part's own join key instead of IndexError-ing on .iloc[0].
                        b_join = _bj_tmp.iloc[0] if not _bj_tmp.empty else str(b_val).strip().lower()
                        plot_adf_sel = adf_30d[(adf_30d["bin"].astype(str).str.strip().str.lower() == b_join) & (adf_30d["currency"].astype(str).str.strip().str.lower() == c_val)].copy()
                        b_df = eval_df[(eval_df["bin_join"] == b_join) & (eval_df["currency_join"] == c_val)].copy()
                
                    # vampMid-level SR / gateway-share charts: map gatewayFid → vampMid via
                    # Master_MID_List, then collapse. Only the CHART frames (daily_gw + local copies
                    # below) are remapped — the gatewayFid-grained tables further down keep b_df /
                    # plot_adf_sel untouched.
                    _mm_sr = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    _f2v_sr = {}
                    if os.path.exists(_mm_sr):
                        _mmd_sr = load_mid_list(_mm_sr)
                        _cc_sr = _norm_cols(_mmd_sr)
                        if _cc_sr.get("gatewayfid") and _cc_sr.get("vampmid"):
                            _f2v_sr = _fid2vamp_from(_mmd_sr, _cc_sr["gatewayfid"], _cc_sr["vampmid"])
                    daily_gw = plot_adf_sel.groupby(["date_clean", "gateway"]).agg(att=("attempts", "sum"), succ=("success", "sum")).reset_index()
                    if _f2v_sr:
                        daily_gw["gateway"] = (daily_gw["gateway"].astype(str).str.strip().str.lower()
                                               .map(_f2v_sr).fillna(daily_gw["gateway"].astype(str)))
                        daily_gw = daily_gw.groupby(["date_clean", "gateway"], as_index=False).agg(
                            att=("att", "sum"), succ=("succ", "sum"))
                    daily_gw["sr"] = np.where(daily_gw["att"] > 0, daily_gw["succ"] / daily_gw["att"], np.nan)
                    daily_tot = daily_gw.groupby("date_clean")["att"].sum().reset_index().rename(columns={"att": "tot_att"})
                    daily_gw = daily_gw.merge(daily_tot, on="date_clean", how="left")
                    daily_gw["share"] = np.where(daily_gw["tot_att"] > 0, daily_gw["att"] / daily_gw["tot_att"], 0)

                    _gw_wf = None   # gateway revenue waterfall (rendered beside the table below)
                    if HAS_PLOTLY:
                        import plotly.express as px
                        import plotly.graph_objects as go
                        axis_layout_config = dict(tickfont=dict(color='#0B1F3A', size=9, family="inherit"), title_font=dict(color='#0B1F3A', size=9, family="inherit"), automargin=True)

                        _leg_top = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                        font=dict(color='#0B1F3A', size=9), title_text=None)
                        # Colour = DARKNESS tied to VOLUME so higher-opacity lines are ALSO darker:
                        # sort the palette by luminance (darkest first) and give the darkest shades to
                        # the highest-volume gateways, matching the opacity ramp below.
                        import plotly.colors as _pc_sr
                        _att_by_gw = daily_gw.groupby("gateway")["att"].sum()
                        _gw_sorted = _att_by_gw.sort_values(ascending=False).index.astype(str).tolist()

                        # [FN-364]
                        def _lum_sr(_hex):
                            _h = str(_hex).lstrip("#")
                            try:
                                _r, _g2, _b2 = int(_h[0:2], 16), int(_h[2:4], 16), int(_h[4:6], 16)
                            except Exception:  # noqa: BLE001
                                return 0.0
                            return 0.299 * _r + 0.587 * _g2 + 0.114 * _b2
                        _pal_sr = sorted(_pc_sr.qualitative.Dark24, key=_lum_sr)   # darkest → lightest
                        _cmap_sr = {g: _pal_sr[i % len(_pal_sr)] for i, g in enumerate(_gw_sorted)}
                        # Legend/trace order = ENGINE SCORE (the Bayesian-smoothed gw_sr the optimiser
                        # used, from the eval frame) descending; fall back to the raw 30D success rate
                        # for any gateway not present in the eval frame.
                        _es_gw = daily_gw.groupby("gateway").apply(
                            lambda d: (d["succ"].sum() / d["att"].sum()) if d["att"].sum() > 0 else 0.0)
                        _eng_score = {}
                        if isinstance(b_df, pd.DataFrame) and "gw_sr" in b_df.columns and not b_df.empty:
                            _bb = b_df.copy()
                            if _f2v_sr and "gateway" in _bb.columns:   # vampMid-level score ordering
                                _bb["gateway"] = (_bb["gateway"].astype(str).str.strip().str.lower()
                                                  .map(_f2v_sr).fillna(_bb["gateway"].astype(str)))
                            _bb["_g"] = ((_bb["gateway"] if "gateway" in _bb.columns else _bb["gateway_join"])
                                         .astype(str).str.strip().str.lower())
                            _bb["_w"] = (pd.to_numeric(_bb.get("cell_att", 1.0), errors="coerce").fillna(0.0)
                                         * pd.to_numeric(_bb.get("share", 1.0), errors="coerce").fillna(0.0))
                            for _gname, _d in _bb.groupby("_g"):
                                _wsum = float(_d["_w"].sum())
                                _eng_score[_gname] = (float((_d["gw_sr"] * _d["_w"]).sum() / _wsum) if _wsum > 0
                                                      else float(pd.to_numeric(_d["gw_sr"], errors="coerce").fillna(0.0).mean()))

                        # [FN-365]
                        def _score_of(_g):
                            return _eng_score.get(str(_g).strip().lower(), float(_es_gw.get(_g, 0.0)))
                        _gw_es_sorted = sorted(_gw_sorted, key=lambda g: -_score_of(g))
                        fig_sr = px.line(daily_gw, x="date_clean", y="sr", color="gateway",
                                         markers=True, color_discrete_map=_cmap_sr,
                                         category_orders={"gateway": _gw_es_sorted})
                        # Opacity ∝ attempts share, with a MUCH steeper contrast (power curve +
                        # low floor) so thin gateways fade right back and high-volume ones read solid.
                        _att_map = {str(_k).strip().lower(): float(_v) for _k, _v in _att_by_gw.items()}
                        _maxa = max(_att_map.values()) if _att_map else 0.0
                        for _tr in fig_sr.data:
                            _s = _att_map.get(str(_tr.name).strip().lower(), 0.0)
                            _frac = (_s / _maxa) if _maxa > 0 else 0.0
                            _tr.opacity = max(0.12, min(1.0, _frac ** 1.4))
                        fig_sr.update_layout(
                            height=460, margin=dict(l=35, r=40, t=28, b=10),  # match fig_sh height
                            yaxis_title=None, xaxis_title=None, legend_title=None,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#0B1F3A', family="inherit"), legend=_leg_top)
                        fig_sr.update_yaxes(tickformat=".0%", showgrid=True, gridcolor='lightgrey', **axis_layout_config)
                        fig_sr.update_xaxes(tickformat="%d-%m", nticks=20, showgrid=False, **axis_layout_config)
                    
                        # Daily historical share bars + one final "Proposed" bar
                        # showing the engine's proposed split, separated by a dark
                        # grey dotted line (like the T-months divider on tab 2).
                        prop_bar = pd.DataFrame(columns=["xlab", "gateway", "share"])
                        if not b_df.empty and {"gateway", "cell_att", "share"}.issubset(b_df.columns):
                            _pb = b_df.copy()
                            if _f2v_sr:   # vampMid-level proposed bar
                                _pb["gateway"] = (_pb["gateway"].astype(str).str.strip().str.lower()
                                                  .map(_f2v_sr).fillna(_pb["gateway"].astype(str)))
                            _pb["_pv"] = _pb["cell_att"] * _pb["share"]
                            _pb = _pb.groupby("gateway", as_index=False)["_pv"].sum()
                            _tot = _pb["_pv"].sum()
                            _pb["share"] = np.where(_tot > 0, _pb["_pv"] / _tot, 0.0)
                            _pb["xlab"] = "Proposed"
                            prop_bar = _pb[["xlab", "gateway", "share"]]

                        # Share chart shows only the LAST 7 DAYS of history (the SR
                        # line chart above and the tables stay on 30D).
                        _dg = daily_gw.copy()
                        if not _dg.empty:
                            _dmax = pd.to_datetime(_dg["date_clean"]).max()
                            _dg = _dg[pd.to_datetime(_dg["date_clean"]) >= (_dmax - pd.Timedelta(days=6))]
                        _dg["xlab"] = pd.to_datetime(_dg["date_clean"]).dt.strftime("%d-%m")
                        _order = list(pd.to_datetime(_dg["date_clean"]).drop_duplicates().sort_values().dt.strftime("%d-%m"))
                        if not prop_bar.empty:
                            _order = _order + ["Proposed"]
                        _combined = pd.concat([_dg[["xlab", "gateway", "share"]], prop_bar], ignore_index=True)

                        # Legend + stack order = highest PROPOSED share → lowest.
                        _gw_prop_sorted = (prop_bar.sort_values("share", ascending=False)["gateway"].astype(str).tolist()
                                           if not prop_bar.empty else
                                           daily_gw.groupby("gateway")["share"].sum().sort_values(ascending=False).index.astype(str).tolist())
                        fig_sh = px.bar(_combined, x="xlab", y="share", color="gateway",
                                        text="share", category_orders={"xlab": _order, "gateway": _gw_prop_sorted})
                        # Show the share value on each bar segment (plotly hides ones too small to fit).
                        fig_sh.update_traces(texttemplate="%{text:.0%}", textposition="inside",
                                             insidetextanchor="middle", textfont=dict(size=9, color="#FFFFFF"),
                                             cliponaxis=False)
                        fig_sh.update_layout(
                            height=460, margin=dict(l=35, r=40, t=28, b=10),
                            barmode='stack', yaxis_title=None, xaxis_title=None, legend_title=None,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#0B1F3A', family="inherit"), legend=_leg_top)
                        fig_sh.update_yaxes(tickformat=".0%", showgrid=True, gridcolor='lightgrey', **axis_layout_config)
                        fig_sh.update_xaxes(type="category", showgrid=False, **axis_layout_config)
                        if not prop_bar.empty and len(_order) >= 2:
                            fig_sh.add_vline(x=len(_order) - 1.5, line_width=2, line_dash="dot", line_color="#555")
                        # fig_sr and fig_sh are rendered below in the two-row layout
                        # (Row A: SR line + Current-vs-Proposed table; Row B: waterfall
                        # + 100% stacked share bar), so nothing is drawn here.

                        # Revenue waterfall by gatewayFid: total current -> per-gateway
                        # delta -> total proposed (30D revenue).
                        if not b_df.empty and {"cell_att", "baseline_share", "gw_sr", "avg_ticket", "exp_rev", "gateway"}.issubset(b_df.columns):
                            _wf = b_df.copy()
                            _wf["pre_rev"] = _wf["cell_att"] * _wf["baseline_share"] * _wf["gw_sr"] * _wf["avg_ticket"]
                            _wg = _wf.groupby("gateway", as_index=False).agg(pre=("pre_rev", "sum"), post=("exp_rev", "sum"))
                            _wg["delta"] = _wg["post"] - _wg["pre"]
                            _wg = _wg[(_wg["pre"].abs() + _wg["post"].abs()) > 0].sort_values("delta", ascending=False)
                            if not _wg.empty:
                                _tot_pre, _tot_post = float(_wg["pre"].sum()), float(_wg["post"].sum())
                                _xs = ["Current"] + _wg["gateway"].tolist() + ["Proposed"]
                                _ys = [_tot_pre] + _wg["delta"].tolist() + [0.0]
                                _labs = [_tot_pre] + _wg["delta"].tolist() + [_tot_post]
                                _meas = ["absolute"] + ["relative"] * len(_wg) + ["total"]
                                _run, _peaks = _tot_pre, [_tot_pre, _tot_post]
                                for _dv in _wg["delta"]:
                                    _run += float(_dv)
                                    _peaks.append(_run)
                                _wlo, _whi = min(_peaks) * 0.95, max(_peaks) * 1.05   # x-min = trough − 5%, x-max = peak + 5%
                                # Every bar (current, proposed, and per-gateway deltas) in $x.xk.
                                _gtext = [f"${_v/1e3:,.1f}k" for _v in _labs]
                                figw = go.Figure(go.Waterfall(
                                    orientation="h", measure=_meas, y=_xs, x=_ys,
                                    text=_gtext, textposition="outside",
                                    textfont=dict(size=8, color='#0B1F3A'),
                                    customdata=_labs,
                                    hovertemplate="%{y}<br>$%{customdata:,.0f}<extra></extra>",
                                    connector=dict(line=dict(color="#B9C6DA")),
                                    increasing=dict(marker=dict(color="#22C36B")),
                                    decreasing=dict(marker=dict(color="#e63748")),
                                    totals=dict(marker=dict(color="#0B1F3A")), showlegend=False))
                                figw.update_layout(
                                    height=460, margin=dict(l=35, r=40, t=14, b=10),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#0B1F3A', family="inherit", size=9))
                                figw.update_xaxes(range=[_wlo, _whi], showgrid=True, gridcolor='lightgrey',
                                                  tickprefix="$", tickfont=dict(color='#0B1F3A', size=9), title=None, automargin=True)
                                figw.update_yaxes(type="category", autorange="reversed", showgrid=False,
                                                  tickfont=dict(color='#0B1F3A', size=9), title=None, automargin=True)
                                _gw_wf = figw
                    else:
                        st.caption("Plotly is required to render these interactive charts.")

                    # --- Current vs Proposed Share Table ---
                    if b_df.empty:
                        b_df = pd.DataFrame(columns=["gateway_join", "gateway", "cell_att", "baseline_share", "share", "gw_sr", "avg_ticket", "exp_succ", "exp_rev"])
                        b_df["curr_vol"] = 0.0; b_df["prop_vol"] = 0.0
                    else:
                        if "cell_att" in b_df.columns and "baseline_share" in b_df.columns: b_df["curr_vol"] = b_df["cell_att"] * b_df["baseline_share"]
                        else: b_df["curr_vol"] = 0.0
                        if "cell_att" in b_df.columns and "share" in b_df.columns: b_df["prop_vol"] = b_df["cell_att"] * b_df["share"]
                        else: b_df["prop_vol"] = 0.0
                
                    if b_df.empty:
                        gw_sh = pd.DataFrame(columns=["gateway_join", "Gateway", "curr_vol", "Expected Attempts", "Expected Success", "Expected_Rev"])
                    else:
                        gw_sh = b_df.groupby("gateway_join").agg(Gateway=("gateway", "first"), curr_vol=("curr_vol", "sum"), Expected_Attempts=("prop_vol", "sum"), Expected_Success=("exp_succ", "sum"), Expected_Rev=("exp_rev", "sum"), avg_ticket=("avg_ticket", "first")).reset_index().rename(columns={"Expected_Attempts": "Expected Attempts"})
                
                    raw_gw = plot_adf_sel.groupby("gateway").agg(raw_att=("attempts", "sum"), raw_succ=("success", "sum"), raw_amount=("succ_amount", "sum")).reset_index().rename(columns={"gateway": "gateway_join"})
                    raw_gw["Raw 30D Success Rate"] = np.where(raw_gw["raw_att"] > 0, raw_gw["raw_succ"] / raw_gw["raw_att"], 0)
                
                    gw_sh = gw_sh.merge(raw_gw, on="gateway_join", how="outer")
                
                    if "Gateway" in gw_sh.columns and "gateway_join" in gw_sh.columns:
                        gw_sh["Gateway"] = gw_sh["Gateway"].fillna(gw_sh["gateway_join"])
                
                    for safe_col in ["Expected Attempts", "Expected Success", "Expected_Rev", "raw_att", "raw_succ", "raw_amount", "curr_vol", "Raw 30D Success Rate", "avg_ticket"]:
                        if safe_col in gw_sh.columns: gw_sh[safe_col] = gw_sh[safe_col].fillna(0)
                        else: gw_sh[safe_col] = 0.0
                
                    # Proposed Share = the engine's allocation (volume-weighted).
                    t_raw = gw_sh["raw_att"].sum()
                    _eng_exp = gw_sh["Expected Attempts"]            # engine cell_att * share
                    _t_eng = _eng_exp.sum()
                    prop_frac = np.where(_t_eng > 0, _eng_exp / _t_eng, 0.0)
                    sr_frac = gw_sh["Raw 30D Success Rate"]          # still a fraction here

                    # 30D-consistent view:
                    #   Expected Attempts = total 30D attempts * proposed share
                    #   Expected Success  = Expected Attempts * 30D SR
                    gw_sh["Expected Attempts"] = t_raw * prop_frac
                    gw_sh["Expected Success"] = gw_sh["Expected Attempts"] * sr_frac
                    # Like-for-like: pre valued at Avg txn value (Bank×Cur) × Raw Successes (30D).
                    gw_sh["Expected Revenue Impact"] = gw_sh["Expected_Rev"] - gw_sh["avg_ticket"] * gw_sh["raw_succ"]

                    gw_sh["Current Share"] = np.where(t_raw > 0, (gw_sh["raw_att"] / t_raw) * 100, 0)
                    gw_sh["Proposed Share"] = prop_frac * 100
                    gw_sh["Shift (pp)"] = gw_sh["Proposed Share"] - gw_sh["Current Share"]
                    gw_sh["Raw 30D Success Rate"] = gw_sh["Raw 30D Success Rate"] * 100

                    # Engine Score = the per-gateway rate the engine actually scores on
                    # (ss["agg_sr"] at currency × parent-bank × gateway), shown as a %.
                    # For most engines that's the Bayesian-SHRUNK success_rate; Thompson uses
                    # its own Beta posterior from the RAW (time-decayed) counts — no κ shrinkage
                    # — so for Thompson show the raw rate, which is what it genuinely scores on.
                    # For a single Bank/Currency it's that profile's score; for the whole portfolio
                    # it's volume-weighted (by all-time attempts) per gateway.
                    _escore = {}
                    _agg = ss.get("agg_sr")
                    _score_engine = ss.get("variations_engine", "softmax")
                    _score_col = "raw_rate" if _score_engine == "thompson" else "success_rate"
                    if _agg is not None and not getattr(_agg, "empty", True) and _score_col in _agg.columns:
                        _a = _agg.copy()
                        _a["_gj"] = _a["gateway"].astype(str).str.strip().str.lower()
                        _a["_sr"] = pd.to_numeric(_a[_score_col], errors="coerce").fillna(0.0)
                        _a["_at"] = pd.to_numeric(_a.get("attempts", 0.0), errors="coerce").fillna(0.0)
                        if selected_bank != "(All Portfolio)":
                            _bv = selected_bank.split(" - ")[0]
                            _cv = selected_bank.split(" - ")[1].lower()
                            _b2b = ss.get("bin_to_bank", {})
                            _pb = str(_b2b.get(_bv, _b2b.get(str(_bv).strip().lower(), _bv))).strip().lower()
                            _a = _a[(_a["currency"].astype(str).str.strip().str.lower() == _cv)
                                    & (_a["bin"].astype(str).str.strip().str.lower() == _pb)]
                            _escore = _a.groupby("_gj")["_sr"].mean().to_dict()
                        else:
                            _escore = (_a.groupby("_gj")
                                       .apply(lambda d: (d["_sr"] * d["_at"]).sum() / max(d["_at"].sum(), 1e-9))
                                       .to_dict())
                    gw_sh["Engine Score"] = (gw_sh["gateway_join"].astype(str).str.strip().str.lower()
                                             .map(_escore).fillna(0.0)) * 100

                    # Order by Engine Score (highest first).
                    gw_sh = gw_sh.sort_values("Engine Score", ascending=False)

                    gw_sh_view = gw_sh.rename(columns={
                        "raw_att": "Attempts Pre", "Expected Attempts": "Attempts Post",
                        "Current Share": "Share Pre", "Proposed Share": "Share Post",
                        "Expected Revenue Impact": "$ Impact",
                    })[["Gateway", "Engine Score", "Attempts Pre", "Attempts Post",
                        "Share Pre", "Share Post", "Shift (pp)", "$ Impact"]].copy()
                    if selected_bank == "(All Portfolio)": gw_sh_view = gw_sh_view.head(20)

                    _tw = gw_sh["Expected Attempts"].sum()
                    _es_total = (float((gw_sh["Engine Score"] * gw_sh["Expected Attempts"]).sum() / _tw)
                                 if _tw > 0 else 0.0)
                    gw_total_row = {
                        "Gateway": "TOTAL", "Engine Score": _es_total,
                        "Attempts Pre": t_raw, "Attempts Post": gw_sh["Expected Attempts"].sum(),
                        "Share Pre": 100.0, "Share Post": 100.0, "Shift (pp)": 0.0,
                        "$ Impact": gw_sh["Expected Revenue Impact"].sum(),
                    }
                    gw_sh_view = pd.concat([gw_sh_view, pd.DataFrame([gw_total_row])], ignore_index=True)

                    # Engine-score colour scale: RELATIVE to the Engine Scores in this
                    # Bank×Currency table — red (lowest) → green (highest).
                    _es_nz = gw_sh_view.loc[gw_sh_view["Gateway"] != "TOTAL", "Engine Score"]
                    _es_max = float(_es_nz.max()) if not _es_nz.empty else 1.0
                    _es_min = float(_es_nz.min()) if not _es_nz.empty else 0.0
                    _es_rng = (_es_max - _es_min) if (_es_max - _es_min) > 1e-9 else 1.0

                    # width:auto + nowrap → each column is only as wide as its longest cell.
                    html_gw = ['<div style="box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow:auto; width:100%; height:460px; background-color: var(--tav-card); border: 1px solid var(--tav-line);">']
                    html_gw.append('<table style="width:100%; border-collapse:collapse; font-size:0.74rem;"><tr>')
                    for col in gw_sh_view.columns:
                        html_gw.append(f'<th style="background-color: var(--tav-red); color:#FFF; padding:6px 10px; white-space:nowrap; text-align:{"left" if col=="Gateway" else "right"};">{col}</th>')
                    html_gw.append('</tr>')
                    for _, r in gw_sh_view.iterrows():
                        is_total = (r["Gateway"] == "TOTAL")
                        t_b = "border-top:2px solid var(--tav-line) !important;" if is_total else ""
                        html_gw.append('<tr>')
                        for col in gw_sh_view.columns:
                            val = r[col]
                            c_sh = "#22C36B" if ("Shift" in col or "Impact" in col) and val > 0 and not is_total else ("#e63748" if ("Shift" in col or "Impact" in col) and val < 0 and not is_total else "var(--tav-ink)")
                            _bg = ""
                            if "Share" in col or col == "Engine Score":
                                str_val = f"{val:.2f}%"
                            elif "Shift" in col:
                                str_val = f"{val:+.2f} pp"
                            elif "Impact" in col:
                                str_val = f"${val:+,.0f}"
                            elif col in ["Attempts Pre", "Attempts Post"]:
                                str_val = f"{val:,.0f}"
                            else:
                                str_val = str(val)
                            # Engine Score: red (lowest in table) → green (highest), relative scale.
                            if col == "Engine Score" and not is_total:
                                _frac = max(0.0, min(1.0, (float(val) - _es_min) / _es_rng))
                                _rr = int(round(230 + (34 - 230) * _frac))
                                _gg = int(round(55 + (195 - 55) * _frac))
                                _bb = int(round(72 + (107 - 72) * _frac))
                                _bg = f"background-color: rgba({_rr},{_gg},{_bb},0.38);"
                            html_gw.append(f'<td style="padding:4px 10px; white-space:nowrap; text-align:{"left" if col=="Gateway" else "right"}; color:{c_sh}; font-weight:{"800" if is_total else "normal"}; {_bg} {t_b}">{str_val}</td>')
                        html_gw.append('</tr>')
                    html_gw.append('</table></div>')

                    # Current vs Proposed Gateway Share moves to the Bank Analysis slot (left of the
                    # Attempts/Successes charts). SR-by-gatewayFid line chart takes the full width here.
                    _gw_share_html = "".join(html_gw)   # header removed
                    # Gateway-share table → left slot; revenue-bridge waterfall → right slot,
                    # both in the Bank Analysis row (top). Fall back to inline if slots are absent.
                    (_gwshare_slot or st).markdown(_gw_share_html, unsafe_allow_html=True)
                    if _gw_wf is not None:
                        (_gwwf_slot or st).plotly_chart(_gw_wf, use_container_width=True)

                    # Revenue bridges (by vampMid / by RPGT) — reserved ABOVE the SR / share charts,
                    # filled later from the pre/post section once _evv / _rp are computed.
                    # (Revenue/Transactions toggle removed — the revenue bridge is the only view.)
                    _br_sort = "Top absolute movers"   # fixed: always show the biggest absolute movers
                    _br_money = True                   # Revenue bridge only (Txn toggle removed)
                    # Row 1 = vampMid bridges (revenue | success); Row 2 = RPGT bridges (revenue |
                    # success). A trailing spacer keeps each pair's combined width within the pre/post
                    # impact table's footprint rather than spanning the whole tab.
                    # Create the bridge column-slots INTO the reserved container above the
                    # bank-blocked / share-chart row, so the bridges render at the top of the tab.
                    with _finbridge_slot:
                        _vmc1, _vmc2, _vmc3 = st.columns([1, 1, 0.9])
                        _vmbr_slot = _vmc1.container()     # revenue by vampMid
                        _vmsr_slot = _vmc2.container()     # success by vampMid
                        _rpc1, _rpc2, _rpc3 = st.columns([1, 1, 0.9])
                        _rpgtbr_slot = _rpc1.container()   # revenue by RPGT
                        _rpgtsr_slot = _rpc2.container()   # success by RPGT

                    # SR + gateway-share charts (now vampMid-level, no headers) render on the
                    # Mid Detail tab, side by side.
                    if HAS_PLOTLY:
                        # Mid Detail's SR-by-date line chart AND the 100%-stacked gateway-share bar chart
                        # are removed; the SR-by-day data now renders as a per-gateway trellis on the
                        # Gateway Detail sub-tab (all Plotly, x-axis in dd-mm).
                        # Keep only the most recent 30 days for these charts — some rows carry stray /
                        # backfilled dates going back years, which otherwise stretch the axis.
                        # Gateway Detail charts are portfolio-wide with their OWN Bank + RPGT filters
                        # (independent of the Bank Analysis selection).
                        with _t_gwdet:
                            _gwf1, _gwf2, _gwfsp = st.columns([1, 1, 4])
                            _bopts = (["(All)"] + sorted(adf_30d["bin"].astype(str).str.strip().unique().tolist())
                                      if "bin" in adf_30d.columns else ["(All)"])
                            _ropts = (["(All)"] + sorted(adf_30d["rpgt"].astype(str).str.title().str.strip().unique().tolist())
                                      if "rpgt" in adf_30d.columns else ["(All)"])
                            _gw_selb = _gwf1.selectbox("Bank", _bopts, key="gwdet_bank")
                            _gw_selr = _gwf2.selectbox("RPGT", _ropts, key="gwdet_rpgt")
                        _src_gw = adf_30d
                        if _gw_selb != "(All)" and "bin" in _src_gw.columns:
                            _src_gw = _src_gw[_src_gw["bin"].astype(str).str.strip() == _gw_selb]
                        if _gw_selr != "(All)" and "rpgt" in _src_gw.columns:
                            _src_gw = _src_gw[_src_gw["rpgt"].astype(str).str.title().str.strip() == _gw_selr]
                        if "date_clean" in _src_gw.columns:
                            _dgf = (_src_gw.groupby(["date_clean", "gateway"], as_index=False)
                                    .agg(att=("attempts", "sum"), succ=("success", "sum")))
                            if _f2v_sr:
                                _dgf["gateway"] = (_dgf["gateway"].astype(str).str.strip().str.lower()
                                                   .map(_f2v_sr).fillna(_dgf["gateway"].astype(str)))
                                _dgf = _dgf.groupby(["date_clean", "gateway"], as_index=False).agg(
                                    att=("att", "sum"), succ=("succ", "sum"))
                            _dgf["sr"] = np.where(_dgf["att"] > 0, _dgf["succ"] / _dgf["att"], np.nan)
                        else:
                            _dgf = daily_gw
                        _dg30 = _dgf.copy()
                        _dg30["_dt"] = pd.to_datetime(_dg30["date_clean"], errors="coerce")
                        _dg30 = _dg30.dropna(subset=["_dt"])
                        if not _dg30.empty:
                            _dg30 = _dg30[_dg30["_dt"] >= (_dg30["_dt"].max() - pd.Timedelta(days=29))]
                        # Dynamic left margin sized to the longest MID name (so y-axis labels aren't clipped).
                        _gw_names = _dg30["gateway"].astype(str).unique().tolist() if not _dg30.empty else []
                        _hm_lm = int(min(320, 12 + (max((len(g) for g in _gw_names), default=10)) * 7))
                        # ---- Gateway Detail: 30D raw SR % vs 30D raw volume, one point per MID (TOP) ----
                        try:
                            _sc = _dg30.groupby("gateway", as_index=False).agg(
                                att=("att", "sum"), succ=("succ", "sum"))
                            _sc["sr"] = np.where(_sc["att"] > 0, _sc["succ"] / _sc["att"], np.nan)
                            _sc = _sc.dropna(subset=["sr"])
                            if not _sc.empty:
                                _scfig = px.scatter(_sc, x="att", y="sr", text="gateway")
                                _scfig.update_traces(
                                    textposition="top center", marker=dict(size=11, color="#0B1F3A"),
                                    textfont=dict(size=9, color="#0B1F3A"),
                                    hovertemplate=("MID: %{text}<br>30D raw volume: %{x:,.0f}"
                                                   "<br>30D raw SR %% : %{y:.2%}<extra></extra>"))
                                _scfig.update_yaxes(tickformat=".0%", title_text=None,
                                                    tickfont=dict(color="#0B1F3A", size=9), gridcolor="lightgrey")
                                _scfig.update_xaxes(title_text=None, tickfont=dict(color="#0B1F3A", size=9))
                                _scfig.update_layout(
                                    height=430, margin=dict(l=45, r=10, t=20, b=20),
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#0B1F3A", family="inherit"))
                                _t_gwdet.plotly_chart(_scfig, use_container_width=True)
                        except Exception as _sce:  # noqa: BLE001
                            _t_gwdet.caption(f"MID scatter unavailable: {type(_sce).__name__}: {_sce}")
                        try:
                            from plotly.subplots import make_subplots as _msub
                            import plotly.graph_objects as _gotr
                            _dg_tr = _dg30.sort_values("_dt").copy()
                            # This gateway's SR, and the SR of ALL OTHER gateways that day (benchmark).
                            _dg_tr["sr"] = np.where(_dg_tr["att"] > 0, _dg_tr["succ"] / _dg_tr["att"], np.nan)
                            _tot_tr = _dg_tr.groupby("_dt").agg(_ts=("succ", "sum"), _ta=("att", "sum"))
                            _dg_tr = _dg_tr.merge(_tot_tr, on="_dt", how="left")
                            _oth_att = _dg_tr["_ta"] - _dg_tr["att"]
                            _dg_tr["sr_excl"] = np.where(_oth_att > 0,
                                                         (_dg_tr["_ts"] - _dg_tr["succ"]) / _oth_att, np.nan)
                            _mids_tr = [m for m in _gw_es_sorted if m in set(_dg_tr["gateway"].astype(str))]
                            _mids_tr += [m for m in sorted(_dg_tr["gateway"].astype(str).unique())
                                         if m not in _mids_tr]
                            _ncol_tr = 3
                            _nrow_tr = int(np.ceil(len(_mids_tr) / _ncol_tr))
                            _fig_tr = _msub(rows=_nrow_tr, cols=_ncol_tr, subplot_titles=_mids_tr,
                                            vertical_spacing=0.06, horizontal_spacing=0.04)
                            _hov_tr = "Date : %{x|%d-%m}<br>SR %% : %{y:.2%}<extra></extra>"
                            for _i, _g in enumerate(_mids_tr):
                                _r, _c = _i // _ncol_tr + 1, _i % _ncol_tr + 1
                                _d = _dg_tr[_dg_tr["gateway"].astype(str) == _g].sort_values("_dt")
                                _x = _d["_dt"]
                                _A = _d["sr"].to_numpy(float)          # this gateway
                                _B = _d["sr_excl"].to_numpy(float)     # all other gateways
                                # GREEN shade where this > others (between others and this), 30% opacity
                                _fig_tr.add_trace(_gotr.Scatter(x=_x, y=_B, mode="lines", line=dict(width=0),
                                    hoverinfo="skip", showlegend=False), row=_r, col=_c)
                                _fig_tr.add_trace(_gotr.Scatter(x=_x, y=np.fmax(_A, _B), mode="lines",
                                    line=dict(width=0), fill="tonexty", fillcolor="rgba(34,195,107,0.30)",
                                    hoverinfo="skip", showlegend=False), row=_r, col=_c)
                                # RED shade where this < others, 30% opacity
                                _fig_tr.add_trace(_gotr.Scatter(x=_x, y=_B, mode="lines", line=dict(width=0),
                                    hoverinfo="skip", showlegend=False), row=_r, col=_c)
                                _fig_tr.add_trace(_gotr.Scatter(x=_x, y=np.fmin(_A, _B), mode="lines",
                                    line=dict(width=0), fill="tonexty", fillcolor="rgba(230,55,72,0.30)",
                                    hoverinfo="skip", showlegend=False), row=_r, col=_c)
                                # dashed grey benchmark (all others, 50%) + dark-ink 'this gateway' on top
                                _fig_tr.add_trace(_gotr.Scatter(x=_x, y=_B, mode="lines",
                                    line=dict(color="#888888", dash="dash"), opacity=0.5, showlegend=False,
                                    hovertemplate=_hov_tr), row=_r, col=_c)
                                _fig_tr.add_trace(_gotr.Scatter(x=_x, y=_A, mode="lines",
                                    line=dict(color="#0B1F3A"), showlegend=False,
                                    hovertemplate=_hov_tr), row=_r, col=_c)
                            _fig_tr.update_annotations(font=dict(size=11, color="#0B1F3A"))
                            _fig_tr.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="lightgrey",
                                                 tickfont=dict(color="#0B1F3A", size=9))
                            _fig_tr.update_xaxes(showticklabels=False, showgrid=False)
                            _fig_tr.update_layout(
                                height=max(468, 273 * _nrow_tr), margin=dict(l=30, r=20, t=30, b=10),
                                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0B1F3A", family="inherit"))
                            _t_gwdet.plotly_chart(_fig_tr, use_container_width=True)
                        except Exception as _tre:  # noqa: BLE001
                            _t_gwdet.caption(f"Gateway trellis unavailable: {type(_tre).__name__}: {_tre}")


            # ------------------------------------------------------------------
            # Pre / post impact visuals (moved here from the Routing engine tab).
            # These reflect the variation selected with the Risk <-> Conversion slider above.
            # ------------------------------------------------------------------
            with st.container(border=True):
                # Reuse the per-variation eval frame already computed for the top row
                # (precomputed at compute time) instead of recomputing it every rerun.
                _ev = eval_df

                # Revenue table + SR chart render into the slots beneath the Bank table in the
                # Bank×Currency row's left column (all three same width; waterfall spans them).
                with _rpgt_tab_slot:
                    # ---- Table: 30D revenue by RPGT (pre vs post) — header removed ----
                    # PRE uses the MODELLED pre_rev baseline (cell_att × baseline_share × gw_sr ×
                    # avg_ticket) and POST = expected revenue — the SAME basis as the Expected Revenue
                    # card and every other bridge, so all revenue bridges reconcile. Keyed on _rl
                    # (rpgt lowercased) to match _post_l.
                    _post_l = (_ev.assign(_rl=_ev["rpgt"].astype(str).str.strip().str.lower())
                               .groupby("_rl").agg(post=("post_rev", "sum"), disp=("rpgt", "first")))
                    _base_l = (_ev.assign(_rl=_ev["rpgt"].astype(str).str.strip().str.lower())
                               .groupby("_rl")["pre_rev"].sum())
                    _rl_all = sorted(set(_post_l.index) | set(_base_l.index))
                    _rp = pd.DataFrame([{
                        "rpgt": (_post_l.loc[_k, "disp"] if _k in _post_l.index else str(_k).title()),
                        "pre": float(_base_l.get(_k, 0.0)),
                        "post": float(_post_l.loc[_k, "post"]) if _k in _post_l.index else 0.0,
                    } for _k in _rl_all]).sort_values("post", ascending=False)
                    # Success rate per RPGT (pre/post) — added as columns to this table (the
                    # separate SR-by-RPGT bar chart is removed).
                    _srg = _ev.groupby("rpgt", as_index=False).agg(
                        pre_succ=("pre_succ", "sum"), pre_att=("pre_att", "sum"),
                        post_succ=("post_succ", "sum"), post_att=("post_att", "sum"))
                    _srg["_k"] = _srg["rpgt"].astype(str).str.strip().str.lower()
                    _sr_pre, _sr_post = {}, {}
                    for _, r in _srg.iterrows():   # one pass instead of two
                        _k = r["_k"]
                        _sr_pre[_k] = r["pre_succ"] / r["pre_att"] if r["pre_att"] > 0 else 0.0
                        _sr_post[_k] = r["post_succ"] / r["post_att"] if r["post_att"] > 0 else 0.0
                    _srt_pre = (float(_srg["pre_succ"].sum()) / float(_srg["pre_att"].sum())) if float(_srg["pre_att"].sum()) > 0 else 0.0
                    _srt_post = (float(_srg["post_succ"].sum()) / float(_srg["post_att"].sum())) if float(_srg["post_att"].sum()) > 0 else 0.0
                    if not _rp.empty:
                        _rp["delta"] = _rp["post"] - _rp["pre"]
                    # (RPGT revenue table removed; _rp kept for the RPGT revenue bridge.)

                # ---- Table: biggest increases/decreases by bank for a vampMid (header removed) ----
                # Map gatewayFid -> vampMid (Master_MID_List) so the picker groups fids by MID.
                _mm_bi = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                _f2v_bi = {}
                if os.path.exists(_mm_bi):
                    _mmdf_bi = load_mid_list(_mm_bi)
                    _cc_bi = _norm_cols(_mmdf_bi)
                    _gc_bi, _vc_bi = _cc_bi.get("gatewayfid"), _cc_bi.get("vampmid")
                    if _gc_bi and _vc_bi:
                        _f2v_bi = _fid2vamp_from(_mmdf_bi, _gc_bi, _vc_bi)
                _evv = _ev.copy()
                _evv["_vmid"] = (_evv["gateway"].astype(str).str.strip().str.lower().map(_f2v_bi)
                                 .fillna(_evv["gateway"].astype(str)))

                # Before → after volume-share chart (now in Financial Impact, next to the
                # bank-block table) — needs _evv with _vmid, so render it into the reserved slot here.
                _fin_render_share_chart(_evv, _finshare_slot)

                # ---- Portfolio revenue bridges: by vampMid and by RPGT (side by side) ----
                # Same waterfall format as the Bank×Currency bridge. The vampMid bridge uses the
                # per-vampMid pre/post revenue (as the per-vampMid bank table does); the RPGT bridge
                # uses the per-RPGT revenue table (_rp), so each reconciles to its own table.
                if HAS_PLOTLY:
                    _bpre, _bpost = ("pre_rev", "post_rev") if _br_money else ("pre_vol", "post_vol")
                    _vmbr = _evv.groupby("_vmid", as_index=False).agg(
                        pre_rev=("pre_rev", "sum"), post_rev=("post_rev", "sum"),
                        pre_vol=("pre_vol", "sum"), post_vol=("post_vol", "sum"))
                    _vm_wf_all = None
                    _vbi = _bridge_items(_vmbr, "_vmid", _bpre, _bpost, _br_sort, "Other vampMids", _max_n=None)
                    if _vbi is not None:
                        _vm_wf_all = _rev_bridge_waterfall(*_vbi, money=_br_money)
                    # RPGT bridge from the per-RPGT revenue table (_rp) + per-RPGT volume from _ev.
                    _rpgt_wf_all = None
                    try:
                        if _rp is not None and not _rp.empty:
                            _rpx = _rp.rename(columns={"pre": "pre_rev", "post": "post_rev"}).copy()
                            _rpx["_rl"] = _rpx["rpgt"].astype(str).str.strip().str.lower()
                            _rpvol = (_ev.assign(_rl=_ev["rpgt"].astype(str).str.strip().str.lower())
                                      .groupby("_rl").agg(pre_vol=("pre_vol", "sum"), post_vol=("post_vol", "sum")))
                            _rpx = _rpx.merge(_rpvol, on="_rl", how="left").fillna({"pre_vol": 0.0, "post_vol": 0.0})
                            _rbi = _bridge_items(_rpx, "rpgt", _bpre, _bpost, _br_sort, "Other RPGTs", _max_n=None)
                            if _rbi is not None:
                                # Left margin auto-sizes to the RPGT label length.
                                _rpgt_wf_all = _rev_bridge_waterfall(*_rbi, money=_br_money, height=392)
                    except NameError:
                        _rpgt_wf_all = None
                    # Render into the slots reserved above the SR / gateway-share charts
                    # (fall back to inline here if those slots weren't created this run).
                    if _vm_wf_all is not None:
                        (_vmbr_slot or st).plotly_chart(_vm_wf_all, use_container_width=True)
                    if _rpgt_wf_all is not None:
                        (_rpgtbr_slot or st).plotly_chart(_rpgt_wf_all, use_container_width=True)

                    # ---- Approval-rate (success) bridges: current → proposed success %, by vampMid
                    #      and by RPGT (Batch C). Reconciled EXACTLY to the "Expected Success Rate
                    #      (30D)" red card: same denominator (base_att = 30D attempts), same endpoints
                    #      (current = base_sr = base_succ/base_att, proposed = exp_sr = new_succ/
                    #      base_att). Proposed per-group uses the modelled post_succ (sums to new_succ);
                    #      the modelled per-group baseline is rescaled so it sums to the card's
                    #      base_succ, so the Current bar lands on base_sr to the last decimal.
                    try:
                        _sev = _evv.copy()
                        _D = float(base_att) or 1.0                      # card denominator (30D attempts)
                        _sum_pre = float(pd.to_numeric(_sev.get("pre_succ"), errors="coerce")
                                         .fillna(0.0).sum()) or 1.0
                        _cur_scale = float(base_succ) / _sum_pre         # modelled baseline → card base_succ
                        # Validate isolates the BIN-specific overrides: the eval-frame PRE is already
                        # the authoritative baseline (catch-all cells = live actuals, pre == post).
                        # Rescaling to the raw actual base_succ re-introduces a scale factor that
                        # would drift those 0-movement profiles, so keep PRE unscaled here — this also
                        # matches the SR card baseline, which is overridden to Σ pre_succ / base_att.
                        if ss.get("variations_engine") == "validate":
                            _cur_scale = 1.0

                        # [FN-366]
                        def _sr_bridge(_grp_col, _other):
                            _g = _sev.groupby(_grp_col, as_index=False).agg(
                                pre_succ=("pre_succ", "sum"), post_succ=("post_succ", "sum"))
                            _g["pre_rate"] = 100.0 * _g["pre_succ"] * _cur_scale / _D
                            _g["post_rate"] = 100.0 * _g["post_succ"] / _D
                            return _bridge_items(_g, _grp_col, "pre_rate", "post_rate", _br_sort,
                                                 _other, _max_n=None)

                        _vsbi = _sr_bridge("_vmid", "Other vampMids")
                        if _vsbi is not None:
                            (_vmsr_slot or st).plotly_chart(
                                _rev_bridge_waterfall(*_vsbi, money=False, pct=True),
                                use_container_width=True)
                        # by RPGT (title-cased label)
                        _sev["_rl"] = _sev["rpgt"].astype(str).str.strip().str.title()
                        _rsbi = _sr_bridge("_rl", "Other RPGTs")
                        if _rsbi is not None:
                            (_rpgtsr_slot or st).plotly_chart(
                                _rev_bridge_waterfall(*_rsbi, money=False, pct=True, height=392),
                                use_container_width=True)
                    except Exception as _sbe:  # noqa: BLE001
                        (_vmsr_slot or st).caption(
                            f"Approval bridges unavailable: {type(_sbe).__name__}: {_sbe}")

                _gopts = sorted(_evv["_vmid"].dropna().astype(str).unique().tolist())
                # Mid Detail filter + treemap + table + bridge live in a st.fragment, so changing the
                # Mid / Top-banks controls reruns ONLY this section (no full-page whiteout).
                # [FN-367]
                def _md_detail_fragment():
                    if _gopts:
                        _gf1, _gf2, _gf3 = st.columns([1, 1, 4])   # picker + top-N banks control
                        _gsel = _gf1.selectbox("Mid", _gopts, key="tab_imp_vm_sel")
                        _topn = int(_gf2.number_input("Top banks", min_value=5, max_value=200, value=30,
                                                      step=5, key="tab_imp_tm_topn",
                                                      help="Treemap shows the N banks with the most raw 30D attempts."))
                        # ---- Bank treemap (top of tab, just below the filter; FULL-row width) ----
                        # finviz-style: box size = raw 30D attempts, colour = this MID's engine score
                        # vs the TOP MID at each bank (green = this MID is the best choice there).
                        _bank_tm = None
                        try:
                            _agg_tm = ss.get("agg_sr")
                            if (HAS_PLOTLY and _agg_tm is not None and not _agg_tm.empty
                                    and {"bin", "gateway", "attempts", "success_rate"}.issubset(_agg_tm.columns)):
                                _b2b_tm = ss.get("bin_to_bank", {})
                                _tm = _agg_tm.copy()
                                _tm["_vm"] = _tm["gateway"].astype(str).str.strip().str.lower().map(_f2v_bi).astype(str)
                                _tm["parent"] = _map_to_bank(_tm["bin"], _b2b_tm).astype(str).str.upper()
                                _tm["attempts"] = pd.to_numeric(_tm["attempts"], errors="coerce").fillna(0.0)
                                _tm["success_rate"] = pd.to_numeric(_tm["success_rate"], errors="coerce").fillna(0.0)
                                _tm["_wsr"] = _tm["success_rate"] * _tm["attempts"]
                                _gmid = _tm.groupby(["parent", "_vm"], as_index=False).agg(att=("attempts", "sum"), wsr=("_wsr", "sum"))
                                _gmid["score"] = np.where(_gmid["att"] > 0, _gmid["wsr"] / _gmid["att"], np.nan)
                                _selg = _gmid[_gmid["_vm"] == _gsel].set_index("parent")
                                _topg = _gmid.groupby("parent")["score"].max()
                                # vampMid holding the top engine score at each bank (for the tooltip)
                                _gvalid = _gmid.dropna(subset=["score"])
                                _top_rows = (_gvalid.loc[_gvalid.groupby("parent")["score"].idxmax()]
                                             .set_index("parent")) if not _gvalid.empty else pd.DataFrame()
                                _top_mid = (_top_rows["_vm"] if "_vm" in getattr(_top_rows, "columns", [])
                                            else pd.Series(dtype=object))
                                _all_banks = [b for b in _selg.index if float(_selg.loc[b, "att"]) > 0]
                                # There can be thousands of banks/BINs → an unreadable single green
                                # blob. Show only the TOP-N by raw 30D attempts (finviz style: the
                                # biggest banks get the biggest tiles).
                                _all_banks.sort(key=lambda b: float(_selg.loc[b, "att"]), reverse=True)
                                _banks = _all_banks[:_topn]
                                if _banks:
                                    _this = np.array([float(_selg.loc[b, "score"]) for b in _banks])
                                    _att = np.array([float(_selg.loc[b, "att"]) for b in _banks])
                                    _top = np.array([float(_topg.loc[b]) for b in _banks])
                                    _topnm = [str(_top_mid.get(b, _gsel)) for b in _banks]
                                    _topatt = [float(_top_rows.loc[b, "att"]) if b in getattr(_top_rows, "index", [])
                                               else 0.0 for b in _banks]
                                    _gap = (_this - _top) * 100.0                    # ≤ 0 (top MID is the max)
                                    _M = float(np.nanmax(np.abs(_gap))) if np.isfinite(_gap).any() else 1.0
                                    _M = _M if (np.isfinite(_M) and _M > 1e-9) else 1.0
                                    # tile size = raw 30D attempts; colour = how THIS MID's engine score
                                    # compares to the BEST MID at that bank (green = at/near best, red =
                                    # well below). px builds the tree; per-node arrays are aligned to px's
                                    # own node list so the hidden root gets blank text.
                                    # Rank category (best…terrible) for THIS MID at each bank — computed
                                    # BEFORE the tree so it can drive the SUBTREE grouping. Rank 1 = best,
                                    # rank N = terrible; the middle splits by percentile (top quartile =
                                    # really good, bottom quartile = bad).
                                    _CAT_COLORS = {"best": "#1B9E4B", "really good": "#86C34A",
                                                   "average": "#F4C430", "bad": "#EF7D22",
                                                   "terrible": "#E63748"}
                                    _gm_v = _gmid.dropna(subset=["score"])
                                    _gm_v = _gm_v[_gm_v["parent"].isin(_banks)]
                                    _score_lists = _gm_v.groupby("parent")["score"].apply(lambda s: s.to_numpy())
                                    _cat_map, _rank_map = {}, {}
                                    for b, ts in zip(_banks, _this):
                                        _arr = _score_lists.get(b, np.array([float(ts)], dtype=float))
                                        _n = int(_arr.size)
                                        _r = int(np.sum(_arr > float(ts) + 1e-12)) + 1     # 1 = best
                                        _rank_map[b] = (_r, _n)
                                        if _n <= 1 or _r == 1:
                                            _cat_map[b] = "best"
                                        elif _r == _n:
                                            _cat_map[b] = "terrible"
                                        else:
                                            _pct = (_r - 1) / (_n - 1)                     # 0 best … 1 worst
                                            _cat_map[b] = ("really good" if _pct <= 0.25
                                                           else "bad" if _pct >= 0.75 else "average")
                                    # word-wrap long bank names onto multiple rows so they don't overflow
                                    # [FN-368]
                                    def _wrap(_s, _w=16):
                                        _lines, _cur = [], ""
                                        for _wd in str(_s).split():
                                            if _cur and len(_cur) + 1 + len(_wd) > _w:
                                                _lines.append(_cur)
                                                _cur = _wd
                                            else:
                                                _cur = (_cur + " " + _wd) if _cur else _wd
                                        if _cur:
                                            _lines.append(_cur)
                                        return "<br>".join(_lines)
                                    # Tree grain: Banks -> Category -> Bank, so each tile sits inside its rank
                                    # category's subtree. Tile size = raw 30D attempts.
                                    _df_tm = pd.DataFrame({
                                        "Category": [_cat_map.get(b, "average") for b in _banks],
                                        "Bank": list(_banks),
                                        "Attempts": [float(a) for a in _att],
                                    })
                                    _bank_tm = px.treemap(
                                        _df_tm, path=[px.Constant("Banks"), "Category", "Bank"],
                                        values="Attempts")
                                    _lab = list(_bank_tm.data[0].labels)
                                    # Tile text = bank name + this-MID vs top score (category is the subtree
                                    # header, not the tile). Tooltip = bank, engine score, top MID + its score.
                                    _txt_map = {b: "<b>%s<br>%.2f%% vs %.2f%%</b>" % (_wrap(b), t * 100, tp * 100)
                                                for b, t, tp in zip(_banks, _this, _top)}
                                    _hov_map = {}
                                    for b, t, tp, nm, a, ta in zip(_banks, _this, _top, _topnm, _att, _topatt):
                                        _rr, _nn = _rank_map.get(b, (0, 0))
                                        _hov_map[b] = (
                                            "Bank: %s"
                                            "<br>Engine score: %.2f%%  ·  rank %d of %d"
                                            "<br>This MID raw 30D attempts: %s"
                                            "<br>Top MID at bank: %s (%.2f%%)"
                                            "<br>Top MID raw 30D attempts: %s"
                                            % (b, t * 100, _rr, _nn, format(a, ",.0f"),
                                               str(nm), tp * 100, format(ta, ",.0f")))
                                    # Per-NODE arrays aligned to px's node list (root + category nodes + bank
                                    # leaves). Category nodes show their name as the header and take the
                                    # category colour; bank leaves inherit it; the root "Banks" is dark ink.
                                    _node_text, _node_hov, _node_colors = [], [], []
                                    for _l in _lab:
                                        if _l in _cat_map:                        # bank leaf
                                            _node_text.append(_txt_map.get(_l, ""))
                                            _node_hov.append(_hov_map.get(_l, ""))
                                            _node_colors.append(_CAT_COLORS.get(_cat_map[_l], "#808080"))
                                        elif _l in _CAT_COLORS:                   # category subtree node
                                            _node_text.append("<b>%s</b>" % str(_l).upper())
                                            _node_hov.append("<b>%s</b>" % str(_l).upper())
                                            _node_colors.append(_CAT_COLORS.get(_l, "#808080"))
                                        else:                                     # root "Banks"
                                            _node_text.append("")
                                            _node_hov.append("")
                                            _node_colors.append("#0B1F3A")
                                    _bank_tm.update_traces(
                                        text=_node_text, texttemplate="%{text}", textposition="middle center",
                                        insidetextfont=dict(color="white"),
                                        marker=dict(colors=_node_colors, line=dict(width=1, color="#0B1F3A"),
                                                    coloraxis=None),
                                        hovertext=_node_hov,
                                        hovertemplate="%{hovertext}<extra></extra>")
                                    _bank_tm.update_layout(margin=dict(t=4, l=2, r=2, b=2), height=945,
                                                           coloraxis_showscale=False,       # no colour scale
                                                           paper_bgcolor="#0B1F3A",         # dark ink behind tiles
                                                           plot_bgcolor="#0B1F3A")
                        except Exception:  # noqa: BLE001
                            _bank_tm = None
                        if _bank_tm is not None:
                            st.plotly_chart(_bank_tm, use_container_width=True)
                        else:
                            st.caption("Bank treemap unavailable for this vampMid.")
                        # FULL per-vampMid per-bank agg (drives BOTH the sortable table and the bridge's
                        # Current/Proposed totals + 'Other banks' roll-up).
                        _gfull = _evv[_evv["_vmid"].astype(str) == _gsel].groupby("bin", as_index=False).agg(
                            pre_vol=("pre_vol", "sum"), post_vol=("post_vol", "sum"),
                            vol_delta=("vol_delta", "sum"), rev_delta=("rev_delta", "sum"),
                            pre_rev=("pre_rev", "sum"), post_rev=("post_rev", "sum"),
                            pre_share=("baseline_share", "mean"), post_share=("share", "mean"))
                        _gfull["share_delta_pp"] = (_gfull["post_share"] - _gfull["pre_share"]) * 100.0
                        _gt = _gfull.sort_values("rev_delta", ascending=False)
                        if _gt.empty:
                            st.info("No banks for this vampMid.")
                        else:
                            # Sortable table (click any header). green↑/red↓ shading on 30D $ Impact.
                            _disp = _gt[["bin", "rev_delta", "share_delta_pp", "vol_delta",
                                         "pre_vol", "post_vol"]].copy()
                            _disp.columns = ["Bank", "30D $ Impact", "Δ Share (pp)",
                                             "Δ Volume (txns)", "Pre Volume", "Post Volume"]
                            _gmax = float(np.nanmax(np.abs(_disp["30D $ Impact"].to_numpy(dtype=float)))) if len(_disp) else 1.0
                            _gmax = _gmax if _gmax > 1e-9 else 1.0

                            # [FN-369]
                            def _impact_bg(_col):
                                _out = []
                                for _v in _col:
                                    _f = max(-1.0, min(1.0, float(_v) / _gmax))
                                    _rgba = ("rgba(34,195,107,%.3f)" % (0.75 * _f) if _f >= 0
                                             else "rgba(230,55,72,%.3f)" % (0.75 * abs(_f)))
                                    _out.append("background-color: %s; color:#000" % _rgba)
                                return _out
                            _fmt_map = {"30D $ Impact": "${:,.0f}", "Δ Share (pp)": "{:+.2f}",
                                        "Δ Volume (txns)": "{:+,.0f}", "Pre Volume": "{:,.0f}",
                                        "Post Volume": "{:,.0f}"}
                            # Per-vampMid bridge (uses the shared metric + sort controls above;
                            # read from session_state so it's independent of local scope).
                            _gb_money = (ss.get("tab_imp_br_metric", "Revenue") == "Revenue")
                            _gb_sort = "Top absolute movers"   # bank bridge always shows the biggest movers
                            _gbpre, _gbpost = ("pre_rev", "post_rev") if _gb_money else ("pre_vol", "post_vol")
                            _vm_wf = None
                            _gbi = _bridge_items(_gfull, "bin", _gbpre, _gbpost, _gb_sort, "Other banks")
                            if _gbi is not None:
                                _vm_wf = _rev_bridge_waterfall(*_gbi, money=_gb_money)
                            # Sortable table beside the bridge (the bank treemap is at the TOP of the tab).
                            _btc, _bwc = st.columns([1, 1])
                            try:
                                _btc.dataframe(_disp.style.format(_fmt_map).apply(_impact_bg, subset=["30D $ Impact"]),
                                               use_container_width=True, hide_index=True, height=560)
                            except Exception:  # jinja2/Styler unavailable → plain sortable table
                                _btc.dataframe(_disp, use_container_width=True, hide_index=True, height=560)
                            if _vm_wf is not None:
                                _bwc.plotly_chart(_vm_wf, use_container_width=True)
                    else:
                        st.info("No vampMids in this variation's split.")

                with _md_bank_slot:
                    (st.fragment(_md_detail_fragment) if hasattr(st, "fragment") else _md_detail_fragment)()

            # -------------- FILTERABLE ENGINE WORKINGS EXPANDER & DEBUGGER --------------
            # --- Banks per vampMid: how widely each MID is spread across banks ---
            _bpm_split = ss.get("split")
            if (HAS_PLOTLY and _bpm_split is not None and not getattr(_bpm_split, "empty", True)
                    and {"gateway", "bin", "share"}.issubset(getattr(_bpm_split, "columns", []))):
                with _t_engwork.container():
                    try:
                        import plotly.graph_objects as _gbp
                        _f2v = (ss.get("wallet_ctx") or {}).get("fid2vamp") or {}
                        _b2b = ss.get("bin_to_bank", {})
                        _d = _bpm_split.copy()
                        _d["_shp"] = pd.to_numeric(_d["share"], errors="coerce").fillna(0.0)
                        _d["_shb"] = (pd.to_numeric(_d["baseline_share"], errors="coerce").fillna(0.0)
                                      if "baseline_share" in _d.columns else 0.0)
                        _gwl = _d["gateway"].astype(str).str.strip().str.lower()
                        _d["_vm"] = (_gwl.map(_f2v).fillna(_d["gateway"].astype(str)) if _f2v
                                     else _d["gateway"].astype(str))
                        # 'bin' holds the BIN; map it to its parent bank for the distinct-banks view.
                        _d["_pb"] = _map_to_bank(_d["bin"], _b2b).astype(str)

                        # [FN-370]
                        def _spread_fig(_col, _unit):
                            # distinct count of _col per vampMid: Current (baseline_share>0) vs
                            # Proposed (share>0), grouped bars.
                            _pro = _d[_d["_shp"] > 1e-9].groupby("_vm")[_col].nunique()
                            _pre = _d[_d["_shb"] > 1e-9].groupby("_vm")[_col].nunique()
                            _idx = _pro.index.union(_pre.index)
                            if len(_idx) == 0:
                                return None
                            _pro = _pro.reindex(_idx, fill_value=0)
                            _pre = _pre.reindex(_idx, fill_value=0)
                            _order = _pro.sort_values(ascending=True).index[-30:]   # most-spread at top
                            _pro = _pro.reindex(_order); _pre = _pre.reindex(_order)
                            _ylab = [str(m).replace("_", " ").strip()[:40] for m in _order]
                            _f = _gbp.Figure()
                            _f.add_bar(x=_pre.to_numpy(), y=_ylab, orientation="h", name="Current",
                                       marker=dict(color="#8A93A6"),
                                       text=[f"{int(v):,}" for v in _pre.to_numpy()], textposition="outside",
                                       textfont=dict(color="#0B1F3A", size=9), cliponaxis=False,
                                       hovertemplate="%{y}<br>current %{x} " + _unit + "<extra></extra>")
                            _f.add_bar(x=_pro.to_numpy(), y=_ylab, orientation="h", name="Proposed",
                                       marker=dict(color="#e63748"),
                                       text=[f"{int(v):,}" for v in _pro.to_numpy()], textposition="outside",
                                       textfont=dict(color="#0B1F3A", size=9), cliponaxis=False,
                                       hovertemplate="%{y}<br>proposed %{x} " + _unit + "<extra></extra>")
                            # Same styling as the Risk Impact bar charts: light-grey value gridlines,
                            # top-centred horizontal legend, transparent bg, 9px ink text.
                            _f.update_layout(
                                barmode="group", bargap=0.08,
                                height=max(280, 26 * len(_order) + 70),
                                margin=dict(l=10, r=35, t=22, b=4),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#0B1F3A", family="inherit", size=9),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                            font=dict(color="#0B1F3A", size=9), title_text=None),
                                xaxis=dict(title="", showgrid=True, gridcolor="lightgrey",
                                           tickfont=dict(color="#0B1F3A", size=9)),
                                yaxis=dict(showgrid=False, automargin=True,
                                           tickfont=dict(color="#0B1F3A", size=9)))
                            return _f

                        _fig_bin = _spread_fig("bin", "BIN(s)")
                        _fig_bank = _spread_fig("_pb", "bank(s)")
                        if _fig_bin is None and _fig_bank is None:
                            st.info("No routed share in the selected split.")
                        else:
                            _sc1, _sc2 = st.columns(2)
                            _sc1.markdown("**Distinct BINs (current vs proposed)**")
                            if _fig_bin is not None:
                                _sc1.plotly_chart(_fig_bin, use_container_width=True)
                            _sc2.markdown("**Distinct banks (current vs proposed)**")
                            if _fig_bank is not None:
                                _sc2.plotly_chart(_fig_bank, use_container_width=True)
                    except Exception as _e:  # noqa: BLE001
                        st.info(f"Banks-per-vampMid chart unavailable: {type(_e).__name__}: {_e}")

            # --- GA workings: convergence / step-size / violation / genome / population ---
            _hist_rev = ss.get("ga_hist_rev"); _hist_safe = ss.get("ga_hist_safe")
            if HAS_PLOTLY and (_hist_rev or _hist_safe):
                import pandas as _pd
                import plotly.graph_objects as _gof

                # [FN-371]
                def _conv_df(h):
                    if not h:
                        return None
                    _names = ["gen", "best", "gen_best", "gen_mean", "sigma", "viol", "eps", "cands"]
                    # History rows can be ragged (main run vs re-projection rounds record different
                    # widths). Pad short rows to the widest width with None (rather than dropping
                    # columns) so the DataFrame never trips on "N columns passed, data had M" and the
                    # convergence / σ / violation charts keep every field a row does have.
                    try:
                        _w = max(len(_r) for _r in h)
                    except TypeError:
                        return None
                    _w = min(int(_w), len(_names))
                    if _w <= 0:
                        return None
                    _rows = [tuple(_r[:_w]) + (None,) * (_w - min(len(_r), _w)) for _r in h]
                    return _pd.DataFrame(_rows, columns=_names[:_w])

                # Effort scale for the candidate-based charts below: the convergence trace is ONE
                # (winning) seed, but N seeds ran in PARALLEL, so `_seed_scale` maps the per-seed
                # candidate x-axis up to the RUN TOTAL (e.g. 51k → 409,600).
                _tot_cands = int(ss.get("last_ga_cands", 0) or 0)
                _d2s = _conv_df(_hist_safe)
                _seed_scale = 1.0
                if _d2s is not None and len(_d2s) and "cands" in _d2s.columns:
                    _perseed = float(_d2s["cands"].max())
                    _seed_scale = (_tot_cands / _perseed) if (_tot_cands > 0 and _perseed > 0) else 1.0

                # NEW: engine score vs. candidate splits EVALUATED (the true search-effort x-axis),
                # plus the marginal gain per 10k candidates — shows where the search plateaus, i.e.
                # whether more budget (seeds/restarts/generations) would still be buying improvement.
                # (Removed: "📈 Engine score vs candidate splits evaluated (winning seed)" chart
                #  + its marginal-gain sub-chart, per request.)

                _t_engwork.markdown("##### 📈 GA scoring over time (convergence)")
                with _t_engwork.container():
                    st.caption("How the search's score improved each generation, for each end of the "
                               "dial. Higher is better; 'best so far' is the best plan yet, 'population "
                               "mean' the average candidate.")

                    # [FN-372]
                    def _mk_conv(h, title):
                        d = _conv_df(h)
                        if d is None:
                            return None
                        f = _gof.Figure()
                        f.add_scatter(x=d["gen"], y=d["best"], mode="lines", name="best so far",
                                      line=dict(color="#1D9E75"))
                        f.add_scatter(x=d["gen"], y=d["gen_mean"], mode="lines", name="population mean",
                                      line=dict(color="#B4B2A9", dash="dot"))
                        f.update_layout(height=300, margin=dict(l=10, r=10, t=36, b=10), title=title,
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="#0B1F3A", family="inherit", size=9),
                                        xaxis=dict(title="", showgrid=False,
                                                   tickfont=dict(color="#0B1F3A", size=9)),
                                        yaxis=dict(title="", showgrid=False,
                                                   tickfont=dict(color="#0B1F3A", size=9)),
                                        legend=dict(orientation="h", y=1.16, font=dict(color="#0B1F3A", size=9)))
                        return f
                    _cvc1, _cvc2 = st.columns(2)
                    _figr = _mk_conv(_hist_rev, "Revenue-max endpoint (dial 100)")
                    _figs = _mk_conv(_hist_safe, "Risk-min endpoint (dial 0)")
                    if _figr is not None:
                        _cvc1.plotly_chart(_figr, use_container_width=True)
                    if _figs is not None:
                        _cvc2.plotly_chart(_figs, use_container_width=True)

                    _fsig = _gof.Figure(); _has_sig = False
                    for _h, _nm, _dash, _col in ((_hist_rev, "dial 100", "solid", "#378ADD"),
                                                 (_hist_safe, "dial 0", "dot", "#D4537E")):
                        _d = _conv_df(_h)
                        if _d is not None and "sigma" in _d:
                            _has_sig = True
                            _fsig.add_scatter(x=_d["gen"], y=_d["sigma"], mode="lines", name=_nm,
                                              line=dict(color=_col, dash=_dash))
                    # Step size (σ) | Winning tilt genome — two per row.
                    _gwr1, _gwr2 = st.columns(2)
                    if _has_sig:
                        _gwr1.markdown("###### Step size (σ) — big steps early, refines late, jumps on restart")
                        _fsig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                            font=dict(color="#0B1F3A", family="inherit", size=9),
                                            xaxis=dict(title="", showgrid=False,
                                                       tickfont=dict(color="#0B1F3A", size=9)),
                                            yaxis=dict(title="", type="log", showgrid=False,
                                                       tickfont=dict(color="#0B1F3A", size=9)),
                                            legend=dict(orientation="h", y=1.15, font=dict(color="#0B1F3A", size=9)))
                        _gwr1.plotly_chart(_fsig, use_container_width=True)
                    _genome_g = ss.get("ga_genome"); _mlabels_g = ss.get("ga_mid_labels")
                    if _genome_g is not None and _mlabels_g:
                        _gg = np.asarray(_genome_g, float); _Mg = len(_mlabels_g)
                        if _gg.size >= 3 * _Mg and _Mg > 0:
                            _gwr2.markdown("###### 🧬 Winning tilt genome (per vampMid) — θr risk-tilt, "
                                           "θq revenue-tilt, gain")
                            _matg = np.vstack([_gg[:_Mg], _gg[_Mg:2 * _Mg], _gg[2 * _Mg:3 * _Mg]])
                            _fhm = _gof.Figure(_gof.Heatmap(
                                z=_matg, x=[str(m)[:22] for m in _mlabels_g],
                                y=["θr risk-tilt", "θq revenue-tilt", "gain"],
                                colorscale=[[0.0, "#D85A30"], [0.5, "#EEF3FB"], [1.0, "#1D9E75"]],
                                zmid=0.0, colorbar=dict(
                                    title=dict(text="value", font=dict(color="#0B1F3A")),
                                    tickfont=dict(color="#0B1F3A"))))
                            _fhm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                               font=dict(color="#0B1F3A", family="inherit", size=9),
                                               xaxis=dict(tickangle=-45, tickfont=dict(color="#0B1F3A", size=9)),
                                               yaxis=dict(tickfont=dict(color="#0B1F3A", size=9)))
                            _gwr2.plotly_chart(_fhm, use_container_width=True)

                    # [FN-373]
                    def _mk_viol(h, title):
                        d = _conv_df(h)
                        if d is None or "viol" not in d:
                            return None
                        f = _gof.Figure()
                        f.add_scatter(x=d["gen"], y=d["viol"], mode="lines", name="violation",
                                      line=dict(color="#D85A30"), fill="tozeroy",
                                      fillcolor="rgba(216,90,48,0.12)")
                        if "eps" in d:
                            f.add_scatter(x=d["gen"], y=d["eps"], mode="lines", name="ε tolerance",
                                          line=dict(color="#888780", dash="dot"))
                        f.update_layout(height=280, margin=dict(l=10, r=10, t=36, b=10), title=title,
                                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                        font=dict(color="#0B1F3A", family="inherit", size=9),
                                        xaxis=dict(title="", showgrid=False,
                                                   tickfont=dict(color="#0B1F3A", size=9)),
                                        yaxis=dict(title="", showgrid=False,
                                                   tickfont=dict(color="#0B1F3A", size=9)),
                                        legend=dict(orientation="h", y=1.16, font=dict(color="#0B1F3A", size=9)))
                        return f
                    _fvr = _mk_viol(_hist_rev, "Revenue-max endpoint (dial 100)")
                    _fvs = _mk_viol(_hist_safe, "Risk-min endpoint (dial 0)")
                    if _fvr is not None or _fvs is not None:
                        st.markdown("###### Violation → feasibility (breach falls to 0 as ε tightens)")
                        _vvc1, _vvc2 = st.columns(2)
                        if _fvr is not None:
                            _vvc1.plotly_chart(_fvr, use_container_width=True)
                        if _fvs is not None:
                            _vvc2.plotly_chart(_fvs, use_container_width=True)

                # (Winning-tilt genome heatmap moved up to sit 2-up beside the Step-size chart.)
                # (Removed: "⚖️ Feasibility-first population (final generation)" chart, per request.)

            # --- Time-decay weighting curve (how older attempts count less in the SR estimate) ---
            if HAS_PLOTLY:
                _t_engwork.markdown("##### ⏳ Time-decay weighting (how older data counts less)")
                with _t_engwork.container():
                    _dc_on = bool(ss.get("apply_decay", ss.get("apply_decay_cb", True)))
                    _hl = float(ss.get("decay_half_inp", 15) or 15)
                    import numpy as _np2
                    import plotly.graph_objects as _gdc
                    # Build the x-axis from the ACTUAL attempts dates (window ending at attempts_end),
                    # so weight is shown per calendar day rather than an abstract "age in days".
                    _adf_dc = ss.get("adf")
                    _dates = None
                    try:
                        if _adf_dc is not None and not getattr(_adf_dc, "empty", True):
                            _dcol = ("date" if "date" in _adf_dc.columns
                                     else ("Date" if "Date" in _adf_dc.columns else None))
                            if _dcol:
                                _dser = pd.to_datetime(_adf_dc[_dcol], errors="coerce").dropna()
                                if not _dser.empty:
                                    _end = _dser.max().normalize()
                                    _start = _dser.min().normalize()
                                    _dates = pd.date_range(_start, _end, freq="D")
                    except Exception:  # noqa: BLE001
                        _dates = None
                    if _dates is None or len(_dates) < 2:
                        # Fallback: a synthetic window ending today if no dated attempts are loaded.
                        _end = pd.Timestamp.today().normalize()
                        _dates = pd.date_range(_end - pd.Timedelta(days=int(max(90.0, _hl * 4.0))), _end, freq="D")
                    _aged = (_dates[-1] - _dates).days.to_numpy().astype(float)
                    _w = _np2.power(0.5, _aged / max(_hl, 1e-9)) if _dc_on else _np2.ones_like(_aged)
                    _fdc = _gdc.Figure()
                    _fdc.add_scatter(x=_dates, y=_w, mode="lines", name="weight",
                                     line=dict(color="#1D9E75", width=2), fill="tozeroy",
                                     fillcolor="rgba(29,158,117,0.12)",
                                     hovertemplate="%{x|%Y-%m-%d}<br>weight %{y:.0%}<extra></extra>")
                    if _dc_on:
                        _hldate = _dates[-1] - pd.Timedelta(days=int(_hl))
                        _fdc.add_shape(type="line", x0=_hldate, x1=_hldate, y0=0.0, y1=0.5,
                                       line=dict(color="#D85A30", dash="dot", width=1))
                        _fdc.add_scatter(x=[_hldate], y=[0.5], mode="markers+text",
                                         marker=dict(color="#D85A30", size=9),
                                         text=[f"half-life {_hl:.0f}d → 50%"], textposition="top left",
                                         textfont=dict(size=9, color="#0B1F3A"), showlegend=False,
                                         hovertemplate="half-life %{x|%Y-%m-%d} → 50%<extra></extra>")
                    _fdc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="#0B1F3A", family="inherit", size=9),
                                       showlegend=False,
                                       xaxis=dict(title="", showgrid=False,
                                                  tickfont=dict(color="#0B1F3A", size=9)),
                                       yaxis=dict(title="", range=[0.0, 1.05], tickformat=".0%", showgrid=False,
                                                  tickfont=dict(color="#0B1F3A", size=9)))
                    st.plotly_chart(_fdc, use_container_width=True)

            _t_engwork.markdown("##### ⚙️ Algorithm Scoring Workings (Pre-Softmax) & Granular BIN Impact")
            with _t_engwork.container():
                st.markdown("<div style='font-size: 0.85rem; color: #0B1F3A; margin-bottom: 1rem;'>This table exposes the granular BIN/Currency level details. You can sort and filter any column by clicking the column headers.</div>", unsafe_allow_html=True)
            
                debug_mode = st.checkbox("🛠️ Toggle Debug Diagnostics (Click if table is missing or empty)")
            
                sr_df = ss["sr"].copy()
                # Collapse BINs into their parent Bank so the debug table is at
                # the Bank x Currency grain (matching the selection & engine).
                _b2b_s = ss.get("bin_to_bank", {})
                if _b2b_s and "bin" in sr_df.columns:
                    sr_df["bin"] = _map_to_bank(sr_df["bin"], _b2b_s).astype(str)

                if sr_df.empty or "bin" not in sr_df.columns:
                    st.info("This diagnostic is only available for an engine-computed split "
                            "(**2 · Routing engine**); a validated split doesn't build the "
                            "granular per-BIN success-rate cache it needs.")
                elif selected_bank != "(All Portfolio)":
                    b_val = selected_bank.split(" - ")[0]
                    c_val = selected_bank.split(" - ")[1].lower()
                
                    if debug_mode:
                        st.write("**Diagnostics:**")
                        st.write(f"- Looking for BIN: `{b_val}` and Currency: `{c_val}`")
                        st.write(f"- Total rows available in Engine Cache: `{len(sr_df)}`")
                        st.write(f"- Sample of unique BINs in cache: `{sr_df['bin'].astype(str).unique()[:10]}`")
                    
                    sr_df = sr_df[(sr_df["bin"].astype(str).str.upper() == b_val.upper()) & (sr_df["currency"].astype(str).str.lower() == c_val)]
                
                    if debug_mode:
                        st.write(f"- Rows found after filtering: `{len(sr_df)}`")
                        if len(sr_df) == 0:
                            st.error("🚨 Filter returned 0 rows! This means the BIN doesn't exist in the memory cache. You MUST click 'Compute split variations' in Tab 2 to rebuild the cache.")
                
                if not sr_df.empty:
                    att_col = "attempts" if "attempts" in sr_df.columns else "raw_attempts"
                    succ_col = "success" if "success" in sr_df.columns else "raw_successes"
                    rate_col = "success_rate" if "success_rate" in sr_df.columns else "smoothed_rate"
                
                    sr_df[att_col] = pd.to_numeric(sr_df[att_col], errors='coerce').fillna(0)
                    sr_df[succ_col] = pd.to_numeric(sr_df[succ_col], errors='coerce').fillna(0)
                    sr_df[rate_col] = pd.to_numeric(sr_df[rate_col], errors='coerce').fillna(0)
                
                    sr_df["weighted_sr"] = sr_df[rate_col] * sr_df[att_col]
                
                    workings = sr_df.groupby(["bin", "currency", "gateway"]).agg(
                        All_Time_Attempts=(att_col, "sum"),
                        All_Time_Success=(succ_col, "sum"),
                        Weighted_SR=("weighted_sr", "sum")
                    ).reset_index()
                
                    # Engine Score is calculated ONLY at the ALL_RPGTS level:
                    # it must be the exact aggregated smoothed rate the engine
                    # exponentiates (agg_sr), keyed by (currency, parent bank,
                    # gateway) - NOT a per-RPGT volume-weighted average.
                    agg_sr_lk = ss.get("agg_sr")
                    b2b = ss.get("bin_to_bank", {})
                    # [FN-374]
                    def _parent(b):
                        return str(b2b.get(b, b2b.get(str(b), b))).strip().lower()
                    _kappa = float(ss.get("shrink_kappa", 300.0))
                    if agg_sr_lk is not None and not agg_sr_lk.empty:
                        a = agg_sr_lk.copy()
                        a["_cj"] = a["currency"].astype(str).str.strip().str.lower()
                        a["_pj"] = a["bin"].astype(str).str.strip().str.lower()
                        a["_gj"] = a["gateway"].astype(str).str.strip().str.lower()
                        _acols = ["_cj", "_pj", "_gj", "attempts", "success", "prior_rate", "success_rate"]
                        if "kappa" in a.columns:
                            _acols.append("kappa")
                        a = a[_acols].drop_duplicates(["_cj", "_pj", "_gj"])
                        workings["_cj"] = workings["currency"].astype(str).str.strip().str.lower()
                        workings["_pj"] = workings["bin"].map(_parent)
                        workings["_gj"] = workings["gateway"].astype(str).str.strip().str.lower()
                        workings = workings.merge(a, on=["_cj", "_pj", "_gj"], how="left")
                        workings["Engine Score (Smoothed SR)"] = workings["success_rate"].fillna(0.0)
                        # Show All-Time at the SAME parent-bank grain the engine
                        # actually scores on (it pools every BIN under the bank),
                        # so every column reconciles with the Engine Score:
                        #   Engine Score = (All-Time Success + κ*prior) / (All-Time Attempts + κ)
                        # κ is the fixed value, or the per-Bank×Currency Empirical-Bayes estimate.
                        workings["All_Time_Attempts"] = workings["attempts"].fillna(0.0)
                        workings["All_Time_Success"] = workings["success"].fillna(0.0)
                        workings["Prior SR %"] = workings["prior_rate"].fillna(0.0) * 100
                        if "kappa" in workings.columns:
                            workings["κ used"] = pd.to_numeric(workings["kappa"], errors="coerce").fillna(_kappa)
                        else:
                            workings["κ used"] = _kappa
                        workings["Bayesian Adj Attempts"] = workings["All_Time_Attempts"] + workings["κ used"]
                        workings["Bayesian Adj Success"] = workings["All_Time_Success"] + workings["κ used"] * workings["prior_rate"].fillna(0.0)
                        # UN-decayed parent attempts (same grain) for verifying decay.
                        _raw = ss.get("agg_raw_att")
                        if _raw is not None:
                            _r = _raw.copy()
                            _r["_cj"] = _r["currency"].astype(str).str.strip().str.lower()
                            _r["_pj"] = _r["parent_bank"].astype(str).str.strip().str.lower()
                            _r["_gj"] = _r["gateway"].astype(str).str.strip().str.lower()
                            _r = _r[["_cj", "_pj", "_gj", "attempts"]].rename(columns={"attempts": "All-Time Attempts (raw)"}).drop_duplicates(["_cj", "_pj", "_gj"])
                            workings = workings.merge(_r, on=["_cj", "_pj", "_gj"], how="left")
                            workings["All-Time Attempts (raw)"] = pd.to_numeric(workings.get("All-Time Attempts (raw)", 0), errors="coerce").fillna(0.0)
                        else:
                            workings["All-Time Attempts (raw)"] = 0.0
                        workings = workings.drop(columns=[c for c in ["_cj", "_pj", "_gj", "attempts", "success", "prior_rate", "success_rate", "kappa"] if c in workings.columns])
                    else:
                        # Fallback = the (bank, currency, gateway) grouped mean, looked up PER ROW by
                        # key (reindex on workings' own keys) rather than assigned by position — a
                        # key-sorted .values array would otherwise attach scores to the wrong cell.
                        _grp_fb = sr_df.groupby(["bin", "currency", "gateway"])[rate_col].mean()
                        _grp_fb_aligned = _grp_fb.reindex(
                            pd.MultiIndex.from_frame(workings[["bin", "currency", "gateway"]])).to_numpy()
                        workings["Engine Score (Smoothed SR)"] = np.where(workings["All_Time_Attempts"] > 0, workings["Weighted_SR"] / workings["All_Time_Attempts"], _grp_fb_aligned)
                        for _c in ["Prior SR %", "Bayesian Adj Attempts", "Bayesian Adj Success", "All-Time Attempts (raw)", "κ used"]:
                            workings[_c] = 0.0
                    workings["All-Time Raw SR"] = np.where(workings["All_Time_Attempts"] > 0, workings["All_Time_Success"] / workings["All_Time_Attempts"], 0)
                
                    workings["bin_join"] = workings["bin"].astype(str).str.strip().str.lower()
                    workings["currency_join"] = workings["currency"].astype(str).str.strip().str.lower()
                    workings["gateway_join"] = workings["gateway"].astype(str).str.strip().str.lower()
                
                    if not b_df.empty:
                        _agg_kw = dict(
                            Gateway=("gateway", "first"),
                            curr_vol=("curr_vol", "sum"),
                            Expected_Attempts=("prop_vol", "sum"),
                            Expected_Success=("exp_succ", "sum"),
                            Expected_Rev=("exp_rev", "sum"))
                        # Baseline revenue at the SAME per-RPGT ticket as Post (eval frame pre_rev,
                        # summed over the gateway's RPGT profiles) → "Pre Revenue (Adj)" now reconciles
                        # with the Financial Impact tables and with Post Revenue at the RPGT grain.
                        if "pre_rev" in b_df.columns:
                            _agg_kw["Pre_Rev"] = ("pre_rev", "sum")
                        # Baseline (pre) attempts/successes at the 30D-attempts basis, for the
                        # success-rate reconciliation chain (Baseline Success = Baseline Attempts × SR).
                        if "pre_succ" in b_df.columns:
                            _agg_kw["Baseline_Success"] = ("pre_succ", "sum")
                        if "pre_att" in b_df.columns:
                            _agg_kw["Baseline_Attempts"] = ("pre_att", "sum")
                        gw_sh_det = b_df.groupby(["bin_join", "currency_join", "gateway_join"]).agg(
                            **_agg_kw).reset_index()
                    else:
                        gw_sh_det = pd.DataFrame(columns=["bin_join", "currency_join", "gateway_join", "Gateway", "curr_vol", "Expected_Attempts", "Expected_Success", "Expected_Rev", "Pre_Rev", "Baseline_Success", "Baseline_Attempts"])
                
                    if not plot_adf_sel.empty:
                        raw_gw_det = plot_adf_sel.groupby(["bin", "currency", "gateway"]).agg(
                            Raw_Gateway=("gateway", "first"),
                            raw_att=("attempts", "sum"),
                            raw_succ=("success", "sum"),
                            raw_amount=("succ_amount", "sum")
                        ).reset_index().rename(columns={"bin": "bin_join", "currency": "currency_join", "gateway": "gateway_join"})
                        raw_gw_det["bin_join"] = raw_gw_det["bin_join"].astype(str).str.strip().str.lower()
                        raw_gw_det["currency_join"] = raw_gw_det["currency_join"].astype(str).str.strip().str.lower()
                        raw_gw_det["gateway_join"] = raw_gw_det["gateway_join"].astype(str).str.strip().str.lower()
                    else:
                        raw_gw_det = pd.DataFrame(columns=["bin_join", "currency_join", "gateway_join", "Raw_Gateway", "raw_att", "raw_succ", "raw_amount"])
                
                    raw_gw_det["Raw 30D Success Rate"] = np.where(raw_gw_det["raw_att"] > 0, raw_gw_det["raw_succ"] / raw_gw_det["raw_att"], 0)
                
                    gw_sh_det = gw_sh_det.merge(raw_gw_det, on=["bin_join", "currency_join", "gateway_join"], how="outer")
                
                    if "Gateway" in gw_sh_det.columns and "Raw_Gateway" in gw_sh_det.columns:
                        gw_sh_det["Gateway"] = gw_sh_det["Gateway"].fillna(gw_sh_det["Raw_Gateway"]).fillna(gw_sh_det["gateway_join"])
                    elif "Gateway" not in gw_sh_det.columns:
                        gw_sh_det["Gateway"] = gw_sh_det["gateway_join"]
                
                    gw_sh_det["BIN"] = gw_sh_det["bin_join"].str.upper()
                    gw_sh_det["Currency"] = gw_sh_det["currency_join"].str.upper()
                
                    for safe_col in ["Expected_Attempts", "Expected_Success", "Expected_Rev", "raw_att", "raw_succ", "raw_amount", "curr_vol", "Raw 30D Success Rate"]:
                        gw_sh_det[safe_col] = gw_sh_det.get(safe_col, 0).fillna(0)
                
                    gw_sh_det["Expected Revenue Impact"] = gw_sh_det["Expected_Rev"] - gw_sh_det["raw_amount"]
                
                    t_raw_g = gw_sh_det.groupby(["BIN", "Currency"])["raw_att"].transform("sum")
                    t_prop_g = gw_sh_det.groupby(["BIN", "Currency"])["Expected_Attempts"].transform("sum")
                
                    gw_sh_det["Current Share"] = np.where(t_raw_g > 0, (gw_sh_det["raw_att"] / t_raw_g), 0)
                    gw_sh_det["Proposed Share"] = np.where(t_prop_g > 0, (gw_sh_det["Expected_Attempts"] / t_prop_g), 0)
                    gw_sh_det["Shift (pp)"] = (gw_sh_det["Proposed Share"] - gw_sh_det["Current Share"]) * 100
                
                    workings_full = gw_sh_det.merge(workings, on=["bin_join", "currency_join", "gateway_join"], how="left")
                
                    workings_full["Gateway"] = workings_full["Gateway"].fillna(workings_full["gateway_join"])
                    workings_full["BIN"] = workings_full["BIN"].fillna(workings_full["bin_join"].str.upper())
                    workings_full["Currency"] = workings_full["Currency"].fillna(workings_full["currency_join"].str.upper())
                
                    # 19ez: same guard, same reason — these come from optional enrichment blocks
                    # above, and four of the reads below are the BARE form with no pd.to_numeric,
                    # so a missing column crashed on `.fillna` rather than reading as zero.
                    # 19fm: "Avg txn value (Bank x Cur)" IS DELIBERATELY NOT IN THIS LIST.
                    # 19ez put it here and that CAUSED a crash. The `_bcv` merge ~20 lines below
                    # brings a column of exactly that name; pandas will not overwrite an existing
                    # column on merge, it SUFFIXES both sides into `..._x` / `..._y`. So ensuring
                    # the column first made the plain name DISAPPEAR at the merge, the next line's
                    # `.get(name, 0)` returned the bare int default, and `pd.to_numeric(0).fillna`
                    # raised "AttributeError: 'int' object has no attribute 'fillna'" — the exact
                    # error ensure_cols exists to prevent, reintroduced by the fix for it.
                    # THE RULE: never ensure a column that a later merge supplies. Nothing between
                    # here and that merge reads it, and the if/else there defines it on both
                    # branches, so it needs no default.
                    ensure_cols(workings_full, (
                        ("All_Time_Attempts", 0.0), ("All-Time Raw SR", 0.0),
                        ("Engine Score (Smoothed SR)", 0.0), ("raw_amount", 0.0),
                        ("raw_succ", 0.0),
                        ("Expected_Rev", 0.0), ("Expected_Attempts", 0.0),
                        ("Expected_Success", 0.0), ("curr_vol", 0.0),
                        ("Baseline_Success", 0.0)))
                    workings_full["All-Time Attempts"] = workings_full.get("All_Time_Attempts", 0).fillna(0)
                    workings_full["All-Time Raw SR"] = workings_full.get("All-Time Raw SR", 0).fillna(0)
                    workings_full["Engine Score (Smoothed SR)"] = workings_full.get("Engine Score (Smoothed SR)", 0).fillna(0)
                    for _bc in ["Prior SR %", "Bayesian Adj Attempts", "Bayesian Adj Success", "All-Time Attempts (raw)", "κ used"]:
                        workings_full[_bc] = workings_full.get(_bc, 0)
                        workings_full[_bc] = pd.to_numeric(workings_full[_bc], errors="coerce").fillna(0)
                    # Raw 30D amount (revenue) and average value per attempt.
                    # Flag cross-border gateways (Engine Score already includes the penalty).
                    _xb = ss.get("xborder_fids", set())
                    workings_full["Cross-border?"] = np.where(
                        workings_full["gateway_join"].astype(str).str.strip().str.lower().isin(_xb),
                        "⚠️ x-border", "")
                    workings_full["Raw 30D Amount"] = pd.to_numeric(workings_full.get("raw_amount", 0), errors="coerce").fillna(0)
                    # Avg value per successful txn at the Bank x Currency level -
                    # the SAME figure that drives every revenue-impact number.
                    _bcv = cache.get("bc_val")
                    if _bcv is not None:
                        # 19fm: DROP FIRST, MERGE SECOND. A merge that brings a column the left
                        # frame already has produces `..._x` / `..._y` and NEITHER is the name the
                        # code reads — a silent rename that surfaces as a scalar-default crash
                        # further down. Dropping makes the merge authoritative and idempotent, so
                        # it survives both the 19ez-style pre-ensure and a re-render on a frame
                        # that already went through this block once.
                        workings_full = workings_full.drop(
                            columns=["Avg txn value (Bank x Cur)"], errors="ignore")
                        workings_full = workings_full.merge(
                            _bcv[["currency_join", "bin_join", "avg_txn_value"]].rename(
                                columns={"avg_txn_value": "Avg txn value (Bank x Cur)"}),
                            on=["currency_join", "bin_join"], how="left")
                        workings_full["Avg txn value (Bank x Cur)"] = pd.to_numeric(
                            workings_full.get("Avg txn value (Bank x Cur)", 0), errors="coerce").fillna(0)
                    else:
                        workings_full["Avg txn value (Bank x Cur)"] = 0.0

                    # Revenue impact, valued at the PER-RPGT ticket so both pre and post track the
                    # RPGT mix and reconcile with the Financial Impact tables (same eval frame).
                    #   Pre Revenue (Adj) = Σ_RPGT (per-RPGT ticket × baseline successes)  [eval pre_rev]
                    #   Post Revenue      = Σ_RPGT (per-RPGT ticket × proposed successes)  [eval exp_rev]
                    #   Expected Revenue Impact = Post Revenue − Pre Revenue (Adj)
                    # Both come from the eval frame's per-RPGT-ticket revenue, so the delta equals the
                    # Impact tab's Δ Revenue. (Falls back to Bank×Cur ticket × raw successes for any
                    # gateway missing from the eval frame, e.g. raw-only rows.)
                    _raw_succ = pd.to_numeric(workings_full.get("raw_succ", 0), errors="coerce").fillna(0)
                    _pre_fallback = workings_full["Avg txn value (Bank x Cur)"] * _raw_succ
                    # Pre Revenue (Adj) on the RAW basis (per user): value the ACTUAL observed 30D
                    # successes at the per-RPGT ticket, per RPGT, summed to the gateway. Baseline share
                    # does NOT enter (unlike the modelled cell_att × baseline_share × gw_sr). Built from
                    # plot_adf_sel (observed successes, has RPGT) × cell_agg's per-RPGT ticket; the full
                    # per-(bank,cur,RPGT,gateway) frame (_raw_rpgt) is reused by the per-RPGT breakdown.
                    _raw_rpgt = None
                    try:
                        _ca_t = cache.get("cell_agg") if isinstance(cache, dict) else None
                        if (_ca_t is not None and not plot_adf_sel.empty
                                and {"rpgt", "currency", "bin", "gateway", "success"}.issubset(plot_adf_sel.columns)
                                and {"rpgt_join", "currency_join", "bin_join", "rpgt_ticket"}.issubset(_ca_t.columns)):
                            _tk = _ca_t[["rpgt_join", "currency_join", "bin_join", "rpgt_ticket"]].copy()
                            if "avg_ticket" in _ca_t.columns:
                                _tk["avg_ticket"] = _ca_t["avg_ticket"].to_numpy()
                            _rs = plot_adf_sel.copy()
                            _rs = _rs.assign(
                                rpgt_join=_rs["rpgt"].astype(str).str.strip().str.lower(),
                                currency_join=_rs["currency"].astype(str).str.strip().str.lower(),
                                # 19ey: a KEYWORD, not a quoted column name — which is why the
                                # 19eo/19er/19ew sweeps could not see it. The groupby two lines
                                # below asks for `bin_join`, so this built a frame that could
                                # never be grouped, the bare `except` below swallowed the
                                # KeyError, `_raw_rpgt` fell to None, and the failure surfaced
                                # 200 lines later as an AttributeError on `.fillna`.
                                bin_join=_rs["bin"].astype(str).str.strip().str.lower(),
                                gateway_join=_rs["gateway"].astype(str).str.strip().str.lower(),
                                _succ=pd.to_numeric(_rs["success"], errors="coerce").fillna(0.0),
                                _att=pd.to_numeric(_rs.get("attempts", 0), errors="coerce").fillna(0.0))
                            _rs = _rs.groupby(["rpgt_join", "currency_join", "bin_join", "gateway_join"],
                                              as_index=False).agg(raw_succ=("_succ", "sum"), raw_att=("_att", "sum"))
                            _rs = _rs.merge(_tk, on=["rpgt_join", "currency_join", "bin_join"], how="left")
                            _tkr = pd.to_numeric(_rs["rpgt_ticket"], errors="coerce")
                            _tkr = _tkr.fillna(pd.to_numeric(_rs.get("avg_ticket"), errors="coerce")).fillna(0.0)
                            _rs["ticket"] = _tkr
                            _rs["pre_rev_raw"] = _rs["raw_succ"] * _rs["ticket"]
                            _raw_rpgt = _rs
                            _pre_raw_map = _rs.groupby("gateway_join", as_index=False)["pre_rev_raw"].sum()
                            if "gateway_join" in workings_full.columns:
                                # 19fm: same drop-first rule as the `_bcv` merge above. `pre_rev_raw`
                                # is not currently pre-created anywhere, so this is prevention, not
                                # a fix — and it is the cheap half of the pair.
                                workings_full = workings_full.drop(
                                    columns=["pre_rev_raw"], errors="ignore")
                                workings_full = workings_full.merge(_pre_raw_map, on="gateway_join", how="left")
                    except Exception as _rre:  # noqa: BLE001
                        # 19ey: SAY SO. This swallowed a KeyError from the `bin_join` rename and
                        # left `_raw_rpgt` as None, so the failure re-emerged 200 lines later as
                        # "AttributeError: 'int' object has no attribute 'fillna'" — an error
                        # that named neither the column nor the frame that was actually missing.
                        # The fallback stays (this block is optional enrichment, not the table),
                        # but a silent one costs more to diagnose than it ever saves.
                        _raw_rpgt = None
                        try:
                            st.caption(f"Raw 30D per-RPGT enrichment unavailable "
                                       f"({type(_rre).__name__}: {_rre}) — the Raw Attempts / "
                                       "Raw Successes / Pre Revenue (Adj) columns will read 0. "
                                       "The rest of the table is unaffected.")
                        except Exception:  # noqa: BLE001
                            pass
                    if "pre_rev_raw" in workings_full.columns:
                        workings_full["Pre Revenue (Adj)"] = pd.to_numeric(
                            workings_full["pre_rev_raw"], errors="coerce").fillna(_pre_fallback)
                    else:
                        workings_full["Pre Revenue (Adj)"] = _pre_fallback
                    workings_full["Post Revenue"] = pd.to_numeric(workings_full.get("Expected_Rev", 0), errors="coerce").fillna(0)
                    workings_full["Expected Revenue Impact"] = workings_full["Post Revenue"] - workings_full["Pre Revenue (Adj)"]

                    # ---- Reconciliation chains (trace every figure from visible columns) ----
                    _exp_att = pd.to_numeric(workings_full.get("Expected_Attempts", 0), errors="coerce").fillna(0.0)
                    _exp_succ = pd.to_numeric(workings_full.get("Expected_Success", 0), errors="coerce").fillna(0.0)
                    _base_att = pd.to_numeric(workings_full.get("Baseline_Attempts",
                                              workings_full.get("curr_vol", 0)), errors="coerce").fillna(0.0)
                    _base_succ = pd.to_numeric(workings_full.get("Baseline_Success", 0), errors="coerce").fillna(0.0)
                    # SUCCESS-RATE chain: Baseline/Expected Successes = Attempts × SR applied.
                    workings_full["Baseline Attempts (30D)"] = _base_att
                    workings_full["Baseline Success (30D)"] = _base_succ
                    workings_full["SR applied (30D)"] = np.where(_exp_att > 0, _exp_succ / _exp_att, 0.0)  # fraction
                    # REVENUE chain: the per-RPGT-blended ticket ACTUALLY used (Post Rev ÷ Expected
                    # Success), so Post Revenue = Expected Success × this ticket, and Pre Revenue =
                    # Baseline Success × the per-RPGT ticket. Contrast with the Bank×Cur ticket column.
                    workings_full["Eff. Ticket (per-RPGT)"] = np.where(
                        _exp_succ > 0, workings_full["Post Revenue"] / _exp_succ, 0.0)
                    # ALLOCATION chain: the floor / max-share parameters, plus the NET move from the raw
                    # softmax share to the final proposed share. Floor, max-share cap and the cross-profile
                    # VAMP/MID enforcement are applied together (not per-gateway-decomposable), so their
                    # combined effect is shown as one reconcilable shift = Proposed − Softmax (pre-floor).
                    workings_full["Exploration floor %"] = float(ss.get("exploration_floor", 0.0) or 0.0) * 100.0
                    workings_full["Max share cap %"] = float((ss.get("wallet_ctx") or {}).get("max_share", 0.97)) * 100.0
                    # (the Floor+cap+enforce shift needs Softmax Share (pre-floor); added after that block)

                    # Softmax workings, laid out exactly as the engine computes
                    # them (ALL_RPGTS level): weighting = e^(engine_score * k),
                    # proposed share = weighting / total weighting in the profile.
                    _temp = ss.get("softmax_temperature")
                    _tmethod = ss.get("temp_method", "Manual")
                    _celltemp = ss.get("cell_temperature", {}) or {}
                    _show_softmax = ss.get("variations_engine") == "softmax" and (bool(_temp) or bool(_celltemp))
                    if _show_softmax:
                        if _celltemp:
                            _fb = float(_temp) if _temp else 0.17
                            workings_full["Temperature (cell)"] = [
                                _celltemp.get(f"{c}|{b}", _fb) for c, b in
                                zip(workings_full["currency_join"], workings_full["bin_join"])]
                            _k = workings_full["Temperature (cell)"].astype(float) * 100.0   # per-profile multiplier
                        else:
                            _k = float(_temp) * 100.0            # dial 0.16 -> k = 16
                        _es = workings_full["Engine Score (Smoothed SR)"].astype(float)  # fraction
                        workings_full["k applied (score x k)"] = _es * _k
                        workings_full["Euler's constant"] = np.e
                        workings_full["Weighting"] = np.exp(_es * _k)
                        workings_full["Total Weighting"] = workings_full.groupby(["BIN", "Currency"])["Weighting"].transform("sum")
                        workings_full["Softmax Share (pre-floor)"] = np.where(
                            workings_full["Total Weighting"] > 0,
                            workings_full["Weighting"] / workings_full["Total Weighting"], 0.0)
                        # ALLOCATION chain: net move from the raw softmax share to the final proposed
                        # share = exploration floor + max-share cap + cross-profile VAMP/MID enforcement,
                        # combined (they're not per-gateway-decomposable). = Proposed − Softmax(pre-floor).
                        workings_full["Floor+cap+enforce shift (pp)"] = (
                            pd.to_numeric(workings_full["Proposed Share"], errors="coerce").fillna(0.0)
                            - pd.to_numeric(workings_full["Softmax Share (pre-floor)"], errors="coerce").fillna(0.0)
                        ) * 100.0

                    # Genetic engine has NO per-profile pre-softmax score. Instead show its OWN
                    # workings: the revenue-greedy REFERENCE (dial-100 waterfall: fill the best
                    # gateways up to the max share), the TILT the GA applied, and the FINAL share.
                    _is_genetic = ss.get("variations_engine") in ("genetic", "genetic_numba")
                    if _is_genetic:
                        _gcap = float((ss.get("wallet_ctx") or {}).get("max_share", 0.97))
                        # Explicit per-profile loop (robust: groupby.apply returning a Series can be
                        # coerced to a DataFrame in some pandas versions).
                        _ref_col = pd.Series(0.0, index=workings_full.index)
                        for _grp_key, _idxs in workings_full.groupby(["BIN", "Currency"]).groups.items():
                            _s = workings_full.loc[_idxs, "Engine Score (Smoothed SR)"].astype(float).to_numpy()
                            _n = len(_s)
                            _ref = np.zeros(_n)
                            _rem = 1.0
                            for _pos in np.argsort(-_s, kind="stable"):
                                if _rem <= 1e-12:
                                    break
                                _take = min(_gcap, _rem)
                                _ref[_pos] = _take
                                _rem -= _take
                            if _rem > 1e-9 and _n > 0:
                                _ref += _rem / _n
                            _ref_col.loc[_idxs] = _ref
                        workings_full["Reference Share (waterfall)"] = _ref_col
                        workings_full["Final Share"] = workings_full["Proposed Share"].astype(float)
                        workings_full["Tilt (pp)"] = (workings_full["Final Share"]
                                                      - workings_full["Reference Share (waterfall)"]) * 100.0
                        _genetic_cols = ["Reference Share (waterfall)", "Tilt (pp)", "Final Share"]
                    else:
                        _genetic_cols = []

                    workings_full = workings_full.sort_values(["BIN", "Currency", "Expected_Attempts", "raw_att"], ascending=[True, True, False, False])

                    if _show_softmax:
                        _softmax_cols = ["k applied (score x k)", "Euler's constant", "Weighting",
                                         "Total Weighting", "Softmax Share (pre-floor)"]
                        if "Temperature (cell)" in workings_full.columns:
                            _softmax_cols = ["Temperature (cell)"] + _softmax_cols
                    else:
                        _softmax_cols = []
                    # ALLOCATION-chain columns (floor / max-share params + the net floor+cap+enforce
                    # shift bridging Softmax pre-floor → final Proposed share).
                    _alloc_cols = ["Exploration floor %", "Max share cap %"]
                    if "Floor+cap+enforce shift (pp)" in workings_full.columns:
                        _alloc_cols.append("Floor+cap+enforce shift (pp)")
                    # Time-decay suffix: these columns feed the engine score and
                    # ARE decayed when the time-decay toggle is on.
                    _ta = " (time-adj)" if ss.get("apply_decay") else ""
                    _ATT, _SR = "All-Time Attempts" + _ta, "All-Time Raw SR" + _ta
                    _BAA, _BAS = "Bayesian Adj Attempts" + _ta, "Bayesian Adj Success" + _ta
                    workings_view = workings_full[[
                        "BIN", "Currency", "Gateway", "Cross-border?",
                        "All-Time Attempts (raw)", "All-Time Attempts", "All-Time Raw SR", "Prior SR %", "κ used",
                        "Bayesian Adj Attempts", "Bayesian Adj Success",
                        "Engine Score (Smoothed SR)",
                        *_softmax_cols,
                        *_genetic_cols,
                        *_alloc_cols,
                        "raw_att", "raw_succ", "Raw 30D Success Rate", "Raw 30D Amount",
                        # SUCCESS-RATE chain: Attempts × SR = Successes (baseline and expected).
                        "Baseline Attempts (30D)", "Baseline Success (30D)",
                        "Expected_Attempts", "Expected_Success", "SR applied (30D)",
                        "Current Share", "Proposed Share", "Shift (pp)",
                        # REVENUE chain: Success × ticket = Revenue (per-RPGT ticket actually used).
                        "Avg txn value (Bank x Cur)", "Eff. Ticket (per-RPGT)",
                        "Pre Revenue (Adj)", "Post Revenue", "Expected Revenue Impact"
                    ]].rename(columns={
                        "raw_att": "Raw Attempts (30D)",
                        "raw_succ": "Raw Successes (30D)",
                        "Expected_Attempts": "Expected Attempts (30D)",
                        "Expected_Success": "Expected Success (30D)",
                        "All-Time Attempts": _ATT, "All-Time Raw SR": _SR,
                        "Bayesian Adj Attempts": _BAA, "Bayesian Adj Success": _BAS,
                    })

                    workings_view[_SR] *= 100
                    workings_view["Engine Score (Smoothed SR)"] *= 100
                    workings_view["Raw 30D Success Rate"] *= 100
                    workings_view["SR applied (30D)"] *= 100
                    workings_view["Current Share"] *= 100
                    workings_view["Proposed Share"] *= 100
                    if _show_softmax:
                        workings_view["Softmax Share (pre-floor)"] *= 100
                    if _is_genetic:
                        workings_view["Reference Share (waterfall)"] *= 100
                        workings_view["Final Share"] *= 100

                    # (The old single combined table is replaced by the three headed tables below:
                    # Revenue Impact Workings, Pre-Processing Workings, Allocation Workings.)

                    # ============ Three headed workings tables (all at per-RPGT grain) ============
                    # 1) Revenue Impact Workings  2) Pre-Processing Workings  3) Allocation Workings.
                    # Grain = Bank x Currency x Gateway x RPGT. Score/softmax columns are pooled at
                    # Bank x Currency (the grain the engine scores on) and repeat across a gateway's RPGTs.
                    if (not b_df.empty) and {"rpgt", "gateway"}.issubset(b_df.columns):
                        _b = b_df.copy()
                        for _kj, _sc0 in (("gateway_join", "gateway"), ("bin_join", "bin"), ("currency_join", "currency")):
                            if _kj not in _b.columns:
                                _b[_kj] = _b[_sc0].astype(str).str.strip().str.lower()
                        _grp = ["bin_join", "currency_join", "rpgt", "gateway_join"]
                        _aggmap = {}
                        for _src, _dst in [("post_att", "Expected Attempts"), ("post_succ", "Expected Success"),
                                           ("post_rev", "Post Revenue")]:
                            if _src in _b.columns:
                                _aggmap[_dst] = (_src, "sum")
                        for _src, _dst in [("gateway", "Gateway"), ("baseline_share", "Current Share"),
                                           ("share", "Proposed Share"), ("gw_sr", "SR (30D)"),
                                           ("avg_ticket", "Ticket (per-RPGT)")]:
                            if _src in _b.columns:
                                _aggmap[_dst] = (_src, "first")
                        if any(_k in _aggmap for _k in ("Post Revenue", "Expected Success")):
                            _wr = _b.groupby(_grp, as_index=False).agg(**_aggmap)
                            # RAW observed 30D + raw-basis Pre Revenue (case-insensitive rpgt join)
                            if _raw_rpgt is not None and not getattr(_raw_rpgt, "empty", True):
                                _rr = _raw_rpgt[["bin_join", "currency_join", "rpgt_join", "gateway_join",
                                                 "raw_succ", "raw_att", "pre_rev_raw"]].copy()
                                _wr["_rpgt_l"] = _wr["rpgt"].astype(str).str.strip().str.lower()
                                _wr = _wr.merge(_rr, how="left",
                                                left_on=["bin_join", "currency_join", "_rpgt_l", "gateway_join"],
                                                right_on=["bin_join", "currency_join", "rpgt_join", "gateway_join"])
                                _wr = _wr.drop(columns=["_rpgt_l", "rpgt_join"], errors="ignore")
                            # 19ez: the merge above is CONDITIONAL on `_raw_rpgt`, so when that
                            # enrichment does not run these three columns never arrive and every
                            # `.get(col, 0)` below returns a bare int. Guarantee them as real
                            # Series first — the same fix app_common._impact_eval_frame already
                            # applies to its own inputs.
                            ensure_cols(_wr, (("raw_att", 0.0), ("raw_succ", 0.0),
                                              ("pre_rev_raw", 0.0)))
                            _wr["Raw Attempts (30D)"] = pd.to_numeric(_wr["raw_att"], errors="coerce").fillna(0.0)
                            _wr["Raw Successes (30D)"] = pd.to_numeric(_wr["raw_succ"], errors="coerce").fillna(0.0)
                            _wr["Pre Revenue (Adj)"] = pd.to_numeric(_wr["pre_rev_raw"], errors="coerce").fillna(0.0)
                            # merge POOLED score / softmax / allocation / genetic columns (broadcast per RPGT)
                            _pool = [c for c in ["Cross-border?", "All_Time_Attempts", "All_Time_Success", "All-Time Raw SR",
                                                 "Prior SR %", "κ used", "Bayesian Adj Attempts", "Bayesian Adj Success",
                                                 "Engine Score (Smoothed SR)", "Temperature (cell)", "k applied (score x k)",
                                                 "Euler's constant", "Weighting", "Total Weighting", "Softmax Share (pre-floor)",
                                                 "Exploration floor %", "Max share cap %", "Reference Share (waterfall)",
                                                 "Tilt (pp)", "Final Share"] if c in workings_full.columns]
                            if _pool and {"bin_join", "currency_join", "gateway_join"}.issubset(workings_full.columns):
                                _pfm = workings_full[["bin_join", "currency_join", "gateway_join"] + _pool].drop_duplicates(
                                    ["bin_join", "currency_join", "gateway_join"])
                                _wr = _wr.merge(_pfm, on=["bin_join", "currency_join", "gateway_join"], how="left")
                            # derived (on FRACTIONS, before the %-scaling below). A column may be
                            # absent for some engines (e.g. no Softmax Share for genetic), so read via
                            # a helper that always returns a Series (never a scalar → no .fillna crash).
                            # [FN-375]
                            def _S(_name, _d=0.0):
                                if _name in _wr.columns:
                                    return pd.to_numeric(_wr[_name], errors="coerce").fillna(_d)
                                return pd.Series(_d, index=_wr.index, dtype=float)
                            _kap = _S("κ used")
                            _prior = _S("Prior SR %") / 100.0
                            _wr["Bayesian Attempts Adjustment"] = _kap
                            _wr["Bayesian Success Adjustment"] = _kap * _prior
                            _pfrac = _S("Proposed Share")
                            if "Softmax Share (pre-floor)" in _wr.columns:
                                _sfrac = pd.to_numeric(_wr["Softmax Share (pre-floor)"], errors="coerce").fillna(_pfrac)
                                _wr["Floor+cap+enforce shift (pp)"] = (_pfrac - _sfrac) * 100.0
                            _wr["Raw SR % (All-Time)"] = _S("All-Time Raw SR") * 100.0
                            _wr["Bank"] = _wr["bin_join"].astype(str).str.upper()
                            _wr["Currency"] = _wr["currency_join"].astype(str).str.upper()
                            if "Gateway" not in _wr.columns:
                                _wr["Gateway"] = _wr["gateway_join"]
                            _wr = _wr.rename(columns={"rpgt": "RPGT"})
                            for _pc in ["Current Share", "Proposed Share", "SR (30D)", "Engine Score (Smoothed SR)",
                                        "Softmax Share (pre-floor)", "Reference Share (waterfall)", "Final Share"]:
                                if _pc in _wr.columns:
                                    _wr[_pc] = pd.to_numeric(_wr[_pc], errors="coerce").fillna(0.0) * 100.0
                            _wr = _wr.sort_values(["Bank", "Currency", "Gateway", "RPGT"]).reset_index(drop=True)

                            # -------- 1) Revenue Impact Workings --------
                            st.markdown("<h4 style='color:#0B1F3A;margin:0.4rem 0 0.2rem;'>Revenue Impact Workings</h4>", unsafe_allow_html=True)
                            _c1 = [c for c in ["Bank", "Currency", "Gateway", "RPGT", "SR (30D)", "Ticket (per-RPGT)",
                                               "Current Share", "Proposed Share", "Raw Attempts (30D)", "Raw Successes (30D)",
                                               "Expected Attempts", "Expected Success", "Pre Revenue (Adj)", "Post Revenue"] if c in _wr.columns]
                            st.dataframe(_wr[_c1], use_container_width=True, hide_index=True, column_config={
                                "SR (30D)": st.column_config.NumberColumn(format="%.2f%%"),
                                "Ticket (per-RPGT)": st.column_config.NumberColumn(format="$%.2f"),
                                "Current Share": st.column_config.NumberColumn(format="%.2f%%"),
                                "Proposed Share": st.column_config.NumberColumn(format="%.2f%%"),
                                "Raw Attempts (30D)": st.column_config.NumberColumn(format="%d"),
                                "Raw Successes (30D)": st.column_config.NumberColumn(format="%d"),
                                "Expected Attempts": st.column_config.NumberColumn(format="%d", help="cell attempts × proposed share (Σ per cell = Σ Raw Attempts)."),
                                "Expected Success": st.column_config.NumberColumn(format="%d", help="Expected Attempts × SR."),
                                "Pre Revenue (Adj)": st.column_config.NumberColumn(format="$%.0f", help="Raw Successes × Ticket (per-RPGT)."),
                                "Post Revenue": st.column_config.NumberColumn(format="$%.0f", help="Expected Success × Ticket (per-RPGT)."),
                            })

                            # -------- 2) Pre-Processing & Engine Score Workings --------
                            st.markdown("<h4 style='color:#0B1F3A;margin:0.4rem 0 0.2rem;'>Pre-Processing &amp; Engine Score Workings</h4>", unsafe_allow_html=True)
                            _c2map = {"Cross-border?": "Cross Border", "All_Time_Attempts": "Raw Attempts (All-Time)",
                                      "All_Time_Success": "Raw Successes (All-Time)",
                                      "Bayesian Adj Attempts": "Bayesian Adj Attempts (time-adj)",
                                      "Bayesian Adj Success": "Bayesian Adj Successes (time-adj)"}
                            # RPGT column ONLY when the Engine Score grain is per-RPGT. When the score is
                            # pooled at Bank×Currency it's identical across a gateway's RPGTs, so drop RPGT
                            # and collapse to one row per gateway.
                            _score_rpgt = bool(ss.get("score_by_rpgt", False))
                            _c2cols = (["Bank", "Currency", "Gateway"] + (["RPGT"] if _score_rpgt else [])
                                       + ["Cross-border?", "All_Time_Attempts", "All_Time_Success", "Raw SR % (All-Time)",
                                          "Bayesian Attempts Adjustment", "Bayesian Success Adjustment",
                                          "Bayesian Adj Attempts", "Bayesian Adj Success", "Engine Score (Smoothed SR)"])
                            _c2src = [c for c in _c2cols if c in _wr.columns]
                            _c2 = _wr[_c2src].rename(columns=_c2map)
                            if not _score_rpgt:
                                _c2 = _c2.drop_duplicates(["Bank", "Currency", "Gateway"]).reset_index(drop=True)
                            st.dataframe(_c2, use_container_width=True, hide_index=True, column_config={
                                "Raw Attempts (All-Time)": st.column_config.NumberColumn(format="%.0f", help="All-time attempts (time-decay-weighted when decay is on) — the grain the engine scores on."),
                                "Raw Successes (All-Time)": st.column_config.NumberColumn(format="%.0f"),
                                "Raw SR % (All-Time)": st.column_config.NumberColumn(format="%.2f%%"),
                                "Bayesian Attempts Adjustment": st.column_config.NumberColumn(format="%.1f", help="κ — the smoothing volume added to attempts."),
                                "Bayesian Success Adjustment": st.column_config.NumberColumn(format="%.2f", help="κ × prior — the amount added to successes."),
                                "Bayesian Adj Attempts (time-adj)": st.column_config.NumberColumn(format="%.1f", help="All-Time Attempts + κ."),
                                "Bayesian Adj Successes (time-adj)": st.column_config.NumberColumn(format="%.1f", help="All-Time Successes + κ × prior."),
                                "Engine Score (Smoothed SR)": st.column_config.NumberColumn(format="%.2f%%", help="Adj Successes ÷ Adj Attempts. Pooled at Bank×Currency, so it repeats across a gateway's RPGTs."),
                            })

                            # -------- 3) Allocation Workings --------
                            st.markdown("<h4 style='color:#0B1F3A;margin:0.4rem 0 0.2rem;'>Allocation Workings</h4>", unsafe_allow_html=True)
                            _c3 = ["Bank", "Currency", "Gateway", "RPGT", "Engine Score (Smoothed SR)",
                                   "Temperature (cell)", "k applied (score x k)", "Euler's constant", "Weighting",
                                   "Total Weighting", "Softmax Share (pre-floor)", "Reference Share (waterfall)", "Tilt (pp)",
                                   "Exploration floor %", "Max share cap %", "Proposed Share", "Floor+cap+enforce shift (pp)"]
                            _c3 = [c for i, c in enumerate(_c3) if c in _wr.columns and c not in _c3[:i]]
                            st.dataframe(_wr[_c3], use_container_width=True, hide_index=True, column_config={
                                "Engine Score (Smoothed SR)": st.column_config.NumberColumn(format="%.2f%%"),
                                "Temperature (cell)": st.column_config.NumberColumn(format="%.3f"),
                                "k applied (score x k)": st.column_config.NumberColumn(format="%.4f"),
                                "Euler's constant": st.column_config.NumberColumn(format="%.5f"),
                                "Weighting": st.column_config.NumberColumn(format="%.2f"),
                                "Total Weighting": st.column_config.NumberColumn(format="%.2f"),
                                "Softmax Share (pre-floor)": st.column_config.NumberColumn(format="%.2f%%", help="Softmax share before floor/cap/enforcement. Pooled at Bank×Currency."),
                                "Reference Share (waterfall)": st.column_config.NumberColumn(format="%.2f%%", help="Genetic revenue reference (dial-100 waterfall)."),
                                "Tilt (pp)": st.column_config.NumberColumn(format="%+.2f pp"),
                                "Exploration floor %": st.column_config.NumberColumn(format="%.2f%%"),
                                "Max share cap %": st.column_config.NumberColumn(format="%.2f%%"),
                                "Proposed Share": st.column_config.NumberColumn(format="%.2f%%"),
                                "Floor+cap+enforce shift (pp)": st.column_config.NumberColumn(format="%+.2f pp", help="Proposed − Softmax(pre-floor): net effect of floor + max-share cap + VAMP/MID enforcement."),
                            })
                        else:
                            st.caption("No per-RPGT revenue detail available for this selection.")
                    else:
                        st.caption("No per-RPGT detail available for this selection (e.g. a validate / parsed-rules split).")
                elif not debug_mode:
                    st.info("Table is empty. Please check the 'Toggle Debug Diagnostics' box above to find out why.")

        # [FN-376]
        def _prepost_render(mode):
            if os.path.exists(pp_path):
                # Reuse the granular projection already computed for the VAMP table above
                # (identical args) instead of projecting again.
                _gr_floor3 = (0.0 if os.environ.get("ROUTING_PROJ_FLOOR", "0") == "0"
                              else float(ss.get("exploration_floor", 0.0) or 0.0))
                _wcp3, _uop3, _ = _cap_pairs(
                    os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                    input_json_path("routing_restrictions.json"))
                _gr = _gr_shared if _gr_shared is not None else _c_prepost_granular(
                    pp_path, projection_cache_sig(pp_path, prop_items, _gr_floor3,
                                                  wallet_incapable_pairs=_wcp3,
                                                  usa_only_pairs=_uop3),
                    prop_items, excluded_mids, _kill_eff, _m0s, _scoped_rpgts,
                    frozenset(str(x).strip().lower() for x in ((ss.get("wallet_ctx") or {}).get("incapable") or set())),
                    frozenset(str(x).strip().lower() for x in ((ss.get("wallet_ctx") or {}).get("usa_only") or set())),
                    exploration_floor=_gr_floor3,
                    wallet_incapable_pairs=_wcp3, usa_only_pairs=_uop3,
                    # 19df — the same max-share cap the search applies. This site usually reuses
                    # `_gr_shared` from :4205 and only computes its own frame when that is None;
                    # if the two disagreed on the cap, which branch ran would change the numbers.
                    max_share=float((ss.get("wallet_ctx") or {}).get("max_share", 0.97)))
                if _gr is None or getattr(_gr, "empty", True):
                    _ink_caption("No pro-rata rows available.")
                else:
                    _gr = _gr.copy()
                    _gr["Currency"] = _gr["Currency"].astype(str).str.upper()
                    # 6 filter boxes squeezed into the left 60% (a 4-unit spacer takes the rest),
                    # so each input is ~40% narrower than spanning the full width. Rendered into
                    # the Risk Impact tab slot so the filters sit with the RPGT table + bar charts.
                    fg = st.columns([1, 1, 1, 1, 1, 1, 4])

                    # Find the vampMid with the highest VAMP Post M0 count to use as default
                    _top_mid_val = "(All)"
                    if not vp.empty and "VAMP Post M0" in vp.columns:
                        _vp_valid = vp[vp["VAMP Post M0"] > 0]
                        if not _vp_valid.empty:
                            _top_mid_val = str(_vp_valid.sort_values("VAMP Post M0", ascending=False).iloc[0]["vampMid"])

                    # [FN-377]
                    def _optsel(col, container, label, def_val="(All)"):
                        opts = ["(All)"] + sorted(_gr[col].astype(str).unique().tolist())
                        idx = opts.index(def_val) if def_val in opts else 0
                        return container.selectbox(label, opts, index=idx, key=f"{mode}_pp_{col}")

                    _f_mid = _optsel("vampMid", fg[0], "vampMid", def_val=_top_mid_val)
                    _f_rpgt = _optsel("RPGT", fg[1], "RPGT")
                    _f_bin = _optsel("BIN", fg[2], "BIN")
                    _f_cur = _optsel("Currency", fg[3], "Currency")
                    _per_opts = ["(All)"] + sorted(_gr["period"].unique().tolist())
                    # Default the period filter to 1 when present, else '(All)'.
                    _per_def = next((i for i, _o in enumerate(_per_opts)
                                     if str(_o) == "1" or _o == 1), 0)
                    _f_per = fg[4].selectbox("period", _per_opts, index=_per_def, key=f"{mode}_pp_period")
                    _f_t = fg[5].selectbox("t", ["(All)"] + sorted(_gr["t"].unique().tolist()), key=f"{mode}_pp_t")

                    # Frame filtered by the cell fields only (period/t excluded)
                    # so the monthly chart spans all periods.
                    _fp = _gr.copy()
                    for _c, _v in [("vampMid", _f_mid), ("RPGT", _f_rpgt), ("BIN", _f_bin), ("Currency", _f_cur)]:
                        if _v != "(All)":
                            _fp = _fp[_fp[_c].astype(str) == _v]
                    # Table adds the period/t filters on top.
                    _flt = _fp.copy()
                    if _f_per != "(All)":
                        _flt = _flt[_flt["period"] == _f_per]
                    if _f_t != "(All)":
                        _flt = _flt[_flt["t"] == _f_t]
                    _numcols = ["VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]
                    # Aggregate to vampMid × BIN × period (drop RPGT / Currency / t), summing counts.
                    _show = (_flt.groupby(["vampMid", "BIN", "period"], as_index=False)[_numcols].sum()
                             .sort_values(["VAMP_Post", "period"], ascending=[False, True]))
                    _sv = _show.head(400)
                    _dcols = ["vampMid", "BIN", "period"] + _numcols
                    _tot = {c: float(_show[c].sum()) for c in _numcols}

                    # [FN-378]
                    def _dfmt(_c, _v):
                        if _c == "period":
                            return f"{int(_v)}"
                        if _c in ("vampMid", "BIN"):
                            return str(_v)
                        return f"{float(_v):,.0f}"   # counts as whole numbers

                    # Granular detail table, tightly hugging the text. Fixed height to match
                    # the combined space of the two charts beside it (VAMP + Transactions,
                    # each 216px, plus their headers and the spacer ≈ 496px).
                    _dh = ['<div style="display:inline-block; max-width:100%; box-shadow:0 4px 12px rgba(0,0,0,0.08); '
                           'border-radius:0; overflow:auto; height:560px; '
                           'background-color:var(--tav-card); border:1px solid var(--tav-line);">']

                    # OVERRIDE: width:auto !important overrides Streamlit's global 100% width rule
                    _dh.append('<table style="width:auto !important; border-collapse:collapse; font-family:inherit; '
                               'font-size:0.72rem; line-height:1.1;"><tr>')
                    
                    for _c in _dcols:
                        _al = "left" if _c in ("vampMid", "BIN") else "right"
                        # Wrap numeric headers at underscores so the column shrinks to its widest VALUE;
                        # match body-row padding + drop the fixed height so header height == row height.
                        # Header on ONE line; column auto-sizes to fit the header AND its values.
                        _hdr = _mc_hdr(_c)
                        _hws = "nowrap"
                        _dh.append(f'<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; '
                                   f'padding:1px 6px; text-align:{_al}; white-space:{_hws}; position:sticky; top:0; '
                                   f'width:1%; box-sizing:border-box; vertical-align:middle;">{_hdr}</th>')
                    _dh.append('</tr>')
                    
                    for _, _rr in _sv.iterrows():
                        _dh.append('<tr>')
                        for _c in _dcols:
                            _al = "left" if _c in ("vampMid", "BIN") else "right"
                            _dh.append(f'<td style="padding:1px 6px; text-align:{_al}; color:#000000; white-space:nowrap; width:1%;">{_dfmt(_c, _rr[_c])}</td>')
                        _dh.append('</tr>')
                        
                    # Sticky TOTAL row (across all filtered rows).
                    _dh.append('<tr>')
                    for _c in _dcols:
                        _al = "left" if _c in ("vampMid", "BIN") else "right"
                        _tv = "TOTAL" if _c == "vampMid" else (_dfmt(_c, _tot[_c]) if _c in _numcols else "")
                        _dh.append(f'<td style="padding:2px 6px; text-align:{_al}; color:#000000; font-weight:800; '
                                   f'position:sticky; bottom:0; background-color:var(--tav-card); '
                                   f'border-top:2px solid var(--tav-line); white-space:nowrap; width:1%;">{_tv}</td>')
                    _dh.append('</tr>')
                    _dh.append('</table></div>')
                    
                    # ---- Lifetime table (vampMid × BIN × period). Uses ALL detail filters EXCEPT 't'
                    # (VAMP is summed over every age t = the cohort's LIFETIME VAMP). ----
                    # No raw calendar-period filter here — a cohort's lifetime spans several calendar
                    # months, so we keep all rows and filter by ORIGIN period on the result instead.
                    _lt = _fp.copy()
                    _lt["period"] = pd.to_numeric(_lt["period"], errors="coerce").fillna(0).astype(int)
                    _lt["t"] = pd.to_numeric(_lt["t"], errors="coerce").fillna(0).astype(int)
                    _lt["orig_m"] = _lt["period"] - _lt["t"]
                    _vamp_lt = _lt.groupby(["vampMid", "BIN", "orig_m"], as_index=False).agg(
                        VAMP_Pre=("VAMP_Pre", "sum"), VAMP_Post=("VAMP_Post", "sum"))
                    _txn_o = (_lt[_lt["t"] == 0].groupby(["vampMid", "BIN", "period"], as_index=False)
                              .agg(VI_Txn_Pre=("VI_Txn_Pre", "sum"), VI_Txn_Post=("VI_Txn_Post", "sum"))
                              .rename(columns={"period": "orig_m"}))
                    _lt_tbl = _txn_o.merge(_vamp_lt, on=["vampMid", "BIN", "orig_m"], how="outer").fillna(0.0)
                    _lt_tbl = _lt_tbl[_lt_tbl["orig_m"].between(0, 5)].rename(columns={"orig_m": "period"})
                    if _f_per != "(All)":              # filter by ORIGIN period (the displayed column)
                        _lt_tbl = _lt_tbl[_lt_tbl["period"] == int(_f_per)]
                    _lt_tbl = _lt_tbl.sort_values(["vampMid", "BIN", "period"])
                    _ltcols = ["vampMid", "BIN", "period", "VI_Txn_Pre", "VI_Txn_Post", "VAMP_Pre", "VAMP_Post"]
                    _ltnum = ["VI_Txn_Pre", "VI_Txn_Post", "VAMP_Pre", "VAMP_Post"]
                    _lttot = {c: float(_lt_tbl[c].sum()) for c in _ltnum}

                    # [FN-379]
                    def _ltfmt(_c, _v):
                        if _c == "period":
                            return f"{int(_v)}"
                        if _c in ("vampMid", "BIN"):
                            return str(_v)
                        return f"{float(_v):,.0f}"

                    # [FN-380]
                    def _ltcw(_c):   # ~30% narrower overall: shrink every column's font + padding
                        return ("padding:2px 4px; font-size:0.5rem;" if _c == "vampMid"
                                else "padding:1px 1px; font-size:0.35rem;")

                    _lth = ['<div style="display:inline-block; max-width:100%; box-shadow:0 4px 12px rgba(0,0,0,0.08); '
                            'border-radius:0; overflow:auto; height:560px; background-color:var(--tav-card); '
                            'border:1px solid var(--tav-line);">']
                    _lth.append('<table style="width:auto !important; border-collapse:collapse; font-family:inherit; '
                                'font-size:0.72rem; line-height:1.1;"><tr>')
                    # VAMP columns are cohort LIFETIME VAMP (summed over all ages t) → label as such.
                    _lthdr = {"VAMP_Pre": "Lifetime VAMP_Pre", "VAMP_Post": "Lifetime VAMP_Post",
                              "period": "origin period"}
                    for _c in _ltcols:
                        _al = "left" if _c in ("vampMid", "BIN") else "right"
                        # Non-vampMid headers wrap at underscores (<wbr>) so the column shrinks to
                        # the value width instead of the long header — ~40%+ narrower.
                        _disp = _mc_hdr(_lthdr.get(_c, _c))
                        # Header on ONE line; column auto-sizes to fit the header AND its values.
                        _hdr = _disp
                        _ws = "nowrap"
                        _lth.append(f'<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; '
                                    f'{_ltcw(_c)} text-align:{_al}; white-space:{_ws}; position:sticky; top:0; '
                                    f'width:1%; box-sizing:border-box; vertical-align:middle;">{_hdr}</th>')
                    _lth.append('</tr>')
                    for _, _rr in _lt_tbl.iterrows():
                        _lth.append('<tr>')
                        for _c in _ltcols:
                            _al = "left" if _c in ("vampMid", "BIN") else "right"
                            _lth.append(f'<td style="{_ltcw(_c)} text-align:{_al}; color:#000; white-space:nowrap; width:1%;">{_ltfmt(_c, _rr[_c])}</td>')
                        _lth.append('</tr>')
                    _lth.append('<tr>')
                    for _c in _ltcols:
                        _al = "left" if _c in ("vampMid", "BIN") else "right"
                        _tv = "TOTAL" if _c == "vampMid" else (_ltfmt(_c, _lttot[_c]) if _c in _ltnum else "")
                        _lth.append(f'<td style="{_ltcw(_c)} text-align:{_al}; color:#000; font-weight:800; position:sticky; '
                                    f'bottom:0; background-color:var(--tav-card); border-top:2px solid var(--tav-line); '
                                    f'white-space:nowrap; width:1%;">{_tv}</td>')
                    _lth.append('</tr></table></div>')

                    # ---- New: vampMid × RPGT aggregate (same filters as the detail table) ----
                    _rt = (_flt.groupby(["vampMid", "RPGT"], as_index=False)
                           .agg(VAMP_Pre=("VAMP_Pre", "sum"), VAMP_Post=("VAMP_Post", "sum"),
                                VI_Txn_Pre=("VI_Txn_Pre", "sum"), VI_Txn_Post=("VI_Txn_Post", "sum"))
                           .sort_values("VAMP_Post", ascending=False))
                    _rtcols = ["vampMid", "RPGT", "VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]
                    _rtnum = ["VAMP_Pre", "VAMP_Post", "VI_Txn_Pre", "VI_Txn_Post"]
                    _rttot = {c: float(_rt[c].sum()) for c in _rtnum}
                    _rth = ['<div style="display:inline-block; max-width:100%; margin-bottom:1.5rem; '
                            'box-shadow:0 4px 12px rgba(0,0,0,0.08); '
                            'border-radius:0; overflow:auto; max-height:560px; background-color:var(--tav-card); '
                            'border:1px solid var(--tav-line);">']
                    # Table shrunk (font 0.72rem → 0.43rem → 0.34rem, a further 20% + tighter padding).
                    _rth.append('<table style="width:auto !important; border-collapse:collapse; font-family:inherit; '
                                'font-size:0.34rem; line-height:1.1;"><tr>')
                    for _c in _rtcols:
                        _al = "left" if _c in ("vampMid", "RPGT") else "right"
                        # Wrap numeric headers at underscores so each column fits its value width.
                        _hdr = _mc_hdr(_c) if _c in ("vampMid", "RPGT") else _mc_hdr(_c).replace("_", "_<wbr>")
                        _ws = "nowrap" if _c in ("vampMid", "RPGT") else "normal"
                        _rth.append(f'<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; '
                                    f'padding:2px 3px; text-align:{_al}; white-space:{_ws}; position:sticky; top:0; width:1%;">{_hdr}</th>')
                    _rth.append('</tr>')
                    for _, _rr in _rt.iterrows():
                        _rth.append('<tr>')
                        for _c in _rtcols:
                            _al = "left" if _c in ("vampMid", "RPGT") else "right"
                            _val = str(_rr[_c]) if _c in ("vampMid", "RPGT") else f"{float(_rr[_c]):,.0f}"
                            _rth.append(f'<td style="padding:1px 6px; text-align:{_al}; color:#000; white-space:nowrap; width:1%;">{_val}</td>')
                        _rth.append('</tr>')
                    _rth.append('<tr>')
                    for _c in _rtcols:
                        _al = "left" if _c in ("vampMid", "RPGT") else "right"
                        _tv = "TOTAL" if _c == "vampMid" else (f"{_rttot[_c]:,.0f}" if _c in _rtnum else "")
                        _rth.append(f'<td style="padding:2px 6px; text-align:{_al}; color:#000; font-weight:800; position:sticky; '
                                    f'bottom:0; background-color:var(--tav-card); border-top:2px solid var(--tav-line); '
                                    f'white-space:nowrap; width:1%;">{_tv}</td>')
                    _rth.append('</tr></table></div>')

                    # Per-tab layout: "detail" mode shows the detail + lifetime tables; "impact"
                    # mode shows the vampMid × RPGT table beside the VAMP/Txn bar charts.
                    if mode == "detail":
                        # Detail + lifetime tables — column widths cut to 30% of before (−70%);
                        # a trailing spacer absorbs the freed room. Tables cap at the narrow column
                        # (max-width:100%) and scroll horizontally for any overflow.
                        _tcols = st.columns([3, 0.11, 3, 13.78], gap="small")
                        _tcols[0].markdown("".join(_dh), unsafe_allow_html=True)
                        _tcols[2].markdown("".join(_lth), unsafe_allow_html=True)
                    else:
                        # Impact tab: vampMid × RPGT table + the VAMP and Txn bar charts, all three
                        # side by side on one row.
                        _rlo = st.columns([1, 1, 1], gap="medium")
                        _rlo[0].markdown("".join(_rth), unsafe_allow_html=True)
                        _ts_slot = _rlo[1].container()    # VAMP bar chart
                        _ts_slot2 = _rlo[2].container()   # Txn bar chart

                    # Bar chart: actual months (thermometer) leading into the
                    # forecast, with forecast VAMP Pre vs Post (day-scaled).
                    if HAS_PLOTLY:
                        _fs2 = ss.get("forecast_settings", {}) or {}
                        _bd = pd.to_datetime(_fs2.get("month_0", date.today().replace(day=1)))
                        # month / company / scheme for the ACTUALS caches: derive from the loaded
                        # output folder (…/<month>/<company>/<scheme>/) so the actuals match the SAME
                        # brand + scheme as the forecast being viewed. Fall back to forecast_settings.
                        _odsegs = os.path.normpath(str(out_dir or "")).split(os.sep)
                        if len(_odsegs) >= 3 and _odsegs[-1] in ("visa", "mastercard"):
                            _mv2, _cmp2, _sch2 = _odsegs[-3], _odsegs[-2], _odsegs[-1]
                        else:
                            _mv2, _cmp2 = _fs2.get("month_var"), _fs2.get("company")
                            _sch2 = "mastercard" if _is_mc_disp else "visa"
                        import plotly.graph_objects as _gob
                        _rows = []
                        _ratio = {}   # month label -> VAMP ratio (%)
                        _act_ts_df = None   # actual VAMP by (period, age t) for the T-stacked chart
                        _post_m = _fp.groupby("period")["VAMP_Post"].sum()
                        _post_txn = _fp.groupby("period")["VI_Txn_Post"].sum()
                        for _m in range(6):
                            _md = _bd + pd.DateOffset(months=_m)
                            _lab = _md.strftime("%m-%y")
                            # VAMP_Post is ALREADY calendar-day: the actuarial engine's carryover
                            # system applied days/30.4167 (flex_ratio) and carried the residual
                            # forward. So plot VAMP_Post directly — re-applying the day factor here
                            # double-scaled the forecast bars (and broke reconciliation with the
                            # Risk-tab VAMP table and the detail/lifetime tables, which show
                            # VAMP_Post straight). The ratio already uses raw VAMP_Post (no _fac).
                            _rows.append({"month": _lab, "order": _m,
                                          "series": "Forecast Post", "VAMP": float(_post_m.get(_m, 0.0))})
                            _pt = float(_post_txn.get(_m, 0.0))
                            if _pt > 0:
                                _ratio[_lab] = float(_post_m.get(_m, 0.0)) / _pt * 100.0
                        # Actual VAMP from the thermometer (un-normalised for bars, raw for ratio).
                        # Prefer the gatewayFid-grained actuals cache (from fcast_query_gatewayfid.sql)
                        # so actuals can be reconciled to the forecast's vampMid grain; fall back to
                        # the standard thermometer cache if it hasn't been generated yet.
                        # Caches live under data/build_baseline_cached_input_data/<month>/<company>/<scheme>/ (19fj rename).
                        # Mastercard files carry a '_mc' suffix and have no gatewayFid-grained variant.
                        _cache_dir = os.path.join(PROJECT_ROOT, "data", "cache", str(_mv2), str(_cmp2), _sch2)
                        _sfx = "_mc" if _sch2 == "mastercard" else ""
                        _th_gw = os.path.join(_cache_dir, f"thermometer_data_gwfid_{_mv2}{_sfx}_fcp_v3.parquet")
                        _th_std = os.path.join(_cache_dir, f"thermometer_data_{_mv2}{_sfx}_fcp_v3.parquet")
                        _th = _th_gw if os.path.exists(_th_gw) else _th_std
                        _act, _act_raw = {}, {}
                        _act_v_rp = None   # actual VAMP by (period, RPGT-title) for the by-RPGT chart
                        _act_t_rp = None   # actual txns by (period, RPGT-title) for the by-RPGT chart
                        if os.path.exists(_th):
                            _av = _c_read_parquet(_th, _mtime(_th)).copy()
                            # Mastercard thermometer stores chargebacks as 'cb_count'; the code below
                            # is written for the Visa 'vamp_count' — alias so both schemes work.
                            if "vamp_count" not in _av.columns and "cb_count" in _av.columns:
                                _av = _av.rename(columns={"cb_count": "vamp_count"})
                            # gatewayFid cache: map gatewayFid -> vampMid (Master_MID_List) so the
                            # vampMid filter matches the forecast grain on the actuals too.
                            if "gatewayFid" in _av.columns and "vampMid" not in _av.columns:
                                _f2v = {}
                                _mmp_c = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                                if os.path.exists(_mmp_c):
                                    try:
                                        _mmd_c = load_mid_list(_mmp_c)
                                        _cc2 = _norm_cols(_mmd_c)
                                        if _cc2.get("gatewayfid") and _cc2.get("vampmid"):
                                            _f2v = _fid2vamp_from(_mmd_c, _cc2["gatewayfid"], _cc2["vampmid"])
                                    except Exception:  # noqa: BLE001
                                        _f2v = {}
                                _gwl = _av["gatewayFid"].astype(str).str.strip().str.lower()
                                _av["vampMid"] = _gwl.map(_f2v).fillna(_av["gatewayFid"].astype(str).str.strip())
                            for _cc in ("Company", "company"):
                                if _cmp2 and _cc in _av.columns:
                                    _av = _av[_av[_cc].astype(str).str.lower().str.strip() == str(_cmp2).lower().strip()]
                            if _f_rpgt != "(All)" and "rpgt" in _av.columns:
                                _av = _av[_av["rpgt"].astype(str).str.title() == _f_rpgt.title()]
                            if _f_bin != "(All)" and "bin" in _av.columns:
                                _av = _av[_av["bin"].astype(str).str.title() == _f_bin.title()]
                            if _f_mid != "(All)":
                                _vmcol = next((c for c in ["vampMid", "vamp_mid", "mid", "gateway"]
                                               if c in _av.columns), None)
                                if _vmcol is not None:
                                    _av = _av[_av[_vmcol].astype(str).str.strip() == str(_f_mid)]
                            if not _av.empty and "period" in _av.columns:
                                _pc = _av["period"].fillna(0).astype(int)
                                _tc = _av["time_to_event_months"].fillna(0).astype(int) if "time_to_event_months" in _av.columns else 0
                                _mb = _pc + 1 + _tc
                                _dm = {int(m): calendar.monthrange((_bd - pd.DateOffset(months=int(m))).year,
                                                                    (_bd - pd.DateOffset(months=int(m))).month)[1]
                                       for m in _mb.unique()}
                                _av["_un"] = (_av["vamp_count"].fillna(0).astype(float) / 30.4167) * _mb.map(_dm)
                                _act = _av.groupby("period")["_un"].sum().to_dict()
                                _act_raw = _av.groupby("period")["vamp_count"].sum().to_dict()
                                if "rpgt" in _av.columns:
                                    _av["_rpt"] = _av["rpgt"].astype(str).str.title()
                                    _act_v_rp = _av.groupby(["period", "_rpt"])["_un"].sum()
                                # Actual VAMP split by age t (same normalised basis
                                # as forecast VAMP_Post) for the T-stacked chart.
                                if "time_to_event_months" in _av.columns:
                                    _att = _av.copy()
                                    _att["_t"] = _att["time_to_event_months"].fillna(0).astype(int)
                                    _act_ts_df = (_att.groupby(["period", "_t"], as_index=False)["vamp_count"]
                                                  .sum().rename(columns={"_t": "t", "vamp_count": "VAMP"}))
                        # Actual transactions from the gateway-mapping cache (for the ratio).
                        _gm = os.path.join(_cache_dir, f"gateway_mapping_data_{_mv2}{_sfx}_fcp_v3.parquet")
                        _act_txn = {}
                        if os.path.exists(_gm):
                            _gmd = _c_read_parquet(_gm, _mtime(_gm)).copy()
                            # Mastercard mapping stores 'mastercard_trx_count'; alias to the Visa name
                            # ('visa_trx_count') the code below uses.
                            if "visa_trx_count" not in _gmd.columns and "mastercard_trx_count" in _gmd.columns:
                                _gmd = _gmd.rename(columns={"mastercard_trx_count": "visa_trx_count"})
                            for _cc in ("Company", "company"):
                                if _cmp2 and _cc in _gmd.columns:
                                    _gmd = _gmd[_gmd[_cc].astype(str).str.lower().str.strip() == str(_cmp2).lower().strip()]
                            if _f_rpgt != "(All)" and "rpgt" in _gmd.columns:
                                _gmd = _gmd[_gmd["rpgt"].astype(str).str.title() == _f_rpgt.title()]
                            if _f_bin != "(All)" and "bin" in _gmd.columns:
                                _gmd = _gmd[_gmd["bin"].astype(str).str.title() == _f_bin.title()]
                            # The transactions cache is at gatewayFid grain (no vampMid), so map
                            # gatewayFid -> vampMid (Master_MID_List) and filter to the selected
                            # MID — otherwise the ratio DENOMINATOR is the whole company's txns
                            # while the numerator (actual VAMP) is already vampMid-filtered, which
                            # made the actuals VAMP ratio far too low.
                            if _f_mid != "(All)" and "gatewayFid" in _gmd.columns:
                                _f2vg = {}
                                _mmp_g = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                                if os.path.exists(_mmp_g):
                                    try:
                                        _mmd_g = load_mid_list(_mmp_g)
                                        _cc3 = _norm_cols(_mmd_g)
                                        if _cc3.get("gatewayfid") and _cc3.get("vampmid"):
                                            _f2vg = _fid2vamp_from(_mmd_g, _cc3["gatewayfid"], _cc3["vampmid"])
                                    except Exception:  # noqa: BLE001
                                        _f2vg = {}
                                _gwlg = _gmd["gatewayFid"].astype(str).str.strip().str.lower()
                                _gmd = _gmd[_gwlg.map(_f2vg).fillna(
                                    _gmd["gatewayFid"].astype(str).str.strip()) == str(_f_mid)]
                            if "period" in _gmd.columns and "visa_trx_count" in _gmd.columns:
                                _act_txn = _gmd.groupby("period")["visa_trx_count"].sum().to_dict()
                                if "rpgt" in _gmd.columns:
                                    _gmd["_rpt"] = _gmd["rpgt"].astype(str).str.title()
                                    _act_t_rp = _gmd.groupby(["period", "_rpt"])["visa_trx_count"].sum()
                        for _p in range(3):
                            _td = _bd - pd.DateOffset(months=_p + 1)
                            _lab = _td.strftime("%m-%y")
                            _rows.append({"month": _lab, "order": -(_p + 1),
                                          "series": "Actual", "VAMP": float(_act.get(_p, 0.0))})
                            _at = float(_act_txn.get(_p, 0.0))
                            if _at > 0:
                                _ratio[_lab] = float(_act_raw.get(_p, 0.0)) / _at * 100.0
                        _cdf = pd.DataFrame(_rows).sort_values("order")
                        # Bar label: x.xk when >= 1,000, else the whole number.
                        _cdf["_lbl"] = _cdf["VAMP"].apply(
                            lambda v: f"{v/1000:.1f}k" if abs(v) >= 1000 else f"{v:,.0f}")
                        _order = _cdf["month"].drop_duplicates().tolist()
                        # Same bar formatting as the Transactions chart: one bar per
                        # month (each is either Actual OR Forecast), single trace so
                        # every bar is centred under its date label, low bargap.
                        _afont2 = dict(color='#0B1F3A', size=8, family="inherit")
                        # Two named traces (Actual / Forecast) so the chart shows a legend. Each month
                        # is either actual OR forecast, so the 'other' trace is 0 there → barmode=stack
                        # renders a single centred bar per month.
                        _is_act = (_cdf["series"] == "Actual").tolist()
                        _mon = _cdf["month"].tolist(); _vmp = _cdf["VAMP"].tolist(); _lbls = _cdf["_lbl"].tolist()
                        _fig = _gob.Figure()
                        _fig.add_trace(_gob.Bar(
                            x=_mon, y=[_v if _a else 0 for _v, _a in zip(_vmp, _is_act)],
                            name=_mc_hdr("Actual VAMP"), marker_color="#9AA8C0",
                            text=[_l if _a else "" for _l, _a in zip(_lbls, _is_act)],
                            textposition="inside", textfont=dict(size=9, color='#FFFFFF'), cliponaxis=False))
                        _fig.add_trace(_gob.Bar(
                            x=_mon, y=[_v if not _a else 0 for _v, _a in zip(_vmp, _is_act)],
                            name=_mc_hdr("Forecast VAMP"), marker_color="#e63748",
                            text=[_l if not _a else "" for _l, _a in zip(_lbls, _is_act)],
                            textposition="inside", textfont=dict(size=9, color='#FFFFFF'), cliponaxis=False))
                        
                        _bv = _cdf.loc[_cdf["VAMP"] > 0, "VAMP"]
                        _ylo = float(_bv.min()) * 0.8 if not _bv.empty else 0.0
                        _yhi = float(_cdf["VAMP"].max()) * 1.1 if _cdf["VAMP"].max() > 0 else 1.0
                        
                        # Calculate min/max bounds for the VAMP ratio axis (min - 20%)
                        _ratio_vals = [_ratio.get(_mo) for _mo in _order if _ratio.get(_mo) is not None]
                        _y2lo = max(0.0, float(min(_ratio_vals)) * 0.8) if _ratio_vals else 0.0
                        _y2hi = float(max(_ratio_vals)) * 1.1 if _ratio_vals else 100.0

                        _ratio_y = [_ratio.get(_mo) for _mo in _order]
                        _fig.add_trace(_gob.Scatter(
                            x=_order, y=_ratio_y, name=_mc_hdr("VAMP ratio"),
                            mode="lines+markers+text", yaxis="y2", connectgaps=True,
                            line=dict(color="#22C36B", width=2), marker=dict(size=5),
                            text=[(f"{_v:.1f}%" if _v is not None else "") for _v in _ratio_y],
                            textposition="top center", textfont=dict(size=8, color="#22C36B")))
                        
                        # Left VAMP axis shown as x.xxk (thousands, 2 dp) via explicit k-scaled ticks.
                        _yticks = list(np.linspace(_ylo, _yhi, 5))
                        # Header removed above the chart → give the space back to the plot.
                        _fig.update_layout(
                            height=270, margin=dict(l=35, r=35, t=22, b=4), bargap=0.08, barmode="stack",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#0B1F3A', family="inherit"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                        font=dict(color='#0B1F3A', size=8), title_text=None),
                            yaxis=dict(range=[_ylo, _yhi], showgrid=True, gridcolor='lightgrey', tickfont=_afont2,
                                       title=None, tickmode="array", tickvals=_yticks,
                                       ticktext=[f"{_v/1000:.2f}k" for _v in _yticks]),
                            yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=_afont2, ticksuffix="%", range=[_y2lo, _y2hi]))
                        
                        _fig.update_xaxes(type="category", categoryorder="array", categoryarray=_order,
                                          showgrid=False, tickfont=_afont2, title=None)
                        _n_act = int(_cdf[_cdf["order"] < 0]["month"].nunique())
                        if 0 < _n_act < len(_order):
                            _fig.add_vline(x=_n_act - 0.5, line_width=2, line_dash="dot", line_color="#555")
                        # (_fig is rendered below, beside the Transactions chart.)

                        # ---- Transactions by month: actuals (gateway-mapping) → forecast post.
                        # One bar per month (each month is either Actual OR Forecast),
                        # so a single trace keeps every bar centred under its date
                        # label; a low bargap makes the bars wider / closer together.
                        _txr = []
                        for _m in range(6):
                            _lab = (_bd + pd.DateOffset(months=_m)).strftime("%m-%y")
                            _txr.append({"month": _lab, "order": _m, "series": "Forecast Post",
                                         "Txn": float(_post_txn.get(_m, 0.0))})
                        for _p in range(3):
                            _lab = (_bd - pd.DateOffset(months=_p + 1)).strftime("%m-%y")
                            _txr.append({"month": _lab, "order": -(_p + 1), "series": "Actual",
                                         "Txn": float(_act_txn.get(_p, 0.0))})
                        _txdf = pd.DataFrame(_txr).sort_values("order")
                        _txo = _txdf["month"].drop_duplicates().tolist()
                        _txdf["_lbl"] = _txdf["Txn"].apply(lambda v: f"{v/1000:.1f}k" if abs(v) >= 1000 else f"{v:,.0f}")
                        # Dynamic y-axis min/max (like the VAMP chart) so month-to-month
                        # variation is visible rather than dwarfed by a 0-based axis.
                        _txv = _txdf.loc[_txdf["Txn"] > 0, "Txn"]
                        _txlo = float(_txv.min()) * 0.9 if not _txv.empty else 0.0
                        _txhi = float(_txdf["Txn"].max()) * 1.1 if _txdf["Txn"].max() > 0 else 1.0
                        # Two named traces (Actual / Forecast) → legend; one is 0 per month so
                        # barmode=stack shows a single centred bar.
                        _tx_act = (_txdf["series"] == "Actual").tolist()
                        _txm = _txdf["month"].tolist(); _txv2 = _txdf["Txn"].tolist(); _txl = _txdf["_lbl"].tolist()
                        _txfig = _gob.Figure()
                        _txfig.add_trace(_gob.Bar(
                            x=_txm, y=[_v if _a else 0 for _v, _a in zip(_txv2, _tx_act)],
                            name="Actual Txns", marker_color="#9AA8C0",
                            text=[_l if _a else "" for _l, _a in zip(_txl, _tx_act)],
                            textposition="inside", textfont=dict(size=9, color='#FFFFFF'), cliponaxis=False))
                        _txfig.add_trace(_gob.Bar(
                            x=_txm, y=[_v if not _a else 0 for _v, _a in zip(_txv2, _tx_act)],
                            name="Forecast Txns", marker_color="#e63748",
                            text=[_l if not _a else "" for _l, _a in zip(_txl, _tx_act)],
                            textposition="inside", textfont=dict(size=9, color='#FFFFFF'), cliponaxis=False))
                        # VAMP ratio line on the right axis — same format as the VAMP chart above.
                        _txratio_y = [_ratio.get(_mo) for _mo in _txo]
                        _txfig.add_trace(_gob.Scatter(
                            x=_txo, y=_txratio_y, name=_mc_hdr("VAMP ratio"),
                            mode="lines+markers+text", yaxis="y2", connectgaps=True,
                            line=dict(color="#22C36B", width=2), marker=dict(size=5),
                            text=[(f"{_v:.1f}%" if _v is not None else "") for _v in _txratio_y],
                            textposition="top center", textfont=dict(size=8, color="#22C36B")))
                        # Header removed above the chart → give the space back to the plot.
                        _txfig.update_layout(
                            height=270, margin=dict(l=35, r=35, t=22, b=4), bargap=0.08, barmode="stack",
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#0B1F3A', family="inherit"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                        font=dict(color='#0B1F3A', size=8), title_text=None),
                            yaxis=dict(range=[_txlo, _txhi], showgrid=True, gridcolor='lightgrey',
                                       tickfont=_afont2, title=None),
                            yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=_afont2,
                                        ticksuffix="%", range=[_y2lo, _y2hi]))
                        _txfig.update_xaxes(type="category", categoryorder="array", categoryarray=_txo,
                                            showgrid=False, tickfont=_afont2, title=None)
                        if 0 < _n_act < len(_txo):
                            _txfig.add_vline(x=_n_act - 0.5, line_width=2, line_dash="dot", line_color="#555")

                        # ---- New: VAMP-by-RPGT & Transactions-by-RPGT STACKED bars. Actual
                        # months stay single grey bars (actuals aren't RPGT-grained); forecast
                        # months are stacked by RPGT.
                        # On-brand, distinct palette for RPGT bands (red / ink / green anchors +
                        # complementary tones) — cohesive with the rest of the UI but tellable apart.
                        _RPGT_PAL = ["#e63748", "#0B1F3A", "#22C36B", "#3B6EA5", "#F59E0B",
                                     "#7A4EA3", "#1F9D8F", "#C77DFF", "#9AA8C0", "#D98324"]
                        _vamp_fc = (_fp.groupby(["period", "RPGT"])["VAMP_Post"].sum()
                                    if "RPGT" in _fp.columns else None)
                        _txn_fc = (_fp.groupby(["period", "RPGT"])["VI_Txn_Post"].sum()
                                   if "RPGT" in _fp.columns else None)

                        # [FN-381]
                        def _stacked_rpgt_fig(_fc, _act_rp, _order_labels, pct=False):
                            # One trace per RPGT across BOTH actual and forecast months (same colour
                            # per RPGT across the divider). pct=True → 100% stacked (share) bars.
                            # Per-segment labels (value, or share% for pct) hidden below 7% of the bar.
                            _figs = _gob.Figure()
                            _lab2fm = {(_bd + pd.DateOffset(months=_m)).strftime("%m-%y"): _m for _m in range(6)}
                            _lab2am = {(_bd - pd.DateOffset(months=_p + 1)).strftime("%m-%y"): _p for _p in range(3)}
                            _fc_d = {(int(_p), str(_r).title()): float(_v) for (_p, _r), _v in _fc.items()} if _fc is not None else {}
                            _ac_d = {(int(_p), str(_r).title()): float(_v) for (_p, _r), _v in _act_rp.items()} if _act_rp is not None else {}
                            _segs = sorted(set(k[1] for k in _fc_d) | set(k[1] for k in _ac_d))
                            _ymap = {}
                            for _rp in _segs:
                                _yv = []
                                for _l in _order_labels:
                                    if _l in _lab2fm:
                                        _yv.append(_fc_d.get((_lab2fm[_l], _rp), 0.0))
                                    elif _l in _lab2am:
                                        _yv.append(_ac_d.get((_lab2am[_l], _rp), 0.0))
                                    else:
                                        _yv.append(0.0)
                                _ymap[_rp] = _yv
                            # Highest-total RPGT first → sits at the BOTTOM of the stack (first trace).
                            _segs = sorted(_segs, key=lambda _s: sum(_ymap[_s]), reverse=True)
                            _mtot = [sum(_ymap[_s][_j] for _s in _segs) for _j in range(len(_order_labels))]
                            for _i, _rp in enumerate(_segs):
                                _yv = _ymap[_rp]
                                _txt = []
                                for _j, _v in enumerate(_yv):
                                    _tot = _mtot[_j]
                                    if _tot <= 0 or _v <= 0 or (_v / _tot) < 0.07:
                                        _txt.append("")
                                    elif pct:
                                        _txt.append(f"{_v / _tot * 100:.0f}%")
                                    else:
                                        _txt.append(f"{_v/1000:.1f}k" if _v >= 1000 else f"{_v:,.0f}")
                                _figs.add_trace(_gob.Bar(
                                    x=_order_labels, y=_yv, name=str(_rp),
                                    marker_color=_RPGT_PAL[_i % len(_RPGT_PAL)],
                                    text=_txt, texttemplate="%{text}", textposition="inside",
                                    insidetextanchor="middle", textfont=dict(size=7, color="#FFFFFF"),
                                    cliponaxis=False))
                            # Count charts: y-axis floor = the MAX-total RPGT's MINIMUM plotted (non-zero)
                            # monthly value − 20%; ceiling = the tallest stacked total + 5%. The
                            # 100%-stacked (pct) chart keeps its full 0–100% range.
                            _mmax_rp = max(_mtot) if _mtot else 0.0
                            _yaxd = dict(showgrid=True, gridcolor='lightgrey', tickfont=_afont2, title=None,
                                         ticksuffix=("%" if pct else None))
                            if (not pct) and _segs and _mmax_rp > 0:
                                _topseg_vals = [_v for _v in _ymap[_segs[0]] if _v > 0]   # biggest RPGT's monthly values
                                _minv_top = min(_topseg_vals) if _topseg_vals else 0.0
                                if _minv_top > 0:
                                    _yaxd["range"] = [0.8 * _minv_top, _mmax_rp * 1.05]
                            _figs.update_layout(
                                # t=96 reserves enough room ABOVE the plot for the (wrapping) RPGT
                                # legend so it isn't clipped and never overflows onto the bars; the
                                # dotted actual/forecast divider spans only the plot domain (below the
                                # legend), so it no longer crosses the legend text either.
                                height=439, margin=dict(l=35, r=45, t=96, b=4), barmode="stack",
                                barnorm=("percent" if pct else None), bargap=0.08,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#0B1F3A', family="inherit"),
                                legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                                            font=dict(color='#0B1F3A', size=7), title_text=None,
                                            traceorder="normal"),   # highest first; wraps to as many
                                                                    # lines as needed (extra top margin)
                                yaxis=_yaxd)
                            _figs.update_xaxes(type="category", categoryorder="array",
                                               categoryarray=_order_labels, showgrid=False,
                                               tickfont=_afont2, title=None)
                            if 0 < _n_act < len(_order_labels):
                                _figs.add_vline(x=_n_act - 0.5, line_width=2, line_dash="dot", line_color="#555")
                            return _figs

                        _vamp_stack_fig = _stacked_rpgt_fig(_vamp_fc, _act_v_rp, _order)
                        _vamp_pct_fig = _stacked_rpgt_fig(_vamp_fc, _act_v_rp, _order, pct=True)
                        _txn_stack_fig = _stacked_rpgt_fig(_txn_fc, _act_t_rp, _txo)
                        _txn_pct_fig = _stacked_rpgt_fig(_txn_fc, _act_t_rp, _txo, pct=True)

                        # 3 columns: absolute stacked | 100% stacked | existing (+ VAMP ratio).
                        # Headers removed; that space is given back to the (taller) charts.
                        # Tighten the vertical gap between the stacked charts.
                        st.markdown("""<style>
                            [data-testid="stPlotlyChart"] { margin-bottom: 0.15rem !important; }
                        </style>""", unsafe_allow_html=True)
                        # Same column layout as the 3-table row so _c3 lines up with (and matches
                        # the width of) the 3rd table / +ratio charts above it.
                        if mode == "detail":
                            _cc = st.columns([10, 0.36, 10, 0.36, 10], gap="small")
                            _c1, _c2, _c3 = _cc[0], _cc[2], _cc[4]
                            with _c1:
                                st.plotly_chart(_vamp_stack_fig, use_container_width=True)
                                st.plotly_chart(_txn_stack_fig, use_container_width=True)
                            with _c2:
                                st.plotly_chart(_vamp_pct_fig, use_container_width=True)
                                st.plotly_chart(_txn_pct_fig, use_container_width=True)
                            _c3_slot = _c3.container()   # T-stacked charts render here (below)
                        else:
                            # Impact tab: VAMP + Transactions bar charts (with the VAMP ratio line),
                            # side by side with the table (each in its own column of the row above).
                            _ts_slot.plotly_chart(_fig, use_container_width=True)
                            _ts_slot2.plotly_chart(_txfig, use_container_width=True)


                        # VAMP age (T-stacked): VAMP by month, stacked by age t.
                        #   * actuals lead into the forecast (divider between them),
                        #   * age t is grouped past a cap into a single "T{cap}+" band,
                        #   * bands are shades of one colour (blue ramp) ordered by age,
                        #   * a weighted-avg-T line (uncapped t) rides the right axis.
                        _TCAP = 4
                        _ts_rows = []
                        _tsf = _fp[["period", "t", "VAMP_Post"]].copy()
                        _tsf = _tsf[(_tsf["period"] >= 0) & (_tsf["period"] <= 5)]
                        for _, _r in _tsf.iterrows():
                            _ts_rows.append({"month": (_bd + pd.DateOffset(months=int(_r["period"]))).strftime("%m-%y"),
                                             "order": int(_r["period"]), "t": int(_r["t"]),
                                             "VAMP": float(_r["VAMP_Post"])})
                        if _act_ts_df is not None and not _act_ts_df.empty:
                            for _, _r in _act_ts_df.iterrows():
                                _p = int(_r["period"])
                                if _p > 2:            # only the 3 actual months leading into the
                                    continue          # forecast, matching the Transactions chart
                                _ts_rows.append({"month": (_bd - pd.DateOffset(months=_p + 1)).strftime("%m-%y"),
                                                 "order": -(_p + 1), "t": int(_r["t"]),
                                                 "VAMP": float(_r["VAMP"])})
                        _tsd = pd.DataFrame(_ts_rows)
                        if not _tsd.empty and _tsd["VAMP"].abs().sum() > 0:
                            # Weighted-avg T from the uncapped ages.
                            _wser = _tsd.groupby("order").apply(
                                lambda d: (d["t"] * d["VAMP"]).sum() / max(d["VAMP"].sum(), 1e-9))
                            # Cap age for display and label the top band "T{cap}+".
                            _tsd["tc"] = _tsd["t"].clip(upper=_TCAP)
                            _tsd["tlab"] = _tsd["tc"].apply(lambda x: f"T{_TCAP}+" if int(x) >= _TCAP else f"T{int(x)}")
                            _tsg = _tsd.groupby(["month", "order", "tc", "tlab"], as_index=False)["VAMP"].sum()
                            _mo = _tsg.sort_values("order").drop_duplicates("month")
                            _tso = _mo["month"].tolist()
                            _orders = _mo["order"].tolist()
                            # Age ramp anchored on the SAME grey (#9AA8C0) and red (#e63748)
                            # as the Transactions chart — light shade (young t) → full colour
                            # (old t), so actuals read grey and forecast red while the stacked
                            # age bands stay distinguishable.
                            import plotly.colors as _pcol
                            _tvals = sorted(_tsg["tc"].unique())
                            _labfor = lambda v: (f"T{_TCAP}+" if int(v) >= _TCAP else f"T{int(v)}")
                            # Full 0→1 stop range + darker endpoints for MAXIMUM contrast between
                            # adjacent age bands (still grey for actuals, red for forecast).
                            _stops = ([i / max(len(_tvals) - 1, 1) for i in range(len(_tvals))]
                                      if len(_tvals) > 1 else [0.6])
                            _ramp_act = _pcol.sample_colorscale([[0.0, "#D3DCEA"], [1.0, "#1B2740"]], _stops)
                            _ramp_fc  = _pcol.sample_colorscale([[0.0, "#F6A9B2"], [1.0, "#7A0E17"]], _stops)
                            
                            # Per-month stack total → hide labels on thin segments (< 7% of the
                            # bar) so the chart stays readable.
                            _mtot = _tsg.groupby("month")["VAMP"].sum().to_dict()
                            _n_act_ts = int((pd.Series(_orders) < 0).sum())

                            # [FN-382]
                            def _build_tsfig(pct=False):
                                # pct=True → 100% stacked (age-band share per month); labels become %.
                                _f = _gob.Figure()
                                for i, v in enumerate(_tvals):
                                    _t_data = _tsg[_tsg["tc"] == v]
                                    _y_vals, _c_vals, _txt = [], [], []
                                    for mo, order in zip(_tso, _orders):
                                        _match = _t_data[_t_data["month"] == mo]
                                        _yv = float(_match["VAMP"].sum()) if not _match.empty else 0.0
                                        _y_vals.append(_yv)
                                        # Order < 0 = actuals (greys), >= 0 = forecast (reds).
                                        _c_vals.append(_ramp_fc[i] if order >= 0 else _ramp_act[i])
                                        _tot = float(_mtot.get(mo, 0.0))
                                        if _tot <= 0 or _yv <= 0 or (_yv / _tot) < 0.07:
                                            _txt.append("")
                                        elif pct:
                                            _txt.append(f"{_yv / _tot * 100:.0f}%")
                                        else:
                                            _txt.append(f"{_yv/1000:.1f}k" if _yv >= 1000 else f"{_yv:,.0f}")
                                    _lab_col = "#0B1F3A" if (i < len(_stops) and _stops[i] < 0.5) else "#FFFFFF"
                                    _f.add_trace(_gob.Bar(
                                        x=_tso, y=_y_vals, name=_labfor(v), marker_color=_c_vals,
                                        text=_txt, texttemplate="%{text}", textposition="inside",
                                        insidetextanchor="middle", textfont=dict(size=8, color=_lab_col),
                                        cliponaxis=False))
                                _wtt = [float(_wser.get(o, 0.0)) for o in _orders]
                                # Right y-axis minimum = smallest PLOTTED line value − 20% (positive
                                # values only, so a 0/empty order doesn't pin the axis to zero).
                                _wtt_pos = [v for v in _wtt if v > 0]
                                _wtt_lo = (min(_wtt_pos) * 0.8) if _wtt_pos else 0.0
                                _wtt_hi = (max(_wtt) * 1.1) if _wtt else 1.0
                                _f.add_trace(_gob.Scatter(
                                    x=_tso, y=_wtt, name=_mc_hdr("Avg Months to VAMP"), mode="lines+markers+text", yaxis="y2",
                                    line=dict(color="#22C36B", width=2), marker=dict(size=4),
                                    text=[f"{_v:.1f}" for _v in _wtt], textposition="top center",
                                    textfont=dict(size=8, color="#22C36B")))
                                _f.update_layout(height=439, margin=dict(l=35, r=45, t=28, b=10), barmode="stack",
                                                 barnorm=("percent" if pct else None),
                                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                                 font=dict(color='#0B1F3A', family="inherit"),
                                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                                             font=dict(color='#0B1F3A', size=8), title_text=None),
                                                 yaxis=dict(showgrid=True, gridcolor='lightgrey', tickfont=_afont2, title=None,
                                                            ticksuffix=("%" if pct else None)),
                                                 yaxis2=dict(overlaying="y", side="right", showgrid=False, tickfont=_afont2, range=[_wtt_lo, _wtt_hi]))
                                _f.update_xaxes(type="category", showgrid=False, tickfont=_afont2, title=None)
                                if 0 < _n_act_ts < len(_tso):
                                    _f.add_vline(x=_n_act_ts - 0.5, line_width=2, line_dash="dot", line_color="#555")
                                return _f

                            # Absolute + 100%-stacked versions (headerless), height aligned with RPGT charts.
                            if mode == "detail":
                                _c3_slot.plotly_chart(_build_tsfig(pct=False), use_container_width=True)
                                _c3_slot.plotly_chart(_build_tsfig(pct=True), use_container_width=True)





        with _t_risk:
            # -------------- Forecast VAMP impact of the proposed split (M0-5) --------
            # Renders into the slot reserved ABOVE the Bank Impact section.
            with st.container(border=True):
                tp_path = _scheme_norm_export(out_dir, "vamp_t_period_export.csv")
                pp_path = _scheme_norm_export(out_dir, "vamp_t_period_prorata_export.csv")
                _gr_shared = None   # always defined (used by the pre/post render-call guards)
                split_now = ss.get("impact_split", ss.get("split"))   # follows the impact-basis toggle
                if not os.path.exists(tp_path) and not os.path.exists(pp_path):
                    st.caption("No VAMP export found in the pipeline outputs.")
                elif split_now is None or getattr(split_now, "empty", True):
                    st.caption("No proposed split yet — pick a variation above.")
                else:
                    mm_path = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    fid2vamp = {}
                    if os.path.exists(mm_path):
                        _mm = load_mid_list(mm_path)
                        _cc = _norm_cols(_mm)
                        _gcol, _vcol = _cc.get("gatewayfid"), _cc.get("vampmid")
                        if _gcol and _vcol:
                            fid2vamp = _fid2vamp_from(_mm, _gcol, _vcol)
                    sp = split_now.copy()
                    sp["_vm"] = sp["gateway"].astype(str).str.strip().str.lower().map(fid2vamp)
                    sp = sp.dropna(subset=["_vm"])
                    # At Bank×Currency×RPGT grain the split has a distinct share per RPGT, so the
                    # projection must be fed the per-RPGT shares (5-tuples) to reflect that a MID
                    # was moved on one RPGT only. `prop_items_flat` (4-tuples, collapsed) is kept
                    # for the compute_vamp_post_by_mid fallback, which is RPGT-agnostic.
                    _disp_by_rpgt = bool(ss.get("opt_by_rpgt", False)) and "rpgt" in sp.columns
                    if _disp_by_rpgt:
                        sp = sp.drop_duplicates(["currency", "bin", "rpgt", "gateway"])
                        _pdf = sp.groupby(["currency", "bin", "rpgt", "_vm"], as_index=False)["share"].sum()
                        prop_items = tuple((str(c).lower(), str(b), str(rp), str(v), float(s))
                                           for c, b, rp, v, s in
                                           _pdf[["currency", "bin", "rpgt", "_vm", "share"]].itertuples(index=False))
                        _pdf_flat = sp.groupby(["currency", "bin", "_vm"], as_index=False)["share"].sum()
                        prop_items_flat = tuple((str(c).lower(), str(b), str(v), float(s))
                                                for c, b, v, s in _pdf_flat[["currency", "bin", "_vm", "share"]].itertuples(index=False))
                    else:
                        sp = sp.drop_duplicates(["currency", "bin", "gateway"])
                        prop_df = sp.groupby(["currency", "bin", "_vm"], as_index=False)["share"].sum()
                        prop_items = tuple((str(c).lower(), str(b), str(v), float(s))
                                           for c, b, v, s in prop_df[["currency", "bin", "_vm", "share"]].itertuples(index=False))
                        prop_items_flat = prop_items

                    fs_cfg = ss.get("forecast_settings", {})
                    try:
                        _m0 = pd.to_datetime(fs_cfg.get("month_0", date.today().replace(day=1)))
                    except Exception:
                        _m0 = pd.to_datetime(date.today().replace(day=1))
                    _gl = ss.get("split_go_live_date", date.today())

                    if not prop_items:
                        st.warning("Could not map any proposed-split gateways to vampMids "
                                   "(check Master_MID_List). Showing baseline only.")

                    # vampMids fully switched off in gateway_volume_overrides (target=0,
                    # trx/both) — excluded from the post projection. A vampMid counts as
                    # off only if EVERY gatewayFid mapping to it is switched off.
                    from routing_optimiser.s2_forecast.vamp_forecast_pipeline import _canonical_gateway
                    # [FN-383]
                    def _normfid(x):
                        return str(_canonical_gateway(x)).strip().lower()
                    _ovr = ss.get("gateway_volume_overrides") or {}
                    _off_fids = set()
                    _fid_eff = {}
                    for _gwid, _cfg in (_ovr.items() if isinstance(_ovr, dict) else []):
                        if isinstance(_cfg, dict):
                            _tgt = pd.to_numeric(_cfg.get("target"), errors="coerce")
                            _ap = str(_cfg.get("apply_to", "")).strip().lower()
                            if _tgt == 0 and _ap in ("trx", "both"):
                                _off_fids.add(_normfid(_gwid))
                                if _cfg.get("effective_date"):
                                    _fid_eff[_normfid(_gwid)] = str(_cfg.get("effective_date"))
                    _vamp2fids = {}
                    for _f, _v in fid2vamp.items():
                        _vamp2fids.setdefault(_v, set()).add(_normfid(_f))
                    excluded_mids = frozenset(
                        v for v, fids in _vamp2fids.items() if fids and fids <= _off_fids)
                    # 19cv — apply_to:"vamp" (target 0). Same all-fids-off rule as excluded_mids,
                    # but a DIFFERENT consequence: these MIDs keep their transactions and are barred
                    # only from RECEIVING redistributed VAMP. The baseline already zeroes the VAMP
                    # they hold; this closes the flow half.
                    # 19da — the SAME capability the search is given (tab_2_routing_engine builds it from
                    # the same restrictions file and brand), so both sides complete the aged frame
                    # identically. `_cap_sig` carries its identity into the st.cache_data key,
                    # because the callable itself is excluded from the hash.
                    import io as _io
                    import json as _je
                    import impact_calcs as _ic_cap
                    _capability = None
                    _cap_sig = "off"
                    if os.environ.get("ROUTING_INJECT_CAPABLE", "1") != "0":
                        try:
                            _rjp = input_json_path("routing_restrictions.json")
                            _rj = {}
                            if os.path.exists(_rjp):
                                with _io.open(_rjp, encoding="utf-8") as _rfh:
                                    _rj = _je.load(_rfh)
                            # 19ea: the brand comes from forecast_settings. ss.get("company")
                            # is None, which widened this from 38 gateways to 113 across 27
                            # brands and injected a recipient row for every one of them.
                            _capability = _ic_cap.build_capability(restrictions=_rj,
                                                                   brand=run_company(ss))
                            _cap_sig = (f"{_capability.n_gateways}:{_capability.n_mids}:"
                                        f"{_mtime(_rjp) if os.path.exists(_rjp) else 0:.0f}")
                        except Exception:  # noqa: BLE001
                            _capability, _cap_sig = None, "off"
                    # 19dw: the SHARED builder, so tab 3 and the engine cannot drift on the
                    # all-fids-off rule or on gateway-id canonicalisation. Same value as the
                    # inline version it replaces, so the cache key is unchanged.
                    _vamp_off_mids = _ic_cap.build_vamp_off_mids(fid2vamp, _ovr)
                    _bad_ap = _unknown_apply_to(_ovr if isinstance(_ovr, dict) else {})
                    if _bad_ap:
                        st.warning(
                            "gateway_volume_overrides.json contains apply_to value(s) "
                            f"{sorted(_bad_ap)} that are not one of trx / vamp / both / "
                            "inject_from_siblings — those entries are IGNORED, they are not "
                            "applied to anything.")
                    # Effective-date-gated switch-off: only remove a switched-off vampMid
                    # from its effective month onward (mid-month pro-rated), not from M0.
                    _kill_eff = build_kill_eff(_vamp2fids, _fid_eff)
                    _m0s = str(_m0.date())

                    # CONSOLIDATION: compute the granular pro-rata projection ONCE and derive the
                    # per-MID VAMP table from it (mid_table_from_granular is numerically identical
                    # to _c_vamp_post_prorata). The same _gr_shared frame is reused by the
                    # filterable detail table below, so the Impact tab runs one projection here
                    # instead of two. Falls back to the non-pro-rata path when pp is missing.
                    _gr_shared = None
                    _wc0 = ss.get("wallet_ctx") or {}
                    _wcin = frozenset(str(x).strip().lower() for x in (_wc0.get("incapable") or set()))
                    _uonly = frozenset(str(x).strip().lower() for x in (_wc0.get("usa_only") or set()))
                    # ENFORCED shares (post cap / wallet / USA-Non-USA) so the projection routes
                    # exactly where the deployed config would. It no longer "reproduces the
                    # pipeline's back-fill gateways (WoodForest/Authorize)" — that back-fill was
                    # invented share and is deleted; those MIDs now appear only where the optimiser
                    # actually put them.
                    # build_split_exports is a bit heavy, so cache per (dial, basis, go-live).
                    _proj_prop = prop_items
                    try:
                        _brand_ep = str((ss.get("forecast_settings", {}) or {}).get("company", "TotalAV"))
                        _ep_key = (round(float(picked_w), 4), bool(_basis_compressed), str(_gl))
                        _ep_cache = ss.get("_enf_prop_cache") or {}
                        if _ep_cache.get("key") == _ep_key and _ep_cache.get("val"):
                            _proj_prop = _ep_cache["val"]
                        elif split_now is not None and not getattr(split_now, "empty", True):
                            _ep = enforced_prop_items(
                                split_now, _brand_ep, str(_gl),
                                wallet_incapable=set(_wc0.get("incapable", set())),
                                fid2vamp=_wc0.get("fid2vamp"),
                                mid_list_path=os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                                usa_only=set(_wc0.get("usa_only", set())),
                                country_pres=_wc0.get("country_pres", {}),
                                max_share=float(_wc0.get("max_share", 0.97)))
                            if _ep:
                                _proj_prop = _ep
                                ss["_enf_prop_cache"] = {"key": _ep_key, "val": _ep}
                    except Exception:  # noqa: BLE001
                        _proj_prop = prop_items   # fall back to the raw split on any failure
                    # BACKUP-BLEND: fold the backup files' catch-all (BIN=Other) re-adds into the
                    # proposed shares so the projection matches what the pipeline ACTUALLY routes
                    # (tab 5) — e.g. Braintree re-added at 10.6% where the split zeroed it. No-op
                    # unless a backup folder is set on tab 1. Kill-switch: ROUTING_BACKUP_BLEND=0.
                    _bcatch = ss.get("backup_catchall") or {}
                    if _bcatch and _proj_prop and os.environ.get("ROUTING_BACKUP_BLEND", "1") != "0":
                        try:
                            from routing_optimiser.s5_deliver.backup_blend import blend_prop_items as _bpi
                            _proj_prop = _bpi(_proj_prop, _bcatch, fid2vamp)
                        except Exception:  # noqa: BLE001
                            pass   # any failure → keep the un-blended enforced split
                    # Exploration floor for the projection (replicates the engine's per-profile floor so
                    # 0%-rule incumbents keep >= floor). Kill-switch: ROUTING_PROJ_FLOOR=0 disables it
                    # (to compare against the old flat-rule projection). Default = the run's floor.
                    _proj_floor = (0.0 if os.environ.get("ROUTING_PROJ_FLOOR", "0") == "0"
                                   else float(ss.get("exploration_floor", 0.0) or 0.0))
                    _wcp0, _uop0, _ = _cap_pairs(
                        os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                        input_json_path("routing_restrictions.json"))
                    if os.path.exists(pp_path):
                        _gr_shared = _c_prepost_granular(pp_path,
                                                         projection_cache_sig(pp_path, _proj_prop, _proj_floor,
                                                                              wallet_incapable_pairs=_wcp0,
                                                                              usa_only_pairs=_uop0),
                                                         _proj_prop, excluded_mids,
                                                         _kill_eff, _m0s, _scoped_rpgts, _wcin, _uonly,
                                                         exploration_floor=_proj_floor,
                                                         wallet_incapable_pairs=_wcp0,
                                                         usa_only_pairs=_uop0,
                                                         vamp_off_mids=_vamp_off_mids,
                                                         cap_sig=_cap_sig,
                                                         _capability=_capability,
                                                         # 19df — delivery gets the SAME max-share
                                                         # cap the search applies. `_wc0` already
                                                         # hands this exact value to the enforcement
                                                         # call at :4182, so the two now agree by
                                                         # construction rather than by luck.
                                                         max_share=float(_wc0.get("max_share", 0.97)))
                        # 19dx: keep every Total AV vampMid, active or not (Cardworks, EPX,
                        # Merrick and Bancard have real historic VAMP), and drop only
                        # off-brand rows that are entirely zero.
                        _brand_mids = _ic_cap.brand_vamp_mids(
                            locals().get("_mid_list_e") or os.path.join(
                                PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                            run_company(ss))
                        # 19ea — FAIL LOUD. An empty set means the brand matched nothing, and
                        # the filter's own rule is then to keep every row. That is the right
                        # default (never hide a real row) but it must not be SILENT: it looked
                        # exactly like the filter working on a book with no other brands.
                        if not _brand_mids:
                            st.warning(
                                f"Brand filter unavailable — no vampMid in the Master MID List "
                                f"matches '{run_company(ss)}'. Every row is shown, including "
                                "other brands. Check the brand column spelling.")
                        vp = mid_table_from_granular(_gr_shared, keep_mids=_brand_mids)
                    else:
                        vp = compute_vamp_post_by_mid(tp_path, prop_items_flat, str(_m0.date()), str(_gl),
                                                      excluded_mids, _kill_eff, _mtime(tp_path))

                    # VALIDATE MODE ONLY: build the per-MID table from the pipeline's granular
                    # bin_rpgt_impact_export.csv (the SAME source the Financial Impact table uses
                    # below), so both tables — and the Validate Split table — tie exactly. The
                    # reconciliation guard confirms it aggregates to mid_level. Engine flow untouched.
                    if ss.get("variations_engine") == "validate":
                        _gp_risk = _validate_granular_from_bin_rpgt(out_dir)
                        if _gp_risk is not None and not _gp_risk.empty:
                            vp = mid_table_from_granular(_gp_risk,
                                                         keep_mids=locals().get("_brand_mids"))
                    vp = vp.sort_values("VAMP M0", ascending=False)

                    col_groups, cols = [], ["vampMid"]
                    for m in range(6):
                        grp = [f"VAMP M{m}", f"VI Txn M{m}", f"VAMP Post M{m}", f"VI Txn Post M{m}"]
                        cols.extend(grp)
                        col_groups.append(grp)
                    total = {"vampMid": "TOTAL"}
                    for c in cols:
                        if c != "vampMid":
                            total[c] = vp[c].sum()
                    vp_view = pd.concat([vp[cols], pd.DataFrame([total])], ignore_index=True)

                    html = ['<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow-x:auto; width:100%; background-color:var(--tav-card); border:1px solid var(--tav-line);">']
                    html.append('<table style="width:100%; border-collapse:collapse; font-family:inherit; font-size:0.68rem; line-height:1.1;"><tr>')
                    html.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; text-align:left; position:sticky; left:0; width:1%; white-space:nowrap;">{_mc_hdr("vampMid")}</th>')
                
                    # Reduced spacing from 24px to 12px
                    html.append('<th style="background-color:var(--tav-card); border:none; width:8px; min-width:8px; padding:0;"></th>')
                
                    for gi, grp in enumerate(col_groups):
                        for c in grp:
                            html.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; text-align:right; white-space:nowrap; width:1%;">{_mc_hdr(c)}</th>')
                        # Reduced spacing from 24px to 12px
                        html.append('<th style="background-color:var(--tav-card); border:none; width:8px; min-width:8px; padding:0;"></th>')
                    html.append('</tr>')
                
                    for _, r in vp_view.iterrows():
                        is_total = (r["vampMid"] == "TOTAL")
                        tb = "border-top:2px solid var(--tav-line);" if is_total else ""
                        wt = "800" if is_total else "normal"
                    
                        _bgmap = {}
                        if not is_total:
                            for c in cols:
                                if c.startswith("VAMP"):
                                    _txn = c.replace("VAMP", "VI Txn")
                                    _vv = float(r[c]); _tt = float(r[_txn]) if _txn in r.index else 0.0
                                    _rt = (_vv / _tt) if _tt > 0 else 0.0
                                    if _rt > 0.015 and _vv > 1500:
                                        _bgmap[c] = _bgmap[_txn] = "background-color:rgba(230,55,72,0.30);"
                                    elif _rt > 0.012 and _vv > 1200:
                                        _bgmap[c] = _bgmap[_txn] = "background-color:rgba(245,158,11,0.38);"
                    
                        html.append('<tr>')
                        html.append(f'<td style="padding:2px 8px; text-align:left; color:#000000; font-weight:{"800" if is_total else "600"}; {tb} position:sticky; left:0; background-color:var(--tav-card); width:1%; white-space:nowrap;">{r["vampMid"]}</td>')
                    
                        # Reduced spacing from 24px to 12px
                        html.append(f'<td style="width:8px; min-width:8px; padding:0; {tb}"></td>')
                    
                        for grp in col_groups:
                            for c in grp:
                                ital = "font-style:italic;" if "Post" in c else ""
                                _bg = _bgmap.get(c, "")
                                html.append(f'<td style="padding:2px 6px; text-align:right; color:#000000; font-weight:{wt}; {ital} {_bg} {tb} white-space:nowrap; width:1%;">{r[c]:,.0f}</td>')
                        
                            # Reduced spacing from 24px to 12px
                            html.append(f'<td style="width:8px; min-width:8px; padding:0; {tb}"></td>')
                        html.append('</tr>')
                    html.append('</table></div>')
                    st.markdown("".join(html), unsafe_allow_html=True)

                    # ---- Per-MID constraint check (projected vs target for THIS dial split) ----
                    _mid_rules = ss.get("mid_constraints") or []
                    if _mid_rules:
                        _vpi = vp.copy()
                        _vpi["_k"] = _vpi["vampMid"].astype(str).str.strip().str.lower()
                        _vpi = _vpi.set_index("_k")
                        _mlabel = {"txn": "VI Txn", "vamp": "VAMP", "vamp_pct": "VAMP %"}

                        # [FN-384]
                        def _proj_metric(_mid, _month, _metric):
                            _k = str(_mid).strip().lower()
                            if _k not in _vpi.index:
                                return None
                            _r = _vpi.loc[_k]
                            if isinstance(_r, pd.DataFrame):
                                _r = _r.iloc[0]
                            _mos = [int(_month)] if _month is not None else [0, 1, 2, 3]
                            _vv = sum(float(_r.get(f"VAMP Post M{m}", 0.0) or 0.0) for m in _mos)
                            _tt = sum(float(_r.get(f"VI Txn Post M{m}", 0.0) or 0.0) for m in _mos)
                            if _metric == "txn":
                                return _tt
                            if _metric == "vamp":
                                return _vv
                            return (_vv / _tt * 100.0) if _tt > 0 else 0.0

                        # MIDs carrying BOTH a VAMP(-type) rule and a Txn ceiling — competing.
                        _mid_metrics = {}
                        for _rr in _mid_rules:
                            _mid_metrics.setdefault(str(_rr.get("vampMid")).strip().lower(), set()).add(
                                _rr.get("metric", "txn"))

                        _feas_rows = []   # violated constraints + their minimal relaxation (feasibility report)
                        for _rr in _mid_rules:
                            _mid = str(_rr.get("vampMid")).strip()
                            _mo = _rr.get("month")
                            _rp = _rr.get("rpgt")
                            _mtr = _rr.get("metric", "txn")
                            _tg = float(_rr.get("target") or 0.0)
                            _tl = float(_rr.get("tol") or 0.0)
                            _dir = str(_rr.get("direction", "range"))
                            # constraint TYPE: range = two-sided ±tol; ceiling = upper bound only;
                            # floor = lower bound only. vamp_pct is always ceiling-only.
                            _is_pct = (_mtr == "vamp_pct")
                            _hi_on = _is_pct or (_dir in ("range", "ceiling"))
                            _lo_on = (not _is_pct) and (_dir in ("range", "floor"))
                            _lo = _tg * (1.0 - _tl)
                            _hi = _tg * (1.0 + _tl)
                            _pj = _proj_metric(_mid, _mo, _mtr)
                            _scope = (f"M{_mo}" if _mo is not None else "M0–M3") + ("" if _rp is None else f" · {_rp}")
                            # [FN-385]
                            def _fmt(v):
                                if v is None:
                                    return "—"
                                return (f"{v:.2f}%" if _is_pct else f"{v:,.0f}")
                            if _pj is None:
                                _stat, _bg, _why = "no data", "", "vampMid not in the forecast baseline"
                            elif ((not _hi_on or _pj <= _hi + 1e-6) and (not _lo_on or _pj >= _lo - 1e-6)):
                                _stat, _bg, _why = "✓ met", "background-color: rgba(34,195,107,0.28);", ""
                            else:
                                _below = _lo_on and _pj < _lo
                                _edge = _lo if _below else _hi
                                _over = abs(_pj - _edge)
                                _pct = (_over / _edge * 100.0) if _edge > 0 else 0.0
                                _dirn = "under" if _below else "over"
                                _stat = "✗ violated"
                                _bg = "background-color: rgba(230,55,72,0.28);"
                                _both = {"vamp", "txn"} <= _mid_metrics.get(_mid.lower(), set()) or \
                                        {"vamp_pct", "txn"} <= _mid_metrics.get(_mid.lower(), set())
                                _cause = ("competing VAMP + Txn targets on this MID" if _both else
                                          "target not reachable with the VAMP cap / other MID caps")
                                _need_tol = (abs(_pj / _tg - 1.0) * 100.0) if _tg > 0 else 0.0
                                _bandstr = (f"≤ {_fmt(_hi)}" if _dir == "ceiling"
                                            else f"≥ {_fmt(_lo)}" if _dir == "floor"
                                            else f"{_fmt(_lo)}–{_fmt(_hi)}")
                                _relax = (f"→ widen Tol to ≥ {_need_tol:.0f}% ({_dir} {_bandstr}) to satisfy at this split"
                                          if not _is_pct else f"→ widen the VAMP% ceiling to include {_pj:.2f}%")
                                _why = f"{_dirn}-{_dir} by {_fmt(_over)} ({_pct:+.0f}%); {_cause}. {_relax}"
                                _feas_rows.append({
                                    "mid": _mid, "scope": _scope, "metric": _mlabel.get(_mtr, _mtr),
                                    "type": _dir, "prio": int(_rr.get("priority", 1) or 1),
                                    "target": (f"{_tg:.2f}%" if _is_pct else f"{_tg:,.0f}"),
                                    "now": _fmt(_pj), "dirn": _dirn,
                                    "need": (f"Tol ≥ {_need_tol:.0f}%  ·  or Target → {_fmt(_pj)}"
                                             if not _is_pct else f"raise ceiling ≥ {_pj:.2f}%")})
                        # ---- FEASIBILITY REPORT: smallest per-constraint relaxations that would turn
                        # each violated row green AT THIS SPLIT (widen its tolerance to cover the
                        # projected value, or move its target there). A concrete, achievable set —
                        # relaxing every listed row makes the current split satisfy all of them. ----
                        # Rendered directly into the top-row slot (replaces the old status table); no
                        # header text / caption. Only shows when there are unmet constraints.
                        if _feas_rows:
                            _fh = ['<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow:auto; '
                                   'width:100%; background:var(--tav-card); border:1px solid var(--tav-line);">']
                            _fh.append('<table style="width:100%; border-collapse:collapse; font-size:0.6rem; line-height:1.1;"><tr>')
                            for _c in ["vampMid", "Scope", "Metric", "Type", "Prio", "Target", "Now", "Miss", "Minimal relaxation"]:
                                _al = "right" if _c in ("Target", "Now", "Prio") else "left"
                                _fh.append(f'<th style="background:var(--tav-red); color:#FFF; font-weight:bold; '
                                           f'padding:2px 5px; text-align:{_al}; white-space:nowrap;">{_c}</th>')
                            _fh.append('</tr>')
                            # highest priority-NUMBER first (lowest priority) — cheapest to relax first.
                            for _fr in sorted(_feas_rows, key=lambda r: -int(r.get("prio", 1))):
                                _cells = [("vampMid", "left", _fr["mid"]), ("Scope", "left", _fr["scope"]),
                                          ("Metric", "left", _fr["metric"]), ("Type", "left", _fr["type"]),
                                          ("Prio", "right", str(_fr.get("prio", 1))),
                                          ("Target", "right", _fr["target"]), ("Now", "right", _fr["now"]),
                                          ("Miss", "left", _fr["dirn"]), ("Minimal relaxation", "left", _fr["need"])]
                                _fh.append('<tr>')
                                for _c, _al, _val in _cells:
                                    _fh.append(f'<td style="padding:2px 5px; text-align:{_al}; color:#000; '
                                               f'white-space:nowrap;">{_val}</td>')
                                _fh.append('</tr>')
                            _fh.append('</table></div>')
                            _con_slot.markdown("".join(_fh), unsafe_allow_html=True)

            # Filterable pre/post section (its OWN filter state on this tab):
            # vampMid × RPGT table + VAMP/Transactions bar charts.
            if _gr_shared is not None:
                _prepost_render("impact")
