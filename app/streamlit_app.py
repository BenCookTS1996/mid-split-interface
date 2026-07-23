"""
Transaction Routing Optimiser - Streamlit UI (tabbed).

One tab per stage of the flow:
  1. Forecast   - pick settings, calculate & cache the baseline forecast
  2. Engine     - choose engine + risk/conversion slider + constraints -> split
  3. Impact     - the proposed split, its outputs, and impact dashboards
  4. Compress   - volume-weighted k-means -> config count
  5. Configs    - generate JSON configs and download everything

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import calendar
import datetime
import io
import json
import logging
import os
import sys
import zipfile
from datetime import date

import warnings

import pandas as pd
import streamlit as st
import yaml
import numpy as np

# Silence the benign divide-by-zero / 0-0 RuntimeWarnings from the MANY guarded
# `np.where(denom > 0, num / denom, fallback)` share/rate calculations: np.where
# evaluates num/denom for every row (including denom==0) before selecting, so the
# fallback is always used but numpy still emits the warning. These divisions are all
# intentionally guarded, so the message is pure noise. Targeted to the exact messages
# only — real errors and all other warnings are untouched.
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from routing_optimiser import (  # noqa: E402
    HardConstraints, OptimiserSettings, SoftConstraints, build_cell_problems,
    build_configs, build_pipeline_config, cell_baseline_vs_proposed,
    compress_split, count_config_rules, engine_choices, gateway_success_rates,
    gateway_volume_shift, headline_impact, key_contributors, list_sql_files,
    load_forecast, load_success_data, optimise_split, portfolio_summary,
    prepare_inputs, run_sql_file, run_vamp_pipeline, sweep_slider)
from routing_optimiser.engines import ENGINES, get_engine  # noqa: E402

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# Shared brand mark (favicon + red-banner logo).
_BRAND_ICON = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAASFBMVEVHcEzILDfJLDi2JS21JS68KDHfNETmN0jmN0jkNkfmN0isIiqsIirDKjThNUW4Jy3KLDivIyu8KDHEKzTmN0jEKzS0Ji3KLTfpm+j9AAAAGHRSTlMAFTKMsMD7/8y0jtz/9OQGd/I9/2LkIteIBpNuAAABKklEQVR4AWXTWaKDMAgFUDLXhpiBRve/0xcnGl7Pn4JecIAvpY11PryWt1bwS5uImHw4rW8NUi4RuX62FHEXZXGoYbZMHTqedR+EtfH1Zx0p/LOqO99edR+YTCl4Ste5JUwKDzCM49cHhs/rG3Jsa/DU77JsefOE6ObF1WflOTVeNAj67tBPAoKkaL0zHDcIG6WzY4GIFwVCI+r+WAwqXhoIhYj24IPnBgOCo2H13nNEFBm607B7v/KQWGBi6JBWv/CaGLW8wZB2X6Dhw3HI5uiSSPOjHmyGk7J0S7uCbJDZDQbNdUolA7SILDaA0ol1fU6ME9tpUq7Ie9NasSaaPUNttcZYiascwGu1TkwOIBeXrPy1SiehjwWlZqaW3Wj4kVUzbk+pO9MUsD95+yDzjpcQNgAAAABJRU5ErkJggg=="

st.set_page_config(page_title="Routing Optimiser", layout="wide",
                   initial_sidebar_state="collapsed", page_icon=_BRAND_ICON)

# Theme: green primary, light header with black text, red metric cards.
st.markdown("""
<style>
  :root {
    --tav-green: #22C36B; --tav-green-dark: #1AA85C;
    --tav-red: #e63748;
    --tav-header-bg: #FFFFFF; --tav-ink: #0B1F3A; --tav-muted: #475467;
    --tav-line: #DCE6F5; --tav-card: #FFFFFF;
  }
  /* square corners everywhere (no rounded corners) */
  [data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stTable"],
  [data-testid="stExpander"], details, summary, [data-testid="stPlotlyChart"],
  [data-testid="stNotification"], [data-testid="stAlert"], [data-baseweb="notification"],
  div[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stImage"] img,
  .stButton > button, .stDownloadButton > button, .stSelectbox div[data-baseweb="select"] > div,
  .stTextInput input, .stNumberInput input, .stDateInput input,
  /* BaseWeb wrappers (the rounded border lives here, not on the inner <input>),
     plus multiselect tags, dropdown menus, calendars and any raw input/textarea. */
  [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"],
  [data-baseweb="select"] > div, [data-baseweb="tag"], [data-baseweb="popover"],
  [data-baseweb="menu"], [data-baseweb="calendar"], [role="listbox"],
  .stMultiSelect [data-baseweb="select"] > div, .stTextArea textarea, input, textarea,
  .stTabs [data-baseweb="tab"] { border-radius: 0 !important; }
  /* spacing */
  .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 100%;
    padding-left: 1rem; padding-right: 1rem;}
  div[data-testid="stVerticalBlock"] {gap: 0.35rem;}
  div[data-testid="stHorizontalBlock"] {gap: 0.6rem;}
  hr {margin: 0.3rem 0;}
  h5 {margin: 0.35rem 0 0.15rem 0; color: var(--tav-ink); font-weight: 700;}

  /* st.caption text in black (Streamlit defaults to muted grey) */
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
  div[data-testid="stCaptionContainer"] p, .stCaption, .stCaption * {
    color: #000000 !important;
  }

  /* branded header banner (now red, white text) */
  .tav-header {
    display:flex; align-items:center; gap:16px;
    background: var(--tav-red); color: #ffffff;
    border:1px solid var(--tav-line); border-radius:0; padding:18px 22px; margin-bottom:14px;
  }
  .tav-header .tav-title {font-size:1.45rem; font-weight:800; letter-spacing:-.01em; color:#ffffff;}
  .tav-header .tav-sub {font-size:.9rem; color:#ffffff; margin-top:2px; opacity: 0.9;}
  .tav-badge {display:flex; align-items:center; justify-content:center;
    width:46px; height:46px; background:#fff;
    border:1px solid var(--tav-line); border-radius:0;}

  /* buttons */
  .stButton > button, .stDownloadButton > button {
    border-radius:0; font-weight:600; border:1px solid var(--tav-line);
    background:#fff; color:var(--tav-ink); transition:all .15s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    border-color:var(--tav-green); color:var(--tav-green-dark);
  }
  .stButton > button[kind="primary"], [data-testid="stBaseButton-primary"],
  .stDownloadButton > button[kind="primary"] {
    background:var(--tav-green); border:1px solid var(--tav-green); color:var(--tav-ink);
    box-shadow:0 4px 12px rgba(34,195,107,.28);
  }
  .stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background:var(--tav-green-dark); border-color:var(--tav-green-dark); color:var(--tav-ink);
  }

  /* tabs */
    .stTabs [data-baseweb="tab-list"] {
    gap: 4px; 
    border-bottom: 2px solid var(--tav-line);
    background-color: #FFFFFF !important;
    box-shadow: none !important;
  }
    /* Remove excessive whitespace between navbar tabs and the content below */
  [data-testid="stTabPanel"], .stTabs [data-baseweb="tab-panel"] {
    padding-top: 0.25rem !important;
  }
  div[data-testid="stVerticalBlock"] > .element-container:has(h3) {
    margin-top: -0.5rem !important;
  }            
            

  .stTabs [data-baseweb="tab"] {
    font-weight:600; color:var(--tav-muted); border-radius:0; padding:6px 14px;
  }
  .stTabs [aria-selected="true"] {color:var(--tav-green) !important;}
  .stTabs [data-baseweb="tab-highlight"] {background:var(--tav-green);}





            
  /* metric cards (red with ink text) */
  [data-testid="stMetric"] {
    background:var(--tav-red); border:2px solid var(--tav-red);
    border-radius:0; padding:10px 12px;
    min-height:112px;                        /* equal-height cards (fits the 2-line SR delta) */
    display:flex; flex-direction:column; justify-content:center;
  }
  [data-testid="stMetric"] label, [data-testid="stMetricLabel"] {color:var(--tav-ink) !important;}
  /* card text (higher specificity beats the generic label rule below) */
  [data-testid="stMetric"] [data-testid="stMetricLabel"] p {font-size:12px !important; line-height:1.15 !important;}
  [data-testid="stMetricValue"] {color:var(--tav-ink); font-weight:800; font-size:21px !important;}
  /* delta: keep the sizing but DON'T force a colour, so Streamlit's conditional
     green(up)/red(down) arrow + text applies (via delta_color on each st.metric) */
  [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {font-size:11px !important; line-height:1.2 !important;}

  /* help ('?') tooltip: SOLID white with dark ink text (50%-opacity white read as grey before) */
  [data-testid="stTooltipContent"], div[data-baseweb="tooltip"], div[data-baseweb="tooltip"] > div,
  div[role="tooltip"] {
    background: #FFFFFF !important;
    color: var(--tav-ink) !important;
    border: 1px solid #C9D6EA !important;
    border-radius: 0 !important;
    box-shadow: 0 2px 8px rgba(11,31,58,0.15) !important;
  }
  [data-testid="stTooltipContent"] *, div[data-baseweb="tooltip"] *, div[role="tooltip"] * {
    color: var(--tav-ink) !important; background: transparent !important;
  }

  /* main UI background */
  .stApp {background-color: #F7FAFF;}

  /* sidebar (now red) */
  [data-testid="stSidebar"] {background: var(--tav-red); border-right: 1px solid var(--tav-line);}
  [data-testid="stSidebar"] h1 {color: #ffffff; font-size:1.15rem;}
  [data-testid="stSidebar"] .stMarkdown p {color: #ffffff;}

  /* Enforce ink color for labels, checkboxes, and radio buttons */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stCheckbox"] p,
  [data-testid="stRadio"] p,
  [data-testid="stMarkdownContainer"] p {
    color: var(--tav-ink) !important;
  }

  /* input styling (card background, ink text, thicker red border) */
  div[data-baseweb="input"] > div, 
  div[data-baseweb="select"] > div {
    background-color: var(--tav-card) !important;
    border: 2px solid var(--tav-red) !important;
  }
  div[data-baseweb="input"] input, 
  div[data-baseweb="select"] span,
  div[data-baseweb="select"] div {
    color: var(--tav-ink) !important;
  }
  div[data-baseweb="input"] > div:focus-within, 
  div[data-baseweb="select"] > div:focus-within {
    border-color: var(--tav-green) !important;
    box-shadow: 0 0 0 1px var(--tav-green) !important;
  }

  /* File Uploader / Drag and Drop buttons and Dropzone */
  [data-testid="stFileUploaderDropzone"] {
    background-color: var(--tav-red) !important;
    border: 2px dashed var(--tav-card) !important;
  }
  [data-testid="stFileUploaderDropzone"] * {
    color: var(--tav-card) !important;
  }
  [data-testid="stFileUploaderDropzone"] svg {
    fill: var(--tav-card) !important;
  }
  [data-testid="stFileUploader"] button {
    background-color: var(--tav-card) !important;
    color: var(--tav-ink) !important;
    border: 2px solid var(--tav-card) !important;
  }
  [data-testid="stFileUploader"] button:hover {
    background-color: var(--tav-ink) !important;
    color: var(--tav-card) !important;
    border-color: var(--tav-ink) !important;
  }

  /* Input text size: labels + values across all widgets -> 10px */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] label,
  div[data-baseweb="select"] div,
  div[data-baseweb="select"] input,
  div[data-baseweb="input"] input,
  .stNumberInput input, .stTextInput input, .stDateInput input,
  [data-testid="stSelectbox"] div[role="button"],
  [data-baseweb="popover"] li,
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"],
  .stSlider [data-testid="stThumbValue"],
  [data-testid="stMetricLabel"] p {
    font-size: 12px !important;
  }

  /* Table text size -> 9px across all tabs.
     Covers custom HTML tables (rendered via st.markdown) and native
     st.table / st.dataframe / st.data_editor grids. The !important beats
     the (non-important) inline font-size set on each HTML <td>/<th>. */
  [data-testid="stMarkdownContainer"] table,
  [data-testid="stMarkdownContainer"] table th,
  [data-testid="stMarkdownContainer"] table td,
  [data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td,
  [data-testid="stDataFrame"] div, [data-testid="stDataFrameResizable"] div,
  [data-testid="stDataEditor"] div {
    font-size: 9px !important;
  }

  /* Tables fill the full width of their card. */
  [data-testid="stMarkdownContainer"] table { width: 100% !important; }

  /* --- Equal-height side-by-side layout -------------------------------
     Streamlit lays a row of columns out as a flex row, so sibling columns
     already stretch to the height of the tallest one. These rules make each
     column's content stack fill that height, and let an HTML-table card grow
     to fill the leftover space so a table beside a chart (or another table)
     lines up with it dynamically — no hard-coded pixel heights. Rows in a
     filled table distribute the extra height proportionally. Scoped to
     horizontal blocks so full-width tables keep their natural height. */
  [data-testid="stHorizontalBlock"] { align-items: stretch; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] {
    height: 100%;
  }
  [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(table) {
    flex: 1 1 auto;
  }
  [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(table),
  [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(table) [data-testid="stMarkdown"],
  [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(table) [data-testid="stMarkdownContainer"] {
    display: flex; flex-direction: column;
  }
  [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(table) [data-testid="stMarkdownContainer"] > div {
    flex: 1 1 auto; height: 100%;
  }
  [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(table) table {
    height: 100%;
  }
</style>
""", unsafe_allow_html=True)

_HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
os.chdir(PROJECT_ROOT)  # Ensures the app always operates out of the project root
SQL_DIR = os.path.join(_HERE, "..", "queries")
CACHE_DIR = os.path.join(_HERE, "..", ".cache")
INPUTS_DIR = os.path.join(_HERE, "..", "config", "inputs")
GCP_PROJECT = "sapient-tangent-172609"  # matches the VAMP repo's BigQuery project


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


def _chart_title(text: str, container=None):
    """Render a chart title ABOVE the chart (outside the plot area), so titles
    sit consistently above every figure rather than inside the plot canvas."""
    html = ("<div style='font-weight:700; color:#0B1F3A; font-size:0.9rem; "
            "line-height:1.25; margin:6px 0 14px 4px; position:relative; z-index:3;'>"
            + str(text) + "</div>")
    (container or st).markdown(html, unsafe_allow_html=True)


def _ink_caption(md: str):
    """Render a caption in ink (near-black) rather than Streamlit's default grey."""
    import re as _re
    html = _re.sub(r"`([^`]+)`", r"<code>\1</code>", md)
    st.markdown(
        f"<div style='color:var(--tav-ink); font-size:0.82rem; line-height:1.35;'>{html}</div>",
        unsafe_allow_html=True)


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


def _save_ga_perf(d):
    """Persist the GA timing calibration so the estimate survives Streamlit restarts."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        import json as _json
        with open(os.path.join(CACHE_DIR, "ga_perf.json"), "w") as _f:
            _json.dump(d, _f)
    except Exception:
        pass


# Impact-tab calc/export/cache helpers live in impact_calcs.py (kept out of this
# file for size). Imported here so all call sites resolve unchanged.
from impact_calcs import (  # noqa: E402
    build_split_exports, build_kill_eff, compute_vamp_post_by_mid, compute_vamp_post_from_prorata,
    compute_vamp_prepost_granular, process_wallet_incapable, enforced_prop_items,
    enforced_split_frame,
    rpgt_avg_ticket, mid_revenue_month_table, mid_table_from_granular,
    pool_targeted_compression,
    _c_prepost_granular, _c_read_parquet, _c_vamp_post_prorata, _mtime,
)


# Default gateway (vampMid) allow-list for the attempts_success.sql query.
# The query is scoped to these gatewayFids; edit here to broaden/narrow the pull.
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
ss = st.session_state

# RPGTs (transaction types) used across the pipeline and templates.
RPGT_LIST = [
    "Monthly Initial", "Annual Sub Sale", "Addon Sale", "Upgrades",
    "Monthly Renewal", "Annual Sub Renewal", "P6M Renewals", "Addon Renewal",
]
COMPANIES = ["TotalAV", "Total Drive", "Total Adblock", "Total Cleaner", "Total VPN"]


# --- cached expensive steps -------------------------------------------------
def resolve_attempts(attempts_export: str, _key: str):
    """Success-rate data (a ROUTING input, not the forecast).

    Priority: an exported file if given; else run attempts_success.sql on
    BigQuery. There is no sample fallback — if BigQuery fails, we raise.
    """
    if attempts_export and os.path.exists(attempts_export):
        return attempts_export, "exported file"
    sql_path = os.path.join(SQL_DIR, "attempts_success.sql")
    return run_sql_file(sql_path, CACHE_DIR, use_cache=True,
                        fallback_csv=None, project=GCP_PROJECT)


# Per-step helpers used to live here (step_attempts / step_success_rates /
# step_forecast) for tab 1's routing-cell build. Tab 3 now owns the routing
# flow and calls the backend directly, so those helpers were removed.


def bar(df, x, y, title, color=None):
    if HAS_PLOTLY:
        fig = px.bar(df, x=x, y=y, color=color, title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360,
                          legend=dict(font=dict(color='#0B1F3A', size=8)))
        fig.update_xaxes(tickfont=dict(color='#0B1F3A', size=8))
        fig.update_yaxes(tickfont=dict(color='#0B1F3A', size=8))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(title)
        st.bar_chart(df.set_index(x)[y])


class StreamlitLogHandler(logging.Handler):
    """Streams log records live by calling a sink with each formatted line."""

    def __init__(self, sink):
        super().__init__(logging.INFO)
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.sink(self.format(record))
        except Exception:  # noqa: BLE001
            pass



# --- sidebar removed (not needed) — hide it and its collapse control entirely.
st.markdown("""<style>
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    button[title="Show sidebar"] { display: none !important; }
    div[data-testid="stAppViewContainer"] > section.main { margin-left: 0 !important; }
</style>""", unsafe_allow_html=True)

APP_BUILD = "2026-07-23l"  # ROOT FIX: renormalise eval-frame share/baseline_share per cell at source → undoes BIN-volume dilution consistently (shares 100%, ΣExpected=ΣRaw, Post Revenue/$ Impact un-diluted everywhere)
st.markdown(f"""
<div class="tav-header">
  <div class="tav-badge">
    <img src="{_BRAND_ICON}" width="40" height="40" alt="Logo" />
  </div>
  <div class="tav-htext">
    <div class="tav-title">Transaction Routing Optimiser</div>
    <div class="tav-sub">Payments &amp; Risk &nbsp;·&nbsp; maximise authorisation rates while staying inside VAMP limits
      &nbsp;·&nbsp; <span style="opacity:0.7;">build {APP_BUILD}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

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
    ev["post_vol"] = pd.to_numeric(ev.get("volume"), errors="coerce").fillna(_cv * ev["share"])
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


tab_fc, tab_eng, tab_imp, tab_cfg = st.tabs([
    "1 · Baseline & Validate",
    "2 · Routing engine",
    "3 · Split, outputs & impact",
    "4 · Generate configs",
])

# ============================================================================
# TAB 1 - Forecast
# ============================================================================
with tab_fc:
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
        def _load_default_json(name):
            try:
                with open(os.path.join(INPUTS_DIR, name)) as f:
                    return json.load(f)
            except Exception:
                return None

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
        attempts_export = ""
        run_live, use_yaml_asis, reuse_cached_curves = True, False, True
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
        settings_hidden = False
        if use_prev:
            # Forecast-outputs folder on the LEFT; Split Go Live date on the RIGHT (the other
            # inputs are hidden in this mode, so it's shown here so it's always available).
            _pv1, _pv2 = st.columns(2)
            prev_dir = _pv1.text_input(
                "Forecast outputs folder (the data/outputs/<MONTH>/<COMPANY>/ folder)", "")
            split_go_live = _pv2.date_input(
                "Split Go Live date", value=ss.get("split_go_live_date", m0_date), key="sgl_hidden",
                help="Date the proposed split goes live. Drives the mid-month pro-rata "
                     "element in the forecast export and the tab-4 VAMP Post projection.")
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
                    if len(parts) >= 2:
                        company, month_var = parts[-1], parts[-2]
                        try:
                            m_int = datetime.datetime.strptime(month_var.capitalize(), "%b").month
                            m0_date = m0_date.replace(month=m_int)
                        except ValueError:
                            pass
                    st.success(f"Valid forecast found — {company} ({month_var}). Other "
                               f"inputs hidden. Click **Load forecast**, then open tab 2.")
                else:
                    st.error("Missing/invalid outputs: "
                             + (", ".join(miss) if miss else "not a folder")
                             + f".  Looked in: {prev_dir}")

        _fc_log_slot = None   # forecast run-log slot (assigned in ROW 2 right column)

        if not settings_hidden:
            # --- ROW 1: Run Identity & Data Sources ---
            r1_c1, r1_c2 = st.columns(2, gap="large")
        
            with r1_c1:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>1 · Run identity</h5>", unsafe_allow_html=True)
                    id_c1, id_c2 = st.columns(2)
                    company = id_c1.selectbox("Company", COMPANIES,
                                              help="Brand this forecast is for.")
                    scheme = id_c2.selectbox("Card Scheme", ["visa", "mastercard"],
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
                        "Split Go Live date", value=ss.get("split_go_live_date", m0_date),
                        key="sgl_ident",
                        help="Date the proposed split goes live. Drives the mid-month pro-rata "
                             "element in the forecast export and the tab-4 VAMP Post projection.")
                    ss["split_go_live_date"] = split_go_live
                    # Live actuals (moved here from its own section) — blend in real recent results.
                    # Nudge the checkbox down so it vertically centres on the Split Go Live INPUT
                    # box (left column), not up level with its label.
                    st.markdown("<style>.st-key-use_live_actuals_cb{margin-top:1.75rem;}</style>",
                                unsafe_allow_html=True)
                    use_live_actuals = id_c4.checkbox("Use Live Actuals", value=False,
                                                      key="use_live_actuals_cb",
                                                      help="Blend in real recent results.")
                    if use_live_actuals:
                        _la1, _la2 = st.columns(2)
                        actuals_start_date = _la1.date_input("Actuals Start Date", value=date.today())
                        actuals_end_date = _la2.date_input("Actuals End Date", value=date.today())
                        if actuals_start_date > actuals_end_date:
                            actuals_valid = False
                            st.error("Actuals Start Date must be on or before End Date.")

                    # --- Backup rules folder: the deployed pipeline blends the exported split with
                    #     the backup files' catch-all (BIN=Other) rows, which re-add gateways the split
                    #     zeroed/omitted. Point tab 2 (optimiser) + tab 3 (impact) at those backups so
                    #     they optimise/project against what is ACTUALLY routed (matches tab 5). ---
                    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
                    _bk_dir = st.text_input(
                        "Backup rules folder (catch-all BIN=Other re-adds)",
                        value=ss.get("backup_rules_dir", "data/backup_rules/"), key="backup_rules_dir_input",
                        help="Folder holding the backup split files (e.g. mid_split_*_Eff_Backup_*). Their "
                             "catch-all 'BIN=Other' rows re-add gateways your exported split zeroed/omitted, "
                             "exactly as the deployed pipeline (tab 5) does. Tabs 2 & 3 blend these in so their "
                             "numbers match tab 5. Leave blank to use the raw exported split only.")
                    ss["backup_rules_dir"] = _bk_dir
                    try:
                        from routing_optimiser.backup_blend import parse_backup_catchall as _pbc
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
                                    from routing_optimiser.forecast_pipeline import _canonical_gateway as _cgbc
                                    _ovp_bc = os.path.join(PROJECT_ROOT, "config", "inputs", "gateway_volume_overrides.json")
                                    _off_bc = set()
                                    if os.path.exists(_ovp_bc):
                                        for _gw, _cfg in (_jbc.load(open(_ovp_bc)) or {}).items():
                                            if isinstance(_cfg, dict) and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                                                    and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
                                                _off_bc.add(str(_cgbc(_gw)).strip().lower())
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

            with r1_c2:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>2 · Data sources</h5>", unsafe_allow_html=True)
                    if st.button("Test BigQuery connection"):
                        with st.spinner("Querying BigQuery..."):
                            try:
                                from google.cloud import bigquery
                                _df = (bigquery.Client(project=GCP_PROJECT)
                                       .query("select 1 as x").to_dataframe())
                                st.success(f"BigQuery OK — reached {GCP_PROJECT}, got "
                                           f"{int(_df.iloc[0, 0])}. Live runs will work.")
                            except Exception as exc:  # noqa: BLE001
                                import traceback
                                st.error(f"BigQuery failed: {type(exc).__name__}: {exc}")
                                with st.expander("Full traceback"):
                                    st.code(traceback.format_exc())
                    use_yaml_asis = st.checkbox(
                        "Use config/settings.yaml as-is (repo parity)", value=False,
                        help="Run the pipeline straight from config/settings.yaml, like your repo.")
                    reuse_cached_curves = st.checkbox(
                        "Reuse cached actuarial curves", value=True,
                        help="Reuse cached reference_curves (load_curves_from_cache).")

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

            # --- ROW 2: M0 Txn Weightings & Live Actuals ---
            r2_c1, r2_c2 = st.columns(2, gap="large")
        
            with r2_c1:
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>3 · M0 Transaction Weightings</h5>", unsafe_allow_html=True)
                    m0_total = st.number_input(f"M0 {company} - {scheme} - Total", 0, 50_000_000,
                                               318_077, step=1000,
                                               help="Total starting transactions for month 0.")
                    rpgt_assumed = {}
                    w_cols = st.columns(2)
                    for i, rpgt in enumerate(RPGT_LIST):
                        rpgt_assumed[rpgt] = w_cols[i % 2].number_input(
                            rpgt, 0, 50_000_000, 0, step=500, key=f"assumed_{rpgt}",
                            help="Assumed month-0 volume for this type.")
                    allocated = sum(rpgt_assumed.values())

                # Section 5 now sits below section 3 in the LEFT column.
                with st.container(border=True):
                    st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>5 · Configs & Overrides</h5>", unsafe_allow_html=True)

                    def _read_json(upload, default_path, label):
                        if upload is not None:
                            try:
                                data = json.load(upload)
                                return data
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"Could not parse {label}: {exc}")
                                return None
                        if os.path.exists(default_path):
                            with open(default_path) as f:
                                data = json.load(f)
                            return data
                        return None

                    g1, g2 = st.columns(2)
                    test_gw_file = g1.file_uploader("Test Gateways (JSON)", type=["json"])
                    thermo_file = g2.file_uploader("Thermometer Config", type=["json"])
                    override_file = st.file_uploader("Gateway Volume Overrides", type=["json"])

                    test_gateways = _read_json(test_gw_file,
                                               os.path.join(INPUTS_DIR, "test_gateways.json"),
                                               "Test Gateways") or {}
                    thermometer_config = _read_json(
                        thermo_file, os.path.join(INPUTS_DIR, "thermometer_config.json"),
                        "thermometer config")
                    gateway_volume_overrides = _read_json(
                        override_file, os.path.join(INPUTS_DIR, "gateway_volume_overrides.json"),
                        "gateway volume overrides")

            with r2_c2:
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

                # Forecast run log renders here (right column, beside Configs & Overrides).
                _fc_log_slot = st.container()



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
            "mid_list_file": "data/mappings/Master_MID_List.csv",
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

        if not settings_hidden:
            with st.expander("Preview assembled settings.yaml (VAMP pipeline schema)"):
                pipeline_config = build_pipeline_config(forecast_settings)
                st.code(yaml.safe_dump(pipeline_config, sort_keys=False), language="yaml")
                st.download_button("Download settings.yaml",
                                   yaml.safe_dump(pipeline_config, sort_keys=False),
                                   file_name="settings.yaml", mime="text/yaml")

        # White label text on this green primary button (default primary text is ink).
        st.markdown("""<style>
            .st-key-calc_cache_btn button, .st-key-calc_cache_btn button * { color: #ffffff !important; }
        </style>""", unsafe_allow_html=True)
        if st.button("Load forecast" if settings_hidden else "Calculate & cache forecast",
                     type="primary", key="calc_cache_btn",
                     disabled=not actuals_valid):
            key = json.dumps(forecast_settings, sort_keys=True)
            # Render the run log into the ROW-2 right-column slot when it exists (normal mode);
            # fall back to full-width in previous-forecast mode where that column is hidden.
            _log_ctx = _fc_log_slot if _fc_log_slot is not None else st
            with _log_ctx.status("Calculating & caching forecast...", expanded=True) as status:
                log_area = st.empty()
                log_lines: list[str] = []

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
                    synth_mode = not run_live and not load_baseline

                    log("── Forecast (risk / VAMP baseline) ──")
                    forecast_path, fc_src = None, "synthesised from attempts"
                    if run_live:
                        log("• Running VAMP pipeline (BigQuery); pipeline logs:")
                        try:
                            if use_yaml_asis:
                                with open(os.path.join(PROJECT_ROOT, "config",
                                                       "settings.yaml")) as _f:
                                    pipeline_config = yaml.safe_load(_f)
                                log("  using config/settings.yaml as-is (repo parity)")
                            else:
                                pipeline_config = build_pipeline_config(forecast_settings)
                            pipeline_out_dir = run_vamp_pipeline(
                                pipeline_config, PROJECT_ROOT, gcp_project=GCP_PROJECT)
                            forecast_path, fc_src = pipeline_out_dir, "VAMP pipeline (live)"
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
                        forecast_path, fc_src = cached_inputs_path, "VAMP pipeline pre (cached)"
                        pipeline_out_dir = (cached_inputs_path
                                            if os.path.isdir(cached_inputs_path)
                                            else os.path.dirname(cached_inputs_path))
                        log(f"• Using cached pipeline baseline: {cached_inputs_path}")
                    else:
                        log("• No pipeline output — baseline will not be built here; "
                            "tab 3 will fetch success data and build routing cells.")

                finally:
                    root_logger.removeHandler(handler)
                    root_logger.setLevel(prev_level)

                status.update(
                    label=f"Forecast ready for {company} ({month_var}); baseline: {fc_src}",
                    state="complete", expanded=False)

            # Clear downstream artifacts so old runs don't linger. Tab 3 owns success
            # rates and routing cells now, so we clear them too — the user runs tab 3
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
        # (reuses tab_validate's renderer). Reads mid_level.csv from the cached forecast output dir.
        try:
            _fc_out = ss.get("pipeline_out_dir")
            _mid_csv = os.path.join(_fc_out, "mid_level.csv") if _fc_out else None
            if _mid_csv and os.path.isfile(_mid_csv):
                from tab_validate import _to_prepost as _tpp, _render_prepost_table as _rpt
                _pre = _tpp(pd.read_csv(_mid_csv))
                _pre = _pre[[c for c in _pre.columns if "Post" not in c]]   # PRE months only
                st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
                st.markdown("<h5 style='margin-top:0; margin-bottom:0.25rem;'>Baseline forecast — VI Txn &amp; VAMP by month</h5>",
                            unsafe_allow_html=True)
                _rpt(_pre, fit_content=True)   # hug content → tight gap between vampMid & months
        except Exception as _e:  # noqa: BLE001
            st.caption(f"(baseline VI/VAMP table unavailable: {type(_e).__name__}: {_e})")


    with _vs:
        import tab_validate
        tab_validate.render(ss, PROJECT_ROOT, GCP_PROJECT)

# ============================================================================
# TAB 3 - Routing engine (choose engine + slider + constraints -> propose split)
# ============================================================================
with tab_eng:
    out_dir = ss.get("pipeline_out_dir")

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
        # 'genetic' is served by the cross-cell tilt GA (genetic_global.run_midtilt_ga), which the
        # app dispatches directly (see engine_key == "genetic" below) — it is NOT a registry engine,
        # so its dropdown option is injected here rather than sourced from ENGINES.
        if "genetic" not in {k for k, _ in choices}:
            choices.append(("genetic", "Genetic algorithm"))
        labels = {k: lbl for k, lbl in choices}
        keys = [k for k, _ in choices]
        # Default to the Genetic algorithm (the production engine); fall back to softmax/first.
        default_idx = (keys.index("genetic") if "genetic" in keys
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
                _cb1, _cb2 = st.columns(2)
                _rpgt_hold_others = _cb1.checkbox(
                    "Hold unselected RPGTs at their current split (pre = post)",
                    value=True, key="eng_rpgt_hold_others",
                    help="ON: the optimised routing is applied ONLY to the selected RPGTs; every "
                         "other RPGT is held on its current baseline split, so its pre and post "
                         "impact are identical. OFF: the single Currency×Bank routing decision "
                         "(still informed only by the selected RPGTs) is applied to ALL RPGTs. "
                         "No effect when all RPGTs are selected.")
                _auto_explore = _cb2.checkbox(
                    "Auto-explore capable-but-untested gateways", value=True, key="eng_auto_explore",
                    help="Automatically make every gateway approved for a cell's currency (from "
                         "Master_MID_List, minus scrubbed/switched-off) an eligible candidate — even "
                         "with zero attempts — so capable-but-untested gateways can earn exploration "
                         "volume without maintaining a manual list. Each untested gateway is seeded at "
                         "the bank×currency AVERAGE success rate as a weak prior, so Thompson explores "
                         "it with a wide posterior and softmax gives it the exploration floor. The VAMP "
                         "cap and all constraints still govern them. Off = today's behaviour (only "
                         "gateways with observed attempts, plus any explore_untested_gateways list).")
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
                if engine_key not in ("genetic", "thompson"):
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
                elif engine_key == "genetic":
                    pass   # nothing shown for genetic
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
                        # Always render the input — inside a form, hiding it behind the selectbox
                        # above wouldn't reveal it until submit. It's simply ignored under Empirical
                        # Bayes (which estimates the smoothing volume per Bank×Currency).
                        _shrink_in = st.number_input(
                            "Bayesian Smoothing Volume", min_value=0, max_value=100_000, value=300, step=50,
                            help="Pseudo-attempts applied to every gateway under Fixed Number smoothing "
                                 "(ignored under Empirical Bayes, which estimates it per Bank×Currency).")
                        shrink = 300 if use_eb else int(_shrink_in)

        # Section 3 (Risk constraints + per-MID editor) on the LEFT, Section 4 (Run Log)
        # on the RIGHT. Both columns were created inside the form above, so every widget
        # placed here is a form member and only takes effect on submit.
        with _c_log:
            with st.container(border=True):
                st.markdown("##### 4. Run Log")
                _run_prog_slot = st.container()   # % complete + ETA bar (filled during a run)
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

        if submit_engine:
            ss["exploration_floor"] = float(floor)   # tab 3 uses this to replicate the engine's floor
            base_settings = OptimiserSettings(
                risk_conversion_weight=0.5, engine=engine_key, engine_params=params,
                hard=HardConstraints(max_gateway_share=max_share, vamp_cap=vamp_cap),
                soft=SoftConstraints(exploration_floor=floor))

            import time as _pt
            _run_t0 = _pt.time()
            # Spinner + "Running…" label (LEFT) share ONE row with the % / ETA progress bar
            # (RIGHT); the live log renders full-width beneath (log_area is redirected to
            # _run_log_slot below). The bar takes 4/5 of the row so it lines up close to the
            # full-width log box under it.
            # Status label wide enough for one line; estimator bar takes less room to its right.
            _hc_l, _hc_r = _run_prog_slot.columns([3, 2])
            status = _hc_l.status("Running pre-processing and split engine…", expanded=False)
            _pbar = _hc_r.progress(0.0, text="0% · starting…")

            # ADAPTIVE ETA: derive the stage-boundary fractions from the LAST run's measured
            # engine (④) + compression (⑥) wall-times (persisted in ga_perf.json), so the % and
            # ETA self-calibrate to this machine + settings — multi-seed, dial count and pool
            # target all shift the engine/compression split, which broke the old fixed fractions.
            _perf0 = _load_ga_perf() or {}
            _E_est = float(_perf0.get("secs", 715.0) or 715.0)           # engine ④ secs (last run)
            _C_est = float(_perf0.get("compress_secs", 527.0) or 527.0)  # compression ⑥ secs
            _PRE_est = 43.0                                              # ①②③ (roughly fixed)
            _T_est = max(_PRE_est + _E_est + _C_est, 1.0)
            _f_cells = 26.0 / _T_est                    # assembling-cells checkpoint
            _f_eng = _PRE_est / _T_est                  # engine start
            _f_eng_end = (_PRE_est + _E_est) / _T_est   # engine done → compression start
            def _eng(fr):                               # within-engine fraction → global fraction
                return _f_eng + float(fr) * (_f_eng_end - _f_eng)
            _f_rmin, _f_enf1, _f_enf2, _f_var = _eng(0.37), _eng(0.80), _eng(0.90), _eng(0.97)
            _t6_0 = None   # compression-stage start (set at stage ⑥; used to persist compress_secs)

            def _progress(frac, label=""):
                frac = max(0.0, min(1.0, float(frac)))
                _pct = int(round(frac * 100))
                _el = _pt.time() - _run_t0
                if frac >= 0.999:
                    _rem = ""
                elif frac <= 0.02:
                    # Too early for a stable linear estimate — trust the machine-calibrated
                    # total (last run's measured engine+compression wall-time).
                    _eta = int(max(_T_est - _el, 1))
                    _rem = f" · ~{_eta}s left (est.)"
                else:
                    # Blend the calibrated total remaining (_T_est − elapsed) with a live
                    # linear extrapolation, weighting the model more early and the live pace
                    # more as the run finishes — smoother + far less jumpy than pure linear
                    # from an early fraction.
                    _eta_model = _T_est - _el
                    _eta_lin = _el * (1.0 - frac) / frac
                    _eta = int(max(frac * _eta_lin + (1.0 - frac) * _eta_model, 1))
                    _rem = f" · ~{_eta}s left (est.)" if _eta >= 1 else " · almost done"
                _txt = f"{_pct}%" + (f" · {label}" if label else "") + _rem
                try:
                    _pbar.progress(frac, text=_txt)
                except Exception:  # noqa: BLE001
                    pass

            with status:
                # Render the live log FULL-WIDTH into the Run-Log column (not inside the
                # narrow spinner column), so the spinner+bar row stays compact on top and
                # the dark log box spans the panel beneath it.
                log_area = _run_log_slot.empty()
                log_lines: list[str] = []
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
                def _stage(name):
                    _now = _pt.time()
                    if _stage_state["name"] is not None:
                        log(f"   ✓ {_stage_state['name']} — finished in {_now - _stage_state['t']:.1f}s")
                    _stage_state["name"] = name
                    _stage_state["t"] = _now
                    log(f"▶ {name} — started")
                def _stage_end():
                    if _stage_state["name"] is not None:
                        log(f"   ✓ {_stage_state['name']} — finished in {_pt.time() - _stage_state['t']:.1f}s")
                        _stage_state["name"] = None

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
                        def _gv(name, default="?"):
                            return _L.get(name, default)
                        def _bmark(modpath):
                            try:
                                return getattr(_il.import_module(modpath), "__build__", "(no __build__)")
                            except Exception as _e:  # noqa: BLE001
                                return f"(import failed: {_e})"
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
                        _pk = {k: params.get(k) for k in ("temperature", "temp_method",
                                                          "explore_cap_total", "explore_cap_each", "n_variations")
                               if isinstance(params, dict) and k in params}
                        _diag(f"      engine_params={_pk}")
                        _diag(f"      auto_explore={_gv('_auto_explore')} · RPGT_scope={('ALL' if not _gv('_sel_rpgts', None) else _gv('_sel_rpgts'))} · "
                              f"hold_unselected_at_baseline={ss.get('eng_rpgt_hold_others')}")
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
                    
                    typo_map = {
                        "MONTHLY INTIIAL": "Monthly Initial", "MONTHLY INITIAL": "Monthly Initial",
                        "ANNUAL SUB SALE": "Annual Sub Sale", "ADDON SALE": "Addon Sale",
                        "UPGRADE": "Upgrades", "UPGRADES": "Upgrades",
                        "MONTHLY RENEWAL": "Monthly Renewal", "ANNUAL SUB RENEWAL": "Annual Sub Renewal",
                        "P6M RENEWALS": "P6M Renewals", "ADDON RENEWAL": "Addon Renewal"
                    }
                    if "rpgt" in adf.columns:
                        adf["rpgt"] = adf["rpgt"].astype(str).str.strip().str.upper().map(typo_map).fillna(adf["rpgt"])
                        
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
                    if "rpgt" in forecast_temp.columns:
                        forecast_temp["rpgt"] = forecast_temp["rpgt"].astype(str).str.strip().str.upper().map(typo_map).fillna(forecast_temp["rpgt"])

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
                            _excl = set()
                            for _gwid, _cfg in (_ovr.items() if isinstance(_ovr, dict) else []):
                                if not isinstance(_cfg, dict):
                                    continue
                                _tgt = pd.to_numeric(_cfg.get("target"), errors="coerce")
                                _ap = str(_cfg.get("apply_to", "")).strip().lower()
                                if _tgt == 0 and _ap in ("trx", "both"):
                                    _excl.add(str(_canonical_gateway(_gwid)).strip().lower())
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
                                        _j = _json.load(open(_pth))
                                        _skip |= {str(_cg_ex(g)).strip().lower() for g in (_j.get(_key, []) if isinstance(_j, dict) else [])}
                                except Exception:  # noqa: BLE001
                                    pass
                            try:
                                _ovp = os.path.join(PROJECT_ROOT, "config", "inputs", "gateway_volume_overrides.json")
                                if os.path.exists(_ovp):
                                    _ov = _json.load(open(_ovp))
                                    for _gw, _cfg in (_ov.items() if isinstance(_ov, dict) else []):
                                        if isinstance(_cfg, dict) and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                                                and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
                                            _skip.add(str(_cg_ex(_gw)).strip().lower())
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
                            log(f"   exploration seeding: {len(_sr_rows)} untested gateway cell(s) seeded "
                                f"(weak prior, attempts=0 → wide Thompson posterior): {_n_sib} from a same-"
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
                        _off_route = set()
                        for _gwid, _cfg in (_ovr_off.items() if isinstance(_ovr_off, dict) else []):
                            if isinstance(_cfg, dict) and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                                    and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
                                _off_route.add(str(_cg_off(_gwid)).strip().lower())
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
                    _N_VARIATIONS = 3   # dials evenly spaced 0→100 (0, 50, 100)
                    weights = [round(float(w), 2) for w in np.linspace(0.0, 1.0, _N_VARIATIONS)]
                    _progress(_f_eng, "Running engine…")   # adaptive ETA (see _f_* above)
                    _stage(f"④ Run {engine_key} engine across the Risk↔Conversion axis")
                    log(f"   {_N_VARIATIONS} dials: {', '.join(str(int(round(w * 100))) for w in weights)}")

                    mapping_df = orig_forecast[["rpgt", "currency", "bank"]].drop_duplicates()
                    mapping_df["parent_bank"] = mapping_df["bank"].map(bin_to_bank).fillna(mapping_df["bank"])
                    mapping_df = mapping_df.rename(columns={"rpgt": "orig_rpgt", "bank": "orig_bank"})

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
                    from routing_optimiser.optimiser import (enforce_mid_vamp_caps, enforce_mid_volume_caps,
                                                             vamp_frontier_lp)
                    from routing_optimiser.genetic_global import run_midtilt_ga as _run_midtilt_ga
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

                    def _scope_base(mid, rpgt, month):
                        d = _pp_tidy[_pp_tidy["_mid"] == str(mid).strip().lower()]
                        if rpgt is not None:
                            d = d[d["_rpgt"] == str(rpgt).strip().lower()]
                        if month is not None:
                            d = d[d["_per"] == int(month)]
                        else:
                            d = d[(d["_per"] >= 0) & (d["_per"] <= 3)]
                        return float(d["txn"].sum()), float(d["vamp"].sum())

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
                    # LEXICOGRAPHIC priority: weight each tier so far above the next that ANY prio-1
                    # violation outweighs the WHOLE of prio-2 (etc.). The greedy/LP minimise this
                    # priority-weighted badness, so they now satisfy ALL prio-1 bands first and treat
                    # lower priorities as a tiebreak — instead of trading a prio-1 to shave a prio-2.
                    # _PRIORITY_GAP just needs to exceed the largest achievable within-tier violation
                    # (bounded by the constraint count, ≤ ~100); 1e4 is comfortably above that.
                    _PRIORITY_GAP = 1.0e4
                    _prio_lookup, _prio_by_mid = {}, {}
                    for _rec in _all_rules:
                        _pmk = str(_rec.get("vampMid", "")).strip().lower()
                        _pp = int(_rec.get("priority", 1) or 1)
                        _prio_lookup[(_pmk, _rec.get("month"), _rec.get("metric", "txn"))] = _pp
                        _prio_by_mid[_pmk] = min(_prio_by_mid.get(_pmk, 99), _pp)
                    def _prio_mult(_p):
                        # p1 → 1, p2 → 1e-4, p3 → 1e-8, … (higher priority = far larger weight).
                        return float(_PRIORITY_GAP ** (1 - max(int(_p), 1)))
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

                    def _mids_over(shares):
                        if vamp_cap is None:
                            return 0
                        vol = _cv * np.asarray(shares, float)
                        t = pd.DataFrame({"m": _vm, "vol": vol, "vr": vol * _rt}).groupby("m").sum()
                        gr = t["vr"] / t["vol"].replace(0, np.nan)
                        return int((gr > float(vamp_cap) + 1e-9).sum())

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

                    def _apply_rpgt_caps(gran):
                        # RPGT-scoped per-MID caps: scale each (vampMid, RPGT) group on the
                        # exploded split down to its allowed volume ratio, WITHIN that RPGT.
                        # Reuses the MID volume scaler with a composite (mid||rpgt) identity
                        # and cell = rpgt|currency|bin, so redistribution stays inside the
                        # RPGT and other RPGTs of the MID are untouched.
                        if not a_max_by_key or gran is None or getattr(gran, "empty", True):
                            return gran
                        gg = gran.copy()
                        _gw = gg["gateway"].astype(str).str.strip().str.lower()
                        _vm = _gw.map(_fid2vamp_l).fillna(_gw)
                        _rp = gg["rpgt"].astype(str).str.strip().str.lower()
                        gg["_ck"] = _vm + "||" + _rp
                        gg["_cell"] = (_rp + "|" + gg["currency"].astype(str).str.lower()
                                       + "|" + gg["bank"].astype(str).str.lower())
                        gg["_cvol"] = pd.to_numeric(gg.get("cell_volume", 0.0), errors="coerce").fillna(0.0)
                        gg["_rrate"] = pd.to_numeric(gg.get("gateway_risk_rate", 0.006), errors="coerce").fillna(0.006)
                        _amax_ck = {f"{m}||{r}": a for (m, r), a in a_max_by_key.items()}
                        _in = gg[["_cell", "gateway", "_ck", "_cvol", "baseline_share", "share", "_rrate"]].rename(
                            columns={"_cell": "cell", "_ck": "vampMid", "_cvol": "cell_vol", "_rrate": "rate"})
                        _capd, _cst = enforce_mid_volume_caps(_in, _amax_ck, max_share=float(max_share))
                        _rpgt_constrained.update(_cst)
                        gg["share"] = _capd["share"].to_numpy()
                        return gg.drop(columns=["_ck", "_cell", "_cvol", "_rrate"])

                    # STAGE 4 — feed the backup catch-all into the GA FITNESS so the engine optimises
                    # against the shares the pipeline will ACTUALLY route (tab 5), not the raw split.
                    # The catch-all is pooled (unweighted mean over pmp/Country) to the optimiser's
                    # coarser currency×bank×rpgt grain and mapped fid→vampMid ONCE here; the exact
                    # per-pmp/Country blend is applied in the tab-3 projection (Stage 3). Empty ⇒
                    # no blend. ROUTING_BACKUP_BLEND=0 disables. NOTE: this steers the GA's band/
                    # badness objective; the hard VAMP-cap enforcement still runs on the raw split.
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

                    def _scale_mids_in_gran(gran, scales):
                        # Scale each listed vampMid's aggregate share in every cell by its
                        # factor (>1 up, <1 down), moving the delta to/from the OTHER gateways
                        # in the cell (cell sum preserved). When REDUCING a MID (#4, cost-aware),
                        # the freed volume is redistributed to the OTHER gateways weighted by their
                        # incremental-revenue rate (so it flows to the best converters × ticket)
                        # instead of by current share; falls back to proportional if no rate data.
                        # WHERE the cut is taken across the MID's cells: uniform by default; when
                        # _cost_order["on"], cost-ordered — cut the cheapest cells first (#4b).
                        # Speedup 3: the per-row STRUCTURE (vampMid, cell code, incremental-revenue
                        # rate, cell_volume) depends only on the gran row layout — not the shares that
                        # change each call — so cache it on a layout signature. Repeated LP/greedy calls
                        # then skip the string ops + the 78k-row _ir loop, and the per-MID cell sums use
                        # np.bincount instead of groupby+map. Numerically identical to the old version.
                        _sig_c = ["gateway", "currency", "bank", "rpgt"] + (
                            ["cell_volume"] if "cell_volume" in gran.columns else [])
                        _sig = hash(pd.util.hash_pandas_object(gran[_sig_c], index=False).to_numpy().tobytes())
                        _st = _smig_cache.get(_sig)
                        if _st is None:
                            _vm_arr = (gran["gateway"].astype(str).str.strip().str.lower().map(_fid2vamp_l)
                                       .fillna(gran["gateway"].astype(str).str.strip().str.lower())).to_numpy()
                            _cells_arr = (gran["rpgt"].astype(str).str.lower() + "|"
                                          + gran["currency"].astype(str).str.lower() + "|"
                                          + gran["bank"].astype(str).str.lower()).to_numpy()
                            _cell_codes, _cuniq = pd.factorize(_cells_arr)
                            _n_cells = len(_cuniq)
                            _have_cv = "cell_volume" in gran.columns
                            _cvr = (pd.to_numeric(gran["cell_volume"], errors="coerce").fillna(0.0).to_numpy()
                                    if _have_cv else None)
                            _ir = None
                            if _inc_rev_joined:
                                _irk = (gran["currency"].astype(str).str.strip().str.lower() + "|"
                                        + gran["bank"].astype(str).str.strip().str.lower() + "|"
                                        + gran["gateway"].astype(str).str.strip().str.lower())
                                _ir = _irk.map(_inc_rev_joined).fillna(0.0).to_numpy(float)
                            _st = (_vm_arr, _cells_arr, _cell_codes, _n_cells, _have_cv, _cvr, _ir)
                            if len(_smig_cache) >= 32:
                                _smig_cache.pop(next(iter(_smig_cache)))
                            _smig_cache[_sig] = _st
                        _vm_arr, _cells_arr, _cell_codes, _n_cells, _have_cv, _cvr, _ir = _st

                        _sh = gran["share"].to_numpy(float).copy()
                        for _mid, _f in scales.items():
                            if abs(float(_f) - 1.0) < 1e-12:
                                continue
                            _ism = (_vm_arr == _mid)
                            if not _ism.any():
                                continue
                            _mt = np.bincount(_cell_codes, weights=_sh * _ism, minlength=_n_cells)[_cell_codes]
                            _ct = np.bincount(_cell_codes, weights=_sh, minlength=_n_cells)[_cell_codes]
                            _ot = _ct - _mt
                            _target = np.where(_ot > 1e-12, np.minimum(float(_f) * _mt, np.maximum(_ct - 1e-12, 0.0)), _mt)
                            # ---- #4b: COST-ORDERED cross-cell cut (only on the DOWN path) ----------
                            # Same TOTAL reduction as uniform, but taken from the cells where cutting
                            # costs the least revenue first: cost = MID-rate − best-alternative-rate in
                            # the cell (cut low/negative-cost cells first; keep the expensive ones). A
                            # cell with no alternative gateway is un-cuttable (kept). Deterministic in
                            # _f, so the joint-LP / greedy finite-difference through it unchanged.
                            if (_cost_order["on"] and _have_cv and _ir is not None
                                    and float(_f) < 1.0 - 1e-12):
                                _mmc = np.bincount(_cell_codes, weights=_sh * _ism, minlength=_n_cells)
                                _omc = np.bincount(_cell_codes, weights=_sh * (~_ism), minlength=_n_cells)
                                _shirc = np.bincount(_cell_codes, weights=_sh * _ism * _ir, minlength=_n_cells)
                                # first-occurrence cell_volume per cell (matches the old agg "first";
                                # reverse-assign so the first row's value wins). cell_volume is constant
                                # within a cell in real data, so this only guards a degenerate input.
                                _cvc = np.zeros(_n_cells); _cvc[_cell_codes[::-1]] = _cvr[::-1]
                                _altc = np.full(_n_cells, -np.inf)
                                np.maximum.at(_altc, _cell_codes, np.where(~_ism, _ir, -np.inf))
                                _hasmid = _mmc > 1e-12
                                _cap = _cvc * _mmc
                                _cut_ok = (_omc > 1e-12) & _hasmid
                                _mrate = np.where(_mmc > 0, _shirc / np.where(_mmc > 0, _mmc, 1.0), 0.0)
                                _altf = np.where(np.isfinite(_altc), _altc, 0.0)
                                _cost = _mrate - _altf              # revenue lost per unit cut
                                _Vkeep = float(_f) * float(_cap[_hasmid].sum())
                                _lock = float(_cap[_hasmid & ~_cut_ok].sum())
                                _rem = max(_Vkeep - _lock, 0.0)    # volume the cuttable cells keep
                                _kappa_arr = np.ones(_n_cells)     # default keep (locked / non-MID cells)
                                for _c in [j for j in np.argsort(-_cost) if _cut_ok[j]]:
                                    _take = min(_cap[_c], _rem)
                                    _kappa_arr[_c] = (_take / _cap[_c]) if _cap[_c] > 0 else 1.0
                                    _rem -= _take
                                _kap = _kappa_arr[_cell_codes]
                                _target = np.where(_mt > 1e-12, _kap * _mt, _target)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                _fm = np.where(_mt > 1e-12, _target / _mt, 1.0)
                                _fo = np.where(_ot > 1e-12, (_ct - _target) / _ot, 1.0)
                            # DEFAULT: proportional-to-current-share redistribution.
                            _sh_next = np.where(_ism, _sh * _fm, _sh * _fo)
                            # COST-AWARE (#4): when REDUCING the MID, give the freed volume to the OTHER
                            # gateways weighted by incremental revenue (best converters × ticket first),
                            # per cell — only where the cell's others carry positive revenue weight.
                            if _ir is not None and float(_f) < 1.0:
                                _wr = np.where(~_ism, np.maximum(_ir, 0.0), 0.0)
                                _wsum = np.bincount(_cell_codes, weights=_wr, minlength=_n_cells)[_cell_codes]
                                _freed = _mt - _target                       # per-cell amount to reallocate
                                _ok = (~_ism) & (_wsum > 1e-12) & (_ot > 1e-12) & (_freed > 1e-15)
                                _gain = np.where(_ok, _freed * (_wr / np.where(_wsum > 0, _wsum, 1.0)), 0.0)
                                # Apply revenue-weighted gain only in cells whose others have weight;
                                # elsewhere keep the proportional result. MID rows unchanged (= target).
                                _cell_ok = np.bincount(_cell_codes, weights=_ok.astype(float),
                                                       minlength=_n_cells)[_cell_codes] > 0
                                _sh_next = np.where(_ism, _sh * _fm,
                                                    np.where(_cell_ok, _sh + _gain, _sh * _fo))
                            _sh = _sh_next
                        _out = gran.copy()
                        _out["share"] = _sh
                        return _out

                    _midband_warned = set()   # MID-sets already warned about (collapse repeats)

                    def _midband_need(_proj):
                        # From a projection {(mid,period):[vamp,txn]}, return (scale-needs, badness).
                        # scale-needs {mid: factor} brings each violating band to its edge; badness is
                        # the scalar total violation (Σ shortfall) — the SAME metric the greedy uses,
                        # so LP and greedy results are directly comparable.
                        _need = {}
                        for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                            _months = [int(_mo)] if _mo is not None else list(range(4))
                            _tol = float(_tl) if _tl is not None else 0.0
                            # constraint TYPE gates which edge binds: ceiling → upper only,
                            # floor → lower only, range → both.
                            _hi = _tg * (1.0 + _tol) if _dir in ("range", "ceiling") else np.inf
                            _lo = _tg * (1.0 - _tol) if _dir in ("range", "floor") else 0.0
                            if _mtr in ("txn", "vamp"):
                                _ixm = 1 if _mtr == "txn" else 0
                                _cur = float(sum(_proj.get((_mk, m), (0.0, 0.0))[_ixm] for m in _months))
                                if _cur <= 0:
                                    continue
                                _e = _need.setdefault(_mk, [np.inf, 0.0])
                                if np.isfinite(_hi) and _cur > _hi + 1e-9:
                                    _e[0] = min(_e[0], _hi / _cur)
                                if _lo > 0 and _cur < _lo - 1e-9:
                                    _e[1] = max(_e[1], _lo / _cur)
                            else:  # vamp_pct — intrinsic rate; retire if over (ceiling only)
                                _tx = float(sum(_proj.get((_mk, m), (0.0, 0.0))[1] for m in _months))
                                _vv = float(sum(_proj.get((_mk, m), (0.0, 0.0))[0] for m in _months))
                                if _tx > 0 and (_vv / _tx) > (_tg / 100.0) + 1e-9:
                                    _need.setdefault(_mk, [np.inf, 0.0])[0] = 0.0
                        _sc, _bad = {}, 0.0
                        for _mk, (_fhi, _flo) in _need.items():
                            # PRIORITY weight: a violation on a low-priority (high-number) MID counts
                            # less toward "badness", so the greedy keeps the split that best satisfies
                            # the HIGH-priority MIDs when they can't all be met.
                            _pw = _prio_mult(_prio_by_mid.get(_mk, 1))
                            if _fhi < 1.0 - 1e-6:
                                _sc[_mk] = _fhi; _bad += _pw * (1.0 - _fhi)
                            elif _flo > 1.0 + 1e-6:
                                _sc[_mk] = _flo; _bad += _pw * (_flo - 1.0)
                        return _sc, _bad

                    def _mid_caps_greedy(gran):
                        # SAFETY FALLBACK (original heuristic): greedily scale each out-of-band MID
                        # and re-project, one at a time, tracking the least-violating split and
                        # stopping on a plateau/oscillation. Returns (best_gran, best_bad, last_sc).
                        _best_gran, _best_bad, _nostep, _sc = gran, np.inf, 0, {}
                        for _ in range(8):
                            _sc, _bad = _midband_need(_project_capped(_prop_items_from_gran(gran)))
                            if _bad < _best_bad - 1e-4:
                                _best_bad, _best_gran, _nostep = _bad, gran, 0
                            else:
                                _nostep += 1
                            if not _sc:
                                return gran, 0.0, {}
                            if _nostep >= 2:
                                break
                            gran = _scale_mids_in_gran(gran, _sc)
                            _mid_gran_constrained.update(_sc.keys())
                        return _best_gran, _best_bad, _sc

                    def _mid_caps_lp(gran0, s0=None):
                        # PRIMARY: joint solve. Pick ONE scale factor per constrained MID that
                        # satisfies ALL bands at once (closest to the reference), instead of greedily
                        # one-at-a-time. Each round: linearise the projected VAMP/Txn around the
                        # current scales (finite differences → a Jacobian), solve a small LP
                        # (minimise total movement s.t. the linearised bands + [0,3] bounds), then
                        # re-linearise (trust-region-capped step, ≤3 rounds). Returns (best_gran,
                        # best_bad, still_sc, moved). Returns badness=inf (→ greedy fallback) if
                        # SciPy/linprog is missing/errors.
                        #   s0 (#28/#2): optional {mid: scale} seed — e.g. the greedy result — so the
                        #   FIRST linearisation is taken around a near-feasible point rather than the
                        #   reference (all-ones), which the finite-difference Jacobian approximates far
                        #   better when the bands are tight.
                        try:
                            from scipy.optimize import linprog
                        except Exception:  # noqa: BLE001
                            return gran0, np.inf, {}, set()
                        _mids = sorted({r[0] for r in _mid_month_rules})
                        if not _mids:
                            return gran0, 0.0, {}, set()
                        n = len(_mids); _ixm = {m: i for i, m in enumerate(_mids)}
                        s = (np.ones(n) if not s0
                             else np.clip(np.array([float(s0.get(m, 1.0)) for m in _mids], dtype=float), 0.0, 3.0))
                        _eps = 1e-3
                        _best_gran, _best_bad, _best_sc = gran0, np.inf, {}

                        def _proj_of(_svec):
                            _g = _scale_mids_in_gran(gran0, {m: float(_svec[_ixm[m]]) for m in _mids})
                            return _g, _project_capped(_prop_items_from_gran(_g))

                        def _rule_val(_proj, _mk, _months, _mtr, _tg):
                            _v = sum(_proj.get((_mk, m), (0.0, 0.0))[0] for m in _months)
                            _t = sum(_proj.get((_mk, m), (0.0, 0.0))[1] for m in _months)
                            if _mtr == "txn":
                                return _t
                            if _mtr == "vamp":
                                return _v
                            return _v - (_tg / 100.0) * _t   # vamp_pct constraint: value <= 0

                        try:
                            for _outer in range(3):
                                _g0, _p0 = _proj_of(s)
                                _sc0, _bad0 = _midband_need(_p0)
                                if _bad0 < _best_bad - 1e-9:
                                    _best_bad, _best_gran, _best_sc = _bad0, _g0, _sc0
                                if not _sc0:
                                    return _g0, 0.0, {}, {m for m in _mids if abs(s[_ixm[m]] - 1.0) > 1e-6}
                                # Jacobian perturbations (speedup 4): the n one-MID-perturbed
                                # projections are independent. The base _proj_of(s) above has already
                                # populated the structural + projection caches for this gran layout
                                # (read-only hits hereafter), so we run the perturbations across a
                                # small THREAD pool with a cache-free projection — no shared-dict
                                # races, results gathered in k-order → bit-identical to the serial
                                # loop. Falls back to serial on any error.
                                def _pert_proj(_k):
                                    _sp = s.copy(); _sp[_k] += _eps
                                    _g = _scale_mids_in_gran(gran0, {m: float(_sp[_ixm[m]]) for m in _mids})
                                    return _project_capped(_prop_items_from_gran(_g), _use_cache=False)
                                try:
                                    from concurrent.futures import ThreadPoolExecutor as _TPE
                                    if n > 1:
                                        with _TPE(max_workers=min(4, n)) as _ex:
                                            _pert = list(_ex.map(_pert_proj, range(n)))
                                    else:
                                        _pert = [_pert_proj(0)]
                                except Exception:  # noqa: BLE001  (serial fallback — identical result)
                                    _pert = [_pert_proj(k) for k in range(n)]
                                _A, _bb = [], []
                                for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                    _months = [int(_mo)] if _mo is not None else list(range(4))
                                    _tol = float(_tl) if _tl is not None else 0.0
                                    _g0v = _rule_val(_p0, _mk, _months, _mtr, _tg)
                                    _J = np.array([(_rule_val(_pert[k], _mk, _months, _mtr, _tg) - _g0v) / _eps
                                                   for k in range(n)])
                                    if _mtr == "vamp_pct":
                                        _A.append(_J); _bb.append(0.0 - _g0v + _J.dot(s))
                                    else:
                                        # ceiling row only for range/ceiling; floor row only for range/floor.
                                        _hi = _tg * (1.0 + _tol) if _dir in ("range", "ceiling") else None
                                        _lo = _tg * (1.0 - _tol) if _dir in ("range", "floor") else 0.0
                                        if _hi is not None:
                                            _A.append(_J); _bb.append(_hi - _g0v + _J.dot(s))
                                        if _lo > 0:
                                            _A.append(-_J); _bb.append(-_lo + _g0v - _J.dot(s))
                                _c = np.concatenate([np.zeros(n), np.ones(n)])   # min Σ u ≈ Σ|s-1|
                                _Au, _bu = [], []
                                for _row, _rb in zip(_A, _bb):
                                    _Au.append(np.concatenate([_row, np.zeros(n)])); _bu.append(_rb)
                                for k in range(n):
                                    _r1 = np.zeros(2 * n); _r1[k] = 1.0; _r1[n + k] = -1.0; _Au.append(_r1); _bu.append(1.0)
                                    _r2 = np.zeros(2 * n); _r2[k] = -1.0; _r2[n + k] = -1.0; _Au.append(_r2); _bu.append(-1.0)
                                _bounds = [(0.0, 3.0)] * n + [(0.0, None)] * n
                                _res = linprog(_c, A_ub=np.array(_Au), b_ub=np.array(_bu), bounds=_bounds, method="highs")
                                if not getattr(_res, "success", False):
                                    break
                                _snew = np.clip(np.asarray(_res.x[:n], dtype=float), 0.0, 3.0)
                                # Trust region: cap each round's step so a poor linearisation can't
                                # overshoot into a worse point (the loop keeps the best regardless).
                                _snew = s + np.clip(_snew - s, -1.0, 1.0)
                                _step = float(np.max(np.abs(_snew - s)))
                                s = _snew
                                if _step < 1e-4:
                                    break
                        except Exception as _e:  # noqa: BLE001
                            log(f"   [Warning] joint-LP per-MID solve errored ({_e}); using greedy fallback.")
                            return gran0, np.inf, {}, set()
                        _gf, _pf = _proj_of(s)
                        _scf, _badf = _midband_need(_pf)
                        if _badf < _best_bad - 1e-9:
                            _best_bad, _best_gran, _best_sc = _badf, _gf, _scf
                        _moved = {m for m in _mids if abs(s[_ixm[m]] - 1.0) > 1e-6}
                        return _best_gran, _best_bad, _best_sc, _moved

                    def _apply_mid_caps(gran):
                        # Per-MID month/aggregate caps on the PROJECTED VAMP/Txn (what tab 4 shows).
                        # Primary: a JOINT LP solve that picks every per-MID scale together. If it
                        # satisfies the bands, use it. Otherwise run the greedy heuristic too and
                        # KEEP WHICHEVER is least-violating — so the joint solve can only help.
                        if gran is None or getattr(gran, "empty", True):
                            return gran
                        if _T0 is not None and _mid_month_rules:
                            _lg, _lb, _lsc, _lmoved = _mid_caps_lp(gran)
                            if _lb <= 1e-6:                        # LP satisfied every band
                                _mid_gran_constrained.update(_lmoved)
                                if "lp_ok" not in _midband_warned:
                                    _midband_warned.add("lp_ok")
                                    log(f"   per-MID joint LP: SATISFIED all bands (moved {len(_lmoved)} MID(s)); "
                                        "greedy not needed.")
                                return _lg
                            _gg, _gb, _gsc = _mid_caps_greedy(gran)   # infeasible -> compare with greedy
                            # #28/#2: re-run the joint LP SEEDED from greedy's (near-feasible) per-MID
                            # scales, so its first linearisation starts close to the answer rather than
                            # at the reference. Then keep the LEAST-VIOLATING of {LP, greedy, greedy-
                            # seeded LP}. Badness is priority-weighted (lexicographic via _prio_mult), so
                            # "least-violating" already means "best on the prio-1 bands first". Taking the
                            # min can only match or beat the previous LP-vs-greedy pick.
                            _s0 = None
                            try:
                                _vm0 = (gran["gateway"].astype(str).str.strip().str.lower().map(_fid2vamp_l)
                                        .fillna(gran["gateway"].astype(str).str.strip().str.lower())).to_numpy()
                                _vmg = (_gg["gateway"].astype(str).str.strip().str.lower().map(_fid2vamp_l)
                                        .fillna(_gg["gateway"].astype(str).str.strip().str.lower())).to_numpy()
                                _sh0 = gran["share"].to_numpy(float); _shg = _gg["share"].to_numpy(float)
                                _s0 = {}
                                for _m in sorted({r[0] for r in _mid_month_rules}):
                                    _t0 = float(_sh0[_vm0 == _m].sum())
                                    _s0[_m] = (float(_shg[_vmg == _m].sum()) / _t0) if _t0 > 1e-12 else 1.0
                            except Exception:  # noqa: BLE001
                                _s0 = None
                            _l2g, _l2b, _l2sc, _l2moved = (_mid_caps_lp(gran, s0=_s0) if _s0
                                                           else (gran, np.inf, {}, set()))
                            _cands = [(_lb, _lg, _lsc, _lmoved, "joint LP"),
                                      (_gb, _gg, _gsc, set(), "greedy"),
                                      (_l2b, _l2g, _l2sc, _l2moved, "joint LP (greedy-seeded)")]
                            _cb, _final, _fsc, _fmoved, _method = min(_cands, key=lambda c: c[0])
                            if _fmoved:
                                _mid_gran_constrained.update(_fmoved)
                            if "lp_cmp" not in _midband_warned:
                                _midband_warned.add("lp_cmp")
                                log(f"   per-MID solve residuals: LP {_lb:.1f} · greedy {_gb:.1f} · "
                                    f"greedy-seeded LP {_l2b:.1f} → using {_method} ({_cb:.1f}). If none "
                                    "reaches 0 the band set is jointly INFEASIBLE with the VAMP cap on this "
                                    "data (not a solver miss).")
                            if _fsc:
                                _wkey = frozenset(str(k) for k in _fsc)
                                if _wkey not in _midband_warned:
                                    _midband_warned.add(_wkey)
                                    log(f"   [Warning] per-MID band did not fully converge ({_method}); "
                                        f"{len(_fsc)} MID(s) cannot satisfy their target band together with the "
                                        "VAMP cap. Per-MID detail (projected vs target; minimal band-widening to clear "
                                        "this MID in isolation):")
                                    try:
                                        _pf = _project_capped(_prop_items_from_gran(_final))
                                        _seen_r = set()
                                        for (_mk, _mo, _mtr, _tg, _tl, _dir) in _mid_month_rules:
                                            if _mk not in _fsc:
                                                continue
                                            _months = [int(_mo)] if _mo is not None else list(range(4))
                                            _tol = float(_tl) if _tl is not None else 0.0
                                            _mostr = f"M{int(_mo)}" if _mo is not None else "M0-3"
                                            _prio = _prio_by_mid.get(_mk, 1)
                                            _rk = (_mk, _mostr, _mtr, _dir)
                                            if _rk in _seen_r:
                                                continue
                                            _seen_r.add(_rk)
                                            if _mtr in ("txn", "vamp"):
                                                _ix = 1 if _mtr == "txn" else 0
                                                _cur = float(sum(_pf.get((_mk, m), (0.0, 0.0))[_ix] for m in _months))
                                                _hi = _tg * (1.0 + _tol); _lo = _tg * (1.0 - _tol)
                                                if _dir in ("range", "ceiling") and _cur > _hi + 1e-9:
                                                    log(f"        • {_mk} · {_mtr} {_mostr} · prio {_prio} · projected "
                                                        f"{_cur:,.0f} > ceiling {_hi:,.0f} → widen target to ≥ {_cur:,.0f} "
                                                        f"(+{_cur - _hi:,.0f}) to clear")
                                                elif _dir in ("range", "floor") and _cur < _lo - 1e-9:
                                                    log(f"        • {_mk} · {_mtr} {_mostr} · prio {_prio} · projected "
                                                        f"{_cur:,.0f} < floor {_lo:,.0f} → lower target to ≤ {_cur:,.0f} "
                                                        f"(−{_lo - _cur:,.0f}) to clear")
                                            else:   # vamp_pct — intrinsic VAMP rate ceiling
                                                _v = float(sum(_pf.get((_mk, m), (0.0, 0.0))[0] for m in _months))
                                                _t = float(sum(_pf.get((_mk, m), (0.0, 0.0))[1] for m in _months))
                                                _rate = (_v / _t) if _t > 0 else 0.0
                                                if _rate > (_tg / 100.0) + 1e-9:
                                                    log(f"        • {_mk} · vamp% {_mostr} · prio {_prio} · projected "
                                                        f"{_rate*100:.2f}% > ceiling {_tg:.2f}% → raise ceiling to ≥ "
                                                        f"{_rate*100:.2f}% to clear")
                                    except Exception as _de:  # noqa: BLE001
                                        log(f"        (per-MID diagnostic unavailable: {type(_de).__name__}: {_de})")
                            return _final
                        # Fallback: routing-space MID-total scale (no pro-rata; ceilings only).
                        if a_max_by_mid:
                            gg = gran.copy()
                            _vm = gg["gateway"].astype(str).str.strip().str.lower().map(_fid2vamp_l).fillna(
                                gg["gateway"].astype(str).str.strip().str.lower())
                            _in = pd.DataFrame({
                                "cell": (gg["rpgt"].astype(str).str.lower() + "|" + gg["currency"].astype(str).str.lower()
                                         + "|" + gg["bank"].astype(str).str.lower()),
                                "gateway": gg["gateway"].astype(str), "vampMid": _vm,
                                "cell_vol": pd.to_numeric(gg.get("cell_volume", 0.0), errors="coerce").fillna(0.0),
                                "baseline_share": gg["baseline_share"].to_numpy(float), "share": gg["share"].to_numpy(float),
                                "rate": pd.to_numeric(gg.get("gateway_risk_rate", 0.006), errors="coerce").fillna(0.006)})
                            _capd, _cst = enforce_mid_volume_caps(_in, a_max_by_mid, max_share=float(max_share))
                            _mid_gran_constrained.update(_cst)
                            gg["share"] = _capd["share"].to_numpy()
                            return gg
                        return gran

                    _phase_logged = {"done": False, "n": 0}

                    # ---- shared enforcement scorers (used by the #6 iterate-selection inside
                    # _enforce_once AND the #8 reference-feedback loop in _restrict_and_recap) ----
                    _VAMP_W = 1.0e9   # VAMP-cap compliance ranks far above every per-MID band residual

                    def _combined_bad(_g):
                        # ONE priority-weighted badness: VAMP-cap-over count (weighted huge) + per-MID
                        # band badness (itself prio-1 ≫ prio-2 via _prio_mult). Lower is better.
                        _vo = 0
                        _bb = 0.0
                        try:
                            if _mid_month_rules or (_bpool_rpgt or _bpool_all):
                                _proj = _project_capped(_prop_items_from_gran(_g))
                                if _mid_month_rules:
                                    _bb = float(_midband_need(_proj)[1])
                                if vamp_cap is not None and (_bpool_rpgt or _bpool_all):
                                    from collections import defaultdict as _ddc
                                    _vv, _tt = _ddc(float), _ddc(float)
                                    for (_mk, _per), _val in _proj.items():
                                        _vv[_mk] += float(_val[0]); _tt[_mk] += float(_val[1])
                                    _vo = sum(1 for _mk in _tt
                                              if _tt[_mk] > 0 and _vv[_mk] / _tt[_mk] > float(vamp_cap) + 1e-9)
                            if vamp_cap is not None and not (_bpool_rpgt or _bpool_all):
                                _vo = int(_mids_over_granular(_g))
                        except Exception:  # noqa: BLE001
                            return float("inf")   # unscoreable → never preferred
                        return _VAMP_W * float(_vo) + _bb

                    def _band_bad(_g):
                        # Just the per-MID band residual (0 = every band satisfied by the projection).
                        if not _mid_month_rules:
                            return 0.0
                        try:
                            return float(_midband_need(_project_capped(_prop_items_from_gran(_g)))[1])
                        except Exception:  # noqa: BLE001
                            return 0.0

                    def _rev_proxy(_g):
                        # Expected incremental revenue Σ cell_volume·share·(success_rate·ticket), using the
                        # #4a per-(currency,bank,gateway) incremental-revenue rates. Guards the #8 feedback
                        # against a compliance gain that quietly costs revenue. Falls back to volume·share
                        # (a monotone proxy) when no rate data — still forbids gross revenue loss.
                        if _g is None or getattr(_g, "empty", True):
                            return 0.0
                        _cv = pd.to_numeric(_g.get("cell_volume", 0.0), errors="coerce").fillna(0.0).to_numpy()
                        _sh = pd.to_numeric(_g["share"], errors="coerce").fillna(0.0).to_numpy()
                        if not _inc_rev_rate:
                            return float((_cv * _sh).sum())
                        _rr = np.array([float(_inc_rev_rate.get(
                            (str(c).strip().lower(), str(b).strip().lower(), str(gw).strip().lower()), 0.0))
                            for c, b, gw in zip(_g["currency"], _g["bank"], _g["gateway"])], float)
                        return float((_cv * _sh * _rr).sum())

                    def _cost_order_worth(_g):
                        # Cheap gate for the #4b A/B: only worth an extra enforcement solve if some
                        # CONSTRAINED MID spans ≥2 cells with DIFFERENT incremental-revenue rates —
                        # otherwise cost-ordered cutting == uniform and there's nothing to gain.
                        try:
                            if not _mid_gran_constrained or not _inc_rev_rate:
                                return False
                            _gg = _g.copy()
                            _vm = (_gg["gateway"].astype(str).str.strip().str.lower().map(_fid2vamp_l)
                                   .fillna(_gg["gateway"].astype(str).str.strip().str.lower()))
                            _cell = (_gg["rpgt"].astype(str).str.lower() + "|" + _gg["currency"].astype(str).str.lower()
                                     + "|" + _gg["bank"].astype(str).str.lower())
                            _irv = np.array([float(_inc_rev_rate.get(
                                (str(c).strip().lower(), str(b).strip().lower(), str(gw).strip().lower()), 0.0))
                                for c, b, gw in zip(_gg["currency"], _gg["bank"], _gg["gateway"])], float)
                            _t = pd.DataFrame({"vm": _vm.to_numpy(), "cell": _cell.to_numpy(), "ir": _irv})
                            for _m in _mid_gran_constrained:
                                _sub = _t[_t["vm"] == _m]
                                if _sub["cell"].nunique() < 2:
                                    continue
                                _per = _sub.groupby("cell")["ir"].mean()
                                if float(_per.max() - _per.min()) > 1e-9 * max(abs(float(_per.max())), 1.0):
                                    return True
                            return False
                        except Exception:  # noqa: BLE001
                            return False

                    def _enforce_once(gran):
                        import time as _time
                        _phase_logged["n"] += 1
                        _tlog = True   # log the phase breakdown for EVERY enforcement pass
                        _t = _time.time()
                        gran = _restrict(gran)
                        if gran is None or getattr(gran, "empty", True):
                            return gran
                        _t_elig = _time.time() - _t; _t = _time.time()
                        if vamp_cap is not None:
                            for _ in range(2):
                                g = _mid_cap_granular(gran)
                                _in = g[["cell", "gateway", "_vm", "cell_vol", "rate", "share"]].rename(columns={"_vm": "vampMid"})
                                _capd, _rr, _ss2 = enforce_mid_vamp_caps(
                                    _in, cap=float(vamp_cap), floor=float(floor), max_share=float(max_share))
                                _g2 = gran.copy()
                                _g2["share"] = _capd["share"].to_numpy()
                                _g2 = _restrict(_g2)
                                if np.allclose(_g2["share"].to_numpy(), gran["share"].to_numpy(), atol=1e-6):
                                    gran = _g2
                                    break
                                gran = _g2
                        _t_vamp = _time.time() - _t
                        # Per-MID caps, RPGT caps and the VAMP cap can each re-breach the others
                        # (per-MID redistribution can push a MID back over VAMP; a VAMP shave can
                        # push a MID out of its band). Iterate the trio to a FIXED POINT — bounded
                        # (max 3), deterministic, compliance-improving — stopping as soon as the
                        # split stops moving, instead of a single pass. (G3 / one-pass fix)
                        _t_mid = _t_rpgt = _t_reenf = 0.0
                        _vamp_reenf = 0
                        # #6 (single priority-weighted objective): score each iterate with the hoisted
                        # _combined_bad (VAMP-cap compliance ≫ prio-1 bands ≫ prio-2 bands). The VAMP-cap
                        # and per-MID passes can oscillate, so KEEP THE LEAST-VIOLATING ITERATE, not the
                        # last one — can't regress (only picks among computed iterates).
                        _best_g, _best_cb = None, None
                        for _oi in range(3):
                            _prev = gran["share"].to_numpy().copy()
                            _t = _time.time()
                            gran = _apply_mid_caps(gran)
                            _t_mid += _time.time() - _t; _t = _time.time()
                            gran = _apply_rpgt_caps(gran)
                            _t_rpgt += _time.time() - _t; _t = _time.time()
                            if vamp_cap is not None and _mids_over_granular(gran) > 0:
                                g = _mid_cap_granular(gran)
                                _in = g[["cell", "gateway", "_vm", "cell_vol", "rate", "share"]].rename(columns={"_vm": "vampMid"})
                                _capd, _, _ = enforce_mid_vamp_caps(
                                    _in, cap=float(vamp_cap), floor=float(floor), max_share=float(max_share))
                                _g2 = gran.copy(); _g2["share"] = _capd["share"].to_numpy()
                                gran = _restrict(_g2)
                            _t_reenf += _time.time() - _t
                            _cb_it = _combined_bad(gran)
                            if _best_cb is None or _cb_it < _best_cb - 1e-12:
                                _best_cb, _best_g = _cb_it, gran.copy()
                            if np.allclose(gran["share"].to_numpy(), _prev, atol=1e-6):
                                break
                        if _best_g is not None:
                            if not np.allclose(_best_g["share"].to_numpy(), gran["share"].to_numpy(), atol=1e-9):
                                log("   per-MID/VAMP joint selection (#6): kept the least-violating iterate "
                                    "(VAMP-cap compliance first, then prio-1 ≫ prio-2 bands) — the passes "
                                    "oscillated, so the final pass was not the best.")
                            gran = _best_g
                        # BACKUP-BLEND enforcement: the passes above shaved the RAW split to the cap,
                        # but the pipeline re-adds the backup catch-all, which can push a MID back
                        # over. Iterate with a progressively TIGHTER raw cap until the ACTUAL routed
                        # (blended) VAMP is compliant, so the EXPORTED split stays under cap AFTER the
                        # re-adds. Conservative (only ever reduces further), reuses the tested
                        # enforce + projection. No-op without a backup folder / when disabled.
                        if (vamp_cap is not None and (_bpool_rpgt or _bpool_all)
                                and _mids_over_blended(gran) > 0):
                            _t = _time.time()
                            _cap_t = float(vamp_cap)
                            for _bt in range(6):
                                _cap_t *= 0.90
                                _gb = _mid_cap_granular(gran)
                                _inb = _gb[["cell", "gateway", "_vm", "cell_vol", "rate", "share"]].rename(columns={"_vm": "vampMid"})
                                _capdb, _, _ = enforce_mid_vamp_caps(
                                    _inb, cap=_cap_t, floor=float(floor), max_share=float(max_share))
                                _g2b = gran.copy(); _g2b["share"] = _capdb["share"].to_numpy()
                                gran = _restrict(_g2b)
                                if _mids_over_blended(gran) == 0:
                                    break
                            _t_reenf += _time.time() - _t
                            log(f"   backup-blend VAMP tightening: raw cap {float(vamp_cap):.4f} → "
                                f"{_cap_t:.4f} so the routed (blended) VAMP meets the "
                                f"{float(vamp_cap):.4f} cap ({_mids_over_blended(gran)} MID(s) still over).")
                        _vamp_reenf = (int(_mids_over_blended(gran))
                                       if (vamp_cap is not None and (_bpool_rpgt or _bpool_all))
                                       else (int(_mids_over_granular(gran)) if vamp_cap is not None else 0))
                        if _tlog:
                            _tot = _t_elig + _t_vamp + _t_mid + _t_rpgt + _t_reenf
                            log(f"   [timing · enforce pass #{_phase_logged['n']}] total {_tot:.1f}s = "
                                f"eligibility {_t_elig:.1f}s · VAMP-cap {_t_vamp:.1f}s · "
                                f"per-MID caps {_t_mid:.1f}s · RPGT caps {_t_rpgt:.1f}s · "
                                f"VAMP re-check {_t_reenf:.1f}s "
                                f"(rows={len(gran):,})")
                            if _vamp_reenf > 0:
                                log(f"   [Warning] {_vamp_reenf} MID(s) still over the VAMP cap after re-enforcement "
                                    "— per-MID targets and VAMP cap conflict; VAMP took priority.")
                            _phase_logged["done"] = True
                        return gran

                    def _restrict_and_recap(gran):
                        # #8 — ENGINE↔ENFORCEMENT γ FEEDBACK (safe, bounded, non-regressing).
                        # The per-cell engine reference is built with NO knowledge of the cross-cell
                        # per-MID bands, so enforcement alone has to drag chronically-over MIDs into
                        # their bands after the fact — fighting the reference and shedding revenue.
                        # Here we CLOSE THE LOOP: enforce once, read the residual per-MID band shadow
                        # price γ (the scale each still-violating MID needs), feed it back by tilting the
                        # REFERENCE SEED away from those MIDs' high-risk gateways (revenue-weighted
                        # redistribution, via _scale_mids_in_gran), then re-enforce from that better seed.
                        # A tilted seed starts closer to feasible, so enforcement distorts less.
                        # STRICTLY GUARDED: adopt a feedback round ONLY on a Pareto improvement — combined
                        # badness strictly lower AND revenue not meaningfully worse — so it can never
                        # regress on either compliance or revenue. Bounded to 2 rounds; a no-op (zero
                        # extra cost) whenever the first enforcement already satisfies the bands.
                        _base = _enforce_once(gran)
                        if (_base is None or getattr(_base, "empty", True) or not _mid_month_rules):
                            return _base
                        _bb0 = _combined_bad(_base)
                        _rev0 = _rev_proxy(_base)
                        _seed = gran                       # the reference seed we keep tilting
                        for _fb in range(2):
                            if _band_bad(_base) <= 1e-9:    # bands already met → nothing to feed back
                                break
                            try:
                                _needs = _midband_need(_project_capped(_prop_items_from_gran(_base)))[0]
                            except Exception:  # noqa: BLE001
                                break
                            _needs = {_m: float(_f) for _m, _f in (_needs or {}).items()
                                      if np.isfinite(_f) and abs(float(_f) - 1.0) > 1e-6}
                            if not _needs:
                                break
                            _seed_t = _scale_mids_in_gran(_seed, _needs)   # tilt reference by the γ needs
                            _cand = _enforce_once(_seed_t)
                            if _cand is None or getattr(_cand, "empty", True):
                                break
                            _bb1 = _combined_bad(_cand)
                            _rev1 = _rev_proxy(_cand)
                            # Pareto guard: strictly better compliance, no meaningful revenue loss.
                            if _bb1 < _bb0 - 1e-12 and _rev1 >= _rev0 - 1e-6 * max(abs(_rev0), 1.0):
                                log(f"   engine↔enforcement feedback (#8) round {_fb + 1}: reference tilted "
                                    f"off {len(_needs)} over-band MID(s) → badness {_bb0:.6g}→{_bb1:.6g}, "
                                    f"revenue ${_rev0:,.0f}→${_rev1:,.0f} (adopted).")
                                _base, _bb0, _rev0, _seed = _cand, _bb1, _rev1, _seed_t
                            else:
                                log(f"   engine↔enforcement feedback (#8) round {_fb + 1}: tilted seed did "
                                    f"not Pareto-improve (badness {_bb0:.6g}→{_bb1:.6g}, revenue "
                                    f"${_rev0:,.0f}→${_rev1:,.0f}) — kept the un-tilted result.")
                                break
                        # #4b — COST-ORDERED cross-cell cut (Pareto-guarded A/B). Uniform scaling
                        # cuts a MID by the SAME fraction in every cell; #4b takes the SAME total cut
                        # but from the cells where it costs the least revenue first (MID-rate − best-
                        # alternative-rate), keeping the expensive cells. Re-enforce with cost-ordering
                        # ON and adopt ONLY on a Pareto improvement — revenue up with no compliance
                        # loss, or better compliance with no revenue loss — so it can never regress.
                        # Gated to when a MID was actually cut AND its cells differ in cost (else the
                        # cost-ordered result equals uniform and the extra solve is skipped).
                        if _inc_rev_rate and _cost_order_worth(_base):
                            _cost_order["on"] = True
                            try:
                                _c4 = _enforce_once(gran)
                            except Exception as _e4:  # noqa: BLE001
                                _c4 = None
                                log(f"   cost-ordered cutting (#4b) errored ({_e4}); kept uniform.")
                            finally:
                                _cost_order["on"] = False
                            if _c4 is not None and not getattr(_c4, "empty", True):
                                _bb4 = _combined_bad(_c4)
                                _rev4 = _rev_proxy(_c4)
                                _eps_r = 1e-6 * max(abs(_rev0), 1.0)
                                _pareto = ((_rev4 > _rev0 + _eps_r and _bb4 <= _bb0 + 1e-9)
                                           or (_bb4 < _bb0 - 1e-12 and _rev4 >= _rev0 - _eps_r))
                                if _pareto:
                                    log(f"   cost-ordered cutting (#4b): adopted — cut cheapest cells first "
                                        f"(same total cut); revenue ${_rev0:,.0f}→${_rev4:,.0f}, badness "
                                        f"{_bb0:.6g}→{_bb4:.6g}.")
                                    _base, _bb0, _rev0 = _c4, _bb4, _rev4
                                else:
                                    log(f"   cost-ordered cutting (#4b): no Pareto gain (revenue "
                                        f"${_rev0:,.0f}→${_rev4:,.0f}, badness {_bb0:.6g}→{_bb4:.6g}) — "
                                        "kept uniform.")
                        return _base

                    def _mids_over_granular(gran):
                        if vamp_cap is None or gran is None or getattr(gran, "empty", True):
                            return 0
                        g = _mid_cap_granular(gran)
                        _v = g["cell_vol"] * g["share"]
                        _t = pd.DataFrame({"m": g["_vm"], "vol": _v, "vr": _v * g["rate"]}).groupby("m").sum()
                        _gr = _t["vr"] / _t["vol"].replace(0, np.nan)
                        return int((_gr > float(vamp_cap) + 1e-9).sum())

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

                    def _make_frontier_share(tmpl, rev_sh, comp_sh):
                        # TRUE FRONTIER (replaces the linear share blend for the intermediate dials).
                        # For each dial w, solve for the min-movement-from-the-REVENUE-reference split
                        # whose whole-book aggregate VAMP rate ≤ budget(w), where budget sweeps from
                        # the compliant endpoint's rate (w=0) up to the revenue endpoint's rate (w=1).
                        # Each point is Pareto-optimal (max revenue retention at that risk budget) and
                        # monotonic. Falls back to the linear blend if SciPy/HiGHS is missing, the LP
                        # is infeasible, or the frontier point isn't lower-risk than the blend — so it
                        # can only improve the middle dials, never regress them.
                        rev_sh = np.asarray(rev_sh, float); comp_sh = np.asarray(comp_sh, float)
                        _mc = None
                        try:
                            _m = _mid_cap_granular(tmpl)
                            _cv = pd.to_numeric(_m["cell_vol"], errors="coerce").fillna(0.0).to_numpy()
                            _rt = pd.to_numeric(_m["rate"], errors="coerce").fillna(0.0).to_numpy()
                            _inp0 = _m[["cell", "gateway", "_vm", "cell_vol", "rate"]].rename(columns={"_vm": "vampMid"})
                            _mc = True
                        except Exception:  # noqa: BLE001
                            _mc = None

                        def _aggr(sh):
                            v = _cv * np.asarray(sh, float); tot = float(v.sum())
                            return float((v * _rt).sum() / tot) if tot > 1e-12 else 0.0
                        _r_rev = _aggr(rev_sh) if _mc else 0.0
                        _r_comp = _aggr(comp_sh) if _mc else 0.0

                        def _fshare(w):
                            w = float(w)
                            if w >= 1.0 - 1e-9:
                                return rev_sh
                            if w <= 1e-9:
                                return comp_sh
                            _blend = w * rev_sh + (1.0 - w) * comp_sh
                            if not _mc or vamp_cap is None:
                                return _blend
                            _budget = _r_comp + w * (_r_rev - _r_comp)
                            _inp = _inp0.copy(); _inp["share"] = rev_sh
                            try:
                                _adj = vamp_frontier_lp(_inp, cap=float(vamp_cap), agg_cap=float(_budget),
                                                        floor=float(floor), max_share=float(max_share))
                            except Exception:  # noqa: BLE001
                                _adj = None
                            if _adj is None:
                                return _blend
                            _x = pd.to_numeric(_adj["share"], errors="coerce").fillna(0.0).to_numpy()
                            if _x.shape != rev_sh.shape or not np.isfinite(_x).all():
                                return _blend
                            # Keep the frontier point only if it's at least as low-risk as the blend
                            # (it targets ≤ budget AND min-moves from revenue, so it dominates/ties);
                            # otherwise fall back to the blend. Guarantees no regression.
                            return _x if _aggr(_x) <= _aggr(_blend) + 1e-9 else _blend
                        return _fshare

                    if engine_key == "genetic":
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
                        ctx = {
                            "n_row": len(G), "n_mid": len(_mids_u),
                            "cell_starts": _cell_starts, "cell_counts": _counts,
                            "elig": np.ones(len(G)),   # bans/wallet handled by the post-step
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
                            "breach_fixed": 250.0,   # big fixed hit (× MID revenue) on ANY breach, + quadratic (5× stronger)
                        }
                        # CROSS-CELL per-MID tilt GA (flaw #4 rebuild). The genome is ONE tilt per
                        # vampMid (~20 dims), which shifts a MID's volume from its HIGH-risk cells to
                        # its LOW-risk cells — directly controlling the per-MID CROSS-cell VAMP rate
                        # (the actual constraint) that the old per-cell tilt could never touch. Freed
                        # share redistributes per cell in proportion to the revenue reference (more
                        # revenue-efficient than greedy's lowest-rate dumping), so it can BEAT greedy
                        # on MIDs whose risk varies across cells — and it's tiny/fast (seconds, not
                        # the old 1h24m). dial 100 = softmax revenue reference; dial 0 = this GA's best
                        # compliant split; blended, both endpoints hard-enforced (2 VAMP solves).
                        ctx["ref_share"] = _ref_share_G          # θ=0 decode base (revenue-optimal)
                        _n_cells = int(len(_counts)); _n_mid = int(len(_mids_u))
                        _ga_pop = int(np.clip(round(4 * _n_mid), 30, 80))   # sized to the tiny genome
                        _ga_gen = 80
                        _ga_pat = 12
                        _N_SEED = 3          # multi-seed: keep the fittest of N random starts
                        _GA_GAIN_MAX = 3.5   # wider per-MID gain range (was 2.0) → more cross-MID reach
                        _rev_of = lambda _sh: float((np.asarray(_sh, float) * _rev_coef).sum())
                        log(f"   GA (cross-cell per-MID tilt, {_n_mid} vampMids): pop={_ga_pop}, gen cap={_ga_gen} "
                            "(adaptive σ + early stop). dial 0 = risk-MINIMISED compliant; dial 99 = max-revenue "
                            "compliant; blended between (monotonic frontier); dial 100 = uncapped revenue ceiling.")
                        def _ga_true_breach(_sh):
                            """Total RELATIVE band breach of the TRUE pro-rata projection for `_sh`
                            (0 ⇒ every month band satisfied by the REAL projection, not the proxy)."""
                            if not _mid_month_rules:
                                return 0.0
                            _agg = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                            _agg["share"] = np.asarray(_sh, float)
                            _agg["volume"] = _agg["cell_volume"] * _agg["share"]
                            _proj = _project_capped(_prop_items_from_gran(_explode(_agg)))
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

                        def _ga_solve_with_correction(_risk_min_w, _seed=42, _rounds=2, _band_w=8.0,
                                                      _warm=None, _band_fix=20.0):
                            """Run the tilt GA, then RE-PROJECT & CORRECT (like the greedy): re-anchor
                            the band proxy at the GA's own split via the TRUE projection and re-run,
                            accepting a round ONLY if the true-projection band breach actually drops.
                            Bounded (≤ _rounds) and no-regression. `_band_w` scales the per-MID band
                            penalty (tougher at the dial-0 risk-min endpoint). `_warm` seeds a prior
                            run's genome into the population (free reach). Returns (shares, info)."""
                            ctx["risk_min_w"] = float(_risk_min_w)
                            ctx["band_weight"] = float(_band_w)
                            ctx["band_fixed"] = float(_band_fix)
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
                            if int(_N_SEED) > 1:
                                try:
                                    import time as _st_t
                                    from joblib import Parallel, delayed
                                    # Backend is env-selectable: loky (processes, default — true
                                    # parallelism but pickles the ~600k-row ctx per worker) vs
                                    # threading (no pickling; the GA's heavy numpy/pandas ops release
                                    # the GIL). Both give byte-identical results (each seed is
                                    # deterministic); pick whichever is faster on this box.
                                    _ga_backend = os.environ.get("ROUTING_GA_PARALLEL_BACKEND", "loky").strip().lower()
                                    if _ga_backend not in ("loky", "threading", "multiprocessing"):
                                        _ga_backend = "loky"
                                    _t_par0 = _st_t.time()
                                    _seed_results = Parallel(
                                        n_jobs=min(int(_N_SEED), os.cpu_count() or 1),
                                        backend=_ga_backend)(
                                        delayed(_run_midtilt_ga)(
                                            ctx, lam=50.0, pop_size=_ga_pop, generations=_ga_gen,
                                            seed=_seed + _s, auto=True, patience=_ga_pat,
                                            warm_start=_warm, gain_max=_GA_GAIN_MAX)
                                        for _s in range(int(_N_SEED)))
                                    log(f"   multi-seed GA: {int(_N_SEED)} seeds in PARALLEL ({_ga_backend}, "
                                        f"{min(int(_N_SEED), os.cpu_count() or 1)} workers) in "
                                        f"{_st_t.time() - _t_par0:.1f}s.")
                                except Exception as _pe:  # noqa: BLE001
                                    log(f"   parallel multi-seed GA unavailable ({type(_pe).__name__}: "
                                        f"{_pe}); running seeds sequentially.")
                                    _seed_results = None
                            _sh, _info = None, None
                            if _seed_results is not None:
                                for _shc, _infoc in _seed_results:            # seed order preserved
                                    if _info is None or _infoc["best_fit"] > _info["best_fit"]:
                                        _sh, _info = _shc, _infoc
                            else:
                                import time as _st_t
                                _t_seq0 = _st_t.time()
                                for _s in range(int(_N_SEED)):
                                    _shc, _infoc = _run_midtilt_ga(
                                        ctx, lam=50.0, pop_size=_ga_pop, generations=_ga_gen,
                                        seed=_seed + _s, auto=True, patience=_ga_pat,
                                        warm_start=_warm, gain_max=_GA_GAIN_MAX)
                                    if _info is None or _infoc["best_fit"] > _info["best_fit"]:
                                        _sh, _info = _shc, _infoc
                                if int(_N_SEED) > 1:
                                    log(f"   multi-seed GA: {int(_N_SEED)} seeds SEQUENTIAL in "
                                        f"{_st_t.time() - _t_seq0:.1f}s (parallel unavailable).")
                            if ctx["midband"]:
                                try:
                                    _br = _ga_true_breach(_sh)
                                    for _r in range(int(_rounds)):
                                        if _br <= 1e-6:
                                            break   # real projection already satisfies every band
                                        ctx["midband"] = (_build_ga_bands(_sh) or None)   # re-project
                                        _sh2, _info2 = _run_midtilt_ga(ctx, lam=50.0, pop_size=_ga_pop,
                                                                       generations=_ga_gen, seed=_seed,
                                                                       auto=True, patience=_ga_pat,
                                                                       warm_start=_info.get("genome"),
                                                                       gain_max=_GA_GAIN_MAX)
                                        _br2 = _ga_true_breach(_sh2)
                                        if _br2 < _br - 1e-9:
                                            _sh, _info, _br = _sh2, _info2, _br2   # accept: truly better
                                        else:
                                            break
                                    log(f"   GA re-projection correction: true-band breach {_br:.4g} "
                                        "(0 = all month bands satisfied by the real pro-rata projection).")
                                except Exception as _e:  # noqa: BLE001
                                    log(f"   [Warning] GA re-projection correction skipped ({_e}); "
                                        "using proxy-only GA result.")
                            ctx["risk_min_w"] = 0.0
                            ctx["band_weight"] = 8.0   # reset to defaults for anything downstream
                            ctx["band_fixed"] = 20.0
                            return _sh, _info

                        import time as _gatime
                        _ga_wall0 = _gatime.time()
                        _progress(_f_eng, "GA revenue-max endpoint…")
                        _rep_comp_G, _inf = _ga_solve_with_correction(0.0)
                        _ga_wall = _gatime.time() - _ga_wall0
                        ss["ga_perf"] = {"secs": float(_ga_wall), "budget": int(_ga_pop * _ga_gen),
                                         "nvar": 1, "n": int(len(G))}
                        _save_ga_perf(ss["ga_perf"])   # survives restarts
                        # SAFETY: adopt the GA compliant endpoint only if it is aggregate-VAMP compliant
                        # AND retains ≥ the greedy shave's revenue — else fall back to greedy. No regression.
                        def _agg_mid_ok(_sh):
                            # Guard the GA endpoint on the aggregate per-vampMid VAMP-rate cap only.
                            # The standalone per-MID VOLUME cap was dropped — per-MID rules are now
                            # enforced via the month bands (projection space), not a routing-volume
                            # ceiling — so it's no longer part of the adoption guard.
                            _v = _cvol * np.asarray(_sh, float)
                            for _mi, _r in enumerate(_mid_rows):
                                _vol = float(_v[_r].sum())
                                if vamp_cap is not None and _vol > 1e-12:
                                    if float((_v[_r] * _rkr[_r]).sum()) / _vol > float(vamp_cap) + 1e-9:
                                        return False
                            return True
                        _use_rep = _agg_mid_ok(_rep_comp_G) and (_rev_of(_rep_comp_G) >= _rev_of(_comp_share_G) - 1e-6)
                        _comp_endpoint_G = _rep_comp_G if _use_rep else _comp_share_G   # dial 99 = max-revenue compliant
                        log(f"   GA revenue-max compliant endpoint in {_fmt_secs(_ga_wall)}: revenue GA "
                            f"${_rev_of(_rep_comp_G):,.0f} vs greedy ${_rev_of(_comp_share_G):,.0f} → using "
                            f"{'GA (better)' if _use_rep else 'greedy'}.")
                        _progress(_f_rmin, "GA risk-min endpoint…")
                        # SECOND GA run for the SAFE (dial-0) endpoint: same setup + a risk-minimisation
                        # term so it tilts each MID further toward its low-risk cells while staying
                        # compliant. mu is auto-scaled from the reference so the risk term is a bounded
                        # fraction (~risk_aversion) of reference revenue — trades some revenue for lower
                        # aggregate VAMP without degenerating (the θ-tilt keeps it revenue-shaped per cell).
                        _rev_ref = max(_rev_of(_ref_share_G), 1.0)
                        _vamp_ref = max(float((_cvol * _ref_share_G * _rkr).sum()), 1.0)
                        _risk_aversion = 0.5
                        _safe_wall0 = _gatime.time()
                        # risk-min endpoint (dial 0): same GA + re-projection correction, with the
                        # risk-min term AND a TOUGHER per-MID band penalty (4× the dial-99 weight) so
                        # dial 0 sits inside every band harder. Intermediate dials inherit this via the
                        # frontier blend between the dial-0 and dial-99 endpoints.
                        _safe_G, _inf2 = _ga_solve_with_correction(
                            _risk_aversion * _rev_ref / _vamp_ref, _band_w=3375.0, _band_fix=8100.0,
                            _warm=_inf.get("genome"))   # dial-0: bands ×5 again (675→3375, 1620→8100); warm-start
                        # (_ga_solve_with_correction resets risk_min_w / band_weight on exit.)
                        _ga_wall_tot = _ga_wall + (_gatime.time() - _safe_wall0)
                        # Use the risk-min GA for dial 0 only if it is compliant; else fall back to the
                        # revenue-max endpoint (dial 0 == dial 99, frontier collapses but never regresses).
                        _safe_endpoint_G = _safe_G if _agg_mid_ok(_safe_G) else _comp_endpoint_G
                        _rate_of = lambda _sh: (float((_cvol * np.asarray(_sh, float) * _rkr).sum())
                                                / max(float((_cvol * np.asarray(_sh, float)).sum()), 1e-9))
                        log(f"   GA risk-min (dial 0) endpoint: aggregate VAMP rate {_rate_of(_safe_endpoint_G):.4f} "
                            f"vs revenue-max (dial 99) {_rate_of(_comp_endpoint_G):.4f}; revenue "
                            f"${_rev_of(_safe_endpoint_G):,.0f} vs ${_rev_of(_comp_endpoint_G):,.0f}.")
                        def _endpoint_agg(_shares):
                            _agg = G.drop(columns=["_cellk", "_ref_share", "_comp_share"]).copy()
                            _agg["share"] = np.asarray(_shares, dtype=float)
                            _agg["volume"] = _agg["cell_volume"] * _agg["share"]
                            return _agg
                        def _enforce_endpoint(_shares):
                            return _restrict_and_recap(_explode(_endpoint_agg(_shares)))
                        _progress(_f_enf1, "Enforcing caps (1/2)…")
                        _rev_gran = _enforce_endpoint(_comp_endpoint_G)    # dial 99↓ = max-revenue compliant (GA)
                        _progress(_f_enf2, "Enforcing caps (2/2)…")
                        _comp_gran = _enforce_endpoint(_safe_endpoint_G)   # dial 0 = risk-minimised compliant (GA)
                        _progress(_f_var, "Building variations…")
                        # DIAL 100 (per spec): RAW softmax revenue reference — eligibility only, NO
                        # VAMP / MID / max-share caps — the unconstrained revenue ceiling. May breach.
                        _raw100 = _restrict(_explode(_endpoint_agg(_ref_share_G)))
                        _keyc = ["rpgt", "currency", "bank", "gateway"]
                        variations = []
                        if (_rev_gran is not None and not getattr(_rev_gran, "empty", True)
                                and _comp_gran is not None and not getattr(_comp_gran, "empty", True)):
                            _tmpl = _rev_gran.reset_index(drop=True).copy()
                            _cm = (_comp_gran[_keyc + ["share"]].drop_duplicates(_keyc)
                                   .rename(columns={"share": "_comp_share_g"}))
                            _tmpl = _tmpl.merge(_cm, on=_keyc, how="left")
                            _rev_sh = pd.to_numeric(_tmpl["share"], errors="coerce").fillna(0.0).to_numpy()
                            _comp_sh = pd.to_numeric(_tmpl["_comp_share_g"], errors="coerce").fillna(
                                _tmpl["share"]).to_numpy()
                            _cvol_g = pd.to_numeric(_tmpl.get("cell_volume", 0.0), errors="coerce").fillna(0.0).to_numpy()
                            _tmpl = _tmpl.drop(columns=["_comp_share_g"])
                            _fshare = _make_frontier_share(_tmpl, _rev_sh, _comp_sh)   # true frontier (falls back to blend)
                            for i, w in enumerate(weights, 1):
                                if w >= 1.0 - 1e-9 and _raw100 is not None and not getattr(_raw100, "empty", True):
                                    _rg = _raw100
                                    summ = portfolio_summary(_rg)
                                    _mo = int(_mids_over_granular(_rg))
                                    log(f"   ── GA variation {i}/{len(weights)} · slider 100 "
                                        f"(Risk↔Conversion): RAW reference — uncapped revenue ceiling, may breach; "
                                        f"MIDs over cap={_mo}, succ={summ['expected_success_rate']:.4f}, "
                                        f"risk={summ['expected_risk_rate']:.4f}")
                                else:
                                    _bl = _fshare(float(w))   # true-frontier point at this dial (blend fallback inside)
                                    _rg = _tmpl.copy()
                                    _rg["share"] = _bl
                                    _rg["volume"] = _cvol_g * _bl
                                    summ = portfolio_summary(_rg)
                                    _mo = int(_mids_over_granular(_rg))
                                    log(f"   ── GA variation {i}/{len(weights)} · slider {int(round(w * 100))} "
                                        f"(Risk↔Conversion): blend {int(round(w * 100))}% revenue / "
                                        f"{int(round((1 - w) * 100))}% compliant; MIDs over cap={_mo}, "
                                        f"succ={summ['expected_success_rate']:.4f}, risk={summ['expected_risk_rate']:.4f}")
                                variations.append({
                                    "weight": float(w), "split": _rg, "settings": ref_settings,
                                    "mids_over_cap": _mo,
                                    **{k: v for k, v in summ.items() if k != "volume"},
                                    "volume": summ["volume"],
                                })
                        log(f"   GA total wall time (cross-cell per-MID tilt + blend): {_fmt_secs(_ga_wall_tot)} "
                            f"(2 GA runs [revenue-max + risk-min] × {_n_mid} vampMid tilts + 2 enforcements).")
                    elif not changed:
                        log("✅ Reference (conversion-optimal) split already meets every per-vampMid "
                            "VAMP cap — dial 99↓ identical (compliant); dial 100 = RAW reference (uncapped).")
                        ref_gran = _restrict_and_recap(_explode(ref_agg))
                        _raw100 = _restrict(_explode(ref_agg))   # dial 100: eligibility only, no caps
                        _, ref_summ = _summ_from_shares(ref_share)
                        _mo = int(_mids_over_granular(ref_gran))
                        _raw_summ = portfolio_summary(_raw100) if (_raw100 is not None and not getattr(_raw100, "empty", True)) else ref_summ
                        _raw_mo = int(_mids_over_granular(_raw100)) if (_raw100 is not None and not getattr(_raw100, "empty", True)) else _mo
                        variations = []
                        for w in weights:
                            if w >= 1.0 - 1e-9 and _raw100 is not None and not getattr(_raw100, "empty", True):
                                variations.append({
                                    "weight": float(w), "split": _raw100, "settings": ref_settings,
                                    "mids_over_cap": _raw_mo,
                                    **{k: v for k, v in _raw_summ.items() if k != "volume"},
                                    "volume": _raw_summ["volume"],
                                })
                            else:
                                variations.append({
                                    "weight": float(w), "split": ref_gran, "settings": ref_settings,
                                    "mids_over_cap": _mo,
                                    **{k: v for k, v in ref_summ.items() if k != "volume"},
                                    "volume": ref_summ["volume"],
                                })
                    else:
                        log(f"   adjusted split to meet per-vampMid VAMP caps "
                            f"(retired {len(retired)} MID(s){'; ' + str(len(still_over)) + ' still over after retiring' if still_over else ''}); "
                            "dial 100 = RAW reference (uncapped ceiling); 99↓ endpoint-blend to compliant (2 VAMP solves).")
                        # ENDPOINT-BLEND for the COMPLIANT positions (dial 99↓0): enforce ONLY the two
                        # compliant endpoints (VAMP-enforced revenue reference + best compliant), then
                        # linearly blend the enforced granular shares between them. A blend of two
                        # VAMP-compliant splits stays compliant (the per-MID rate is a Möbius function
                        # of the mix), so the middles need no re-enforcement — 2 solves instead of 5.
                        # DIAL 100 IS DIFFERENT: per spec it's the RAW engine reference — eligibility
                        # only (bans / wallet), NO VAMP / MID / max-share caps — so it can BREACH. It
                        # is the unconstrained revenue ceiling (a diagnostic, not deployable if it
                        # breaches). Its "MIDs over cap" count surfaces exactly how much it breaches.
                        _rev_gran = _restrict_and_recap(_explode(_summ_from_shares(ref_share)[0]))
                        _comp_gran = _restrict_and_recap(_explode(_summ_from_shares(comp_share)[0]))
                        _raw100 = _restrict(_explode(_summ_from_shares(ref_share)[0]))   # dial 100: eligibility only
                        _keyc = ["rpgt", "currency", "bank", "gateway"]
                        variations = []
                        if (_rev_gran is not None and not getattr(_rev_gran, "empty", True)
                                and _comp_gran is not None and not getattr(_comp_gran, "empty", True)):
                            _tmpl = _rev_gran.reset_index(drop=True).copy()
                            _cm = (_comp_gran[_keyc + ["share"]].drop_duplicates(_keyc)
                                   .rename(columns={"share": "_comp_share_g"}))
                            _tmpl = _tmpl.merge(_cm, on=_keyc, how="left")
                            _rev_sh = pd.to_numeric(_tmpl["share"], errors="coerce").fillna(0.0).to_numpy()
                            _comp_sh = pd.to_numeric(_tmpl["_comp_share_g"], errors="coerce").fillna(
                                _tmpl["share"]).to_numpy()
                            _cvol_g = pd.to_numeric(_tmpl.get("cell_volume", 0.0), errors="coerce").fillna(0.0).to_numpy()
                            _tmpl = _tmpl.drop(columns=["_comp_share_g"])
                            _fshare = _make_frontier_share(_tmpl, _rev_sh, _comp_sh)   # true frontier (falls back to blend)
                            for i, w in enumerate(weights, 1):
                                if w >= 1.0 - 1e-9 and _raw100 is not None and not getattr(_raw100, "empty", True):
                                    # DIAL 100: raw reference, uncapped — may breach (revenue ceiling).
                                    _rg = _raw100
                                    summ = portfolio_summary(_rg)
                                    _mo = int(_mids_over_granular(_rg))
                                    _tag = "RAW reference — uncapped, may breach"
                                else:
                                    _bl = _fshare(float(w))   # true-frontier point at this dial (blend fallback inside)
                                    _rg = _tmpl.copy()
                                    _rg["share"] = _bl
                                    _rg["volume"] = _cvol_g * _bl
                                    summ = portfolio_summary(_rg)
                                    _mo = int(_mids_over_granular(_rg))
                                    _tag = "endpoint-blend"
                                variations.append({
                                    "weight": float(w), "split": _rg, "settings": ref_settings,
                                    "mids_over_cap": _mo,
                                    **{k: v for k, v in summ.items() if k != "volume"},
                                    "volume": summ["volume"],
                                })
                                log(f"   variation {i}/{len(weights)}: w={w:.2f} succ={summ['expected_success_rate']:.4f} "
                                    f"risk={summ['expected_risk_rate']:.4f} MIDs-over={_mo} ({_tag})")
                                try:
                                    _sh = _rg["share"].to_numpy(float)
                                    _bs = (_rg["baseline_share"].to_numpy(float) if "baseline_share" in _rg.columns else _sh)
                                    _active = int((_sh > 1e-6).sum())
                                    _l1 = float(np.abs(_sh - _bs).sum())
                                    _diag(f"      ↳ gateways receiving volume={_active}/{len(_sh)} · Σ|Δshare vs baseline|={_l1:.1f} · "
                                          f"total volume={summ.get('volume', 0):,.0f} · aggregate VAMP rate={summ['expected_risk_rate']:.4%}")
                                except Exception as _e:  # noqa: BLE001
                                    _diag(f"      ↳ [variation diag failed: {_e}]")

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
                        # LAZY: precompute ONLY the DEFAULT dial (0, the min weight) now; the other
                        # dials are compressed ON-DEMAND in tab 3 the first time they're viewed/exported
                        # (same output, just deferred). Since the dial defaults to 0, this removes ~2/3
                        # of this stage from the run. The on-demand path (tab 3) caches its result into
                        # the same ss['_pool_comp'] so it's only ever computed once per dial.
                        _def_w_pc = min(float(_v["weight"]) for _v in variations) if variations else 0.0
                        _stage(f"⑥ Pre-compute pool-targeted compression for the DEFAULT dial "
                               f"{int(round(_def_w_pc * 100))} (others on-demand; target <= {_maxN_pc:,})")
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
                        # Build one job for the default dial only (sig + weight + its ideal split).
                        _jobs_pc = []
                        for _v in variations:
                            _w_pc = float(_v["weight"])
                            if abs(_w_pc - _def_w_pc) > 1e-9:
                                continue   # non-default dial → deferred to on-demand in tab 3
                            _sig_pc = (_w_pc, _maxN_pc, ss.get("variations_engine"), _bk_pc,
                                       str(_gl_pc), "sales", _ms_pc)
                            _jobs_pc.append((_sig_pc, _w_pc, _v["split"].copy()))
                        _nvar_pc = max(len(_jobs_pc), 1)

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
                        _pc_results = None
                        if _nvar_pc > 1:
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
                                        mid_list_path=_mid_list_pc, mode="sales")
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

                    _stage_end()
                    if _t6_0 is not None:   # persist compression secs → next run's adaptive ETA
                        try:
                            _gp = dict(ss.get("ga_perf") or {})
                            # EMA-smooth so a single slow/fast (cache-hit) run doesn't poison the
                            # next estimate — it converges to the typical compression time.
                            _new_cs = float(_pt.time() - _t6_0)
                            _old_cs = float(_gp.get("compress_secs", _new_cs) or _new_cs)
                            _gp["compress_secs"] = 0.5 * _old_cs + 0.5 * _new_cs
                            ss["ga_perf"] = _gp
                            _save_ga_perf(_gp)
                        except Exception:  # noqa: BLE001
                            pass
                    log(f"✅ Total run time {_pt.time() - _run_t0:.1f}s")
                    _progress(1.0, "Done")
                    status.update(label=f"Generated {len(variations)} variations.", state="complete", expanded=False)
                    st.success("Variations ready — open tab 4.")
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
                    status.update(label="FAILED", state="error", expanded=True)
                    try:
                        _pbar.empty()   # clear the % bar so it doesn't look stuck mid-run
                    except Exception:  # noqa: BLE001
                        pass
                finally:
                    root_logger.removeHandler(handler)
                    root_logger.setLevel(prev_level)
                    try:                        # persist the full log so it survives tab switches
                        ss["last_run_log"] = "\n".join(log_lines)
                    except Exception:  # noqa: BLE001
                        pass

        # Keep the last run log visible after switching tabs: Streamlit clears the live status
        # container on every rerun, so when we're NOT mid-run re-render the stored text.
        if not submit_engine and ss.get("last_run_log"):
            with _run_log_slot:
                with st.expander("Run Log (last run)", expanded=True):
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


# ============================================================================
# TAB 4 - Split, outputs & impact (UI Tab 3)
# ============================================================================
with tab_imp:

    # --- Populate impact from a 'Validate Split' run: build a single "variation" from the
    #     VALIDATED split (parsed from the exported rule files) + the attempts/success window,
    #     so this tab's pre/post tables + bridges render WITHOUT running the routing engine.
    #     Fails loudly (this path can't be exercised offline — BigQuery + real rules). ---
    _vpr = ss.pop("validate_populate_req", None)
    if _vpr:
        with st.status("Populating impact from the validated split…", expanded=True) as _vst:
            try:
                from routing_optimiser.backup_blend import parse_rules_to_split as _prs
                _split_v = _prs(_vpr.get("rules_dir", ""))
                if getattr(_split_v, "empty", True):
                    raise ValueError(f"No routing rules parsed from {_vpr.get('rules_dir')!r}.")
                st.write(f"Parsed validated split: {len(_split_v)} cell×gateway rows.")
                _scheme_v = str(_vpr.get("scheme", "visa") or "visa")
                _sqlp = {"START_DATE": _vpr.get("attempts_start"), "END_DATE": _vpr.get("attempts_end"),
                         "COMPANY": _vpr.get("company"), "CARD_SCHEME": _scheme_v,
                         "BIN_PREFIX": "4" if _scheme_v == "visa" else "5",
                         "GATEWAY_FIDS": DEFAULT_GATEWAY_FIDS}
                _sqlf = os.path.join(SQL_DIR, "attempts_success.sql")
                if not os.path.exists(_sqlf):
                    raise FileNotFoundError("attempts_success.sql not found.")
                _ap, _asrc = run_sql_file(_sqlf, CACHE_DIR, use_cache=True, fallback_csv=None,
                                          project=GCP_PROJECT, params=_sqlp)
                st.write(f"Attempts/success source: {_asrc}")
                _adf_v = load_success_data(_ap)
                _typo_v = {"MONTHLY INTIIAL": "Monthly Initial", "MONTHLY INITIAL": "Monthly Initial",
                           "ANNUAL SUB SALE": "Annual Sub Sale", "ADDON SALE": "Addon Sale",
                           "UPGRADE": "Upgrades", "UPGRADES": "Upgrades", "MONTHLY RENEWAL": "Monthly Renewal",
                           "ANNUAL SUB RENEWAL": "Annual Sub Renewal", "P6M RENEWALS": "P6M Renewals",
                           "ADDON RENEWAL": "Addon Renewal"}
                if "rpgt" in _adf_v.columns:
                    _adf_v["rpgt"] = (_adf_v["rpgt"].astype(str).str.strip().str.upper()
                                      .map(_typo_v).fillna(_adf_v["rpgt"]))
                ss["adf"] = _adf_v
                ss.setdefault("bin_to_bank", {})   # v1: raw-bank alignment (BIN on both sides)
                ss["opt_by_rpgt"] = True
                ss.pop("cached_base_30d_metrics", None)
                _cache_v = _ensure_base_30d_metrics()
                if _cache_v is None:
                    raise ValueError("Base 30-day metrics could not be built from the attempts data.")
                _eval_v = _impact_eval_frame(_split_v, _cache_v, by_rpgt=True)
                ss["variations"] = [{"weight": 0.0, "split": _split_v, "settings": {}, "eval_df": _eval_v}]
                ss["split"] = _split_v
                ss["selected_variation_weight"] = 0.0
                ss["variations_engine"] = "validate"
                ss["_comp_eval_cache"] = {}
                ss.setdefault("sr", pd.DataFrame())
                ss.setdefault("problems", {})
                ss.setdefault("mid_constraints", [])
                _vst.update(label=f"Impact populated from the validated split "
                                  f"({len(_split_v)} split rows, {_eval_v.shape[0]} eval rows).",
                            state="complete", expanded=False)
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
        st.info("Compute variations in tab 2 first.")
    else:
        variations = ss["variations"]
        weights = [v["weight"] for v in variations]

        # 30D baseline metrics (cell/gateway success rates, avg ticket, base totals)
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
            picked_w = _sld_col.select_slider(
                "**Risk  ↔  Conversion**", options=weights,
                value=_def_w,
                format_func=lambda w: f"{int(round(w * 100))}",
                help="Dial: safer routing ↔ more revenue.")
            ss["selected_variation_weight"] = picked_w
            # Impact basis toggle, rendered directly beneath the dial (label hidden).
            _maxN = int(ss.get("max_configs", 0) or 0)
            if _maxN > 0:
                _basis = _sld_col.radio(
                    "Impact basis", ["No Compression", "Compressed Rules"],
                    index=1,   # default to Compressed Rules (what the exported configs deliver)
                    horizontal=True, key="impact_basis_sel", label_visibility="collapsed",
                    help="No Compression = the uncompressed per-BIN split. Compressed Rules = the "
                         "split trimmed so the generated pool count stays within your target — the "
                         "impact the exported configs will actually deliver. Precomputed for every "
                         "dial position when you generate variations.")
                _basis_compressed = _basis.startswith("Compressed")
                # Shrink the radio option text and force the two options SIDE BY SIDE on one
                # line beneath the dial (the dial column is narrow, so horizontal=True alone
                # wraps them): no-wrap radiogroup + a min-width that lets it use the empty
                # space beneath the first metric card.
                st.markdown("""<style>
                    .st-key-impact_basis_sel label p { font-size: 11px !important; white-space: nowrap; }
                    .st-key-impact_basis_sel div[role="radiogroup"] { flex-wrap: nowrap !important; gap: 0.45rem !important; }
                    .st-key-impact_basis_sel { min-width: 250px; }
                </style>""", unsafe_allow_html=True)
            else:
                _basis_compressed = False

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
                from routing_optimiser.connector_pool_configs import (
                    BRANDS as _POOL_BRANDS0, company_to_brand_key as _co2brand0)
                _brand_key_e = _co2brand0(_company_e0)
                _brand_name_e = _POOL_BRANDS0.get(_brand_key_e, {}).get("name", _company_e0)
            except Exception:  # noqa: BLE001
                _brand_key_e, _brand_name_e = "tav", _company_e0
            _mid_list_e = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
            _csrc = split_ideal.copy()
            if "cell_volume" not in _csrc.columns:
                _csrc["cell_volume"] = (_csrc.groupby(["rpgt", "currency", "bank"])["volume"].transform("sum")
                                        if "volume" in _csrc.columns else 1.0)
            _raw_cells = int(_csrc.groupby(["rpgt", "currency", "bank"]).ngroups)
            # Signature of everything the pool-targeted result depends on (tab 4 uses the
            # default 'sales' mode for its cards/export; tab 6 keys its own mode separately).
            _pool_sig = (float(picked_w), _maxN, ss.get("variations_engine"), _brand_key_e,
                         str(_gl_e0), "sales", round(float(_wc_e.get("max_share", 0.97)), 4))
            _pool_cache = ss.get("_pool_comp") or {}
            _comp_long = None; _comp_stats = None
            if _maxN > 0 and _pool_sig in _pool_cache:
                _comp_long = _pool_cache[_pool_sig]["long"]
                _comp_stats = _pool_cache[_pool_sig]["stats"]

            def _run_pool_compression():
                """Compute + cache the pool-targeted compression for the CURRENT settings."""
                return pool_targeted_compression(
                    ss, split_ideal, target_pools=_maxN, sig=_pool_sig, wallet_ctx=_wc_e,
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
                # gateways new to a cell via the cluster centroid get baseline_share 0.
                _cl = _comp_long.copy()
                if "baseline_share" in split_ideal.columns:
                    _bl = split_ideal[["rpgt", "currency", "bank", "gateway", "baseline_share"]].drop_duplicates(
                        ["rpgt", "currency", "bank", "gateway"])
                    _cl = _cl.merge(_bl, on=["rpgt", "currency", "bank", "gateway"], how="left")
                    _cl["baseline_share"] = _cl["baseline_share"].fillna(0.0)
                _cl["volume"] = _cl["cell_volume"] * _cl["share"]
                _impact_split = _cl
            elif _basis_compressed and _comp_long is None:
                _basis_compressed = False   # compression unavailable/failed → uncompressed view
            ss["impact_split"] = _impact_split
            split = _impact_split   # everything below (charts, tables) uses the chosen basis

            # eval_df drives EVERY revenue / success-rate / bank / gateway view. Build it from the
            # ENFORCED + backup-blended split (post cap / wallet / USA / <2-gateway back-fill + the
            # backup catch-all re-adds) — the SAME routing basis the VAMP projection uses — so those
            # charts reconcile with the risk tables instead of showing the raw optimiser split.
            # A Validate-Split run already carries the routed shares in its parsed rules, so it uses
            # the eval frame built during populate; the enforcement is not re-applied.
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
                _enf["bank"] = _enf["bank"].map(
                    lambda b: _b2b.get(b, _b2b.get(str(b).strip().lower(), b))).astype(str)
                _enf = _enf.groupby(["rpgt", "currency", "bank", "gateway"], as_index=False)["share"].mean()
                _t = _enf.groupby(["rpgt", "currency", "bank"])["share"].transform("sum")
                _enf["share"] = (_enf["share"] / _t).where(_t > 0, 0.0)
                # Backup catch-all re-adds (e.g. Braintree) at gateway grain, per (rpgt,currency,bank).
                _bc = ss.get("backup_catchall") or {}
                if _bc and os.environ.get("ROUTING_BACKUP_BLEND", "1") != "0":
                    from collections import defaultdict as _dd
                    from routing_optimiser.backup_blend import blend_cell_shares as _bcs
                    _acc, _cnt = _dd(lambda: _dd(float)), _dd(int)
                    for (_cur, _rp, _pmp, _ct), _gw in _bc.items():
                        _cnt[(_cur, _rp)] += 1
                        for _g, _v in _gw.items():
                            _acc[(_cur, _rp)][str(_g).strip().lower()] += float(_v)
                    _pooled = {k: {g: v / max(_cnt[k], 1) for g, v in gw.items()} for k, gw in _acc.items()}
                    _rows = []
                    for (_rp, _cur, _bnk), _grp in _enf.groupby(["rpgt", "currency", "bank"]):
                        _spec = {str(r["gateway"]): float(r["share"]) for _, r in _grp.iterrows()}
                        _ca = _pooled.get((str(_cur).strip().lower(), str(_rp).strip().lower()), {})
                        _eff = _bcs(_spec, _ca) if _ca else _spec
                        for _g, _s in _eff.items():
                            _rows.append({"rpgt": _rp, "currency": _cur, "bank": _bnk, "gateway": _g, "share": _s})
                    if _rows:
                        _enf = pd.DataFrame(_rows)
                if "baseline_share" in _spl.columns:
                    # OUTER-merge so gateways that were routed PRE but dropped to 0 POST still appear
                    # (post share 0), and back-fill gateways new POST get baseline 0 — both show in Δ.
                    _bl = _spl[["rpgt", "currency", "bank", "gateway", "baseline_share"]].drop_duplicates(
                        ["rpgt", "currency", "bank", "gateway"])
                    _enf = _enf.merge(_bl, on=["rpgt", "currency", "bank", "gateway"], how="outer")
                    _enf["share"] = _enf["share"].fillna(0.0)
                    _enf["baseline_share"] = _enf["baseline_share"].fillna(0.0)
                # Carry per-cell volume through so the eval frame can size pre/post volume + revenue.
                # (The ideal split has `cell_volume`; the enforced split from build_split_exports does
                # NOT — without this, _impact_eval_frame's cell_volume/volume would be missing.)
                _cvsrc = None
                if "cell_volume" in _spl.columns:
                    _cvsrc = _spl.groupby(["rpgt", "currency", "bank"], as_index=False)["cell_volume"].first()
                elif "volume" in _spl.columns:
                    _cvsrc = (_spl.groupby(["rpgt", "currency", "bank"], as_index=False)["volume"].sum()
                              .rename(columns={"volume": "cell_volume"}))
                if _cvsrc is not None:
                    _enf = _enf.merge(_cvsrc, on=["rpgt", "currency", "bank"], how="left")
                    _enf["cell_volume"] = pd.to_numeric(_enf["cell_volume"], errors="coerce").fillna(0.0)
                    _enf["volume"] = _enf["cell_volume"] * _enf["share"]
                # Belt-and-suspenders: drop switched-off gateways (target=0, trx/both) from the eval
                # split and renormalise each cell's post-share, so a turned-off gateway can NEVER show
                # routed share in the revenue view regardless of how it entered (engine candidate,
                # backup pool, or the baseline outer-merge).
                try:
                    import json as _je
                    from routing_optimiser.forecast_pipeline import _canonical_gateway as _cge
                    _ovp_e = os.path.join(PROJECT_ROOT, "config", "inputs", "gateway_volume_overrides.json")
                    _off_e = set()
                    if os.path.exists(_ovp_e):
                        for _gw, _cfg in (_je.load(open(_ovp_e)) or {}).items():
                            if isinstance(_cfg, dict) and pd.to_numeric(_cfg.get("target"), errors="coerce") == 0 \
                                    and str(_cfg.get("apply_to", "")).strip().lower() in ("trx", "both"):
                                _off_e.add(str(_cge(_gw)).strip().lower())
                    if _off_e and not _enf.empty and "gateway" in _enf.columns:
                        _gc = _enf["gateway"].map(_cge).astype(str).str.strip().str.lower()
                        _enf = _enf[~_gc.isin(_off_e)].copy()
                        if "share" in _enf.columns:
                            _tt = _enf.groupby(["rpgt", "currency", "bank"])["share"].transform("sum")
                            _enf["share"] = (_enf["share"] / _tt).where(_tt > 0, 0.0)
                            if "cell_volume" in _enf.columns:
                                _enf["volume"] = _enf["cell_volume"] * _enf["share"]
                except Exception:  # noqa: BLE001
                    pass
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

            new_succ, new_rev = eval_df["post_succ"].sum(), eval_df["post_rev"].sum()
            exp_sr = new_succ / base_att if base_att > 0 else 0
            rev_change = new_rev - base_rev_adj

            # Hand-rendered red cards: BIG colour-coded change on top, small pre→post beneath,
            # everything inside the card. (st.metric can't put the coloured change first with a
            # sub-line inside the same card.) Help text shows on card hover.
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
            _rcard(_m2c, "Expected Revenue (30D)",
                   f"{'▲' if rev_change >= 0 else '▼'} ${rev_change:+,.0f}",
                   _GRN if rev_change >= 0 else _INK,
                   f"${base_rev_adj:,.0f} → ${new_rev:,.0f}",
                   tip="Monthly revenue this routing is expected to bring in.")
            # --- Pools card: change vs ideal (fewer pools is good → green) ---
            if _comp_stats is not None:
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

        _t_risk, _t_fin, _t_riskdetail, _t_middet, _t_bank, _t_engwork = st.tabs(
            ["Risk Impact", "Financial Impact", "Risk Detail", "Mid Detail", "Bank Detail", "Engine Workings"])
        with _t_middet:
            _md_bank_slot = st.container()    # per-vampMid bank table + revenue bridge
            _md_charts_slot = st.container()  # vampMid-level SR + gateway-share charts

        with _t_fin:
            # Revenue-by-vampMid × month table renders into this slot, positioned
            # ABOVE the Bank x Currency Impact section (its code lives further down).
            _rev_slot = st.container(border=True)

            # -------------- Bank Impact Table Layout --------
            # Renders into the "Bank Detail" sub-tab (its RPGT-table slot is filled later).
            with _t_bank.container(border=True):
                st.markdown("##### Bank x Currency Impact")
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
                cell_impact = _eval_bk.groupby(["rpgt_join", "currency_join", "bank_join"]).agg(exp_succ=("exp_succ", "sum"), exp_rev=("exp_rev", "sum")).reset_index()
                cell_full = _cellagg_bk.merge(cell_impact, on=["rpgt_join", "currency_join", "bank_join"], how="left").fillna(0)
                bank_display_map = eval_df[["bank_join", "bank"]].drop_duplicates().set_index("bank_join")["bank"].to_dict()

                bank_table = cell_full.groupby(["bank_join", "currency_join"]).agg(old_att=("cell_att", "sum"), old_succ=("cell_succ", "sum"), old_rev=("cell_rev", "sum"), new_succ=("exp_succ", "sum"), new_rev=("exp_rev", "sum"), avg_ticket=("avg_ticket", "first")).reset_index()
                # Baseline revenue valued at the avg ticket, consistent with new_rev.
                bank_table["old_rev"] = bank_table["avg_ticket"] * bank_table["old_succ"]
                bank_table["Bank"] = (bank_table["bank_join"].map(bank_display_map).fillna(bank_table["bank_join"]).astype(str)
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

                # (RPGT / Sort by / Order filters render together on one row above.)
                _nb = int(len(bank_table))
                _rows = 20

                bank_view = (bank_table.sort_values(_sort_by, ascending=(_sort_dir == "Ascending"))
                             .head(int(_rows)).reset_index(drop=True))

                _cols = ["Bank", "Attempts", "Baseline Success", "Expected Success",
                         "Old Success Rate", "New Success Rate", "30D $ Impact"]

                def _fmt_cell(col, v):
                    if col == "Bank":
                        _s = str(v)
                        return (_s[:30] + "…") if len(_s) > 30 else _s
                    if col in ("Old Success Rate", "New Success Rate"):
                        return f"{float(v):.2f}%"
                    if col == "30D $ Impact":
                        return f"${float(v):+,.0f}"
                    return f"{float(v):,.0f}"

                # HTML table styled like the 'Forecast VAMP impact' table below.
                # Height is not fixed here: the equal-height flex CSS stretches this
                # card to match the revenue-bridge waterfall beside it, and scrolls
                # if the content is taller.
                # Numeric columns ~half width (smaller font + tight padding + wrapping headers);
                # the Bank column keeps its size. Table hugs content so overall width ~halves.
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

                # Shared revenue-bridge waterfall builder (used by the top-of-tab bridge AND the
                # per-vampMid bridge below), so every bridge has the SAME format. X-axis: min =
                # lower of Current/Proposed − $20k; max = Current + all increases + $20k. Y labels
                # show the FULL name (automargin + a wide left margin), never truncated.
                def _rev_bridge_waterfall(pre, post, names, deltas):
                    if not HAS_PLOTLY:
                        return None
                    import plotly.graph_objects as _gwf
                    pre, post = float(pre), float(post)
                    _dl = [float(d) for d in deltas]
                    _xs = ["Current"] + [str(n) for n in names] + ["Proposed"]
                    _ys = [pre] + _dl + [0.0]
                    _labs = [pre] + _dl + [post]
                    _meas = ["absolute"] + ["relative"] * len(_dl) + ["total"]
                    _incsum = sum(d for d in _dl if d > 0)
                    _lo = min(pre, post) - 20000.0
                    _hi = pre + _incsum + 20000.0
                    if _hi <= _lo:
                        _hi = _lo + 1.0
                    _text = [(f"${_v/1e6:,.2f}M" if (_i == 0 or _i == len(_labs) - 1) else f"${_v/1e3:,.1f}k")
                             for _i, _v in enumerate(_labs)]
                    _wf = _gwf.Figure(_gwf.Waterfall(
                        orientation="h", measure=_meas, y=_xs, x=_ys,
                        text=_text, textposition="outside", textfont=dict(size=8, color='#0B1F3A'),
                        connector=dict(line=dict(color="#B9C6DA")),
                        increasing=dict(marker=dict(color="#22C36B")),
                        decreasing=dict(marker=dict(color="#e63748")),
                        totals=dict(marker=dict(color="#0B1F3A")), showlegend=False))
                    _wf.update_layout(
                        height=560, margin=dict(l=170, r=40, t=14, b=10),   # wide left margin for full names
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#0B1F3A', family="inherit"))
                    _wf.update_xaxes(range=[_lo, _hi], showgrid=True, gridcolor='lightgrey', tickprefix="$",
                                     tickfont=dict(color='#0B1F3A', size=8), title=None)
                    _wf.update_yaxes(type="category", autorange="reversed", showgrid=False,
                                     tickfont=dict(color='#0B1F3A', size=8), title=None, automargin=True,
                                     tickmode="array", tickvals=_xs, ticktext=[str(s) for s in _xs])  # full names
                    return _wf

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
                        _bank_wf = _rev_bridge_waterfall(_pre, _post, _names, _deltas)

                # One row: Bank table + RPGT revenue table + SR-by-RPGT chart stacked in the left
                # column (all same width); revenue-bridge waterfall spanning them on the right.
                # The RPGT visuals are built in the Pre/post section and render into these slots.
                _left_col, _right_col = st.columns([1, 2])
                _left_col.markdown("".join(_h), unsafe_allow_html=True)
                _rpgt_tab_slot = _left_col.container()
                if _bank_wf is not None:
                    _right_col.plotly_chart(_bank_wf, use_container_width=True)


            # -------------- 30D revenue by vampMid × month (pre vs post) --------------
            # Same layout as the Risk-tab VAMP table, but VI Txn + $Revenue. Revenue =
            # RPGT-level avg ticket (from the actuals month before Month 0) × VI Txn for
            # that RPGT in that vampMid, summed to the vampMid. Renders into the slot
            # reserved above the Bank x Currency Impact section.
            with _rev_slot:
                _pp_r = os.path.join(out_dir, "vamp_t_period_prorata_export.csv")
                if not os.path.exists(_pp_r):
                    st.info("No pro-rata export found — revenue-by-vampMid table unavailable.")
                elif split is None or getattr(split, "empty", True):
                    st.info("No proposed split yet.")
                else:
                    _mm_r = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    _f2v_r = {}
                    if os.path.exists(_mm_r):
                        _mmd_r = pd.read_csv(_mm_r)
                        _cc_r = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmd_r.columns}
                        if _cc_r.get("gatewayfid") and _cc_r.get("vampmid"):
                            _f2v_r = dict(zip(_mmd_r[_cc_r["gatewayfid"]].astype(str).str.strip().str.lower(),
                                              _mmd_r[_cc_r["vampmid"]].astype(str).str.strip()))
                    _spr = split.copy()
                    _spr["_vm"] = _spr["gateway"].astype(str).str.strip().str.lower().map(_f2v_r)
                    _spr = _spr.dropna(subset=["_vm"])
                    if bool(ss.get("opt_by_rpgt", False)) and "rpgt" in _spr.columns:
                        _spr = _spr.drop_duplicates(["currency", "bank", "rpgt", "gateway"])
                        _pdf = _spr.groupby(["currency", "bank", "rpgt", "_vm"], as_index=False)["share"].sum()
                        _prop_r = tuple((str(c).lower(), str(b), str(rp), str(v), float(s))
                                        for c, b, rp, v, s in
                                        _pdf[["currency", "bank", "rpgt", "_vm", "share"]].itertuples(index=False))
                    else:
                        _spr = _spr.drop_duplicates(["currency", "bank", "gateway"])
                        _pdf = _spr.groupby(["currency", "bank", "_vm"], as_index=False)["share"].sum()
                        _prop_r = tuple((str(c).lower(), str(b), str(v), float(s))
                                        for c, b, v, s in _pdf[["currency", "bank", "_vm", "share"]].itertuples(index=False))
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
                    except Exception:  # noqa: BLE001
                        pass   # keep the raw _prop_r on any failure
                    from routing_optimiser.forecast_pipeline import _canonical_gateway as _cg_r
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
                    _floor_r = (0.0 if os.environ.get("ROUTING_PROJ_FLOOR", "1") == "0"
                                else float(ss.get("exploration_floor", 0.0) or 0.0))
                    _gran_r = _c_prepost_granular(
                        _pp_r, _mtime(_pp_r), _prop_r, _excl_r, _kill_r, _m0_r, _scoped_rpgts,
                        frozenset(str(x).strip().lower() for x in (_wc_r.get("incapable") or set())),
                        frozenset(str(x).strip().lower() for x in (_wc_r.get("usa_only") or set())),
                        exploration_floor=_floor_r)
                    _tick_r = rpgt_avg_ticket(cache.get("cell_agg"))
                    _rev_tbl = mid_revenue_month_table(_gran_r, _tick_r, months=range(6))
                    _rev_tbl = _rev_tbl.sort_values("$Revenue M0", ascending=False)
                    _numc = [c for c in _rev_tbl.columns if c != "vampMid"]
                    _tot_r = {"vampMid": "TOTAL", **{c: float(_rev_tbl[c].sum()) for c in _numc}}
                    _rvv = pd.concat([_rev_tbl, pd.DataFrame([_tot_r])], ignore_index=True)
                    _grp6 = [[f"VI Txn M{m}", f"$Revenue M{m}", f"VI Txn Post M{m}", f"$Revenue Post M{m}"]
                             for m in range(6)]
                    # Colour scales for post-vs-pre change (same green↑/red↓ as other tables).
                    # VI Txn and $Revenue are coloured independently (different units/magnitudes).
                    _rev_maxabs = 0.0
                    _vi_maxabs = 0.0
                    for _m in range(6):
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
                               'position:sticky; left:0; width:1%; white-space:nowrap;">vampMid</th>')
                    _rh.append(_sp)
                    for _grp in _grp6:
                        for _c in _grp:
                            _rh.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; '
                                       f'text-align:right; white-space:nowrap; width:1%;">{_c.replace("$Revenue", "$Amt")}</th>')
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
                    st.markdown("".join(_rh), unsafe_allow_html=True)

            _gwshare_slot = None   # 'Current vs Proposed Gateway Share' table renders here (top-left)
            _gwwf_slot = None      # 'Revenue bridge by gateway' waterfall renders here (top-right)

            # -------------- Bank Analysis (own filters) --------
            # Renders into the "Bank Detail" sub-tab (its gateway-share / bridge slots filled later).
            with _t_bank.container(border=True):
                st.markdown("##### Bank Analysis")
                # Match the styling of the other Plotly charts on this tab.
                st.markdown("""<style>
                    [data-testid="stPlotlyChart"] { background-color: var(--tav-card) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important; border: 1px solid var(--tav-line) !important; border-radius: 0 !important; padding: 12px !important; margin-bottom: 1rem; }
                </style>""", unsafe_allow_html=True)

                _adf_raw = ss.get("adf")
                if _adf_raw is None or getattr(_adf_raw, "empty", True):
                    st.caption("No attempts data loaded — compute a split in tab 3 first.")
                elif not HAS_PLOTLY:
                    st.caption("Plotly is required to render these charts.")
                else:
                    import plotly.express as px
                    a = _adf_raw.copy()
                    # Collapse BINs into their parent Bank (Bank x Currency grain).
                    _b2b_d = ss.get("bin_to_bank", {})
                    if _b2b_d and "bank" in a.columns:
                        a["bank"] = a["bank"].map(
                            lambda b: _b2b_d.get(b, _b2b_d.get(str(b).strip().lower(), b))).astype(str)
                    _dc2 = "date" if "date" in a.columns else ("Date" if "Date" in a.columns else None)
                    a["bank_currency"] = a["bank"].astype(str).str.strip() + " - " + a["currency"].astype(str).str.strip().str.upper()
                    bc_opts = sorted(a["bank_currency"].dropna().unique().tolist())
                    gw_opts = sorted(a["gateway"].astype(str).str.strip().dropna().unique().tolist())

                    def _idx(opts, val):
                        return opts.index(val) if val in opts else 0

                    # Inputs squeezed into the left half (a 4-unit spacer takes the rest).
                    fc1, fc2, fc3, fc4, _fcsp = st.columns([1, 1, 1, 1, 4])
                    sel_bc = fc1.selectbox("Bank / Currency", bc_opts, index=_idx(bc_opts, "JPMORGAN CHASE BANK NA - USD"), key="raw_daily_bc")
                    sel_gw = fc2.selectbox("Gateway", gw_opts, index=_idx(gw_opts, "adyen-usd-tav"), key="raw_daily_gw")
                    sel_end = fc4.date_input("End date", value=pd.to_datetime("2026-07-05").date(), key="raw_daily_end")
                    sel_start = fc3.date_input("Start date", value=sel_end - datetime.timedelta(days=30), key="raw_daily_start")

                    if not _dc2:
                        st.caption("Attempts data has no date column.")
                    else:
                        a["_d"] = pd.to_datetime(a[_dc2], errors="coerce")
                        _bv = sel_bc.split(" - ")[0].strip().lower()
                        _cv = sel_bc.split(" - ")[1].strip().lower()
                        mask = ((a["bank"].astype(str).str.strip().str.lower() == _bv)
                                & (a["currency"].astype(str).str.strip().str.lower() == _cv)
                                & (a["gateway"].astype(str).str.strip().str.lower() == sel_gw.strip().lower())
                                & (a["_d"] >= pd.to_datetime(sel_start)) & (a["_d"] <= pd.to_datetime(sel_end)))
                        d = a[mask]
                        if d.empty:
                            st.caption("No rows match the selected filters / date range.")
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
                            _afont = dict(color='#0B1F3A', size=8, family="inherit")

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
                                            font=dict(color="#0B1F3A", size=8)),
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
                                            font=dict(color="#0B1F3A", size=8)))
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
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

                st.markdown("""<style>
                    [data-testid="stPlotlyChart"] { background-color: var(--tav-card) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important; border: 1px solid var(--tav-line) !important; border-radius: 0 !important; padding: 12px !important; margin-bottom: 2rem; }
                </style>""", unsafe_allow_html=True)
            
                eval_df["bank_currency"] = eval_df["bank"] + " - " + eval_df["currency"].str.upper()
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
                        b_join = eval_df.loc[(eval_df["bank"] == b_val) & (eval_df["currency_join"] == c_val), "bank_join"].iloc[0]
                        plot_adf_sel = adf_30d[(adf_30d["bank"].astype(str).str.strip().str.lower() == b_join) & (adf_30d["currency"].astype(str).str.strip().str.lower() == c_val)].copy()
                        b_df = eval_df[(eval_df["bank_join"] == b_join) & (eval_df["currency_join"] == c_val)].copy()
                
                    # vampMid-level SR / gateway-share charts: map gatewayFid → vampMid via
                    # Master_MID_List, then collapse. Only the CHART frames (daily_gw + local copies
                    # below) are remapped — the gatewayFid-grained tables further down keep b_df /
                    # plot_adf_sel untouched.
                    _mm_sr = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                    _f2v_sr = {}
                    if os.path.exists(_mm_sr):
                        _mmd_sr = pd.read_csv(_mm_sr)
                        _cc_sr = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmd_sr.columns}
                        if _cc_sr.get("gatewayfid") and _cc_sr.get("vampmid"):
                            _f2v_sr = dict(zip(_mmd_sr[_cc_sr["gatewayfid"]].astype(str).str.strip().str.lower(),
                                               _mmd_sr[_cc_sr["vampmid"]].astype(str).str.strip()))
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
                        axis_layout_config = dict(tickfont=dict(color='#0B1F3A', size=8, family="inherit"), title_font=dict(color='#0B1F3A', size=10, family="inherit"), automargin=True)
                    
                        _leg_top = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                        font=dict(color='#0B1F3A', size=8), title_text=None)
                        # Colour = DARKNESS tied to VOLUME so higher-opacity lines are ALSO darker:
                        # sort the palette by luminance (darkest first) and give the darkest shades to
                        # the highest-volume gateways, matching the opacity ramp below.
                        import plotly.colors as _pc_sr
                        _att_by_gw = daily_gw.groupby("gateway")["att"].sum()
                        _gw_sorted = _att_by_gw.sort_values(ascending=False).index.astype(str).tolist()

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
                                    connector=dict(line=dict(color="#B9C6DA")),
                                    increasing=dict(marker=dict(color="#22C36B")),
                                    decreasing=dict(marker=dict(color="#e63748")),
                                    totals=dict(marker=dict(color="#0B1F3A")), showlegend=False))
                                figw.update_layout(
                                    height=460, margin=dict(l=35, r=40, t=14, b=10),
                                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#0B1F3A', family="inherit"))
                                figw.update_xaxes(range=[_wlo, _whi], showgrid=True, gridcolor='lightgrey',
                                                  tickprefix="$", tickfont=dict(color='#0B1F3A', size=8), title=None, automargin=True)
                                figw.update_yaxes(type="category", autorange="reversed", showgrid=False,
                                                  tickfont=dict(color='#0B1F3A', size=8), title=None, automargin=True)
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
                    # For a single Bank/Currency it's that cell's score; for the whole portfolio
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
                                    & (_a["bank"].astype(str).str.strip().str.lower() == _pb)]
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
                    _brc1, _brc2 = st.columns(2)
                    _vmbr_slot = _brc1.container()
                    _rpgtbr_slot = _brc2.container()

                    # SR + gateway-share charts (now vampMid-level, no headers) render on the
                    # Mid Detail tab, side by side.
                    if HAS_PLOTLY:
                        _mdcc = _md_charts_slot.columns(2)
                        _mdcc[0].plotly_chart(fig_sr, use_container_width=True)
                        _mdcc[1].plotly_chart(fig_sh, use_container_width=True)


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
                    # PRE uses the SAME baseline as the top-of-tab card (actual 30D successes ×
                    # avg ticket) and POST = expected revenue, so the Δ TOTAL reconciles exactly
                    # with the '30D Revenue Change' card (new_rev − base_rev_adj).
                    _post_l = (_ev.assign(_rl=_ev["rpgt"].astype(str).str.strip().str.lower())
                               .groupby("_rl").agg(post=("post_rev", "sum"), disp=("rpgt", "first")))
                    _base_l = (cell_agg.assign(_r=pd.to_numeric(cell_agg.get("avg_ticket", 0), errors="coerce").fillna(0)
                                               * pd.to_numeric(cell_agg.get("cell_succ", 0), errors="coerce").fillna(0))
                               .groupby("rpgt_join")["_r"].sum())
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
                    _sr_pre = {r["_k"]: (r["pre_succ"] / r["pre_att"] if r["pre_att"] > 0 else 0.0) for _, r in _srg.iterrows()}
                    _sr_post = {r["_k"]: (r["post_succ"] / r["post_att"] if r["post_att"] > 0 else 0.0) for _, r in _srg.iterrows()}
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
                    _mmdf_bi = pd.read_csv(_mm_bi)
                    _cc_bi = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmdf_bi.columns}
                    _gc_bi, _vc_bi = _cc_bi.get("gatewayfid"), _cc_bi.get("vampmid")
                    if _gc_bi and _vc_bi:
                        _f2v_bi = dict(zip(_mmdf_bi[_gc_bi].astype(str).str.strip().str.lower(),
                                           _mmdf_bi[_vc_bi].astype(str).str.strip()))
                _evv = _ev.copy()
                _evv["_vmid"] = (_evv["gateway"].astype(str).str.strip().str.lower().map(_f2v_bi)
                                 .fillna(_evv["gateway"].astype(str)))

                # ---- Portfolio revenue bridges: by vampMid and by RPGT (side by side) ----
                # Same waterfall format as the Bank×Currency bridge. The vampMid bridge uses the
                # per-vampMid pre/post revenue (as the per-vampMid bank table does); the RPGT bridge
                # uses the per-RPGT revenue table (_rp), so each reconciles to its own table.
                if HAS_PLOTLY:
                    _vmbr = _evv.groupby("_vmid", as_index=False).agg(
                        pre_rev=("pre_rev", "sum"), post_rev=("post_rev", "sum"))
                    _vmbr["delta"] = _vmbr["post_rev"] - _vmbr["pre_rev"]
                    _vmbr = _vmbr[(_vmbr["pre_rev"].abs() + _vmbr["post_rev"].abs()) > 0]
                    _vm_wf_all = None
                    if not _vmbr.empty:
                        # Top 7 increases + top 7 decreases; the rest roll into 'Other vampMids'.
                        _vinc = _vmbr[_vmbr["delta"] > 0].sort_values("delta", ascending=False).head(7)
                        _vdec = _vmbr[_vmbr["delta"] < 0].sort_values("delta", ascending=True).head(7)
                        _vtop = pd.concat([_vinc, _vdec]).sort_values("delta", ascending=False)
                        _vpre = float(_vmbr["pre_rev"].sum()); _vpost = float(_vmbr["post_rev"].sum())
                        _vother = (_vpost - _vpre) - float(_vtop["delta"].sum())
                        _vhas = len(_vmbr) > len(_vtop)
                        _vnames_a = _vtop["_vmid"].astype(str).tolist() + (["Other vampMids"] if _vhas else [])
                        _vdeltas_a = _vtop["delta"].tolist() + ([_vother] if _vhas else [])
                        _vm_wf_all = _rev_bridge_waterfall(_vpre, _vpost, _vnames_a, _vdeltas_a)
                    # RPGT bridge from the per-RPGT revenue table (_rp: rpgt / pre / post / delta).
                    _rpgt_wf_all = None
                    try:
                        if _rp is not None and not _rp.empty:
                            _rpb = _rp.sort_values("delta", ascending=False)
                            _rpgt_wf_all = _rev_bridge_waterfall(
                                float(_rpb["pre"].sum()), float(_rpb["post"].sum()),
                                _rpb["rpgt"].astype(str).tolist(), _rpb["delta"].tolist())
                    except NameError:
                        _rpgt_wf_all = None
                    # Render into the slots reserved above the SR / gateway-share charts
                    # (fall back to inline here if those slots weren't created this run).
                    if _vm_wf_all is not None:
                        (_vmbr_slot or st).plotly_chart(_vm_wf_all, use_container_width=True)
                    if _rpgt_wf_all is not None:
                        (_rpgtbr_slot or st).plotly_chart(_rpgt_wf_all, use_container_width=True)

                _gopts = sorted(_evv["_vmid"].dropna().astype(str).unique().tolist())
                with _md_bank_slot:
                    if _gopts:
                        _gf1, _gf2 = st.columns([1, 5])   # picker ≈ 1/6 width (~75% narrower than before)
                        _gsel = _gf1.selectbox("vampMid", _gopts, key="tab_imp_vm_sel", label_visibility="collapsed")
                        _grows = 10   # 'Rows to show' filter removed — fixed at 10 rows
                        # FULL per-vampMid per-bank agg (drives BOTH the table's top-10 and the bridge's
                        # Current/Proposed totals + 'Other banks' roll-up).
                        _gfull = _evv[_evv["_vmid"].astype(str) == _gsel].groupby("bank", as_index=False).agg(
                            pre_vol=("pre_vol", "sum"), post_vol=("post_vol", "sum"),
                            vol_delta=("vol_delta", "sum"), rev_delta=("rev_delta", "sum"),
                            pre_rev=("pre_rev", "sum"), post_rev=("post_rev", "sum"),
                            pre_share=("baseline_share", "mean"), post_share=("share", "mean"))
                        _gfull["share_delta_pp"] = (_gfull["post_share"] - _gfull["pre_share"]) * 100.0
                        _gt = _gfull.sort_values("rev_delta", ascending=False).head(_grows)
                        if _gt.empty:
                            st.info("No banks for this vampMid.")
                        else:
                            # Widened table (readable font, FULL Bank names) + green↑/red↓ on 30D $ Impact.
                            _gmax = float(np.nanmax(np.abs(_gt["rev_delta"].to_numpy(dtype=float))))
                            _gmax = _gmax if _gmax > 1e-9 else 1.0
                            _gcols = ["Bank", "30D $ Impact", "Δ Share (pp)", "Δ Volume (txns)", "Pre Volume", "Post Volume"]
                            _gh = ['<div style="box-shadow:0 4px 12px rgba(0,0,0,0.08); border-radius:0; overflow:auto; '
                                   'width:100%; background:var(--tav-card); border:1px solid var(--tav-line);">']
                            _gh.append('<table style="width:100%; border-collapse:collapse; font-size:0.62rem; line-height:1.15;"><tr>')
                            for _c in _gcols:
                                _al = "left" if _c == "Bank" else "right"
                                _gh.append(f'<th style="background:var(--tav-red); color:#FFF; font-weight:bold; '
                                           f'padding:3px 8px; text-align:{_al}; white-space:nowrap;">{_c}</th>')
                            _gh.append('</tr>')
                            for _, _r in _gt.iterrows():
                                _rd = float(_r["rev_delta"])
                                _frac = max(-1.0, min(1.0, _rd / _gmax))
                                _bg = (f"background-color: rgba(34,195,107,{0.75 * _frac:.3f});" if _frac >= 0
                                       else f"background-color: rgba(230,55,72,{0.75 * abs(_frac):.3f});")
                                _gh.append('<tr>')
                                _gh.append(f'<td style="padding:3px 8px; text-align:left; color:#000; white-space:nowrap;">{_r["bank"]}</td>')
                                _gh.append(f'<td style="padding:3px 8px; text-align:right; color:#000; {_bg} white-space:nowrap;">${_rd:,.0f}</td>')
                                _gh.append(f'<td style="padding:3px 8px; text-align:right; color:#000; white-space:nowrap;">{float(_r["share_delta_pp"]):+.2f}</td>')
                                _gh.append(f'<td style="padding:3px 8px; text-align:right; color:#000; white-space:nowrap;">{float(_r["vol_delta"]):+,.0f}</td>')
                                _gh.append(f'<td style="padding:3px 8px; text-align:right; color:#000; white-space:nowrap;">{float(_r["pre_vol"]):,.0f}</td>')
                                _gh.append(f'<td style="padding:3px 8px; text-align:right; color:#000; white-space:nowrap;">{float(_r["post_vol"]):,.0f}</td>')
                                _gh.append('</tr>')
                            _gh.append('</table></div>')
                            # Per-vampMid revenue bridge (same format as the top-of-tab bridge).
                            _vm_wf = None
                            _gpre = float(_gfull["pre_rev"].sum()); _gpost = float(_gfull["post_rev"].sum())
                            _bfi = _gfull[(_gfull["pre_rev"].abs() + _gfull["post_rev"].abs()) > 0].copy()
                            _binc = _bfi[_bfi["rev_delta"] > 0].sort_values("rev_delta", ascending=False).head(7)
                            _bdec = _bfi[_bfi["rev_delta"] < 0].sort_values("rev_delta", ascending=True).head(7)
                            _bfi = pd.concat([_binc, _bdec]).sort_values("rev_delta", ascending=False)
                            if not _bfi.empty:
                                _hoth = len(_gfull) > len(_bfi)
                                _oth = (_gpost - _gpre) - float(_bfi["rev_delta"].sum())
                                _vnames = _bfi["bank"].astype(str).tolist() + (["Other banks"] if _hoth else [])
                                _vdeltas = _bfi["rev_delta"].tolist() + ([_oth] if _hoth else [])
                                _vm_wf = _rev_bridge_waterfall(_gpre, _gpost, _vnames, _vdeltas)
                            # Table (left, width −60%: 3→1.2 with a trailing spacer) + revenue bridge.
                            _btc, _bwc, _bsp = st.columns([1.2, 2, 1.8])
                            _btc.markdown("".join(_gh), unsafe_allow_html=True)
                            if _vm_wf is not None:
                                _bwc.plotly_chart(_vm_wf, use_container_width=True)
                    else:
                        st.info("No vampMids in this variation's split.")

            # -------------- FILTERABLE ENGINE WORKINGS EXPANDER & DEBUGGER --------------
            with _t_engwork.expander("⚙️ View Algorithm Scoring Workings (Pre-Softmax) & Granular BIN Impact", expanded=True):
                st.markdown("<div style='font-size: 0.85rem; color: #0B1F3A; margin-bottom: 1rem;'>This table exposes the granular BIN/Currency level details. You can sort and filter any column by clicking the column headers.</div>", unsafe_allow_html=True)
            
                debug_mode = st.checkbox("🛠️ Toggle Debug Diagnostics (Click if table is missing or empty)")
            
                sr_df = ss["sr"].copy()
                # Collapse BINs into their parent Bank so the debug table is at
                # the Bank x Currency grain (matching the selection & engine).
                _b2b_s = ss.get("bin_to_bank", {})
                if _b2b_s and "bank" in sr_df.columns:
                    sr_df["bank"] = sr_df["bank"].map(
                        lambda b: _b2b_s.get(b, _b2b_s.get(str(b).strip().lower(), b))).astype(str)

                if selected_bank != "(All Portfolio)":
                    b_val = selected_bank.split(" - ")[0]
                    c_val = selected_bank.split(" - ")[1].lower()
                
                    if debug_mode:
                        st.write("**Diagnostics:**")
                        st.write(f"- Looking for BIN: `{b_val}` and Currency: `{c_val}`")
                        st.write(f"- Total rows available in Engine Cache: `{len(sr_df)}`")
                        st.write(f"- Sample of unique BINs in cache: `{sr_df['bank'].astype(str).unique()[:10]}`")
                    
                    sr_df = sr_df[(sr_df["bank"].astype(str).str.upper() == b_val.upper()) & (sr_df["currency"].astype(str).str.lower() == c_val)]
                
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
                
                    workings = sr_df.groupby(["bank", "currency", "gateway"]).agg(
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
                    def _parent(b):
                        return str(b2b.get(b, b2b.get(str(b), b))).strip().lower()
                    _kappa = float(ss.get("shrink_kappa", 300.0))
                    if agg_sr_lk is not None and not agg_sr_lk.empty:
                        a = agg_sr_lk.copy()
                        a["_cj"] = a["currency"].astype(str).str.strip().str.lower()
                        a["_pj"] = a["bank"].astype(str).str.strip().str.lower()
                        a["_gj"] = a["gateway"].astype(str).str.strip().str.lower()
                        _acols = ["_cj", "_pj", "_gj", "attempts", "success", "prior_rate", "success_rate"]
                        if "kappa" in a.columns:
                            _acols.append("kappa")
                        a = a[_acols].drop_duplicates(["_cj", "_pj", "_gj"])
                        workings["_cj"] = workings["currency"].astype(str).str.strip().str.lower()
                        workings["_pj"] = workings["bank"].map(_parent)
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
                        workings["Engine Score (Smoothed SR)"] = np.where(workings["All_Time_Attempts"] > 0, workings["Weighted_SR"] / workings["All_Time_Attempts"], sr_df.groupby(["bank", "currency", "gateway"])[rate_col].mean().values)
                        for _c in ["Prior SR %", "Bayesian Adj Attempts", "Bayesian Adj Success", "All-Time Attempts (raw)", "κ used"]:
                            workings[_c] = 0.0
                    workings["All-Time Raw SR"] = np.where(workings["All_Time_Attempts"] > 0, workings["All_Time_Success"] / workings["All_Time_Attempts"], 0)
                
                    workings["bank_join"] = workings["bank"].astype(str).str.strip().str.lower()
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
                        # summed over the gateway's RPGT cells) → "Pre Revenue (Adj)" now reconciles
                        # with the Financial Impact tables and with Post Revenue at the RPGT grain.
                        if "pre_rev" in b_df.columns:
                            _agg_kw["Pre_Rev"] = ("pre_rev", "sum")
                        # Baseline (pre) attempts/successes at the 30D-attempts basis, for the
                        # success-rate reconciliation chain (Baseline Success = Baseline Attempts × SR).
                        if "pre_succ" in b_df.columns:
                            _agg_kw["Baseline_Success"] = ("pre_succ", "sum")
                        if "pre_att" in b_df.columns:
                            _agg_kw["Baseline_Attempts"] = ("pre_att", "sum")
                        gw_sh_det = b_df.groupby(["bank_join", "currency_join", "gateway_join"]).agg(
                            **_agg_kw).reset_index()
                    else:
                        gw_sh_det = pd.DataFrame(columns=["bank_join", "currency_join", "gateway_join", "Gateway", "curr_vol", "Expected_Attempts", "Expected_Success", "Expected_Rev", "Pre_Rev", "Baseline_Success", "Baseline_Attempts"])
                
                    if not plot_adf_sel.empty:
                        raw_gw_det = plot_adf_sel.groupby(["bank", "currency", "gateway"]).agg(
                            Raw_Gateway=("gateway", "first"),
                            raw_att=("attempts", "sum"),
                            raw_succ=("success", "sum"),
                            raw_amount=("succ_amount", "sum")
                        ).reset_index().rename(columns={"bank": "bank_join", "currency": "currency_join", "gateway": "gateway_join"})
                        raw_gw_det["bank_join"] = raw_gw_det["bank_join"].astype(str).str.strip().str.lower()
                        raw_gw_det["currency_join"] = raw_gw_det["currency_join"].astype(str).str.strip().str.lower()
                        raw_gw_det["gateway_join"] = raw_gw_det["gateway_join"].astype(str).str.strip().str.lower()
                    else:
                        raw_gw_det = pd.DataFrame(columns=["bank_join", "currency_join", "gateway_join", "Raw_Gateway", "raw_att", "raw_succ", "raw_amount"])
                
                    raw_gw_det["Raw 30D Success Rate"] = np.where(raw_gw_det["raw_att"] > 0, raw_gw_det["raw_succ"] / raw_gw_det["raw_att"], 0)
                
                    gw_sh_det = gw_sh_det.merge(raw_gw_det, on=["bank_join", "currency_join", "gateway_join"], how="outer")
                
                    if "Gateway" in gw_sh_det.columns and "Raw_Gateway" in gw_sh_det.columns:
                        gw_sh_det["Gateway"] = gw_sh_det["Gateway"].fillna(gw_sh_det["Raw_Gateway"]).fillna(gw_sh_det["gateway_join"])
                    elif "Gateway" not in gw_sh_det.columns:
                        gw_sh_det["Gateway"] = gw_sh_det["gateway_join"]
                
                    gw_sh_det["BIN"] = gw_sh_det["bank_join"].str.upper()
                    gw_sh_det["Currency"] = gw_sh_det["currency_join"].str.upper()
                
                    for safe_col in ["Expected_Attempts", "Expected_Success", "Expected_Rev", "raw_att", "raw_succ", "raw_amount", "curr_vol", "Raw 30D Success Rate"]:
                        gw_sh_det[safe_col] = gw_sh_det.get(safe_col, 0).fillna(0)
                
                    gw_sh_det["Expected Revenue Impact"] = gw_sh_det["Expected_Rev"] - gw_sh_det["raw_amount"]
                
                    t_raw_g = gw_sh_det.groupby(["BIN", "Currency"])["raw_att"].transform("sum")
                    t_prop_g = gw_sh_det.groupby(["BIN", "Currency"])["Expected_Attempts"].transform("sum")
                
                    gw_sh_det["Current Share"] = np.where(t_raw_g > 0, (gw_sh_det["raw_att"] / t_raw_g), 0)
                    gw_sh_det["Proposed Share"] = np.where(t_prop_g > 0, (gw_sh_det["Expected_Attempts"] / t_prop_g), 0)
                    gw_sh_det["Shift (pp)"] = (gw_sh_det["Proposed Share"] - gw_sh_det["Current Share"]) * 100
                
                    workings_full = gw_sh_det.merge(workings, on=["bank_join", "currency_join", "gateway_join"], how="left")
                
                    workings_full["Gateway"] = workings_full["Gateway"].fillna(workings_full["gateway_join"])
                    workings_full["BIN"] = workings_full["BIN"].fillna(workings_full["bank_join"].str.upper())
                    workings_full["Currency"] = workings_full["Currency"].fillna(workings_full["currency_join"].str.upper())
                
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
                        workings_full = workings_full.merge(
                            _bcv[["currency_join", "bank_join", "avg_txn_value"]].rename(
                                columns={"avg_txn_value": "Avg txn value (Bank x Cur)"}),
                            on=["currency_join", "bank_join"], how="left")
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
                                and {"rpgt", "currency", "bank", "gateway", "success"}.issubset(plot_adf_sel.columns)
                                and {"rpgt_join", "currency_join", "bank_join", "rpgt_ticket"}.issubset(_ca_t.columns)):
                            _tk = _ca_t[["rpgt_join", "currency_join", "bank_join", "rpgt_ticket"]].copy()
                            if "avg_ticket" in _ca_t.columns:
                                _tk["avg_ticket"] = _ca_t["avg_ticket"].to_numpy()
                            _rs = plot_adf_sel.copy()
                            _rs = _rs.assign(
                                rpgt_join=_rs["rpgt"].astype(str).str.strip().str.lower(),
                                currency_join=_rs["currency"].astype(str).str.strip().str.lower(),
                                bank_join=_rs["bank"].astype(str).str.strip().str.lower(),
                                gateway_join=_rs["gateway"].astype(str).str.strip().str.lower(),
                                _succ=pd.to_numeric(_rs["success"], errors="coerce").fillna(0.0),
                                _att=pd.to_numeric(_rs.get("attempts", 0), errors="coerce").fillna(0.0))
                            _rs = _rs.groupby(["rpgt_join", "currency_join", "bank_join", "gateway_join"],
                                              as_index=False).agg(raw_succ=("_succ", "sum"), raw_att=("_att", "sum"))
                            _rs = _rs.merge(_tk, on=["rpgt_join", "currency_join", "bank_join"], how="left")
                            _tkr = pd.to_numeric(_rs["rpgt_ticket"], errors="coerce")
                            _tkr = _tkr.fillna(pd.to_numeric(_rs.get("avg_ticket"), errors="coerce")).fillna(0.0)
                            _rs["ticket"] = _tkr
                            _rs["pre_rev_raw"] = _rs["raw_succ"] * _rs["ticket"]
                            _raw_rpgt = _rs
                            _pre_raw_map = _rs.groupby("gateway_join", as_index=False)["pre_rev_raw"].sum()
                            if "gateway_join" in workings_full.columns:
                                workings_full = workings_full.merge(_pre_raw_map, on="gateway_join", how="left")
                    except Exception:  # noqa: BLE001
                        _raw_rpgt = None
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
                    # softmax share to the final proposed share. Floor, max-share cap and the cross-cell
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
                                zip(workings_full["currency_join"], workings_full["bank_join"])]
                            _k = workings_full["Temperature (cell)"].astype(float) * 100.0   # per-cell multiplier
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
                        # share = exploration floor + max-share cap + cross-cell VAMP/MID enforcement,
                        # combined (they're not per-gateway-decomposable). = Proposed − Softmax(pre-floor).
                        workings_full["Floor+cap+enforce shift (pp)"] = (
                            pd.to_numeric(workings_full["Proposed Share"], errors="coerce").fillna(0.0)
                            - pd.to_numeric(workings_full["Softmax Share (pre-floor)"], errors="coerce").fillna(0.0)
                        ) * 100.0

                    # Genetic engine has NO per-cell pre-softmax score. Instead show its OWN
                    # workings: the revenue-greedy REFERENCE (dial-100 waterfall: fill the best
                    # gateways up to the max share), the TILT the GA applied, and the FINAL share.
                    _is_genetic = ss.get("variations_engine") == "genetic"
                    if _is_genetic:
                        _gcap = float((ss.get("wallet_ctx") or {}).get("max_share", 0.97))
                        # Explicit per-cell loop (robust: groupby.apply returning a Series can be
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

                    st.dataframe(
                        workings_view,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            _SR: st.column_config.NumberColumn(format="%.2f%%"),
                            "Engine Score (Smoothed SR)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Raw 30D Success Rate": st.column_config.NumberColumn(format="%.2f%%"),
                            "Current Share": st.column_config.NumberColumn(format="%.2f%%"),
                            "Proposed Share": st.column_config.NumberColumn(format="%.2f%%"),
                            "Shift (pp)": st.column_config.NumberColumn(format="%+.2f pp"),
                            "Pre Revenue (Adj)": st.column_config.NumberColumn(format="$%.0f", help="ACTUAL raw 30D successes valued at the per-RPGT avg ticket (Σ over RPGTs of raw successes × per-RPGT ticket). Baseline share does NOT enter."),
                            "Post Revenue": st.column_config.NumberColumn(format="$%.0f", help="Expected successful revenue under the proposed split (expected successes × avg ticket)"),
                            "Expected Revenue Impact": st.column_config.NumberColumn(format="$%+.0f", help="Post Revenue − Pre Revenue (Adj)"),
                            "All-Time Attempts (raw)": st.column_config.NumberColumn(format="%.0f", help="Parent-bank grain, UN-decayed. Compare to the time-adj column: if decay is on, time-adj should be lower."),
                            _ATT: st.column_config.NumberColumn(format="%.1f", help="Parent-bank grain (pools all BINs); time-decay-weighted when the decay toggle is on - the grain/values the engine scores on"),
                            "Prior SR %": st.column_config.NumberColumn(format="%.2f%%", help="Bank x Currency prior the gateway is shrunk toward"),
                            "κ used": st.column_config.NumberColumn(format="%.1f", help="Smoothing volume applied: the Fixed Number, or the per-Bank×Currency Empirical-Bayes estimate"),
                            _BAA: st.column_config.NumberColumn(format="%.1f", help="All-Time (time-adj) Attempts + kappa"),
                            _BAS: st.column_config.NumberColumn(format="%.1f", help="All-Time (time-adj) Success + kappa * prior. Engine Score = Adj Success / Adj Attempts"),
                            "Raw Attempts (30D)": st.column_config.NumberColumn(format="%d"),
                            "Raw Successes (30D)": st.column_config.NumberColumn(format="%d"),
                            "Raw 30D Amount": st.column_config.NumberColumn(format="$%.0f", help="30D successful transaction value (revenue)"),
                            "Avg txn value (Bank x Cur)": st.column_config.NumberColumn(format="$%.2f", help="Avg value per successful txn pooled over ALL RPGTs at the Bank×Currency level. Shown for contrast; the revenue figures now use the per-RPGT ticket (next column)."),
                            "Eff. Ticket (per-RPGT)": st.column_config.NumberColumn(format="$%.2f", help="The per-RPGT-blended ticket actually used = Post Revenue ÷ Expected Success. Post Revenue = Expected Success × this; Pre Revenue = Baseline Success × the per-RPGT ticket."),
                            "Baseline Attempts (30D)": st.column_config.NumberColumn(format="%d", help="Current-split attempts (cell 30D attempts × baseline share), summed over RPGTs. Baseline SR = Baseline Success ÷ Baseline Attempts."),
                            "Baseline Success (30D)": st.column_config.NumberColumn(format="%d", help="Current-split successes; drives Pre Revenue (Adj) = Baseline Success × per-RPGT ticket."),
                            "SR applied (30D)": st.column_config.NumberColumn(format="%.2f%%", help="Proposed-share-weighted 30D success rate = Expected Success ÷ Expected Attempts (so Expected Success = Expected Attempts × this). The baseline side uses Baseline Success ÷ Baseline Attempts."),
                            "Expected Attempts (30D)": st.column_config.NumberColumn(format="%d", help="Proposed-split attempts (cell 30D attempts × proposed share), summed over RPGTs."),
                            "Expected Success (30D)": st.column_config.NumberColumn(format="%d", help="Expected Attempts × SR applied. Drives Post Revenue."),
                            "Exploration floor %": st.column_config.NumberColumn(format="%.2f%%", help="Minimum share every eligible gateway is floored to before capping/enforcement."),
                            "Max share cap %": st.column_config.NumberColumn(format="%.2f%%", help="Maximum share any single gateway may hold."),
                            "Floor+cap+enforce shift (pp)": st.column_config.NumberColumn(format="%+.2f pp", help="Net move from the raw Softmax (pre-floor) share to the final Proposed share = exploration floor + max-share cap + cross-cell VAMP/MID enforcement, combined (applied together, not per-gateway-decomposable)."),
                            "Temperature (cell)": st.column_config.NumberColumn(format="%.3f"),
                            "k applied (score x k)": st.column_config.NumberColumn(format="%.4f"),
                            "Euler's constant": st.column_config.NumberColumn(format="%.5f"),
                            "Weighting": st.column_config.NumberColumn(format="%.2f"),
                            "Total Weighting": st.column_config.NumberColumn(format="%.2f"),
                            "Softmax Share (pre-floor)": st.column_config.NumberColumn(format="%.2f%%"),
                            "Reference Share (waterfall)": st.column_config.NumberColumn(format="%.2f%%", help="Genetic's dial-100 revenue reference: fill the best-converting gateways up to the max share."),
                            "Tilt (pp)": st.column_config.NumberColumn(format="%+.2f pp", help="How the genetic tilt moved this gateway away from the revenue reference toward VAMP compliance."),
                            "Final Share": st.column_config.NumberColumn(format="%.2f%%", help="The genetic engine's proposed share after tilting."),
                        }
                    )

                    # ---- PER-RPGT breakdown: each gateway row above = Σ of its RPGT rows here.
                    # This is the grain the revenue is actually summed over (per-RPGT ticket ×
                    # successes), so it reconciles the collapsed Bank×Currency×gateway rows.
                    # (a checkbox, not an expander — the Engine Workings panel is itself an expander,
                    # and Streamlit forbids nesting expanders.)
                    if st.checkbox("Show per-RPGT breakdown — trace each gateway's attempts, successes, "
                                   "ticket & revenue by RPGT", value=False, key="engwork_rpgt_breakdown"):
                        if (not b_df.empty) and {"rpgt", "gateway"}.issubset(b_df.columns):
                            _b = b_df.copy()
                            if "gateway_join" not in _b.columns:
                                _b["gateway_join"] = _b["gateway"].astype(str).str.strip().str.lower()
                            if "bank_join" not in _b.columns:
                                _b["bank_join"] = _b["bank"].astype(str).str.strip().str.lower()
                            if "currency_join" not in _b.columns:
                                _b["currency_join"] = _b["currency"].astype(str).str.strip().str.lower()
                            _grp = ["bank_join", "currency_join", "rpgt", "gateway_join"]
                            # share is now a proper per-cell distribution at the SOURCE (_impact_eval_frame
                            # renormalises it), so post_att / post_succ / post_rev are correct here with no
                            # per-table rescaling — Σ Expected Attempts = Σ Raw, and shares sum to 100%.
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
                                _rpt = _b.groupby(_grp, as_index=False).agg(**_aggmap)
                                # RAW observed 30D attempts/successes + RAW-basis Pre Revenue (raw successes
                                # × per-RPGT ticket). Match on a LOWERCASED rpgt key (the breakdown keeps
                                # rpgt title-case, _raw_rpgt lowercases).
                                if _raw_rpgt is not None and not getattr(_raw_rpgt, "empty", True):
                                    _rr = _raw_rpgt[["bank_join", "currency_join", "rpgt_join", "gateway_join",
                                                     "raw_succ", "raw_att", "pre_rev_raw"]].copy()
                                    _rpt["_rpgt_l"] = _rpt["rpgt"].astype(str).str.strip().str.lower()
                                    _rpt = _rpt.merge(
                                        _rr, how="left",
                                        left_on=["bank_join", "currency_join", "_rpgt_l", "gateway_join"],
                                        right_on=["bank_join", "currency_join", "rpgt_join", "gateway_join"])
                                    _rpt = _rpt.drop(columns=["_rpgt_l", "rpgt_join"], errors="ignore")
                                _rpt["Raw Attempts (30D)"] = pd.to_numeric(_rpt.get("raw_att", 0), errors="coerce").fillna(0.0)
                                _rpt["Raw Successes (30D)"] = pd.to_numeric(_rpt.get("raw_succ", 0), errors="coerce").fillna(0.0)
                                _rpt["Pre Revenue (Adj)"] = pd.to_numeric(_rpt.get("pre_rev_raw"), errors="coerce").fillna(0.0)
                                _rpt["BIN"] = _rpt["bank_join"].astype(str).str.upper()
                                _rpt["Currency"] = _rpt["currency_join"].astype(str).str.upper()
                                if "Gateway" not in _rpt.columns:
                                    _rpt["Gateway"] = _rpt["gateway_join"]
                                _rpt = _rpt.rename(columns={"rpgt": "RPGT"})
                                for _pc in ["Current Share", "Proposed Share", "SR (30D)"]:
                                    if _pc in _rpt.columns:
                                        _rpt[_pc] = pd.to_numeric(_rpt[_pc], errors="coerce").fillna(0.0) * 100.0
                                _order = [c for c in ["BIN", "Currency", "Gateway", "RPGT",
                                                      "SR (30D)", "Ticket (per-RPGT)", "Current Share", "Proposed Share",
                                                      "Raw Attempts (30D)", "Raw Successes (30D)",
                                                      "Expected Attempts", "Expected Success",
                                                      "Pre Revenue (Adj)", "Post Revenue"] if c in _rpt.columns]
                                _rpt = _rpt[_order].sort_values(["BIN", "Currency", "Gateway", "RPGT"]).reset_index(drop=True)
                                st.caption("Full RPGT grain (SR, ticket, shares & revenue are per-RPGT). "
                                           "Pre Revenue (Adj) = Raw Successes × Ticket; Post Revenue = Expected Success × "
                                           "Ticket. Shares sum to 100% per RPGT cell and Σ Expected Attempts = Σ Raw Attempts.")
                                st.dataframe(
                                    _rpt, use_container_width=True, hide_index=True,
                                    column_config={
                                        "SR (30D)": st.column_config.NumberColumn(format="%.2f%%", help="This RPGT's gateway 30D success rate."),
                                        "Ticket (per-RPGT)": st.column_config.NumberColumn(format="$%.2f", help="This RPGT's avg ticket (Bank×Currency×RPGT)."),
                                        "Current Share": st.column_config.NumberColumn(format="%.2f%%"),
                                        "Proposed Share": st.column_config.NumberColumn(format="%.2f%%"),
                                        "Raw Attempts (30D)": st.column_config.NumberColumn(format="%d", help="Actual observed 30D attempts for this gateway × RPGT."),
                                        "Raw Successes (30D)": st.column_config.NumberColumn(format="%d", help="Actual observed 30D successes; drives Pre Revenue (Adj) = Raw Successes × Ticket."),
                                        "Expected Attempts": st.column_config.NumberColumn(format="%d", help="cell attempts × proposed share (Σ per cell = Σ Raw Attempts)."),
                                        "Expected Success": st.column_config.NumberColumn(format="%d", help="Expected Attempts × SR."),
                                        "Pre Revenue (Adj)": st.column_config.NumberColumn(format="$%.0f", help="Raw Successes × Ticket (per-RPGT)."),
                                        "Post Revenue": st.column_config.NumberColumn(format="$%.0f", help="Expected Success × Ticket (per-RPGT)."),
                                    })
                            else:
                                st.caption("No per-RPGT revenue detail available for this selection.")
                        else:
                            st.caption("No per-RPGT detail available for this selection (e.g. a validate / parsed-rules split).")
                elif not debug_mode:
                    st.info("Table is empty. Please check the 'Toggle Debug Diagnostics' box above to find out why.")

        def _prepost_render(mode):
            if os.path.exists(pp_path):
                # Reuse the granular projection already computed for the VAMP table above
                # (identical args) instead of projecting again.
                _gr = _gr_shared if _gr_shared is not None else _c_prepost_granular(
                    pp_path, _mtime(pp_path), prop_items, excluded_mids, _kill_eff, _m0s, _scoped_rpgts,
                    frozenset(str(x).strip().lower() for x in ((ss.get("wallet_ctx") or {}).get("incapable") or set())),
                    frozenset(str(x).strip().lower() for x in ((ss.get("wallet_ctx") or {}).get("usa_only") or set())),
                    exploration_floor=(0.0 if os.environ.get("ROUTING_PROJ_FLOOR", "1") == "0"
                                       else float(ss.get("exploration_floor", 0.0) or 0.0)))
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

                    # Frame filtered by the profile fields only (period/t excluded)
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
                        _hdr = _c
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

                    def _ltfmt(_c, _v):
                        if _c == "period":
                            return f"{int(_v)}"
                        if _c in ("vampMid", "BIN"):
                            return str(_v)
                        return f"{float(_v):,.0f}"

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
                        _disp = _lthdr.get(_c, _c)
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
                    _rth.append('<table style="width:auto !important; border-collapse:collapse; font-family:inherit; '
                                'font-size:0.72rem; line-height:1.1;"><tr>')
                    for _c in _rtcols:
                        _al = "left" if _c in ("vampMid", "RPGT") else "right"
                        # Wrap numeric headers at underscores so each column fits its value width.
                        _hdr = _c if _c in ("vampMid", "RPGT") else _c.replace("_", "_<wbr>")
                        _ws = "nowrap" if _c in ("vampMid", "RPGT") else "normal"
                        _rth.append(f'<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; '
                                    f'padding:3px 4px; text-align:{_al}; white-space:{_ws}; position:sticky; top:0; width:1%;">{_hdr}</th>')
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
                        # Impact tab: vampMid × RPGT table (left) + VAMP/Txn bar charts (right).
                        _rlo = st.columns([1, 1], gap="medium")
                        _rlo[0].markdown("".join(_rth), unsafe_allow_html=True)
                        _ts_slot = _rlo[1].container()

                    # Bar chart: actual months (thermometer) leading into the
                    # forecast, with forecast VAMP Pre vs Post (day-scaled).
                    if HAS_PLOTLY:
                        import plotly.express as _pxp
                        _fs2 = ss.get("forecast_settings", {}) or {}
                        _bd = pd.to_datetime(_fs2.get("month_0", date.today().replace(day=1)))
                        _mv2, _cmp2 = _fs2.get("month_var"), _fs2.get("company")
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
                        _th_gw = os.path.join(PROJECT_ROOT, "data", "cache", str(_mv2), str(_cmp2),
                                              f"thermometer_data_gwfid_{_mv2}_fcp_v3.parquet")
                        _th_std = os.path.join(PROJECT_ROOT, "data", "cache", str(_mv2), str(_cmp2),
                                               f"thermometer_data_{_mv2}_fcp_v3.parquet")
                        _th = _th_gw if os.path.exists(_th_gw) else _th_std
                        _act, _act_raw = {}, {}
                        _act_v_rp = None   # actual VAMP by (period, RPGT-title) for the by-RPGT chart
                        _act_t_rp = None   # actual txns by (period, RPGT-title) for the by-RPGT chart
                        if os.path.exists(_th):
                            _av = _c_read_parquet(_th, _mtime(_th)).copy()
                            # gatewayFid cache: map gatewayFid -> vampMid (Master_MID_List) so the
                            # vampMid filter matches the forecast grain on the actuals too.
                            if "gatewayFid" in _av.columns and "vampMid" not in _av.columns:
                                _f2v = {}
                                _mmp_c = os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv")
                                if os.path.exists(_mmp_c):
                                    try:
                                        _mmd_c = pd.read_csv(_mmp_c)
                                        _cc2 = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmd_c.columns}
                                        if _cc2.get("gatewayfid") and _cc2.get("vampmid"):
                                            _f2v = dict(zip(_mmd_c[_cc2["gatewayfid"]].astype(str).str.strip().str.lower(),
                                                            _mmd_c[_cc2["vampmid"]].astype(str).str.strip()))
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
                        _gm = os.path.join(PROJECT_ROOT, "data", "cache", str(_mv2), str(_cmp2),
                                           f"gateway_mapping_data_{_mv2}_fcp_v3.parquet")
                        _act_txn = {}
                        if os.path.exists(_gm):
                            _gmd = _c_read_parquet(_gm, _mtime(_gm)).copy()
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
                                        _mmd_g = pd.read_csv(_mmp_g)
                                        _cc3 = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mmd_g.columns}
                                        if _cc3.get("gatewayfid") and _cc3.get("vampmid"):
                                            _f2vg = dict(zip(_mmd_g[_cc3["gatewayfid"]].astype(str).str.strip().str.lower(),
                                                             _mmd_g[_cc3["vampmid"]].astype(str).str.strip()))
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
                            name="Actual VAMP", marker_color="#9AA8C0",
                            text=[_l if _a else "" for _l, _a in zip(_lbls, _is_act)],
                            textposition="inside", textfont=dict(size=9, color='#FFFFFF'), cliponaxis=False))
                        _fig.add_trace(_gob.Bar(
                            x=_mon, y=[_v if not _a else 0 for _v, _a in zip(_vmp, _is_act)],
                            name="Forecast VAMP", marker_color="#e63748",
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
                            x=_order, y=_ratio_y, name="VAMP ratio",
                            mode="lines+markers+text", yaxis="y2", connectgaps=True,
                            line=dict(color="#22C36B", width=2), marker=dict(size=5),
                            text=[(f"{_v:.1f}%" if _v is not None else "") for _v in _ratio_y],
                            textposition="top center", textfont=dict(size=8, color="#22C36B")))
                        
                        # Left VAMP axis shown as x.xxk (thousands, 2 dp) via explicit k-scaled ticks.
                        _yticks = list(np.linspace(_ylo, _yhi, 5))
                        # Header removed above the chart → give the space back to the plot.
                        _fig.update_layout(
                            height=270, margin=dict(l=35, r=45, t=22, b=4), bargap=0.08, barmode="stack",
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
                            x=_txo, y=_txratio_y, name="VAMP ratio",
                            mode="lines+markers+text", yaxis="y2", connectgaps=True,
                            line=dict(color="#22C36B", width=2), marker=dict(size=5),
                            text=[(f"{_v:.1f}%" if _v is not None else "") for _v in _txratio_y],
                            textposition="top center", textfont=dict(size=8, color="#22C36B")))
                        # Header removed above the chart → give the space back to the plot.
                        _txfig.update_layout(
                            height=270, margin=dict(l=35, r=45, t=22, b=4), bargap=0.08, barmode="stack",
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
                                yaxis=dict(showgrid=True, gridcolor='lightgrey', tickfont=_afont2, title=None,
                                           ticksuffix=("%" if pct else None)))
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
                            # Impact tab: VAMP + Transactions bar charts (with the VAMP ratio line).
                            _ts_slot.plotly_chart(_fig, use_container_width=True)
                            _ts_slot.plotly_chart(_txfig, use_container_width=True)


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
                                    x=_tso, y=_wtt, name="Avg Months to VAMP", mode="lines+markers+text", yaxis="y2",
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
                tp_path = os.path.join(out_dir, "vamp_t_period_export.csv")
                pp_path = os.path.join(out_dir, "vamp_t_period_prorata_export.csv")
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
                        _mm = pd.read_csv(mm_path)
                        _cc = {str(c).lower().replace(" ", "").replace("_", ""): c for c in _mm.columns}
                        _gcol, _vcol = _cc.get("gatewayfid"), _cc.get("vampmid")
                        if _gcol and _vcol:
                            fid2vamp = dict(zip(_mm[_gcol].astype(str).str.strip().str.lower(),
                                                _mm[_vcol].astype(str).str.strip()))
                    sp = split_now.copy()
                    sp["_vm"] = sp["gateway"].astype(str).str.strip().str.lower().map(fid2vamp)
                    sp = sp.dropna(subset=["_vm"])
                    # At Bank×Currency×RPGT grain the split has a distinct share per RPGT, so the
                    # projection must be fed the per-RPGT shares (5-tuples) to reflect that a MID
                    # was moved on one RPGT only. `prop_items_flat` (4-tuples, collapsed) is kept
                    # for the compute_vamp_post_by_mid fallback, which is RPGT-agnostic.
                    _disp_by_rpgt = bool(ss.get("opt_by_rpgt", False)) and "rpgt" in sp.columns
                    if _disp_by_rpgt:
                        sp = sp.drop_duplicates(["currency", "bank", "rpgt", "gateway"])
                        _pdf = sp.groupby(["currency", "bank", "rpgt", "_vm"], as_index=False)["share"].sum()
                        prop_items = tuple((str(c).lower(), str(b), str(rp), str(v), float(s))
                                           for c, b, rp, v, s in
                                           _pdf[["currency", "bank", "rpgt", "_vm", "share"]].itertuples(index=False))
                        _pdf_flat = sp.groupby(["currency", "bank", "_vm"], as_index=False)["share"].sum()
                        prop_items_flat = tuple((str(c).lower(), str(b), str(v), float(s))
                                                for c, b, v, s in _pdf_flat[["currency", "bank", "_vm", "share"]].itertuples(index=False))
                    else:
                        sp = sp.drop_duplicates(["currency", "bank", "gateway"])
                        prop_df = sp.groupby(["currency", "bank", "_vm"], as_index=False)["share"].sum()
                        prop_items = tuple((str(c).lower(), str(b), str(v), float(s))
                                           for c, b, v, s in prop_df[["currency", "bank", "_vm", "share"]].itertuples(index=False))
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
                    from routing_optimiser.forecast_pipeline import _canonical_gateway
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
                    # ENFORCED shares (post cap / wallet / USA-Non-USA / <2-gateway back-fill) so the
                    # projection reproduces the pipeline's back-fill gateways (WoodForest/Authorize).
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
                            from routing_optimiser.backup_blend import blend_prop_items as _bpi
                            _proj_prop = _bpi(_proj_prop, _bcatch, fid2vamp)
                        except Exception as _bbe:  # noqa: BLE001
                            pass   # any failure → keep the un-blended enforced split
                    # Exploration floor for the projection (replicates the engine's per-cell floor so
                    # 0%-rule incumbents keep >= floor). Kill-switch: ROUTING_PROJ_FLOOR=0 disables it
                    # (to compare against the old flat-rule projection). Default = the run's floor.
                    _proj_floor = (0.0 if os.environ.get("ROUTING_PROJ_FLOOR", "1") == "0"
                                   else float(ss.get("exploration_floor", 0.0) or 0.0))
                    if os.path.exists(pp_path):
                        _gr_shared = _c_prepost_granular(pp_path, _mtime(pp_path), _proj_prop, excluded_mids,
                                                         _kill_eff, _m0s, _scoped_rpgts, _wcin, _uonly,
                                                         exploration_floor=_proj_floor)
                        vp = mid_table_from_granular(_gr_shared)
                    else:
                        vp = compute_vamp_post_by_mid(tp_path, prop_items_flat, str(_m0.date()), str(_gl),
                                                      excluded_mids, _kill_eff)
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
                    html.append('<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; text-align:left; position:sticky; left:0; width:1%; white-space:nowrap;">vampMid</th>')
                
                    # Reduced spacing from 24px to 12px
                    html.append('<th style="background-color:var(--tav-card); border:none; width:8px; min-width:8px; padding:0;"></th>')
                
                    for gi, grp in enumerate(col_groups):
                        for c in grp:
                            html.append(f'<th style="background-color:var(--tav-red); color:#FFF; padding:3px 6px; text-align:right; white-space:nowrap; width:1%;">{c}</th>')
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

        with _t_riskdetail:
            if _gr_shared is not None:
                _prepost_render("detail")



# ============================================================================
# TAB 4 - Generate ConnectorPool JSON configs from the proposed split
# ============================================================================
with tab_cfg:
    # Config-generator tab body lives in tab_configs.py (per-tab split).
    import tab_configs
    tab_configs.render(ss, PROJECT_ROOT)


