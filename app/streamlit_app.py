"""
Transaction Routing Optimiser — Streamlit UI. THIS FILE is the thin entry point / tab
orchestrator; each tab's body lives in its OWN module now (see the FILE MAP below).

WHAT IT DOES
------------
Given a baseline traffic forecast plus recent attempts/success data, it proposes how to SPLIT
each cell's volume across payment gateways to trade conversion off against risk (VAMP), then
turns the chosen split into deployable ConnectorPool JSON configs. It's a four-station assembly
line — one Streamlit tab per station (labels exactly as shown in the UI):

  1 · Baseline & Validate      build & cache the baseline "pre" forecast (and validate a split).
  2 · Routing engine           choose the engine + risk constraints; SEARCH for the split.
  3 · Split, outputs & impact  inspect the split, its VAMP/revenue before→after, dashboards.
  4 · Generate configs         compress to a pool budget and generate/download the JSON configs.

DATA FLOW (state is carried between stations in st.session_state, aliased `ss`)
------------------------------------------------------------------------------
  attempts + baseline forecast
        └─> ss["problems"]      one CellProblem per RPGT×Currency×Bank
              └─ (tab 2 runs the engine: genetic / softmax / thompson / portfolio)
              └─> ss["variations"]   the produced split(s) — now a SINGLE dial-0 entry
                    └─> ss["split"] + ss["settings"]   the delivered "long" split table
                          └─> (tab 4 → tab_4_generate_configs.render) ConnectorPool JSON configs

KEY session_state KEYS
----------------------
  ss["forecast"], ss["sr"]      baseline forecast + per-gateway success rates
  ss["problems"]                list[CellProblem] the engines solve
  ss["variations"]              produced split variation(s); tabs 3 & 4 read this
  ss["split"], ss["settings"]   the delivered split + the settings that produced it
  ss["wallet_ctx"]              eligibility context (bans / wallet / USA) for enforcement + export

FILE MAP — each tab lives in its OWN file; this script just wires them up
------------------------------------------------------------------------
19ft: every tab file is now named `tab_<tab>[_<sub-tab>]_<what it is>`, so the filename alone
says where in the UI it renders and in what order the user meets it. The numbers are the tab
labels the user sees, not an import order.

  TAB FILES — one `render()` each
    app/tab_1_1_build_baseline.py        Tab 1 · sub-tab 1  Build Baseline
                                         ALSO hosts tab 1's sub-tab bar and delegates 1·2 / 1·3
    app/tab_1_2_validate_split.py        Tab 1 · sub-tab 2  Validate Split
    app/tab_1_3_config_validation.py     Tab 1 · sub-tab 3  Config Validation
    app/tab_2_routing_engine.py          Tab 2             Routing engine
    app/tab_3_split_outputs_impact.py    Tab 3             Split, outputs & impact
    app/tab_4_generate_configs.py        Tab 4             Generate configs
                                         (also rendered INSIDE 1·3 with key_prefix="cv_")

  SUPPORT FILES — no `render()`, deliberately unnumbered because they belong to no one tab
    app/streamlit_app.py    THIS FILE — imports, CSS, session setup, st.tabs(), the render() calls
    app/app_common.py       shared constants, the log handler, path resolvers (input_json_path,
                            run_company, run_scheme) and the helpers every tab reuses
    app/impact_calcs.py     the VAMP pre/post projection + split-template builder. This one is
                            really BACKEND that happens to live in app/ — it is @st.cache_data-
                            decorated, which is what keeps it on this side of the fence.

WHERE THE HEAVY LIFTING LIVES (this file is mostly ORCHESTRATION + UI glue)
--------------------------------------------------------------------------
  src/routing_optimiser/…    engines (softmax/thompson/portfolio/genetic), optimiser, band
                             scoring/projection, numba kernels, success rates, eligibility.
  app/impact_calcs.py        VAMP pre/post projection + production split-template builder.
  app/tab_4_generate_configs.py         the Configs-tab body (builds ss["configs"]).

CURRENT-BEHAVIOUR NOTE (post-simplification)
--------------------------------------------
The engine delivers a SINGLE dial-0 variation: the multi-dial slider / Pareto frontier and the
post-hoc VAMP/band ENFORCEMENT layer have been removed. The delivered split is the raw engine
output with ELIGIBILITY only (bans / wallet-incapable / USA-only) applied.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import warnings

import streamlit as st

# Silence the benign divide-by-zero / 0-0 RuntimeWarnings from the MANY guarded
# `np.where(denom > 0, num / denom, fallback)` share/rate calculations: np.where
# evaluates num/denom for every row (including denom==0) before selecting, so the
# fallback is always used but numpy still emits the warning. These divisions are all
# intentionally guarded, so the message is pure noise. Targeted to the exact messages
# only — real errors and all other warnings are untouched.
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Route Numba's compile cache to a STABLE sibling dir (NOT __pycache__, which the routine
# "clear __pycache__ before every run" wipes → a ~2–3 min cold recompile every single run).
# Set it HERE, before anything can import numba, so it's authoritative regardless of import
# order (numba_kernels also setdefault()s the same path as a backup). Matches
# numba_kernels._NB_CACHE_DIR = <src>/routing_optimiser/_numba_cache.
# CRITICAL: keep this OFF any cloud-synced / FUSE-backed folder (Downloads, iCloud/Dropbox/
# OneDrive-managed dirs). Numba's on-disk cache uses file locks + mmap, which HANG on FUSE —
# the symptom is the parallel GA workers launching and then wedging forever with no progress
# (and stray `.fuse_hidden*` files piling up in the cache). The project dir lives under
# Downloads (synced), so we route the cache to the LOCAL OS temp dir instead — a real local
# disk, never synced. It survives app restarts within a session (only a reboot clears it, then
# one ~90s cold recompile), and `clear __pycache__` never touches it.
import tempfile as _tempfile
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(_tempfile.gettempdir(), "routing_optimiser_numba_cache"))
try:
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
except Exception:  # noqa: BLE001 - a read-only dir must never stop the app loading
    pass

# Engine / optimiser / impact functions are imported by the tab modules that use them —
# this orchestrator intentionally pulls in almost nothing from the backend.

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
  /* CONSISTENT TYPE SCALE — one size per role so spacing/weight read uniformly app-wide:
       h4  = tab / section titles      (largest)
       h5  = card titles               (the `##### N. …` panel headers)
       h6  = chart / sub-section titles (the `###### …` labels above individual charts)
       caption = helper text under a control. */
  h4 {font-size: 1.18rem; font-weight: 800; letter-spacing: -0.01em;
      color: var(--tav-ink); margin: 0.5rem 0 0.25rem 0;}
  h5 {font-size: 1.00rem; font-weight: 700; letter-spacing: -0.005em;
      color: var(--tav-ink); margin: 0.35rem 0 0.15rem 0;}
  h6 {font-size: 0.82rem; font-weight: 700; text-transform: none;
      color: var(--tav-muted); margin: 0.55rem 0 0.1rem 0;}
  [data-testid="stCaptionContainer"] {font-size: 0.8rem; line-height: 1.35;}

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

  /* Multiselect: the selected-value tags are [data-baseweb="tag"] (not caught by the rule
     above), so pin them AND the label to the same 12px input scale — label == values. */
  .stMultiSelect [data-testid="stWidgetLabel"] p,
  .stMultiSelect [data-baseweb="tag"],
  .stMultiSelect [data-baseweb="tag"] span,
  .stMultiSelect [data-baseweb="tag"] div {
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

# Shared constants, the log handler, and small helpers now live in app_common.py so each
# tab can import them from its own file. (Tab bodies are being split out one at a time.)
from app_common import ss, PROJECT_ROOT, SQL_DIR, GCP_PROJECT, APP_BUILD
import tab_1_1_build_baseline
import tab_2_routing_engine
import tab_3_split_outputs_impact
os.chdir(PROJECT_ROOT)  # Ensures the app always operates out of the project root




























# Per-step helpers used to live here (step_attempts / step_success_rates /
# step_forecast) for tab 1's routing-cell build. Tab 3 now owns the routing
# flow and calls the backend directly, so those helpers were removed.





# --- sidebar removed (not needed) — hide it and its collapse control entirely.
st.markdown("""<style>
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    button[title="Show sidebar"] { display: none !important; }
    div[data-testid="stAppViewContainer"] > section.main { margin-left: 0 !important; }
</style>""", unsafe_allow_html=True)

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





# ─────────────────────────── Startup preflight / environment check ───────────────────────────
# First thing the app shows: a clear checklist (packages → gcloud → credentials → BigQuery access)
# so a fresh clone surfaces "here's what's missing and how to fix it" instead of a stack trace when
# the first query runs. Non-blocking — cached / previously-run outputs still work below even if
# BigQuery isn't reachable. Cached per session; re-run via the buttons. Can't test gcloud/BigQuery
# outside a real environment, so this is best-effort and fails safe.
# [FN-288]
def _find_gcloud():
    """Resolve the gcloud binary — PATH first, then common install locations. A GUI/venv-launched
    Streamlit process often has a narrower PATH than the user's shell, so shutil.which alone can
    miss a gcloud that's actually installed and working."""
    import shutil, os as _os
    _p = shutil.which("gcloud")
    if _p:
        return _p
    _home = _os.path.expanduser("~")
    for _cand in (_os.path.join(_home, "google-cloud-sdk", "bin", "gcloud"),
                  "/usr/local/bin/gcloud", "/opt/homebrew/bin/gcloud",
                  "/usr/local/google-cloud-sdk/bin/gcloud", "/snap/bin/gcloud",
                  _os.path.join(_home, ".local", "bin", "gcloud")):
        if _os.path.exists(_cand):
            return _cand
    return None


# [FN-289]
def _query_table_refs(queries_dir):
    """Best-effort scan of every .sql in `queries_dir` for fully-qualified BigQuery table refs
    (`project.dataset.table` or `dataset.table`, backticked or after FROM / JOIN). Returns
    {table_ref: [source .sql filenames]}. Skips single-identifier CTE/aliases and any ref whose
    name is templated (contains '{'), which can't be resolved statically."""
    import re as _re
    import glob as _glob
    import os as _os
    refs = {}
    if not _os.path.isdir(queries_dir):
        return refs
    # Allow '*' in the dataset/table parts so BigQuery WILDCARD tables (e.g. `ds.transactions_*`)
    # are captured WITH their star — otherwise the prefix alone looks like a missing table (404).
    _bt = _re.compile(r"`([A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-\$\*]+){1,2})`")
    _fj = _re.compile(r"\b(?:FROM|JOIN)\s+`?([A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-\$\*]+){1,2})`?",
                      _re.IGNORECASE)
    # 19fh: RECURSIVE. queries/ now holds queries/visa/ and queries/mastercard/ alongside the
    # shared files, and a non-recursive glob would silently check only the shared ones — i.e.
    # report "all tables reachable" while never looking at either pipeline's own SQL.
    for _f in sorted(set(_glob.glob(_os.path.join(queries_dir, "*.sql")))
                     | set(_glob.glob(_os.path.join(queries_dir, "*", "*.sql")))):
        try:
            with open(_f, encoding="utf-8", errors="ignore") as _fh:
                _sql = _fh.read()
        except Exception:  # noqa: BLE001
            continue
        # 19fh: keep the scheme subfolder in the reported name (visa/fcast_query.sql), so a
        # table listed against both schemes' queries is distinguishable.
        _rel = _os.path.relpath(_f, queries_dir)
        _name = _rel if _os.sep in _rel else _os.path.basename(_f)
        for _t in set(_bt.findall(_sql)) | set(_fj.findall(_sql)):
            _t = _t.strip().strip("`")
            if "{" in _t or "}" in _t or _t.count(".") < 1:
                continue
            refs.setdefault(_t, set()).add(_name)
    return {k: sorted(v) for k, v in refs.items()}


# [FN-290]
def _check_table_access(project, tables):
    """DRY-RUN a `SELECT 1 FROM <table> LIMIT 0` per table (free — no bytes billed) to test that the
    signed-in user can resolve AND read it. Returns (ok, denied, unverified):
      • denied[(t, why)]     — a real 403 permission block (the thing worth flagging red),
      • unverified[(t, why)] — 404 / partition-filter / other probe errors that are commonly just
        wildcard or partitioned tables the naive `SELECT 1 … LIMIT 0` can't validate — NOT a
        confirmed access problem, so surfaced only as a soft amber note."""
    from google.cloud import bigquery
    _client = bigquery.Client(project=project)
    _cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    ok, denied, unverified = [], [], []
    for _t in tables:
        try:
            _client.query("SELECT 1 FROM `%s` LIMIT 0" % _t, job_config=_cfg)
            ok.append(_t)
        except Exception as _e:  # noqa: BLE001
            _code = getattr(_e, "code", None)
            if _code == 403:
                denied.append((_t, "no permission (403)"))
            elif _code == 404:
                unverified.append((_t, "probe couldn't resolve it (404 — likely a wildcard/date-sharded table)"))
            else:
                unverified.append((_t, f"probe inconclusive ({_code or type(_e).__name__})"))
    return ok, denied, unverified


# [FN-291]
def _run_preflight(project):
    """Return [(label, status, detail, fix)]; status ∈ ok / warn / fail / skip."""
    out = []
    import sys as _sys                                # 0. Python version vs the pinned target
    _vi = _sys.version_info
    _pyv = f"{_vi.major}.{_vi.minor}.{_vi.micro}"
    if _vi[:2] == (3, 8):
        out.append(("Python version", "ok", f"{_pyv} — matches the pinned target (3.8)", ""))
    elif (3, 9) <= (_vi.major, _vi.minor) <= (3, 11):
        out.append(("Python version", "ok",
                    f"{_pyv} — supported by the pinned dependencies (the project targets 3.8)", ""))
    else:
        out.append(("Python version", "warn", f"{_pyv} — outside the supported range",
                    "Use Python 3.8–3.11 (3.8.10 is the tested target). The pinned requirements "
                    "(numba 0.58 / llvmlite 0.41, etc.) won't install or run on 3.12+ or below 3.8."))
    try:                                              # 1. client libraries importable
        from google.cloud import bigquery  # noqa: F401
        import google.auth  # noqa: F401
        out.append(("Python packages", "ok", "BigQuery client libraries importable", ""))
    except Exception as _e:  # noqa: BLE001
        out.append(("Python packages", "fail", f"{type(_e).__name__}: {_e}",
                    "Install dependencies:  pip install -r requirements.txt"))
        return out
    _adc = False                                      # 2. Application Default Credentials
    try:
        import google.auth
        from google.auth.transport.requests import Request as _GReq
        _creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not getattr(_creds, "valid", False):
            _creds.refresh(_GReq())
        _adc = bool(getattr(_creds, "valid", False))
        out.append(("Google credentials (ADC)", "ok" if _adc else "fail",
                    "valid — BigQuery calls are authenticated" if _adc else "present but could not be refreshed",
                    "" if _adc else "Use 'Sign in to Google Cloud' below, or run "
                    "'gcloud auth application-default login' in a terminal, then Re-check."))
    except Exception as _e:  # noqa: BLE001
        out.append(("Google credentials (ADC)", "fail", f"{type(_e).__name__}: {str(_e)[:180]}",
                    "Use 'Sign in to Google Cloud' below, or run "
                    "'gcloud auth application-default login' in a terminal, then Re-check."))
    if _adc:                                          # 3. BigQuery query permission (trivial SELECT 1)
        try:
            from google.cloud import bigquery
            list(bigquery.Client(project=project).query("SELECT 1 AS ok").result(timeout=30))
            out.append(("BigQuery access", "ok", f"test query ran on {project}", ""))
        except Exception as _e:  # noqa: BLE001
            out.append(("BigQuery access", "fail", f"{type(_e).__name__}: {str(_e)[:180]}",
                        f"Signed in, but no query access to {project}. Ask an admin to grant BigQuery "
                        "Job User + Data Viewer on that project."))
    else:
        out.append(("BigQuery access", "skip", "sign in first, then re-check", ""))
    if _adc:                                          # 3b. per-table access across every queries/*.sql
        try:
            _refs = _query_table_refs(SQL_DIR)
            if not _refs:
                out.append(("Query table access", "warn",
                            f"no table references found in {SQL_DIR} (nothing to check)", ""))
            else:
                _n_sql = len({_s for _srcs in _refs.values() for _s in _srcs})
                _ok_t, _denied, _unver = _check_table_access(project, list(_refs))
                if _denied:                            # only a real 403 is a hard failure
                    _lst = "\n".join(
                        f"- `{_t}` — {_why}  (used in: {', '.join(_refs.get(_t, []))})"
                        for _t, _why in _denied)
                    out.append(("Query table access", "fail",
                                f"{len(_denied)} of {len(_refs)} referenced table(s) DENIED "
                                f"(permission):\n\n{_lst}",
                                "Ask an admin to grant BigQuery read/query access to the listed tables "
                                "(or their datasets)."))
                elif _unver:                           # 404 / wildcard / partition — NOT a confirmed block
                    _lst = "\n".join(
                        f"- `{_t}` — {_why}  (used in: {', '.join(_refs.get(_t, []))})"
                        for _t, _why in _unver)
                    out.append(("Query table access", "warn",
                                f"{len(_ok_t)} accessible; {len(_unver)} could NOT be auto-verified "
                                f"(usually fine — wildcard / date-sharded / partitioned tables the simple "
                                f"probe can't check):\n\n{_lst}",
                                "These are NOT confirmed permission problems — ignore if your run works; "
                                "only investigate a specific table if a query actually fails on it."))
                else:
                    out.append(("Query table access", "ok",
                                f"all {len(_ok_t)} table(s) referenced across {_n_sql} .sql file(s) "
                                "are accessible", ""))
        except Exception as _e:  # noqa: BLE001
            out.append(("Query table access", "warn",
                        f"table-access scan could not complete ({type(_e).__name__}: {str(_e)[:120]})", ""))
    else:
        out.append(("Query table access", "skip", "sign in first, then re-check", ""))
    # 4. gcloud CLI — ONLY needed to (re)authenticate. Credentials work WITHOUT it (ADC reads a
    # stored file), so this is never a problem while ADC is valid; it only matters if you must sign in.
    _gc = _find_gcloud()
    if _gc:
        out.append(("gcloud CLI", "ok", f"found ({_gc})", ""))
    elif _adc:
        out.append(("gcloud CLI", "ok",
                    "not detected on PATH — not needed (credentials already work; gcloud is only used "
                    "by the in-app sign-in button)", ""))
    else:
        out.append(("gcloud CLI", "warn", "not found on PATH",
                    "Only needed to sign in. Either install the Cloud CLI "
                    "(https://cloud.google.com/sdk/docs/install), or run "
                    "'gcloud auth application-default login' yourself in a terminal, then Re-check."))
    return out


# [FN-292]
def _render_preflight(project):
    _pf = ss.get("_preflight")
    if _pf is None:
        with st.spinner("Checking environment…"):
            _pf = _run_preflight(project)
        ss["_preflight"] = _pf
    _icon = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}
    _has_issue = any(_c[1] in ("fail", "warn") for _c in _pf)
    _title = "Environment check — ⚠️ action needed" if _has_issue else "Environment check — ✅ ready"
    with st.expander(_title, expanded=_has_issue):
        # Render at 12px to match the widget-label scale (e.g. "Exported rules folder").
        _esc = lambda _s: str(_s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Render ALL rows inside ONE markdown container. Rendering each row as its own st.markdown
        # made Streamlit's block spacing collapse the 12px lines into each other (the overlap);
        # a single container + per-row margins spaces them cleanly. Ink text (#0B1F3A) throughout.
        _rows_html = []
        for _lbl, _stt, _detail, _fix in _pf:
            _r = (f"<div style='margin:0 0 6px;'>"
                  f"{_icon.get(_stt, '•')} <b>{_esc(_lbl)}</b> — {_esc(_detail)}")
            if _fix:
                _r += f"<br><span style='padding-left:1.4em;'>↳ {_esc(_fix)}</span>"
            _r += "</div>"
            _rows_html.append(_r)
        # Check text on the LEFT; the action buttons stacked vertically in a column on the RIGHT.
        _txt_col, _btn_col = st.columns([3, 1])
        _txt_col.markdown(
            "<div style='font-size:12px; line-height:1.45; color:#0B1F3A;'>"
            + "".join(_rows_html) + "</div>",
            unsafe_allow_html=True)
        _gc = _find_gcloud()
        _reqs = os.path.join(PROJECT_ROOT, "requirements.txt")
        _do_recheck = _btn_col.button("Re-check", key="_pf_recheck", use_container_width=True)
        _do_signin = (_btn_col.button("Sign in to Google Cloud", key="_pf_signin", type="primary",
                                      use_container_width=True) if _gc else False)
        _do_pip = (_btn_col.button("Install / update packages", key="_pf_pip",
                                   use_container_width=True) if os.path.exists(_reqs) else False)
        if _do_recheck:
            ss.pop("_preflight", None)
            st.rerun()
        if _do_signin:
            import subprocess as _sp2
            with st.spinner("A browser window should have opened — complete Google sign-in there. "
                            "(If not, check the terminal where you launched Streamlit.)"):
                try:
                    _r = _sp2.run([_gc, "auth", "application-default", "login"],
                                  capture_output=True, text=True, timeout=300)
                    if _r.returncode != 0:
                        st.error("gcloud sign-in did not complete:\n\n" + (_r.stderr or "")[-800:])
                except Exception as _e:  # noqa: BLE001
                    st.error(f"Could not launch gcloud: {type(_e).__name__}: {_e}")
            ss.pop("_preflight", None)
            st.rerun()
        if _do_pip:
            # In-app install for a RUNNING app (e.g. requirements.txt changed after a git pull, or an
            # optional package is missing). Installs into THIS interpreter's environment via pip.
            # Newly-installed packages only take effect after a restart (already-imported modules stay).
            import subprocess as _sp3, sys as _sys3
            with st.spinner("pip install -r requirements.txt … (first run can take a few minutes)"):
                try:
                    _r = _sp3.run([_sys3.executable, "-m", "pip", "install", "-r", _reqs],
                                  capture_output=True, text=True, timeout=1800)
                    if _r.returncode == 0:
                        ss.pop("_preflight", None)
                        st.success("Packages installed / up to date. Restart the app (Ctrl+C, then "
                                   "re-launch) so any newly-installed packages take effect.")
                    else:
                        st.error("pip failed:\n\n" + ((_r.stdout or "") + (_r.stderr or ""))[-1500:])
                except Exception as _e:  # noqa: BLE001
                    st.error(f"Could not run pip: {type(_e).__name__}: {_e}")


_render_preflight(GCP_PROJECT)

tab_fc, tab_eng, tab_imp, tab_cfg = st.tabs([
    "1 · Baseline & Validate",
    "2 · Routing engine",
    "3 · Split, outputs & impact",
    "4 · Generate configs",
])

# --- Readiness gate: tabs 3 (impact) & 4 (configs) only have anything to show once a run
# (or a validated split) has produced `variations`. Until then, dim the two top-level tab
# labels (with a hover tooltip) so they LOOK inactive, and show a calm placeholder inside
# them instead of an empty page / a raw "compute first" info box. ---
_HAS_RUN = bool(ss.get("variations"))


if not _HAS_RUN:
    # Scope to TOP-LEVEL tabs only: `:not([tab-panel] …)` excludes every nested tab-list
    # (tab 1's and tab 3's sub-tabs live inside a tab-panel). nth-of-type counts the tab
    # buttons, so (3) and (4) are the Impact and Configs tabs.
    st.markdown("""<style>
      button[data-baseweb="tab"]:not([data-baseweb="tab-panel"] button[data-baseweb="tab"]):nth-of-type(3),
      button[data-baseweb="tab"]:not([data-baseweb="tab-panel"] button[data-baseweb="tab"]):nth-of-type(4) {
          opacity: 0.4 !important; cursor: not-allowed; position: relative;
      }
      button[data-baseweb="tab"]:not([data-baseweb="tab-panel"] button[data-baseweb="tab"]):nth-of-type(3):hover::after,
      button[data-baseweb="tab"]:not([data-baseweb="tab-panel"] button[data-baseweb="tab"]):nth-of-type(4):hover::after {
          content: "Run the engine first"; position: absolute; top: 100%; left: 0;
          white-space: nowrap; background: #0B1F3A; color: #fff; font-size: 0.72rem;
          font-weight: 600; padding: 3px 8px; margin-top: 4px; z-index: 1000;
      }
    </style>""", unsafe_allow_html=True)

# ============================================================================
# TAB 1 — Baseline & Validate  (build/cache the baseline forecast; validate a split)
# ============================================================================
with tab_fc:
    tab_1_1_build_baseline.render()

# ============================================================================
# TAB 2 — Routing engine  (choose engine + constraints -> search for & propose the split)
# ============================================================================
with tab_eng:
    tab_2_routing_engine.render()




# ============================================================================
# TAB 3 — Split, outputs & impact  (split tables, VAMP pre/post, financial impact, dashboards)
# ============================================================================
with tab_imp:
    tab_3_split_outputs_impact.render()


# ============================================================================
# TAB 4 - Generate ConnectorPool JSON configs from the proposed split
# ============================================================================
with tab_cfg:
    # Config-generator tab body lives in tab_4_generate_configs.py (per-tab split).
    import tab_4_generate_configs
    tab_4_generate_configs.render(ss, PROJECT_ROOT)




