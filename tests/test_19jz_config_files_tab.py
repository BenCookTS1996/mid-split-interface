"""19jz - Config Validation folded into '4 · Config Files'; the lost chart function replaced.

WHAT BEN ASKED FOR, and what each part of this test pins:

  1  everything on the 'Config Validation' sub-tab moves to tab 4 in the SAME format, tab 4 is
     renamed '4 · Config Files' and is never locked, and the sub-tab is deleted as duplication;
  2  Configs folder defaults to data/config_validation/config_lookup/<COMPANY>/<SCHEME>;
  3  the bucket.bpid input becomes a TEST-GROUP PERCENTAGE, bpid = 10000 - pct x 100;
  4  Exported rules folder defaults to data/exported_rules/<COMPANY>/<SCHEME>;
  5  clicking Generate JSON configs must not white out the other tabs;
  6  the 'Generated N ConnectorPool config(s)' line drops to input text size;
  7  the 'Generated as visa (scheme filter `vi`) from ...' line goes;
  8  the RuntimeError from the missing render_config_profile_charts goes;
  9  the Forecast outputs folder input moves under its own tickbox, defaulting to
     data/outputs/_validate/<MONTH>/<COMPANY>/<SCHEME>;
 10  the settings.yaml preview goes below the pre/post table at column width;
 11  favicon + banner logo become the TSC roundel SVG.

THREE THINGS FOUND WHILE DOING IT, each with its own check below, because each was a live
defect rather than a request:

  * `del _lost` sat under `for _lost in _LOST_IN_OVERWRITE_2026_08_26:` in app_common. That
    loop never BINDS `_lost` when the dict is empty - and emptying the dict is exactly what
    replacing the last lost symbol does - so the next import of app_common would have died on
    NameError and taken every tab with it.
  * the readiness gate dimmed tab 4 with the tooltip "Run the engine first", but the generator
    has had no variations guard for several builds. It was dimming a working tab.
  * Validate Split's rules-folder default ran the company through `.replace(" ", "")`, so
    'Total Drive' produced `data/exported_rules/TotalDrive/visa` - a folder that does not
    exist. TotalAV has no space, which is why nobody saw it.

THE WHITEOUT was not a spinner problem. Streamlit builds every tab panel in ONE script run, in
script order, and generation BLOCKS that run - so any panel built after the generator is blank
while it works. The generator ran inside tab 1, so tabs 2, 3 and 4 had not been built yet.
Moving it to the last tab is the fix; the spinner is only so tab 4 itself says it is busy.
That is why check 5 is an ORDER assertion, not a search for a spinner.
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SA = (APP / "streamlit_app.py").read_text(encoding="utf-8")
BB = (APP / "tab_1_1_build_baseline.py").read_text(encoding="utf-8")
VS = (APP / "tab_1_2_validate_split.py").read_text(encoding="utf-8")
T4 = (APP / "tab_4_generate_configs.py").read_text(encoding="utf-8")
AC = (APP / "app_common.py").read_text(encoding="utf-8")
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def code(src):
    """Source minus comment-only lines, so a comment quoting a deleted phrase cannot make a
    'the phrase is gone' check pass or fail on the strength of its own explanation."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


SAC, BBC, VSC, T4C, ACC = code(SA), code(BB), code(VS), code(T4), code(AC)


# === 1. THE SUB-TAB IS GONE AND THE MODULE IS RENAMED ===================================
# AST, not a substring search. `code()` strips comments but NOT docstrings, and this build
# added several docstrings that EXPLAIN the deleted sub-tab by name - so a plain
# `"Config Validation" not in BB` would fail on the strength of its own explanation. That is
# the same trap that made three checks in earlier builds pass or fail for the wrong reason.
def _live_strings(src):
    """Every string literal that is actually EVALUATED - docstrings excluded."""
    _t = ast.parse(src)
    _docs = set()
    for _n in ast.walk(_t):
        if isinstance(_n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _b = getattr(_n, "body", None) or []
            if _b and isinstance(_b[0], ast.Expr) and isinstance(_b[0].value, ast.Constant) \
                    and isinstance(_b[0].value.value, str):
                _docs.add(id(_b[0].value))
    return {_n.value for _n in ast.walk(_t)
            if isinstance(_n, ast.Constant) and isinstance(_n.value, str)
            and id(_n) not in _docs}


def _imported_names(src):
    _out = set()
    for _n in ast.walk(ast.parse(src)):
        if isinstance(_n, ast.Import):
            _out |= {_a.name for _a in _n.names}
        elif isinstance(_n, ast.ImportFrom) and _n.module:
            _out.add(_n.module)
    return _out


check("1  the Config Validation sub-tab is deleted",
      'st.tabs(["Build Baseline", "Validate Split"])' in BBC
      and "Config Validation" not in _live_strings(BB),
      "the name survives only in docstrings that explain WHY it went")
check("1  no module imports tab_1_3_config_validation any more",
      not any("tab_1_3_config_validation" in _imported_names(_s)
              for _s in (SA, BB, VS, T4, AC)))
# AST again: `tab_1_3_config_validation.py` appears in two docstrings that record the rename,
# and a `"tab_1_3_config_validation." in src` test matches the `.py` in those sentences.
def _attr_targets(src):
    return {_n.value.id for _n in ast.walk(ast.parse(src))
            if isinstance(_n, ast.Attribute) and isinstance(_n.value, ast.Name)}


check("1  ...and no code calls into it either",
      not any("tab_1_3_config_validation" in _attr_targets(_s)
              for _s in (SA, BB, VS, T4, AC)),
      "the name survives only in the two docstrings that record the rename")
check("1  the lookup module is renamed to what it now is",
      (APP / "config_profile_lookup.py").is_file()
      and not (APP / "tab_1_3_config_validation.py").exists(),
      "a module called tab_1_3_* when no such tab exists is a lie the reader has to disprove")
check("1  the tab is renamed", '"4 · Config Files",' in SAC
      and "4 · Generate configs" not in SAC)


# === 2. NEVER LOCKED ====================================================================
# The gate is CSS keyed on nth-of-type. Tab 4 must not appear in it, and the tooltip that told
# people to run the engine first must not be attached to it.
_gate = SAC[SAC.index("_HAS_RUN = bool("):]
_gate = _gate[:_gate.index("# ====")] if "# ====" in _gate else _gate
check("2  tab 4 is out of the readiness-gate CSS",
      "nth-of-type(4)" not in _gate and "nth-of-type(3)" in _gate,
      "tab 3 still gated, tab 4 not")
# `_variations` is READ (the risk/conversion dial is only shown when a split exists), which is
# not the same as gating on it. What must not exist is an early return: the generator builds
# from the exported-rules FOLDER, so no split, forecast or engine run is required.
check("2  and the generator never returns early for want of a split",
      "if not _variations:" not in T4C and "if True:" in T4C)


# === 3. TAB 4 IS THE CONFIG VALIDATION LAYOUT ===========================================
import tab_4_generate_configs as t4
import config_profile_lookup as cpl
import inspect

check("3  render(ss, PROJECT_ROOT) is the page",
      str(inspect.signature(t4.render)) == "(ss, PROJECT_ROOT)")
check("3  the generator body is still its own function",
      hasattr(t4, "render_generator")
      and "key_prefix" in inspect.signature(t4.render_generator).parameters)
_page = T4C[T4C.index("def render(ss, PROJECT_ROOT):"):]
check("3  two columns, generator left, lookup right - the sub-tab's own format",
      "_gen_col, _lookup_col = st.columns(2)" in _page
      and '"##### Generate Configs"' in _page
      and '"##### Look up configs by profile"' in _page)
check("3  the generator's own duplicate lookup panel is suppressed",
      "show_find=False" in _page,
      "or there would be two lookups on one screen - the duplication being removed")
check("3  the lookup column moved with it, as a function",
      hasattr(cpl, "render_lookup_panel") and hasattr(cpl, "render_profile_lookup")
      and not hasattr(cpl, "render"))


# === 4. THE FOLDER DEFAULTS =============================================================
check("4  Configs folder -> data/config_validation/config_lookup/<COMPANY>/<SCHEME>",
      'os.path.join("data", "config_validation", "config_lookup", _co, _sch)' in code(
          (APP / "config_profile_lookup.py").read_text(encoding="utf-8")))
check("4  Exported rules folder -> data/exported_rules/<COMPANY>/<SCHEME>",
      'os.path.join("data", "exported_rules", _company_c, _active_scheme)' in T4C)
check("4  Forecast outputs folder -> data/outputs/_validate/<MONTH>/<COMPANY>/<SCHEME>",
      'os.path.join("data", "outputs", "_validate", _month,' in VSC)
check("4  and the company segment KEEPS its space, because the folders on disk do",
      '.replace(" ", "")' not in VSC,
      "'Total Drive' -> data/exported_rules/Total Drive/visa, not .../TotalDrive/visa")
# the paths are only right if they exist; TotalAV/visa is the one this repo carries
for _p in ("data/config_validation/config_lookup/TotalAV/visa",
           "data/exported_rules/TotalAV/visa",
           "data/outputs/_validate/AUG/TotalAV/visa"):
    check("4  ...and %s is a real folder" % _p, (ROOT / _p).is_dir())


# === 5. bucket.bpid IS A PERCENTAGE =====================================================
check("5  the input is a test-group percentage", '"Test group %"' in T4C
      and '"bucket.bpid <"' not in T4C)
check("5  the ceiling is DERIVED, so a percentage can never reach generate_configs",
      "_ctrl_bpid = 10000 - (int(_ctrl_pct) * 100)" in T4C)
check("5  whole percentages only, capped at 99",
      "min_value=0, max_value=99, value=1, step=1" in T4C,
      "100% would derive `bucket.bpid Lt 0`, which matches nothing, silently")
for _pct, _want in ((1, 9900), (5, 9500), (0, 10000), (99, 100)):
    check("5  %d%% -> bucket.bpid Lt %s" % (_pct, f"{_want:,}"),
          10000 - (_pct * 100) == _want)


# === 6. THE TWO TEXT CHANGES ============================================================
check("6  the 'Generated as <scheme> (scheme filter ...)' caption is deleted",
      "scheme filter" not in T4C)
check("6  ...but the scheme is still RECORDED, where the download name reads it",
      '"scheme_filter": _cfg_scheme' in T4C)
check("6  the 'Generated N ConnectorPool config(s)' line is at input text size",
      "_sbc.success(_note)" not in T4C and "font-size:12px" in T4C
      and "_sbc.markdown(" in T4C)


# === 7. THE WHITEOUT: SCRIPT ORDER, NOT A SPINNER =======================================
check("7  tab 4 is the LAST panel built, so tabs 1-3 exist before generation blocks",
      SAC.index("with tab_cfg:") > max(SAC.index("with tab_fc:"),
                                       SAC.index("with tab_eng:"),
                                       SAC.index("with tab_imp:")))
check("7  the generator no longer runs from inside tab 1",
      "tab_4_generate_configs" not in BBC)
check("7  and tab 4's own panel says it is working", "st.spinner(" in T4C)


# === 8. THE LOST CHART FUNCTION =========================================================
# The import itself is half the test: `del _lost` under an empty-dict loop would raise here.
import app_common

check("8  app_common still imports with the lost-symbol dict EMPTY",
      app_common._LOST_IN_OVERWRITE_2026_08_26 == {},
      "the `del _lost` NameError this would have caused is fixed")
check("8  render_config_profile_charts is a real function, not the raising stub",
      callable(app_common.render_config_profile_charts)
      and not getattr(app_common.render_config_profile_charts, "_lost_stub", False))
check("8  it is DRAW-ONLY: returns None on every path",
      not any(isinstance(_n, ast.Return) and _n.value is not None
              for _n in ast.walk(next(f for f in ast.walk(ast.parse(AC))
                                      if isinstance(f, ast.FunctionDef)
                                      and f.name == "render_config_profile_charts"))),
      "which is why writing it fresh cannot move a number")
check("8  an empty match set is a message, not a crash",
      app_common.render_config_profile_charts([]) is None)
check("8  the stub machinery SURVIVES for the next time", "_lost_symbol" in ACC
      and "_LOST_IN_OVERWRITE_2026_08_26 = {}" in ACC)
check("8  the stub message no longer names a symbol that is not in the dict",
      "active_gateway_fids a wrong guess" not in AC)


# === 9. THE FORECAST OUTPUTS FOLDER MOVED ==============================================
_cb_at = VSC.index("v_use_prev = st.checkbox(")
_form_at = VSC.index('with st.form("validate_form"')
_input_at = VSC.index('"Forecast outputs folder", key="validate_prev_dir"')
check("9  the input sits BELOW its own tickbox", _input_at > _cb_at)
check("9  ...and OUTSIDE the form, in the same column as the tickbox",
      _input_at < _form_at,
      "a form would defer the tickbox's effect to submit, so the field could never appear")
check("9  v_prev_dir is bound unconditionally before the conditional widget (19jf)",
      VSC.index('v_prev_dir = ss.get("validate_prev_dir", "")') < _input_at)
check("9  it follows the Card Scheme, like the rules folder beside it",
      'ss["validate_prev_dir"] = _v_prev_default(_company, _sc)' in VSC)


# === 10. THE settings.yaml PREVIEW MOVED ===============================================
check("10  the preview renders AFTER the pre/post table",
      BBC.index("Preview assembled settings.yaml")
      > BBC.index("baseline VI/VAMP table unavailable"))
check("10  ...at one COLUMN wide", "_yaml_col = st.columns(2)[0]" in BBC)
check("10  its old slot in the row-2 reserve is gone, not left dangling",
      "_yaml_slot" not in BBC)
check("10  the run log keeps the reserve", "_fc_log_slot = _grow_box.container()" in BBC)


# === 11. THE BRAND MARK =================================================================
check("11  the TSC roundel is the brand mark",
      "totalsecurity.com/_r/c/6/_ptd/Core/Brand/Logos/TSCLogo" in SAC
      and "logo-alt.svg" in SAC)
check("11  the old embedded PNG is KEPT as the offline fallback",
      "_BRAND_ICON_PNG = \"data:image/png;base64," in SAC)
check("11  set_page_config can never be what kills startup",
      SAC.count("st.set_page_config(") == 3 and "except Exception:" in
      SAC[SAC.index("try:\n    st.set_page_config("):
          SAC.index("try:\n    st.set_page_config(") + 900],
      "it is the FIRST Streamlit call; a rejected page_icon would have been a stack trace")


# === 12. IT ACTUALLY RUNS ==============================================================
# Every check above reads source. This one runs the app.
_cwd = os.getcwd()
try:
    os.chdir(str(ROOT))
    from streamlit.testing.v1 import AppTest
    # ABSOLUTE: AppTest resolves a relative path against the file that CALLS it, which is this
    # test in tests/, not the repo root.
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=420)
    at.run()
    check("12  the app runs with NO exceptions", len(at.exception) == 0,
          "; ".join(str(_e)[:200] for _e in at.exception))
    _labels = [t.label for t in at.tabs]
    check("12  the tab bar reads as asked",
          "4 · Config Files" in _labels and "Config Validation" not in _labels
          and "Build Baseline" in _labels and "Validate Split" in _labels,
          str(_labels))
    _vals = {}
    for _ti in at.text_input:
        _vals.setdefault(_ti.label, _ti.value)
    check("12  Exported rules folder default", _vals.get("Exported rules folder", "").replace(
        "\\", "/") == "data/exported_rules/TotalAV/visa", _vals.get("Exported rules folder"))
    check("12  Configs folder default", _vals.get("Configs folder", "").replace("\\", "/")
          == "data/config_validation/config_lookup/TotalAV/visa", _vals.get("Configs folder"))
    check("12  the lookup rendered over real configs on disk",
          any("Loaded" in _c.value and "config(s) from" in _c.value for _c in at.caption))
    _md = [_m.value for _m in at.markdown]
    check("12  both charts rendered - this is the RuntimeError Ben hit, gone",
          any("priority × connector count" in _m for _m in _md)
          and any("Pools per connector" in _m for _m in _md)
          and not any("is MISSING" in _m for _m in _md)
          and not any("chart" in _c.value.lower() and "skipped" in _c.value.lower()
                      for _c in at.caption))
    check("12  the Test group % input is there and the raw bpid one is not",
          any(_n.label == "Test group %" for _n in at.number_input)
          and not any("bpid" in (_n.label or "") for _n in at.number_input))
    check("12  Forecast outputs folder is HIDDEN until its tickbox is ticked",
          "Forecast outputs folder" not in _vals)
    _cb = [_c for _c in at.checkbox if _c.label.startswith("Load a previously-run forecast")]
    check("12  ...and one tickbox reveals it", len(_cb) == 1)
    if _cb:
        _cb[0].set_value(True).run()
        check("12  ticking it does not raise", len(at.exception) == 0,
              "; ".join(str(_e)[:200] for _e in at.exception))
        _rev = {_ti.label: _ti.value for _ti in at.text_input}
        check("12  Forecast outputs folder default (MONTH from Month 0)",
              _rev.get("Forecast outputs folder", "").replace("\\", "/").startswith(
                  "data/outputs/_validate/")
              and _rev.get("Forecast outputs folder", "").replace("\\", "/").endswith(
                  "/TotalAV/visa"),
              _rev.get("Forecast outputs folder"))
finally:
    os.chdir(_cwd)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
