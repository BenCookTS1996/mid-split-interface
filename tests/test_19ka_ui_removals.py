"""19ka - four removals and a layout move, all on Ben's instruction.

  1  'Use config/vamp_settings.yaml as-is (repo parity)' - checkbox AND the code behind it.
     Ticking it made the pipeline read config/<scheme>_settings.yaml straight off disk and
     IGNORE every widget on the Build Baseline tab. Default was OFF, so nothing about a normal
     run changes; what goes is the ability to produce a forecast whose inputs are not the ones
     on screen.
  2  'Enforce VAMP cap' on tab 2 - always on. Its default was already True, so no default and
     no result moves. What goes with it is the OFF path, `vamp_cap = None`.
  3  the TSC roundel SVG in the banner and favicon - REVERTED to the embedded PNG.
  4  the `selector: bucket.bpid Lt 9,900` echo under the Test group % input.
  5  tab 3: the four red cards move to the far left, and 'Export Templates' fits on one line.

WHY EVERY SOURCE CHECK HERE IS AST-BASED. `code()` strips comments but not docstrings, and
this build's comments explain each removal BY NAME - they quote 'Enforce VAMP cap',
'use_yaml_asis' and the logo URL. A plain `"Enforce VAMP cap" not in src` would pass or fail on
the strength of its own explanation. That already bit three checks in 19jz.

WHAT THIS TEST CANNOT PROVE, stated so nobody reads more into it than it earns: tab 3's cards
and Export button need `variations`, `adf` and `cached_base_30d_metrics` - a full engine run's
worth of state - so item 5 is verified STRUCTURALLY (column order, widths, the nowrap rule) and
not visually. Items 1-4 are driven through a live AppTest, and item 2's check is made
non-vacuous by forcing tab 2's Risk Constraints panel to render first.
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SA = (APP / "streamlit_app.py").read_text(encoding="utf-8")
BB = (APP / "tab_1_1_build_baseline.py").read_text(encoding="utf-8")
T2 = (APP / "tab_2_routing_engine.py").read_text(encoding="utf-8")
T3 = (APP / "tab_3_split_outputs_impact.py").read_text(encoding="utf-8")
T4 = (APP / "tab_4_generate_configs.py").read_text(encoding="utf-8")
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def _tree(src):
    return ast.parse(src)


def _live_strings(src):
    """Every string literal that is actually EVALUATED - docstrings excluded."""
    t = _tree(src)
    docs = set()
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = getattr(n, "body", None) or []
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                docs.add(id(b[0].value))
    return [n.value for n in ast.walk(t)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def _names(src):
    """Every identifier the code READS or WRITES (so a deleted variable is provably gone)."""
    return {n.id for n in ast.walk(_tree(src)) if isinstance(n, ast.Name)}


def _assigned(src, name):
    """How many times `name` is the target of an assignment (incl. tuple unpacking)."""
    c = 0
    for n in ast.walk(_tree(src)):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                for sub in ([t] if not isinstance(t, (ast.Tuple, ast.List)) else t.elts):
                    if isinstance(sub, ast.Name) and sub.id == name:
                        c += 1
    return c


def _has_string(src, needle):
    return any(needle in s for s in _live_strings(src))


# === 1. 'Use config/vamp_settings.yaml as-is' ===========================================
check("1  the use_yaml_asis variable is gone from the code entirely",
      "use_yaml_asis" not in _names(BB),
      "it survives only in the two comments that record the removal")
check("1  the checkbox label is no longer rendered",
      not _has_string(BB, "vamp_settings.yaml as-is")
      and not _has_string(BB, "repo parity"))
check("1  the run log no longer echoes it",
      not _has_string(BB, "use config/vamp_settings.yaml as-is="))
check("1  ...and the NOTE warning that the widgets were being ignored is gone with it",
      not _has_string(BB, "the widget settings below are "))
check("1  the off-disk read path is deleted - the config is always BUILT",
      "yaml.safe_load" not in BB and "build_pipeline_config(forecast_settings)" in BB)
check("1  yaml is still imported and used, for the settings.yaml PREVIEW",
      "yaml.safe_dump" in BB)
check("1  the surviving Data Sources checkbox is still there",
      _has_string(BB, "Reuse cached actuarial curves"))


# === 2. 'Enforce VAMP cap' ==============================================================
check("2  the vamp_on variable is gone", "vamp_on" not in _names(T2))
check("2  the checkbox label is no longer rendered",
      not _has_string(T2, "Enforce VAMP cap"))
check("2  its widget key is gone, and so is the CSS nudge that only existed for it",
      not _has_string(T2, "vamp_on_cb") and ".st-key-vamp_on_cb" not in T2)
check("2  the VAMP cap % input SURVIVES - the cap is enforced, not removed",
      _has_string(T2, "VAMP cap (%)") and _has_string(T2, "vamp_cap_inp"))
check("2  vamp_cap is assigned exactly ONCE, so it is always a float",
      _assigned(T2, "vamp_cap") == 1, "%d assignment(s)" % _assigned(T2, "vamp_cap"))
# The consequence, stated rather than hidden: the `is None` arms are now unreachable. They are
# LEFT IN PLACE - they still behave correctly (always-true / never-taken) and deleting a dozen
# branches across 16,000 lines was not what was asked for. This check exists so the next person
# knows they are dead rather than discovering it the hard way.
_dead = T2.count("vamp_cap is None")
check("2  the now-unreachable `vamp_cap is None` arms are counted, not silently left",
      _dead >= 1, "%d arm(s) are dead but harmless - reported, not deleted" % _dead)


# === 3. THE BRAND MARK IS BACK TO THE EMBEDDED PNG ======================================
check("3  _BRAND_ICON is an embedded data: URI again",
      any(s.startswith("data:image/png;base64,") for s in _live_strings(SA)))
check("3  no live string reaches totalsecurity.com",
      not _has_string(SA, "totalsecurity.com"),
      "the URL survives only in the comment recording the revert")
check("3  the dead _BRAND_ICON_PNG fallback name went with it",
      "_BRAND_ICON_PNG" not in _names(SA))
check("3  set_page_config stays guarded - that guard is about startup, not the icon",
      SA.count("st.set_page_config(") == 2 and "except Exception:" in SA)
check("3  the banner still renders the mark", _has_string(SA, 'width="40" height="40"')
      or 'src="{_BRAND_ICON}"' in SA)


# === 4. THE bucket.bpid ECHO =============================================================
# Narrow, on purpose. `bucket.bpid Lt` STILL appears in a live string - the input's help text
# says "the generated selector is `bucket.bpid Lt (10000 - pct x 100)`", which is check 4c
# below and is the whole reason the echo was redundant. What must be gone is the RENDERED echo:
# the markdown call under the widget. Check 6 proves nothing reaches the page.
check("4  the `selector: bucket.bpid Lt ...` echo is gone",
      not _has_string(T4, "selector: <code>bucket.bpid")
      and "_cgv.markdown(" not in T4)
check("4  the derivation itself is untouched",
      "_ctrl_bpid = 10000 - (int(_ctrl_pct) * 100)" in T4)
check("4  ...and the mapping is still explained where it belongs, in the help text",
      _has_string(T4, "10000 - pct x 100"))


# === 5. TAB 3: CARDS LEFT, BUTTON ON ONE LINE ===========================================
_cols = None
for n in ast.walk(_tree(T3)):
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) \
            and isinstance(n.value, ast.Call) \
            and getattr(getattr(n.value.func, "attr", None), "__str__", lambda: "")() == "columns":
        _ids = [e.id for e in n.targets[0].elts if isinstance(e, ast.Name)]
        if "_m1c" in _ids:
            _cols = (_ids, [a.value for a in n.value.args[0].elts])
check("5  the metric-card row was found in the AST", _cols is not None)
if _cols:
    _ids, _w = _cols
    check("5  the four red cards are the FIRST four columns",
          _ids[:4] == ["_m1c", "_m2c", "_cc_col", "_cf_col"], str(_ids))
    check("5  the dial follows them rather than leading", _ids.index("_sld_col") == 4)
    check("5  the Export column is last and is now the widest of the non-report columns",
          _ids[-1] == "_exp_col" and _w[-1] > max(_w[:5]),
          "widths %s" % (_w,))
    check("5  the Export column grew from its old 0.6", _w[-1] >= 1.0, "now %s" % _w[-1])
    check("5  _con_col keeps its 2.4, so the feasibility table is not re-narrowed",
          _w[_ids.index("_con_col")] == 2.4)
check("5  one line is a PROPERTY, not a bet on font metrics",
      ".st-key-export_splits_btn button[kind=\"primary\"]" in T3
      and "white-space: nowrap !important" in T3)
# Walk back from each `white-space: nowrap` to the selector that opens its block, rather than
# pattern-matching the text around it: the first version of this check matched its OWN scoped
# rule as if it were a global one, because a scoped selector also ends in `...primary"] {`.
def _nowrap_selectors(src):
    _lines = src.splitlines()
    _out = []
    for _i, _l in enumerate(_lines):
        if "white-space: nowrap" not in _l:
            continue
        for _j in range(_i, -1, -1):
            if "{" in _lines[_j]:
                _out.append(_lines[_j].strip())
                break
    return _out


_nw = _nowrap_selectors(T3)
check("5  ...and every nowrap rule is scoped to THAT button only",
      bool(_nw) and all(".st-key-export_splits_btn" in _s for _s in _nw),
      "selectors: %s" % _nw,
      )


# === 6. IT ACTUALLY RUNS ================================================================
_cwd = os.getcwd()
try:
    os.chdir(str(ROOT))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=420)
    # NON-VACUOUS: without a cached forecast tab 2 short-circuits to a placeholder and its Risk
    # Constraints panel never renders - so "the checkbox is gone" would be true for the wrong
    # reason. Seed a real cached-forecast dir from the repo so the panel is actually built.
    _seed = ROOT / "data" / "outputs" / "SEP" / "TotalAV" / "visa"
    at.session_state["pipeline_out_dir"] = str(_seed)
    at.session_state["forecast_settings"] = {"company": "TotalAV", "card_scheme": "visa",
                                             "month_var": "SEP", "month_0": "2026-09-01"}
    at.run()
    check("6  the app runs with NO exceptions", len(at.exception) == 0,
          "; ".join(str(e)[:200] for e in at.exception))
    _nums = {n.label for n in at.number_input}
    check("6  tab 2's Risk Constraints panel DID render (so check 2 means something)",
          "VAMP cap (%)" in _nums, "the seeded forecast dir is %s" % _seed.is_dir())
    _cbs = {c.label for c in at.checkbox}
    check("6  'Enforce VAMP cap' is not on the page",
          not any("Enforce VAMP cap" in c for c in _cbs))
    check("6  'Use config/vamp_settings.yaml as-is' is not on the page",
          not any("as-is" in c for c in _cbs), str(sorted(_cbs)))
    check("6  'Reuse cached actuarial curves' survives beside it",
          any("Reuse cached actuarial curves" in c for c in _cbs))
    _md = [m.value for m in at.markdown]
    check("6  no bucket.bpid echo is rendered",
          not any("bucket.bpid Lt" in m for m in _md))
    check("6  the banner img is the embedded PNG",
          any("data:image/png;base64," in m and "<img" in m for m in _md))
finally:
    os.chdir(_cwd)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
