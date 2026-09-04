"""19jy - [scaffold-timing] instruments stage 4.1, and [feas-par] prints again.

WHY THIS BUILD EXISTS. Stage 4.1 was 194.7s on the 2026-09-04 11:47 run. Only about 48s of
it had any breakdown - [cap-timing] and [band-setup], which split themselves. The five
biggest blocks in the run were each a GAP BETWEEN two log lines with nothing measured
inside: 47.0s, 30.9s, 22.1s, 18.6s and 16.2s. Twice in this session a step was priced by
READING the code and the resulting prototype measured 4.1x SLOWER, and 1.034x. So: measure
first.

THE 47.0s TURNED OUT TO BE A DEAD DIAGNOSTIC, not a missing one. `band_greedy_shares_multi`
returns from its `n_starts <= 1` fast path BEFORE the `_dt` timing at the bottom of the
function, so at the shipped `n_starts=1` the caller's `par_info` dict was never written.
That silently disabled BOTH of tab_2's reports on the stage:

  * `[feas-par]`, its wall time - which is why 47.0s of a 194.7s stage had no name; and
  * the WINNING SEED CHECKSUM, written specifically to settle whether this stage is
    deterministic after three same-input runs produced band breach 0.7159 / 0.7157 / 0.7159.

Neither has printed since n_starts was fixed at 1. Both are measurement only.

WHAT THIS TEST GUARDS
  1  the harness is bound UNCONDITIONALLY (the 19jf trap): every name a later `_sc_mark`
     call needs is bound on every path, so no mark can raise NameError because the branch
     that would have bound it did not run;
  2  `_sc_mark` and `_sc_report` swallow - a measurement fault can never reach a run;
  3  the marks COVER the five gaps the run log actually showed;
  4  `par_info` is filled on the `_n <= 1` path, and the two reports it gates are reachable;
  5  the timing is READ-ONLY: no mark or report result is assigned or tested, and the
     `_n <= 1` path still returns the same two objects from the same call.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
SS = (ROOT / "src/routing_optimiser/s4_search/seed_search.py").read_text(encoding="utf-8")
BP = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def code(src):
    """Source with comment-only lines removed, so a comment quoting a phrase can never
    satisfy (or defeat) a check about the CODE."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


T2C, SSC = code(T2), code(SS)
T2L = T2.splitlines()


# === 1. THE 19jf TRAP: the harness is bound unconditionally ==============================
# 19jf: wrapping code in a conditional makes every name it BINDS conditional. `_sc_mark` is
# called from 27 places, several inside branches that can be skipped; if the harness itself
# sat in a branch, a run that skipped it and reached a mark would die on NameError - in a
# block whose whole purpose is to be unable to affect the run.
tree = ast.parse(T2)
_render = next((n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "render"), None)
check("1  render() is where the harness lives", _render is not None)

# The property that actually matters is not "depth 0" - render() itself sits inside
# `if submit_engine:` and more - but that the harness's enclosing BRANCH CHAIN is a PREFIX
# of every mark's. If it is, then any path that reaches a mark has already run the binding.
# A `try:` body and a `with` body are not branches (they run); an `if`/`else`/`for`/`while`
# body and an `except` handler are.
BRANCHY = (ast.If, ast.For, ast.AsyncFor, ast.While)


def _paths(fn):
    """{id(stmt): tuple of (id(owner), field) branch frames} for every statement in fn's
    own scope (nested defs are followed, since the marks are all in render() itself)."""
    out = {}

    def walk(node, path):
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            for ch in items:
                if not isinstance(ch, ast.AST):
                    continue
                p = path
                if isinstance(node, BRANCHY) and field in ("body", "orelse"):
                    p = path + ((id(node), field),)
                elif isinstance(node, ast.ExceptHandler) and field == "body":
                    p = path + ((id(node), field),)
                elif isinstance(node, ast.Try) and field == "orelse":
                    p = path + ((id(node), field),)
                if isinstance(ch, ast.stmt):
                    out[id(ch)] = p
                walk(ch, p)

    walk(fn, ())
    return out


P = _paths(_render)


def _stmt_path(pred):
    for n in ast.walk(_render):
        if isinstance(n, ast.stmt) and pred(n):
            return P.get(id(n)), n
    return None, None


_h_import, _ = _stmt_path(
    lambda n: isinstance(n, ast.Import)
    and any((a.asname or a.name) == "_sc_time" for a in n.names))
_h_assign, _ = _stmt_path(
    lambda n: isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "_sctm" for t in n.targets))
_h_mark, _ = _stmt_path(lambda n: isinstance(n, ast.FunctionDef) and n.name == "_sc_mark")
_h_rep, _ = _stmt_path(lambda n: isinstance(n, ast.FunctionDef) and n.name == "_sc_report")

check("1  all four harness names are bound in the SAME branch frame",
      _h_import is not None and _h_import == _h_assign == _h_mark == _h_rep,
      "%d frame(s) deep" % (len(_h_import) if _h_import else -1))

_calls = []
for n in ast.walk(_render):
    if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id in ("_sc_mark", "_sc_report")):
        _calls.append((n.value.func.id, n.lineno, P.get(id(n))))

check("1  every _sc_mark / _sc_report call was located", len(_calls) == 28,
      "%d call(s)" % len(_calls))

_bad = [(f, ln) for f, ln, pth in _calls
        if pth is None or pth[:len(_h_mark)] != _h_mark]
check("1  the harness frame is a PREFIX of every call's frame - so no path can reach a "
      "mark without having bound it",
      not _bad, "offenders: %r" % (_bad[:4],))

# and it is bound before them in SOURCE order too, which is what actually executes here
_first = min(ln for _, ln, _ in _calls)
_h_lineno = next(n.lineno for n in ast.walk(_render)
                 if isinstance(n, ast.Import)
                 and any((a.asname or a.name) == "_sc_time" for a in n.names))
check("1  ...textually, on the line the harness opens", _h_lineno < _first,
      "harness line %d, first call line %d" % (_h_lineno, _first))

check("1  the harness is defined before the first _sc_mark call",
      T2.index("def _sc_mark(") < T2.index('_sc_mark("'))


# === 2. A MEASUREMENT FAULT CANNOT REACH THE RUN ========================================
def _fn(src, name):
    return next((n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)


for _f in ("_sc_mark", "_sc_report"):
    _node = _fn(T2, _f)
    check("2  %s exists" % _f, _node is not None)
    if _node is None:
        continue
    _stmts = [s for s in _node.body
              if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    check("2  %s's whole body is one try/except" % _f,
          len(_stmts) == 1 and isinstance(_stmts[0], ast.Try)
          and len(_stmts[0].handlers) == 1,
          "so nothing it does can propagate")

# The RUNTIME proof, not just the shape: lift the two closures out of tab_2 verbatim and
# drive them for real, including with a clock that raises.
_lo = T2L.index("                    # [FN-330]")
_hi = T2L.index("                    mid_rate = {}")
_ns = {}
exec("\n".join(  # noqa: S102 - tab_2's own harness source, run as-is
    ["def _mk(log):",
     "    import time as _sc_time",
     "    _sctm = {'t0': _sc_time.perf_counter(), 't': _sc_time.perf_counter(),",
     "             'rows': []}"]
    + ["    " + l[20:] for l in T2L[_lo:_hi]]
    + ["    return _sc_mark, _sc_report, _sctm"]), _ns)

_lines = []
_mark, _report, _state = _ns["_mk"](_lines.append)
_mark("alpha")
_mark("beta")
_report()
check("2  the real harness runs and prints a table",
      any("[scaffold-timing]" in l for l in _lines)
      and any("alpha" in l for l in _lines) and any("beta" in l for l in _lines))
check("2  the table carries a TOTAL the rows sum to",
      any("TOTAL" in l for l in _lines))


class _BadClock(object):
    def perf_counter(self):
        raise RuntimeError("clock exploded")


_lines2 = []
_mark2, _report2, _st2 = _ns["_mk"](_lines2.append)
_mark2.__defaults__ = (_st2, _BadClock())
_report2.__defaults__ = (_st2, _BadClock())
_mark2("this must not raise")
_report2()
check("2  a mark whose clock RAISES does not propagate", True)
check("2  a report whose clock RAISES says so instead of dying",
      any("unavailable" in l for l in _lines2), "and the run continues")


# === 3. THE MARKS COVER THE GAPS THE RUN LOG SHOWED =====================================
WANT = [
    # the 18.6s gap: harness start -> [inject]
    "pro-rata export -> MID VAMP rates", "ref_agg cap enforcement",
    "per-MID rule parse", "_pp_full: FULL pro-rata re-read",
    "build_capability", "inject_capable_rows",
    # the 30.9s gap: [inject] -> [rpgt-scope]
    "candidate-door coverage set", "_pp_full.copy()", "12 derived key columns on _P",
    "9-key groupby collapse", "rpgt scope mask", "[rpgt-scope] diagnostic",
    # the 16.2s gap
    "profile scope filter + _T0 slice", "back-fill grid",
    # the 1.8s gap into the cap scaffold
    "_T0 statics", "[emask] pair mask", "_Pc slice for capped MIDs",
    # the already-measured blocks, so this table reconciles against their own tables
    "cap scaffold + [scaffold-recon]", "[band-setup]",
    # the 22.1s gap
    "_profiles_layout build", "_build_elig_op",
    # the 47.0s gap - the reason this build exists
    "seed stage 1: band_greedy_shares_multi",
]
for _w in WANT:
    check("3  a mark names %r" % _w, ('_sc_mark("%s' % _w) in T2C)

_n_marks = sum(1 for l in T2L if l.strip().startswith('_sc_mark("'))
check("3  every mark is a bare statement, never part of an expression",
      _n_marks == T2C.count('_sc_mark("'), "%d mark(s)" % _n_marks)
check("3  exactly one report call, immediately before the 4.2 header",
      T2C.count("_sc_report()") == 1
      and T2.index("_sc_report()") < T2.index('_substep("④·2'))


# === 4. [feas-par] AND THE SEED CHECKSUM ARE REACHABLE AGAIN ============================
_n1 = SSC.split("if _n <= 1:")[1].split("return best_s")[0]
check("4  the _n <= 1 path fills par_info", "if isinstance(par_info, dict):" in _n1)
check("4  ...and times itself, since it returns before the _dt below",
      "_dt1 = _time.perf_counter() - _t1" in _n1)
check("4  it reports parallel=False / workers=1, which is what actually happened",
      "par_info.update(parallel=False, workers=1, starts=int(_n)," in _n1)
check("4  the docstring no longer names the deleted ROUTING_FEAS_PAR as a revert",
      "`ROUTING_FEAS_PAR=0` restores" not in SS)

import routing_optimiser.s4_search.seed_search as _ss

_calls = []


def _fake_greedy(x, *a, **k):
    _calls.append(1)
    return ("SHARES", (0.0, 0.0))


_orig = _ss.band_greedy_shares
try:
    _ss.band_greedy_shares = _fake_greedy
    _keys, _par = [], {}
    _out = _ss.band_greedy_shares_multi(
        [0.5, 0.5], [0], [2], [1.0, 1.0], {}, ["a", "b"], (), None,
        n_starts=1, rng_seed=0, keys_out=_keys, par_info=_par)
finally:
    _ss.band_greedy_shares = _orig

check("4  par_info is NON-EMPTY on the shipped n_starts=1 path", bool(_par), repr(_par))
check("4  it carries the wall time [feas-par] prints",
      isinstance(_par.get("secs"), float) and _par["secs"] >= 0.0)
check("4  starts == 1 and parallel is False, so tab_2 takes the SERIAL branch",
      _par.get("starts") == 1 and _par.get("parallel") is False)
check("4  the greedy still ran exactly once", len(_calls) == 1)
check("4  it returns the same two objects the greedy gave it",
      _out == ("SHARES", (0.0, 0.0)))
check("4  keys_out is unchanged in shape", _keys == [(0, 0.0, 0.0)])
check("4  tab_2's checksum block is gated on _bg_par, which is now filled",
      "if _bg_par:" in T2C and "WINNING SEED CHECKSUM" in T2)


# === 5. READ-ONLY: the timing cannot move an answer =====================================
check("5  no mark or report result is ever assigned or tested",
      "= _sc_mark(" not in T2C and "= _sc_report(" not in T2C
      and "if _sc_mark(" not in T2C and "if _sc_report(" not in T2C)
check("5  the harness state is only ever reached through the two closures",
      T2C.count("_sctm") == 3, "the dict literal + the two default args")
check("5  the _n <= 1 path calls _greedy(base) exactly once, as before",
      _n1.count("_greedy(base)") == 1)
check("5  nothing in seed_search's shipped arithmetic changed",
      "_par = False" in SSC and "best_s, best_key = _out[0]" in SSC)
check("5  band_projection records 19jy", "19jy-scaffold-timing" in BP)
check("5  both files compile",
      bool(compile(T2, "tab_2", "exec")) and bool(compile(SS, "seed_search", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
