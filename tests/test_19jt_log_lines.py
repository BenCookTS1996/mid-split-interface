"""19jt - four run-log lines from the 2026-09-04 10:48 log.

Two are formatting. Two are not:

  * `enforced` in the reconciliation chain. There is no enforcement PASS any more - dials,
    tilts and the enforcement pass were all removed, and the wall-time line said so on the
    same run - so a stage called "enforced" named machinery that does not exist. The STEP is
    real and still runs: it is the value after build_split_exports and the backup catch-all
    blend, which [rung] decomposes under the name ENFORCEMENT. Renamed to `exported`, for the
    function that produces it.

  * TWO lines counted "wallet-incapable gatewayFid(s)" and disagreed - 17 on [emask-grain],
    18 on `eligibility:`. That was not an error: they apply DIFFERENT RULES to the same MID
    list. [emask-grain] ORs capability over a (vampMid, currency) pair's active fids;
    `eligibility:` takes the fids explicitly flagged processWallet=FALSE. Ben kept the
    per-fid read - the one you can trace back to a row in the CSV - so [emask-grain]'s counts
    are gone. What stays there is what only IT can say: the grain the mask applies at, and the
    cross-brand warning, which no per-fid count can reach because it is a property of the pair.

Checked at source level, like 19ip and 19js: these are display facts with nothing to return.
"""
import ast, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)


# ═══ 1. the dial-0 variation: one stat per line ══════════════════════════════════════════
check("1  the GENETIC branch's one-line form is gone",
      "GA single variation (dial 0): MIDs over cap=" not in T2)
# the softmax/thompson branch keeps its own one-liner. It is a DIFFERENT engine's line, only
# one of the two ever prints in a run, and this run does not take that path - so it is left
# alone rather than changed unmeasured.
check("1  ...and the non-genetic branch still has its own, untouched",
      "single variation (dial 0): MIDs over cap=" in T2
      and T2.count("(informational — no cap enforced), succ=") == 1)
check("1  the header stands alone", 'log("   ── GA single variation (dial 0)")' in T2)
check("1  each stat is its own indented line, in a fixed-width column",
      all(_s in T2 for _s in ("'MIDs over cap':<18", "'success rate':<18", "'risk rate':<18")))
check("1  ...and the three values still come from the same two sources as before",
      "summ['expected_success_rate']" in T2 and "summ['expected_risk_rate']" in T2
      and "_mo:>10,}" in T2)

# ═══ 2. the wall-time line ═══════════════════════════════════════════════════════════════
check("2  the wall time still prints", 'log(f"   GA total wall time: {_fmt_secs(_ga_wall_tot)}")' in T2)
check("2  ...and the parenthetical describing removed machinery does not",
      "one full-matrix GA; no dials, no tilts, no enforcement pass" not in T2)

# ═══ 3. the chain stage ══════════════════════════════════════════════════════════════════
check("3  the chain builds an `exported` stage, not an `enforced` one",
      '_chain += f" → exported {float(_enfnow):,.0f}"' in T2
      and '_chain += f" → enforced ' not in T2)
_flat = " ".join(T2.split())
check("3  the header names the same five stages the chain builds",
      'raw \\u2192 GA-fitness \\u2192 shipped " "\\u2192 exported \\u2192 delivered' in _flat
      and "\\u2192 enforced" not in T2)
check("3  the rename is explained where it is, and names what the step really is",
      "19jt: `enforced` RENAMED to `exported`" in T2
      and "build_split_exports" in T2)
check("3  [rung]'s ENFORCEMENT term is UNTOUCHED, so the two logs stay linkable",
      "build_split_exports + backup blend rewrote the " in T2
      and "ENFORCEMENT" in T2,
      "the chain says `exported`, [rung] still decomposes the same step as ENFORCEMENT")
# the stage still reads the same value it always did - a rename must not move a number
_src_lines = T2.splitlines()
_exp_i = next((_i for _i, _l in enumerate(_src_lines) if "→ exported {float(_enfnow)" in _l), -1)
check("3  ...and it is still `_enf_by_midl`'s value, so only the LABEL changed",
      _exp_i > 0 and "_enfnow = _enf_by_midl.get(_midl)" in T2)

# ═══ 4. ONE count of wallet-incapable fids, not two that disagree ════════════════════════
_code = [_l for _l in _src_lines if not _l.lstrip().startswith("#")]
check("4  exactly ONE line of CODE emits a wallet-incapable gatewayFid count",
      sum(1 for _l in _code if "wallet-incapable gatewayFid(s)" in _l) == 1,
      "the phrase also appears in the comment explaining why the other one went")
check("4  the survivor is the `eligibility:` line - the per-fid read",
      'f"{len(_wf):,} wallet-incapable gatewayFid(s), "' in T2
      and "process_wallet_incapable(_mmp)" in T2)
check("4  [emask-grain]'s own counts are gone",
      "wallet-incapable \"\n" not in T2
      and "USA-only gatewayFid(s) for \"" not in T2
      and "THE MASK IS UNCHANGED" not in T2)
check("4  ...and it points at where the counts went, rather than going silent",
      "Counts are on " in T2 and "`eligibility:` line below, per gatewayFid" in T2)
check("4  what only [emask-grain] can say SURVIVES: the grain it masks at",
      "wallet/USA capability is masked at (vampMid, \"\n" in T2
      or "capability is masked at (vampMid, " in T2)
check("4  ...and the cross-brand warning, which is a property of the PAIR",
      "carry fids from ANOTHER brand" in T2 and "cross_brand_pairs" in T2)
check("4  ...and the source line",
      'log(f"      source: {_pair_src}")' in T2)
check("4  the reason both lines existed, and why this one lost, is recorded at the deletion",
      "19jt: THE COUNTS ARE GONE FROM HERE" in T2
      and "they apply\n" in T2 and "DIFFERENT RULES to the same MID list" in T2)

# ═══ 5. nothing bound only inside a new conditional is read outside it (the 19jf trap) ════
_fn = next((n for n in ast.walk(ast.parse(T2))
            if isinstance(n, ast.FunctionDef) and n.name == "render"), None)
check("5  tab_2.render parsed", _fn is not None)
if _fn is not None:
    _b, _r = {}, {}
    for _n in ast.walk(_fn):
        if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Store):
            _b.setdefault(_n.id, []).append(_n.lineno)
        elif isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Load):
            _r.setdefault(_n.id, []).append(_n.lineno)
    _bad = [nm for nm in ("_emv", "_mo", "summ", "_enfnow", "_chain", "_wf", "_uf")
            if _r.get(nm) and min(_r[nm]) < min(_b.get(nm, [10 ** 9]))]
    check("5  every name these lines read is bound before its first read", not _bad,
          f"read-before-bind: {_bad}" if _bad else "checked by parsing render(), not by eye")

check("6  tab_2 still compiles", bool(compile(T2, "tab_2", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
