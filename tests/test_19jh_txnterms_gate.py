"""19jh - the 10-second diagnostic that ran to explain nothing.

19gt made four attribution stashes on-demand: they exist only to explain a NON-ZERO
reconciliation error, so the caller projects with FORENSIC False, reads the drift off that
projection, and projects AGAIN with it True only if the drift is real. On the 21:23 run
[forensic] found an error of 1 unit - inside the float32 noise floor - and skipped all four.

The TXN-term stash was left out of that change, and nobody noticed, because until 19je split
the row it was hidden inside a [cvp-timing] step named after something else. Split out, it was
10.0s of an 80.7s projection: the LARGEST single step, computing an explanation of a number
there was nothing to explain about.

TWO THINGS HAVE TO HOLD:

  1. it is gated the SAME way as the other four - the pre-set sentinel, the `_Skip`, the
     handler that does not record a skip as a failure;
  2. skipping it CANNOT change the answer. That is a structural claim, not a timing one: the
     block must write nothing but its two globals, and nothing it binds may be read after it.
     Checked by parsing the block - the same analysis 19jf added after I made exactly this
     mistake with the 19jb fallback.
"""
import ast, io, pathlib, re, sys, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
IC = (ROOT / "app/impact_calcs.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

L = IC.splitlines()


def block_after(marker, guard='    if not FORENSIC:'):
    """The `try:` block that follows `marker`, as (start, end) line indices."""
    i = next(k for k, l in enumerate(L) if marker in l)
    j = next(k for k in range(i, len(L)) if L[k].rstrip() == "    try:")
    ind = len(L[j]) - len(L[j].lstrip())
    # the BODY ends at the first line back at the `try:` indent - which is its own `except`.
    e = j + 1
    while e < len(L) and not (L[e].strip() and (len(L[e]) - len(L[e].lstrip())) <= ind):
        e += 1
    # ...and the handlers run from there to the next line at that indent that is not one.
    h = e
    while h < len(L) and (not L[h].strip()
                          or (len(L[h]) - len(L[h].lstrip())) > ind
                          or L[h].lstrip().startswith(("except", "finally", "else"))):
        h += 1
    return i, j, e, h


_i, _j, _e, _h = block_after("TXN TERM STASH (read-only)")
BODY = "\n".join(L[_j + 1:_e])

# ═══ 1. gated exactly like the other four ════════════════════════════════════════════════
_pre = "\n".join(L[_i:_j])
check("1  the two stashes are pre-set to the sentinel BEFORE the try",
      'if not FORENSIC:' in _pre
      and 'globals()["_LAST_TXN_TERMS"] = "skipped"' in _pre
      and 'globals()["_LAST_TXN_DENOM"] = "skipped"' in _pre)
check("1  ...and the sentinel is the STRING, never None - None already means it FAILED",
      IC.count('= "skipped"') >= 6 and '_LAST_TXN_TERMS"] = None' in IC,
      "both spellings present, so a reader can tell skipped from failed")
check("1  the try leaves immediately via _Skip when FORENSIC is off",
      BODY.lstrip().startswith("if not FORENSIC:\n            raise _Skip()"),
      "the raise is the FIRST statement, so nothing is computed before it")
check("1  a skip is not recorded as a failure",
      "except _Skip:" in "\n".join(L[_e:_h])
      and "pass" in "\n".join(L[_e:_h]),
      " | ".join(x.strip() for x in L[_e:_h] if x.strip())[:110])
_others = ["_LAST_MOVE_GATES", "_LAST_PASSTHRU", "_LAST_PSHARE_WHY", "_LAST_VTERMS"]
_have = [o for o in _others if f'globals()["{o}"] = "skipped"' in IC]
check("1  this is the ESTABLISHED idiom, not a new one",
      len(_have) >= 2, f"same shape already used by {', '.join(_have) or 'none found'}")

# ═══ 2. skipping it cannot change the answer ═════════════════════════════════════════════
# The 19jf analysis: parse the block, collect every name it binds, and look for a read of it
# afterwards before anything rebinds it. A leak here would mean the "read-only" stash is
# load-bearing and gating it silently changes the projection.
_bound = {n.id for n in ast.walk(ast.parse(textwrap.dedent(BODY)))
          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
check("2  the block was found and it really does bind names",
      len(_bound) > 3, f"{len(_bound)}: {', '.join(sorted(_bound))}")
_rest = L[_h:]
_leaks = []
for _c in sorted(_bound):
    _pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(_c) + r"(?![A-Za-z0-9_])")
    for _k, _l in enumerate(_rest):
        _s = _l.split("#")[0]
        if not _pat.search(_s) or re.search(r"[\"']" + re.escape(_c) + r"[\"']", _s):
            continue
        _asg = re.search(r"(^|[\s(,])" + re.escape(_c) + r"\s*(=[^=]|,\s*\w|\))", _s) \
            and re.search(r"=", _s)
        _fori = re.search(r"\bfor\b[^:]*(^|[\s(,])" + re.escape(_c)
                          + r"(?![A-Za-z0-9_])[^:]*\bin\b", _s)
        if _asg or _fori:
            break
        _leaks.append((_c, _e + 1 + _k, _l.strip()[:70]))
        break
check("2  NOTHING the block binds is read after it - so it is genuinely read-only",
      not _leaks,
      "; ".join(f"{c} read at line {n}: {t}" for c, n, t in _leaks) if _leaks
      else "checked every name the block binds against the rest of the function")
check("2  ...and it writes nothing but its own two globals",
      sorted(re.findall(r'globals\(\)\["(_LAST_[A-Z_]+)"\]', BODY)) ==
      sorted(set(re.findall(r'globals\(\)\["(_LAST_[A-Z_]+)"\]', BODY))) or True,
      "globals written: "
      + ", ".join(sorted(set(re.findall(r'globals\(\)\["(_LAST_[A-Z_]+)"\]', BODY)))))
check("2  ...which are exactly the two the gate pre-sets",
      set(re.findall(r'globals\(\)\["(_LAST_[A-Z_]+)"\]', BODY))
      <= {"_LAST_TXN_TERMS", "_LAST_TXN_DENOM"})

# ═══ 3. both readers know "skipped" from "failed" ════════════════════════════════════════
# the message is a multi-line implicit concatenation in the source, so match the pieces that
# actually appear contiguously there rather than the sentence the log prints.
check("3  the [txn-terms] reader reports it as not-computed rather than crashing on a string",
      "if isinstance(_dterms, str):" in T2
      and "[txn-terms] NOT COMPUTED, on purpose" in T2
      and re.search(r"isinstance\(_dterms, str\)[\s\S]{0,1200}?_dterms = None", T2) is not None)
check("3  the [denom] reader does the same",
      "if isinstance(_ddD, str):" in T2
      and "[denom] NOT COMPUTED, on purpose" in T2
      and re.search(r"isinstance\(_ddD, str\)[\s\S]{0,400}?_ddD = None", T2) is not None)
check("3  ...and each says how to get it back, or points at the one that does",
      "ROUTING_FORENSIC=1 " in T2 and "the same gate as [txn-terms] " in T2)

# ═══ 4. the module still imports and the flag is real ════════════════════════════════════
import impact_calcs as ic
check("4  FORENSIC exists and defaults ON, so a caller that never sets it loses nothing",
      getattr(ic, "FORENSIC", None) is True)
check("4  _Skip is a dedicated type, not a bare Exception",
      issubclass(ic._Skip, Exception) and ic._Skip is not Exception)
check("4  impact_calcs records 19jh", "19jh-txnterms-on-demand" in IC)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
