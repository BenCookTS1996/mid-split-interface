"""19jq - seven run-log lines, three of which were saying something untrue.

Ben read the 2026-09-04 09:52 log and asked for wording changes. Three of them turned out to
be defects wearing a wording problem's clothes, and this test pins the FACTS behind each fix
rather than the prose, so a later edit that reverts the substance fails even if it keeps the
words:

  1. "Σ attempts" and "attempt rows" were identical because the column lookup asked for
     `initialattempt` - the name in attempts_success.sql - while s1_extract renames it to
     `attempts` long before tab_2 sees the frame. The lookup missed on every run and the
     column silently fell through to `len(group)`. The test pins BOTH ends of that rename.
  2. "fc rows" are forecast ROWS, and a forecast row is one gateway within a profile - never
     a profile. Both are now shown, named for what they are.
  3. The pooled-prior fallback has NO GRAIN: it is one attempts-weighted number for the whole
     run. The old line claimed gateways were "scored on a POOLED AVERAGE", which the ④ table
     in the same log then contradicted with a count of 0.

And two are scope, not wording: [emask-grain] and the eligibility line reported capability
counts across EVERY brand in the Master MID list. The counts are now this run's company; the
MASK IS NOT, and the test asserts that separation directly.
"""
import ast, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
AC = (ROOT / "app/app_common.py").read_text(encoding="utf-8")
import app_common as ac

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)


# ═══ 1. the attempts column: the FIX, and the rename that made it necessary ══════════════
_ac_node = None
for _n in ast.walk(ast.parse(T2)):
    if (isinstance(_n, ast.Assign) and len(_n.targets) == 1
            and isinstance(_n.targets[0], ast.Name) and _n.targets[0].id == "_ac"
            and isinstance(_n.value, ast.Call)):
        _ac_node = _n
_names = []
if _ac_node is not None:
    _names = [x.value for x in ast.walk(_ac_node)
              if isinstance(x, ast.Constant) and isinstance(x.value, str)]
check("1  the RPGT table looks for `attempts` FIRST, then `initialattempt`",
      _names[:2] == ["attempts", "initialattempt"], f"searches {_names[:2]}")
check("1  ...and the old initialattempt-only lookup is gone",
      '_ac = ("initialattempt" if "initialattempt" in _rt_a.columns' not in T2)

# THE ROOT CAUSE, pinned at its source: if either half of this rename moves, the lookup above
# starts missing again and the column silently becomes a row count for another few months.
_SQL = (ROOT / "queries/attempts_success.sql").read_text(encoding="utf-8")
_SR = (ROOT / "src/routing_optimiser/s1_extract/success_rates.py").read_text(encoding="utf-8")
check("1  the SQL still emits `initialattempt`", "initialattempt" in _SQL)
check("1  ...and s1_extract still renames it to `attempts` before tab_2 sees it",
      '"initialattempt"): "attempts"' in _SR or 'initialattempt"): "attempts"' in _SR,
      "that rename is WHY the old lookup missed")
check("1  the header says `attempt count`, and `attempt rows` is kept beside it",
      "'attempt count':>15" in T2 and "'attempt rows':>14" in T2
      and "'Σ attempts':>13" not in T2,
      "the old HEADER is gone; the phrase survives in the comment explaining why")
check("1  a run with NO attempts column to sum says so instead of printing a row count twice",
      "attempt count is a ROW COUNT this run" in T2)

# ═══ 2. forecast rows vs forecast profiles ═══════════════════════════════════════════════
check("2  `fc rows` is renamed to `forecast rows` - a row is one gateway within a profile",
      "'forecast rows':>15" in T2 and "'fc rows'" not in T2)
check("2  ...and the profile count Ben actually wanted is a column of its own",
      "'forecast profiles':>19" in T2 and "_rt_prof" in T2)
check("2  the table states which key a profile is, so the number is checkable",
      "a forecast ROW is one gateway within a profile" in T2)

# ═══ 3. the pooled fallback has no grain, and the run counts it elsewhere ════════════════
check("3  the old claim is gone", "gateways are scored on a POOLED AVERAGE" not in T2)
check("3  the line names the REAL fallback: one global attempts-weighted average",
      "ONE global attempts-weighted average" in T2
      and "not a pooled average at any grain of its own" in T2)
check("3  ...and points at the row that counts what actually took it",
      "gateway-rows on POOLED prior" in T2)
# the claim about the fallback is a claim about data_loader, so read data_loader.
_DL = (ROOT / "src/routing_optimiser/s1_extract/data_loader.py").read_text(encoding="utf-8")
check("3  data_loader really does fall back to ONE number for the whole run",
      '_global_rate = float(sr["success"].sum() / sr["attempts"].sum())' in _DL
      and _DL.count("succ.append(_global_rate)") >= 2,
      "an unqualified global mean - no grain, in either loader")

# ═══ 4. require-forecast, [inject], the build string ═════════════════════════════════════
check("4  [require-forecast] says which side the dropped profiles came from",
      "in attempt data, not in the forecast" in T2 and "dropped - not present in forecast" not in T2)
check("4  [inject] is shorter and speaks in cells",
      "[inject] aged cells" in T2 and "aged frame {_n0:,}" not in T2)
check("4  the maximum-revenue-reference line prints the NEWEST tag plus a count",
      "_obt, _obn, _obw = _newest_tag(_ob)" in T2
      and 'reference (build {_obs})' in T2)
check("4  ...using the SAME ordering rule as the build-markers table, not a second copy",
      T2.count("_newest_tag(") == 2 and "_re_bm.findall" not in T2,
      "one call in _bmark, one on the reference line; the regex moved to app_common")

# the ordering rule itself, on the two cases that have caught it out before
check("4  newest_build_tag orders by GENERATION CODE, not by position",
      ac.newest_build_tag("x+19gw-eval-cost+19gu-decode-cap+19ee-maxshare-repair")[:2]
      == ("19gw-eval-cost", 3),
      "genetic_fullmatrix's history is genuinely out of order and read 19ee before 19hw")
check("4  ...and flags that the history IS out of order",
      ac.newest_build_tag("x+19gw-a+19ee-b")[2] is True
      and ac.newest_build_tag("19ee-b+19gw-a")[2] is False)
check("4  ...and survives a single tag, an empty one and None",
      ac.newest_build_tag("solo") == ("solo", 0, False)
      and ac.newest_build_tag("") == ("", 0, False)
      and ac.newest_build_tag(None) == ("", 0, False))

# ═══ 5. capability counts are scoped to the company; THE MASK IS NOT ═════════════════════
_ML = str(ROOT / "data/mappings/Master_MID_List.csv")
_RR = str(ROOT / "config/inputs/visa/visa_routing_restrictions.json")
_have = os.path.exists(_ML) and os.path.exists(_RR)
if not _have:
    print("  ..    5 NOT RUN: the MID list or the visa restrictions JSON is not on this disk.")
else:
    _wc, _uo, _src = ac.capability_pairs(_ML, _RR)
    _v = ac.capability_company_view(_ML, _RR, "TotalAV")
    check("5  the company view resolves at all", _v is not None, f"source {_src}")
    if _v:
        check("5  it is a STRICT SUBSET of the all-brand sets, never something new",
              _v["wallet_pairs"] < _wc and _v["usa_pairs"] <= _uo
              and len(_v["wallet_pairs"]) < len(_wc),
              f"TotalAV {len(_v['wallet_pairs'])} of {len(_wc)} wallet pair(s), "
              f"{len(_v['usa_pairs'])} of {len(_uo)} USA-only")
        check("5  a (vampMid, currency) pair is NOT always single-brand, so this matters",
              len(_v["cross_brand_pairs"]) > 0,
              f"{len(_v['cross_brand_pairs'])} of TotalAV's pair(s) carry another brand's fids - "
              "('paypal', 'gbp') alone spans nine brands on this MID list")
        check("5  every fid it reports really belongs to that brand",
              all(ac._norm_brand(ac.LAST_CAP_FID_BRAND.get(_f, "")) == "totalav"
                  for _f in (_v["wallet_fids"] | _v["usa_fids"])),
              f"{len(_v['wallet_fids'])} wallet fid(s), {len(_v['usa_fids'])} USA-only fid(s)")
        check("5  a brand nobody has returns nothing rather than everything",
              len(ac.capability_company_view(_ML, _RR, "NoSuchBrand")["wallet_pairs"]) == 0,
              "failing OPEN here would silently report another brand's gateways")
        check("5  the all-brand totals are still carried, so the log can say what it filtered",
              _v["all_wallet_pairs"] == len(_wc) and _v["all_usa_pairs"] == len(_uo))
    # THE MEMO USED TO DROP HALF ITS OWN OUTPUT
    _n_meta, _n_fid, _n_split = (len(ac.LAST_CAP_PAIR_META), len(ac.LAST_CAP_FID_BRAND),
                                 len(ac.LAST_CAP_PAIR_SPLITS))
    _wc2, _uo2, _ = ac.capability_pairs(_ML, _RR)          # memo HIT
    check("5  a memo hit restores the module-level records, not just the pairs",
          (_wc2 == _wc and _uo2 == _uo and len(ac.LAST_CAP_PAIR_META) == _n_meta
           and len(ac.LAST_CAP_FID_BRAND) == _n_fid
           and len(ac.LAST_CAP_PAIR_SPLITS) == _n_split),
          "before 19jq the second caller got the pairs and an EMPTY disagreement list")

# SUPERSEDED BY 19jt, and rewritten rather than deleted so the trail is readable. 19jq put
# company-scoped COUNTS on [emask-grain]; 19jt took them off again, because the `eligibility:`
# line was counting the same thing under a different rule (pair-OR vs explicit
# processWallet=FALSE) and the two disagreed - 17 vs 18 - which reads as an error. The
# per-fid read won. What 19jq actually built is still in use: `capability_company_view` is
# what the surviving cross-brand warning is computed from, and section 5 above tests it
# directly. So the assertions here move to the two things 19jt kept.
check("5  [emask-grain] still states the GRAIN the mask applies at - 19hh's whole reason",
      "capability is masked at (vampMid, " in T2 and "in the search and in delivery alike" in T2)
check("5  ...and the cross-brand warning, which only the PAIR view can see",
      "cross_brand_pairs" in T2 and "carry fids from ANOTHER brand" in T2)
check("5  ...and its counts are gone, pointing at the line that kept them (19jt)",
      "THE MASK IS UNCHANGED" not in T2 and "NOT SCOPED TO THIS COMPANY" not in T2
      and "`eligibility:` line below, per gatewayFid" in T2)

# ═══ 6. the eligibility line ═════════════════════════════════════════════════════════════
check("6  it counts gatewayFids, not the mixed fid+vampMid set",
      "wallet-incapable gatewayFid(s)" in T2 and "USA-only gatewayFid(s)" in T2
      and "wallet-incapable id(s)" not in T2 and "USA-only id(s)" not in T2)
check("6  the percentages are gone", "global wallet share" not in T2
      and "global Non-USA share" not in T2)
check("6  the fid-only sets are bound UNCONDITIONALLY (the 19jf trap)",
      "_wal_fids = set()" in T2 and "_uo_fids = set()" in T2)
check("6  ...and a USA restriction that is NOT enforced reports no count",
      "_uo_fids = set()     # 19jq: not enforced, so do not report a count" in T2)
check("6  ENFORCEMENT is untouched - the mixed set is still what is populated and used",
      "_wallet_incapable.add(_f)" in T2 and "_usa_only.add(_f)" in T2
      and "_wallet_incapable.add(_fid2vamp_l[_f])" in T2)

# ═══ 7. nothing bound only inside a new conditional is read outside it ═══════════════════
_fn = next((n for n in ast.walk(ast.parse(T2))
            if isinstance(n, ast.FunctionDef) and n.name == "render"), None)
check("7  tab_2.render parsed", _fn is not None)
if _fn is not None:
    _binds = {}
    for _n in ast.walk(_fn):
        if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Store):
            _binds.setdefault(_n.id, []).append(_n.lineno)
    _reads = {}
    for _n in ast.walk(_fn):
        if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Load):
            _reads.setdefault(_n.id, []).append(_n.lineno)
    _bad = [nm for nm in ("_ac", "_pk_cols", "_rt_prof", "_wal_fids", "_uo_fids",
                          "_emv", "_obt", "_obn", "_obw", "_obs")
            if _reads.get(nm) and min(_reads[nm]) < min(_binds.get(nm, [10 ** 9]))]
    check("7  every name 19jq introduced is bound before its first read", not _bad,
          f"read-before-bind: {_bad}" if _bad else "checked by parsing render(), not by eye")

check("8  both files still compile", bool(compile(T2, "tab_2", "exec"))
      and bool(compile(AC, "app_common", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
