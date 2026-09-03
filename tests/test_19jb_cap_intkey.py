"""19jb - one int64 codebook for the scaffold's cross-frame lookups.

[cap-timing] 19ja split the two rows that were 39.9s of a 51.7s table, and both turned out
to be the same mechanism seen twice:

  * `_Pc -> _T0` builds a 7-part string per row of _T0 AND of _Pc before pandas hashes
    either one;
  * the three maps over _P build tuple-keyed dicts and then walk them with np.fromiter -
    one Python tuple construction and one hash per row of a multi-million-row frame.

Both are what [gk-code] already replaced with an int64 key elsewhere. What makes this one
easy to be sure about is that the key decides only WHICH source row a destination row reads:
the value is the same float, gathered rather than hashed. No arithmetic changes and no float
sum is reassociated.

So the tests below are about the KEY inducing the same equivalence as the string join and the
tuple, including the two duplicate policies (`set_index(...).to_dict()` keeps the LAST,
`drop_duplicates()` before it keeps the FIRST) - the detail that would be silently wrong.
The fixture reproduces all five lookups of the call site side by side, old and new.
"""
import pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")

from routing_optimiser.s4_search.band_projection import (
    joint_col_codes, mix_codes, joint_lookup)

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def bits(a):
    return np.asarray(a).astype(np.float64).view(np.int64)


# ═══ a fixture shaped like the real frames ═══════════════════════════════════════════════
rng = np.random.default_rng(19)
# SMALL domains on purpose. The point of this fixture is the HIT path - duplicate keys,
# keep-last vs keep-first, a _Pc row landing on the right _T0 row - and a wide key space
# would make almost every lookup a miss and test nothing but the default.
CUR = ["usd", "eur", "gbp"]
BIN = [f"4{n:05d}" for n in range(5)]
RPGT = ["annual sub sale", "monthly initial", "upgrades"]
PMP = ["card", "wallet"]
CTRY = ["us", "gb", "de"]
MIDL = [f"mid{n}" for n in range(4)]

def frame(n, per_lo, per_hi, tmax, seed):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "_cur": r.choice(CUR, n), "_bin": r.choice(BIN, n), "_rpgt": r.choice(RPGT, n),
        "_pmp": r.choice(PMP, n), "_ctry": r.choice(CTRY, n), "_midl": r.choice(MIDL, n),
        "_per": r.integers(per_lo, per_hi, n).astype(np.int64),
        "_t": r.integers(0, tmax + 1, n).astype(np.int64)})
    d["_om"] = (d["_per"] - d["_t"]).astype(np.int64)
    d["_fcp"] = r.random(n)
    d["_pr"] = r.random(n)
    d["_vc"] = r.random(n) * 100.0
    d["_bf"] = (r.random(n) < 0.12).astype(int)
    return d

_T0 = frame(4000, 20, 26, 0, 1)
_P = frame(9000, 20, 26, 5, 2)
_Pc = frame(7000, 20, 26, 5, 3)
# duplicate keys on purpose, so keep-last vs keep-first is actually exercised
_T0 = pd.concat([_T0, _T0.iloc[:400].assign(_fcp=lambda d: d["_fcp"] + 1.0)],
                ignore_index=True)
_P = pd.concat([_P, _P[_P["_t"] == 0].iloc[:600].assign(_fcp=lambda d: d["_fcp"] + 1.0,
                                                        _pr=lambda d: d["_pr"] + 1.0)],
               ignore_index=True)


# ═══ 1. THE OLD CODE, verbatim ═══════════════════════════════════════════════════════════
def old_arrays(T0, P, Pc):
    t0_join = (T0["_cur"] + "|" + T0["_bin"] + "|" + T0["_rpgt"] + "|" + T0["_pmp"] + "|"
               + T0["_ctry"] + "|" + T0["_midl"] + "|" + T0["_per"].astype(str)).to_numpy()
    bf_mask = (T0["_bf"].to_numpy() > 0) if "_bf" in T0.columns else np.zeros(len(T0), bool)
    pc_join = (Pc["_cur"] + "|" + Pc["_bin"] + "|" + Pc["_rpgt"] + "|" + Pc["_pmp"] + "|"
               + Pc["_ctry"] + "|" + Pc["_midl"] + "|" + Pc["_om"].astype(str)).to_numpy()
    valid = ~bf_mask
    t0_pos = pd.Series(np.where(valid)[0], index=t0_join[valid])
    t0_pos = t0_pos[~t0_pos.index.duplicated(keep="last")]
    Pc_to_t0 = t0_pos.reindex(pc_join).fillna(-1).to_numpy().astype(np.int64)

    P0 = P[P["_t"] == 0]
    fcp_orig_map = P0.set_index(
        ["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_midl", "_per"])["_fcp"].to_dict()
    prapp_map = (P0.drop_duplicates(["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per"])
                 .set_index(["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per"])["_pr"].to_dict())
    P_origin = (P["_per"] - P["_t"]).to_numpy()
    fcp_o_P = np.fromiter(
        (fcp_orig_map.get((c, b, r, pm, ct, ml, o), 0.0)
         for c, b, r, pm, ct, ml, o in
         zip(P["_cur"], P["_bin"], P["_rpgt"], P["_pmp"], P["_ctry"], P["_midl"], P_origin)),
        dtype=float, count=len(P))
    mvraw = P["_vc"].to_numpy(float) * fcp_o_P
    Pw = P.assign(_mvraw=mvraw)
    mvp_map = Pw.groupby(["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_per", "_t"],
                         observed=True)["_mvraw"].sum().to_dict()
    pc_prapp_a = np.fromiter(
        (prapp_map.get((c, b, r, pm, ct, p), 0.0)
         for c, b, r, pm, ct, p in
         zip(Pc["_cur"], Pc["_bin"], Pc["_rpgt"], Pc["_pmp"], Pc["_ctry"], Pc["_per"])),
        dtype=float, count=len(Pc))
    movedvpool = np.fromiter(
        (mvp_map.get((c, b, r, pm, ct, p, t), 0.0)
         for c, b, r, pm, ct, p, t in
         zip(Pc["_cur"], Pc["_bin"], Pc["_rpgt"], Pc["_pmp"], Pc["_ctry"], Pc["_per"], Pc["_t"])),
        dtype=float, count=len(Pc)) * pc_prapp_a
    return dict(Pc_to_t0=Pc_to_t0, fcp_o_P=fcp_o_P, mvraw=mvraw,
                pc_prapp_a=pc_prapp_a, Pc_movedvpool_a=movedvpool)


# ═══ 2. THE NEW CODE, composed exactly as the call site composes it ══════════════════════
def new_arrays(T0, P, Pc):
    CKC = ["_cur", "_bin", "_rpgt", "_pmp", "_ctry", "_midl"]
    r = joint_col_codes([{n: d[n] for n in CKC} for d in (T0, P, Pc)], CKC, sep="|")
    if r is None:
        return None
    ckp, ckrad, cklen = r
    c_T0, c_P, c_Pc = ckp
    org = (P["_per"] - P["_t"]).to_numpy()
    def off(arrs):
        lo = min(int(np.min(a)) for a in arrs if len(a))
        hi = max(int(np.max(a)) for a in arrs if len(a))
        return lo, int(hi - lo + 1)
    pmn, prd = off([np.asarray(T0["_per"]), np.asarray(P["_per"]), org,
                    np.asarray(Pc["_per"]), np.asarray(Pc["_om"])])
    tmn, trd = off([np.asarray(P["_t"]), np.asarray(Pc["_t"])])
    def pcode(v, m=None):
        return np.asarray(v, np.int64) - (pmn if m is None else m)
    K_T0 = mix_codes(c_T0 + [pcode(T0["_per"])], ckrad + [prd], n=cklen[0])
    K_Pcom = mix_codes(c_Pc + [pcode(Pc["_om"])], ckrad + [prd], n=cklen[2])
    K_Pper = mix_codes(c_P + [pcode(P["_per"])], ckrad + [prd], n=cklen[1])
    K_Porg = mix_codes(c_P + [pcode(org)], ckrad + [prd], n=cklen[1])
    K5_P = mix_codes(c_P[:5] + [pcode(P["_per"])], ckrad[:5] + [prd], n=cklen[1])
    K5_Pc = mix_codes(c_Pc[:5] + [pcode(Pc["_per"])], ckrad[:5] + [prd], n=cklen[2])
    K6_P = mix_codes(c_P[:5] + [pcode(P["_per"]), pcode(P["_t"], tmn)],
                     ckrad[:5] + [prd, trd], n=cklen[1])
    K6_Pc = mix_codes(c_Pc[:5] + [pcode(Pc["_per"]), pcode(Pc["_t"], tmn)],
                      ckrad[:5] + [prd, trd], n=cklen[2])
    bf = (T0["_bf"].to_numpy() > 0)
    val = ~bf
    Pc_to_t0 = joint_lookup(K_T0[val], np.where(val)[0], K_Pcom,
                            default=-1, keep="last").astype(np.int64)
    p0 = (P["_t"].to_numpy() == 0)
    fcp_o_P = joint_lookup(K_Pper[p0], P["_fcp"].to_numpy(float)[p0], K_Porg,
                           default=0.0, keep="last")
    pc_prapp_a = joint_lookup(K5_P[p0], P["_pr"].to_numpy(float)[p0], K5_Pc,
                              default=0.0, keep="first")
    mvraw = P["_vc"].to_numpy(float) * fcp_o_P
    mv = pd.Series(mvraw, index=P.index).groupby(
        pd.Series(K6_P, index=P.index), observed=True).sum()
    movedvpool = joint_lookup(mv.index.to_numpy(), mv.to_numpy(float), K6_Pc,
                              default=0.0, keep="last") * pc_prapp_a
    return dict(Pc_to_t0=Pc_to_t0, fcp_o_P=fcp_o_P, mvraw=mvraw,
                pc_prapp_a=pc_prapp_a, Pc_movedvpool_a=movedvpool)


OLD = old_arrays(_T0, _P, _Pc)
NEW = new_arrays(_T0, _P, _Pc)
check("1  the int64 codebook was buildable on the fixture", NEW is not None)
for _k in ("Pc_to_t0", "fcp_o_P", "mvraw", "pc_prapp_a", "Pc_movedvpool_a"):
    check(f"1  `{_k}` is bit-identical to the string/tuple-key version",
          np.array_equal(bits(OLD[_k]), bits(NEW[_k])),
          f"max|d| {float(np.abs(np.asarray(OLD[_k], float) - np.asarray(NEW[_k], float)).max()):.3e}")
_hitr = float((OLD["Pc_to_t0"] >= 0).mean())
check("1  ...and the fixture exercises the HIT path, not just the default",
      int((OLD["Pc_to_t0"] >= 0).sum()) >= 1000
      and int((OLD["Pc_to_t0"] < 0).sum()) >= 1000
      and int((OLD["fcp_o_P"] > 0).sum()) >= 1000
      and int((OLD["fcp_o_P"] == 0).sum()) >= 1000,
      f"{int((OLD['Pc_to_t0'] >= 0).sum()):,} of {len(_Pc):,} _Pc rows found a _T0 row "
      f"({_hitr:.0%}); {int((OLD['fcp_o_P'] > 0).sum()):,} of {len(_P):,} _P rows found an "
      "origin fcp")
check("1  ...and duplicate keys really are present, so keep-last/keep-first matters",
      bool(pd.Series(NEW is not None and 1).any())
      and int(pd.Index(_T0.index).size) > int(pd.Index(
          (_T0["_cur"] + _T0["_bin"] + _T0["_rpgt"] + _T0["_pmp"] + _T0["_ctry"]
           + _T0["_midl"] + _T0["_per"].astype(str)).unique()).size),
      "the _T0 join key is not unique on this fixture, which is the case that would break")

# THE DUPLICATE POLICY IS THE PART THAT WOULD BE SILENTLY WRONG
_dupk = np.array([7, 7, 7, 3], np.int64)
_dupv = np.array([10.0, 20.0, 30.0, 40.0])
check("2  keep='last' matches `set_index(...).to_dict()` on a duplicated key",
      list(joint_lookup(_dupk, _dupv, np.array([7, 3, 9], np.int64), default=0.0,
                        keep="last")) == [30.0, 40.0, 0.0])
check("2  keep='first' matches `drop_duplicates()` before the dict",
      list(joint_lookup(_dupk, _dupv, np.array([7, 3, 9], np.int64), default=0.0,
                        keep="first")) == [10.0, 40.0, 0.0])
check("2  a source value that IS NaN comes back as NaN, not as the default",
      np.isnan(joint_lookup(np.array([1], np.int64), np.array([np.nan]),
                            np.array([1, 2], np.int64), default=-7.0)[0])
      and joint_lookup(np.array([1], np.int64), np.array([np.nan]),
                       np.array([1, 2], np.int64), default=-7.0)[1] == -7.0)

# CROSS-FRAME COMPARABILITY, which is the whole promise
_fa = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
_fb = pd.DataFrame({"a": ["z", "q", "x"], "b": [3, 9, 1]})
_p, _r, _l = joint_col_codes([_fa, _fb], ["a", "b"])
_ka = mix_codes(_p[0], _r, n=_l[0]); _kb = mix_codes(_p[1], _r, n=_l[1])
check("3  the same tuple in two different frames gets the same key",
      _ka[2] == _kb[0] and _ka[0] == _kb[2] and _kb[1] not in set(_ka.tolist()))
check("3  ...and a key is never shared by two different tuples",
      len(set(_ka.tolist()) | set(_kb.tolist())) == 4)

# stringify: int 403163 and str "403163" are ONE key for a join, TWO for a tuple
_sa = pd.DataFrame({"a": [403163]})
_sb = pd.DataFrame({"a": ["403163"]})
_ps, _rs, _ls = joint_col_codes([_sa, _sb], ["a"], stringify=True)
_pr_, _rr, _lr = joint_col_codes([_sa, _sb], ["a"], stringify=False)
check("3  stringify=True merges 403163 and '403163', as a '|'-join does",
      mix_codes(_ps[0], _rs, n=1)[0] == mix_codes(_ps[1], _rs, n=1)[0])
check("3  stringify=False keeps them apart, as a tuple key does",
      mix_codes(_pr_[0], _rr, n=1)[0] != mix_codes(_pr_[1], _rr, n=1)[0])

check("4  a value containing the separator DECLINES rather than quietly fixing the join",
      joint_col_codes([pd.DataFrame({"a": ["x|y"]})], ["a"], sep="|") is None)
check("4  ...and without a sep it is allowed, because no join is being replaced",
      joint_col_codes([pd.DataFrame({"a": ["x|y"]})], ["a"]) is not None)
_nn = pd.DataFrame({"a": [None, np.nan, "None", "nan"]})
_pn, _rn, _ln = joint_col_codes([_nn], ["a"])
_kn = mix_codes(_pn[0], _rn, n=4)
check("4  None and NaN do not collapse into one code, and neither swallows its own spelling",
      len(set(_kn.tolist())) == len(set(map(str, [None, np.nan, "None", "nan"]))),
      f"codes {sorted(set(_kn.tolist()))}")
check("4  mixed-radix overflow past 2**62 declines instead of wrapping",
      mix_codes([np.zeros(1, np.int64)] * 4, [2 ** 20] * 4) is None)
check("4  an empty frame is handled, not crashed on",
      joint_col_codes([pd.DataFrame({"a": []})], ["a"]) is not None)

# ═══ 5. the wiring ═══════════════════════════════════════════════════════════════════════
check("5  the int key is ON by default and revertible",
      'os.environ.get("ROUTING_CAP_INTKEY", "1") != "0"' in T2)
check("5  it VERIFIES itself against the code it replaces, and that is on by default",
      'os.environ.get("ROUTING_CAPKEY_VERIFY", "1") != "0"' in T2
      and "VERIFY FAILED on " in T2)
check("5  a verify failure ships the STRING/TUPLE keys, not the int key",
      "_CK_NEW = None\n                                    _CKEY[\"why\"] = (\"VERIFY FAILED on \"" in T2)
check("5  the comparison is on BIT PATTERNS, not allclose",
      ".astype(np.float64).view(np.int64)," in T2)
check("5  sep='|' is passed, because the _Pc -> _T0 half replaces a '|'-join",
      '_ckr = _jcc(_ckf, _CKC, sep="|")' in T2)
check("5  the moved-pool groupby keeps its own rows in their own order",
      "THE GROUPBY KEEPS ITS OWN ROWS IN THEIR OWN ORDER" in T2)
check("5  [cap-key] says which way it went, either way", '[cap-key] ON - ' in T2 and '[cap-key] OFF - ' in T2)
check("5  the original string/tuple code is KEPT as the fallback, not deleted",
      "_fcp_orig_map = _P0.set_index(" in T2 and "np.fromiter(" in T2)


# ═══ 6. THE FALLBACK BLOCK DEFINES NOTHING THE CODE BELOW NEEDS ══════════════════════════
# 19jf. Wrapping the string/tuple-key code in `if _CK_NEW is None or _CK_VERIFY:` made every
# name it assigns conditional. Two of them - `_SEP` and `_Pc_vc_a` - are not lookups at all
# and are read hundreds of lines further down, so the FIRST run with the verify off died on
# `local variable '_SEP' referenced before assignment`. The run before it passed only because
# the verify made the block execute anyway, which is what hid it.
#
# So this is not a check that those two names are hoisted. It is a check that there is no
# THIRD one: the block is parsed, every name it binds is collected, and each is looked for in
# the rest of the file before anything rebinds it.
import ast as _ast, re as _re, textwrap as _tw

_L = T2.splitlines()
_g = next(i for i, l in enumerate(_L) if l.strip() == "if _CK_NEW is None or _CK_VERIFY:")
_ind = len(_L[_g]) - len(_L[_g].lstrip())
_e = _g + 1
while _e < len(_L) and not (_L[_e].strip() and (len(_L[_e]) - len(_L[_e].lstrip())) <= _ind):
    _e += 1
_bound = {n.id for n in _ast.walk(_ast.parse(_tw.dedent("\n".join(_L[_g + 1:_e]))))
          if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Store)}
check("6  the fallback block was found and it really does bind names",
      len(_bound) > 5, f"{len(_bound)} name(s): {', '.join(sorted(_bound))}")

# names the ADOPT block sets on the fast path, so they are defined either way
_adopted = {"_Pc_to_t0", "_fcp_o_P", "_pc_prapp_a", "_Pc_movedvpool_a"}
_leaks = []
for _c in sorted(_bound - _adopted):
    _pat = _re.compile(r"(?<![A-Za-z0-9_])" + _re.escape(_c) + r"(?![A-Za-z0-9_])")
    for _i, _l in enumerate(_L[_e:]):
        _s = _l.split("#")[0]
        if not _pat.search(_s) or _re.search(r"[\"']" + _re.escape(_c) + r"[\"']", _s):
            continue                                  # not a use, or a same-spelled string key
        _asg = _re.search(r"(^|[\s(,])" + _re.escape(_c) + r"\s*(=[^=]|,\s*\w|\))", _s) \
            and _re.search(r"=", _s)
        _fori = _re.search(r"\bfor\b[^:]*(^|[\s(,])" + _re.escape(_c)
                           + r"(?![A-Za-z0-9_])[^:]*\bin\b", _s)
        if _asg or _fori:
            break                                     # rebound before it is ever read
        _leaks.append((_c, _e + 1 + _i, _l.strip()[:70]))
        break
check("6  NO name is defined only by the fallback and then read below it",
      not _leaks,
      "; ".join(f"{c} read at line {n}: {t}" for c, n, t in _leaks) if _leaks
      else "checked every name the block binds against the rest of the file")
# by LINE against the guard the scan found, not by `T2.find` on the guard's text: the 19jf
# comment above the hoist quotes that line verbatim, so a text search finds the comment first.
_hoist = {_n: next((_i for _i, _l in enumerate(_L) if _l.strip().startswith(_n + " = ")), -1)
          for _n in ("_Pc_vc_a", "_SEP")}
check("6  ...and the two that bit are hoisted ABOVE the block, where every path sets them",
      all(0 <= _v < _g for _v in _hoist.values()),
      f"guard at line {_g + 1}; " + ", ".join(f"{_n} first bound at line {_v + 1}"
                                              for _n, _v in _hoist.items()))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
