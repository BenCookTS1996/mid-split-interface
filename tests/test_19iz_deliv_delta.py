"""19iz - the deliver delta: the same search, a fifth of the delivery, bit for bit.

`_deliver_full` was 364.9 ms of every 839 ms evaluation on the 19:38 run - the largest row
left after 19ix took the decode down. It is per profile for exactly the reason the decode
was: blocked-caps, eligibility, the cap water-fill and the exploration floor are all
np.add.reduceat / np.repeat along ONE profile's own rows.

Three things have to hold and each is tested here rather than argued:

  1. THE MATH. Delivering a SUB-LAYOUT of profiles is bit-identical to delivering the full
     array and reading those profiles out of it - with real bans, wallet/USA masks, blocked
     rows and a live cap. This is the claim the whole change rests on.
  2. THE ENGINE. run_fullmatrix_ga with the delta armed returns the same split, the same
     fitness and the same history as with it off, on a problem that exercises crossover,
     mutation, elitism and a restart.
  3. THE GUARD. A corrupted subset delivery is CAUGHT by the per-generation self-check, the
     delta disables itself, and the run still finishes on the correct answer.
"""
import importlib.util, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GA_P = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")
GA_SRC = (ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")

from routing_optimiser.s3_problem.eligibility import apply_elig_pop

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)


# ═══ the delivery transform, built from the same expressions tab_2 ships ══════════════════
CAP = 0.62
FLOOR = 0.0

def _sub_rows(cs, cc, hit):
    cs = np.asarray(cs, np.intp); cc = np.asarray(cc, np.intp)
    sel = np.where(np.asarray(hit, bool))[0]
    scc = cc[sel]
    scs = np.zeros(scc.size, np.intp)
    if scc.size:
        np.cumsum(scc[:-1], out=scs[1:])
    t = int(scc.sum())
    rows = (np.arange(t, dtype=np.intp) + np.repeat(cs[sel] - scs, scc)) if t else np.zeros(0, np.intp)
    return rows, scs, scc

def _block(X, blk, cs, cc, fl):
    """tab_2's `_fm_block_narrow` semantics: only the profiles carrying a blocked row move."""
    if blk is None:
        return X
    prof_of = np.repeat(np.arange(np.asarray(cs).size), cc)
    hit = np.zeros(np.asarray(cs).size, bool)
    hit[prof_of[np.asarray(blk, bool)]] = True
    if not hit.any():
        return np.array(X, float, copy=True)
    rows, scs, scc = _sub_rows(cs, cc, hit)
    sub = np.ascontiguousarray(X[:, rows])
    bm = np.asarray(blk, bool)[None, rows]
    capd = np.where(bm, np.minimum(sub, fl), sub)
    freed = sub - capd
    recip = np.where(bm, 0.0, capd)
    fc = np.repeat(np.add.reduceat(freed, scs, axis=1), scc, axis=1)
    rc = np.repeat(np.add.reduceat(recip, scs, axis=1), scc, axis=1)
    has = rc > 1e-12
    add = np.where(has, recip * fc / np.where(has, rc, 1.0), 0.0)
    Y = np.array(X, float, copy=True)
    Y[:, rows] = np.where(has, capd + add, sub)
    return Y

def _cap(X, cs, cc, cap=CAP):
    """tab_2's `_fm_cap`, expression for expression."""
    X = np.asarray(X, float)
    o = X > cap
    if not o.any():
        return X
    exc = np.add.reduceat(np.where(o, X - cap, 0.0), cs, axis=1)
    room = np.where(~o & (X > 1e-12) & (X < cap), cap - X, 0.0)
    pool = np.add.reduceat(room, cs, axis=1)
    ok = (exc > 0.0) & (pool > 1e-12)
    if not ok.any():
        return X
    okr = np.repeat(ok, cc, axis=1)
    f = np.repeat(np.where(ok, exc / np.where(pool > 1e-12, pool, 1.0), 0.0), cc, axis=1)
    add = room * f
    return np.where(okr & o, cap, np.where(okr, X + add, X))

def _floor(X, cs, cc, fl, bs):
    if fl <= 0.0:
        return X
    X = np.asarray(X, float)
    nz = X > 1e-12
    live = (nz | bs[None, :]) if bs is not None else nz
    n = np.add.reduceat(live.astype(float), cs, axis=1)
    flc = np.repeat(np.where(n > 0.0, np.minimum(fl, 1.0 / np.maximum(n, 1.0)), 0.0), cc, axis=1)
    Y = np.where(live, np.maximum(X, flc), X)
    s = np.repeat(np.add.reduceat(Y, cs, axis=1), cc, axis=1)
    return np.where(s > 1e-12, Y / np.where(s > 1e-12, s, 1.0), Y)

def _elig_sub(A, rows, scs, scc, op):
    if op is None:
        return A
    o2 = {"profile_starts": np.asarray(scs, np.intp),
          "profile_counts": np.asarray(scc, np.intp),
          "n_rows": int(np.asarray(rows).size),
          "has_ban": op.get("has_ban"), "has_w": op.get("has_w"), "has_u": op.get("has_u")}
    for k in ("ban", "w_incap", "w_wf", "u_incap", "u_wf"):
        v = op.get(k)
        o2[k] = None if v is None else np.asarray(v)[np.asarray(rows, np.intp)]
    return apply_elig_pop(A, o2)


# ═══ 1. THE MATH: a sub-layout delivers to the same bits ══════════════════════════════════
rng = np.random.default_rng(19)
NP = 40
counts = rng.integers(1, 7, NP).astype(np.intp)
cs = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
N = int(counts.sum())
P = 9

blk = np.zeros(N, bool); blk[rng.choice(N, 11, replace=False)] = True
ban = np.zeros(N, bool); ban[rng.choice(N, 7, replace=False)] = True
w_incap = np.zeros(N, bool); w_incap[rng.choice(N, 13, replace=False)] = True
u_incap = np.zeros(N, bool); u_incap[rng.choice(N, 9, replace=False)] = True
w_wf = np.where(rng.random(N) < 0.4, rng.random(N), 0.0)
u_wf = np.where(rng.random(N) < 0.4, rng.random(N), 0.0)
sf_base = rng.random(N) < 0.5
OP = {"profile_starts": cs, "profile_counts": counts, "ban": ban, "has_ban": True,
      "w_incap": w_incap, "w_wf": w_wf, "has_w": True,
      "u_incap": u_incap, "u_wf": u_wf, "has_u": True, "n_rows": N}

X = rng.random((P, N))
X[0, cs[3]:cs[3] + counts[3]] = 0.0                     # an all-zero profile
X[1, cs[5]] = -0.0                                       # a signed zero
X = np.where(rng.random((P, N)) < 0.08, 0.0, X)
seg = np.repeat(np.add.reduceat(X, cs, axis=1), counts, axis=1)
X = np.where(seg > 0, X / np.where(seg > 0, seg, 1.0), X)

def deliver_full_arr(A, fl=FLOOR):
    A = _block(np.asarray(A, float), blk, cs, counts, 0.01)
    A = apply_elig_pop(A, dict(OP))
    A = _cap(A, cs, counts)
    return _floor(A, cs, counts, fl, sf_base)

def deliver_rows_arr(A_full, rows, scs, scc, rowc, fl=FLOOR):
    r = np.asarray(rows, np.intp)
    sub = np.asarray(A_full, float)[np.asarray(rowc, np.intp), r][None, :]
    sub = _block(sub, np.asarray(blk, bool)[r], scs, scc, 0.01)
    sub = _elig_sub(sub, r, scs, scc, OP)
    sub = _cap(sub, np.asarray(scs, np.intp), np.asarray(scc, np.intp))
    return _floor(sub, np.asarray(scs, np.intp), np.asarray(scc, np.intp), fl,
                  None if sf_base is None else np.asarray(sf_base, bool)[r])

def bits(a):
    return np.asarray(a, float).view(np.int64)

for _fl, _lbl in ((0.0, "floor off (the shipped default)"), (0.02, "floor ARMED")):
    ref = deliver_full_arr(X, _fl)
    # every (candidate, profile) pair, so the comparison covers the whole array
    pp = np.tile(np.arange(NP, dtype=np.intp), P)
    pc = np.repeat(np.arange(P, dtype=np.intp), NP)
    scc = counts[pp]
    scs = np.zeros(scc.size, np.intp); np.cumsum(scc[:-1], out=scs[1:])
    rows = np.arange(int(scc.sum()), dtype=np.intp) + np.repeat(cs[pp] - scs, scc)
    rowc = np.repeat(pc, scc)
    got = np.zeros_like(ref)
    got[rowc, rows] = np.asarray(deliver_rows_arr(X, rows, scs, scc, rowc, _fl))[0]
    check(f"1  EVERY profile through the sub-layout == the full delivery, bit for bit ({_lbl})",
          np.array_equal(bits(ref), bits(got)),
          f"max|d| {float(np.abs(ref - got).max()):.3e}")

# a genuine SUBSET: one scattered handful of pairs, gathered from the reference elsewhere
ref = deliver_full_arr(X, 0.0)
sel = rng.random((P, NP)) < 0.15
pc, pp = np.where(sel)
pc = pc.astype(np.intp); pp = pp.astype(np.intp)
scc = counts[pp]
scs = np.zeros(scc.size, np.intp); np.cumsum(scc[:-1], out=scs[1:])
rows = np.arange(int(scc.sum()), dtype=np.intp) + np.repeat(cs[pp] - scs, scc)
rowc = np.repeat(pc, scc)
got = np.array(ref, float, copy=True)
got[rowc, rows] = np.asarray(deliver_rows_arr(X, rows, scs, scc, rowc, 0.0))[0]
check("1  a SCATTERED subset of (candidate, profile) pairs recomputes to the same bits",
      np.array_equal(bits(ref), bits(got)),
      f"{int(sel.sum())} of {P * NP} pairs recomputed")
check("1  ...and that subset really did touch a blocked profile, a ban and a wallet mask",
      bool(blk[rows].any() and ban[rows].any() and (w_wf[rows] > 0).any()))

# the has_* flags must NOT be recomputed on the slice: a subset with no banned row still
# owes the ban stage's trailing renormalise.
_o_recomputed = dict(OP)
_rows2, _scs2, _scc2 = _sub_rows(cs, counts, ~np.isin(np.arange(NP), np.unique(
    np.repeat(np.arange(NP), counts)[ban])))
if _rows2.size:
    _a = np.asarray(X, float)[:, _rows2]
    _with = _elig_sub(_a, _rows2, _scs2, _scc2, OP)
    _o_off = dict(OP); _o_off["has_ban"] = False
    _without = _elig_sub(_a, _rows2, _scs2, _scc2, _o_off)
    check("1  carrying has_ban VERBATIM matters - dropping it on a ban-free slice changes bits",
          not np.array_equal(bits(_with), bits(_without)),
          "the ban stage ends in a renormalise, which is not the identity")

# ═══ 2. THE ENGINE: armed == unarmed, on a real run_fullmatrix_ga ═════════════════════════
# 19kd: nothing to pop - the delta is unconditional and has no env var. The flag is set per
# arm on the throwaway module `run()` imports.
_sp = importlib.util.spec_from_file_location("ga_19iz", GA_P)
ga = importlib.util.module_from_spec(_sp); sys.modules["ga_19iz"] = ga; _sp.loader.exec_module(ga)

NMID = 4
mid_id = np.repeat(np.arange(NMID), int(np.ceil(N / NMID)))[:N].astype(np.int64)
ctx = {"n_row": N, "n_mid": NMID,
       "profile_starts": cs, "profile_counts": counts,
       "elig": np.ones(N), "base": np.full(N, 0.25),
       "profile_vol": np.repeat(rng.random(NP) * 900 + 100, counts),
       "sr": 0.5 + 0.4 * rng.random(N), "risk": 0.01 * rng.random(N),
       "mid_id": mid_id, "mid_rows": None, "vamp_cap": None,
       "max_share": CAP, "floor": FLOOR}
prob, meta = ga.problem_from_ctx(ctx, soft_cap_mult=1.0)
colmap = np.asarray(meta["keep_idx"])[prob.order]
NROW = int(ctx["n_row"])

# the kept -> full profile map, built exactly as tab_2's [dlv-map] builds it
_pof = np.repeat(np.arange(cs.size, dtype=np.intp), counts)
_kr = _pof[np.asarray(colmap, np.intp)]
_lo = np.minimum.reduceat(_kr, np.asarray(prob.profile_start, np.intp))
_hi = np.maximum.reduceat(_kr, np.asarray(prob.profile_start, np.intp))
MAP_OK = bool(np.array_equal(_lo, _hi)
              and np.array_equal(np.sort(_lo), np.arange(cs.size))
              and np.array_equal(counts[_lo], np.asarray(prob.profile_len, np.intp)))
check("2  the kept-grain -> full-grain profile map is a verified bijection on this fixture",
      MAP_OK)
CMINV = np.full(NROW, -1, np.intp); CMINV[np.asarray(colmap, np.intp)] = np.arange(colmap.size)
DMAP = {"jmap": np.asarray(_lo, np.intp), "cs": cs, "cc": counts, "nrow": NROW}

_SCAT = {"buf": None}
def deliver_full_fn(sh):
    Xk = np.asarray(sh, float)
    one = Xk.ndim == 1
    if one:
        Xk = Xk[None, :]
    Pn = Xk.shape[0]
    b = _SCAT["buf"]
    if b is None or b.shape[0] < Pn:
        b = np.zeros((Pn, NROW), float); _SCAT["buf"] = b
    f = b[:Pn]
    f[:, colmap] = Xk
    d = deliver_full_arr(f, FLOOR)          # a NEW array, like the unfused tab_2 path
    return d[0] if one else d

def gather_fn(fd):
    D = np.asarray(fd, float)
    one = D.ndim == 1
    if one:
        D = D[None, :]
    d = D[:, colmap]
    s = np.repeat(np.add.reduceat(d, prob.profile_start, axis=1), prob.profile_len, axis=1)
    d = np.where(s > 1e-12, d / np.where(s > 1e-12, s, 1.0), d)
    return d[0] if one else d

POISON = {"on": False}
def deliver_rows_fn(sh, rows, scs, scc, rowc):
    r = np.asarray(rows, np.intp)
    ki = CMINV[r]
    S = np.asarray(sh, float)
    A = np.where(ki >= 0, S[np.asarray(rowc, np.intp), np.maximum(ki, 0)], 0.0)[None, :]
    A = _block(A, np.asarray(blk, bool)[r], scs, scc, 0.01)
    A = _elig_sub(A, r, scs, scc, OP)
    A = _cap(A, np.asarray(scs, np.intp), np.asarray(scc, np.intp))
    A = _floor(A, np.asarray(scs, np.intp), np.asarray(scc, np.intp), FLOOR,
               np.asarray(sf_base, bool)[r])
    if POISON["on"]:
        A = np.asarray(A, float).copy()
        A[0, 0] = A[0, 0] * (1.0 + 1e-13) + 1e-18
    return A

_BAND = {"n": 0}
def band_penalty_fn(fd):
    D = np.asarray(fd, float)
    _BAND["n"] += 1
    tot = np.zeros((D.shape[0], NMID))
    for m in range(NMID):
        tot[:, m] = D[:, mid_id == m].sum(axis=1)
    return np.maximum(tot - 0.30 * N / NMID, 0.0).sum(axis=1)

KW = dict(reference_shares=meta["reference_kept"], pop_size=16, generations=14,
          patience=50, n_seeds=1, restarts=2, restart_mode="lean", seed=0,
          log_fn=None, numba=False, band_penalty_fn=band_penalty_fn,
          deliver_full_fn=deliver_full_fn, gather_fn=gather_fn,
          obj_full=meta.get("obj_full"))

def run(delta, poison=False, rows_fn=deliver_rows_fn, dmap=DMAP, logs=None):
    POISON["on"] = poison
    _sp2 = importlib.util.spec_from_file_location("ga_run", GA_P)
    m = importlib.util.module_from_spec(_sp2); sys.modules["ga_run"] = m; _sp2.loader.exec_module(m)
    # 19kd: through the module constant - ROUTING_DELIV_DELTA is deleted. A fresh module per
    # call, so the flag cannot leak between arms. The OFF arm is KEPT deliberately: it is the
    # reference every bit-identity check below diffs against.
    m._DELIV_DELTA_ON = bool(delta)
    _SCAT["buf"] = None
    kw = dict(KW)
    if logs is not None:
        kw["log_fn"] = logs.append
    best, info = m.run_fullmatrix_ga(prob, deliver_rows_fn=rows_fn, deliver_map=dmap, **kw)
    POISON["on"] = False
    return m, np.asarray(best, float), info

_m0, best_off, info_off = run(False)
_l1 = []
_m1, best_on, info_on = run(True, logs=_l1)
D1 = getattr(_m1, "_LAST_DELIV_DELTA", {})

check("2  the deliver delta ENGAGED (it is not silently falling back to the full delivery)",
      int(D1.get("gathered", 0)) > 0,
      f"gathered {D1.get('gathered')}, full {D1.get('full')}, "
      f"re-delivered {D1.get('prof_re')}/{D1.get('prof_tot')} profile-deliveries")
check("2  ...and it stayed on for the whole run", bool(D1.get("on")), str(D1.get("why", "")))
check("2  the SPLIT is bit-identical with the delta armed",
      np.array_equal(bits(best_off), bits(best_on)),
      f"max|d| {float(np.abs(best_off - best_on).max()):.3e}")
for _k in ("success_rate", "violation", "band_breach", "generations", "evaluated"):
    if _k in info_off and _k in info_on:
        _a, _b = info_off[_k], info_on[_k]
        check(f"2  info['{_k}'] is unchanged",
              (np.array_equal(bits(_a), bits(_b)) if isinstance(_a, (float, np.floating))
               else _a == _b), f"{_a} vs {_b}")
check("2  the whole history matches generation for generation",
      len(info_off.get("history", [])) == len(info_on.get("history", []))
      and all(np.array_equal(bits(np.asarray(x, float)), bits(np.asarray(y, float)))
              for x, y in zip(info_off.get("history", []), info_on.get("history", []))))
check("2  the self-check ran on every gathering generation, not once",
      int(D1.get("checks", 0)) >= 2, f"{D1.get('checks')} generation(s) checked")
check("2  [deliv-delta] reported itself in the run log",
      any("[deliv-delta]" in str(x) for x in _l1))

# ═══ 3. THE GUARD: a corrupted subset delivery is caught ══════════════════════════════════
_l2 = []
_m2, best_poison, info_poison = run(True, poison=True, logs=_l2)
D2 = getattr(_m2, "_LAST_DELIV_DELTA", {})
check("3  a poisoned subset delivery is CAUGHT by the per-generation self-check",
      not bool(D2.get("on")) and bool(D2.get("why")), str(D2.get("why", "(not caught)")))
check("3  ...it says so in the run log, loudly",
      any("SELF-CHECK FAILED" in str(x) and "[deliv-delta]" in str(x) for x in _l2))
check("3  ...and the run still finishes on the CORRECT split",
      np.array_equal(bits(best_off), bits(best_poison)),
      f"max|d| {float(np.abs(best_off - best_poison).max()):.3e}")

# a caller that wires nothing must be a no-op, not an error
_l3 = []
_m3, best_none, _ = run(True, rows_fn=None, dmap=None, logs=_l3)
check("3  armed with nothing wired says so and changes no answer",
      np.array_equal(bits(best_off), bits(best_none))
      and any("[deliv-delta] NOT USED" in str(x) for x in _l3))

# ═══ 4. the wiring, at source level ══════════════════════════════════════════════════════
# 19kd: REVERSED, on Ben's instruction. It WAS default-off, and that cost the 2026-09-04
# 15:25 run 193s against the 11:47 run for the same 11,840 candidates and the same shipped
# answer - 22 split(s)/s against 75 - with no line in the log to say why. There is no env var
# now; a source-level constant that only this test flips is not a switch anyone can forget.
check("4  the delta is ON by default, with no env var anywhere",
      "_DELIV_DELTA_ON = True" in GA_SRC
      and 'os.environ.get("ROUTING_DELIV_DELTA"' not in GA_SRC
      and 'os.environ.get("ROUTING_EVAL_DELTA"' not in GA_SRC)
check("4  ...and it is a NAME, not an inlined literal, so the OFF reference survives",
      'bool(_DELIV_DELTA_ON and _have_full' in GA_SRC)
# COMMENT-STRIPPED, because 19kd's own comment QUOTES the `elif _DLV["why"]:` it deleted - a
# raw substring test passes or fails on the strength of its own explanation. That trap has bitten
# five checks across 19jz, 19ka, 19kb and 19kc.
_GA_CODE = "\n".join(l for l in GA_SRC.splitlines() if not l.lstrip().startswith("#"))
check("4  ...and it reports its state EVERY run, armed or not",
      "[deliv-delta] NOT USED" in _GA_CODE
      and 'elif _DLV["why"]:' not in _GA_CODE,
      "the old `elif _DLV[why]` printed nothing at all when the switch defaulted off")
check("4  it rides on the decode delta's provenance, so it can never be armed alone",
      'if _DLV["on"] and _DLT["on"] and _prov is not None:' in GA_SRC)
check("4  the bootstrap delivery is COPIED - `_deliver_full` hands back a reused buffer",
      '_DLV["last"] = np.array(_fd, float, copy=True)' in GA_SRC)
check("4  the cache follows the SAME elite permutation as the population",
      '_DLV["arr"] = (None if (_dlv_new is None or _DLV["arr"] is None)\n'
      '                                   else np.vstack([_DLV["arr"][_el_idx], _dlv_new]))' in GA_SRC)
check("4  ...and a restart starts a fresh one",
      '_DLV["arr"] = _DLV["last"] if _DLV["on"] else None' in GA_SRC)
check("4  tab_2 VERIFIES the profile map instead of assuming it",
      "[dlv-map]" in T2 and "np.array_equal(np.sort(_dm_lo)," in T2
      and "_fm_dlv_map = None" in T2)
check("4  ...and passes None for both hooks when it does not hold",
      "if _fm_dlv_map is None:\n                                        _fm_deliv_rows = None" in T2)
check("4  the subset transform calls the SHIPPED cap and floor, not copies of them",
      "_X = _fm_cap(_X, _cs=_cs2, _cc=_cc2," in T2 and "_X = _fm_floor(_X, _cs=_cs2, _cc=_cc2," in T2)
check("4  eligibility carries has_ban/has_w/has_u over verbatim",
      '"has_ban": _op.get("has_ban"),' in T2)
check("4  blocked-caps on the subset follows the LIVE full-width path",
      'if not _st["use"]:\n                                            return _fm_block_full(' in T2)
check("4  both files record 19iz",
      "19iz-deliv-delta" in GA_SRC and "19iz" in T2)
check("4  the 19ix note no longer says this cannot be done",
      "cannot be asked for a subset of profiles" not in GA_SRC)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
