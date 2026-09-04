"""19iq - the blocked-row rule at the water-fill that SHIPS, and the flag that gets it there.

Threading `blocked_pairs` into five water-fills means five call-site chains and five chances to
forget one; build_split_exports alone has five callers across tab_3 and tab_4. So the flag rides
on the ROW instead: `_apply_blocked_caps` stamps `_blocked` on the split it floors, and every
frame derived from that split carries it. A water-fill with no column has no information and
applies no rule - which is a different statement from "nothing is blocked", and is treated as
such.

The rule itself stays REFUSED until all five sites are wired (blocked_fill.arming_verdict), so
this commit changes no delivered number. What it must prove is that it changes none: the export
is asserted bit-identical with the column present, absent, and with the rule requested.
"""
import importlib.util, os, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

from routing_optimiser.s4_search import blocked_fill as bf   # noqa: E402
import app_common as ac                                       # noqa: E402
import impact_calcs as ic                                     # noqa: E402

CAP = 0.97
FLOOR = 0.01

# ── the split: one profile whose leader is over the cap, with a blocked sibling ──────────
# BIN 111111: leader over the cap, a NON-blocked sibling with room for the whole excess, and
#             a blocked row below the floor. Nothing forces the blocked row to receive, so the
#             excess here is entirely AVOIDABLE - the rule would move it.
#             (The shares must breach the cap once the export normalises the profile to 1, which
#             is why the two small rows are 0.005 and not 0.39/0.01.)
# BIN 222222: leader 0.99 and the ONLY under-cap sibling is blocked -> the exception.
rows = [
    ("rpgt-a", "usd", "111111", "adyen-usd-tav",      0.990),
    ("rpgt-a", "usd", "111111", "braintree-usd-tav",  0.005),
    ("rpgt-a", "usd", "111111", "woodforest-usd-tav", 0.005),
    ("rpgt-a", "usd", "222222", "adyen-usd-tav",      0.99),
    ("rpgt-a", "usd", "222222", "woodforest-usd-tav", 0.01),
]
split = pd.DataFrame(rows, columns=["rpgt", "currency", "bin", "gateway", "share"])
blocked_pairs = {("111111", "woodforest-usd-tav"), ("222222", "woodforest-usd-tav")}

# ── 1. the flag is stamped, and only on the blocked rows ────────────────────────────────
out, ncap = ac._apply_blocked_caps(split, blocked_pairs, FLOOR)
check("_apply_blocked_caps stamps a _blocked column", "_blocked" in out.columns)
check("...True on exactly the blocked rows",
      out["_blocked"].tolist() == [False, False, True, False, True],
      str(out["_blocked"].tolist()))
check("...and no share is created or destroyed by the flooring pass",
      abs(float(out["share"].sum()) - float(split["share"].sum())) < 1e-12,
      f"{float(out['share'].sum()):.9f} vs {float(split['share'].sum()):.9f}")

# pairs supplied that match NOTHING is a fact, not an absence
none_match, _ = ac._apply_blocked_caps(split, {("999999", "nope")}, FLOOR)
check("pairs that match nothing still stamp the column, all-False",
      "_blocked" in none_match.columns and not none_match["_blocked"].any())

# no pairs at all -> no column, because there is no information to record
no_pairs, _ = ac._apply_blocked_caps(split, set(), FLOOR)
check("no blocked pairs leaves NO column (absence of information, not a False claim)",
      "_blocked" not in no_pairs.columns)

# ── 2. the export is BIT-IDENTICAL with the column, without it, and with the rule asked for
# 19kg: the settings this used to set in the environment are module constants on the module
# that reads them, so `**sw` rebinds them there and restores them exactly as the env dance did.
def _export(_df, **sw):
    _old = {k: getattr(ic, k) for k in sw}
    for k, v in sw.items():
        setattr(ic, k, v)
    try:
        _ex = ic.build_split_exports(_df, "TotalAV", "2026-01-01", max_share=CAP)
        return {k: v.copy() for k, v in _ex.items()}
    finally:
        for k, v in _old.items():
            setattr(ic, k, v)

def _numbers(ex):
    _o = []
    for k in sorted(ex, key=str):
        _d = ex[k]
        _o.append(_d.select_dtypes(include=[float, int]).to_numpy(float))
    return _o

base = _export(split)
withcol = _export(out)
asked = _export(out, _SW_BLOCK_NOFILL=True)
_b, _w, _a = _numbers(base), _numbers(withcol), _numbers(asked)
check("the export is bit-identical whether or not the split carries _blocked",
      len(_b) == len(_w) and all(np.array_equal(np.nan_to_num(x).view(np.int64),
                                                np.nan_to_num(y).view(np.int64))
                                 for x, y in zip(_b, _w)))
check("...and bit-identical again with the rule REQUESTED, because it is refused until 5/5",
      len(_b) == len(_a) and all(np.array_equal(np.nan_to_num(x).view(np.int64),
                                                np.nan_to_num(y).view(np.int64))
                                 for x, y in zip(_b, _a)))
_v = ic._LAST_BLK_FILL
check("the refusal is recorded with the reason, not silently dropped",
      _v is not None and _v["armed"] is False and "not wired for it" in _v["msg"],
      str(_v and _v["msg"])[:110])
check("_cap_rows is registered as wired, so the verdict counts it",
      "_cap_rows" in bf.wired())
check("...and the sites still missing are named",
      set(bf.missing()) <= {"_fm_cap", "_max_share_waterfill", "band_kernel_profile",
                            "band_kernel_flat"} and bf.missing(), str(bf.missing()))

# ── 3. the PRICING ran on the shipping path, which is the point of this commit ───────────
check("the rule was priced on the delivered water-fill",
      _v["pairs"] == 2 and _v["sweeps"] >= 1,
      f"pairs={_v['pairs']}, sweeps={_v['sweeps']}, on_blocked={_v['on_blocked']:.6g}, "
      f"unavoidable={_v['unavoidable']:.6g}, avoidable={_v['avoidable']:.6g}")
check("the unmodified water-fill really does lift blocked rows (so there IS something to fix)",
      _v["on_blocked"] > 1e-9, f"{_v['on_blocked']:.6g} of share went onto blocked rows")
check("...and some of it was avoidable - a non-blocked sibling had room and did not get it",
      _v["avoidable"] > 1e-9, f"{_v['avoidable']:.6g}")
check("the mask fact is recorded for this site",
      bf._MASK_FACT.get("_cap_rows", (False, ""))[0] is True,
      str(bf._MASK_FACT.get("_cap_rows")))

# ── 4. and the rule, applied directly, does what the pricing says it would ──────────────
# _cap_rows' own layout: one row per profile, one column per gateway.
V = np.array([[0.99, 0.00, 0.01],       # blocked col 2, roomy col 1 absent -> exception
              [0.60, 0.39, 0.01]])
V[0] = [0.99, 0.00, 0.01]
blk = np.array([False, False, True])
room = np.where((V < CAP) & (V > 1e-12), CAP - V, 0.0)
exc = np.where(V > CAP, V - CAP, 0.0).sum(1, keepdims=True)
fb = np.where(room.sum(1, keepdims=True) > 1e-12,
              room / np.where(room.sum(1, keepdims=True) > 1e-12,
                              room.sum(1, keepdims=True), 1.0) * exc, 0.0)
add = bf.two_stage_add_rowwise(room, blk, exc, fb)
check("row 0 (only under-cap sibling is blocked): the exception fires, the blocked row receives",
      add[0, 2] > 1e-9, f"{add[0, 2]:.6g}")
check("row 1 (nothing over cap): no excess, so no add anywhere",
      float(np.abs(add[1]).max()) < 1e-15)
# a roomy NON-blocked sibling exists, and the profile really does breach (sums to 1 already,
# so there is a genuine 0.02 of excess to place - the previous fixture normalised the breach
# away and the mass check below was passing on 0.000000 of 0.000000)
V2 = np.array([[0.99, 0.005, 0.005]])
room2 = np.where((V2 < CAP) & (V2 > 1e-12), CAP - V2, 0.0)
exc2 = np.where(V2 > CAP, V2 - CAP, 0.0).sum(1, keepdims=True)
fb2 = np.where(room2.sum(1, keepdims=True) > 1e-12,
               room2 / np.where(room2.sum(1, keepdims=True) > 1e-12,
                                room2.sum(1, keepdims=True), 1.0) * exc2, 0.0)
add2 = bf.two_stage_add_rowwise(room2, np.array([False, False, True]), exc2, fb2)
check("with a roomy non-blocked sibling the blocked row gets NOTHING",
      abs(float(add2[0, 2])) < 1e-15, f"{float(add2[0, 2]):.3e}")
check("...and the whole excess still lands (mass conserved)",
      abs(float(add2.sum()) - float(exc2.sum())) < 1e-12,
      f"placed {float(add2.sum()):.9f} of {float(exc2.sum()):.9f}")

# ── 5. the residual bug the refactor fixed: excess > all room must never lose mass ───────
# The blocked row sits at ZERO share, so it is not a recipient and holds no room; the
# non-blocked room (0.28) is SMALLER than the excess (0.68). The pre-refactor split capped the
# primary stage at the non-blocked room and handed the remainder to a fallback pool of zero,
# which DROPPED it. `_factors` returns split=False for exactly this profile, so the caller's own
# formula runs and places the whole excess (overshooting, which the next sweep re-sheds - the
# unmodified behaviour).
st = np.array([0], np.intp); ct = np.array([3], np.intp)
X = np.array([[0.98, 0.02, 0.00]])
CAP2 = 0.30
blk5 = np.array([False, False, True])
got = bf.waterfill_once(X, CAP2, blk5, st, ct)
check("a profile whose blocked row holds NO room does not lose the residual",
      abs(float(got.sum()) - float(X.sum())) < 1e-12,
      f"{float(got.sum()):.9f} vs {float(X.sum()):.9f}")
# and it is byte-identical to the SAME water-fill with nothing blocked, which is the safety
# property that makes the rule wirable into five sites: it may only change profiles it reaches.
# (The reference has to be this function with blocked=all-False, not a hand-written formula -
# `room * (excess/pool)` and `(room/pool) * excess` differ in the last bit, and asserting
# bit-identity against the wrong association measures my own transcription, not the rule.)
_none5 = bf.waterfill_once(X, CAP2, np.zeros(3, bool), st, ct)
check("...and is byte-identical to the same water-fill with nothing blocked",
      np.array_equal(got.view(np.int64), _none5.view(np.int64)),
      f"max|d| {float(np.abs(got - _none5).max()):.3e}")
_room5 = np.where((~(X > CAP2)) & (X > 1e-12) & (X < CAP2), CAP2 - X, 0.0)
_exc5 = np.add.reduceat(np.where(X > CAP2, X - CAP2, 0.0), st, axis=1)
# the guard is what makes that true: without it the fallback stage would have been handed
# max(excess - non_blocked_room, 0) with nowhere to put it.
_, _, _e_p, _e_f = bf.split_room(_room5, blk5, _exc5, st, ct)
check("...and the residual the naive split would have dropped is named",
      float(_e_f.sum()) > 1e-9,
      f"{float(_e_f.sum()):.6g} of share had no fallback pool to land in")

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
