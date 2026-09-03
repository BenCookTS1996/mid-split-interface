"""19ii — the blocked-row water-fill rule and the arithmetic that prices it.

THE RULE (Ben, 2026-09-03): a bank-blocked gateway stays at the exploration floor, unless the
profile needs it to keep its other gateways under the max-share cap.

The safety property that makes it landable: with no blocked row in a profile, the rule must be
BYTE-IDENTICAL to the unmodified water-fill. Anything else is a change to profiles nobody asked
to change.
"""
import importlib.util, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_sp = importlib.util.spec_from_file_location(
    "blocked_fill", str(ROOT / "src/routing_optimiser/s4_search/blocked_fill.py"))
bf = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(bf)

FAIL = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok: FAIL.append(name)

CAP = 0.97

def plain_waterfill(X, cap, starts, counts):
    """The UNMODIFIED rule, as every caller implements it today: recipients on share alone."""
    X = np.asarray(X, float).copy()
    o = X > cap
    if not o.any():
        return X
    exc = np.add.reduceat(np.where(o, X - cap, 0.0), starts, axis=1)
    room = np.where((~o) & (X > 1e-12) & (X < cap), cap - X, 0.0)
    pool = np.add.reduceat(room, starts, axis=1)
    f = np.repeat(np.where(pool > 1e-12, exc / np.where(pool > 1e-12, pool, 1.0), 0.0),
                  counts, axis=1)
    return np.where(o, cap, X) + room * f

# ── a book with three profiles of four gateways ─────────────────────────────────────────
starts = np.array([0, 4, 8], np.intp)
counts = np.array([4, 4, 4], np.intp)

# profile 0: one row over cap, TWO roomy non-blocked siblings, one blocked row at the floor
#            -> the blocked row must get NOTHING; the siblings have plenty of room.
# profile 1: one row over cap, the ONLY under-cap sibling is blocked
#            -> the blocked row MUST receive, or the cap cannot hold. The exception.
# profile 2: nothing over cap -> untouched.
X = np.array([[1.20, 0.30, 0.40, 0.010,
               1.30, 0.960, 0.960, 0.010,
               0.25, 0.25, 0.25, 0.25]])
blk = np.array([False, False, False, True,
                False, True, True, True,
                False, False, False, True])

got = bf.waterfill_once(X, CAP, blk, starts, counts)
plain = plain_waterfill(X, CAP, starts, counts)

check("mass is conserved per profile",
      np.allclose(np.add.reduceat(got, starts, axis=1),
                  np.add.reduceat(X, starts, axis=1), atol=1e-12),
      f"max drift {float(np.abs(np.add.reduceat(got - X, starts, axis=1)).max()):.2e}")
check("profile 0: the blocked row stays exactly at the floor",
      got[0, 3] == X[0, 3],
      f"{got[0, 3]:.6f} (the unmodified rule lifts it to {plain[0, 3]:.6f})")
check("profile 0: and the unmodified rule really does lift it (so this is a real difference)",
      plain[0, 3] > X[0, 3] + 1e-9)
check("profile 0: the excess went to the non-blocked siblings instead",
      abs((got[0, 1] + got[0, 2]) - (X[0, 1] + X[0, 2] + (X[0, 0] - CAP))) < 1e-12)
check("profile 1: the blocked rows DO receive — the exception fires",
      got[0, 5] > X[0, 5] + 1e-9 and got[0, 6] > X[0, 6] + 1e-9)
check("profile 1: and the over-cap row is brought to the cap",
      abs(got[0, 4] - CAP) < 1e-12)
check("profile 2: a profile with nothing over cap is untouched, bit for bit",
      np.array_equal(got[0, 8:].view(np.int64), X[0, 8:].view(np.int64)))

# ── THE SAFETY PROPERTY: no blocked row ⇒ byte-identical to the unmodified rule ──────────
rng = np.random.default_rng(19)
for _t in range(200):
    P, nprof, per = 3, 5, 6
    st = np.arange(0, nprof * per, per, dtype=np.intp)
    ct = np.full(nprof, per, np.intp)
    Y = rng.random((P, nprof * per))
    Y = Y / np.repeat(np.add.reduceat(Y, st, axis=1), ct, axis=1)
    if _t % 3 == 0:                       # force some rows over the cap
        Y[:, ::per] = 0.99
        Y = Y / np.repeat(np.add.reduceat(Y, st, axis=1), ct, axis=1)
    none_blocked = np.zeros(Y.shape[1], bool)
    a = bf.waterfill_once(Y, CAP, none_blocked, st, ct)
    b = plain_waterfill(Y, CAP, st, ct)
    if not np.array_equal(a.view(np.int64), b.view(np.int64)):
        check("SAFETY: with no blocked row the rule is byte-identical to the unmodified one",
              False, f"diverged on trial {_t}, max|d| {float(np.abs(a - b).max()):.3e}")
        break
else:
    check("SAFETY: with no blocked row the rule is byte-identical to the unmodified one "
          "(200 random books)", True)

# ── the rule may only ever change profiles containing a blocked row WITH ROOM ────────────
bad = 0
for _t in range(200):
    P, nprof, per = 2, 4, 5
    st = np.arange(0, nprof * per, per, dtype=np.intp)
    ct = np.full(nprof, per, np.intp)
    Y = rng.random((P, nprof * per)); Y[:, ::per] = 3.0
    Y = Y / np.repeat(np.add.reduceat(Y, st, axis=1), ct, axis=1)
    bl = rng.random(Y.shape[1]) < 0.25
    a = bf.waterfill_once(Y, CAP, bl, st, ct)
    b = plain_waterfill(Y, CAP, st, ct)
    moved = np.abs(a - b) > 1e-12
    prof_moved = np.add.reduceat(moved.astype(float), st, axis=1) > 0
    prof_has_blk = np.add.reduceat(np.repeat(bl[None, :], P, 0).astype(float), st, axis=1) > 0
    if bool((prof_moved & ~prof_has_blk).any()):
        bad += 1
check("the rule never touches a profile that has no blocked row", bad == 0,
      f"{bad} of 200 random books changed an all-unblocked profile")

# ── unavoidable_excess: the pricing arithmetic ──────────────────────────────────────────
exc = np.array([[0.23, 0.33, 0.0]])
room = np.where((~(X > CAP)) & (X > 1e-12) & (X < CAP), CAP - X, 0.0)
need = bf.unavoidable_excess(exc, room, blk, starts)
check("profile 0's excess is fully absorbable without a blocked row ⇒ need 0",
      abs(float(need[0, 0])) < 1e-12, f"{float(need[0, 0]):.6g}")
check("profile 1's excess is NOT ⇒ need > 0 and never exceeds the excess",
      float(need[0, 1]) > 1e-9 and float(need[0, 1]) <= float(exc[0, 1]) + 1e-12,
      f"need {float(need[0, 1]):.6g} of excess {float(exc[0, 1]):.6g}")
check("a profile with no excess needs nothing", abs(float(need[0, 2])) < 1e-12)
check("with nothing blocked, nothing is ever unavoidable",
      float(np.abs(bf.unavoidable_excess(exc, room, np.zeros_like(blk), starts)).max()) < 1e-12)

# ── the measurement wiring is present in the shipped _fm_cap ────────────────────────────
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
check("[blk-fill] measures inside _fm_cap and reports beside [deliv-cap]",
      "[blk-fill]" in T2 and '_st["bf_need"]' in T2 and "THE AVOIDABLE PART IS" in T2)
check("[blk-fill] is read-only — it never assigns to the returned array",
      "_Y = " not in T2.split('19ii [blk-fill]: WHAT THIS WATER-FILL')[1].split('_st["hit"] += 1')[0])

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
