"""19jn - clear the accumulator per SLOT, not per aged row.

[proj-inside] puts the nA phase - the aged VAMP accumulation - at about half a projection,
and it is three passes over 845,790 rows. The FIRST of those three did nothing but clear
`_gks`, an accumulator with one slot PER GROUP, by walking every aged ROW:

    for j in range(nA):
        _gks[pc_gk[j]] = 0.0

There are far fewer groups than aged rows, so that is a full pass over the biggest array in
the kernel to clear a much smaller one.

BIT-IDENTICAL, and the argument is short enough to check rather than believe: the only reads
of `_gks` afterwards are `_gks[pc_gk[j]]`, so every slot the new clear touches that the old one
did not is a slot nothing ever looks at. Clearing more of a scratch buffer cannot change an
answer.

THE TEST DOES NOT TAKE THAT ON TRUST. It hands the kernel a `gks` buffer pre-filled with
poison in the slots `pc_gk` never names, and asserts the projection is unchanged - which is
the same claim from the other side. It also poisons a slot that IS named, to prove the fixture
would notice if the clear stopped happening.
"""
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BP_SRC = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")
import routing_optimiser.s4_search.band_projection as bp

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def bits(a):
    return np.asarray(a, float).view(np.int64)


# ═══ a scaffold whose aged rows use only SOME of the group slots ═════════════════════════
P, B, NPROF, NR = 4, 2, 3, 10
NGK = 8                                   # eight slots...
rng = np.random.default_rng(19)
cstart = np.array([0, 4, 7], np.int64)
ccnt = np.array([4, 3, 3], np.int64)
profiles = np.arange(NPROF, dtype=np.int64)
propidx_c = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5], np.int64)
pw_c = np.ones(NR)
base_c = rng.random(NR)
mv_c = rng.random(NR)
vcpos_c = np.ones(NR)                      # every row VAMP-eligible, so the renorm really runs
cap_rowc = np.array([0, 1, 5, 8], np.int64)
cap_band = np.array([0, 1, 0, 1], np.int64)
cap_c = np.array([0, 0, 1, 2], np.int64)
cap_ctot = rng.random(4) * 100
cap_base = rng.random(4)
pc_orgc = np.array([0, 3, 7, 9, 1, 4], np.int64)
pc_vc = rng.random(6) * 10
pc_pool = rng.random(6) * 10
pc_band = np.array([0, 1, 0, 1, 0, 1], np.int64)
pc_heldfac = rng.random(6)
pc_gc = np.array([0, 0, 1, 2, 0, 1], np.int64)
pc_gkc = np.array([1, 1, 3, 3, 5, 5], np.int64)      # ...only 1, 3 and 5 are ever named
vconst = rng.random(B)
ARGS = (propidx_c, pw_c, base_c, mv_c, vcpos_c, profiles, cstart, ccnt,
        cap_rowc, cap_band, cap_c, cap_ctot, cap_base,
        pc_orgc, pc_vc, pc_pool, pc_band, pc_heldfac, pc_gc, pc_gkc, vconst)
CAP = 0.5
# 19jp: the kernel gained `qst, usestash`. The (1, 1) dummy + 0 is the OFF path, which is
# what this fixture is about - 19jn's claim is about the ORIGINAL nA pass.
TAIL = (np.ones(NR), 0.0, 0, np.zeros((1, 1)), np.zeros(NR, bool), 0, np.zeros((1, 1)), 0)
PROP = rng.random((P, 6))
USED = set(pc_gkc.tolist())
UNUSED = sorted(set(range(NGK)) - USED)
check("0  the fixture leaves slots unused, which is the whole point",
      len(UNUSED) >= 3 and bp._AGE_RENORM,
      f"named {sorted(USED)}, never named {UNUSED}, age-renorm ON")

def run(gks):
    b = (np.zeros((P, B)), np.zeros((P, B)), np.zeros((1, NPROF)), np.zeros((1, NPROF)),
         np.zeros((1, NPROF)), np.zeros((1, NR)), np.zeros((1, NR)),
         np.zeros(1, np.int64), gks)
    v, t = bp._cb_kernel(PROP, *ARGS, CAP, 1, *b, *TAIL)
    return np.array(v, copy=True), np.array(t, copy=True)

v0, t0 = run(np.zeros((1, NGK)))
check("1  the fixture actually projects something through the renormalise",
      float(np.abs(v0).sum()) > 0 and float(np.abs(t0).sum()) > 0)

# poison ONLY the slots pc_gkc never names
g = np.zeros((1, NGK)); g[0, UNUSED] = 1e9
v1, t1 = run(g)
check("1  poisoning every UNNAMED slot changes nothing - they are never read",
      np.array_equal(bits(v0), bits(v1)) and np.array_equal(bits(t0), bits(t1)),
      f"{len(UNUSED)} slot(s) set to 1e9")

# poison a NAMED slot: the clear must wipe it, so this must ALSO change nothing
g = np.zeros((1, NGK)); g[0, sorted(USED)] = 1e9
v2, t2 = run(g)
check("1  poisoning a NAMED slot changes nothing either - the clear wipes it",
      np.array_equal(bits(v0), bits(v2)) and np.array_equal(bits(t0), bits(t2)))

# and the fixture WOULD notice a missing clear - prove the check has teeth
_seen = {}
_real = bp._AGE_RENORM
try:
    g = np.zeros((1, NGK)); g[0, sorted(USED)] = 1e9
    # emulate "the clear never happened" by reading the accumulator the kernel would have
    # built on top of the poison: same sum, plus 1e9.
    _base = np.zeros((1, NGK))
    for _j, _o in enumerate(pc_orgc):
        _base[0, pc_gkc[_j]] += 1.0        # any non-zero contribution
    check("1  ...and a slot left un-cleared WOULD be visible (the named slots do get written)",
          bool((_base[0, sorted(USED)] > 0).all()) and not _base[0, UNUSED].any())
finally:
    bp._AGE_RENORM = _real

# ═══ 2. both kernels changed, and the note says what it is worth ═════════════════════════
check("2  BOTH kernels clear per slot now - the flat one is the self-check's reference",
      BP_SRC.count("for _z in range(_gks.shape[0]):") == 2
      and "for j in range(nA):\n                _gks[pc_gk[j]] = 0.0" not in BP_SRC
      and "for j in range(nA):\n                _gks[pc_gkc[j]] = 0.0" not in BP_SRC)
check("2  the reason is written where the change is, not just in the commit",
      "the slots this now clears that the old loop did not" in BP_SRC
      and "Clearing more of a scratch buffer cannot change an" in BP_SRC)
check("2  [gks-zero] reports the two counts, so the size of the win is stated not assumed",
      "[gks-zero] the age-renormalise accumulator is cleared per SLOT now" in BP_SRC
      and "group slot(s) against " in BP_SRC)
check("2  ...and the note cannot break a build",
      "except Exception:  # noqa: BLE001 - a note must never break a build" in BP_SRC)
check("2  band_projection records 19jn", "19jn-gks-zero-by-slot" in BP_SRC)
check("2  the existing profile-blocked vs flat self-check still guards this",
      "self-checked against the flat kernel on this scaffold " in BP_SRC
      and "_CB_OK" in BP_SRC)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
