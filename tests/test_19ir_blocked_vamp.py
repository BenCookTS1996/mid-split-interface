"""19ir - the blocked-row rule at the VAMP forecast's water-fill (site 3 of 5).

`_max_share_waterfill` is a THIRD implementation of the 0.97 cap, and it cannot express a
gatewayFid: its row identity is (BIN, vampMid, Currency). So it reads the canonical key that
19ij proved is an identity for the fid within a brand-scoped run. If this site and the two
fine-grained ones ever disagree about which rows are blocked, a blocked gateway is held at the
floor by one stage and lifted off it by another - and the difference lands in reconciliation
error, which is the failure class this series exists to close.

The rule is still REFUSED (3 of 5 wired), so nothing delivered changes yet. What has to be
proven is exactly that, plus that the plumbing carries the keys and the cache key moves with
them.
"""
import inspect, io, os, pathlib, sys
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
import impact_calcs as ic                                    # noqa: E402

CAP = 0.97
EPS = 1e-12

# ── 1. registration and the refusal ─────────────────────────────────────────────────────
check("site 3 registers at import, so the arming gate counts it",
      "_max_share_waterfill" in bf.wired(), str(bf.wired()))
_armed, _msg = bf.arming_verdict(True)
check("the rule is still refused, and names what is left",
      _armed is False and "band_kernel" in _msg, _msg[:100])
check("...and the two band kernels are what remain",
      set(bf.missing()) == {"_fm_cap", "band_kernel_profile", "band_kernel_flat"},
      str(bf.missing()))

# ── 2. the water-fill itself: bit-identical unarmed, correct armed ──────────────────────
# Three profiles x three rows. grp is the profile key; `live` is the kernel's psum > 0.
t0 = pd.DataFrame({
    "Currency": ["USD"] * 9,
    "BIN": ["111111"] * 3 + ["222222"] * 3 + ["333333"] * 3,
    "RPGT": ["r"] * 9,
    "_pmp": ["non_gp_ap"] * 9,
    "_ctry": ["_all_"] * 9,
    "period": [5] * 9,
    "vampMid": (["Adyen_TAV", "Braintree_TAV", "WoodForest_TAV"] * 3),
})
grp = ["Currency", "BIN", "RPGT", "_pmp", "_ctry", "period"]
#   111111: over the cap, a roomy NON-blocked sibling -> avoidable
#   222222: over the cap, the only under-cap sibling is blocked -> the exception
#   333333: nothing over the cap -> untouched
sh = np.array([0.99, 0.005, 0.005,
               0.99, 0.000, 0.010,
               0.50, 0.30, 0.20])
live = np.ones(9, bool)
blk = np.array([False, False, True, False, False, True, False, False, True])

plain = ic._max_share_waterfill(sh.copy(), t0, grp, CAP, live)
priced = ic._max_share_waterfill(sh.copy(), t0, grp, CAP, live, blocked=blk)
check("with the rule refused the water-fill is BIT-IDENTICAL to the unmasked call",
      np.array_equal(plain.view(np.int64), priced.view(np.int64)),
      f"max|d| {float(np.abs(plain - priced).max()):.3e}")
_v = ic._LAST_BLK_FILL_VAMP
check("...and it was PRICED anyway, which is what makes 'the rule moves nothing' a measurement",
      _v is not None and _v["rows"] == 3 and _v["sweeps"] >= 1,
      f"rows={_v['rows']}, sweeps={_v['sweeps']}, on_blocked={_v['on_blocked']:.6g}, "
      f"unavoidable={_v['unavoidable']:.6g}, avoidable={_v['avoidable']:.6g}")
check("the unmodified rule lifts blocked rows off the floor here",
      priced[2] > sh[2] + 1e-9, f"row 2: {sh[2]:.4f} -> {priced[2]:.6f}")
check("some of that was avoidable (111111 had a roomy non-blocked sibling)",
      _v["avoidable"] > 1e-9, f"{_v['avoidable']:.6g}")
check("and some was NOT (222222's only under-cap sibling is the blocked one)",
      _v["unavoidable"] > 1e-9, f"{_v['unavoidable']:.6g}")

# armed, with every site force-registered, the rule must actually apply
_saved = set(bf._WIRED)
try:
    for _s in bf.SITES:
        bf.register(_s)
    os.environ["ROUTING_BLOCK_NOFILL"] = "1"
    ruled = ic._max_share_waterfill(sh.copy(), t0, grp, CAP, live, blocked=blk)
finally:
    os.environ.pop("ROUTING_BLOCK_NOFILL", None)
    bf._WIRED.clear(); bf._WIRED.update(_saved)
check("ARMED: the blocked row in a profile WITH a roomy sibling stays where it was",
      abs(ruled[2] - sh[2]) < 1e-15, f"{ruled[2]:.9f} vs {sh[2]:.9f} (unarmed {priced[2]:.9f})")
check("ARMED: the roomy sibling absorbed it instead",
      ruled[1] > priced[1] + 1e-9, f"{priced[1]:.6f} -> {ruled[1]:.6f}")
check("ARMED: the exception still fires where the cap could not otherwise hold",
      ruled[5] > sh[5] + 1e-9, f"row 5: {sh[5]:.4f} -> {ruled[5]:.6f}")
check("ARMED: the over-cap rows are still brought to the cap",
      abs(ruled[0] - CAP) < 1e-9 and abs(ruled[3] - CAP) < 1e-9,
      f"{ruled[0]:.6f}, {ruled[3]:.6f}")
check("ARMED: share is conserved per profile",
      max(abs(ruled[0:3].sum() - sh[0:3].sum()),
          abs(ruled[3:6].sum() - sh[3:6].sum()),
          abs(ruled[6:9].sum() - sh[6:9].sum())) < 1e-12,
      f"{ruled[0:3].sum():.9f} / {ruled[3:6].sum():.9f} / {ruled[6:9].sum():.9f}")
check("ARMED: the profile with nothing over the cap is untouched, bit for bit",
      np.array_equal(ruled[6:9].view(np.int64), sh[6:9].view(np.int64)))

# ── 3. the canonical key, and the memo ──────────────────────────────────────────────────
_mid = ROOT / "data/mappings/Master_MID_List.csv"
if _mid.exists():
    _keys = ic.blocked_keys_for({("111111", "woodforest-usd-tav")}, str(_mid))
    check("(bin, gatewayFid) is canonicalised to (bin, vampMid, currency)",
          len(_keys) == 1 and len(next(iter(_keys))) == 3, str(sorted(_keys)))
    check("...and the fid's own vampMid/currency is what came back",
          next(iter(_keys))[0] == "111111" and "woodforest" in next(iter(_keys))[1],
          str(sorted(_keys)))
    check("the memo returns the SAME object on a second call (one mid-list read for three sites)",
          ic.blocked_keys_for({("111111", "woodforest-usd-tav")}, str(_mid)) is _keys)
else:
    check("Master_MID_List.csv present for the canonical-key check", False, "file missing")
check("an unknown fid canonicalises to nothing rather than guessing",
      ic.blocked_keys_for({("111111", "not-a-real-fid")}, str(_mid)) == frozenset())
check("no pairs -> no keys, so the site records NO MASK rather than an empty claim",
      ic.blocked_keys_for(set(), str(_mid)) == frozenset())

# ── 4. the plumbing: every projection site carries the keys, and the cache key moves ────
_sig_a = ic.projection_cache_sig("nope", [("a", 1.0)], 0.0)
_sig_b = ic.projection_cache_sig("nope", [("a", 1.0)], 0.0,
                                 blocked_keys=frozenset({("1", "v", "usd")}))
check("the blocked keys are part of the projection cache key",
      _sig_a != _sig_b, f"{_sig_a[-12:]} vs {_sig_b[-12:]}")
check("...and the same keys give the same key (stable ordering)",
      ic.projection_cache_sig("nope", [("a", 1.0)], 0.0,
                              blocked_keys=frozenset({("1", "v", "usd")})) == _sig_b)

for _fn in (ic.compute_vamp_prepost_granular, ic._c_prepost_granular, ic.projection_cache_sig):
    check(f"{_fn.__name__} takes blocked_keys",
          "blocked_keys" in inspect.signature(_fn).parameters)

T3 = (ROOT / "app/tab_3_split_outputs_impact.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
# each of the three sites passes the keys TWICE - once to the projection and once to its own
# projection_cache_sig, because a key that is not in the signature can be served a frame
# computed at the other setting.
_t3n = {n: T3.count(f"blocked_keys={n}") for n in ("_blk_r", "_blk3", "_blk0")}
check("tab 3 passes the keys at all THREE of its projections, and into all three cache keys",
      all(v == 2 for v in _t3n.values()) and T3.count("blocked_keys=") == 6,
      f"{_t3n}, {T3.count('blocked_keys=')} mentions")
check("...from ONE helper reading the pairs tab 2 recorded",
      "def _blk_keys_t3()" in T3 and 'ss.get("blocked_pairs")' in T3)
check("tab 2's delivery projection passes them too",
      "blocked_keys=_pj_blk" in T2 and "_blk_keys_for(_blk_pairs_pre or set()" in T2)
check("both delivered water-fills report their pricing in the run log",
      "[blk-fill] delivered water-fill" in T2 and "[blk-fill] VAMP forecast water-fill" in T2)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
