"""19is - the rule at the search's own water-fill, and at the band projector's numpy reference.

Two sites move here:

  * `_fm_cap` (tab_2) has had the blocked mask since 19ii but only MEASURED with it. It now
    applies the rule when armed, and the build registers the site.
  * `PopulationBandProjector._cap_pshare` - the flat kernel's numpy twin, and the reference
    every self-check diffs against. It has to move WITH the kernels or the check fails by
    construction, which is what 19hv's exploration floor did to the profile-blocked self-check.
    The two numba kernels themselves come in 19it, so `band_kernel_*` stay unregistered and the
    rule stays refused.

The awkward fact this commit deals with: the live fitness projector is built at tab_2:5350 and
the auto-block detection runs at tab_2:6099 - AFTER it. So there is nothing to pass at
construction, and `set_blocked_keys` exists to hand it over in between rather than moving a
detection pass earlier to suit an argument list.
"""
import numpy as np
import pandas as pd
import os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

from routing_optimiser.s4_search import blocked_fill as bf              # noqa: E402
from routing_optimiser.s4_search import band_projection as bp           # noqa: E402

T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
BP = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")

CAP = 0.97

# ── 1. _fm_cap: registered, and applying rather than only measuring ─────────────────────
check("tab_2 registers _fm_cap at import, so the arming gate counts the site",
      "_BFM_T2.register(\"_fm_cap\")" in T2)
check("_fm_cap builds the add through blocked_fill, not a private copy of the rule",
      "_bfm2.two_stage_add(" in T2 and "fallback_add=_add" in T2)
check("...and passes its OWN unmodified expression as the fallback",
      "_add = _room * _f" in T2)
check("the arming verdict is read ONCE, not per call",
      '_DCAP["bf_on"] = bool(_bf_ok and _fm_blk_row is not None)' in T2
      and T2.count("arming_verdict(") == 1)
check("armed and APPLIED stay separate facts",
      '"bf_on": False, "bf_applied": 0' in T2 and '_st["bf_applied"] += 1' in T2)
check("a failure inside the rule disables it rather than breaking the search",
      '_st["bf_on"] = False   # never break the search' in T2)
check("_fm_cap records whether it actually got a mask this run",
      'saw_mask("_fm_cap"' in T2)
check("the stale SCOPE note (\"_cap_rows ... has no mask\") is gone",
      "which applies "
      "the same rule in the same order but has no mask" not in T2
      and "[blk-fill] STATUS: " in T2)

# ── 2. the projector: the mask can arrive AFTER construction ────────────────────────────
check("set_blocked_keys exists and says why",
      "def set_blocked_keys(self, blocked_keys):" in BP
      and "runs LATER in the same function" in BP)
check("...and it invalidates BOTH numba argument caches",
      "self._nbcache = None\n        self._cb = None" in BP)
check("tab_2 calls it at the point the mask first exists",
      "_pbp_live.set_blocked_keys(_pk)" in T2)
check("...and says so in the run log, including the ZERO-match case",
      "[blk-fill] band projector: {_nblk:,} scaffold " in T2
      and "ZERO rows matched" in T2)
check("the two diagnostic projector builds are left without a key on purpose",
      T2.count("blocked_keys=_pbp_blk") == 1)

# ── 3. _cap_pshare applies the rule, from the one definition of it ──────────────────────
check("_cap_pshare uses blocked_fill._factors rather than its own arithmetic",
      "_BFM._factors(self._profilesum(_rp)[:, gc]," in BP)
check("...and keeps the unmodified add verbatim where the rule cannot reach",
      "_add = np.where(_spl, _rp * _fp + _rf * _ff, _add)" in BP)
check("band_projection imports blocked_fill at MODULE level (a build-time capability claim)",
      "from routing_optimiser.s4_search import blocked_fill as _BFM" in BP
      and BP.index("import blocked_fill as _BFM") < BP.index("def _pop_band_kernel_impl"))
check("the two kernels are NOT registered yet, so the rule is still refused",
      set(bf.missing()) >= {"band_kernel_profile", "band_kernel_flat"}, str(bf.missing()))

# ── 4. _cap_pshare, behaviourally, on a real projector ──────────────────────────────────
# Minimal scaffold: one currency, three BINs, three MIDs each, one period.
_rows = []
for _b, _shares in (("111111", (0.99, 0.005, 0.005)),
                    ("222222", (0.99, 0.000, 0.010)),
                    ("333333", (0.50, 0.30, 0.20))):
    for _m, _s in zip(("adyen_tav", "braintree_tav", "woodforest_tav"), _shares):
        _rows.append({"cur": "usd", "bin": _b, "rpgt": "r", "pmp": "non_gp_ap",
                      "ctry": "_all_", "mid": _m, "midl": _m, "per": 0,
                      "vi": 1000.0, "vc": 10.0, "pr": 1.0, "fcp": 1.0, "bf": 0.0,
                      "excl": False, "emask": False, "iscap": True, "_av": 1000.0,
                      "keep": 1.0, "_share": _s})
T0 = pd.DataFrame(_rows)
Pc = pd.DataFrame({"cur": [], "bin": [], "rpgt": [], "pmp": [], "ctry": [], "mid": [],
                   "midl": [], "per": [], "t": [], "vc": []})
proj = bp.PopulationBandProjector(T0, Pc, np.zeros(0), [("adyen_tav", 0)],
                                  max_share=CAP, by_profile=True)
_nR = len(proj._gcode)
check("the projector reduced the scaffold and kept every row",
      _nR == 9, f"nR={_nR}")
_n = proj.set_blocked_keys({("111111", "woodforest_tav", "usd"),
                            ("222222", "woodforest_tav", "usd"),
                            ("333333", "woodforest_tav", "usd")})
check("set_blocked_keys matched the blocked rows by (bin, midl, cur)",
      _n == 3 and list(np.where(proj._pblk)[0]) == [2, 5, 8],
      f"{_n} row(s): {list(np.where(proj._pblk)[0])}")
check("...and an unknown key matches nothing rather than guessing",
      proj.set_blocked_keys({("999999", "nope", "usd")}) == 0)

_sh = np.array([[0.99, 0.005, 0.005, 0.99, 0.000, 0.010, 0.50, 0.30, 0.20]])
# `act` is per ROW, not per profile: project_pop builds it as _profilesum(pr)[:, gcode] > 0.
_act = np.ones((1, _nR), bool)
proj.set_blocked_keys({("111111", "woodforest_tav", "usd"),
                       ("222222", "woodforest_tav", "usd")})
_unarmed = proj._cap_pshare(_sh.copy(), _act)
check("REFUSED: _cap_pshare is bit-identical to the pre-19is water-fill",
      not proj._blk_armed, f"armed={proj._blk_armed}")

_saved = set(bf._WIRED)
try:
    for _s in bf.SITES:
        bf.register(_s)
    os.environ["ROUTING_BLOCK_NOFILL"] = "1"
    proj.set_blocked_keys({("111111", "woodforest_tav", "usd"),
                           ("222222", "woodforest_tav", "usd")})
    check("ARMED: the projector reports the rule as armed", proj._blk_armed)
    _armed = proj._cap_pshare(_sh.copy(), _act)
finally:
    os.environ.pop("ROUTING_BLOCK_NOFILL", None)
    bf._WIRED.clear(); bf._WIRED.update(_saved)

check("ARMED: the blocked row with a roomy sibling is left exactly where it was",
      abs(_armed[0, 2] - _sh[0, 2]) < 1e-15,
      f"{_armed[0, 2]:.9f} vs {_sh[0, 2]:.9f} (refused: {_unarmed[0, 2]:.9f})")
check("...and the refused run really did lift it (so this is a real difference)",
      _unarmed[0, 2] > _sh[0, 2] + 1e-9, f"{_unarmed[0, 2]:.6f}")
check("ARMED: the roomy non-blocked sibling absorbed it instead",
      _armed[0, 1] > _unarmed[0, 1] + 1e-9,
      f"{_unarmed[0, 1]:.6f} -> {_armed[0, 1]:.6f}")
check("ARMED: the exception fires where the cap could not otherwise hold",
      _armed[0, 5] > _sh[0, 5] + 1e-9, f"{_sh[0, 5]:.4f} -> {_armed[0, 5]:.6f}")
check("ARMED: over-cap rows still land on the cap",
      abs(_armed[0, 0] - CAP) < 1e-9 and abs(_armed[0, 3] - CAP) < 1e-9)
check("ARMED: share is conserved per profile",
      max(abs(_armed[0, 0:3].sum() - 1.0), abs(_armed[0, 3:6].sum() - 1.0),
          abs(_armed[0, 6:9].sum() - 1.0)) < 1e-12,
      f"{_armed[0, 0:3].sum():.9f} / {_armed[0, 3:6].sum():.9f} / {_armed[0, 6:9].sum():.9f}")
check("ARMED: the profile with nothing over the cap is untouched, bit for bit",
      np.array_equal(_armed[0, 6:9].view(np.int64), _sh[0, 6:9].view(np.int64)))
check("ARMED: a profile with no blocked row is bit-identical to the refused run",
      np.array_equal(_armed[0, 6:9].view(np.int64), _unarmed[0, 6:9].view(np.int64)))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
