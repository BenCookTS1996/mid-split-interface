"""19it - the blocked-row rule inside both numba kernels (sites 4 and 5 of 5).

WHAT HAD TO BE TRUE, and is checked here rather than argued:

  1. UNARMED, the kernels are byte-for-byte what they were. The two extra per-profile pools are
     accumulated ALONGSIDE the existing `rsum`, not instead of it, so the unarmed branch is the
     pre-19it expression verbatim.
  2. ARMED, each kernel agrees with `_cap_pshare` - the numpy reference every projector
     self-check diffs against - BIT for BIT. That is why the rule uses the same
     factor-then-multiply association as blocked_fill._factors, and why the two pools are
     accumulated in row order: adding 0.0 is exact, so a masked profile sum in numpy equals a
     skip-the-blocked-rows loop in the kernel.
  3. The flat and profile-blocked kernels agree with EACH OTHER armed. 19hv's exploration floor
     had to SKIP that self-check because it only reached one of the two kernels, leaving the
     profile-blocked path unverified on any armed run. This rule reaches both.

With numba installed these run as compiled kernels; without it, `_njit` degrades to a no-op
decorator and the same bodies run as Python - the arithmetic under test is identical either way,
and the test says which mode it ran in.
"""
import os, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# 19kg: ROUTING_PROJ_FLOAT32 was deleted before this commit; float32 is a source-level
# setting now. The identity below needs ONE dtype on both sides, which `_ident` enforces
# directly by comparing dtypes rather than by asking for a dtype in the environment.

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

CAP = 0.97
NBIN = 24
KEYS = frozenset({(f"{100000 + b}", "woodforest_tav", "usd") for b in range(NBIN)})


def _scaffold(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(NBIN):
        for m in ("adyen_tav", "braintree_tav", "woodforest_tav", "paysafe_tav"):
            rows.append({"cur": "usd", "bin": f"{100000 + b}", "rpgt": "r",
                         "pmp": "non_gp_ap", "ctry": "_all_", "mid": m, "midl": m,
                         "per": 0, "vi": float(rng.integers(100, 5000)),
                         "vc": float(rng.integers(0, 50)), "pr": 1.0, "fcp": 1.0,
                         "bf": 0.0, "excl": False, "emask": False, "iscap": True,
                         "_av": 1000.0, "keep": 1.0})
    T0 = pd.DataFrame(rows)
    Pc = pd.DataFrame({"cur": [], "bin": [], "rpgt": [], "pmp": [], "ctry": [], "mid": [],
                       "midl": [], "per": [], "t": [], "vc": []})
    return T0, Pc


def _fresh(env, **flags):
    """Re-import band_projection under `env`, then set any module FLAGS given by keyword.

    Its env switches are read at IMPORT so numba can fold them, so an env-selected path can only
    be chosen by re-importing.

    19jx: ROUTING_PROJ_PROFILEBLOCK was DELETED - the profile-blocked kernel is unconditional -
    so this test selects the flat path the way `lift_ab_report` selects the unlifted one: by
    setting the module global. The flat kernel itself is untouched and is still the reference
    the profile-blocked self-check diffs against, which is the whole reason the switch could go.
    """
    for _m in [k for k in list(sys.modules) if k.startswith("routing_optimiser")]:
        del sys.modules[_m]
    os.environ.update(env)
    from routing_optimiser.s4_search import band_projection as bp
    from routing_optimiser.s4_search import blocked_fill as bf
    for _k, _v in flags.items():
        setattr(bp, _k, _v)
    if "_PROJ_CB_ON" in flags:                # _CB_OK["use"] is seeded from it at import
        bp._CB_OK["use"] = bool(flags["_PROJ_CB_ON"])
    for _s in bf.SITES:          # this test owns two of the five; the rule needs all of them
        bf.register(_s)
    return bp, bf


def _run(bp, keys, armed, P=6, seed=3):
    # 19kg: no environment variable to set. `_SW_BLOCK_NOFILL` is the setting, and it survives
    # as a NAME on the module that reads it for exactly this reason - so a test can still drive
    # both paths. Set on the module, not the environment, and read at call time either way.
    bp._SW_BLOCK_NOFILL = bool(armed)
    T0, Pc = _scaffold()
    proj = bp.PopulationBandProjector(T0, Pc, np.zeros(0), [("adyen_tav", 0)],
                                      max_share=CAP, by_profile=True)
    proj.set_blocked_keys(keys)
    rng = np.random.default_rng(seed)
    # ** 4 skews the proposal hard, so plenty of rows breach 0.97 and the water-fill runs
    pr = rng.random((P, max(len(proj.prop_keys), 1))) ** 4
    return {"numpy": proj.project_pop(pr), "numba": proj.project_pop_numba(pr),
            "armed": bool(proj._blk_armed), "nblk": int(proj._pblk.sum())}


def _ident(a, b):
    # dtype-agnostic: `.view(int64)` only works on float64, and float32 lives in the
    # profile-blocked path (the module's own _f32_eq note).
    if a[0].dtype != b[0].dtype or a[1].dtype != b[1].dtype:
        return False
    return a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


# ── which mode is this? ─────────────────────────────────────────────────────────────────
try:
    import numba as _nb
    _MODE = f"compiled (numba {_nb.__version__})"
except Exception:  # noqa: BLE001
    _MODE = "PURE PYTHON (numba absent - the same kernel bodies, uncompiled)"
print(f"  ..    kernels under test: {_MODE}")

# ── 1. the profile-blocked path (the default) ───────────────────────────────────────────
bp1, bf1 = _fresh({}, _PROJ_CB_ON=True)
check("both kernels register, so the rule can reach 5 of 5",
      not bf1.missing(), f"missing: {bf1.missing()}")
cb_un = _run(bp1, KEYS, armed=False)
check("the mask reached the projector but the rule is refused by default",
      cb_un["nblk"] == NBIN and not cb_un["armed"],
      f"nblk={cb_un['nblk']}, armed={cb_un['armed']}")
check("UNARMED: the profile-blocked kernel == the numpy reference, bit for bit",
      _ident(cb_un["numpy"], cb_un["numba"]))
cb_ar = _run(bp1, KEYS, armed=True)
check("the rule armed once every site was registered", cb_ar["armed"])
check("ARMED: the profile-blocked kernel == the numpy reference, bit for bit",
      _ident(cb_ar["numpy"], cb_ar["numba"]))
check("ARMED: and it CHANGED the projection, so the kernel really applied the rule",
      not _ident(cb_un["numba"], cb_ar["numba"]),
      f"max|dtxn| {float(np.abs(cb_un['numba'][1] - cb_ar['numba'][1]).max()):.6g}")

# ── 2. the flat path ────────────────────────────────────────────────────────────────────
bp2, bf2 = _fresh({}, _PROJ_CB_ON=False)
check("profile-blocking is off for this half of the test", not bp2._PROJ_CB_ON)
fl_un = _run(bp2, KEYS, armed=False)
fl_ar = _run(bp2, KEYS, armed=True)
check("UNARMED: the flat kernel == the numpy reference, bit for bit",
      _ident(fl_un["numpy"], fl_un["numba"]))
check("ARMED: the flat kernel == the numpy reference, bit for bit",
      _ident(fl_ar["numpy"], fl_ar["numba"]))

# ── 3. the two kernels against EACH OTHER - the check 19hv's floor had to skip ──────────
check("the two kernels agree with each other UNARMED",
      _ident(fl_un["numba"], cb_un["numba"]),
      f"max|dtxn| {float(np.abs(fl_un['numba'][1] - cb_un['numba'][1]).max()):.3e}")
check("...and ARMED, which is the self-check 19hv's exploration floor had to skip",
      _ident(fl_ar["numba"], cb_ar["numba"]),
      f"max|dtxn| {float(np.abs(fl_ar['numba'][1] - cb_ar['numba'][1]).max()):.3e}")

# ── 4. the parallel compile takes the new arguments ─────────────────────────────────────
bp3, bf3 = _fresh({"ROUTING_PROJ_PARALLEL": "1"}, _PROJ_CB_ON=False)
pa = _run(bp3, KEYS, armed=True, P=24)
check("the PARALLEL compile takes the two new arguments and agrees with numpy",
      _ident(pa["numpy"], pa["numba"]),
      f"max|dtxn| {float(np.abs(pa['numpy'][1] - pa['numba'][1]).max()):.3e}")

# ── 4b. importing tab_2 is enough to reach 5/5 ──────────────────────────────────────────
# band_projection registers the two kernel sites when it IMPORTS, and tab_2 imported it lazily
# inside `_get_pbp` - so an arming verdict read before the first projector build said "the band
# kernels are not wired" when they were, merely not yet imported. tab_2 imports it at module
# level from 19it, which makes registration deterministic.
try:
    sys.path.insert(0, str(ROOT / "app"))
    for _m in [k for k in list(sys.modules) if k.startswith("routing_optimiser")]:
        del sys.modules[_m]
    import tab_2_routing_engine as _t2   # noqa: F401
    from routing_optimiser.s4_search import blocked_fill as _bf2
    check("importing tab_2 alone registers all five sites",
          not _bf2.missing(), f"missing: {_bf2.missing()}")
    check("...so a verdict read before the first projector build can actually arm",
          _bf2.arming_verdict(True)[0] is True, _bf2.arming_verdict(True)[1][:90])
except Exception as _e:  # noqa: BLE001
    check("importing tab_2 alone registers all five sites", False,
          f"{type(_e).__name__}: {_e}")

# ── 5. no blocked row anywhere => armed is indistinguishable from refused ───────────────
bp4, bf4 = _fresh({}, _PROJ_CB_ON=True)
z_un = _run(bp4, frozenset(), armed=False)
z_ar = _run(bp4, frozenset(), armed=True)
check("with NO blocked row, arming changes nothing at all",
      _ident(z_un["numba"], z_ar["numba"]) and _ident(z_un["numpy"], z_ar["numpy"]))
check("...and the projector says so rather than claiming to be armed",
      not z_ar["armed"] and z_ar["nblk"] == 0)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
