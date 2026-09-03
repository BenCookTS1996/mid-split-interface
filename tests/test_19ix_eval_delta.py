"""19ix - the delta decode. Same search, same numbers, 20% less work.

THE CLAIM. Crossover is uniform PER PROFILE, and the softmax is per segment. So an unmutated
profile's decoded shares in a child are the SAME operations on the SAME inputs as in the parent
it came from - bit-identical by construction, not to a tolerance. The decode is 225 ms of every
1,107 ms evaluation and re-does 14,852 profiles per child when about 1% of them changed.

WHAT THIS TEST HAS TO PROVE, in order:
  1. the whole search is BIT-IDENTICAL with the delta on and off - same shipped split, same
     success rate, same breach, same RNG consumption;
  2. the gather actually ENGAGED (a test that silently falls back proves nothing);
  3. only the mutated profiles were re-decoded;
  4. the self-check catches a desynchronised cache, because that is the one way this can be
     wrong and it would otherwise be wrong SILENTLY.
"""
import importlib.util, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GA_P = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def load(name):
    sp = importlib.util.spec_from_file_location(name, GA_P)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m

# ── a book big enough to have something to gather: 60 profiles x 4 rows ─────────────────
NP, PER = 60, 4
N_ROW = NP * PER
starts = (np.arange(NP) * PER).astype(np.int64)
counts = np.full(NP, PER, np.int64)
rng = np.random.default_rng(19)
ctx = {
    "n_row": N_ROW, "n_mid": 4,
    "profile_starts": starts, "profile_counts": counts,
    "elig": np.ones(N_ROW, float),
    "base": np.full(N_ROW, 1.0 / PER),
    "profile_vol": np.repeat(rng.integers(200, 5000, NP).astype(float), PER),
    "sr": rng.uniform(0.55, 0.95, N_ROW),
    "risk": rng.uniform(0.005, 0.06, N_ROW),
    "mid_id": np.tile(np.arange(4), NP).astype(np.int64),
    "mid_rows": None, "vamp_cap": 0.02, "max_share": 0.97, "floor": 0.0,
}

def band_pen(fd):
    F = np.atleast_2d(np.asarray(fd, float))
    return np.abs(F[:, ::3].sum(axis=1) - 0.9) * 0.01

def hooks(colmap, n_row):
    def deliver_full(sh):
        X = np.atleast_2d(np.asarray(sh, float))
        F = np.zeros((X.shape[0], n_row))
        F[:, colmap] = X
        return F
    return deliver_full

def run(delta, gens=8, pop=16, seed=4):
    if delta:
        os.environ["ROUTING_EVAL_DELTA"] = "1"
    else:
        os.environ.pop("ROUTING_EVAL_DELTA", None)
    m = load("ga_delta_" + ("on" if delta else "off"))
    p, meta = m.problem_from_ctx(ctx, soft_cap_mult=1.0)
    colmap = np.asarray(meta["keep_idx"])[p.order]
    best, info = m.run_fullmatrix_ga(
        p, reference_shares=meta["reference_kept"], pop_size=pop, generations=gens, elite=3,
        patience=99, seed=seed, numba=False, band_penalty_fn=band_pen,
        deliver_full_fn=hooks(colmap, N_ROW), obj_full=meta.get("obj_full"))
    return m, np.asarray(best, float), info

m_off, best_off, info_off = run(False)
m_on, best_on, info_on = run(True)

# ── 1. BIT-IDENTICAL ────────────────────────────────────────────────────────────────────
check("the shipped split is BIT-IDENTICAL with the delta on",
      best_off.shape == best_on.shape
      and np.array_equal(best_off.view(np.int64), best_on.view(np.int64)),
      f"max|d| {float(np.abs(best_off - best_on).max()):.3e}"
      if best_off.shape == best_on.shape else f"{best_off.shape} vs {best_on.shape}")
for _k in ("success_rate", "band", "other", "feasible", "improved_over_seed", "evaluated"):
    if _k in info_off or _k in info_on:
        check(f"info['{_k}'] is unchanged", info_off.get(_k) == info_on.get(_k),
              f"{info_off.get(_k)} vs {info_on.get(_k)}")

# ── 2. IT ACTUALLY ENGAGED ──────────────────────────────────────────────────────────────
# (the module keeps its counters on the closure, so read them off the run's own info stash)
_g = getattr(m_on, "_LAST_DELTA", None)
check("the delta reports itself in the run info", _g is not None, str(_g))
if _g:
    check("...and it GATHERED most of the candidates it saw",
          _g["gathered"] > 0 and _g["gathered"] > _g["full"],
          f"gathered {_g['gathered']:,} vs full {_g['full']:,}")
    check("...and it re-decoded only a small fraction of profiles",
          0 < _g["prof_re"] < 0.2 * max(_g["prof_tot"], 1),
          f"{_g['prof_re']:,} of {_g['prof_tot']:,} "
          f"({100.0 * _g['prof_re'] / max(_g['prof_tot'], 1):.1f}%)")
    check("...and the self-check ran EVERY gathered generation, not once",
          _g["checks"] >= 2, f"{_g['checks']} check(s)")
    check("...and it never turned itself off", _g["on"] and not _g["why"], str(_g["why"]))

_h = getattr(m_off, "_LAST_DELTA", None)
check("with the switch off nothing is gathered at all",
      _h is None or (_h["gathered"] == 0 and _h["full"] > 0),
      str(_h and (_h["gathered"], _h["full"])))

# ── 3. THE GUARD CATCHES A DESYNCHRONISED CACHE ─────────────────────────────────────────
# Corrupt one row of the parent cache mid-run and confirm the check fires. This is the failure
# the whole optimisation risks, so it is tested rather than argued.
os.environ["ROUTING_EVAL_DELTA"] = "1"
m_bad = load("ga_delta_bad")
_orig = m_bad._segment_softmax
_state = {"n": 0}
def _poison(logits, ps, pl, ms=None):
    """Corrupt the delta's OWN sub-decode by one ulp, and see whether the check notices."""
    _out = _orig(logits, ps, pl, ms)
    # The delta's OWN sub-decode is the only call with fewer columns than the full row set: it
    # decodes just the mutated profiles, through a segment layout this optimisation builds
    # itself. Getting that layout wrong is the most likely way for the gather to be subtly
    # wrong, so that is what gets corrupted here - by one ulp, which is the smallest error the
    # check must still catch. Full-width calls (the seed, [decode-cap], and the check's own
    # reference) are untouched; poisoning the reference would make the check agree with the
    # corruption, which is the opposite of the point.
    _a = np.asarray(_out)
    if _a.ndim == 2 and _a.shape[1] < N_ROW:
        _state["n"] += 1
        _out = _a.astype(float).copy()
        _out[0, 0] = np.nextafter(_out[0, 0], 1.0)
    return _out
m_bad._segment_softmax = _poison
p3, meta3 = m_bad.problem_from_ctx(ctx, soft_cap_mult=1.0)
_cm3 = np.asarray(meta3["keep_idx"])[p3.order]
try:
    m_bad.run_fullmatrix_ga(p3, reference_shares=meta3["reference_kept"], pop_size=16,
                            generations=8, elite=3, patience=99, seed=4, numba=False,
                            band_penalty_fn=band_pen, deliver_full_fn=hooks(_cm3, N_ROW),
                            obj_full=meta3.get("obj_full"))
    _b = getattr(m_bad, "_LAST_DELTA", None)
    check("a one-ulp cache corruption is CAUGHT and the delta disables itself",
          _b is not None and not _b["on"] and "did NOT match" in str(_b["why"]),
          str(_b and _b["why"])[:80])
except Exception as _e:
    check("a one-ulp cache corruption is CAUGHT and the delta disables itself", False,
          f"the run raised instead: {type(_e).__name__}: {_e}")
finally:
    os.environ.pop("ROUTING_EVAL_DELTA", None)

# ── 4. the masks that make it possible cost no extra draws ──────────────────────────────
m4 = load("ga_delta_masks")
_r1 = np.random.default_rng(7)
_r2 = np.random.default_rng(7)
_a = _r1.standard_normal(N_ROW); _b = _r1.standard_normal(N_ROW)
_ = _r2.standard_normal(N_ROW); __ = _r2.standard_normal(N_ROW)
_c1 = m4._child_fused(_a, _b, 0.1, 0.4, starts, counts, _r1)
_c2, _pk, _hit = m4._child_fused(_a, _b, 0.1, 0.4, starts, counts, _r2, want_masks=True)
check("want_masks returns the SAME child, bit for bit",
      np.array_equal(np.asarray(_c1).view(np.int64), np.asarray(_c2).view(np.int64)))
check("...and the same generator state, so no draw was added or reordered",
      repr(_r1.bit_generator.state) == repr(_r2.bit_generator.state))
check("...and the masks describe the child: every unmutated profile equals ONE parent",
      all(np.array_equal(np.asarray(_c2)[s:s + PER].view(np.int64),
                         (_a if _pk[i] else _b)[s:s + PER].view(np.int64))
          for i, s in enumerate(starts) if not _hit[i]))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
