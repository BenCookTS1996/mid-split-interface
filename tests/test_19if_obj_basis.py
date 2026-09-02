"""19if — the delivered objective's BASIS. Four things must hold.

D1  the objective is scored on the FULL-GRAIN delivered array, not on gather_fn's
    per-profile-renormalised copy of it (which is a shape, not a share).
D2  the SEED is scored on the same basis as its challengers, so best_key and top_key
    are the same quantity.
OFF ROUTING_DECODE_OBJ unset must be BIT-IDENTICAL to 19ie.
"""
import importlib.util, io, os, sys
import numpy as np

import pathlib, subprocess, tempfile

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m

ROOT = pathlib.Path(__file__).resolve().parents[1]
GA = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")
# The bit-identity reference is 19ie (ee9a5b6), the commit immediately before 19if - NOT HEAD,
# which would make the OFF check trivially true the moment 19if landed.
REF_REV = "ee9a5b6"
_tmp = tempfile.mkdtemp(prefix="19if_ref_")
REF = str(pathlib.Path(_tmp) / "ga_ref.py")
with io.open(REF, "wb") as _f:
    _f.write(subprocess.check_output(
        ["git", "-C", str(ROOT), "show",
         REF_REV + ":src/routing_optimiser/s4_search/genetic_fullmatrix.py"]))

sys.path.insert(0, str(ROOT / "src"))

# ── a small book with (a) config-banned rows, so keep_idx < n_row, and (b) rows the
#    DELIVERY transform zeroes, so the delivered split does not sum to 1 per profile.
N_ROW = 12
starts = np.array([0, 4, 8], np.int64)
counts = np.array([4, 4, 4], np.int64)
rng = np.random.default_rng(11)
ctx = {
    "n_row": N_ROW, "n_mid": 3,
    "profile_starts": starts, "profile_counts": counts,
    "elig": np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], float),   # rows 3, 9 config-banned
    "base": np.full(N_ROW, 0.25),
    "profile_vol": np.repeat(np.array([1000.0, 700.0, 450.0]), 4),
    "sr":   np.array([.90,.82,.75,.60,.88,.79,.71,.85,.93,.55,.68,.77]),
    "risk": np.array([.010,.020,.030,.050,.012,.024,.031,.018,.009,.060,.028,.015]),
    "mid_id": np.array([0,1,2,0,1,2,0,1,2,0,1,2], np.int64),
    "mid_rows": None, "vamp_cap": 0.02, "max_share": 0.97, "floor": 0.0,
}

def build(mod):
    p, meta = mod.problem_from_ctx(ctx, soft_cap_mult=1.0)
    colmap = np.asarray(meta["keep_idx"])[p.order]          # sorted-kept -> full column
    return p, meta, colmap

# ── the delivery transform: eligibility zeroes two rows that the GENOME still carries.
#    This is what makes the delivered split sum to < 1 per profile.
DELIV_ZERO = np.array([2, 6], np.intp)

def make_hooks(colmap, n_row):
    def deliver_full(sh):
        X = np.atleast_2d(np.asarray(sh, float))
        F = np.zeros((X.shape[0], n_row))
        F[:, colmap] = X
        F[:, DELIV_ZERO] = 0.0                      # eligibility
        return F
    def gather(fd):                                  # tab_2's _fm_gather, renormalise included
        D = np.atleast_2d(np.asarray(fd, float))
        d = D[:, colmap]
        seg = np.repeat(np.add.reduceat(d, np.asarray(_KS, np.intp), axis=1), _KC, axis=1)
        return np.where(seg > 1e-12, d / np.where(seg > 1e-12, seg, 1.0), d)
    return deliver_full, gather

def band_pen(fd):
    F = np.atleast_2d(np.asarray(fd, float))
    return np.abs(F[:, ::3].sum(axis=1) - 0.9) * 0.01     # small, deterministic, non-constant

FAIL = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)

# ══ 1. OFF IS BIT-IDENTICAL TO HEAD ══════════════════════════════════════════════════════
os.environ.pop("ROUTING_DECODE_OBJ", None)
res = {}
for tag, path in (("head", REF), ("new", GA)):
    mod = load("ga_" + tag, path)
    p, meta, colmap = build(mod)
    _KS, _KC = p.profile_start, p.profile_len
    globals()["_KS"], globals()["_KC"] = _KS, _KC
    dfull, gath = make_hooks(colmap, N_ROW)
    kw = dict(reference_shares=meta["reference_kept"], pop_size=12, generations=6, elite=3,
              patience=99, seed=3, numba=False, band_penalty_fn=band_pen,
              deliver_full_fn=dfull, gather_fn=gath)
    if tag == "new":
        kw["obj_full"] = meta["obj_full"]
    best, info = mod.run_fullmatrix_ga(p, **kw)
    res[tag] = (np.asarray(best, float), info, mod, p, meta, colmap, dfull, gath)

b_h, i_h = res["head"][0], res["head"][1]
b_n, i_n = res["new"][0], res["new"][1]
check("OFF: shipped split is bit-identical to 19ie",
      b_h.shape == b_n.shape and np.array_equal(b_h.view(np.int64), b_n.view(np.int64)),
      f"{b_h.shape} vs {b_n.shape}")
check("OFF: success rate is bit-identical to 19ie",
      float(i_h["success_rate"]) == float(i_n["success_rate"]),
      f"{i_h['success_rate']!r} vs {i_n['success_rate']!r}")
check("OFF: seed success rate is bit-identical to 19ie",
      float(i_h["seed_success_rate"]) == float(i_n["seed_success_rate"]))

# ══ 2. D1 — WHAT THE RENORMALISE DOES TO THE NUMBER ══════════════════════════════════════
mod, p, meta, colmap, dfull, gath = res["new"][2:]
_KS, _KC = p.profile_start, p.profile_len
OF = meta["obj_full"]
sh = mod._segment_softmax(np.log(np.clip(rng.random((5, p.profile_id.size)), 1e-6, None)),
                          p.profile_start, p.profile_len, p.max_share)
fd = dfull(sh)
total_vol = float(p.vol[p.profile_start].sum())

v_full = mod._success_rate(fd, OF.vol, OF.succ, OF.total_vol)          # 19if
v_renorm = mod._success_rate(gath(fd), p.vol, p.succ, total_vol)        # pre-19if
kept_raw = np.asarray(fd, float)[:, colmap]                             # gather, NO renormalise
v_keptraw = mod._success_rate(kept_raw, p.vol, p.succ, total_vol)

check("D1: full-grain view carries the same denominator as the kept problem",
      abs(float(OF.total_vol) - total_vol) <= 1e-9 * total_vol,
      f"{OF.total_vol!r} vs {total_vol!r}")
check("D1: full-grain == un-renormalised gather (no share lives outside the genome here)",
      np.allclose(v_full, v_keptraw, rtol=0, atol=1e-15),
      f"max |d| {float(np.abs(v_full - v_keptraw).max()):.3e}")
check("D1: the renormalised objective is a DIFFERENT, HIGHER number (the defect is real)",
      bool(np.all(v_renorm > v_full + 1e-6)),
      f"mean d {float((v_renorm - v_full).mean()):+.6f}")
psum_full = np.add.reduceat(fd, OF.profile_start, axis=1)
psum_ren = np.add.reduceat(gath(fd), p.profile_start, axis=1)
check("D1: delivered profiles do NOT sum to 1; the renormalised copy does",
      bool(psum_full.min() < 0.999) and bool(np.allclose(psum_ren, 1.0, atol=1e-9)),
      f"delivered min {float(psum_full.min()):.6f}")

# ══ 3. D2 — THE SEED IS ON THE SAME BASIS AS ITS CHALLENGERS ═════════════════════════════
os.environ["ROUTING_DECODE_OBJ"] = "1"
mod2 = load("ga_armed", GA)
p2, meta2, colmap2 = build(mod2)
_KS, _KC = p2.profile_start, p2.profile_len
dfull2, gath2 = make_hooks(colmap2, N_ROW)
OF2 = meta2["obj_full"]
logs = []
best2, info2 = mod2.run_fullmatrix_ga(
    p2, reference_shares=meta2["reference_kept"], pop_size=16, generations=10, elite=4,
    patience=99, seed=3, numba=False, band_penalty_fn=band_pen, log_fn=lambda s: logs.append(s),
    deliver_full_fn=dfull2, gather_fn=gath2, obj_full=OF2)

s0 = mod2._segment_softmax(
    np.log(np.clip(np.asarray(meta2["reference_kept"], float)[p2.order], 1e-6, None))[None, :],
    p2.profile_start, p2.profile_len, p2.max_share)
seed_deliv = float(mod2._success_rate(dfull2(s0), OF2.vol, OF2.succ, OF2.total_vol)[0])
seed_raw = float(mod2._success_rate(s0, p2.vol, p2.succ, float(p2.vol[p2.profile_start].sum()))[0])
check("D2: the recorded seed score IS the delivered-basis one",
      abs(float(info2["seed_success_rate"]) - seed_deliv) <= 1e-12,
      f"info {float(info2['seed_success_rate']):.9f} vs delivered {seed_deliv:.9f} "
      f"(raw was {seed_raw:.9f})")
check("D2: the raw basis really is higher, which is what made the incumbent unbeatable",
      seed_raw > seed_deliv + 1e-9, f"{seed_raw:.9f} > {seed_deliv:.9f}")
check("D2: with both bases aligned the search can beat its own seed",
      float(info2["success_rate"]) >= float(info2["seed_success_rate"]) - 1e-15,
      f"best {float(info2['success_rate']):.9f} vs seed {float(info2['seed_success_rate']):.9f}")
check("D2: [obj-basis] is emitted and reports the re-score",
      any("[obj-basis]" in s and "re-scored on the delivered split" in s for s in logs))
check("D2: [obj-basis] finds the full-grain view lines up",
      any("[obj-basis]" in s and "same denominator" in s for s in logs)
      and not any("⚠⚠" in s and "obj-basis" in s for s in logs))
# 19ig: this toy's delivery hook ZEROES rows without redistributing, so it genuinely loses
# mass — unlike the live pipeline, where eligibility blends the incapable share onto capable
# siblings and the delivered split sums to 1. That makes it the positive control for
# [obj-check]'s mass guard: the warning MUST fire here, and (per the 23:01 run) must NOT fire
# on a real book.
check("D1: [obj-check]'s mass guard fires on a delivery hook that really loses mass",
      any("[obj-check]" in s and "LOST OR GAINED MASS" in s for s in logs))
check("D1: [obj-check] prices the renormalise against the pre-19if objective",
      any("[obj-check]" in s and "PRE-19if objective" in s for s in logs))
# gather must be untouched by the hot loop: count calls
calls = {"n": 0}
def counting_gather(fd, _g=gath2):
    calls["n"] += 1
    return _g(fd)
mod3 = load("ga_count", GA)
p3, meta3, colmap3 = build(mod3)
_KS, _KC = p3.profile_start, p3.profile_len
dfull3, _ = make_hooks(colmap3, N_ROW)
mod3.run_fullmatrix_ga(p3, reference_shares=meta3["reference_kept"], pop_size=16, generations=10,
                       elite=4, patience=99, seed=3, numba=False, band_penalty_fn=band_pen,
                       deliver_full_fn=dfull3, gather_fn=counting_gather,
                       obj_full=meta3["obj_full"])
check("D1: gather_fn is called at most twice (the [obj-check] pricing + the 19ia return path), "
      "not once per evaluation",
      calls["n"] <= 2, f"{calls['n']} call(s) over 10 generations x 16 candidates")

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
