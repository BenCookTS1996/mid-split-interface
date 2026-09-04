"""19ig — the bincount accumulation must be BIT-IDENTICAL to np.add.at, on shapes that
matter, including the awkward ones (empty mids, single candidate, unsorted mid_id)."""
import importlib.util, os, pathlib, sys
import numpy as np

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m

ROOT = pathlib.Path(__file__).resolve().parents[1]
GA = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")
sys.path.insert(0, str(ROOT / "src"))

FAIL = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok: FAIL.append(name)

rng = np.random.default_rng(19)
CASES = [
    ("live shape        P=35 R=140154 M=15", 35, 140154, 15),
    ("single candidate  P=1  R=9001   M=15", 1, 9001, 15),
    ("one mid           P=8  R=5000   M=1",  8, 5000, 1),
    ("empty mid present P=6  R=3000   M=9",  6, 3000, 9),
    ("wide mids         P=4  R=2000   M=300",4, 2000, 300),
]

# 19kg: there is no environment variable to set any more. The flag survives as a NAME
# for exactly this reason - a test can still A/B the two code paths by rebinding it.
new = load("ga_bc", GA)
old = load("ga_addat", GA); old._VIOL_BINCOUNT = False
check("the switch actually selects different code",
      new._VIOL_BINCOUNT is True and old._VIOL_BINCOUNT is False)

for label, P, R, M in CASES:
    w = rng.random((P, R)) * 1000.0
    wr = w * (rng.random(R) * 0.05)
    mid = rng.integers(0, M, R).astype(np.int64)
    if label.startswith("empty mid"):
        mid = np.where(mid == 3, 4, mid)          # mid 3 gets no rows at all
    a_n, b_n = new._mid_accum(w, wr, mid, M)
    a_o, b_o = old._mid_accum(w, wr, mid, M)
    same = (np.array_equal(a_n.view(np.int64), a_o.view(np.int64))
            and np.array_equal(b_n.view(np.int64), b_o.view(np.int64)))
    check("bit-identical: " + label, same,
          "" if same else f"max|d| {float(np.abs(a_n - a_o).max()):.3e}")

# the self-check the run relies on must actually fire and must PASS
fresh = load("ga_sc", GA)
check("self-check has not fired before the first call",
      fresh._VIOL_FACT["checked"] is False and fresh._VIOL_FACT["msg"] == "")
w = rng.random((5, 4000)) * 10.0
fresh._mid_accum(w, w * 0.02, rng.integers(0, 7, 4000).astype(np.int64), 7)
check("self-check fires on the first call and reports PASSED",
      fresh._VIOL_FACT["checked"] is True and fresh._VIOL_FACT["same"] is True
      and "SELF-CHECK PASSED" in fresh._VIOL_FACT["msg"])
check("self-check does not re-run on the second call",
      (lambda before: (fresh._mid_accum(w, w * 0.02,
                                        np.zeros(4000, np.int64), 7),
                       fresh._VIOL_FACT["msg"] == before)[1])(fresh._VIOL_FACT["msg"]))

# and _violation itself must agree end to end
v_new = load("ga_v1", GA)
v_old = load("ga_v0", GA); v_old._VIOL_BINCOUNT = False
N_ROW, M = 6000, 11
starts = np.arange(0, N_ROW, 10, dtype=np.int64)
ctx = {"n_row": N_ROW, "n_mid": M,
       "profile_starts": starts, "profile_counts": np.full(starts.size, 10, np.int64),
       "elig": np.ones(N_ROW), "base": np.full(N_ROW, 0.1),
       "profile_vol": np.repeat(rng.random(starts.size) * 500.0, 10),
       "sr": rng.random(N_ROW), "risk": rng.random(N_ROW) * 0.06,
       "mid_id": rng.integers(0, M, N_ROW).astype(np.int64),
       "mid_rows": None, "vamp_cap": 0.02, "max_share": 0.97, "floor": 0.0}
p1, _ = v_new.problem_from_ctx(ctx, soft_cap_mult=1.0)
p0, _ = v_old.problem_from_ctx(ctx, soft_cap_mult=1.0)
sh = v_new._segment_softmax(np.log(np.clip(rng.random((9, N_ROW)), 1e-6, None)),
                            p1.profile_start, p1.profile_len, p1.max_share)
x1 = v_new._violation(sh, p1); x0 = v_old._violation(sh, p0)
check("_violation end to end is bit-identical",
      np.array_equal(np.asarray(x1).view(np.int64), np.asarray(x0).view(np.int64)),
      f"max|d| {float(np.abs(x1 - x0).max()):.3e}")

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
