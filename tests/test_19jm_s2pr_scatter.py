"""19jm - prop_raw is a scatter, not a matmul.

[bpf-inside] on the 22:30 run put `shares -> prop_raw` at 39.1 ms and `prop_raw ->
C-contiguous` at 126.1 ms - 61.6s between them. 19jk had already established that the second
one is ALL transpose and no allocation, so the pair is: multiply by a sparse matrix, then
rewrite the answer the other way round.

THE MATRIX IS A PERMUTATION. [inc-build] constructs it from `np.ones`, and the live run reads
154,405 non-zeros over 154,405 share columns and 154,405 reachable prop-keys - one non-zero
per column, one per row, every weight exactly 1.0. So `prop_raw[p, k] = shares[p, col(k)]`:
a scatter. There is no sum to reassociate and no multiply to reorder, which is what makes it
bit-identical BY CONSTRUCTION rather than by argument. Measured 229.5 -> 71.8 ms, 3.2x.

THAT IS A PROPERTY OF THIS DATA, NOT A LAW - the builder appends one (row, col) per matching
BIN, so a column can carry several. The detection has to REFUSE anything that is not a
permutation, and the three ways it can fail are tested below alongside the equivalence.
"""
import pathlib, re, sys
import numpy as np
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def bits(a):
    return np.asarray(a, float).view(np.int64)


# ═══ the detection, lifted verbatim from tab_2 so the test cannot drift from it ══════════
def is_perm(inc):
    coo = inc.tocoo()
    N = int(inc.shape[1])
    return bool(coo.nnz == N
                and np.array_equal(np.sort(coo.col), np.arange(N))
                and np.unique(coo.row).size == coo.nnz
                and np.all(np.asarray(coo.data) == 1.0))

def row_of_col(inc):
    coo = inc.tocoo()
    r = np.empty(int(inc.shape[1]), np.intp)
    r[np.asarray(coo.col, np.intp)] = np.asarray(coo.row, np.intp)
    return r


# ═══ 1. the equivalence, on a live-shaped permutation ════════════════════════════════════
rng = np.random.default_rng(19)
N, K = 4_000, 9_000                     # the live ratio: 47.8% of keys reachable, unsorted
rows = rng.permutation(K)[:N]
INC = sp.csr_matrix((np.ones(N), (rows, np.arange(N))), shape=(K, N))
check("1  the fixture IS a permutation, and an unsorted one",
      is_perm(INC) and not np.all(np.diff(rows) > 0),
      f"{N:,} column(s) onto {N:,} of {K:,} key(s)")

R = row_of_col(INC)

def scatter(shares, buf):
    X = np.asarray(shares, float)
    one = X.ndim == 1
    if one:
        X = X[None, :]
    o = buf[:X.shape[0]]
    for p in range(X.shape[0]):
        o[p, R] = X[p]
    return o[0] if one else o

for _P in (35, 40, 1):
    S = np.ascontiguousarray(rng.random((_P, N)))
    ref = np.ascontiguousarray(np.asarray(shares_to_prop_raw(S, INC)), dtype=float)
    got = scatter(S, np.zeros((_P, K)))
    check(f"1  bit-identical to the sparse matmul at P={_P}",
          ref.shape == got.shape and np.array_equal(bits(ref), bits(got)))
    check(f"1  ...and the scatter's output is C-contiguous, so nothing has to make it so (P={_P})",
          got.flags["C_CONTIGUOUS"])
S1 = np.ascontiguousarray(rng.random(N))
check("1  bit-identical on a 1-D single candidate, and returns 1-D",
      np.array_equal(bits(np.asarray(shares_to_prop_raw(S1, INC))[0]),
                     bits(scatter(S1, np.zeros((1, K)))))
      and scatter(S1, np.zeros((1, K))).ndim == 1)

# THE INVARIANT THE BUFFER RESTS ON: the unmapped keys are written once, at np.zeros, and
# every later call writes exactly the same mapped columns - so they stay zero forever.
BUF = np.zeros((40, K))
_unmapped = np.setdiff1d(np.arange(K), R)
for _P in (40, 1, 35, 40, 35, 1):
    scatter(np.ascontiguousarray(rng.random((_P, N)) + 5.0), BUF)
check("1  the unmapped prop-keys are still exactly zero after six calls at three widths",
      not BUF[:, _unmapped].any(),
      f"{_unmapped.size:,} of {K:,} key(s) are unreachable and never written again")
# ...and a narrower call must not leave a wider one's rows readable as its own
S35 = np.ascontiguousarray(rng.random((35, N)))
scatter(np.ascontiguousarray(rng.random((40, N))), BUF)
check("1  a wide call then a narrow one: every returned row is the CALLER's, not a leftover",
      np.array_equal(bits(scatter(S35, BUF)),
                     bits(np.ascontiguousarray(np.asarray(shares_to_prop_raw(S35, INC)), float))))


# ═══ 2. it REFUSES anything that is not a permutation ════════════════════════════════════
_bad = {
    "a column feeding TWO prop-keys (the builder does this per matching BIN)":
        sp.csr_matrix((np.ones(N + 1), (np.append(rows, rows[0] + 1 if rows[0] + 1 < K else 0),
                                        np.append(np.arange(N), 0))), shape=(K, N)),
    "two columns feeding ONE prop-key (a real sum, whose order matters)":
        sp.csr_matrix((np.ones(N), (np.where(np.arange(N) == 1, rows[0], rows), np.arange(N))),
                      shape=(K, N)),
    "a weight that is not 1.0":
        sp.csr_matrix((np.where(np.arange(N) == 3, 0.5, 1.0), (rows, np.arange(N))), shape=(K, N)),
    "a column with no prop-key at all":
        sp.csr_matrix((np.ones(N - 1), (rows[:-1], np.arange(N - 1))), shape=(K, N)),
}
for _nm, _m in _bad.items():
    check(f"2  refused: {_nm}", not is_perm(_m))
check("2  ...and the good one is still accepted after all that", is_perm(INC))


# ═══ 3. the wiring ═══════════════════════════════════════════════════════════════════════
check("3  tab_2 checks all four conditions before it trusts the shortcut",
      "np.array_equal(np.sort(_sc_coo.col)," in T2
      and "np.unique(_sc_coo.row).size == _sc_coo.nnz" in T2
      and "np.all(np.asarray(_sc_coo.data) == 1.0)" in T2
      and "_sc_coo.nnz == _sc_N" in T2)
check("3  ...and self-checks against the matmul on the first LIVE call as well",
      "[s2pr-perm] \\u2713 SELF-CHECK PASSED" in T2
      and "[s2pr-perm] \\u26a0 SELF-CHECK FAILED" in T2
      and '_st["on"] = False' in T2)
check("3  a failed self-check ships the MATMUL's answer, not the scatter's",
      re.search(r'_st\["on"\] = False[\s\S]{0,900}?return _ref\[0\] if _one else _ref', T2)
      is not None)
check("3  the buffer is np.zeros, because the unmapped keys are never written again",
      "_b = _st[\"a\"] = np.zeros((_P, _K), float)" in T2
      and "the rest stay the zeros they were" in T2)
check("3  it declines when a backup catch-all folds the prop vector after it is built",
      "a backup catch-all is configured" in T2)
check("3  `_SW_S2PR_SCATTER = False` reverts, and the log says which way it went",
      "_SW_S2PR_SCATTER = True" in T2 and "elif (not _SW_S2PR_SCATTER):" in T2
      and "[s2pr-perm] the incidence is a PERMUTATION" in T2
      and "[s2pr-perm] OFF - " in T2)
check("3  the measured expectation is on record, not implied",
      "229.5 ms" in T2 and "71.8 ms" in T2)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
