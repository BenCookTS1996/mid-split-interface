"""19jk - the 148 ms that moves 90 MB and computes nothing.

[bpf-inside] on the 21:57 run split the biggest row in the search and found this:

    project (the kernel)                   383 ms   149.0s
    prop_raw -> C-contiguous               148 ms    57.6s   <- one line
    shares -> prop_raw                      47 ms    16.0s
    the 15-band penalty loop                 0.9 ms   0.3s

`shares_to_prop_raw` returns `(incidence @ shares.T).T` - a TRANSPOSED VIEW - and `penalty`
then makes it C-contiguous for the kernel. That is a full-width strided rewrite of a
(35 x 323,063) array, 90 MB, 338 times, and each one was a FRESH allocation whose pages had
to be faulted in before they could be written.

19jk does the SAFE half: the rewrite stays (removing it needs the array built the other way
round, which changes the order the sparse matmul sums in - a separate, riskier change), but
the allocation goes. ~30 GB of churn over a run.

A COPY IS A COPY, so this is bit-identical by construction and not by measurement - there is
no arithmetic in it to reassociate. The tests below assert it anyway, because "by
construction" is exactly what I said about the 19jb fallback block before it killed a run.
"""
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BS = (ROOT / "src/routing_optimiser/s4_search/band_scoring.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
import routing_optimiser.s4_search.band_scoring as bs

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def bits(a):
    return np.asarray(a, float).view(np.int64)


# ═══ a projector and a penalty, shaped like the live one ═════════════════════════════════
rng = np.random.default_rng(19)
SPECS = [bs.BandSpec(midl="m1", months=(1,), metric="vamp", ceil=9.0, floor=None, weight=1.0),
         bs.BandSpec(midl="m2", months=(1,), metric="txn", ceil=None, floor=40.0, weight=0.125),
         bs.BandSpec(midl="m3", months=(1,), metric="vamp", ceil=4.0, floor=None, weight=0.125)]

class _Proj:
    band_order = [("m1", 1), ("m2", 1), ("m3", 1)]
    seen = []

    def project_pop_numba(self, prop_raw):
        _p = np.asarray(prop_raw, float)
        # record the LAYOUT and the VALUES the kernel would actually have seen
        _Proj.seen.append((_p.flags["C_CONTIGUOUS"], _p.dtype, np.array(_p, copy=True)))
        return (np.stack([_p.sum(axis=1) * 0.4, _p.sum(axis=1) * 0.2, _p.sum(axis=1) * 0.1], 1),
                np.stack([_p.sum(axis=1) * 3.0, _p.sum(axis=1) * 1.5, _p.sum(axis=1) * 0.5], 1))

def mk():
    return bs.ExactBandPenalty(_Proj(), SPECS, breach_fixed=0.3, breach_quad=1.0)

# the live shape's defining feature: the caller hands over a TRANSPOSED view
P, K = 6, 900
base = rng.random((K, P)) * 20.0
VIEW = base.T                                   # (P, K), F-contiguous - what s2pr returns
check("0  the fixture reproduces the real input: a transposed, NON-contiguous view",
      VIEW.shape == (P, K) and not VIEW.flags["C_CONTIGUOUS"])


# ═══ 1. bit-identical to the ascontiguousarray it replaces ═══════════════════════════════
def reference(pen, pr, detail=None):
    """penalty() as it was before 19jk: allocate a fresh contiguous copy every call."""
    _saved = pen._contig
    pen._contig = lambda x: np.ascontiguousarray(x, dtype=float)
    try:
        return np.asarray(pen.penalty(pr, detail_out=detail), float)
    finally:
        pen._contig = _saved

CASES = [("the transposed view (the live case)", VIEW),
         ("an already C-contiguous float64 array", np.ascontiguousarray(VIEW)),
         ("a 1-D single candidate", VIEW[0].copy()),
         ("a 1-D slice of the view", base[:, 0]),
         ("float32 input", VIEW.astype(np.float32)),
         ("integer input", (VIEW * 100).astype(np.int64)),
         ("a reversed-stride view", np.ascontiguousarray(VIEW)[:, ::-1]),
         ("a zero-row population", VIEW[:0])]
for _nm, _pr in CASES:
    _ref = reference(mk(), _pr)
    _got = np.asarray(mk().penalty(_pr), float)
    check(f"1  bit-identical on {_nm}",
          _ref.shape == _got.shape and np.array_equal(bits(_ref), bits(_got)),
          f"n={_ref.size}")

# the array the KERNEL sees must be identical too, not just the penalty that comes out
_Proj.seen.clear(); reference(mk(), VIEW); _r_seen = _Proj.seen[-1]
_Proj.seen.clear(); mk().penalty(VIEW);    _g_seen = _Proj.seen[-1]
check("1  ...and the array handed to the kernel is byte-for-byte the same, same layout",
      _r_seen[0] == _g_seen[0] and _r_seen[1] == _g_seen[1]
      and np.array_equal(bits(_r_seen[2]), bits(_g_seen[2])),
      f"C-contiguous={_g_seen[0]}, dtype={_g_seen[1]}")

# detail_out still works, and repeated calls do not contaminate each other through the buffer
_p = mk()
_d1, _d2 = {}, {}
_a = np.asarray(_p.penalty(VIEW, detail_out=_d1), float)
_other = np.ascontiguousarray(base.T * 1.7)
_p.penalty(_other)                                   # overwrite the buffer in between
_b = np.asarray(_p.penalty(VIEW, detail_out=_d2), float)
check("1  a second call with a DIFFERENT array in between returns the same answer",
      np.array_equal(bits(_a), bits(_b)))
check("1  detail_out is unaffected", _d1["per_spec"].shape == (P, 3)
      and np.array_equal(bits(_d1["per_spec"]), bits(_d2["per_spec"])))


# ═══ 2. the buffer really is reused ══════════════════════════════════════════════════════
_p = mk()
_p.penalty(VIEW)
_b1 = _p._pr_buf
for _ in range(5):
    _p.penalty(VIEW)
check("2  the same buffer object serves every call at one shape",
      _p._pr_buf is _b1 and _p._pr_stat["alloc"] == 1 and _p._pr_stat["copied"] == 6,
      f"{_p._pr_stat}")
_p.penalty(base[:, :3].T)                            # a different P
check("2  a shape change reallocates rather than writing out of bounds",
      _p._pr_stat["alloc"] == 2 and _p._pr_buf.shape == (3, K))
_p2 = mk()
_p2.penalty(np.ascontiguousarray(VIEW))
check("2  an array that is ALREADY C-contiguous float64 is passed through, not copied",
      _p2._pr_stat["passthru"] == 1 and _p2._pr_stat["copied"] == 0
      and _p2._pr_buf is None)
check("2  ...and two instances do not share a buffer",
      mk()._pr_buf is None and _p._pr_buf is not None)


# ═══ 3. the wiring ═══════════════════════════════════════════════════════════════════════
check("3  the allocation is what went, not the rewrite - and the docstring says so",
      "It removes the ALLOCATION" in BS and "np.copyto(_b, _a)" in BS
      and "changes the order the sparse matmul sums in" in BS)
check("3  the escaping-buffer hazard is stated where the buffer is made",
      "THE BUFFER ESCAPES INTO `project`" in BS and "_stash_ab" in BS)
check("3  [bpf-inside] reports whether the buffer was actually reused",
      "the C-contiguous row " in T2 and "_pr_stat" in T2)
check("3  penalty() no longer allocates per call",
      "_bp_t0 = _bs_time.perf_counter()\n        prop_raw = self._contig(prop_raw)" in BS)
check("3  ...and neither does report(), which the heartbeat calls beside it",
      BS.count("prop_raw = self._contig(prop_raw)") == 2
      # the one surviving mention is `_contig`'s own docstring, naming what it replaces
      and BS.count("np.ascontiguousarray(prop_raw, dtype=float)") == 1)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
