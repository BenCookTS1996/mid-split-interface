# Full-matrix BIN-grain GA — wiring plan (opt-in engine)

Status after this session: **Stage 1 (core) is built and offline-validated.** The
new module `src/routing_optimiser/genetic_fullmatrix.py`
(`__build__ = "2026-08-12-fullmatrix-binmatrix-ga-dualceiling-adaptivetol"`) is
self-contained numpy, `py_compile`-clean, and passes a synthetic multi-cell /
multi-MID smoke test: VWSR improved over the safe seed (0.813 → 0.860), stayed
feasible, and the per-mid VAMP rates sat right at the soft ceiling (0.0109 vs
hard 0.009 / soft 0.011) — i.e. the adaptive-tolerance boundary-hugging works.

Nothing existing was touched — Genetic / GA - Numba are untouched.

## What the core already does
- Full-matrix genome: one logit per (cell × eligible gateway) at BIN grain;
  per-cell softmax → shares (any allocation reachable, no tilt/anchor limit).
- BIN-grain scoring: VWSR objective + per-vampMid VAMP on the rates you pass in.
- Dual VAMP ceilings per mid (hard `max_vamp` + soft `max_pass_vamp`, cheaper
  route passes) + adaptive VWSR tolerance so the search hugs the boundary.
- Elite seeding from a caller reference + **never-worse-than-seed guarantee**.
- Returns `(best_shares, info)`; `info` flags `feasible`, `improved_over_seed`,
  and states compliance is SOFT in-search — the caller MUST run enforcement.

## Remaining stages (need the live env to validate — no BQ/Streamlit in-container)

### Stage 2 — data feed  ✅ DONE (offline-validated)
- Added `max_pass_vamp` (soft ceiling) to `HardConstraints` — defaults to `None`
  → falls back to `vamp_cap` (single hard wall). Other engines ignore it.
- Added `genetic_fullmatrix.build_fullmatrix_problem(cell_problems, hard, *,
  mid_caps=None, exploration_floor=None)` → `(problem, meta)`. It consumes the
  app's EXISTING `CellProblem` list at its finest grain (no pooling):
  - `succ` = `cell.success_rates` (ALREADY EB-shrunk upstream by
    `gateway_success_rates` inside `build_cell_problems` — no re-shrink needed).
  - `risk` = `cell.risk_rates` (per-gateway VAMP, from `bin_rpgt_impact_export`).
  - `mid_id` = global index of the gateway/MID NAME, so the same MID across cells
    is one vampMid whose cap applies to its cross-cell aggregate.
  - `max_share` from `hard.max_gateway_share`; dual caps from `vamp_cap` /
    `max_pass_vamp`, with optional per-MID overrides via `mid_caps`
    (`{mid_name: (hard, soft)}`, from the tab-3 editor).
  - `meta` returns `gw_names`, `mid_names`, and `baseline_shares` (for the elite
    seed).
- Validated: `test_fullmatrix_builder.py` — from a neutral (infeasible) seed the
  GA delivered a feasible, higher-VWSR split by a cross-cell shift, MID rates at
  the soft cap. **Still needs a run against the LIVE cell problems** to confirm
  the real MID grouping / cap wiring behaves as expected.

### Stage 3 — register as opt-in engine (`app/tab2_engine.py`)  🟡 DRAFTED (apply live)

The `ctx`-native adapter (`problem_from_ctx` / `reconstruct_full_split`) is built
and offline-validated (`test_fullmatrix_ctx.py`: eligibility drop + reconstruct +
cross-cell boundary-hug all pass). The tab change is small and additive. Apply
these three edits at your machine, clear `__pycache__`, restart Streamlit.

**Line numbers are approximate — match on the anchor text.**

**(a) Dropdown option (~line 114, next to the `genetic_numba` append):**
```python
        if "genetic_fullmatrix" not in {k for k, _ in choices}:
            choices.append(("genetic_fullmatrix",
                            "GA - Full matrix (BIN grain, experimental)"))
```
Also include it wherever `genetic`/`genetic_numba` are treated as genetic, e.g.:
```python
        _is_genetic = engine_key in ("genetic", "genetic_numba", "genetic_fullmatrix")
```
(and the analogous `_pre_engine in (...)` checks at ~156 / ~705, and ~945 / ~3088).

**(b) Compute the full-matrix endpoint (inside the genetic block, right AFTER
`ctx` is fully assembled — after `ctx["ref_share"] = _ref_share_G`, ~line 3437,
and after `_comp_share_G` exists, ~line 4205):**
```python
        if engine_key == "genetic_fullmatrix":
            from routing_optimiser.genetic_fullmatrix import (
                problem_from_ctx, run_fullmatrix_ga, reconstruct_full_split)
            # soft ceiling: ride to 1.25x the hard cap in-search; enforcement
            # tightens back to the hard cap. Pull per-MID overrides from the
            # tab-3 editor if present (index-aligned to _mids_u).
            _fm_problem, _fm_meta = problem_from_ctx(ctx, soft_cap_mult=1.25)
            _fm_best, _fm_info = run_fullmatrix_ga(
                _fm_problem,
                reference_shares=_fm_meta["reference_kept"],  # kept-row seed
                pop_size=int(ss.get("ga_seeds", 8)) * 8,
                generations=200, seed=0, log_fn=log)
            _fm_full = reconstruct_full_split(_fm_best, _fm_meta)
            log(f"   [full-matrix] vwsr={_fm_info['vwsr']:.5f} "
                f"viol={_fm_info['violation']:.2e} feasible={_fm_info['feasible']} "
                f"improved={_fm_info['improved_over_seed']}")
            # Hand BOTH endpoints the same full-matrix split; the frontier blend
            # + existing enforcement below run unchanged on it.
            _comp_endpoint_G = _fm_full
            _safe_endpoint_G = _fm_full
```
Place this so it SHORT-CIRCUITS the tilt CMA-ES endpoint search for this engine
(guard the `_comp_endpoint_G = _comp_share_G` / risk-min CMA-ES block with
`if engine_key != "genetic_fullmatrix":`).

**(c) Enforcement is automatic.** The delivery lines (~4567)
`_deliver_G = _safe_endpoint_G` → `_ga_gran = _explode(_endpoint_agg(_deliver_G))`
already run the VAMP-cap LP + band projection + eligibility on whatever `_G` you
assign. Because we set `_safe_endpoint_G = _fm_full`, the soft (≤1.25× cap)
in-search split is tightened to the HARD cap here. **Do not add a second
enforcement path — reuse this one.** Never ship `_fm_full` raw.

**(d) History chart / efficiency readout:** `_fm_info["history"]` is a list of
`(gen, best_vwsr, best_viol)` — feed it to the same chart the tilt GA uses.

Risk to watch on the live run: `problem_from_ctx` assumes `ctx['sr']` is the
EB-shrunk success rate and `ctx['risk']` the period-0 VAMP rate (confirmed by the
ctx assembly at ~3411). If a future refactor renames those keys, the adapter must
be updated in lockstep.

### Stage 4 — speed (fused numba kernel)  ✅ DONE (numpy path validated; numba path unvalidated in-container)
- Added `_fused_eval_kernel` (segment-softmax + VWSR + violation in ONE pass,
  numba-compatible scalar loops) + `make_fused_eval(problem, use_numba=...)` in
  `genetic_fullmatrix.py`. `run_fullmatrix_ga(..., numba=True)` uses it.
- **Verify-or-fallback**: `make_fused_eval` runs the njit kernel against the numpy
  path on a random sample (allclose rtol 1e-7 / atol 1e-8); any mismatch OR a
  missing numba OR a kernel error returns the numpy evaluator. The kernel can
  never change results.
- Validated in-container (numba ABSENT, so the kernel ran as plain Python via the
  identity decorator): bit-exact vs the vectorised numpy path (max diff 4.4e-16);
  `use_numba=True` correctly fell back to numpy; GA unchanged.
- **Caveat**: the actual njit-COMPILED path could not run here (numba not
  installed). On your machine it will compile + verify on first use; watch the run
  log line `evaluator backend=numba/numpy` to confirm which path ran. If it says
  `numpy (verify mismatch ...)` the guard caught a discrepancy — send me the
  max_dv/max_dx and I'll chase it.

### TRUE BIN GRAIN — APPLIED
The run-dispatch gate (`if engine_key in (...)` at ~3097) plus 954/161/714 now
include `genetic_fullmatrix`, so it actually enters the genetic branch and the
override runs (earlier it silently fell into the non-genetic path). AND: for this
engine the `bin_to_bank` map is forced to IDENTITY (~1254), so `parent_bank ==
bank == BIN` and the whole pipeline (ctx, endpoints, delivery, explode,
compression) is built per BIN — the genome is now genuinely BIN-grain, not
parent-bank grain. Cost: more cells (~10k+), slower. Watch for the log line
"[full-matrix] TRUE BIN GRAIN: parent-bank collapse DISABLED".

### Stage 3 tab edits — APPLIED (not just drafted)
`tab2_engine.py` now has: the dropdown option, `_is_genetic`/`_use_numba`
membership, and the delivery-site override that seeds the full-matrix GA from the
KNOWN-COMPLIANT `_comp_share_G` (soft_cap == hard_cap, so feasible by
construction) and swaps its split into `_safe_endpoint_G`/`_comp_endpoint_G`.
NOTE: this path's general VAMP-cap enforcement is OFF for every engine (eligibility
projection only), which is why the full-matrix seed is the compliant greedy+LP
split rather than a soft-cap ride. Re-enabling boundary-hugging (ride to a soft
cap, then LP-tighten) requires re-wiring an explicit enforce step on `_fm_full`.

### Stage 5 — compression (`kmeans_compress.py`)
Expand the BIN-grain split up to the production config contract unchanged; the
full-matrix output is already at BIN grain so no broadcast step is needed.

## Guardrails reminder
- After any code change: `find . -name __pycache__ -type d -exec rm -rf {} +`
  then fully restart Streamlit (Ctrl+C, not just close the tab).
- Bump `__build__` on every file whose behaviour changes.
- Keep it opt-in / additive on the compliance-critical path; crash loudly, no
  silent fallbacks.
