# Scope — moving the optimiser **decision** grain from cell to sub-cell

**Goal:** let the genetic optimiser *decide* at the grain production deploys —
`bank × currency × RPGT × paymentMethodProvider × Country` (**sub-cell**) — instead of deciding at
`bank × currency × RPGT` (**cell**) and letting `build_split_exports` expand/concentrate it afterward.

**Design decision (locked):** the **decision grain** moves to sub-cell, but the **scoring (success-rate)
grain stays at cell**, and the **forecast pipeline / pro-rata export are untouched**. So:

- Success rates are **not** split by pmp/Country (removes the data-sparsity risk). Every sub-cell of a cell
  shares the same gateway conversion rates.
- The GA routes sub-cells differently **only** to satisfy their different **VAMP exposure and eligibility**
  (wallet, USA/Non-USA) — not conversion. That is precisely the lever that closes the scored↔delivered gap.
- Sub-cell **volume/VAMP** come from the pro-rata export (already the finest grain); the forecast pipeline
  keeps producing cell-grain volume, which we split to sub-cells by the export's VI-Txn fractions.

**Why:** the scored↔delivered VAMP gap exists only because the GA optimises coarser than it deploys, then
broadcasts. Deciding at sub-cell makes scored == delivered **by construction**, and the expensive
`build_split_exports` expansion is no longer needed inside the loop (the GA decides sub-cell shares directly;
eligibility + cap are cheap and already sub-cell-native).

---

## Verdict up front

- **No data blocker.** pmp/Country are already in the attempts, the pro-rata export, the VAMP band projector
  (`band_projection._GRPK`), and the eligibility masks (`_T0_emask_a`). The exporter already emits per-pmp/
  Country rows.
- **Sparsity risk removed** by keeping success rates at cell grain (per your direction).
- Remaining work is threading the finer **decision** grain through three places + unwinding one broadcast.
- Does **not** move the joint-infeasibility wall (Braintree/WorldPay ~+8–11% over regardless of grain).

---

## NOT changing (per direction)

| Item | Why unchanged |
|---|---|
| `vamp_forecast_pipeline` (`_prorata_to_pre`, `_normalise_pre`) | pro-rata export stays the finest source; forecast keeps cell-grain volume |
| `success_rates.gateway_success_rates` | success-rate grain = scoring grain = cell; not split by pmp/Country |

Consequence to stay conscious of: **decision grain ≠ scoring grain.** Sub-cells of the same cell see
identical gateway conversion rates; they differ only by VAMP and eligibility. Intended — that's the whole point.

---

## Already sub-cell-native (free)

| Subsystem | Where |
|---|---|
| VAMP / band projector | `band_projection.py` `_GRPK = cur\|bin\|rpgt\|pmp\|ctry\|per` |
| Eligibility masks (wallet / USA-only) | `tab2_engine.py:2447` `_T0_emask_a` |
| Deliverable exporter (output rows) | `impact_calcs.build_split_exports:1373,1513` |
| Sub-cell volume + VAMP data | `vamp_t_period_prorata_export.csv` |

---

## What must change (the decision path)

**3. GA cell key → sub-cell (+ the volume glue).** `M`
`tab2_engine.py:2203` / `:2872` build the cell key `currency|bank|rpgt`; extend it to
`currency|bank|rpgt|pmp|ctry` (+ a grain option at `:495`). Two supporting joins at assembly time:
- **succ (broadcast):** attach each sub-cell's gateway success rate = its parent cell's rate (no
  `success_rates` change; just don't split — map cell rate onto each sub-cell row).
- **vol (split):** distribute each cell's forecast volume across its sub-cells by the pro-rata export's
  VI-Txn fractions (a join, not a forecast_pipeline change).
`ctx['cell_starts']/cell_counts` then carry sub-cell segments automatically — **`FullMatrixProblem` and
`run_fullmatrix_ga` need no change** (grain-agnostic; they consume whatever `cell_id` they're handed).

**4. prop_key / band projection → include pmp/ctry (unwind the broadcast — the linchpin).** `M`
`band_projection._prop_key:178` excludes pmp/ctry so one cell share maps onto every sub-cell row; the GA
prop-dict groupby (`tab2_engine.py:2955–2962`) collapses to `(cur,bin,[rpgt,]mid)`. Add a sub-cell variant
of `_prop_key` (include pmp/ctry) and key the prop dict at sub-cell grain, so the GA's per-sub-cell shares
are scored as-is instead of broadcast. This is the single structural assumption to break.

**5. build_split_exports → consume sub-cell shares (stop re-expanding).** `M`
`impact_calcs.py:1508–1519` takes one per-cell share and fans it across pmp/Country sub-rows. It must accept
a split already carrying `paymentMethodProvider`/`Country` and key its base-normalisation on the sub-cell
instead of re-expanding. Output schema already has the columns.

---

## Risks & unknowns (after the refinement)

- **Search size / perf (now the main one).** `n_cells` and genome length `R` grow up to ~6× (realistically
  ~1.6–3× — many pmp×Country combos are empty). Slower search, more memory; likely needs pop/generation
  retuning. Watch splits/s and total wall time.
- **Volume-split fidelity.** Splitting cell forecast volume to sub-cells by the export's VI fractions assumes
  the export's sub-cell mix is representative — worth a sanity check, but low risk (it's the deploy grain).
- **Compression (tab 5).** If pools/compression are on (`max_pools_target > 0`), the k-means step
  reintroduces its own scored↔delivered divergence — separate concern.
- **Infeasibility wall unchanged.** Grain doesn't create routing headroom that isn't there.

*(The success-rate sparsity risk from the earlier draft is gone — we're not splitting success rates.)*

---

## Recommended staging

- **Stage 1 — sub-cell scaffold + volume glue.** Item 3: build the sub-cell cell key, broadcast succ, split
  volume via pro-rata VI fractions. Verify per-sub-cell volumes sum back to cell totals and rates look sane.
- **Stage 2 — flip the decision grain + prop key.** Item 4: sub-cell `_prop_key` + prop dict. Verify
  scored == delivered on a small case (reconciliation test, same pattern as the timing/cap fixes).
- **Stage 3 — exporter + config-gen.** Item 5; full validation + the mandatory `__pycache__` clear + restart.

---

## What I can / can't validate

- **Can (in-container):** `py_compile`, synthetic unit tests, grain-threading + volume-split correctness, and
  a scored==delivered reconciliation against the real `compute_vamp_prepost_granular`.
- **Can't:** real convergence/perf at the larger genome, or config-gen against live BigQuery — needs your runs.

---

## Overall

A **medium project** across ~three files (`tab2_engine.py`, `band_projection.py`, `impact_calcs.py`), with
**no change to `vamp_forecast_pipeline` or `success_rates`**. The optimiser already scores VAMP and emits configs
at sub-cell grain; this makes the *decision* grain match — sub-cells differentiated by VAMP/eligibility, not
conversion. Biggest practical watch-item is now **search size/perf**, not data sparsity.
