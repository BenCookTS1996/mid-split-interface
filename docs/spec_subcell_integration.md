# Implementation spec — sub-cell decision grain (coupled change (a)–(e))

Single indivisible change: the GA *decides* at **sub-cell** = `currency × bin × rpgt × pmp × Country`
while *scoring conversion* stays at **cell** = `currency × bin × rpgt`. Gated behind a new grain option;
the existing cell-grain path stays byte-identical when the option is off.

## Design decision that unblocks it (the duplicate-key fix)

Make the **sub-cell the unique cell identity end-to-end**. Concretely:

- `bank` stays the raw **BIN** (so the band scaffold, keyed `bin/pmp/ctry` separately, still aligns).
- `pmp` and `ctry` are carried as **separate aligned fields** on `CellProblem` → `ref_agg` → `G`.
- The **cell key** (`G["cell"]`), `cell_starts`, the **band incidence**, the **prop_key**, and the
  **exporter** all key on the *sub-cell* `(cur,bin,rpgt,pmp,ctry[,gateway])` — **not** on
  `(bank,gateway)`.

Result: every `G` row is uniquely `(cur,bin,rpgt,pmp,ctry,gateway)` → **no duplicate keys**, so the
incidence and exporter stay unambiguous. Conversion (`succ`) is broadcast from cell grain; VAMP bands +
eligibility are already sub-cell-native; volume is apportioned by the pro-rata VI-fraction glue.

## Foundation (DONE + tested)

- `routing_optimiser.subcell` — `subcell_vi_fractions`, `expand_forecast_to_subcells` (volume glue).
- `CellProblem.pmp` / `.ctry` (defaulted `"_all_"`, backward-compatible).
- `routing_optimiser.data_loader.build_subcell_problems` — sub-cell assembler, rates broadcast.

## Ordered edits

**(a) `optimiser.optimise_split`** — emit `pmp`/`ctry` in the long frame.
`src/routing_optimiser/optimiser.py:463-501`: add `_c_pmp/_c_ctry` accumulators
(`p.pmp`, `p.ctry`), append per row, add `"pmp"`/`"ctry"` to the output dict. Backward-compatible
(existing CellProblems → `"_all_"`; existing consumers ignore the extra cols). **Testable now.**

**(d) `band_projection._prop_key` sub-cell variant** — add `by_subcell`.
`src/routing_optimiser/band_projection.py`: `_prop_key(df, by_rpgt, by_subcell=False)` →
`cur|bin|rpgt|pmp|ctry|mid` when `by_subcell`. Thread `by_subcell` through `_prop_raw`,
`PopulationBandProjector.__init__` (+`self.by_subcell`), `project_pop_from_props`, and `BandProjector`.
The scaffold `_GRPK` is already `cur|bin|rpgt|pmp|ctry|per`, so the sub-cell prop_key aligns row-for-row
(no broadcast). **Testable now** (reconciliation on a case with ≥2 sub-cells that differ).

**(b)+(c) `tab2_engine.py` assembly** — gated sub-cell branch. **Needs user runs.**
- Dropdown `:495`: add option `"Bank × Currency × RPGT × pmp × Country"`; set
  `_opt_subcell = (_opt_grain == that)`. (Leave Engine-Score-grain options as-is — scoring stays cell.)
- Pro-rata path `_ppf = out_dir/vamp_t_period_prorata_export.csv` is known at `:2179`; compute it (or hoist)
  before the `build_cell_problems` call at `:2015`.
- When `_opt_subcell`, before `build_cell_problems`:
  `sub_fc = expand_forecast_to_subcells(agg_forecast, subcell_vi_fractions(pd.read_csv(_ppf)))`
  then `agg_problems = build_subcell_problems(sub_fc, agg_sr)` (else the existing `build_cell_problems`).
- `optimise_split` now returns `ref_agg` with `pmp`/`ctry` (via (a)); `_mc = ref_agg.copy()` carries them.
- Cell key `:2203`: when `_opt_subcell`,
  `_mc["cell"] = cur|bank|rpgt|pmp|ctry` (else the current `cur|bank|rpgt`). `G["_cellk"]`/`cell_starts`
  (`:3459-3467`) then become sub-cell automatically.
- Incidence / prop-key build (`_get_pbp` and the incidence at `:3801-3815`): pass `by_subcell=_opt_subcell`
  so `prop_keys` and the column→key map are sub-cell; the map keys on `(cur,bin,rpgt,pmp,ctry,gateway)`.

**(e) `impact_calcs.build_split_exports`** — consume sub-cell shares. **Needs user runs.**
`:1368,1508-1519`: if the incoming split carries `pmp`/`Country`, **skip the cell→sub-cell expansion** and
use the rows as-is; base-normalise per `(rpgt,currency,BIN,pmp,Country)` sub-cell. Same for
`enforced_prop_items` / `enforced_split_frame`. (Tab-3 already keys the projection at sub-cell.)

## Validation

- **Now (in-container):** `py_compile`; unit test (a) (pmp/ctry columns present + `"_all_"` default);
  reconcile (d) (sub-cell prop_key projects two differing sub-cells distinctly, matches the oracle).
- **User runs (gated):** select the new grain; first checkpoint = the run ASSEMBLES — watch `cells`,
  `R`, and the **incidence self-check** (`Σprop_raw vs Σshare`, coverage %). Then scored vs tab-3
  delivered should converge (that's the payoff). Expect 2–3 debug rounds on the incidence/exporter keys.

## Risks / notes

- **Perf:** `n_cells`/`R` grow ~1.6–3× (sub-cell fan-out) → slower search; may need pop/gen retune.
- **Incidence coverage** is the canary: if the self-check shows dropped share mass after the switch, the
  column→prop-key map isn't keying on the sub-cell — fix there first.
- **Does not move the joint-infeasibility wall** (Braintree/WorldPay ~+8–11% over) regardless of grain.
- Everything is gated on `_opt_subcell`; cell-grain runs are unaffected.
