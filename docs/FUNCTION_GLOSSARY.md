# Function glossary

A plain-English reference to **every function** in the codebase. Each entry is `[FN-id]` ·
`signature` — **what it does**, plus an *analogy* for non-technical readers. The
**how-it-ties-in** context is the one-line role under each file heading. The same `[FN-id]`
tag sits in the source beside each function, so you can hop between here and the code.

_Total: 431 functions across 37 files · analogy coverage: 431/431._

---

## `app/app_common.py`
*How it ties in:* Shared constants, the live log handler, and small helpers every UI tab reuses.

- **[FN-233]** `StreamlitLogHandler.__init__(self, sink)` — (small helper — see source)  
  *Analogy:* Wires up the on-screen log panel so messages have somewhere to land.
- **[FN-234]** `StreamlitLogHandler.emit(self, record)` — (small helper — see source)  
  *Analogy:* Pins each new log line onto the on-screen panel as it arrives.
- **[FN-235]** `_switched_off_gateways(ov)` — Canonicalised, lower-cased gateway ids that are SWITCHED OFF in an already-loaded  
  *Analogy:* The list of doors currently bolted shut, in tidy standard spelling.
- **[FN-236]** `_variance_gap_temp(agg_sr, anchor, t_ceiling, n_cap)` — Per-Bank×Currency softmax temperature from the STATISTICAL SIGNIFICANCE of  
  *Analogy:* Reads how spread-out the doors' success rates are and sets how sharply to favour the best one.
- **[FN-237]** `_ink_caption(md)` — Render a caption in ink (near-black) rather than Streamlit's default grey.  
  *Analogy:* Prints small-print captions in near-black ink instead of the faded default grey.
- **[FN-238]** `_fmt_secs(s)` — Human-friendly duration, e.g. 45s, 2m 05s, 1h 12m.  
  *Analogy:* Turns a raw seconds count into '2m 05s' that a person can read.
- **[FN-239]** `_load_ga_perf()` — Load the last GA timing calibration from disk (survives restarts).  
  *Analogy:* Remembers how long the search took last time, so the ETA survives a restart.
- **[FN-240]** `_save_ga_perf(d)` — Persist the GA timing calibration so the estimate survives Streamlit restarts.  
  *Analogy:* Jots down this run's timing so next time's estimate is sharper.
- **[FN-241]** `_physical_cpu_count(default)` — Number of PHYSICAL CPU cores (excludes hyperthreads). The seed default uses this so the  
  *Analogy:* Counts the real engines under the hood, ignoring the pretend hyper-thread ones.
- **[FN-242]** `_apply_blocked_caps(split, blocked_pairs, floor)` — Cap the share of any BANK-BLOCKED (bank, gateway) to the exploration floor and redistribute  
  *Analogy:* Throttles any bank-blocked door down to a trickle, so we stop feeding a dead route.
- **[FN-243]** `_ensure_base_30d_metrics()` — Compute & cache the 30-day baseline metrics (profile/gateway success rates,  
  *Analogy:* Works out the last-30-days 'before' picture once and keeps it on hand.
- **[FN-244]** `_impact_eval_frame(split, cache, by_rpgt)` — Per-(rpgt, currency, bank, gateway) pre/post frame for a proposed split,  
  *Analogy:* Lays the 'before' and 'after' numbers side by side for one proposed plan.
- **[FN-245]** `_locked_panel(step_html)` — Calm centered placeholder for a results tab that has no run behind it yet.  
  *Analogy:* A calm 'nothing to show yet — run it first' holding screen.
- **[FN-246]** `_split_df_to_xlsx_bytes(rdf)` — Serialise one split DataFrame to .xlsx bytes for the export ZIP. Primary path uses  
  *Analogy:* Packs one split table into a downloadable Excel file.

## `app/impact_calcs.py`
*How it ties in:* The before→after impact maths + the production split-template builder behind Tabs 3-4.

- **[FN-247]** `build_kill_eff(vamp2fids, fid_eff)` — Build the hashable effective-date switch-off map for the projection.  
  *Analogy:* Builds the calendar of which doors close on which date.
- **[FN-248]** `_mid_keep_fraction(vampmid_series, period_series, kill_eff, month_0)` — Per-row RETAINED fraction (1 − kill) for effective-date-gated switch-offs.  
  *Analogy:* For each row, the slice of traffic that survives after the switch-offs.
- **[FN-249]** `compute_vamp_post_by_mid(tp_path, prop_items, month_0, go_live, excluded_mids, kill_eff, mtime)` — Derive the proposed-split VAMP forecast from the saved baseline export.  
  *Analogy:* Projects each account's future risk if the new plan goes live.
- **[FN-250]** `compute_vamp_post_by_mid._frac_after(m)` — (small helper — see source)  
  *Analogy:* The leftover fraction once a mid-month switch-off is applied.
- **[FN-251]** `compute_vamp_post_from_prorata(pp_path, prop_items, excluded_mids, kill_eff, month_0, scoped_rpgts)` — Accurate proposed-split VAMP forecast using the pipeline pro-rata export.  
  *Analogy:* The precise version of that risk projection, using the pipeline's own month-by-month engine.
- **[FN-252]** `_vamp_post_core(pp, prop_items, excluded_mids, kill_eff, month_0, scoped_rpgts)` — Core projection on a PRE-LOADED pro-rata dataframe, so the per-MID cap  
  *Analogy:* The projection's engine room, run on already-loaded data so it's quick.
- **[FN-253]** `_dump_projection_diag(t0, pp_path, prop_items, enforced, by_rpgt)` — EXCESSIVE diagnostics for the tab-3 vs tab-5 back-fill gap. Writes two files next to the  
  *Analogy:* A firehose of debug detail for chasing a specific mismatch between two tabs.
- **[FN-254]** `_dump_projection_diag._p()` — (small helper — see source)  
  *Analogy:* A little printer for the diagnostic dump.
- **[FN-255]** `_dump_projection_diag._kv(_s)` — (small helper — see source)  
  *Analogy:* Formats one 'name: value' line in the dump.
- **[FN-256]** `_dump_projection_diag._cached_df(_cache, _srcs, _build)` — (small helper — see source)  
  *Analogy:* Keeps a copy of the working table so the dump doesn't rebuild it.
- **[FN-257]** `_dump_projection_diag._build_rules()` — (small helper — see source)  
  *Analogy:* Reconstructs the routing rules for the diagnostic comparison.
- **[FN-258]** `_dump_projection_diag._build_map()` — (small helper — see source)  
  *Analogy:* Rebuilds the lookup the diagnostic needs.
- **[FN-259]** `_inject_backfill_rows(pp, prop)` — #3 ZERO-BASELINE BACK-FILL: build_split_exports can route to gateways (e.g. <2-gateway  
  *Analogy:* Adds placeholder rows for doors that had no history, so new traffic isn't lost.
- **[FN-260]** `compute_vamp_prepost_granular(pp_path, prop_items, excluded_mids, kill_eff, month_0, scoped_rpgts, wallet_incapable, usa_only, exploration_floor)` — Per-ROW baseline vs proposed VAMP / VI-Txn from the pro-rata export.  
  *Analogy:* The finest-grain before-vs-after risk table, row by row.
- **[FN-261]** `mid_table_from_granular(gran)` — Per-vampMid VAMP / VI-Txn M0–5 (pre & post) table, derived by AGGREGATING the  
  *Analogy:* Rolls those rows up into a per-account, month-by-month risk table.
- **[FN-262]** `process_wallet_incapable(mid_list_path)` — Set of gatewayFids (lowercased) that CANNOT process wallet (GOOGLEPAY /  
  *Analogy:* a guest list of which gateways can't accept Apple/Google Pay. We only strike a gateway off when the sheet EXPLICITLY says no (FALSE/0/NO); a blank is treated as "can", so we never wrongly ban a gateway on missing data.
- **[FN-263]** `_cache_data()` — (small helper — see source)  
  *Analogy:* A memory shelf so expensive results aren't recomputed.
- **[FN-264]** `_mtime(path)` — (small helper — see source)  
  *Analogy:* Reads a file's last-changed time, to tell when a cache has gone stale.
- **[FN-265]** `_c_read_parquet(path, m)` — (small helper — see source)  
  *Analogy:* Reads a data file once and remembers it.
- **[FN-266]** `_c_vamp_post_prorata(pp_path, m, prop_items, excluded_mids, kill_eff, month_0, scoped_rpgts)` — (small helper — see source)  
  *Analogy:* The cached wrapper around the precise risk projection.
- **[FN-267]** `_c_prepost_granular(pp_path, m, prop_items, excluded_mids, kill_eff, month_0, scoped_rpgts, wallet_incapable, usa_only, exploration_floor)` — (small helper — see source)  
  *Analogy:* The cached wrapper around the fine-grain before/after table.
- **[FN-268]** `build_split_exports(split, brand, go_live, wallet_incapable, fid2vamp, mid_list_path, usa_only, country_pres, max_share)` — Build the production template (one DataFrame per Brand×RPGT) from a split.  
  *Analogy:* Fills in the official deployment template — one sheet per brand and payment type.
- **[FN-269]** `build_split_exports._incap(gw)` — (small helper — see source)  
  *Analogy:* Marks the doors that can't serve wallet pay in the template.
- **[FN-270]** `build_split_exports._is_usa_only(gw)` — (small helper — see source)  
  *Analogy:* Flags the US-only doors so they aren't offered elsewhere.
- **[FN-271]** `build_split_exports._valid_candidates(cur_l, country, is_wallet)` — Template-column gateways eligible to serve this (currency, country, pmp): currency-  
  *Analogy:* Works out which doors are actually allowed to serve this row.
- **[FN-272]** `build_split_exports._cap_rows(V)` — VECTORISED twin of the old per-Series `_cap_shares`, applied to a whole (rows×gw)  
  *Analogy:* Trims every row's shares down to their caps in one sweep.
- **[FN-273]** `build_split_exports._countries_for(cur, bin_)` — (small helper — see source)  
  *Analogy:* Lists which countries a rule covers.
- **[FN-274]** `enforced_prop_items(split, brand, go_live, wallet_incapable, fid2vamp, mid_list_path, usa_only, country_pres, max_share)` — Proposed shares AFTER the pipeline's enforcement — cap, wallet-incapable zeroing,  
  *Analogy:* what the split looks like once it's passed through production's "rulebook" — the same caps, capability filters and safety back-fills the deployed config would apply — so the impact projection scores what will REALLY be routed, not the raw optimiser output.
- **[FN-275]** `enforced_split_frame(split, brand, go_live, wallet_incapable, fid2vamp, mid_list_path, usa_only, country_pres, max_share)` — Gateway-grain version of :func:`enforced_prop_items`.  
  *Analogy:* The same post-rulebook split, broken down door by door.
- **[FN-276]** `count_pools_for_split(split_long, brand_name, go_live)` — Number of ConnectorPool configs build_split_exports -> generate_configs would  
  *Analogy:* Counts how many deployable config files this split would generate.
- **[FN-277]** `pool_targeted_core(split_ideal)` — Pure (NO session_state) pool-count-targeting compression for `split_ideal`.  
  *Analogy:* The group-to-a-budget engine, with no UI strings attached.
- **[FN-278]** `_pool_disk_key(split_ideal)` — CONTENT hash of everything the compression output depends on: the split's own values  
  *Analogy:* A fingerprint of the inputs, so an unchanged compression is served from cache.
- **[FN-279]** `pool_targeted_compression(ss, split_ideal)` — Run (and cache in ss) the pool-count-targeting compression for `split_ideal`.  
  *Analogy:* Runs the group-to-budget squeeze and remembers the result.
- **[FN-280]** `rpgt_avg_ticket(profile_agg)` — RPGT-level average ticket from the 30D actuals (the window ending just before  
  *Analogy:* The typical payment size per type, from the last 30 days.
- **[FN-281]** `mid_revenue_month_table(granular, rpgt_ticket, months)` — Per-vampMid × month VI Txn + $Revenue (pre/post) from the pro-rata granular.  
  *Analogy:* A per-account, month-by-month tally of transactions and dollars, before vs after.

## `app/live_allocation.py` — DELETED (19ga)

Removed: nothing imported it, and the full-pipeline "Validate Split" sub-tab
(`app/tab_1_2_validate_split.py`) had superseded it. Recover from git history if the
"validate a split from the cached forecast, offline" idea is ever revived.


## `app/streamlit_app.py`
*How it ties in:* App entry point / orchestrator — sets up, creates the tabs, calls each tab's render().

- **[FN-285]** `_chart_title(text, container)` — Render a chart title ABOVE the chart (outside the plot area), so titles  
  *Analogy:* Prints a heading above a chart, outside the plotting area.
- **[FN-286]** `resolve_attempts(attempts_export, _key)` — Success-rate data (a ROUTING input, not the forecast).  
  *Analogy:* Tracks down the attempts data the routing engine needs.
- **[FN-287]** `bar(df, x, y, title, color)` — (small helper — see source)  
  *Analogy:* Draws a simple bar chart for a small inline figure.
- **[FN-288]** `_find_gcloud()` — Resolve the gcloud binary — PATH first, then common install locations. A GUI/venv-launched  
  *Analogy:* Hunts down the Google Cloud command-line tool wherever it's installed.
- **[FN-289]** `_query_table_refs(queries_dir)` — Best-effort scan of every .sql in `queries_dir` for fully-qualified BigQuery table refs  
  *Analogy:* Skims every query to list which database tables it touches.
- **[FN-290]** `_check_table_access(project, tables)` — DRY-RUN a `SELECT 1 FROM <table> LIMIT 0` per table (free — no bytes billed) to test that the  
  *Analogy:* Knocks on each table (for free) to check we're actually allowed in.
- **[FN-291]** `_run_preflight(project)` — Return [(label, status, detail, fix)]; status ∈ ok / warn / fail / skip.  
  *Analogy:* The pre-flight checklist — green/amber/red for each thing that must be ready.
- **[FN-292]** `_render_preflight(project)` — (small helper — see source)  
  *Analogy:* Puts that checklist up on screen.

## `app/tab_1_1_build_baseline.py`
*How it ties in:* Tab 1 UI — build/cache the baseline forecast and validate a split.

- **[FN-293]** `render()` — (small helper — see source)  
  *Analogy:* Draws the whole Baseline & Validate tab.
- **[FN-294]** `render._load_default_json(name)` — (small helper — see source)  
  *Analogy:* Loads the starting config so the form isn't blank.
- **[FN-295]** `render._fetch_m0(_co, _sch)` — (small helper — see source)  
  *Analogy:* Pulls the current-month starting numbers.
- **[FN-296]** `render._read_json(upload, default_path, label)` — (small helper — see source)  
  *Analogy:* Reads a config file into the form.
- **[FN-297]** `render.log(msg)` — (small helper — see source)  
  *Analogy:* Writes a line to this tab's on-screen log.

## `app/tab_2_routing_engine.py`
*How it ties in:* Tab 2 UI — pick an engine + limits and run the search.

- **[FN-298]** `render()` — (small helper — see source)  
  *Analogy:* Draws the whole Routing-engine tab.
- **[FN-299]** `render._budget_panel()` — (small helper — see source)  
  *Analogy:* Shows the time/effort budget for the search and its live ETA.
- **[FN-300]** `render._budget_panel._fmt_eta(_secs)` — (small helper — see source)  
  *Analogy:* Turns the remaining time into a readable countdown.
- **[FN-301]** `render._ratio(cur, prev, lo, hi)` — (small helper — see source)  
  *Analogy:* Works out a simple share/ratio for display.
- **[FN-302]** `render._eng(fr)` — (small helper — see source)  
  *Analogy:* Builds the chosen engine, ready to run.
- **[FN-303]** `render._progress(frac, label)` — (small helper — see source)  
  *Analogy:* Nudges the progress bar along.
- **[FN-304]** `render.log(msg)` — (small helper — see source)  
  *Analogy:* Writes a line to the engine tab's log.
- **[FN-305]** `render._stage(name)` — (small helper — see source)  
  *Analogy:* Announces the start of a named step in the run.
- **[FN-306]** `render._stage_end()` — (small helper — see source)  
  *Analogy:* Marks that step finished, and how long it took.
- **[FN-307]** `render._diag(msg)` — Verbose diagnostic line (same sink as log). Wrapped so a diagnostics  
  *Analogy:* Writes a verbose debug line when detailed tracing is switched on.
- **[FN-308]** `render._gv(name, default)` — (small helper — see source)  
  *Analogy:* Fetches a stored setting, with a fallback if it's missing.
- **[FN-309]** `render._bmark(modpath)` — (small helper — see source)  
  *Analogy:* Times a chunk of work for the diagnostics.
- **[FN-310]** `render._finfo(p)` — (small helper — see source)  
  *Analogy:* Reports a file's size and date for the log.
- **[FN-311]** `render._shape(df, name)` — (small helper — see source)  
  *Analogy:* Prints a table's dimensions as a quick sanity check.
- **[FN-312]** `render._q(a, x)` — (small helper — see source)  
  *Analogy:* Reads a percentile off a list of numbers (the median, the 90th, and so on).
- **[FN-313]** `render._explode(agg_split)` — (small helper — see source)  
  *Analogy:* Fans a parent-bank profile back out to its individual BIN-level banks.
- **[FN-314]** `render._scope_base(mid, rpgt, month)` — (small helper — see source)  
  *Analogy:* Narrows the data down to one account (and optionally one payment-type or month).
- **[FN-315]** `render._rule_a(_tg, _tl, _mtr, _bt, _bv)` — (small helper — see source)  
  *Analogy:* Works out one account's cap target from its transaction/VAMP settings.
- **[FN-316]** `render._prio_mult(_p)` — (small helper — see source)  
  *Analogy:* Turns a rule's priority rank into a weight — higher priority pulls harder.
- **[FN-317]** `render._project_capped(prop_items, _use_cache)` — (small helper — see source)  
  *Analogy:* Runs a split forward while keeping every door under its cap.
- **[FN-318]** `render._mids_over(shares)` — (small helper — see source)  
  *Analogy:* Lists which accounts are over their limit.
- **[FN-319]** `render._summ_from_shares(shares)` — (small helper — see source)  
  *Analogy:* Boils a split down to its headline success and risk numbers.
- **[FN-320]** `render._restrict(spl)` — (small helper — see source)  
  *Analogy:* Applies the eligibility rulebook to a split here in the tab.
- **[FN-321]** `render._mid_cap_granular(gran)` — (small helper — see source)  
  *Analogy:* Enforces each account's cap at the finest grain.
- **[FN-322]** `render._blend_ga(prop)` — (small helper — see source)  
  *Analogy:* Blends the search's result with the backup doors.
- **[FN-323]** `render._prop_items_from_gran(gran)` — (small helper — see source)  
  *Analogy:* Reads the proposed shares out of the fine-grain table.
- **[FN-324]** `render._band_frames()` — Adapter (T0a, Pca, pool, sorted band-set, by_rpgt) for the collapse  
  *Analogy:* Repackages the data into the shape the band-projector expects.
- **[FN-325]** `render._band_collapse_diag(prop_items, label)` — (small helper — see source)  
  *Analogy:* A debug readout of how far the band scaffold collapsed.
- **[FN-326]** `render._band_cost_probe()` — Measure the REAL reduced-scaffold size + one population project_pop time,  
  *Analogy:* Measures the real size and speed of one band projection.
- **[FN-327]** `render._band_compress_probe()` — How far could the EXACT projection scaffold shrink losslessly? The candidate  
  *Analogy:* Tests how much smaller the projection scaffold could safely get.
- **[FN-328]** `render._get_pbp()` — Build & cache the population band projector on this run's real scaffold.  
  *Analogy:* Builds and caches the population band-projector for this run.
- **[FN-329]** `render._band_slope_probe(ref_prop_items, end_prop_items)` — Does a FIRST-ORDER (slope) band model stay accurate over the search's ACTUAL  
  *Analogy:* Checks whether a quick straight-line risk model is accurate enough.
- **[FN-330]** `render._band_slope_probe._col(_ai)` — (small helper — see source)  
  *Analogy:* Handles one column inside that slope check.
- **[FN-331]** `render._band_enabler_probes(ref_prop_items, end_prop_items)` — Measure the two enablers of the 'near-exact + incredibly fast' path:  
  *Analogy:* Measures the two tricks that make the fast-but-accurate band model possible.
- **[FN-332]** `render._band_enabler_probes._val(vmat, tmat, border, row, mkl, mo, mtr)` — (small helper — see source)  
  *Analogy:* Reads one measured value in that probe.
- **[FN-333]** `render._band_enabler_probes._timeit(proj, P, reps)` — (small helper — see source)  
  *Analogy:* A stopwatch for one probe step.
- **[FN-334]** `render._band_enabler_probes._ck(df)` — (small helper — see source)  
  *Analogy:* A quick correctness check inside the probe.
- **[FN-335]** `render._mids_over_granular(gran)` — (small helper — see source)  
  *Analogy:* Lists over-limit accounts at the finest grain.
- **[FN-336]** `render._mids_over_blended(gran)` — (small helper — see source)  
  *Analogy:* Lists over-limit accounts after the backup blend.
- **[FN-337]** `render._colnum(_df, _name)` — (small helper — see source)  
  *Analogy:* Reads a column as numbers, or fills zeros if it's missing.
- **[FN-338]** `render._build_ga_bands(_anchor)` — Month-specific per-MID bands for the GA fitness, CALIBRATED so the  
  *Analogy:* Sets each account's monthly speed-limits for the search to respect.
- **[FN-339]** `render._ga_true_breach(_sh)` — Total RELATIVE band breach of the TRUE pro-rata projection for `_sh`  
  *Analogy:* The honest total by which a plan busts its limits, using the precise projection.
- **[FN-340]** `render._ga_solve_with_correction(_risk_min_w, _seed, _rounds, _band_w, _warm, _band_fix, _ref_gamma, _n_fine, _n_restarts)` — Run the tilt GA, then RE-PROJECT & CORRECT (like the greedy): re-anchor  
  *Analogy:* Runs the search, then re-checks and nudges the result to stay truly within limits.
- **[FN-341]** `render._ga_solve_with_correction._log_seed(_idx, _infoc, _t0, _best_holder)` — Verbose one-line summary for a finished seed (best-effort).  
  *Analogy:* A one-line summary of a finished search attempt.
- **[FN-342]** `render._ga_solve_with_correction._poll_progress(_t0, _emit, _nseed)` — Sum the per-seed progress files and surface the live count in BOTH  
  *Analogy:* Adds up all the parallel workers' progress for the live bar.
- **[FN-343]** `render._ga_solve_with_correction._run_par(_pk, _tasks, _box)` — (small helper — see source)  
  *Analogy:* Fires off the search across several cores at once.
- **[FN-344]** `render._agg_mid_ok(_sh)` — (small helper — see source)  
  *Analogy:* Confirms every account ends up within its limit.
- **[FN-345]** `render._riskmin_key()` — (small helper — see source)  
  *Analogy:* The tie-breaker that picks the lowest-risk plan when scores are level.
- **[FN-346]** `render._endpoint_agg(_shares)` — (small helper — see source)  
  *Analogy:* Aggregates the final numbers at the slider's endpoint.
- **[FN-347]** `render._log_pc(_w, _sta)` — (small helper — see source)  
  *Analogy:* Logs the projected-vs-cap detail for each profile.

## `app/tab_3_split_outputs_impact.py`
*How it ties in:* Tab 3 UI — show the split, its before→after risk/revenue, and dashboards.

- **[FN-348]** `render()` — (small helper — see source)  
  *Analogy:* Draws the whole Split, outputs & impact tab.
- **[FN-349]** `render._run_pool_compression()` — Compute + cache the pool-targeted compression for the CURRENT settings.  
  *Analogy:* Squeezes the split to the target rule-count and caches it.
- **[FN-350]** `render._enforced_blended_eval_split(_spl)` — Enforced (build_split_exports) + backup-blended gateway-grain split, at the  
  *Analogy:* The split as it would deploy — through the rulebook and with backups blended in.
- **[FN-351]** `render._rcard(col, label, big, big_color, small, tip)` — (small helper — see source)  
  *Analogy:* Draws one summary card.
- **[FN-352]** `render._export_ui()` — (small helper — see source)  
  *Analogy:* The download panel for saving the outputs.
- **[FN-353]** `render._export_ui._gather(_split_df, _subdir, _prefix)` — (small helper — see source)  
  *Analogy:* Collects every file to bundle for download.
- **[FN-354]** `render._fin_render_share_chart(_evframe, _target)` — Before → after volume-share chart at vampMid grain, rendered inside Financial Impact.  
  *Analogy:* Draws the before→after volume-share chart, per account.
- **[FN-355]** `render._rev_bridge_waterfall(pre, post, names, deltas, money, pct, wide_min, height, tick_size, left_margin)` — (small helper — see source)  
  *Analogy:* The waterfall showing what pushes revenue up or down between before and after.
- **[FN-356]** `render._rev_bridge_waterfall._fmt(_v)` — (small helper — see source)  
  *Analogy:* Formats a number for the revenue bridge.
- **[FN-357]** `render._bridge_items(_df, _name_col, _pre_col, _post_col, _sort_mode, _other, _max_n)` — Pick the movers for a bridge given a sort mode; roll the rest into one bar.  
  *Analogy:* Picks the biggest movers to show on the bridge.
- **[FN-358]** `render._bank_detail_fragment()` — (small helper — see source)  
  *Analogy:* The expandable per-bank detail block.
- **[FN-359]** `render._bank_detail_fragment._fmt_cell(col, v)` — (small helper — see source)  
  *Analogy:* Formats one table cell (an HTML table cell, not a routing profile).
- **[FN-360]** `render._bank_detail_fragment._bcw(_c)` — (small helper — see source)  
  *Analogy:* Sets a column's padding and font size in the bank table.
- **[FN-361]** `render._bank_detail_fragment._bank_row_html(r, is_total)` — (small helper — see source)  
  *Analogy:* Builds one styled table row for a bank.
- **[FN-362]** `render._idx(opts, val)` — (small helper — see source)  
  *Analogy:* Finds where the current choice sits in a dropdown (or defaults to the first).
- **[FN-363]** `render._style(fig)` — (small helper — see source)  
  *Analogy:* Applies colours and formatting to a table.
- **[FN-364]** `render._lum_sr(_hex)` — (small helper — see source)  
  *Analogy:* Picks black-or-white text so it stays readable on any profile colour.
- **[FN-365]** `render._score_of(_g)` — (small helper — see source)  
  *Analogy:* Reads a plan's headline score for display.
- **[FN-366]** `render._sr_bridge(_grp_col, _other)` — (small helper — see source)  
  *Analogy:* The waterfall for what moves the approval rate.
- **[FN-367]** `render._md_detail_fragment()` — (small helper — see source)  
  *Analogy:* The expandable per-account detail block.
- **[FN-368]** `render._md_detail_fragment._wrap(_s, _w)` — (small helper — see source)  
  *Analogy:* Wraps content in its display box.
- **[FN-369]** `render._md_detail_fragment._impact_bg(_col)` — (small helper — see source)  
  *Analogy:* Colours a profile by how big the change is.
- **[FN-370]** `render._spread_fig(_col, _unit)` — (small helper — see source)  
  *Analogy:* Draws the spread/scatter figure.
- **[FN-371]** `render._conv_df(h)` — (small helper — see source)  
  *Analogy:* Builds the conversion table behind a chart.
- **[FN-372]** `render._mk_conv(h, title)` — (small helper — see source)  
  *Analogy:* Draws the conversion chart.
- **[FN-373]** `render._mk_viol(h, title)` — (small helper — see source)  
  *Analogy:* Draws the violations chart.
- **[FN-374]** `render._parent(b)` — (small helper — see source)  
  *Analogy:* Finds a BIN's parent bank.
- **[FN-375]** `render._S(_name, _d)` — (small helper — see source)  
  *Analogy:* Reads a column as numbers, defaulting to zero when it's absent.
- **[FN-376]** `render._prepost_render(mode)` — (small helper — see source)  
  *Analogy:* Draws the before-vs-after comparison section.
- **[FN-377]** `render._prepost_render._optsel(col, container, label, def_val)` — (small helper — see source)  
  *Analogy:* The picker for which variation to view.
- **[FN-378]** `render._prepost_render._dfmt(_c, _v)` — (small helper — see source)  
  *Analogy:* Formats a number for the before/after table.
- **[FN-379]** `render._prepost_render._ltfmt(_c, _v)` — (small helper — see source)  
  *Analogy:* Formats a label/large number for the table.
- **[FN-380]** `render._prepost_render._ltcw(_c)` — (small helper — see source)  
  *Analogy:* Sizes a column in that table.
- **[FN-381]** `render._prepost_render._stacked_rpgt_fig(_fc, _act_rp, _order_labels, pct)` — (small helper — see source)  
  *Analogy:* Draws the stacked-by-payment-type chart.
- **[FN-382]** `render._prepost_render._build_tsfig(pct)` — (small helper — see source)  
  *Analogy:* Builds the month-by-month time-series chart.
- **[FN-383]** `render._normfid(x)` — (small helper — see source)  
  *Analogy:* Standardises a door's name so two spellings match.
- **[FN-384]** `render._proj_metric(_mid, _month, _metric)` — (small helper — see source)  
  *Analogy:* Shows one projected headline metric.
- **[FN-385]** `render._fmt(v)` — (small helper — see source)  
  *Analogy:* A number formatter for the tab.

## `app/tab_4_generate_configs.py`
*How it ties in:* Tab 4 UI — compress to a pool budget and generate/download the JSON configs.

- **[FN-386]** `render(ss, PROJECT_ROOT)` — Render the config-generation tab.  
  *Analogy:* Draws the config-generation tab — turns the chosen split into deployable files.
- **[FN-387]** `render._pool_bins(_pool)` — All BINs referenced by a pool's card.bin matching-rule expressions.  
  *Analogy:* Lists every card-BIN a pool's rule covers.
- **[FN-388]** `render._pool_bins._walk(o)` — (small helper — see source)  
  *Analogy:* Crawls the rule expression to collect its BINs.

## `app/tab_1_2_validate_split.py`
- **[FN-389]** `_covered_rpgts(merged_dir)` — Lower-cased set of RPGTs that HAVE a rule (from the RPGT column of each rule  
  *Analogy:* Lists which payment types actually have a routing rule written for them.
- **[FN-390]** `_to_prepost(df)` — Rename the pipeline's mid_level.csv columns onto the tab-3 table names.  
  *Analogy:* Relabels the pipeline's columns onto the before/after names the table expects.
- **[FN-391]** `_render_prepost_table(vp, fit_content, bold)` — Same red-header / month-spacer / TOTAL-row table tab 3 uses.  
  *Analogy:* Draws the red-header before/after table with its totals row.
- **[FN-392]** `_read_export_manifest(rules_dir)` — Read _export_manifest.json (the drift-guard stamp) from an exported rules folder.  
  *Analogy:* Reads the little stamp recording when the rules were last exported.
- **[FN-393]** `_drift_check(ss, rules_dir)` — Warn if the rule files in `rules_dir` were exported for a DIFFERENT split than the latest  
  *Analogy:* Warns if the deployed rule files are older than the latest export — a 'these may be stale' alarm.
- **[FN-394]** `_stage_rules(rules_dir, merged_dir)` — Copy every rule file from the exported rules folder into a clean staging dir the  
  *Analogy:* Copies the exported rule files into place, ready to validate.
- **[FN-395]** `render(ss, PROJECT_ROOT, GCP_PROJECT)` — (small helper — see source)  
  *Analogy:* Draws the whole Validate tab.
- **[FN-396]** `render._d(key, fallback)` — (small helper — see source)  
  *Analogy:* A small display helper for the Validate tab.
- **[FN-397]** `render._log(msg)` — (small helper — see source)  
  *Analogy:* Writes a line to the Validate tab's log.
- **[FN-398]** `render._H.emit(self, rec)` — (small helper — see source)  
  *Analogy:* Pins a log line onto the Validate tab's panel.

## `src/routing_optimiser/backup_blend.py`
- **[FN-001]** `_norm(s)` — (small helper — see source)  
  *Analogy:* Rescales a set of shares so they add back up to 100%.
- **[FN-002]** `parse_backup_catchall(backup_dir, rpgt_filter)` — Read every rule file in ``backup_dir`` and return the CATCH-ALL (BIN=Other/All)  
  *Analogy:* Reads the 'catch-all' fallback routes from the rule files.
- **[FN-003]** `blend_profile_shares(specific, catchall)` — Reproduce the pipeline's effective per-profile routing shares.  
  *Analogy:* Works out the real routing once the always-on backup doors are mixed in.
- **[FN-004]** `_catchall_by_vampmid(catchall_profile, fid2vamp)` — Map a profile's catch-all {gatewayFid: pct} onto {vampMid: pct} (summing fids that  
  *Analogy:* Re-expresses the fallback routes in per-account terms.
- **[FN-005]** `blend_prop_items(prop_items, catchall, fid2vamp, by_rpgt)` — Blend the backup catch-all into a projection's prop_items so the projected POST  
  *Analogy:* Folds the fallback routes into a proposed split before projecting its impact.
- **[FN-006]** `blend_prop_items._pooled(cur, rpgt)` — (small helper — see source)  
  *Analogy:* Handles the shared/pooled portion of that blend.
- **[FN-007]** `parse_rules_to_split(rules_dir)` — Reconstruct a proposed-split DataFrame from a folder of exported rule files.  
  *Analogy:* Reverse-engineers a split table back out of a folder of deployed rule files.

## `src/routing_optimiser/band_projection.py`
*How it ties in:* The exact 'what will the risk really be?' projection the per-MID monthly targets are scored against.

- **[FN-008]** `_njit()` — (small helper — see source)  
  *Analogy:* A turbo switch: bolts on the fast compiler if it's installed, otherwise runs the function plain.
- **[FN-009]** `_njit._deco(f)` — (small helper — see source)  
  *Analogy:* The clamp that actually fastens the turbo kit onto the function.
- **[FN-010]** `_pop_band_kernel(prop_raw, propidx, masked, gcode, base, mv_s, vcpos, ctot, pc_org, pc_vc, pc_pool, pc_band, cap_row, cap_band, nprofile, nband, vamp, txn, psum, vpsum, moved, pr, pshare, vshare, mvrow)` — Bit-identical numba equivalent of PopulationBandProjector.project_pop: flat passes over  
  *Analogy:* The engine-room version of the band maths — same numbers, whole population in one fast sweep.
- **[FN-011]** `_prop_key(df, by_rpgt)` — Build each row's bucket address ('cur|bin|mid', or 'cur|bin|rpgt|mid' when by_rpgt).  
  *Analogy:* Writes each row's postal address (currency|bin|mid) so proposals file into the right pigeonhole.
- **[FN-012]** `_prop_raw(T0, prop, by_rpgt)` — Look up each t0 row's proposed share from the `prop` dict by its bucket key.  
  *Analogy:* Reads each row's proposed share off the plan by looking up its address.
- **[FN-013]** `_static(T0)` — Precompute the per-row pieces that DON'T depend on the candidate (done once).  
  *Analogy:* Prepping all the ingredients that never change, once, before the cooking starts.
- **[FN-014]** `_origin_map(T0, Pc)` — Each aged Pc row -> its ORIGIN t0 row index (om==per), excluding back-fill; -1 if none.  
  *Analogy:* A family-tree pointer — links each 'aged' month back to the original month it grew from.
- **[FN-015]** `_shares(T0, prop, by_rpgt, gcode, ngc, base)` — Return (pshare, vshare, psum) for the candidate. `psum` is the per-row (broadcast  
  *Analogy:* For one candidate plan, works out how traffic and value spread across the doors.
- **[FN-016]** `project_reference(T0, Pc, pool, prop, by_rpgt)` — Faithful re-implementation of `_project_capped`'s array math (a readable oracle).  
  *Analogy:* the two-cohort model that ALL three projectors here share: split each MID's volume into a HELD cohort (stays put) and a MOVED cohort (the movable fraction mv=pr·fcp, re-routed across gateways by the candidate's proposed shares). In a profile the candidate leaves empty (psum==0) nothing can move, so the held cohort keeps 100%. The final band value = held volume + the candidate's slice of the redistributed moved pool.
- **[FN-017]** `BandProjector.__init__(self, T0, Pc, pool, bands, by_rpgt)` — (small helper — see source)  
  *Analogy:* Loads the projector with the fixed scaffolding it reuses for every plan.
- **[FN-018]** `BandProjector.project(self, prop)` — (small helper — see source)  
  *Analogy:* Runs one plan forward in time and reports the resulting risk and volume per band.
- **[FN-019]** `PopulationBandProjector.__init__(self, T0, Pc, pool, bands, by_rpgt)` — (small helper — see source)  
  *Analogy:* The same projector, geared to push a whole crowd of plans through at once.
- **[FN-020]** `PopulationBandProjector.project_pop_from_props(self, props)` — (small helper — see source)  
  *Analogy:* Takes a batch of plans described as look-up tables and projects them all.
- **[FN-021]** `PopulationBandProjector._nb_arrays(self)` — Cast static arrays to the numba kernel's dtypes once; pre-filter excl txn rows  
  *Analogy:* Repackaging the ingredients into the exact tins the fast engine expects — done once.
- **[FN-022]** `PopulationBandProjector._nb_buffers(self, P)` — Pre-allocated working buffers for the numba kernel, cached & REUSED across calls  
  *Analogy:* Keeping reusable mixing bowls ready so the fast engine never stops to fetch new ones.
- **[FN-023]** `PopulationBandProjector.project_pop_numba(self, prop_raw)` — Numba-accelerated project_pop — bit-identical, ~7× faster on the real scaffold.  
  *Analogy:* The turbo lane — same answer as the plain version, about 7x quicker.
- **[FN-024]** `PopulationBandProjector._profilesum(self, x)` — (P, nR) -> (P, ngc) segment sum over profile codes via sparse matmul (C-fast).  
  *Analogy:* Totting up each profile's share by sliding a stencil across the rows.
- **[FN-025]** `PopulationBandProjector.project_pop(self, prop_raw)` — prop_raw : (P, K) proposed share per `prop_keys`. Returns (vamp[P,B], txn[P,B]).  
  *Analogy:* The main assembly line: feed in every plan, get back each one's risk and volume per band.

## `src/routing_optimiser/band_scoring.py`
*How it ties in:* Scores how far a candidate breaks each MID's monthly target, fast, for the search.

- **[FN-026]** `build_col_incidence(col_propkeys, prop_keys)` — Build the sparse (K × N) 0/1 lookup that rolls GA columns up to projector rows.  
  *Analogy:* A wiring diagram that rolls the engine's fine-grained columns up into the projector's rows.
- **[FN-027]** `shares_to_prop_raw(shares, incidence)` — (P, N) decoded shares → (P, K) prop_raw = (incidence @ shares.T).T (sparse-safe).  
  *Analogy:* Translates the engine's split into the projector's language with one matrix multiply.
- **[FN-028]** `ExactBandPenalty.__init__(self, projector, specs)` — (small helper — see source)  
  *Analogy:* Loads the referee with the band limits it will police.
- **[FN-029]** `ExactBandPenalty._pen(self, overshoot)` — The fine schedule for being `overshoot` fraction over a band (0 = at/under the limit).  
  *Analogy:* The fine schedule — how big a penalty you pay for how far you overshoot a limit.
- **[FN-030]** `ExactBandPenalty.project(self, prop_raw)` — Project the population to exact per-band VAMP/Txn values (numba path by default).  
  *Analogy:* Works out each plan's exact per-band risk and volume before judging it.
- **[FN-031]** `ExactBandPenalty.penalty(self, prop_raw)` — (P, K) prop_raw → (P,) total band violation to add to each candidate's `viol`.  
  *Analogy:* Adds up how much each plan breaks the band limits, as a single penalty number.

## `src/routing_optimiser/config_generator.py`
- **[FN-032]** `_currency_expr(currency)` — (small helper — see source)  
  *Analogy:* Writes the 'match this currency' clause of a rule.
- **[FN-033]** `_scheme_expr(scheme)` — (small helper — see source)  
  *Analogy:* Writes the 'match this card scheme' clause of a rule.
- **[FN-034]** `_pmp_expr(pmp)` — (small helper — see source)  
  *Analogy:* Writes the 'match this wallet provider' clause of a rule.
- **[FN-035]** `_make_pool(rule, gateway_cols, brand, scheme)` — (small helper — see source)  
  *Analogy:* Assembles one routing rule (a 'pool') from its parts.
- **[FN-036]** `build_configs(compressed, brand, scheme)` — Return {rpgt: [pool, ...]} ready to serialise.  
  *Analogy:* Builds the full set of rules, grouped by payment type.
- **[FN-037]** `write_configs(configs, outdir, brand, date)` — (small helper — see source)  
  *Analogy:* Saves those rules out as the deployable JSON files.

## `src/routing_optimiser/connector_pool_configs.py`
- **[FN-038]** `company_to_brand_key(company)` — Map a company display name to a BRANDS key (defaults to 'tav').  
  *Analogy:* Maps a company's display name to its internal brand code.
- **[FN-039]** `get_priority(term, is_apgp, has_bins, extra_priority_amount)` — (small helper — see source)  
  *Analogy:* Reads a connector's priority ranking.
- **[FN-040]** `get_combinable_provider_sets(providers)` — Determines which providers can be safely combined into a single pool.  
  *Analogy:* Works out which providers can safely share one rule.
- **[FN-041]** `rows_from_dataframe(df, brand_name)` — Replaces the script's ``parse_sheet``: turn a Compressed_Rules template  
  *Analogy:* Turns the compressed split table into the rows the config builder reads.
- **[FN-042]** `make_pool(cfg, name, priority, currencies, bins, providers, scheme_filter, type_selectors, connectors_weighted, country_label)` — (small helper — see source)  
  *Analogy:* Builds one production ConnectorPool entry.
- **[FN-043]** `normalize_weights(connectors, expected_total)` — (small helper — see source)  
  *Analogy:* Rescales a pool's weights so they add up cleanly.
- **[FN-044]** `process_compressed_rows(cfg, rows, scheme_filter, count_only)` — BIN-specific pools (bin != 'Other'). Returns {name: pool}.  
  *Analogy:* Builds the pools for specific card-BINs.
- **[FN-045]** `process_backup_rows(cfg, rows, scheme_filter, count_only)` — Catch-all pools (bin == 'Other'). Returns {name: pool}.  
  *Analogy:* Builds the catch-all fallback pools for everything else.
- **[FN-046]** `emit_pool_generic(cfg, pools)` — pool-generic: every connector seen across `pools`, weight 100, priority 1.  
  *Analogy:* Emits a generic pool listing every connector it saw.
- **[FN-047]** `generate_configs(exports, brand_key, date, scheme, mode, extra_priority_amount, emit_generic, count_only)` — Generate ConnectorPool configs from the export templates.  
  *Analogy:* The config factory — export templates in, deployable ConnectorPool files out.

## `src/routing_optimiser/data_loader.py`
*How it ties in:* Loads the baseline forecast + rates and builds one problem per profile for the engines.

- **[FN-048]** `synthesise_forecast_from_success(success_df, default_risk)` — Build a plausible baseline forecast from the attempts data.  
  *Analogy:* When no official forecast exists, sketches a believable one from recent attempt history — a stand-in until the real numbers arrive.
- **[FN-049]** `load_forecast(path, success_df)` — Load the baseline 'pre' forecast — from a file, a pipeline output directory, or (when  
  *Analogy:* Fetches how things are routed today, from whichever source has it (a file, a pipeline run, or a folder).
- **[FN-050]** `build_profile_problems(forecast, success_rates, default_risk)` — Join forecast volume + baseline split with success/risk rates per profile.  
  *Analogy:* assembling each profile's "briefing pack". For every RPGT×Currency×Bank profile we pull the forecast's volume + current split together with each gateway's success rate, risk rate and evidence, and hand the engine one ProfileProblem it can solve. Gateways with no per-profile attempts fall back to the pooled prior (flagged so the UI can show which are educated guesses rather than measured rates).
- **[FN-051]** `build_profile_problems._nk(x)` — (small helper — see source)  
  *Analogy:* Tidies each label to a standard spelling so two data sources line up when joined.
- **[FN-052]** `prepare_inputs(success_source, forecast_path, shrink_strength)` — Convenience: load everything and return (problems, success_rates, forecast).  
  *Analogy:* The one-stop loader — grabs every input and hands back a ready-to-solve bundle.

## `src/routing_optimiser/eligibility.py`
*How it ties in:* The hard yes/no route rules: bans, wallet-incapable, USA-only.

- **[FN-053]** `load_usa_only(path)` — Explicit list of gatewayFids that can ONLY process country='USA'.  
  *Analogy:* The list of doors that can only handle US-domestic traffic.
- **[FN-054]** `load_explore_gateways(path)` — gatewayFids to treat as ELIGIBLE candidates even with no 30-day attempts, so  
  *Analogy:* The 'give them a chance' list — new doors allowed in even before they've built a track record.
- **[FN-055]** `load_restrictions(path)` — Load and normalise ban rules. Missing/invalid file -> no rules.  
  *Analogy:* Reads the rulebook of which doors are banned for which traffic.
- **[FN-056]** `_resolve_field(field, profile)` — Value for a rule field, aliasing 'bin' onto the 'bank' column (BIN-level  
  *Analogy:* Looks up the column a rule refers to, treating 'bin' and 'bank' as the same thing.
- **[FN-057]** `_row_banned(gw, vmid, profile, rules)` — True if any rule bans this gateway/vampMid for this traffic profile.  
  *Analogy:* Checks one door against the rulebook — is it barred for this payment?
- **[FN-058]** `_rules_signature(rules)` — (small helper — see source)  
  *Analogy:* A fingerprint of the rulebook, so an identical rulebook can be recognised and its answers reused.
- **[FN-059]** `_banned_mask_cached(df, rules, prof_cols)` — (small helper — see source)  
  *Analogy:* Remembers who's banned for a given rulebook so it isn't recomputed every time.
- **[FN-060]** `unenforceable_fields(rules, available_cols)` — Match-fields referenced by rules that can't be enforced at this grain  
  *Analogy:* Flags rules that mention details this data can't actually check — honest about the rulebook's blind spots.
- **[FN-061]** `_renorm(df, group_keys, col)` — Renormalise `col` to sum 1 within each group (leaves all-zero groups).  
  *Analogy:* After removing banned doors, rebalances the rest back to 100% (leaving fully-blocked groups at zero).
- **[FN-062]** `_capability_blend(df, group_cols, incapable, frac_map, default)` — Volume-weighted capability blend, returning the new per-row share array.  
  *Analogy:* an `incapable` gateway is like a vendor that can't take a certain payment type. It keeps only the (1 − frac) share it CAN serve; the `frac` portion of each profile is handed to the vendors that CAN (renormalised among themselves), so no transactions are lost. Used identically for wallet capability (frac = the profile's wallet share) and country capability (frac = the profile's Non-USA share). `frac_map` is keyed by (currency, bank); `default` is used when a profile isn't in the map.
- **[FN-063]** `apply_restrictions(split, rules, fid2vamp, wallet_incapable, wallet_frac, wallet_default, usa_only, nonusa_frac, nonusa_default, group_keys)` — Return the split with bans + wallet capability + country capability enforced.  
  *Analogy:* The bouncer — removes every barred or incapable door and rebalances the rest.
- **[FN-064]** `build_elig_operator(profiles, rules, fid2vamp)` — Precompute static per-row eligibility arrays for a FIXED layout.  
  *Analogy:* Pre-builds the guest list once for a fixed layout, so every plan can be checked instantly.
- **[FN-065]** `build_elig_operator._incap_mask(incapable)` — (small helper — see source)  
  *Analogy:* Marks which doors can't handle this payment type.
- **[FN-066]** `build_elig_operator._wf(frac_map, default)` — (small helper — see source)  
  *Analogy:* Works out how much of each row's traffic is wallet pay that needs re-routing.
- **[FN-067]** `_renorm_pop(X, cs, cc)` — Per-profile renormalise to sum 1, leaving all-zero profiles (matches `_renorm`).  
  *Analogy:* The rebalance-to-100% step, done for a whole batch of plans at once.
- **[FN-068]** `_blend_pop(X, incap, wf, cs, cc)` — Vectorised twin of `_capability_blend` + its trailing `_renorm`, over a population.  
  *Analogy:* The wallet-shift step, applied to a whole batch of plans at once.
- **[FN-069]** `apply_elig_pop(X, op)` — Apply the prebuilt eligibility operator to shares X ((N,) or (P, N)). Reproduces  
  *Analogy:* Runs every plan past the pre-built guest list and rebalances what's left.

## `src/routing_optimiser/engines/__init__.py`
*How it ties in:* The engine registry — pick an engine by name; used by the dropdown.

- **[FN-399]** `get_engine(key, weight, hard, soft)` — (small helper — see source)  
  *Analogy:* The rental desk: hand it an engine name and it hands back the ready-to-use engine.
- **[FN-400]** `engine_choices()` — (key, label) pairs for building a dropdown.  
  *Analogy:* The menu board — lists which engines you can pick, with their friendly labels.

## `src/routing_optimiser/engines/base.py`
*How it ties in:* The shared 'recipe contract' every engine follows: one profile in, shares out, plus reusable helpers.

- **[FN-401]** `ProfileProblem.n(self)` — Number of gateways in this profile (length of every aligned array).  
  *Analogy:* A quick head-count of the doors in this profile.
- **[FN-402]** `BaseEngine.__init__(self, weight, hard, soft)` — (small helper — see source)  
  *Analogy:* Hands a fresh engine its risk dial and the rules it must obey — briefing a chef before they cook.
- **[FN-403]** `BaseEngine._t(self, msg)` — Record one debug/trace line (a no-op unless tracing is switched on).  
  *Analogy:* The engine's flight recorder — jots down each step, but only when someone's watching.
- **[FN-404]** `BaseEngine.solve_traced(self, p)` — Solve one profile AND return the stage-by-stage trace for it.  
  *Analogy:* Solve one profile AND keep the running commentary, so the debug panel can replay every move.
- **[FN-405]** `BaseEngine._bounds(self, p)` — Per-gateway (lower, upper) share bounds from the hard constraints.  
  *Analogy:* Works out each door's allowed min/max share (and remembers it) — the 'at least this, no more than that' limits.
- **[FN-406]** `BaseEngine._project_box_simplex(v, lo, hi)` — Euclidean projection of ``v`` onto ``{x : sum(x)=1, lo<=x<=hi}``.  
  *Analogy:* Snaps a rough set of shares to the nearest valid one that totals 100% and respects every cap — nudging a seating plan until everyone fits and every table is within size.
- **[FN-407]** `BaseEngine._project_qp(self, ref, lo, hi, risk, cap)` — Closest valid split to ``ref`` whose portfolio risk meets a ceiling.  
  *Analogy:* The same snapping, but also keeping blended risk under the ceiling — turn a 'risk tax' dial just enough to drop under the line, no further.
- **[FN-408]** `BaseEngine._score(self, p)` — Per-gateway linear score: reward conversion, penalise risk.  
  *Analogy:* The old points system: points for approvals, minus points for risk (kept only for the dormant engines).
- **[FN-409]** `BaseEngine._ref_cache_key(self, p)` — Fingerprint of everything the reference split depends on EXCEPT the risk dial.  
  *Analogy:* A fingerprint of everything the starting split depends on, so it's built once and reused.
- **[FN-410]** `BaseEngine._ref_param_key(self, p)` — Engine-specific reference parameters (softmax/base default).  
  *Analogy:* The engine-specific part of that fingerprint — each engine declares what its starting split depends on.
- **[FN-411]** `BaseEngine._reference_split(self, p)` — Cached wrapper around `_reference_split_impl`.  
  *Analogy:* Hands back the 'chase-conversion' starting split, computed once and photocopied rather than rewritten each time.
- **[FN-412]** `BaseEngine._reference_split_impl(self, p)` — The slider=100 reference split: conversion only, no risk logic.  
  *Analogy:* Builds that starting split: favour the best doors, but floor every door so none goes dark — reward the top performers while keeping everyone in the game.
- **[FN-413]** `BaseEngine._project_to_vamp(self, p, shares)` — Nudge a split to the closest one that meets the VAMP (risk) cap.  
  *Analogy:* If a split is too risky, nudge it to the closest one that meets the risk cap — pull an over-full glass back to the fill line.
- **[FN-414]** `BaseEngine._finalise(self, p, shares, note)` — Clean up a raw share vector into a valid ProfileSolution.  
  *Analogy:* Plates the dish: clean up the raw split, renormalise, apply the risk cap, and attach the headline success/risk numbers.
- **[FN-415]** `BaseEngine._is_feasible(self, p, shares)` — True only if `shares` satisfies EVERY hard constraint for this profile.  
  *Analogy:* The pass/fail check — does this split break any hard rule (over a cap, on a banned door)?
- **[FN-416]** `BaseEngine.solve(self, p)` — Public entry point: return the chosen split for one profile.  
  *Analogy:* The front door: handle the trivial profiles (0 or 1 door) itself, hand the rest to the engine's own method.
- **[FN-417]** `BaseEngine._solve(self, p)` — Engine-specific split logic. Every concrete engine overrides this.  
  *Analogy:* The blank the concrete engine fills in with its own way of deciding a split.

## `src/routing_optimiser/engines/entropy.py`
*How it ties in:* Entropy engine (dormant) — spread traffic as evenly as the rules allow.

- **[FN-428]** `EntropyEngine._solve(self, p)` — (small helper — see source)  
  *Analogy:* A dormant engine that spreads traffic as evenly as the rules allow (maximum hedge), trading some conversion for spread.
- **[FN-429]** `EntropyEngine._solve.neg_obj(x)` — (small helper — see source)  
  *Analogy:* Its internal score to minimise.
- **[FN-430]** `EntropyEngine._solve.neg_grad(x)` — (small helper — see source)  
  *Analogy:* The slope of that score.

## `src/routing_optimiser/engines/genetic_ref.py`
*How it ties in:* A reference stand-in for the genetic engine (the real search runs separately).

- **[FN-431]** `GeneticRefEngine._solve(self, p)` — (small helper — see source)  
  *Analogy:* A reference stand-in for the genetic engine (the real one runs separately), used for comparison.

## `src/routing_optimiser/engines/portfolio.py`
*How it ties in:* Portfolio engine — diversify like investments, shy away from doors that could spike.

- **[FN-423]** `PortfolioEngine._ref_param_key(self, p)` — (small helper — see source)  
  *Analogy:* Portfolio's fingerprint of what its starting split depends on.
- **[FN-424]** `PortfolioEngine._reference_split_impl(self, p)` — slider=100 reference: mean-CVaR optimal (conversion vs downside VAMP  
  *Analogy:* Picks the split that chases conversion but shies away from doors whose risk could spike — diversifying like a cautious investor.
- **[FN-425]** `PortfolioEngine._reference_split_impl._f(x)` — (small helper — see source)  
  *Analogy:* The score being minimised: reward, minus a penalty for the plausible worst-case risk.
- **[FN-426]** `PortfolioEngine._reference_split_impl._grad(x)` — (small helper — see source)  
  *Analogy:* The slope of that score — which way to nudge the split to improve it.
- **[FN-427]** `PortfolioEngine._reference_split_impl._proj(v)` — (small helper — see source)  
  *Analogy:* Snap a candidate back onto the valid-split set at each step of the search.

## `src/routing_optimiser/engines/softmax.py`
*How it ties in:* Softmax engine — reward the best doors, keep a spread; trim toward the risk cap if needed.

- **[FN-418]** `SoftmaxEngine._solve(self, p)` — Return the profile's split: the reference, trimmed toward the risk cap if needed.  
  *Analogy:* Draw the conversion-chasing split; only if it's too risky, walk it back just enough to sit on the risk line.

## `src/routing_optimiser/engines/thompson.py`
*How it ties in:* Thompson engine — bet on each door by its chance of being the best (explores thin doors).

- **[FN-419]** `_leggauss_cached(m)` — Gauss–Legendre nodes/weights on [-1, 1] for m points. `leggauss` is a pure  
  *Analogy:* Precomputes and remembers the fixed 'measuring points' for the probability maths, so they aren't re-derived for every profile — keeping one ruler handy instead of re-marking one each time.
- **[FN-420]** `ThompsonEngine._ref_param_key(self, p)` — (small helper — see source)  
  *Analogy:* Thompson's fingerprint of what its starting split depends on (its Beta belief + tilt).
- **[FN-421]** `ThompsonEngine._beta_params(self, p)` — Beta(alpha, beta) per gateway — a SELF-CONTAINED posterior from the  
  *Analogy:* Turns each door's wins/losses into a 'how sure are we' belief — lots of data = a narrow, confident belief; little data = a wide, unsure one.
- **[FN-422]** `ThompsonEngine._reference_split_impl(self, p)` — slider=100 reference: analytic probability-of-being-best over SUCCESS.  
  *Analogy:* Gives each door a share equal to its CHANCE of being the best — so barely-tested doors still get a look-in.

## `src/routing_optimiser/vamp_forecast_pipeline.py`
- **[FN-070]** `build_pipeline_config(ui)` — Map the flat Forecast-tab settings onto the pipeline's settings.yaml schema.  
  *Analogy:* Translates the Forecast tab's simple settings into the pipeline's full config.
- **[FN-071]** `run_vamp_pipeline(config, project_root, gcp_project)` — Run the full VAMP pipeline and return the output directory.  
  *Analogy:* Runs the whole forecast pipeline and points to where its outputs landed.
- **[FN-072]** `run_vamp_pipeline._shape(obj, name, warn_empty)` — Log a dataframe's size + leading columns, PLUS the summed transaction count of any  
  *Analogy:* Logs a table's size and first columns as a sanity check.
- **[FN-073]** `_canonical_gateway(name)` — Some pipeline exports contain deprecated instances of a gateway with a  
  *Analogy:* Collapses a door's deprecated aliases onto its one true name.
- **[FN-074]** `_mm_kappa(vg, ng, fallback, kmax)` — Method-of-moments Beta-Binomial concentration (kappa) for one back-off LEVEL,  
  *Analogy:* Measures how alike a group of rates are, to decide how hard to smooth them.
- **[FN-075]** `_hier_vamp_shrink(d, fallback_kappa, kmax)` — FULLY-AUTOMATIC hierarchical empirical-Bayes shrinkage of the per-profile VAMP rate.  
  *Analogy:* Automatically smooths thin risk data by borrowing strength from broader groups.
- **[FN-076]** `_normalise_pre(df)` — Normalise a pipeline export into the optimiser's baseline contract:  
  *Analogy:* Tidies a raw pipeline export into the clean 'before' shape the optimiser wants.
- **[FN-077]** `normalise_pre_from_effective_rate(df)` — (small helper — see source)  
  *Analogy:* The same tidy-up, starting from the effective-rate export instead.
- **[FN-078]** `load_pre_forecast(path)` — Load the pipeline's baseline from its outputs. `path` may be a directory  
  *Analogy:* Loads the pipeline's baseline 'before' picture from its outputs.
- **[FN-079]** `looks_like_effective_rate(df)` — (small helper — see source)  
  *Analogy:* Sniffs a file to tell which export format it is.

## `src/routing_optimiser/ga_oliver.py`
- **[FN-080]** `_renorm_profiles(X, cs, cc, elig, ref)` — Zero non-eligible rows, then renormalise each profile to sum 1 over its eligible rows.  
  *Analogy:* Zeroes banned doors, then rebalances each profile to 100%.
- **[FN-081]** `_mutate(X, cs, cc, elig, ref, rng, mutation_rate, strength)` — Gaussian per-row mutation on a random subset of PROFILES (mirrors Oliver's mutate_split, but  
  *Analogy:* Randomly jiggles some profiles to explore new splits.
- **[FN-082]** `_crossover(A, B, cs, cc, rng, cx_rate)` — Per-profile uniform crossover (mirrors make_children): each profile independently comes from one  
  *Analogy:* Mixes two parent splits profile-by-profile to breed a child.
- **[FN-083]** `_score(X, ctx, cs, cc, elig, cap, floor)` — Cap-at-evaluation (uncapped genome), then this app's _obj_viol → (obj, viol).  
  *Analogy:* Grades a candidate split on revenue and rule-breaking.
- **[FN-084]** `_fast_nondominated_fronts(M)` — M: (P, k) MINIMISATION objectives. Returns list of fronts (each a list of indices).  
  *Analogy:* Sorts candidates into tiers where no one in a tier is beaten on every measure.
- **[FN-085]** `_crowding(M_front)` — Crowding distance for one front. M_front: (m, k).  
  *Analogy:* Rewards candidates in sparse areas to keep the options varied.
- **[FN-086]** `_select_nsga2(M, k)` — Pick k indices from M (P,k_obj) minimisation objectives via NSGA-II (fronts + crowding).  
  *Analogy:* Picks the survivors — best tiers first, variety as the tie-break.
- **[FN-087]** `run(ctx, lam)` — Drop-in for seed_search.run_midtilt_ga. Returns (best_shares (N,), info).  
  *Analogy:* The whole NSGA-II search — an alternative breeding programme.
- **[FN-088]** `_best_index(obj, viol)` — Feasibility-first: among feasible (viol<=tol) pick max obj; else min viol (tie max obj).  
  *Analogy:* Among the rule-abiding finalists, picks the top scorer.
- **[FN-089]** `_feas_scalar(obj, viol)` — Monotone 'higher = better' score: feasible splits always beat infeasible; feasible ranked  
  *Analogy:* A single 'higher is better' score that always ranks legal splits above illegal ones.

## `src/routing_optimiser/genetic_fullmatrix.py`
- **[FN-090]** `_renorm_profiles(X, cs, cc)` — Renormalise each contiguous profile segment of X (P, N) so every profile sums to 1.  
  *Analogy:* Rebalances each profile of the full matrix back to 100%.
- **[FN-091]** `_repair(X, cs, cc, elig, cap, floor)` — Make every row a deployable split: non-negative, eligible-only, per-profile sum 1, then  
  *Analogy:* Fixes any candidate into a deployable split — non-negative, eligible, summing to one.
- **[FN-092]** `_objectives(pop, ctx)` — (revenue [maximise], aggregate expected VAMP count [minimise]) per candidate.  
  *Analogy:* Scores each candidate on revenue and total risk.
- **[FN-093]** `_fast_nondominated_sort(F)` — NSGA-II non-dominated sort. F (P, M) minimisation. Returns list of fronts (each a list  
  *Analogy:* Sorts candidates into unbeaten tiers.
- **[FN-094]** `_crowding(F, idxs)` — Crowding distance for members `idxs` (boundary points = inf).  
  *Analogy:* Keeps a spread of options by favouring the lonely ones.
- **[FN-095]** `_nsga2(pop, ctx, rng, cs, cc, elig, cap, floor, generations, mutation_rate, mutation_sigma, stop_check)` — (small helper — see source)  
  *Analogy:* The multi-objective breeding loop over the whole matrix.
- **[FN-096]** `_nsga2.objs_min(P)` — (small helper — see source)  
  *Analogy:* Restates the goals as 'smaller is better' for the sorter.
- **[FN-097]** `_nsga2.rank_and_crowd(F)` — (small helper — see source)  
  *Analogy:* Ranks by tier, then by variety.
- **[FN-098]** `_nsga2.breed(P, rank, crowd)` — (small helper — see source)  
  *Analogy:* Produces the next generation of candidate matrices.
- **[FN-099]** `_nsga2.breed.better(i, j)` — (small helper — see source)  
  *Analogy:* The head-to-head judge for two candidates.
- **[FN-100]** `run_fullmatrix_ga(ctx, lam)` — Evolve the full per-profile share matrix. Single-objective (revenue − λ·risk) by default;  
  *Analogy:* The whole full-matrix search from start to finish.
- **[FN-101]** `run_fullmatrix_ga.fit_of(P)` — (small helper — see source)  
  *Analogy:* Reads one candidate's fitness.
- **[FN-102]** `run_fullmatrix_ga.pick()` — (small helper — see source)  
  *Analogy:* Crowns the winning matrix at the end.

## `src/routing_optimiser/seed_search.py`
*How it ties in:* The genetic (CMA-ES) search: breed better splits over many rounds.

- **[FN-103]** `_mid_sums(vol, mid_rows, M, S)` — Per-MID column sums of `vol` (P, N) -> (P, M).  
  *Analogy:* Totals each merchant account's traffic across all the rows it appears in.
- **[FN-104]** `_build_mid_incidence(mid_id, M, N)` — Sparse (M, N) 0/1 incidence for `_mid_sums`' fast path: entry (m, n)=1 iff row n  
  *Analogy:* A checklist marking which rows belong to which merchant account, for fast totalling.
- **[FN-105]** `_fitness(pop, ctx, lam)` — Vectorised fitness. Revenue is the SAME quantity tab 4 shows as incremental  
  *Analogy:* The scoreboard — how much extra revenue each candidate plan is expected to earn.
- **[FN-106]** `_risk_z_per_mid(risk, mid_rows, n_mid, N)` — Standardise each vampMid's per-profile risk across ITS rows, so θ_m tilts that  
  *Analogy:* Grades each door's riskiness on a curve within its own account, so a tilt means the same everywhere.
- **[FN-107]** `_cap_floor_shares(X, profile_starts, profile_counts, elig, cap, floor)` — HARD per-profile max-share cap + exploration floor on shares X (P, N), vectorised.  
  *Analogy:* Trims any door over its ceiling, lifts any below its floor, then rebalances back to 100%.
- **[FN-108]** `_mid_over(shares, ctx, include_floor_shortfall)` — Per-candidate per-MID breach magnitude (P, M): the RELATIVE overage (actual/limit − 1,  
  *Analogy:* Measures how far over its limit each merchant account has strayed.
- **[FN-109]** `_ret_z_per_mid(ret, mid_rows, n_mid, N)` — Standardise each vampMid's per-profile REVENUE-efficiency (rev_coef) across ITS  
  *Analogy:* Grades each door's revenue-per-payment on a curve within its own account.
- **[FN-110]** `_leaned_ref(ref, risk, elig, profile_starts, profile_counts, gamma)` — Lean the revenue reference gently toward LOWER global risk (γ ≥ 0): the θ=0  
  *Analogy:* Gives the starting plan a gentle nudge toward the safer doors before the search begins.
- **[FN-111]** `_cap_floor_prep(profile_starts, profile_counts, elig, cap, floor)` — Precompute the per-profile CONSTANTS the max-share/floor water-fill needs, so they  
  *Analogy:* Prepping the cap/floor tools once so the balancing step runs fast.
- **[FN-112]** `_cap_floor_apply(X, prep)` — HARD per-profile floor-then-cap water-fill using PRECOMPUTED constants (`prep` from  
  *Analogy:* Pouring share back and forth between doors until each sits between its floor and ceiling.
- **[FN-113]** `_risk_z_per_profile(risk, profile_starts, profile_counts, N)` — Standardise each PROFILE's per-gateway risk across ITS rows (mirror of `_risk_z_per_mid`  
  *Analogy:* The same risk grading-on-a-curve, but within each profile instead of each account.
- **[FN-114]** `_decode_midtilt3(genome, M, ref, zr, zq, mid_id, profile_starts, profile_counts, elig, cap, floor)` — genome (P, 3M[+K]) = [θr (risk-tilt) | θq (return-tilt) | g (gain) | profileθ (K fine)] -> (P, N).  
  *Analogy:* θr / θq / g are ~three knobs per MID. Turning θr up leans that MID's volume toward its LOW-risk profiles, θq toward its HIGH-revenue profiles, and g raises/lowers its overall presence. This function turns those ~20 knob settings into a full per-gateway split (then applies the hard floor/cap). That tiny genome is why the search is so fast.
- **[FN-115]** `_mid_viol_weights(ctx, M)` — Per-MID VIOLATION weight (VOLUME-WEIGHTING of the feasibility violation).  
  *Analogy:* Weights each account's rule-break by how much traffic it carries, so big accounts count more.
- **[FN-116]** `_obj_viol(shares, ctx)` — Split the score into (objective, violation) for feasibility-first ranking.  
  *Analogy:* Separates 'how good is this plan' from 'how many rules it breaks' — judged in that order.
- **[FN-117]** `_obj_viol._pen(_ov)` — (small helper — see source)  
  *Analogy:* The little meter that converts each overshoot into penalty points.
- **[FN-118]** `_violation_breakdown(shares, ctx, top_k)` — One-shot DECOMPOSITION of the _obj_viol violation for a SINGLE split (diagnostic only —  
  *Analogy:* An itemised receipt showing exactly which rules a single plan broke, and by how much.
- **[FN-119]** `_violation_breakdown._pen(_ov)` — (small helper — see source)  
  *Analogy:* The line-item calculator on that receipt.
- **[FN-120]** `_violation_breakdown._pen_split(_o)` — (small helper — see source)  
  *Analogy:* Splits each penalty line into its separate causes for the receipt.
- **[FN-121]** `_feas_keys(obj, viol, tol)` — Deb feasibility-first ranking as a MINIMISABLE key (improvement #3). Feasible  
  *Analogy:* The sorting rule that always puts rule-abiding plans ahead of rule-breaking ones.
- **[FN-122]** `_cmaes(eval_ov, x0, sigma0, lo, hi)` — Active (μ/μ_w, λ)-CMA-ES over the box [lo, hi] (bounds via clipping). `eval_ov(X)`  
  *Analogy:* CMA-ES searches like a smart swarm. Each generation it samples candidate tilt-vectors from a bell-shaped "cloud", scores them, keeps the best few, then MOVES and RESHAPES the cloud toward them — automatically learning which directions matter and how big a step to take. "Active" means it also nudges the cloud AWAY from the worst directions, and "feasibility-first" means a compliant candidate always beats a non-compliant one when picking the winner.
- **[FN-123]** `run_midtilt_ga(ctx, lam)` — Active-CMA-ES cross-profile per-vampMid tilt search — the live engine (see the block  
  *Analogy:* The whole breeding programme end to end — the actual search that runs in production.
- **[FN-124]** `run_midtilt_ga._decode(G)` — (small helper — see source)  
  *Analogy:* The programme's own translator from dials to a finished split, with caps applied.
- **[FN-125]** `run_midtilt_ga._decode_precap(G)` — (small helper — see source)  
  *Analogy:* The same translation but before trimming — used to read the gradients cleanly.
- **[FN-126]** `run_midtilt_ga._to_actual(V)` — (small helper — see source)  
  *Analogy:* Converts the search's 0-to-1 dials into real-world dial values.
- **[FN-127]** `run_midtilt_ga._to_unit(x)` — (small helper — see source)  
  *Analogy:* The reverse — folds real dial values back onto the 0-to-1 scale.
- **[FN-128]** `run_midtilt_ga._bands_pen(X)` — (small helper — see source)  
  *Analogy:* The referee that adds each plan's band-limit fines to its violation score.
- **[FN-129]** `run_midtilt_ga._bands_pen(X)` — (small helper — see source)  
  *Analogy:* The same referee, wired up to the turbo scorer.
- **[FN-130]** `run_midtilt_ga.eval_ov(V)` — (small helper — see source)  
  *Analogy:* Scores a whole batch of candidate plans in one go.
- **[FN-131]** `run_midtilt_ga.score_of(gu)` — (small helper — see source)  
  *Analogy:* Scores a single candidate plan.
- **[FN-132]** `run_midtilt_ga.better(a, b)` — (small helper — see source)  
  *Analogy:* The tie-breaker judge deciding which of two plans wins — rules first, then revenue.
- **[FN-133]** `run_midtilt_ga._profilerep(v)` — (small helper — see source)  
  *Analogy:* Totals each profile, then stamps that total back onto every door in it.
- **[FN-134]** `run_midtilt_ga._midsum1(v)` — (small helper — see source)  
  *Analogy:* Totals one plan's traffic per merchant account.
- **[FN-135]** `run_midtilt_ga.eval_ov(V)` — (small helper — see source)  
  *Analogy:* The turbo-lane batch scorer, swapped in when the fast engine is available.
- **[FN-136]** `run_midtilt_ga.score_of(gu)` — (small helper — see source)  
  *Analogy:* The turbo-lane single-plan scorer.
- **[FN-137]** `run_midtilt_ga._fine_grad(vec)` — (small helper — see source)  
  *Analogy:* Works out which way to nudge the fine per-profile dials to improve revenue.
- **[FN-138]** `run_midtilt_ga._grads(gu)` — (small helper — see source)  
  *Analogy:* The compass telling the polish step which direction is downhill.
- **[FN-139]** `run_midtilt_ga._fit_to_target(target)` — (small helper — see source)  
  *Analogy:* Reverse-engineers the dial settings that would produce a desired target split.
- **[FN-140]** `run_midtilt_ga._fit_to_target._loss(z)` — (small helper — see source)  
  *Analogy:* The 'how far off target are we' meter that the fitting tries to drive to zero.
- **[FN-141]** `run_midtilt_ga._seed_key(sv)` — (small helper — see source)  
  *Analogy:* Ranks the starting candidates so the best-behaved seeds go first.
- **[FN-142]** `run_midtilt_ga._repair(Vpop)` — (small helper — see source)  
  *Analogy:* A pit-stop that fixes any plan straying over an account limit before it races on.
- **[FN-143]** `run_midtilt_ga._polish_genome(u0)` — #2 memetic: SLSQP with analytic revenue + smooth-violation gradients (Nelder–Mead  
  *Analogy:* After breeding finds a good plan, a careful hand-tuning to squeeze out the last improvement.
- **[FN-144]** `run_midtilt_ga._polish_genome._gc(z)` — (small helper — see source)  
  *Analogy:* The polish step's constraint-checker.
- **[FN-145]** `run_midtilt_ga._polish_genome._sk(z)` — (small helper — see source)  
  *Analogy:* The polish step's scoring key.
- **[FN-146]** `run_midtilt_ga._farthest_start(rng_seed, explored, n_try)` — A unit-box start point maximising the min distance to every already-explored genome  
  *Analogy:* Picks a fresh starting plan as far as possible from ones already tried, to explore new ground.
- **[FN-147]** `run_midtilt_ga._rank_key(sv)` — (small helper — see source)  
  *Analogy:* The final leaderboard rule that crowns the overall winning plan — rules first.

## `src/routing_optimiser/impact.py`
- **[FN-148]** `profile_baseline_vs_proposed(split, avg_ticket)` — Per profile: expected successful transactions and revenue under the baseline  
  *Analogy:* For each profile, the expected approvals and revenue before vs after.
- **[FN-149]** `profile_baseline_vs_proposed.ticket(rpgt)` — (small helper — see source)  
  *Analogy:* The typical payment size used in that calculation.
- **[FN-150]** `headline_impact(profile)` — (small helper — see source)  
  *Analogy:* The top-line 'what changes overall' summary numbers.
- **[FN-151]** `key_contributors(profile, by, top)` — Which banks / currencies / RPGTs drive most of the incremental revenue.  
  *Analogy:* Names the banks, currencies and payment types driving most of the change.
- **[FN-152]** `gateway_volume_shift(split)` — How much volume each gateway gains/loses vs baseline (the 'stolen'  
  *Analogy:* How much traffic each door gains or loses versus today.
- **[FN-153]** `_split_volume(df)` — Per-row profile volume from a split frame, tolerating either column name.  
  *Analogy:* Reads each row's volume from a split, coping with missing columns.
- **[FN-154]** `gateway_move_vs_reference(ref_split, sel_split, keys)` — Per-gateway volume BEFORE (`ref_split`) vs AFTER (`sel_split`), aligned on the  
  *Analogy:* Each door's traffic before vs after, measured against the reference plan.
- **[FN-155]** `traffic_moved_curve(variations, ref_weight)` — Fraction of total volume moved vs the revenue reference, for every dial position.  
  *Analogy:* How much total traffic shifts as the slider moves — the 'how much are we changing?' curve.

## `src/routing_optimiser/kmeans_compress.py`
*How it ties in:* Groups near-identical profiles so fewer deployable rules ship.

- **[FN-156]** `wallet_segment_split(split, wallet_incapable, wallet_frac, wallet_default, fid2vamp, wallet_label, nonwallet_label)` — Add a `pmp` (paymentMethodProvider) dimension so wallet traffic routes only  
  *Analogy:* Splits wallet traffic out by provider, so Apple/Google Pay get their own routing line.
- **[FN-157]** `_cap_and_respill(vec, cap)` — Cap every share at `cap` and re-spill the overflow onto the others (keeps the sum at 1).  
  *Analogy:* pour the excess from any over-full glass into the ones with room, in proportion to how much they already hold, and repeat until none overflow — so a cluster's representative split never puts more than `cap` on a single gateway (it always keeps a backup).
- **[FN-158]** `_weighted_accuracy(X, recon, w)` — % fidelity: 100 = identical. Uses L1 distance on share vectors.  
  *Analogy:* A faithfulness score — 100 means the compressed rules match the originals exactly.
- **[FN-159]** `_fit_k(X, w, k, seed)` — (small helper — see source)  
  *Analogy:* Tries grouping the profiles into k bundles and measures how well it fits.
- **[FN-160]** `compress_split(split, group_keys, rpgt_targets, max_gateway_cap, k_max, seed)` — Returns (compressed_rules, elbow, stats).  
  *Analogy:* The grouping machine — bundles near-identical profiles so fewer rules ship, and reports how good the fit is.
- **[FN-161]** `count_config_rules(compressed)` — Number of JSON routing rules the compressed split will generate.  
  *Analogy:* Counts how many deployable rule files a given grouping would produce.
- **[FN-162]** `compress_to_pool_budget(split, target_pools, count_pools_fn, group_keys, max_gateway_cap, k_max, seed, method, allocation, parallel, count_backend)` — Compress so the GENERATED POOL count is <= target_pools, using as large a profile  
  *Analogy:* Bundles just enough to fit within an allowed number of deployable rules.
- **[FN-163]** `compress_to_pool_budget._no_compression(_reason_feasible)` — (small helper — see source)  
  *Analogy:* The 'leave it as-is' shortcut when everything already fits without grouping.
- **[FN-164]** `compress_to_pool_budget._eval(b)` — (small helper — see source)  
  *Analogy:* Tries one target size and reports the resulting fit and rule-count.
- **[FN-165]** `compress_to_pool_budget._parallel_counts(cls)` — Run the (expensive) count_pools_fn on a list of clusterings, in parallel when  
  *Analogy:* Runs the expensive rule-counting for many groupings at once, across cores.
- **[FN-166]** `compress_to_pool_budget._eval_many(bs)` — Evaluate several budgets at once: build each clustering (cheap), dedupe by  
  *Analogy:* Tries several target sizes together to find the sweet spot faster.
- **[FN-167]** `_build_compress_context(split, group_keys, max_gateway_cap, k_max, seed, method, allocation)` — Precompute everything that DOESN'T depend on the cluster budget: the volume-weighted  
  *Analogy:* Preps everything that doesn't change with the target size, once, up front.
- **[FN-168]** `_compress_with_context(ctx, n_configs)` — Greedy volume-weighted cluster allocation for a given budget, using a prebuilt  
  *Analogy:* Hands out cluster slots greedily, giving the busiest profiles their own rule first.
- **[FN-169]** `_compress_with_context._fit(g, k)` — (small helper — see source)  
  *Analogy:* Tries one grouping and scores its fit.
- **[FN-170]** `_compress_with_context._push_next(g)` — (small helper — see source)  
  *Analogy:* Spends the next cluster slot where it buys the most accuracy.
- **[FN-171]** `_compress_ext(ctx, n_configs)` — OPT-IN compression: alternative cluster METHOD and/or budget ALLOCATION.  
  *Analogy:* An alternative grouping method (Ward) for when the default needs a second opinion.
- **[FN-172]** `_compress_ext._ward_model(g)` — (small helper — see source)  
  *Analogy:* Builds the Ward family-tree of which profiles are most alike.
- **[FN-173]** `_compress_ext._labels_centroids(g, k)` — (small helper — see source)  
  *Analogy:* Reads off which bundle each profile lands in, and each bundle's representative split.
- **[FN-174]** `_compress_ext._acc(g, k)` — (small helper — see source)  
  *Analogy:* Scores how faithful a given grouping is.
- **[FN-175]** `_compress_ext._push(g)` — (small helper — see source)  
  *Analogy:* Spends the next slot where it adds the most fidelity.
- **[FN-176]** `compress_to_budget(split, n_configs, group_keys, max_gateway_cap, k_max, seed, method, allocation)` — Compress a per-profile split to ~n_configs representative rules TOTAL by greedily  
  *Analogy:* Squeezes the plan down to about a target number of representative rules.

## `src/routing_optimiser/menu_picker.py`
- **[FN-177]** `_profile_matrix(split, idx_cols, gateway_cols)` — (profiles x gateways) share matrix (renormalised) + per-profile volume, aligned to  
  *Analogy:* Lays every profile's split out as a tidy grid.
- **[FN-178]** `_build_group_menu(X, w, menu_k, seed)` — Shortlist of candidate split vectors for one group: volume-weighted KMeans centroids  
  *Analogy:* Draws up a shortlist of candidate splits a group could share.
- **[FN-179]** `menu_compress(split, group_keys, menu_k, max_items, max_gateway_cap, seed)` — Compress a per-profile split by MENU PICKING.  
  *Analogy:* Compresses by having each profile pick the closest split off a shared menu.
- **[FN-180]** `menu_compress._item_id(c)` — (small helper — see source)  
  *Analogy:* Gives each menu choice a label.
- **[FN-181]** `menu_compress._distinct_items()` — (small helper — see source)  
  *Analogy:* Counts how many distinct menu choices got used.
- **[FN-182]** `menu_compress._next_available(c)` — (small helper — see source)  
  *Analogy:* Grabs the next free menu slot.

## `src/routing_optimiser/numba_kernels.py`
*How it ties in:* A compiled, fused version of the genetic scoring loop — same maths, much faster.

- **[FN-183]** `njit()` — (small helper — see source)  
  *Analogy:* The turbo switch again — use the fast compiler if present, else run plain Python.
- **[FN-184]** `njit()` — (small helper — see source)  
  *Analogy:* A second turbo switch, for the no-compiler fallback path.
- **[FN-185]** `njit.deco(f)` — (small helper — see source)  
  *Analogy:* The bolt that attaches the compiler to the function.
- **[FN-186]** `_fused_eval(G, M, ref, zr, zq, mid_id, cs, cc, elig, fine_idx, zr_profile, n_fine, nec_col, fl_col, capN_col, has_floor, has_cap, cv, risk, rc, has_vcap, vcap, has_volcap, volcap, n_bands, b_mi, b_bval, b_ceil, b_floor, b_has_ceil, b_has_floor, b_pmul, has_base, base_vol, wm, max_share, floor_val, rmw, has_vfr, vfr, bfix, qwt, pexp, has_elig, ecs, ecc, e_has_ban, e_ban, e_has_w, e_w_incap, e_w_wf, e_has_u, e_u_incap, e_u_wf)` — One fused pass: ACTUAL genome batch G (P, 3M[+K]) -> (obj (P,), viol (P,)).  
  *Analogy:* instead of building each intermediate array and handing it to the next NumPy step (like shipping half-finished parts between factory stations), this does the whole decode → eligibility → score on ONE workbench per candidate. The bulky in-between arrays never exist — which is where the speed comes from — while the maths and the summation ORDER stay the same, so the answer matches the NumPy path to float64 rounding.
- **[FN-187]** `_prep_cols(profile_starts, profile_counts, elig, cap, floor)` — Per-column nec / floor / capN constants, matching `seed_search._cap_floor_prep`  
  *Analogy:* Pre-cutting each door's cap/floor constants so the conveyor never pauses to recompute them.
- **[FN-188]** `make_numba_eval(M, ref, zr, zq, mid_id, profile_starts, profile_counts, elig, cap, floor, fine_idx, zr_profile, n_fine, cv, risk, rc, ctx)` — Return a callable `eval_actual(G)->(obj, viol)` (G in ACTUAL genome space) backed by  
  *Analogy:* Builds and hands you the ready-tuned fast scorer.
- **[FN-189]** `make_numba_eval.eval_actual(G)` — (small helper — see source)  
  *Analogy:* The scorer itself — feed it plans, get back scores and rule-breaks.
- **[FN-190]** `verify(np_eval_actual, nb_eval_actual, sample_G)` — Run NumPy and Numba evals on the SAME actual-space genomes and compare. Returns a dict  
  *Analogy:* Runs the plain and turbo scorers on the same plans to prove they agree to the last digit.

## `src/routing_optimiser/optimiser.py`
*How it ties in:* Runs an engine across every profile and assembles the split; also the per-MID risk-cap enforcers.

- **[FN-191]** `_vamp_cap_lp(df, cap, floor, max_share, agg_cap, _reduce)` — Joint solve for the per-vampMid VAMP cap: the split CLOSEST to the reference  
  *Analogy:* Finds the split closest to the reference that keeps every MID's risk under the cap, in one big 'least movement' solve — rearranging the fewest parcels so no truck is overloaded.
- **[FN-192]** `vamp_frontier_lp(df, cap, agg_cap, floor, max_share)` — Frontier point (public wrapper for `_vamp_cap_lp` with an aggregate budget):  
  *Analogy:* The same, plus a whole-book risk budget — used to trace the risk-vs-revenue trade-off curve.
- **[FN-193]** `_group_indices(labels)` — {label -> ascending row positions}, identical to  
  *Analogy:* A filing clerk sorting rows into labelled trays (one tray per profile or MID) in a single pass.
- **[FN-194]** `_profile_recip_order(profile_rows, rate)` — Per-profile row positions sorted by rate ASCENDING, ties broken by ascending row index.  
  *Analogy:* Pre-sorts each profile's doors cheapest-risk-first, once, so moves don't re-sort every time.
- **[FN-195]** `enforce_mid_vamp_caps(df, cap, floor, max_share, max_iter, step)` — Cross-profile adjustment so each vampMid's AGGREGATE VAMP rate <= cap.  
  *Analogy:* a MID's monitored rate is the volume-weighted average across every profile it runs in — like a student's overall grade averaged across subjects, weighted by credit hours. To pull that average under the limit with the least disruption, we move volume off the MID's WORST profiles onto the cheapest alternative in each; a MID that's over the limit in EVERY profile can't be fixed by re-weighting, so it's retired (dropped) and its volume handed off.
- **[FN-196]** `enforce_mid_vamp_caps._mid_rate(m)` — (small helper — see source)  
  *Analogy:* Reads a MID's current volume-weighted risk rate.
- **[FN-197]** `enforce_mid_vamp_caps._rt(m)` — (small helper — see source)  
  *Analogy:* A fast running-total version of that rate, updated as volume moves instead of recomputed.
- **[FN-198]** `enforce_mid_volume_caps(df, a_max_by_mid, max_share)` — Scale each vampMid's allocated volume down to a_max x its BASELINE volume.  
  *Analogy:* a spend cap per MID. If a MID is routed more volume than a_max × what it historically carried, we shrink every one of its profiles by the same factor (like trimming an over-budget line item proportionally) and hand the freed volume to the cheapest other gateway in each profile.
- **[FN-199]** `optimise_split(problems, settings)` — Solve every profile with the selected engine and assemble the long split table.  
  *Analogy:* Runs the chosen engine over every profile and stacks the answers into one tidy table — the assembly line turning per-profile decisions into the whole plan.
- **[FN-200]** `portfolio_summary(split)` — Volume-weighted headline numbers for a whole split (the book-level scorecard).  
  *Analogy:* Rolls a whole plan up into its headline volume-weighted success and risk numbers.
- **[FN-201]** `sweep_slider(problems, settings, weights)` — Produce split *variations* across the conversion↔risk slider.  
  *Analogy:* Re-solves the whole book at each risk-dial position to trace the trade-off curve (used by the pipeline script).

## `src/routing_optimiser/precluster.py`
- **[FN-202]** `_profile_signatures(ctx)` — One hashable signature per profile. Two profiles share a signature IFF, gateway-for-gateway  
  *Analogy:* Gives each profile a fingerprint, so identical ones can be spotted.
- **[FN-203]** `build_clusters(ctx)` — Group profiles by identical signature. Returns a dict with:  
  *Analogy:* Buckets together profiles with matching fingerprints.
- **[FN-204]** `reduce_ctx(ctx, clusters)` — Build a REDUCED ctx over one representative profile per cluster. Intensive per-gateway fields  
  *Analogy:* Shrinks the problem to one representative per bucket.
- **[FN-205]** `reduce_ctx._take(name)` — (small helper — see source)  
  *Analogy:* Pulls the representative row out of a bucket.
- **[FN-206]** `run_midtilt_ga_preclustered(ctx)` — OPT-IN drop-in for `seed_search.run_midtilt_ga` (same call/return contract). Clusters the  
  *Analogy:* Runs the search on the shrunken problem, then copies answers back — same result, less work.
- **[FN-207]** `run_midtilt_ga_preclustered._red(w)` — (small helper — see source)  
  *Analogy:* The reduced-problem helper inside that run.
- **[FN-208]** `expand_shares(rep_shares, expand)` — Copy each representative profile's per-gateway shares to ALL its member profiles → full (N,)  
  *Analogy:* Copies each representative's split back onto all its identical profiles.

## `src/routing_optimiser/run_bundle.py`
- **[FN-209]** `_write_config(folder, config)` — Write config as YAML if available, else JSON. Returns the path written.  
  *Analogy:* Saves the run's settings as YAML (or JSON if YAML isn't available).
- **[FN-210]** `prune_old_runs(runs_dir, keep)` — Keep the `keep` most-recent run folders under `runs_dir`; delete the rest.  
  *Analogy:* Keeps only the most recent run folders and clears the rest.
- **[FN-211]** `write_run_bundle(runs_dir, config)` — Create runs_dir/<timestamp[_name]>/ with config, log.txt, artifacts.npz, meta.json.  
  *Analogy:* Packages a whole run — config, logs, outputs — into one timestamped folder.
- **[FN-212]** `_stop_path(target)` — A directory -> its _stop file; a file path -> itself.  
  *Analogy:* Works out where the 'please stop' flag file lives.
- **[FN-213]** `request_stop(target)` — Drop the stop flag (target may be a runs dir or an explicit file path).  
  *Analogy:* Drops a 'please stop' flag the running search will notice.
- **[FN-214]** `clear_stop(target)` — (small helper — see source)  
  *Analogy:* Removes that stop flag.
- **[FN-215]** `stop_requested(target)` — (small helper — see source)  
  *Analogy:* Checks whether someone has asked the run to stop.
- **[FN-216]** `_StopCheck.__init__(self, path)` — (small helper — see source)  
  *Analogy:* Sets up the stop-flag watcher.
- **[FN-217]** `_StopCheck.__call__(self)` — (small helper — see source)  
  *Analogy:* Each time it's asked, reports whether to stop.
- **[FN-218]** `make_stop_check(target)` — Return a zero-arg predicate for a GA's `stop_check` param: True once the flag exists.  
  *Analogy:* Hands the search a simple 'should I stop?' button to poll.
- **[FN-219]** `_ProgressWriter.__init__(self, path)` — (small helper — see source)  
  *Analogy:* Sets up the progress-file writer.
- **[FN-220]** `_ProgressWriter.__call__(self, inc, score, fitness)` — (small helper — see source)  
  *Analogy:* Writes the latest progress out so the UI bar can read it.

## `src/routing_optimiser/schema.py`
- **[FN-221]** `gateway_columns(columns)` — Return the gateway/MID columns from a template header.  
  *Analogy:* Picks out which columns of a template are the doors.

## `src/routing_optimiser/sql_runner.py`
- **[FN-222]** `list_sql_files(sql_dir)` — (small helper — see source)  
  *Analogy:* Lists the available query files.
- **[FN-223]** `cache_path_for(sql_path, cache_dir, params)` — Cache filename includes a short hash of the SQL file's own text AND of  
  *Analogy:* Works out a cache filename that changes whenever the query does.
- **[FN-224]** `_substitute_params(sql, params)` — (small helper — see source)  
  *Analogy:* Slots the run's parameters into the query template.
- **[FN-225]** `run_sql_file(sql_path, cache_dir, use_cache, fallback_csv, project, params)` — Return (data_path, source) where source is one of:  
  *Analogy:* Runs a query (or serves its cached result) and hands back the data.

## `src/routing_optimiser/success_rates.py`
*How it ties in:* Turns raw attempts data into trustworthy per-door success rates (smoothed for thin data).

- **[FN-226]** `load_success_data(source)` — Load the attempts/success data from a DataFrame, or a CSV/parquet path.  
  *Analogy:* Reads the raw approve/decline history, from a table or file, and tidies it into shape.
- **[FN-227]** `_apply_time_decay(df, half_life_days, date_col)` — Apply an exponential half-life weight to each row so recent attempts count  
  *Analogy:* Gives recent payments more say than old ones, fading the past out smoothly.
- **[FN-228]** `_empirical_bayes_kappa(grp, scope, fallback, kmax)` — Method-of-moments Beta-Binomial concentration (kappa) per prior_scope group.  
  *Analogy:* kappa asks "do the gateways in this group behave alike?" If their success rates cluster tightly (small true spread), trust the pooled average heavily → big kappa (shrink hard). If they're all over the place, trust each gateway's own data → small kappa (shrink little). It's LEARNED from the data rather than being a fixed dial.
- **[FN-229]** `gateway_success_rates(df, gateway_col, shrink_strength, time_decay_half_life_days, prior_scope, empirical_bayes)` — Returns one row per (rpgt, currency, bank, gateway) with:  
  *Analogy:* The headline approval rate for each door, smoothed so thin data can't fool it.
- **[FN-230]** `detect_blocked_gateways(adf, min_consecutive, date_col)` — Flag (bank, gateway) pairs the acquiring bank appears to have BLOCKED us on.  
  *Analogy:* like spotting a vendor whose card terminal has declined EVERY transaction for days straight — most likely the bank cut them off, so we stop throwing traffic at a dead route (the caller caps that gateway to the exploration floor) instead of bleeding conversions.
- **[FN-231]** `rpgt_gateway_sensitivity(sr_df, avg_ticket, min_attempts)` — How sensitive is each RPGT to WHERE its traffic is routed.  
  *Analogy:* Measures how much each payment type's success depends on which door it's sent through.
- **[FN-232]** `risk_rates_from_forecast(forecast, gateways, default, shrink)` — Expected chargeback/VAMP rate per gateway. In production this comes from  
  *Analogy:* Reads each door's expected chargeback rate from the baseline forecast.
