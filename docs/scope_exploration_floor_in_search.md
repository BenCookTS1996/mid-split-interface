# Scope — modelling the EXPLORATION FLOOR inside the search

Status: **design only, no code written.** Supersedes the 19gw attempt (reverted to default OFF in 19gx).
Written 2026-09-02 against `f48050e` (19ha).

---

## 0. What the 10:17 run established

`ROUTING_PROJ_FLOOR=1`, one variable, search left alone:

* RECONCILIATION ERROR **1 → 1,802**, against a float32 noise floor of 9.5
* `[rung] TXN` — SPLIT **0 (0%)** · ENFORCEMENT **0 (0%)** · PROJECTION MATH **1,801 (100%)**
* `[rung] VAMP` — PROJECTION MATH **1 (100%)**, i.e. nothing
* `[rung2] GRAIN DISPERSION` 0 of 100,191 · PROFILE PRESENCE 0

The GA shipped exactly the split it scored and `build_split_exports` rewrote nothing. The
entire 1,802 is *"identical shares, two different M5 models"*, and all of it is TXN-side.

> **Caveat on the dispersion reading.** `[rung2]` measures the *enforced* shares out of
> `build_split_exports`, which applies **no floor**. Its 0.0% is evidence the *shipped template*
> is grain-flat — it is **not** evidence that the floored `prop_share` is grain-flat. The floor
> is grain-dependent by construction (`min(floor, 1/n_eligible)` counted per profile). Do not
> use `[rung2]` to argue the floor can be applied at profile grain. That argument is what broke 19gw.

---

## 1. The structural finding that makes this tractable

The two group keys are the same six-part key:

| | key |
|---|---|
| delivery — `impact_calcs.py:1546` | `["Currency", "BIN", "RPGT", "_pmp", "_ctry", "period"]` |
| projector — `band_projection.py:148` | `["cur", "bin", "rpgt", "pmp", "ctry", "per"]` |

So `gcode` in the projector kernel **is** delivery's `grp`, and `_pshare[r]` in the kernel **is**
`t0["prop_share"]`. Same object, same grain, same key.

**The projector already runs at the grain the floor needs. Nothing has to be re-grained.**

19gw floored `prop_raw`'s columns — `propidx` grain, i.e. *search-profile* grain — and did it
*before* the `/ _psum[c]` normalisation. Coarser grain, wrong side of the divide. That is the
whole of the error; it was not a tuning problem.

---

## 2. Why the floor is TXN-only, and must stay that way

| | delivery | projector today | after this change |
|---|---|---|---|
| **TXN share** | `prop_share` — **floored**, not profile capped | `_pshare` — capped, not floored | `pshf` — capped **then floored** |
| **VAMP share** | `_vprop` = waterfill(`prop_raw/prop_sum`) — capped, **not floored** | derived from `_pshare` — capped, not floored | **unchanged** |

`impact_calcs.py` says this outright at the `_vprop` block:

> *"DELIBERATELY NOT `prop_share`: that column also carries the 0.01 exploration floor, which the
> search's water-fill does not apply."*

So the VAMP path **already agrees between the two sides** — which is exactly why `[rung] VAMP`
read 1 and `[rung] TXN` read 1,801. The code reading and the measurement corroborate each other.

**Second reason VAMP must stay un-floored:** `_AGE_RENORM` re-bases `_vshare` over
`(profile, period, t)`, and delivery's `_efloor` touches only the `t == 0` slice. A floored
`_vshare` would corrupt the aged pass, where delivery applies no floor at all.

### The landmine

In `_cb_kernel_impl` — **the kernel that actually ships**, `_PROJ_CB_ON` defaults to `"1"` —
`_vshare` is *derived from `_pshare` on the fly* (`_pshare[o] / _vpsum[_cg]`), never
materialised. **Flooring `_pshare` in place would silently floor VAMP too.** Hence a separate
`pshf` buffer rather than an in-place edit. This is the single easiest way to reproduce 19gw's
failure mode from a different direction.

---

## 3. Ordering: floor AFTER the water-fill, and do not re-cap

Delivery's sequence for `prop_share`:

```
_fm_cap at CELL grain (upstream, in _fm_deliv)   # CELL = one gateway in a profile
  -> / prop_sum at PROFILE grain       <- can push a row back above the cap
  -> FLOOR
  -> renormalise
  -> (no re-cap)
```

Therefore: **floor after the kernel's water-fill, renormalise, do not re-cap.** Flooring before
the water-fill lets the cap undo the floor and will not match delivery.

---

## 4. The exact insertion

`impact_calcs.py:1632-1648` is the specification. Transcribe it; do not re-derive it.

### Eligibility clause mapping

| delivery clause | kernel equivalent | status |
|---|---|---|
| `base_share > 0` | `base_c[i] > 0` | available (static) |
| `prop_raw > 0` | `_pr[i] > 0` | available (per candidate) |
| `_keep > 0` | `pw_c[i] > 0` | available — `pw` **is** the `_keep` fraction (19dt) |
| `~_emask_f` (wallet / USA) | — | **MISSING — the one new input** |
| `prop_sum > 0` | `ps > 0.0` | available |

### The one new input

A static 0/1 per-row array. `T0` already carries `pmp`, `ctry` and `midl`, so it is buildable
inside the projector from `wallet_incapable` / `usa_only`.

**Clone `self._vcpos`.** It is already exactly this shape of thing: a static per-row eligibility
mask, threaded into both kernels as `vcpos`, permuted profile-major by `_cb_arrays` into `vcpos_c`.
Same shape, same threading, same lifecycle. This is not a new mechanism — it is a second
instance of one that already works.

Threading route: `vamp_off_mids=frozenset()` is already a `PopulationBandProjector` kwarg;
`exploration_floor`, `wallet_incapable` and `usa_only` follow the identical path from
`tab_2_routing_engine.py` (where `_wallet_incapable` / `_usa_only` are built at ~4305-4352 and
already handed to `impact_calcs`).

### Where, in `_cb_kernel_impl`

Inside the existing `if ps > 0.0:` block, **after** the water-fill and **before** the `vpsum`
pass — four passes over `range(s, e)`:

1. count eligible rows -> `n`
2. `pshf[i] = max(_pshare[i], min(floor, 1.0/n))` where eligible, else `_pshare[i]`
3. sum `pshf` over the profile
4. divide

All contiguous, all already in cache — the profile-blocked layout makes this cheap in a way the
flat kernel does not.

Then the `nC` txn loop reads `pshf[r]` instead of `_pshare[r]`. **Nothing else changes.**

---

## 5. The four bodies that must move together

This is where the real risk lives, and 19gw never had to face it.

1. **`_pop_band_kernel_impl`** (`:183`) — flat; serial + parallel compiles off one body
2. **`_cb_kernel_impl`** (`:464`) — profile-blocked; serial + parallel. **This is the shipping kernel.**
3. **`_pop_band_kernel_fm_cache`** — lazy fastmath compile of (1); measurement only
4. **`_shares()`** (`:1430`) — the pandas path feeding `BandProjector.project()`. Its own comment:
   *"it must move with them or the two in-search paths disagree, which `test_vconserve.py` detects."*

### Structural advantage over 19gw

**`_fm_deliv` is not touched.** It stays a three-stage chain (`_cp(_el(_bl(x)))`), so
`[deliv-fuse]`'s hand-composed self-check keeps matching. 19gw broke it by adding a fourth
stage; this design adds none.

---

## 6. What does NOT change

* **`build_split_exports`** — applies no floor, and must keep applying none. The shipped template
  is *pre-floor*; the live allocation engine floors at runtime. Adding one here would double-apply.
* **`_fm_floor` / `ROUTING_SEARCH_FLOOR`** — become dead. Recommend deleting after one clean run,
  the same way `_repair_maxshare` went in 19gv.
* **The exact LP band solvers** — linearise RAW and have no delivery Jacobian; the floor stays
  invisible to them. Unchanged.
* **The softmax decode / max-share cap** — untouched.

---

## 7. The seed — an open decision for you

The three seed stages judge through `deliver_fn=_seed_dlv` = `_fm_deliv`, the *share-vector*
transform. **The floor cannot reach them by that route** — wrong grain, same reason as 19gw.

* **(a) Leave the seed judging un-floored.** A known, bounded inconsistency: the seed only
  *proposes*, and the GA re-scores everything through the projector. Free. **Recommended.**
* **(b) Route the seed's feasibility test through the projector.** Exact, but costs a projector
  call per seed pass.

Recommend (a) for the first run, with `[seed-basis]` reporting the size of the gap so it is
measured rather than assumed.

---

## 8. Cache identity

`projection_cache_sig` (`impact_calcs.py:2464`) already hashes `exploration_floor`, so delivery
is safe.

On the projector side, **check every numba compile the floor value folds into.** All the kernel
compiles are `cache=False` except the serial flat one. This codebase has already had a wrong
answer served from a stale numba cache — the `_VAMP_CONSERVE` incident documented at
`band_projection.py:426-437`, where a module global folded into a `cache=True` compile and the
kill switch silently did nothing. The floor is the same class of value. Anything it folds into
must be `cache=False`, or the floor must arrive as an argument rather than a global.

---

## 9. Acceptance tests — what must be true before it ships

1. **Floor off, or floor 0 -> bit-identical to today**, on both kernels and `_shares`.
   `np.array_equal` on `vamp` and `txn`, not a tolerance.
2. **Both sides floored -> the `exact` class of `[denom]` collapses to noise.** This is the
   headline test. It does **not** mean total error reaches single digits — see §13 for the
   residual that survives by construction. Judge this test on the `[denom]` class breakdown, not
   on the total.
3. `[rung] TXN PROJECTION MATH` -> ~0.
4. `[rung] VAMP` -> **still ~1**. A VAMP error that moves means the floor leaked into the VAMP
   path. This is the specific 19gw-class regression to watch for.
5. `[deliv-fuse]` still **PASSES**.
6. `_cb_kernel` vs `_pop_band_kernel` vs `_shares` agree **with the floor on** — three
   implementations, one answer. `test_vconserve.py` and `test_proj_parallel.py` cover this.
7. `adyen - totalav - na` delivers **>= its 20,000 floor** (it delivered 19,765 on the 10:17 run),
   or the drift warning explains why not.
8. A fixture test in the 19df style: one profile, known gateway counts, floor 0.01, asserting the
   kernel against a hand-computed `_efloor`.

---

## 10. Cost — expect it to be net FASTER

**Added:** 4 extra passes over live rows, on ~15 existing. `[eval-cost]` prices the band
projector at 148.8s and `_deliver_full` at 146.7s of a 380s eval, so roughly **+60-80s**, ~5% of
a 1,500s run.

**Recovered:**

* `[forensic]` stops firing once reconciliation is back under the bar — **-190.6s**
* `[nw-skip]` starts firing again once the shipped split meets its floors — **-532.0s**

**Net: likely -500s or better**, and the `adyen` compliance miss closes.

---

## 11. Expect a different split, and expect it to look worse

vwsr will fall. The floor deliberately holds share on alternatives the GA would abandon.

That is **correct**: the GA can now see that abandoning a gateway does not remove its VAMP,
because the engine puts it back at runtime. Bands that read "met" may stop reading met — those
were never met in reality; the search was scoring a split that gets modified before it executes.

**Do not read a lower vwsr as a regression on this change.** Read `[rung]` instead.

---

## 12. Implementation order

Each step is separately shippable and separately verifiable.

1. Thread `exploration_floor` + the two eligibility sets into `PopulationBandProjector`; build the
   static row mask beside `_vcpos`; log its size. **No behaviour.** Confirm bit-identical.
2. Add the floor to `_cb_kernel_impl` + the `pshf` buffer, behind `ROUTING_PROJ_SFLOOR`,
   **default OFF**. Confirm bit-identical with it off.
3. Mirror into `_pop_band_kernel_impl` and `_shares`. Cross-check all three with the floor ON.
4. Fixture test against a hand-computed `_efloor`.
5. **One full run with `ROUTING_PROJ_SFLOOR=1` AND `ROUTING_PROJ_FLOOR=1` together.** They must
   move together or they measure different objects — the same rule `_SFLOOR_ON` already carries.
6. Read acceptance tests 2-7 off that log. **Only then** consider defaulting it on.
7. After one clean run: delete `_fm_floor`, `ROUTING_SEARCH_FLOOR`, and the 19gw post-mortem block.

**Worth doing at step 1 regardless:** fix `[deliv-fuse]`'s self-check to call `_fm_deliv_serial`
instead of hand-composing `_cp(_el(_bl(x)))`. It is the reason 19gw's breakage was ambiguous, and
it will break again the moment any fourth stage is added.

---

## 13. What the residual will be — this does NOT reach 0

Three sources survive a correct implementation. Only the first is float32.

### (a) float32 noise — ~9.5
`f32_noise_floor()` measured 9.5 on the 10:17 run, and its `bound = dv_sum + dt_sum` is
deliberately conservative. Irreducible while `_PROJ_F32=True`. Setting `ROUTING_PROJ_FLOAT32=0`
removes it at a throughput cost.

### (b) `[denom]` class 4, "profile absent in IN-SEARCH" — the one that matters

`_inject_backfill_rows` (`impact_calcs.py:1543`) **adds zero-baseline `t0` rows** to the
delivered frame for prop items that have no baseline row. The in-search scaffold has no
counterpart for them, so `_mD[4]` is 0 by construction while `_dlD[4]` carries real delivered
volume.

**There is no kernel row to floor.** Putting the floor in the projector cannot close this term.
Worse, the floor is what *creates* it: with the floor off those profiles sit at baseline on both
sides and net to ~0, which is why total error read 1. Turn the floor on and delivery moves them
while the search cannot see them.

On the 10:17 run, `adyen - totalav - na` split as:

| `[denom]` class | profiles | Δ |
|---|---|---|
| `exact` | 4,888 | **-1,097** ← closes |
| `profile absent in IN-SEARCH` | 266 | **+206** ← **survives** |
| net | | -891 |

The equivalent split for `woodforest - total av` (-782) and `authorize - total av` (+128) was not
printed, so the total residual is **not known** — do not extrapolate adyen's ratio. Read it off
the first run.

This term's owner is **back-fill scope**, not floor modelling. It is separate work: either
inject the matching rows into the scaffold, or exclude back-filled rows from the reconciliation
basis so the number measures what it claims to.

### (c) The cap / grain asymmetry — small, but not structurally zero

Delivery's TXN share is capped at **profile** grain by `_fm_cap` upstream, then renormalised at
**profile** grain (which can push a row back above the cap) and never re-capped. The kernel's
TXN share is capped at **profile** grain by the water-fill. Empirically this contributes ≤ 1
(that is what the floor-off run measures), so it is not worth chasing — but it is a real
asymmetry and it is not guaranteed to stay at 1 once the floor's renormalisation runs on top of it.

### Expected reading after the change

* `[rung] TXN PROJECTION MATH` — the `exact` class gone, the `absent` class remaining
* `[rung] VAMP` — unchanged at ~1
* **Total — well below 1,802, well above 9.5.** Somewhere in the low hundreds is the honest
  expectation, dominated by (b).
* `[forensic]`'s bar is `f32_noise_floor()`-scaled, so **it will still fire.** Either accept that
  cost, or raise the bar to the measured class-4 residual once it is known — a decision to take
  from the first run's log, not now.


---

## 14. The capability-mask grain — DECIDED 2026-09-02

**Decision: the floor uses the SEARCH's (vampMid, currency)-grain mask, and delivery is brought
into line with it.** Recorded here because it changes the delivered number, so it is not a
step-2 implementation detail.

### The two masks

| | grain | built where | over-blocks? |
|---|---|---|---|
| search (`emask`) | (vampMid, **currency**) | tab_2 `_wc_pairs` / `_uo_pairs`, from `Master_MID_List + routing_restrictions` | no |
| delivery (`_emask` / `_emask_f`) | vampMid **only** | `impact_calcs`, from the `wallet_incapable` / `usa_only` NAME SETS | yes |

Delivery over-blocks any vampMid whose fids differ in capability by currency — tab_2's
`[emask]` line names the case: *PaySafe - Total AV is wallet-capable in USD, not in EUR/GBP.*
The vampMid-only test bans it everywhere.

`[ef-mask]` (19he) reports both counts and the disagreement every run.

### CORRECTION (19hh) — the blast radius is much smaller than first written

The first draft of this section listed nine call sites, including `build_split_exports` and tab
3's four enforcement calls. **That was wrong, and reading the code settled it:**

**`build_split_exports` is already FID-GRAIN and already exact.** Its own note, dated
2026-08-17, says the vampMid roll-up was deliberately REMOVED for precisely the PaySafe reason:

> *"Template columns ARE fids, so `g in set` is already exact; the extra `fid2vamp.get(g) in
> set` term rolled the vampMid's capability onto every sibling fid and over-blocked the ones
> that CAN serve … Removed."*

Fids are currency-specific, so fid grain is **finer** than (vampMid, currency). Nothing to do.
`enforced_prop_items` and `enforced_split_frame` delegate to it, so they are fine too — and tab
3's four sites at 543 / 789 / 1343 / 4291 are calls to *those two*, not to the projection.

**So there was exactly ONE coarse consumer: `compute_vamp_prepost_granular`**, because the
pro-rata export carries vampMid and no fid. It had the wallet/USA test written out **twice** —
the `prop_raw` zeroing and the floor block's own `_emask_f` — which is now one shared builder.

### A real gap found while threading it

`tab_2`'s own delivery projection passed **no** `wallet_incapable` / `usa_only` at all, only
`capability`. For the `prop_raw` zeroing that is correct and by design (`_ep` is the enforced
template; the fid-grain mask is already baked in). But **the exploration-floor block's mask is
not gated on `_enforced`**, so with empty sets it was all-False — every row read as
capability-eligible, and the floor's `base_share > 0` clause could resurrect a gateway
enforcement had zeroed. That is the "the floor must never un-mask an ineligible gateway" hazard
with nothing standing in its way. Passing the pairs closes it.

### The work — DONE in 19hh

1. ✅ **Pair builder hoisted** to `app_common.capability_pairs`, memoised on (path, mtime).
   Verified identical to tab_2's inline version on the live MID list: **89 wallet-incapable
   pairs, 4 USA-only pairs**, same provenance label.
2. ✅ **Pair-grain parameters** on `compute_vamp_prepost_granular`, `projection_cache_sig` and
   `_c_prepost_granular`, falling back to the name sets. Verified **byte-identical** to the
   pre-19hh formula with the switch off.
3. ✅ **Call sites migrated**: tab_2's delivery projection and tab 3's three
   `_c_prepost_granular` calls. (Not nine — see the correction above.)
4. ✅ **`ROUTING_EMASK_PAIRS`, default OFF.** The grain is in the cache key, and so is the
   switch, so flipping it cannot be served a projection computed at the other grain.
   `_PROJ_CODE_VER` bumped for the same reason.
5. ⬜ **Step 2 of §12** — wire the floor into the kernel. Now unblocked.

### Measured property of the change

The pair grain **only ever removes a block, never adds one.** On a fixture spanning four
currencies and three payment-method types, every row the two grains disagreed on was
`name-set = masked, pairs = not masked` — 25 of 44. So arming it can only free capacity that a
gateway can genuinely serve; it cannot take capacity away. That is the safe direction, and it
is why this is worth arming before the floor rather than after.

### Does the whole chain still reconcile?

**Not automatically, and not until step 3 above is complete.** The chain has four hand-off
points and the mask has to be the same object at every one:

```
search (projector)  ──►  delivery (compute_vamp_prepost_granular)
                    ──►  tab 3 PRE vs POST impact table
                    ──►  tab 3 exports / validate-split
```

The search and delivery agree once (2) and (3) land. Tab 3's table and its exports read
`compute_vamp_prepost_granular` and `build_split_exports`, so they inherit the fix — **but only
if tab 3's four call sites are migrated in the same change.** Leave any one of them on the name
sets and tab 3 will show a different PRE/POST split from the one the engine reconciled against,
which is the same class of failure as 19gw and would look like a tab 3 bug.

**Acceptance test for this piece, before the floor is touched at all:** with
`ROUTING_EMASK_PAIRS=1` and the floor still OFF everywhere, reconciliation error must stay
inside the float32 noise floor and tab 3's PRE must still agree exactly with the baseline. That
isolates the mask change from the floor change — one variable, as the 10:17 run was.

---

## 15. The acceptance test ran, and it exercised nothing (2026-09-02, 16:19)

`ROUTING_EMASK_PAIRS=1 ./run.command` came back **byte-identical to the unarmed 15:22 run**:
all 15 delivered band values, success rate 0.615322, reconciliation error 1, `[rung]` VAMP 1
(100%) / TXN Σ 0.322, `[forensic]` 1.11 units, 11,840 splits, and `[profiles]` 6,428 keys with
Σ|Δprop| 24.4448 and the same 8 sample keys.

That is not the test passing. **The test could not have failed**, because on tab_2's reconcile
path the mask has no consumer:

| gate | line | on this path | consequence |
|---|---|---|---|
| `_enforced = (_n == 7)` | `impact_calcs.py:1521` | **True** — tab_2 passes `_ep` from `enforced_prop_items` | … |
| `if (_wc_s or _uo_s or _use_pairs) and not _enforced:` | `impact_calcs.py:1650` | **skipped** | the `prop_raw` zeroing never runs |
| `if _efloor > 0.0:` | `impact_calcs.py:1675` | **skipped** — `ROUTING_PROJ_FLOOR` unset, so `_pj_floor` is 0.0 | the floor's eligibility mask never runs |

`_cap_emask()` is therefore never called, and the grain it would have used is irrelevant.
`ROUTING_EMASK_PAIRS` is a **no-op on this path by construction**, not by coincidence. It can
only bite where the frame is RAW (tab 3's three `_c_prepost_granular` calls, 4- and 5-tuple
`prop_items`) or where the exploration floor is non-zero.

The prediction written before the run — "expect the delivered split to move… with 89,955 rows
differing, some band values will shift" — was wrong for this reason, not because the pair grain
does nothing. The 89,955-row figure came from `[ef-mask]`, which compares the two masks
directly; it is still correct, and it is still the size of the disagreement. It just does not
reach the delivered number from here.

### What the log now says (19hr)

`_LAST_EMASK_GRAIN` existed but was never printed, and it recorded the **armed** grain — intent
wearing a fact's clothes, which is exactly what the 19df comment in the same file warns against.
It now records the consumers that actually ran, and tab_2 prints it once beside the projection:

```
[emask-grain] FACT: (vampMid, currency) pairs - NOT APPLIED, no consumer ran: frame is
ENFORCED (7-tuple), so the prop_raw zeroing is skipped; exploration_floor=0 so the floor block
is skipped. ROUTING_EMASK_PAIRS cannot change this call's output.
```

So a run can no longer arm a switch, print a paragraph about how it changes the delivered
number, and change nothing, without saying so on the same page.

### The corrected acceptance test

The floor block is the **only** consumer of the mask on the reconcile path, so isolating the
mask there is not possible — the two have to move together:

- `ROUTING_PROJ_FLOOR=1` alone → baseline for the floor at the coarse name-set grain.
- `ROUTING_EMASK_PAIRS=1 ROUTING_PROJ_FLOOR=1` → the same floor at the pair grain.

The **difference between those two runs** is the pair grain's effect, one variable, with the
floor held fixed. That is two runs, not one, and it needs agreement before it is spent.

Tab 3 is the other route: its three `_c_prepost_granular` calls are on RAW frames, so
`ROUTING_EMASK_PAIRS=1` bites there today with no floor involved. Opening tab 3 after an armed
run tests the mask in isolation for free.

---

## 16. Both sides on the fine grain — the switch defaults ON (19ht)

Ben's call: **both need to do the fine version.** Done, and it turned out to be a smaller change
than §15 implied, because the delivered path was already there.

### There are THREE grains, and only the coarsest was ever wrong

| grain | who | since |
|---|---|---|
| **fid** | `build_split_exports` — its template columns ARE fids | 2026-08-17 |
| **(vampMid, currency)** | the SEARCH, and `compute_vamp_prepost_granular` under the switch | 2026-08-17 / 19hh |
| **vampMid name sets** | what `compute_vamp_prepost_granular` did by default | — |

`enforced_prop_items` takes its shares **straight from `build_split_exports`**, so tab_2's
reconcile path has been on the *finest* grain all along. That is the real reason the 16:19 run
was byte-identical: not just that the mask had no consumer, but that the number it would have
corrected was already correct.

The coarse test survived in exactly two places, both of them closed on the reconcile path: the
`prop_raw` zeroing (RAW frames only) and the exploration-floor eligibility set. Its live
consumers are **tab 3's PRE/POST impact table** and **any floored run**.

### Do the two fine grains agree? Measured, not assumed

The pair grain ORs over a pair's ACTIVE fids; `build_split_exports` tests each fid on its own.
They give the same answer only while a pair's active fids agree with each other. On the live
`Master_MID_List` (2026-09-02):

- **296** `(vampMid, currency)` groups
- **111** of them hold more than one fid
- **5** have fids that disagree on `processWallet`
- **0** disagree among the **ACTIVE** fids

All five are an active fid paired with its inactive `-test` sibling (`adyen-usd-tav-na` vs
`adyen-usd-tav-na-test`, and the same for tdr / tab / tvn / tcl). So the two fine grains agree
exactly today, and the whole disagreement was ever between the coarse test and the other two.

### What shipped

- `ROUTING_EMASK_PAIRS` **defaults ON**. `=0` restores the coarse test — the switch stays,
  because this changes what tab 3 shows and what a floored run computes.
- `impact_calcs.emask_pairs_on()` is now the **one reader** of it. tab_2's `[emask-grain]` line
  calls that instead of doing its own `os.environ.get`, which is how a default flip would
  otherwise have been reported wrongly in the log while behaving correctly in the code.
- `app_common.capability_pairs` records `LAST_CAP_PAIR_SPLITS` — the pairs whose **active** fids
  disagree — and tab_2 logs a ⚠ if it is ever non-empty. It is empty today. The thing that would
  make it non-empty is a **MID-list edit**, which no code change would announce, and on that day
  the search and the exports go back to masking different rows.
- `_PROJ_CODE_VER` → `2026-09-02-19ht-pair-grain-DEFAULT-ON`. The switch was already hashed into
  the cache key by 19hh, so a cached pairs-OFF projection cannot be served to a pairs-ON run
  either way.

### What to expect from the next run

**Nothing, on the headline numbers.** The reconcile path was already fid-grain, so band values,
success rate and reconciliation error should be unchanged from 15:22 — and if they *do* move,
that is the finding, not the feature. The change is visible in **tab 3's PRE/POST table**, which
is the free isolation test §15 was pointing at: it is now on the fine grain by default.

---

## 17. Step 2 is shipped (19hv) — `ROUTING_PROJ_SFLOOR`, default OFF

### What the kernel now does

`_cb_kernel_impl` takes `ef_c`, `efloor`, `usefloor` and a `pshf` lane buffer. With `usefloor == 0`
it takes exactly the branches it took before, and the run is bit-identical — that is step 2's
acceptance test and it is asserted, not argued (`tests/test_19hv_sfloor_kernel.py`).

### The asymmetry, finally read off delivery's own source

`impact_calcs`'s `_vprop` block settles what 19gw guessed at:

| | delivery | the kernel before 19hv | the kernel with SFLOOR on |
|---|---|---|---|
| **TXN** share | `prop_share` — floored, renormalised, **uncapped** | `_pshare` — capped, unfloored | `pshf` — floored, renormalised, uncapped ✅ |
| **VAMP** share | `_vprop` — capped, **unfloored** | `_pshare` — capped, unfloored ✅ | unchanged ✅ |

Delivery's comment says it outright: `_vprop` is *"DELIBERATELY NOT `prop_share`: that column also
carries the 0.01 exploration floor … which the search's water-fill does not apply."*

**This is why the separate buffer is not optional.** In this kernel `vshare` is *derived* from
`_pshare` on the fly, so flooring in place would floor VAMP too — a quantity delivery leaves
unfloored — and the reconciliation error would move in a direction nothing in delivery explains.
The fixture asserts VAMP is byte-identical armed and unarmed.

**Two changes under one switch, on purpose.** `pshf` is built from the *uncapped* proposals
(`_pr/ps`), not from the capped `_pshare`, because delivery's TXN share is uncapped. Flooring the
capped share would produce a third object matching neither side. Half of this is not a smaller
step; it is a wrong one.

### Two guards, both learned from 2026-08-24

While `SFLOOR` is armed:

- **The profile-blocked self-check is SKIPPED**, and says so loudly. Its reference is the flat
  kernel, which does not floor yet (step 3), so the diff would fail *by construction*, the
  `except` would disable profile-blocking for the whole process, **and float32 would die with it
  because it lives inside that same path.** That cascade already happened once and cost a 5h22m
  run. An armed run is therefore an experiment, not a delivered number.
- **The float32 live-width drift is NOT re-measured**, and says so. `_f32_drift`'s float64
  reference is also the flat kernel, so the "drift" would be float32 **plus the entire floor** —
  and `[f32-floor]` would raise the reconciliation bar by it, hiding a real disagreement behind a
  number that is not float32's.

### Fact, not intent

`_SFLOOR_ON` is the request. `_SFLOOR_FACT["applied"]` is whether the kernel ran with
`usefloor > 0`, which also needs a non-zero `exploration_floor` to have reached the projector.
`proj_config()` prints the second, and prints a distinct `REQUESTED AND IS NOT IN EFFECT` line
when they differ — so no run can claim the floor is modelled while scoring an unfloored split.

### Not done in 19hv (§12 steps 3–7)

3. Mirror into `_pop_band_kernel_impl` and `_shares`, so the self-check and the drift measurement
   can run again. **This is the next step and it is what makes an armed run trustworthy.**
4. Fixture against a hand-computed `_efloor` — *done early*, it is the file above.
5. One full run with `ROUTING_PROJ_SFLOOR=1` **and** `ROUTING_PROJ_FLOOR=1` together.
6. Read acceptance tests 2–7 off that log.
7. Then delete `_fm_floor`, `ROUTING_SEARCH_FLOOR` and the 19gw post-mortem block.

**Do not arm this for a delivered run until step 3 lands.** With the self-check off, the only
thing standing behind the profile-blocked kernel is the fixture.
