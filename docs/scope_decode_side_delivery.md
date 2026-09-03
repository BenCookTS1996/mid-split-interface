# Scope — fold the delivery transform into the DECODE, so `deliver` is not needed at all

**Goal (Ben, 2026-09-02):** *"The raw splits need to factor in everything deliver does. I want to
get it to a place where deliver is not needed at all."*

---

## 1. What "deliver" is, and where it already lives

```
_fm_deliv_serial(x) = floor( cap( elig( block(x) ) ) )      # tab_2:6654
```

- **block** — `_apply_blocked_caps`: pins auto-blocked (bank, gateway) rows to the exploration
  floor and redistributes the freed share.
- **elig** — `apply_elig_pop`: bans → 0 + renormalise; wallet blend + renormalise; USA blend +
  renormalise.
- **cap** — the 0.97 max-share water-fill. **LAST, because that is where delivery applies it.**
- **floor** — the 19gw exploration floor. Off by default; 19hv is replacing it.

**One of the four is already in the decode.** 19gu made the cap *a property of the decode*
(`_segment_softmax(..., max_share=...)`), so an over-cap split is not something a candidate can
express. That is the precedent this whole scope follows.

## 2. Why it is worth doing

| | measured, 2026-09-02 17:39 |
|---|---|
| `_deliver_full` share of the search | **147.5s, 38.2% of eval** |
| scored-vs-shipped discrepancy | **6,428 prop-keys, Σ\|Δprop\| 24.4448** |
| RAW vs DELIVERED on the seeds | a knife-edge on every run (19hx) |

If the decode emits the delivered split, all three go away at once: the second transform is not
run, the discrepancy is zero by construction, and there is no second basis to disagree with.

## 3. The enabling fact — eligibility is a MASK at this grain, not a blend

`apply_elig_pop` implements a *blend*: a wallet-incapable gateway keeps the fraction of the
profile's volume that is not wallet. That is machinery from a coarser grain. At the current
optimisation grain a profile is **one payment method and one country**, so the blend fraction is
always 0 or 1 — and `[elig-grain]` measures exactly that, every run:

> capability is a clean yes/no on **100%** of rows for wallet and **100%** for USA-only
> (154,405 and 154,405 of 154,405)

**So at this grain, elig = zero the ineligible rows, renormalise the rest.** That is precisely
what a masked softmax does:

```
softmax_masked(z)_i  =  exp(z_i) / Σ_{eligible} exp(z_j)
                     =  (exp(z_i)/Σ_all) / (Σ_elig exp(z_j)/Σ_all)
                     =  s_i / Σ_{eligible} s_j          ← zero + renormalise, exactly
```

`[elig-grain]` is therefore not a diagnostic here — **it is the precondition**, and it must become
a hard refusal rather than a report before any of this is armed.

## 4. THE BLOCKER, found before writing any code

The composition order is **block → elig → cap**. Moving eligibility into the decode makes it
**elig → block → cap**, and those are not the same function.

- `block` redistributes a blocked row's freed share to the profile's other rows — **including
  ineligible ones** — and `elig` then zeroes those and renormalises, which moves the blocked rows
  off the floor `block` just put them on.
- That is not hypothetical. `[deliv-cap]` measures the same interaction from the other side:
  **9,898,787 (candidate, profile) pairs repaired "that eligibility zeroing had lifted past
  0.97"**.

So eligibility cannot simply move to the front. **It has to move into the decode *in its current
position*, which means block has to move with it.**

## 5. What the 6,428 discrepancy keys actually are

Their worst values are `0.029989, 0.029987, 0.029987, 0.029975, …` — all essentially **0.03**.
`[deliv-cap]` reports **largest single-row move 0.03**. They are the same thing:

> the decode caps at 0.97 → **elig renormalises and lifts rows back over the cap** → `_fm_cap`
> repairs them.

The discrepancy is not diffuse drift and it is not eligibility on its own. **It is the cap being
applied twice with a renormalisation in between**, and the second application is the one that
ships. Fold elig into the decode *before* the decode's cap and the second application has nothing
left to do.

## 6. Target shape

```
decode(z) = cap( elig( block( softmax(z) ) ) )         # the whole of _fm_deliv, in one place
_deliver_full  →  identity
RAW == DELIVERED                                       # by construction, not by tolerance
```

The cap stays LAST, which is the order delivery already uses, so this does not move the cap — it
moves the two passes that precede it to the other side of the decode boundary.

## 7. Steps

1. **Make `[elig-grain]` a refusal, not a report.** If any row's wallet or USA fraction is not a
   hard 0/1, decode-side eligibility is not equivalent and must not arm. No behaviour today
   (it reads 100%/100%), and it is the guard everything else rests on.
2. **`ROUTING_DECODE_DELIV`, default OFF — SHIPPED as 19ia, and it turned out to be one line.**
   The fold does not need the transform physically relocated into `_segment_softmax`. The search
   ALREADY scores `_deliver_full(_segment_softmax(logits))`; it just **returns the bare
   `_segment_softmax(best_logits)`** (genetic_fullmatrix ~2557). Returning the delivered array
   instead makes scored == shipped with no change to the hot loop and no reordering of anything.
   That is the whole of the 6,428 keys.
   - `[decode-deliv]` reports rows moved / worst |Δ| / Σ|Δ| — the discrepancy measured at source.
   - It is a change to WHAT SHIPS, so tab 3's PRE/POST and the delivered M5 can move.
   - **Step 2b, NOT done:** `eval_pop` still computes the SUCCESS RATE from its own internal
     softmax with no block and no eligibility. So the objective is measured on the undelivered
     split while the constraints are measured on the delivered one. Closing that changes what the
     GA optimises and therefore the success rate itself — one axis per run.
3. **Keep `_fm_deliv` in the chain and measure it against a KNOWN FLOOR — not against zero.**
   `[deliv-fixed]` settled this on the 2026-09-02 20:20 run: the transform is **not idempotent**,
   and applying it twice moves **12 of 154,405 rows, worst |Δ| 6.212e-04**. So
   `_fm_deliv(decode(z)) == decode(z)` is the wrong assertion — it can never hold. The right one
   is **≤ ~12 rows and worst |Δ| ≤ ~1e-3, with the actual figures printed**, so a real regression
   stands out against that floor instead of hiding behind a tolerance nobody can see.
   "Reconcile exactly" is unreachable; **12 rows and 6e-04 is what is on offer**, and that is ~0
   in band terms.
4. **Only then remove `_fm_deliv` from `_eval_with_bands`** — and the 147.5s with it.
5. Re-read the 6,428 keys / Σ|Δprop| 24.4448 off the next log. **The target is 0 keys.**

## 8. What this does NOT close

- **float32.** GA-fitness comes from a float32 kernel and delivery is float64, so the 9.5-unit
  noise floor stands whatever happens here.
- **The exploration floor.** That is 19hv's `ROUTING_PROJ_SFLOOR`, a separate axis; the two must
  not be armed in the same run or neither is measurable.
- **`build_split_exports`.** It applies its own fid-grain rules downstream and is not modelled
  here. It has agreed with the search since 19ht (§16 of the floor scope), and this changes
  nothing about that.

## 9. Why this is a scope and not a diff

The 19gw floor attempt cost an 1,855s run by transforming the right rule on the wrong object. The
order-of-operations blocker in §4 is exactly that class of error, and it is invisible until you
look at what `block` hands to `elig`. Step 3 is the design's own check: if `_fm_deliv` is not a
provable no-op afterwards, the fold is wrong and the log says so before anything ships.

---

## 10. Two corrections, after 19ia

### 10a. The 12 rows are NOT float32

`[deliv-fixed]`'s residual — 12 of 154,405 rows, worst |Δ| 6.212e-04 — has nothing to do with
`ROUTING_PROJ_FLOAT32`. That setting narrows the **band projector kernel**; the delivery transform
is a different code path and is **float64 end to end**:

- `_fm_cap` → `np.asarray(_farr, float)`
- `apply_elig_pop` → `np.asarray(X, dtype=float)`
- `[deliv-fuse]` verifies it by **int64 bit-pattern comparison**, which only makes sense on float64

And the size settles it independently: **6.2e-04 on a share is not rounding.** float64 epsilon on
a share of order 0.01–0.97 is ~1e-16 relative; 6.2e-04 is twelve orders of magnitude larger. It is
0.06 of a percentage point — a real move on a real row.

**So it is structural, exactly as predicted in §4:** `block` pins a row to the exploration floor,
the 0.97 water-fill that runs after it sees a row with `share > 1e-12` under the cap and hands it
excess, and the row comes off the floor. Twelve rows is small. It is not noise.

### 10b. `deliver` is NOT redundant, and the 147.5s was overstated

I framed the prize as *"`_deliver_full` is 147.5s, 38.2% of eval — removing it is more than a
third of the run."* That was wrong in a way worth recording.

**The transform still has to run, once per candidate, for ever.** The band penalty is scored on
the *delivered* split — that is the whole point of 19fg and 19go — and you cannot score the
delivered split without computing it. `_deliver_full` inside `_eval_with_bands` is not a duplicate
or a check; it is the thing that produces the number the fitness uses.

What 19ia removed was the **mismatch**, not the computation. What separates the two:

| | can it go? |
|---|---|
| `_deliver_full` in `_eval_with_bands` | **No.** It computes the band penalty's input. |
| the scored-vs-shipped gap (6,428 keys) | **Gone in 19ia** — return the array that was scored. |
| the SECOND cap (`_fm_cap` after `elig`) | **Yes, but only via the §4 reorder** — put `elig` in front of the decode's cap and the second application has nothing to repair. |
| `_fm_deliv` in the never-worse comparison | already deduped by 19fi. |

So "deliver is not needed at all" is true of the **concept** — after 19ia the search's output is
already delivery-final, and nothing downstream needs to transform it again to make it shippable.
It is not true of the **computation**, and the honest saving is the double cap in §5, not 147.5s.

---

## 11. Step 2b — the objective is measured on the WRONG SPLIT

Found while implementing 19ia, and it is the more consequential half.

```
v, x = eval_pop(logits)                                   # SUCCESS RATE + engineering violation
_sh  = _segment_softmax(logits, …, p.max_share)           # decode
_fd  = _deliver_full(_sh)                                 # block -> elig -> cap
_band = band_penalty_fn(_fd)                              # CONSTRAINTS
```

`eval_pop` is a fused numba kernel that does **its own internal softmax** and computes the success
rate from it. It applies the **cap** (19gu put the cap in every decode path, including this one)
but **not `block` and not `elig`**.

**So the GA maximises the success rate of a split it does not ship, subject to bands measured on
the split it does.** Eligibility zeroes rows and renormalises onto the survivors, so the delivered
split's success rate is a *different number* — and generally a **lower** one, because eligibility
removes gateways the objective would otherwise have loaded up.

### What closing it takes

- `_fd` **already exists** at that point in the function, so the success rate can be computed from
  it directly — a weighted sum over the delivered array. Cheap; `eval_pop` is only 22.3 ms/call
  and 2.1% of eval, so this is not a performance question.
- The **engineering violation** (global VAMP cap + max-share) comes out of the same kernel and
  would have to move with it, or the two halves of `x` disagree about which split they describe.
- **Two call paths do not have `_fd`:** the early return when no band scoring is wired
  (`_eval_with_bands`, `_need_band` false) and `_rescore_compress`. Both need a defined answer.

### Why it gets its own run, and its own warning

**This changes the reported success rate.** Not the split, the *headline number* — 0.615322 has
been the same figure for eight runs and it will move. That is not a regression; it is the number
becoming the one that describes what ships. But it means:

- it cannot share a run with `ROUTING_DECODE_DELIV` or `ROUTING_PROJ_SFLOOR`;
- the run before it must be the armed-19ia run, so the split is already delivery-final and only
  the *measurement* changes;
- and the log must say plainly that the success rate is not comparable with any earlier run.

**SHIPPED as 19id, `ROUTING_DECODE_OBJ`, default OFF.**

- `v` (success rate) AND `x` (engineering violation) both move onto the delivered array. They come
  out of one kernel call today, so leaving the violation behind would make the two halves of the
  fitness describe different splits — the defect this closes, not a smaller version of it.
- The delivered array is computed **once** and shared with the band path and the compress term;
  the extra cost is one gather per evaluation against `eval_pop`'s 2.4% of eval.
- **`_rescore_compress` moved with it.** It re-scores the success rate on a codebook refresh, and
  leaving it on `eval_pop`'s basis would let a refresh silently change the objective mid-run —
  the hardest class of bug to see, because the numbers stay plausible.
- **Three distinct "not in effect" messages**, because there are three ways to arm it and get
  nothing: no band penalty and no compression wired (no delivered array is built on that path),
  no delivery hook supplied, and the applied case. An armed run must never quietly score half its
  calls on one basis and half on the other.

**Acceptance test:** the run's success rate WILL differ from 0.615322. That is the switch working.
What must NOT change is feasibility — the bands are already measured on the delivered split, so
arming this changes what the GA *prefers*, not what it is *allowed* to do. A run that comes back
infeasible is a finding, not a tuning problem.

---

## 12. Scope — the 20 residual keys: blocked rows and the water-fill

**Not started. Parked deliberately, because the fix is only correct if the live engine agrees.**

### What is left after 19ia/19ic

`[profiles] PART B` went from **6,428 keys / Σ|Δprop| 24.4448** to **20 keys / 0.0302**. Every one
of the 20 is on a **single BIN — `cad|485097`** — across three RPGTs, with deltas of
`+0.004973` (×3) and `-0.001248` (×5) among the eight shown.

That is the same thing `[deliv-fixed]` measures directly: **applying the transform twice moves 12
of 154,405 rows, worst |Δ| 6.212e-04.**

### The mechanism, again

1. `_fm_block` pins an auto-blocked row to the exploration floor (0.01) and redistributes.
2. `_fm_elig` zeroes ineligible rows and renormalises — which moves everything, including the
   pinned row.
3. `_fm_cap` water-fills at 0.97. Its recipient rule is `share > 1e-12 and share < cap`, **which a
   blocked row sitting at the floor satisfies.** So the cap hands it excess and lifts it off the
   floor the block just put it on.

Apply the transform again and step 1 pins it back down. Hence: no fixed point, and a residual
that no amount of re-application converges away.

### The candidate fix, and why it is not obvious

**Exclude blocked rows from receiving water-fill.** One clause in `_cap_rows`' recipient mask.

But that is **a change to what ships**, and it is only right if the **live allocation engine also
excludes them**. If production's water-fill does hand share to a floored blocked row, then the
current behaviour is *faithful* and "fixing" it would open a new reconciliation gap in the
opposite direction — the search would model something the engine does not do.

**So the first step is not code. It is answering: what does the live engine do when a blocked
gateway sits at the exploration floor and a sibling in the same profile goes over 0.97?**

### Why it is low priority

- **20 keys on one BIN**, Σ|Δprop| 0.0302.
- **Zero effect on any delivered band value** — the 21:12 run's 15 bands are identical to the
  20:37 run's, and `[rung]` reads `Σ|SPLIT| 0 (0%)`.
- It is **three orders of magnitude** below the float32 noise floor the reconciliation detector
  already runs with.

It is a correctness wart, not a number problem. Worth closing once the engine's behaviour is
confirmed; not worth guessing at.

### If it is ever picked up

1. Confirm the live engine's water-fill recipient rule for a floored blocked row.
2. If it excludes them: one clause in `_cap_rows`, mirrored in `_cb_kernel_impl`'s water-fill and
   in `_cap_shares_ref`, behind a switch, and `[deliv-fixed]` becomes the acceptance test — it
   should read **0 rows moved**.
3. If it does not exclude them: the residual is *correct* and this section should be closed with
   that finding recorded, not with a change.

---

## 13. Step 2b's first armed run FAILED, and reading the code found why

### What the 2026-09-02 21:25 run did

**Nothing.** `best success rate 0.59911` at generation 0 and at generation 320. `improved=False`.
11,840 candidates and not one beat the decoded seed — the shipped split IS the seed (checkout
1,104 and woodforest 16,498 are the targeted-move seed's 1,103 / 16,491).

`§11`'s acceptance test said *"what must not change is feasibility"*. Feasibility held. **That was
the wrong test.** The one that mattered — *the search must still improve on its seed* — was not
written down, and it is the one that failed.

### And the cost estimate was wrong by two orders of magnitude

§11 said *"one gather per evaluation against eval_pop's 2.4% of eval"*. Measured:

- `[eval-cost]` **496.1s unaccounted** — untimed work, which is exactly where `_deliver_kept` sits
- ≈ **1,476 ms/call** against `_deliver_full`'s **477 ms/call** — **the gather is 3× the delivery**
- search **475s → 963s**, throughput **25/s → 12/s**, eval **1,234 → 2,649 ms/gen**

### THE CAUSE, found by reading `_fm_gather` (tab_2 ~8154)

```python
def _fm_gather(_fd, _cm=_fm_colmap, _cs=…, _cc=…):
    _d = _D[:, _cm]                                          # gather the kept rows
    _seg = repeat(reduceat(_d, _cs, axis=1), _cc, axis=1)    # per-profile sum
    _d = where(_seg > 1e-12, _d / _seg, _d)                  # ← RENORMALISE TO SUM 1
```

**`_fm_gather` renormalises.** That is correct for the job it was written for — the
compressibility distortion needs a normalised per-profile *shape*. It is **wrong for the success
rate**, because the delivered split does **not** sum to 1 per profile: eligibility zeroes rows, so
kept mass drops below 1, and renormalising **puts back exactly the mass eligibility removed.**

So `_sd` is a normalised shape, not the delivered share, and `_success_rate(_sd)` is not the
success rate of what ships. That is explanation **B**, and it also explains the one number that
looked wrong from the start: the seed printed **0.599109** on the raw basis and the new objective's
best printed **0.59911** — the delivered figure should have differed.

**There is no kept-grain delivery reference inside the GA to have caught this**: tab_2 passes
`deliver_full_fn` and `gather_fn` but **not** `deliver_fn`, so `_deliver` is the identity and
`_deliver_kept`'s fallback is a no-op. The only path to kept grain is the one that renormalises.

### `[obj-check]` (19ie) — the verdict, on the run's own data

Fires once, only when `ROUTING_DECODE_OBJ=1`, on the live population. Four readings, because a
**wrong-but-varying** `_sd` looks exactly like a flat objective from the outside:

1. **Per-profile sums of the scored array.** The delivered split must have mass **below 1** where
   eligibility bit. All 1.0 ⇒ something renormalised ⇒ **B**.
2. **Spread and distinct-value count** of the new objective, beside eval_pop's on the same
   population.
3. **Rank correlation** between the two. A correct delivered objective is a *compression* of the
   raw one, not a scramble. Near zero ⇒ **B**, even with a healthy spread. This is the reading the
   spread test alone could not give.
4. **The raw-best candidate's two scores side by side.** Identical to 1e-9 ⇒ the transform never
   reached the objective ⇒ **B**.

### The fix, once the run confirms it

`gather_fn` cannot be reused for this. Step 2b needs the kept rows of the delivered array
**without the renormalise** — either a second hook, or the success rate computed on the
**full-grain** `_fd` directly with full-grain `vol`/`succ`, **which would also delete the 496s**
because there would be no gather at all.

**Explanation A is not ruled out and cannot be until B is fixed** — the 0.97 cap saturating the
GA's main lever is still a real possibility, and it would be a search-design problem, not a bug.
One thing at a time.

---

## §14 — 19if: the two defects, both fixed, both still behind the switch

The 22:11 run carried `[obj-check]`. **The four readings disagreed**, and the disagreement is
what identified the real fault.

| Check | Reading | Verdict |
|---|---|---|
| 1 — per-profile sums | 100% sum to 1.0 within 1e-9 | ⚠ confirms the renormalise |
| 2 — spread | 0.0556 over 40 distinct values (eval_pop: 0.0564) | not flat |
| 3 — rank correlation | **+0.9972** | not scrambled |
| 4 — transform reaching it | raw-best 0.599572 → delivered 0.598867 (Δ −0.000705) | reaching it, and lower |

Checks 2–4 exonerate the array's variation and ordering. So the zero progress was **not** B, and
not A either. The decisive arithmetic was three numbers from the run's own log:

* population **delivered** max `0.598981`
* incumbent `best_success_rate` `0.599109`
* population **raw** best `0.599572`

The incumbent sits *between* them. `seed_success_rate` is computed at
`genetic_fullmatrix.py:1966` from `s0` — the plain softmax — and `best_success_rate`/`best_key`
are seeded from it. 19id moved the **population** onto the delivered basis inside
`_eval_with_bands` and left the **seed** on the raw one, so the generation loop's
`if top_key > best_key` compared two different quantities. The delivered basis reads
systematically ~0.0007 lower, so no child could ever win. `_key_of` orders on
`(-band, -viol, success_rate)` and `seed_other` was raw-basis too — **two of the three key
components were mixed-basis**. Only `seed_band` was already delivered, which is why the band half
never showed it.

### D1 — the objective moves to full grain

`problem_from_ctx` now returns `meta["obj_full"]`: a `FullMatrixProblem`-shaped view of the same
ctx columns at `n_row` grain, in ctx row order — the order `_deliver_full` returns. The delivered
objective is scored straight off the `(P, n_row)` array. No gather, so no renormalise.

This also deletes the gather from the hot loop. `[eval-cost]` charged **439.0s** of an 853.2s
search to `unaccounted` on the 22:11 run; that row was this call. `gather_fn` itself is
**unchanged** — its renormalise is correct for its two remaining consumers (the compress
distortion, currently unreachable, and 19ia's return path).

### D2 — the seed is re-scored on the basis its challengers use

Before `seed_key` is built, when armed. `[obj-basis]` prints the before/after and then proves the
full-grain view lines up: column count vs `n_row`, profile-volume total vs the kept problem's
denominator (they must match or the two success rates are not the same quantity), and how much
delivered share sits on rows outside the genome — share the kept-grain objective could not see.

The objective is now spelled out **once**, in `_obj_scores`, and the seed, the population, the
codebook re-score and `[decode-loss]` all call it. D2 was not a hard bug to write: the objective
was spelled out at each call site and one site was missed. A future basis change now has one place
to happen, and a missed site is an `AttributeError`, not a silent scale mismatch.

### Tests — `tests/test_19if_obj_basis.py`, 14 checks

Bit-identity of the shipped split, the success rate and the seed score against **19ie**
(`ee9a5b6`, not HEAD) with the switch off; full-grain equals the un-renormalised gather; the
renormalised figure is a different and **higher** number; delivered profiles sum below 1 while the
renormalised copy sums to 1; the recorded seed score is the delivered one; the raw basis really is
higher; and `gather_fn` is called at most **twice** per run rather than once per evaluation.

### Still open

**Explanation A is still not ruled out.** With both bases aligned the search can now move, but
whether it moves *enough* — whether the 0.97 cap saturates the GA's main lever — is a question the
next armed run answers, not this commit. `ROUTING_DECODE_OBJ` stays **OFF** by default.

---

## §15 — 19ig: the 23:01 run. The search works; the ship rule threw the result away.

`ROUTING_DECODE_OBJ=1`, build 19if. **The GA climbed**: `0.59175` at generation 0 →
`0.59774` at generation 320, monotone across all 16 restarts, `improved=True`. D2 was the whole
of the zero progress. **Explanation A is dead** — the delivered objective is climbable and the
operators are fine.

`[obj-basis]` printed the mechanism with the run's own numbers: seed **0.592077 → 0.591691**
on re-score, a −0.000386 that no child could ever make up.

### D1 was a numerical no-op, and my 19ie premise was wrong

`[obj-check]` check 1b priced the renormalise at **mean Δ −0.000000, worst |Δ| 0.000000**.

Reading `eligibility.py:_apply_elig_pop`: eligibility in this pipeline is **mass-preserving**.
A ban zeroes and then `_renorm_pop`s; the wallet and USA-only stages are `_blend_pop`, which
**redistributes** the incapable share onto capable siblings rather than dropping it. A routing
profile must send 100% of its volume somewhere. So the delivered split sums to 1 per profile,
`gather_fn`'s renormalise was a no-op on it, and 19ie's assertion — "the delivered split does
NOT sum to 1" — was false.

The ⚠ that fired on a correct run is deleted. The check now warns on the **opposite** condition:
mass that has gone missing, which would mean the transform lost volume. `tests/test_19if_obj_basis.py`
keeps the toy hook that really does drop mass as the positive control for it.

### The 439s was over-attributed

| | 22:11 (19ie) | 23:01 (19if) |
|---|---|---|
| search | 853.2s | 715.7s |
| `[eval-cost] unaccounted` | 439.0s | 339.2s |

The gather was ~100s, not 439s. `_obj_scores` measures ~142 ms/call at P=35 (≈48s a run), so
~290s was still untimed — `[eval-cost]`'s "no residual to argue about" was not true. 19ig times
`_obj_scores` as its own row and puts every row of that table on **one denominator** (the five
stages were divided by their own sum and `unaccounted` by the true total, so the column read
100% and then another 49.6%).

### `_violation`: bincount, bit-identical

`np.add.at` costs **127.5 ms/call** on the live shape against **30.2 ms** for a per-candidate
`np.bincount` — 4.2×. It only started mattering with 19id; before that the population's
violation came from the numba kernel and this path ran on the seed alone.

Bit-identity is an argument, not a hope: `np.add.at(num.T, mid_id, X.T)` walks r = 0…R−1 in
order into bin `mid_id[r]`, and `bincount(mid_id, weights=X[i])` walks the same rows in the same
order into the same bin, so every output element receives the same float64 additions in the same
sequence. `[viol-bincount]` measures it anyway on the first live call, int64 bit patterns.
`ROUTING_VIOL_BINCOUNT=0` reverts.

### The engineering key is non-zero now, and that is correct

`viol 4.9485`, flat for 320 generations. It is the delivered split's real max-share overage:
`[deliv-cap]` reports **1,818,198 (candidate, profile) pairs where the cap is unsatisfiable** and
`_cap_rows` leaves those at baseline, above 0.97. `[decode-cap]`'s standing line said a non-zero
key means the water-fill failed; under this switch that would accuse it on every armed run, so
the line now states which reading is which — and notes that the key **outranks the success rate**
in `_key_of`, so a candidate that lowers it beats a better-converting one.

### `[nw-conv]` — the ship rule

```
candidate          delivered   scored   drift   GA-fitness
seed exact-proj            0        0      +0   0
GA output                  0        0      +0   0.3
breach gap 0 unit(s)   ⇒ SHIPPING THE SEED INSTEAD
```

`tab_2:9694` is `_nw_pick_ga = _nw_dg <= _nw_ds`, and conversion never enters. Both candidates
were compliant, both delivered breaches printed as 0, and the GA lost on a difference smaller
than the run's own float32 noise floor. What shipped delivers `succ=0.5918` — the seed — instead
of the GA's 0.5977. **The deciding quantity was rounding.**

`ROUTING_NW_CONV=1` makes an *indistinguishable* tie fall through to conversion: the gap must sit
inside the distinguishability bar (`_fx_bar`, which `[f32-floor]` raises to the projector's
measured float32 noise floor) **and** neither candidate may be outside it. Whenever either
candidate genuinely breaches, or the gap is real, nothing fires and the breach-alone rule stands.
With float32 off the floor is 0 and it degenerates to the strict rule by construction.

Conversion is measured **at the decision point** by one function on the delivered basis, not
taken from `_fm_info` — `info["seed_success_rate"]` is the *decoded* seed's, and comparing it
with the GA's would re-introduce exactly the mixed-basis error 19if closed.

**Default OFF.** Unarmed, the block still prints what the rule cost when the tie was
indistinguishable, so the size of the discarded gain accumulates across runs before what ships
changes.

### Still open

`[recon-breakdown]` was ⚠ UNAVAILABLE again. `[profiles]` PART B read 240 keys / Σ|Δprop| 3.7405
rather than 19ia's 20 / 0.0302 — expected, because the **seed** shipped and only the GA's return
path goes through `_deliver_kept`; it is not a regression, and it disappears if the GA ships.

---

## §16 — 19ih: the three parked items, closed

### 1. `[recon-breakdown]` — the message was inverted, not merely unhelpful

`impact_calcs.py` wrote the 19gt sentinel and then **clobbered it three lines later**:

```python
if not FORENSIC:
    globals()["_LAST_VAMP_TERMS"] = "skipped"     # 19gt's sentinel
if FORENSIC:
    try:    ...
    except: globals()["_LAST_VAMP_TERMS"] = None
else:
    globals()["_LAST_VAMP_TERMS"] = None          # <- pre-19gt branch, never removed
```

19gt added the sentinel so a **deliberate skip** could be told from a **genuine absence**, and
19hw taught `[recon-breakdown]` to read it. But the `if FORENSIC: … else:` still carried its
pre-19gt `else`, which fired on exactly the runs the sentinel existed for. So every clean run
printed *"⚠ UNAVAILABLE, **and not by the forensic gate**: the delivered VAMP-terms stash is
missing"* — when the forensic gate is precisely what skipped it. That sent the reader hunting a
defect on every run that had none.

One `if/else` now, so there is no second writer. `_LAST_VAMP_CF_SKIPPED` gets the sentinel too,
and tab_2 no longer iterates it when it is a string (that would have printed one line per
*character*).

### 2. `[lift-ab]` — a blocked A/B against a guessed floor

It timed **all `reps` ON, then all `reps` OFF**, took the mean of each, and compared against a
hard-coded 5%. Two faults compounding:

* **Blocked, not interleaved** — any machine drift *between* the two blocks lands entirely on the
  lift. That is how the same unchanged code printed 0.915x, 1.137x and 1.287x.
* **`_real = abs(_sp - 1.0) > _floor`** — direction was never constrained. The lift only ever
  *skips* passes that are provable no-ops, and the outputs are checked bit-identical, so its true
  speedup **cannot be below 1.0x**. The old test would call **0.915x "a real difference"**.

Now: **interleaved** ON/OFF/ON/OFF so drift is shared between the arms; compared on the **per-arm
minimum** (the least-contaminated sample either arm produced); and the bar is **this machine's own
noise, measured on the same rounds** — the within-arm spread, which is identical code and is
therefore noise by construction. A sub-1.0x reading is now named as a contaminated **clock**, and
reported as UNMEASURED rather than as a lift. Costs ~3× the projections, once per run.

### 3. §12 — the water-fill recipient rule: **do not fix this**

The parked question was whether the live engine excludes a blocked row sitting at the exploration
floor from receiving 0.97 water-fill. It does not:

```python
# impact_calcs._cap_rows — THE LIVE ENGINE
recip = (W > 1e-12) & (~over) & (W < _cap - 1e-12)
```

No blocked-row clause, and `_apply_blocked_caps` runs **before** `build_split_exports` reaches
`_cap_rows` (stage ≥ 3). So the live engine lifts a floored blocked row off the floor exactly as
the search's `block → elig → cap` does.

**The non-idempotence `[deliv-fixed]` reports is faithful to delivery.** Excluding blocked rows in
the search would make it *diverge* from what ships — the opposite of the goal. §12 closes with no
code change, and `[deliv-fixed]` now says so instead of leaving it open.

The residual question is a product one, not a search one: the live engine floors a gateway with
≥100 consecutive failures to 0.01 and then water-fills it back above 0.01. If that is wrong it is
wrong in `_cap_rows`, and that is where it would change.

`tests/test_19ih_parked_items.py` — 16 checks, including a fake-clock drive of `lift_ab_report`
that asserts min/min (not mean/mean), that a 13% difference inside a 40% noise bar is not called
real, and that 0.909x comes back `above_floor=False, impossible=True`.
