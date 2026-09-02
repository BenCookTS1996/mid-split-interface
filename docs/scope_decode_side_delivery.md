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

**Not started. `ROUTING_DECODE_OBJ` is the reserved name.**
