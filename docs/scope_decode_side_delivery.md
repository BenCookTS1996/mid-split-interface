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
2. **`ROUTING_DECODE_DELIV`, default OFF.** Move `block` and `elig` inside the decode, in their
   current order, in front of the decode's existing cap.
3. **Keep `_fm_deliv` in the chain and PROVE it is a no-op.** With the decode doing the work,
   `_fm_deliv(decode(z))` must equal `decode(z)`. Assert it on the live population, the same
   discipline `[deliv-fuse]` and `[decode-cap]` already use. This is the acceptance test, and it
   is what turns "should be equivalent" into a number.
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
