# CLAUDE.md — project context & concept glossary

Context and plain-English definitions for the **Transaction Routing Optimiser**. Aimed at
anyone (technical or not) picking the project up. For setup/run steps and the file layout see
`README.md`; for a per-function reference see `FUNCTION_GLOSSARY.md`.

**The one analogy to anchor everything:** each payment gateway is a **door** a payment can go
through. Some doors get more payments approved; some are riskier (more chargebacks). The tool
decides **what share of each group of payments to send through each door**.

---

## ⚠ VOCABULARY — PROFILE vs CELL (read this before any log line)

**This is the project's terminology. Where the CODE disagrees, the code is wrong, not this.**

| term | means | key | parts |
|---|---|---|---|
| **PROFILE** | a group of payments that get routed together — one currency, one BIN, one RPGT, one payment-method, one country. **No gateway.** | `cur\|bin\|rpgt\|pmp\|ctry` | 5 |
| **CELL** | one **door within** a profile — a profile plus the gateway/MID serving it. This is the thing a share is attached to. | `cur\|bin\|rpgt\|pmp\|ctry\|mid` | 6 |

So: **a profile contains many cells** — roughly 10 on the live scaffold (14,852 profiles,
154,405 cells). A split assigns a share to every cell; the shares of the cells inside one
profile sum to 1.

### The code currently has these INVERTED

Identifiers and many log strings use `cell` for the 5-part group and `prop-key` / `profile` for
the 6-part row — the opposite of the table above. That inversion is why the run log reads
confusingly, and it is being corrected outward-in:

* **run-log prose** — being migrated to the table above (19hg onward)
* **comments** — follow the prose
* **identifiers** (`gcode`, `cell_start`, `n_cells`, `CellProblem`, `by_subcell`, …) — NOT yet
  renamed: ~3,000 occurrences across `app/` and `src/`, including numba kernel parameters and
  dataclass fields. A half-finished identifier rename is worse than none, so it is a dedicated
  mechanical pass, not something to do opportunistically.

**When reading code, translate:** an identifier saying `cell` almost always means this
document's PROFILE; a `prop_key` / `propidx` almost always means this document's CELL.

---

## How the app is laid out (post-refactor)

The Streamlit UI used to be one ~10,000-line file. It's now split so each tab is its own file
and the entry point just wires them together:

```
app/streamlit_app.py                 entry point / orchestrator (imports, setup, st.tabs())
app/app_common.py                    shared constants, log handler, path resolvers, helpers
app/impact_calcs.py                  before -> after impact projection + config templates
app/tab_1_1_build_baseline.py        Tab 1 · 1 — Build Baseline  (+ hosts tab 1's sub-tab bar)
app/tab_1_2_validate_split.py        Tab 1 · 2 — Validate Split
app/tab_1_3_config_validation.py     Tab 1 · 3 — Config Validation
app/tab_2_routing_engine.py          Tab 2     — Routing engine
app/tab_3_split_outputs_impact.py    Tab 3     — Split, outputs & impact
app/tab_4_generate_configs.py        Tab 4     — Generate configs
src/routing_optimiser/               the engines + all the maths (the "brains")
```

NAMING RULE (19ft). A file that renders UI is `tab_<tab>[_<sub-tab>]_<what it is>.py`, numbered
by the labels the user sees. Files with no `render()` (streamlit_app, app_common, impact_calcs)
stay unnumbered because they belong to no single tab. The app/ folder must stay
FLAT: Streamlit puts only the entry script's own directory on sys.path, so `import
tab_2_routing_engine` works and `from tabs.tab_2 import ...` would not.

`app/` is the frontend; `src/` is the backend (engines + pipelines, no Streamlit import).

CONFIG INPUTS (19ft). `config/inputs/` has a `visa/` and a `mastercard/` subfolder holding that
scheme's own prefixed copy of each JSON, with the bare root file as a shared fallback. Nothing
reads a path to these directly — go through `app_common.input_json_path(name[, scheme])`.

Every function in the codebase carries a stable id tag in its docstring/comment, e.g.
`# [FN-042]`, which `FUNCTION_GLOSSARY.md` uses to describe it.

---

## Key concepts (plain English + an analogy each)

- **Cell** — one `bank × currency × transaction-type` group. One routing decision is made per cell.
  *Analogy: one row on a delivery schedule — "parcels of THIS type, to THIS region, via which couriers?"*

- **Gateway / MID** — a payment route (a "door"). A MID is a merchant account at a gateway.

- **Split** — the plan: what % of a cell's payments goes through each door. Adds up to 100%.
  *Analogy: how you divide a pizza between people — the slices must sum to the whole pie.*

- **Success rate** — share of payments approved. Higher is better.

- **Risk / VAMP rate** — share that become chargebacks/fraud flags. Lower is better. **VAMP** is
  Visa's fraud-monitoring programme; go over its limit and you're penalised, so we keep risk under it.
  *Analogy: a speeding limit — cross it and you get fined, so you stay under.*

- **BIN** — the first digits of a card number identifying the issuing bank; the fine-grained level
  real risk and deployed configs use.

- **Baseline / "pre"** — how things are routed today; every proposal is compared against this "before".

- **Eligibility** — hard yes/no route rules: banned doors, doors that can't do wallet (Apple/Google
  Pay), doors that only serve USA traffic. Ineligible doors get zero share.
  *Analogy: a guest list — some doors simply aren't allowed for certain payments.*

- **Pool / config** — one deployable routing rule (a JSON file). "Pools" = how many rules ship.

- **Compression** — grouping near-identical cells so they can share one rule (fewer files to deploy).
  *Analogy: instead of a bespoke recipe card for every dish, one card for all the "near-identical" dishes.*

### Empirical Bayes (the smoothing behind our success rates)

**Short version:** when a gateway has only a little data, don't trust its raw rate on its own —
pull it toward the average of similar gateways, and trust its own number more as it collects more
data. The **"empirical"** part means the group average *and* how hard to pull toward it are
**learned from the data itself**, not set by hand.

**Analogy:** judging a brand-new restaurant that has only **two** reviews. You don't take those
two reviews at face value — you assume it's *probably about as good as similar restaurants nearby*
(the group average), and you only let its own reviews move your opinion as more of them come in. A
gateway with a handful of transactions is treated the same way: it starts near its peer-group
average and earns its own rate as the evidence piles up. This stops a gateway that got lucky on
"2 out of 2" from looking like a flawless 100%.

*(In the code this is `κ` / "shrink strength": think of it as a number of "pretend" prior
transactions mixed in. More real data → the pretend ones matter less → the gateway's own rate wins.)*

---

## The four engines (what "best" means to each, with an analogy)

All four decide the same thing — how to divide each cell's payments between its doors — and all
read the same inputs, so you can switch between them from a dropdown. Ineligible doors are always
filtered out afterwards. **Genetic is the default (what runs in production).**

### 1. Genetic (CMA-ES) — the default

**What it does:** keeps a whole *population* of candidate splits, scores each one, keeps the best
few, then breeds and mutates them into a new generation and repeats — many rounds. Crucially it's
a **CMA-ES** search, which means it also **adapts its own search spread each round**: it learns
which directions of change actually help and takes bigger, smarter steps there, and shrinks away
from the directions that don't.

**Why it's the default:** it's the best at juggling *all* the limits at once (risk caps, per-bank
monthly targets) while keeping conversion high — the other three optimise mostly for conversion and
lean on a simpler risk step.

**Analogy:** **selective breeding.** Start with a field of plants (candidate splits), keep the
strongest, cross-pollinate and mutate them, and repeat over generations — each generation a bit
better than the last. The "CMA-ES adapts its spread" part is like a breeder who, over time, *learns
which traits matter most* and focuses their crossing there instead of breeding at random.

### 2. Softmax — steady, temperature-controlled

**What it does:** sends **more** traffic to the doors that get more payments approved, but never
everything to one door — it always keeps a spread (an exploration floor). A **temperature** dial
controls how sharply it concentrates: *hot* = pile onto the single best door (winner-takes-most),
*cold* = spread evenly (hedge your bets).

**Analogy:** **a manager assigning shifts by skill.** The strongest staff get more hours, but nobody
is benched entirely. The temperature is how aggressively the manager rewards the top performer versus
sharing the hours around.

### 3. Thompson (bandit) — explores thinly-tested doors

**What it does:** mostly backs whichever door is *probably* the best right now, but deliberately
keeps giving newer / less-tested doors a few tries in case one turns out great. The less a door has
been tested (the wider our uncertainty about it), the more benefit of the doubt it gets.

**Analogy:** **a gambler at a row of slot machines.** They mostly play the one paying out best so
far, but keep pulling the others now and then — because a machine they've barely tried might secretly
be the best one, and you only find out by trying. "Explore a little so you never miss a hidden winner."

### 4. Portfolio (mean-CVaR) — avoids nasty surprises

**What it does:** treats doors like investments — conversion is the "return", and a possible **risk
spike** (a sudden jump in chargebacks) is the "danger". Unlike plain variance, which would also
penalise pleasant surprises, it prices only the **bad tail** — the plausible worst case — and
diversifies away from volatile or barely-tested doors. It gives up a sliver of average success rate
in exchange for far fewer disasters. Risk aversion is auto-tuned per cell (no dial to set).

**Analogy:** **spreading your savings across several accounts.** You favour good returns but steer
clear of the wildly unpredictable options that could crash — happy to accept slightly lower average
growth for many fewer sleepless nights. (Contrast with Thompson, which *explores* thin doors; this
one *avoids* them.)

> Softmax, Thompson and Portfolio share the same second half; they differ only in how they build
> the starting "reference" split at the top. Genetic is a different beast — a search, not a formula.
