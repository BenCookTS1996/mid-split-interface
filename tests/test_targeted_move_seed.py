"""Validate solve_targeted_moves (headroom-aware targeted move-operator seed).

Mock projector: a MID's M5 value = Σ over its rows of (share × cell_vol × risk × movable_frac) — i.e.
VAMP ∝ volume (the property the real period-5 data confirmed), and EXACTLY the proxy the operator
uses for its recipient budget, so the budget/steering logic is tested under its own assumptions.

Checks:
  A. sheds a breached ceiling MID onto a genuinely-slack (no-ceiling) sibling → clears it;
  B. STEERS onto the slack sibling and leaves a near-full capped sibling UNTOUCHED (no whack-a-mole);
  C. when every recipient is at its own ceiling (zero headroom) it makes NO move → returns base
     unchanged (never relocates a breach);
  D. respects max_share.

Run: python tests/test_targeted_move_seed.py   (numpy + scipy + repo band_scoring)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser.exact_band_solver import solve_targeted_moves  # noqa: E402

NAMES = ["risky", "midcap", "slack"]
FRAC = 0.8


class MockEB:
    def __init__(self, specs):
        self.specs = specs

    def report(self, prop_raw):
        pr = np.asarray(prop_raw, float).reshape(-1)
        return [{"midl": s.midl, "ceil": s.ceil, "floor": None, "now": float(pr[i])}
                for i, s in enumerate(self.specs)]

    def penalty(self, prop_raw):
        pr = np.asarray(prop_raw, float)
        if pr.ndim == 1:
            pr = pr[None, :]
        tot = np.zeros(pr.shape[0])
        for i, s in enumerate(self.specs):
            if s.ceil is not None:
                tot = tot + np.maximum(pr[:, i] / s.ceil - 1.0, 0.0) ** 2
        return tot


def build(base, ceilings, risks=(0.05, 0.03, 0.001), ncells=4, cell_vol=1000.0, elig_mask=None):
    mid_id, risk, shares, cs, cc = [], [], [], [], []
    a = 0
    for _ in range(ncells):
        cs.append(a); cc.append(3); a += 3
        mid_id += [0, 1, 2]; risk += list(risks); shares += list(base)
    N = 3 * ncells
    mid_id = np.array(mid_id); risk = np.array(risk); shares = np.array(shares, float)
    cs = np.array(cs, np.intp); cc = np.array(cc, np.intp)
    cell_of = np.repeat(np.arange(ncells), 3)
    inc_data = cell_vol * np.maximum(risk, 1e-9) * FRAC      # VAMP per unit share = operator's dens
    inc = sp.csr_matrix((inc_data, (mid_id, np.arange(N))), shape=(3, N))
    specs = [SimpleNamespace(midl=NAMES[i], ceil=ceilings.get(NAMES[i])) for i in range(3)]
    elig = np.ones(N) if elig_mask is None else np.asarray(elig_mask, float)
    cvol = np.full(ncells, cell_vol)
    return MockEB(specs), inc, shares, cs, cc, elig, mid_id, risk, cvol, NAMES


def _now(eb, inc, s):
    return {r["midl"]: r["now"] for r in eb.report((inc @ s[None, :].T).T)}


def test_A_slack_clears():
    eb, inc, s0, cs, cc, elig, mid_id, risk, cvol, names = build((0.5, 0.3, 0.2), {"risky": 60.0})
    out, info = solve_targeted_moves(eb, inc, s0, cs, cc, elig, mid_id=mid_id, risk=risk,
                                     cell_vol=cvol, mid_names=names, max_share=0.97, movable_frac=FRAC)
    now = _now(eb, inc, out)
    assert info["ok"] and info["breach"] < info["breach0"] - 1e-12
    assert now["risky"] <= 60.0 + 1e-6, f"risky not cleared: {now['risky']:.2f}"
    for a, c in zip(cs, cc):
        assert abs(out[a:a + c].sum() - 1.0) < 1e-9
    print(f"A. slack recipient clears risky: {80.0:.0f} → {now['risky']:.0f} (ceil 60)  ✓")


def test_B_steers_to_slack_not_capped():
    # midcap near its ceiling (small headroom), slack has NO ceiling → must steer to slack, leave midcap.
    eb, inc, s0, cs, cc, elig, mid_id, risk, cvol, names = build(
        (0.5, 0.3, 0.2), {"risky": 60.0, "midcap": 40.0})   # midcap now 28.8, hr 11.2; slack inf
    m0 = _now(eb, inc, s0)
    out, info = solve_targeted_moves(eb, inc, s0, cs, cc, elig, mid_id=mid_id, risk=risk,
                                     cell_vol=cvol, mid_names=names, max_share=0.97, movable_frac=FRAC)
    now = _now(eb, inc, out)
    assert info["ok"] and now["risky"] <= 60.0 + 1e-6, "risky should clear via slack"
    assert now["midcap"] <= 40.0 + 1e-6, "capped sibling must not be pushed over its ceiling"
    assert abs(now["midcap"] - m0["midcap"]) < 1e-6, "capped sibling should be left untouched (slack preferred)"
    print(f"B. steered to slack; midcap untouched at {now['midcap']:.1f} (ceil 40); risky {now['risky']:.0f}  ✓")


def test_C_zero_headroom_returns_base():
    # every recipient sits AT its own ceiling → no headroom → no move → base unchanged (never relocate).
    eb, inc, s0, cs, cc, elig, mid_id, risk, cvol, names = build(
        (0.4, 0.3, 0.3), {"risky": 60.0, "midcap": 48.0, "slack": 48.0}, risks=(0.05, 0.05, 0.05))
    # risky now = 0.4*40*4 = 64 > 60; midcap/slack now = 0.3*40*4 = 48 == ceiling (hr 0)
    out, info = solve_targeted_moves(eb, inc, s0, cs, cc, elig, mid_id=mid_id, risk=risk,
                                     cell_vol=cvol, mid_names=names, max_share=0.97, movable_frac=FRAC)
    assert np.array_equal(out, s0), "no recipient headroom → must return base unchanged"
    assert not info["ok"], "no-improvement must report ok=False (breach only relocates)"
    print("C. all recipients at ceiling → no move, base returned (breach not relocated)  ✓")


def test_D_max_share():
    eb, inc, s0, cs, cc, elig, mid_id, risk, cvol, names = build((0.5, 0.3, 0.2), {"risky": 60.0})
    out, info = solve_targeted_moves(eb, inc, s0, cs, cc, elig, mid_id=mid_id, risk=risk,
                                     cell_vol=cvol, mid_names=names, max_share=0.6, movable_frac=FRAC)
    assert out.max() <= 0.6 + 1e-9, f"max_share violated: {out.max()}"
    print(f"D. max_share respected (max {out.max():.3f} ≤ 0.6)  ✓")


if __name__ == "__main__":
    test_A_slack_clears()
    test_B_steers_to_slack_not_capped()
    test_C_zero_headroom_returns_base()
    test_D_max_share()
    print("PASS ✓  headroom-aware move operator: clears via slack, steers around capped siblings, "
          "never relocates a breach, respects max_share")
