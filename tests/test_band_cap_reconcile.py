"""Reconciliation: the GA band projector's per-sub-cell MAX-SHARE CAP (max_share, e.g. 0.97)
must reproduce the DELIVERED projection — i.e. compute_vamp_prepost_granular fed the SAME shares
after the build_split_exports-style water-fill cap.

The max-share cap was proven (on real TotalAV data) to be the ENTIRE scored-vs-delivered VAMP
residual: applying just the 0.97 cap to the GA's coarse shares reproduces tab-3's delivered VAMP
within ±3. This test locks that in on synthetic data where one MID dominates a sub-cell at 98%
(> 97%), so the cap MUST fire:

  * projector with max_share=0.97, fed the UNCAPPED shares, caps internally →
    must equal compute_vamp_prepost_granular fed the hand-capped shares (numpy AND numba);
  * projector with max_share=1.0 (cap OFF) must equal the uncapped oracle (no regression to the
    appearance-timing behaviour).

Pure pandas/numpy(+optional numba); streamlit stubbed so impact_calcs imports offline.
Run: python tests/test_band_cap_reconcile.py
"""
import os
import sys
import types
import tempfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "app"))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")

    def _passthrough(**_kw):
        def _deco(f):
            return f
        return _deco

    _st.cache_data = _passthrough
    _st.experimental_memo = _passthrough
    _st.session_state = {}
    sys.modules["streamlit"] = _st

from impact_calcs import compute_vamp_prepost_granular            # noqa: E402
from routing_optimiser.band_projection import PopulationBandProjector  # noqa: E402

CUR, BIN, RPGT = "usd", "400001", "monthly renewal"
MIDS = ["Mid_A", "Mid_B", "Mid_C"]
RAMP = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0}
FCP = 0.8
VC0 = {"Mid_A": 100.0, "Mid_B": 40.0, "Mid_C": 20.0}
VI0 = {"Mid_A": 1000.0, "Mid_B": 800.0, "Mid_C": 600.0}
PROP = {"Mid_A": 0.98, "Mid_B": 0.015, "Mid_C": 0.005}   # Mid_A > 0.97 → cap fires
CAP = 0.97


def _waterfill(sh, cap=CAP):
    """MID-grain water-fill matching build_split_exports._cap_rows (>=2 gateways, 50 sweeps)."""
    W = dict(sh)
    if sum(1 for v in W.values() if v > 1e-12) < 2:
        return W
    for _ in range(50):
        over = {m: v for m, v in W.items() if v > cap + 1e-12}
        if not over:
            break
        exc = sum(v - cap for v in over.values())
        for m in over:
            W[m] = cap
        room = {m: cap - W[m] for m in W if 1e-12 < W[m] < cap - 1e-12}
        rs = sum(room.values())
        if rs <= 1e-12:
            break
        for m in room:
            W[m] += room[m] / rs * exc
    return W


def _build_export():
    rows = []
    for per in range(6):
        for t in range(per + 1):
            origin = per - t
            for m in MIDS:
                rows.append(dict(
                    vampMid=m, RPGT=RPGT, BIN=BIN, Currency=CUR, period=per, t=t,
                    vampCount=VC0[m] * (0.9 ** t), VI_Txn_Count=(VI0[m] if t == 0 else 0.0),
                    pro_rata=RAMP[origin], fcp1_frac=FCP))
    return pd.DataFrame(rows)


def _oracle(csv, shares):
    prop = tuple((CUR, BIN, RPGT, m, shares[m]) for m in MIDS)
    gran = compute_vamp_prepost_granular(csv, prop, scoped_rpgts=())
    return gran[gran["period"] == 5].groupby("vampMid")["VAMP_Post"].sum()


def _scaffold(exp):
    e = exp.copy()
    e["cur"] = CUR; e["bin"] = BIN; e["rpgt"] = RPGT; e["pmp"] = "_all_"; e["ctry"] = "_all_"
    e["mid"] = e["vampMid"]; e["midl"] = e["vampMid"].str.lower()
    e["per"] = e["period"].astype(int); e["t"] = e["t"].astype(int)
    e["vi"] = e["VI_Txn_Count"].astype(float); e["vc"] = e["vampCount"].astype(float)
    e["pr"] = e["pro_rata"].astype(float); e["fcp"] = e["fcp1_frac"].astype(float)
    capped = {m.lower() for m in MIDS}
    T0 = e[e["t"] == 0].copy()
    T0["bf"] = 0; T0["excl"] = False; T0["emask"] = False
    T0["iscap"] = T0["midl"].isin(capped); T0["_av"] = T0["vi"]
    Pc = e[e["midl"].isin(capped)].copy()
    t0 = e[e["t"] == 0]
    prmap = (t0.drop_duplicates(["cur", "bin", "rpgt", "pmp", "ctry", "per"])
             .set_index(["cur", "bin", "rpgt", "pmp", "ctry", "per"])["pr"].to_dict())
    fcpmap = t0.set_index(["cur", "bin", "rpgt", "pmp", "ctry", "midl", "per"])["fcp"].to_dict()
    ps = e.copy(); ps["origin"] = ps["per"] - ps["t"]
    ps["fcpo"] = [fcpmap.get((r.cur, r.bin, r.rpgt, r.pmp, r.ctry, r.midl, r.origin), 0.0)
                  for r in ps.itertuples()]
    ps["mvraw"] = ps["vc"] * ps["fcpo"]
    agg = ps.groupby(["cur", "bin", "rpgt", "pmp", "ctry", "per", "t"])["mvraw"].sum().to_dict()
    pool = np.array([agg.get((r.cur, r.bin, r.rpgt, r.pmp, r.ctry, r.per, r.t), 0.0)
                     * prmap.get((r.cur, r.bin, r.rpgt, r.pmp, r.ctry, r.per), 0.0)
                     for r in Pc.itertuples()], float)
    return T0.reset_index(drop=True), Pc.reset_index(drop=True), pool


def _proj(T0, Pc, pool, cap, numba):
    pbp = PopulationBandProjector(T0, Pc, pool, bands=[(m.lower(), 5) for m in MIDS],
                                  by_rpgt=True, max_share=cap)
    kpos = {k: j for j, k in enumerate(pbp.prop_keys)}
    pr = np.zeros((1, pbp._K))
    for m in MIDS:
        kk = "|".join([CUR, BIN, RPGT, m])
        if kk in kpos:
            pr[0, kpos[kk]] += PROP[m]     # UNCAPPED shares — projector caps internally
    vamp, _ = (pbp.project_pop_numba(pr) if numba else pbp.project_pop(pr))
    return {m: float(vamp[0, pbp.band_order.index((m.lower(), 5))]) for m in MIDS}


def test_cap_matches_delivered():
    exp = _build_export()
    fd, csv = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    try:
        exp.to_csv(csv, index=False)
        orc_capped = _oracle(csv, _waterfill(PROP))    # delivered = oracle fed the capped shares
        orc_uncap = _oracle(csv, PROP)                 # cap-off reference
        T0, Pc, pool = _scaffold(exp)
        assert any(abs(orc_capped.get(m, 0) - orc_uncap.get(m, 0)) > 1e-3 for m in MIDS), \
            "test is inert: the cap must change VAMP"
        for numba in (False, True):
            on = _proj(T0, Pc, pool, CAP, numba)
            off = _proj(T0, Pc, pool, 1.0, numba)
            tag = "numba" if numba else "numpy"
            for m in MIDS:
                assert abs(on[m] - orc_capped.get(m, 0)) < 1e-5, \
                    f"cap-ON {tag} {m}: {on[m]:.5f} != delivered {orc_capped.get(m,0):.5f}"
                assert abs(off[m] - orc_uncap.get(m, 0)) < 1e-5, \
                    f"cap-OFF {tag} {m}: {off[m]:.5f} != uncapped {orc_uncap.get(m,0):.5f}"
        print(f"cap ON == delivered (oracle+cap); cap OFF == uncapped; numpy & numba  "
              f"(Mid_A M5 {orc_uncap['Mid_A']:.1f}→{orc_capped['Mid_A']:.1f})  ✓")
    finally:
        os.unlink(csv)


if __name__ == "__main__":
    test_cap_matches_delivered()
    print("PASS ✓  max-share cap fold-in reconciles with the delivered projection")
