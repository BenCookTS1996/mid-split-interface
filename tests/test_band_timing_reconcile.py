"""Reconciliation: the GA band projector (routing_optimiser.band_projection.PopulationBandProjector)
must reproduce the tab-3 DELIVERED projection (app.impact_calcs.compute_vamp_prepost_granular)
for the M5 VAMP band values — the "#3 scored == delivered" guarantee.

This pins the APPEARANCE-MONTH timing fix: go-live pro_rata is applied by the month the VAMP
APPEARS (the aged row's own period), while fcp1_frac stays at ORIGINATION. The synthetic export
ramps pro_rata by month and gives each aged (period, t) row its ORIGIN month's pro_rata, so the
old origination-timed projector and the appearance-timed oracle genuinely disagree — i.e. the test
would FAIL against the pre-fix code (the raw gap is ~+26% on the banded MID here).

Isolates timing: single pmp/Country sub-cell, no wallet/USA/cap/back-fill → grain identical, so any
residual difference is timing alone. Pure pandas/numpy(+optional numba); no BigQuery/Streamlit
(streamlit is stubbed so impact_calcs imports).

Run: python tests/test_band_timing_reconcile.py
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

# --- stub streamlit so impact_calcs (which decorates with @st.cache_data) imports offline ------
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

from impact_calcs import compute_vamp_prepost_granular            # noqa: E402  (tab-3 oracle)
from routing_optimiser.band_projection import PopulationBandProjector, BandProjector  # noqa: E402

CUR, BIN, RPGT = "usd", "400001", "monthly renewal"
MIDS = ["Mid_A", "Mid_B", "Mid_C"]
RAMP = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0}   # go-live pro_rata by APPEARANCE month
FCP = 0.8
VC0 = {"Mid_A": 100.0, "Mid_B": 40.0, "Mid_C": 20.0}
VI0 = {"Mid_A": 1000.0, "Mid_B": 800.0, "Mid_C": 600.0}
PROP = {"Mid_A": 0.2, "Mid_B": 0.5, "Mid_C": 0.3}


def _build_export(ramp):
    rows = []
    for per in range(6):
        for t in range(per + 1):
            origin = per - t
            for mid in MIDS:
                rows.append(dict(
                    vampMid=mid, RPGT=RPGT, BIN=BIN, Currency=CUR, period=per, t=t,
                    vampCount=VC0[mid] * (0.9 ** t), VI_Txn_Count=(VI0[mid] if t == 0 else 0.0),
                    pro_rata=ramp[origin], fcp1_frac=FCP))    # aged rows carry ORIGIN-month pro_rata
    return pd.DataFrame(rows)


def _oracle_m5(csv):
    prop = tuple((CUR, BIN, RPGT, m, PROP[m]) for m in MIDS)   # 5-tuple = by_rpgt, non-enforced
    gran = compute_vamp_prepost_granular(csv, prop, scoped_rpgts=())
    return gran[gran["period"] == 5].groupby("vampMid")["VAMP_Post"].sum()


def _build_scaffold(exp):
    """T0/Pc/pool at band_projection's column names, with the APPEARANCE-timed pool
    (pr[appearance] × Σ_mid vc·fcp[origin]) — the same recipe as tab2_engine's _Pc_movedvpool_a."""
    e = exp.copy()
    e["cur"] = CUR; e["bin"] = BIN; e["rpgt"] = RPGT; e["pmp"] = "_all_"; e["ctry"] = "_all_"
    e["mid"] = e["vampMid"]; e["midl"] = e["vampMid"].str.lower()
    e["per"] = e["period"].astype(int); e["t"] = e["t"].astype(int)
    e["vi"] = e["VI_Txn_Count"].astype(float); e["vc"] = e["vampCount"].astype(float)
    e["pr"] = e["pro_rata"].astype(float); e["fcp"] = e["fcp1_frac"].astype(float)
    capped = {m.lower() for m in MIDS}   # all three MIDs banded → all appear in Pc for comparison
    T0 = e[e["t"] == 0].copy()
    T0["bf"] = 0; T0["excl"] = False; T0["emask"] = False
    T0["iscap"] = T0["midl"].isin(capped); T0["_av"] = np.where(T0["excl"], 0.0, T0["vi"])
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


def _proj_m5(T0, Pc, pool, numba):
    bands = [(m.lower(), 5) for m in MIDS]
    pbp = PopulationBandProjector(T0, Pc, pool, bands=bands, by_rpgt=True)
    kpos = {k: j for j, k in enumerate(pbp.prop_keys)}
    pr = np.zeros((1, pbp._K))
    for m in MIDS:
        kk = "|".join([CUR, BIN, RPGT, m])   # prop key keeps MID case (cur/rpgt are lowercased)
        if kk in kpos:
            pr[0, kpos[kk]] += PROP[m]
    vamp, _ = (pbp.project_pop_numba(pr) if numba else pbp.project_pop(pr))
    return {m: float(vamp[0, pbp.band_order.index((m.lower(), 5))]) for m in MIDS}


def _run(ramp, label):
    exp = _build_export(ramp)
    fd, csv = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    try:
        exp.to_csv(csv, index=False)
        orc = _oracle_m5(csv)
        T0, Pc, pool = _build_scaffold(exp)
        for numba in (False, True):
            got = _proj_m5(T0, Pc, pool, numba)
            for m in MIDS:
                o = float(orc.get(m, 0.0))
                assert abs(got[m] - o) < 1e-6, \
                    f"[{label}] {'numba' if numba else 'numpy'} {m}: proj {got[m]:.6f} != oracle {o:.6f}"
        # non-population BandProjector (GATE-1 diagnostic collapse) must agree too
        bp = BandProjector(T0, Pc, pool, bands=[(m.lower(), 5) for m in MIDS], by_rpgt=True)
        bpo = bp.project({(CUR, BIN, RPGT, m): PROP[m] for m in MIDS})
        for m in MIDS:
            o = float(orc.get(m, 0.0)); v = bpo[(m.lower(), 5)][0]
            assert abs(v - o) < 1e-6, f"[{label}] BandProjector {m}: {v:.6f} != oracle {o:.6f}"
        print(f"{label}: projector (numpy & numba) == tab-3 oracle for all MIDs  "
              f"(Mid_A M5={float(orc['Mid_A']):.3f})  ✓")
    finally:
        os.unlink(csv)


def test_appearance_timing_matches_oracle():
    _run(RAMP, "A. ramped pro_rata (origination != appearance)")


def test_constant_prorata_invariant():
    _run({k: 0.7 for k in range(6)}, "B. flat pro_rata (timing-invariant sanity)")


if __name__ == "__main__":
    test_appearance_timing_matches_oracle()
    test_constant_prorata_invariant()
    print("PASS ✓  GA band projector reconciles with the tab-3 delivered projection (appearance timing)")
