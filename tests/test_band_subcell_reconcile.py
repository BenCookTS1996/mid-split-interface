"""Reconciliation: PopulationBandProjector with by_subcell=True reproduces the tab-3 delivered
projection when a cell's SUB-CELLS carry DIFFERENT per-gateway shares — the thing the old
cell-grain broadcast could not represent.

One cell (usd|400001|monthly renewal), two sub-cells:
  * non_gp_ap/usa : MID A share 0.30, MID B 0.70
  * applepay/usa  : MID A share 0.90, MID C 0.10
MID A therefore has DIFFERENT shares per sub-cell (0.30 vs 0.90). by_subcell must project each
sub-cell distinctly and match compute_vamp_prepost_granular fed the same enforced 7-tuples.

streamlit stubbed; pure pandas/numpy(+optional numba). Run: python tests/test_band_subcell_reconcile.py
"""
import os, sys, types, tempfile
import numpy as np, pandas as pd

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_H, "..", "app"))
sys.path.insert(0, os.path.join(_H, "..", "src"))
if "streamlit" not in sys.modules:
    _st = types.ModuleType("streamlit")
    _pt = lambda **k: (lambda f: f)
    _st.cache_data = _pt; _st.experimental_memo = _pt; _st.session_state = {}
    sys.modules["streamlit"] = _st
from impact_calcs import compute_vamp_prepost_granular   # noqa: E402
from routing_optimiser.band_projection import PopulationBandProjector  # noqa: E402

CUR, BIN, RPGT = "usd", "400001", "monthly renewal"
RAMP = {0: .5, 1: .6, 2: .7, 3: .8, 4: .9, 5: 1.0}; FCP = 0.8
# (pmp, ctry) -> {mid: (share, vc0, vi0)}
SUB = {
    ("non_gp_ap", "usa"): {"A": (0.30, 100.0, 1000.0), "B": (0.70, 40.0, 800.0)},
    ("applepay", "usa"):  {"A": (0.90, 60.0, 600.0),  "C": (0.10, 20.0, 500.0)},
}
BANDS = ["a"]   # band MID A at M5


def _export():
    rows = []
    for (pmp, ctry), mids in SUB.items():
        for per in range(6):
            for t in range(per + 1):
                for mid, (_sh, vc0, vi0) in mids.items():
                    rows.append(dict(vampMid=mid, RPGT="Monthly Renewal", BIN=BIN, Currency=CUR,
                                     paymentMethodProvider=pmp, Country=ctry, period=per, t=t,
                                     vampCount=vc0 * (0.9 ** t), VI_Txn_Count=(vi0 if t == 0 else 0.0),
                                     pro_rata=RAMP[per - t], fcp1_frac=FCP))
    return pd.DataFrame(rows)


def _oracle(csv):
    prop = tuple((CUR, BIN, "Monthly Renewal", pmp, ctry, mid, sh)
                 for (pmp, ctry), mids in SUB.items() for mid, (sh, *_r) in mids.items())
    g = compute_vamp_prepost_granular(csv, prop, scoped_rpgts=())
    return g[g["period"] == 5].groupby("vampMid")["VAMP_Post"].sum()


def _scaffold(exp):
    e = exp.copy()
    e["cur"] = CUR; e["bin"] = BIN; e["rpgt"] = RPGT
    e["pmp"] = e["paymentMethodProvider"].str.lower(); e["ctry"] = e["Country"].str.lower()
    e["mid"] = e["vampMid"]; e["midl"] = e["vampMid"].str.lower()
    e["per"] = e["period"].astype(int); e["t"] = e["t"].astype(int)
    e["vi"] = e["VI_Txn_Count"].astype(float); e["vc"] = e["vampCount"].astype(float)
    e["pr"] = e["pro_rata"].astype(float); e["fcp"] = e["fcp1_frac"].astype(float)
    capped = {"a"}
    T0 = e[e["t"] == 0].copy()
    T0["bf"] = 0; T0["excl"] = False; T0["emask"] = False
    T0["iscap"] = T0["midl"].isin(capped); T0["_av"] = T0["vi"]
    Pc = e[e["midl"].isin(capped)].copy()
    t0 = e[e["t"] == 0]
    prm = t0.drop_duplicates(["cur", "bin", "rpgt", "pmp", "ctry", "per"]).set_index(
        ["cur", "bin", "rpgt", "pmp", "ctry", "per"])["pr"].to_dict()
    fm = t0.set_index(["cur", "bin", "rpgt", "pmp", "ctry", "midl", "per"])["fcp"].to_dict()
    ps = e.copy(); ps["o"] = ps["per"] - ps["t"]
    ps["fo"] = [fm.get((r.cur, r.bin, r.rpgt, r.pmp, r.ctry, r.midl, r.o), 0.0) for r in ps.itertuples()]
    ps["mv"] = ps["vc"] * ps["fo"]
    ag = ps.groupby(["cur", "bin", "rpgt", "pmp", "ctry", "per", "t"])["mv"].sum().to_dict()
    pool = np.array([ag.get((r.cur, r.bin, r.rpgt, r.pmp, r.ctry, r.per, r.t), 0.0)
                     * prm.get((r.cur, r.bin, r.rpgt, r.pmp, r.ctry, r.per), 0.0) for r in Pc.itertuples()], float)
    return T0.reset_index(drop=True), Pc.reset_index(drop=True), pool


def _proj(T0, Pc, pool, numba):
    pbp = PopulationBandProjector(T0, Pc, pool, bands=[(m, 5) for m in BANDS],
                                  by_rpgt=True, by_subcell=True)
    prop = {(CUR, BIN, RPGT, pmp, ctry, mid): sh
            for (pmp, ctry), mids in SUB.items() for mid, (sh, *_r) in mids.items()}
    # build prop_raw in prop_keys order via project_pop_from_props
    v, _ = (pbp.project_pop_numba(_mat(pbp, prop)) if numba else pbp.project_pop(_mat(pbp, prop)))
    return {m: float(v[0, pbp.band_order.index((m, 5))]) for m in BANDS}, pbp


def _mat(pbp, prop):
    from routing_optimiser.band_projection import _prop_key_str
    kpos = {k: j for j, k in enumerate(pbp.prop_keys)}
    pr = np.zeros((1, pbp._K))
    for k, v in prop.items():
        kk = _prop_key_str(k, True, True)
        if kk in kpos:
            pr[0, kpos[kk]] += v
    return pr


def test_subcell_matches_oracle():
    exp = _export()
    fd, csv = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    try:
        exp.to_csv(csv, index=False)
        orc = _oracle(csv); T0, Pc, pool = _scaffold(exp)
        (vn, pbp) = _proj(T0, Pc, pool, False); (vb, _) = _proj(T0, Pc, pool, True)
        # prop_keys must be sub-cell (6 fields) and A must appear TWICE (once per sub-cell)
        akeys = [k for k in pbp.prop_keys if k.endswith("|A")]
        assert len(akeys) == 2, f"MID A should have 2 sub-cell prop-keys, got {akeys}"
        oA = float(orc.get("A", 0.0))
        assert abs(vn["a"] - oA) < 1e-5, f"numpy A {vn['a']:.6f} != oracle {oA:.6f}"
        assert abs(vb["a"] - oA) < 1e-5, f"numba A {vb['a']:.6f} != oracle {oA:.6f}"
        print(f"by_subcell: MID A has 2 distinct sub-cell keys; numpy & numba == oracle "
              f"(A M5={oA:.3f})  ✓")
    finally:
        os.unlink(csv)


if __name__ == "__main__":
    test_subcell_matches_oracle()
    print("PASS ✓  by_subcell prop_key reconciles per-sub-cell shares with the delivered projection")
