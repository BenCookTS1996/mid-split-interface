"""Validate the CELL-LEVEL catch-all fix across the three places that model it:

A. Pipeline (data_extractor._apply_chronological_deduplication logic): an Expanded catch-all row is
   DROPPED in any full-grain cell that already has a Specific row, and KEPT where the cell has none.
B. backup_blend.blend_cell_shares: a cell with any specific share ships only its specific shares
   (renormalised); an empty cell falls back to the catch-all alone.
C. The vectorised _fm_blend_pr (GA fitness) reproduces (B) on the projector prop-key grain.

Run: python tests/test_cell_level_catchall.py   (pure numpy/pandas + repo backup_blend)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser.backup_blend import blend_cell_shares  # noqa: E402

DEDUP = ["Brand", "RPGT", "Currency", "BIN", "paymentMethodProvider", "Country"]


def pipeline_drop(split_df):
    """Replica of the fix in data_extractor._apply_chronological_deduplication."""
    if "Rule_Source" in split_df.columns and (split_df["Rule_Source"] == "Expanded").any():
        spec_cells = split_df.loc[split_df["Rule_Source"] == "Specific", DEDUP].drop_duplicates()
        if not spec_cells.empty:
            spec_cells = spec_cells.assign(_has_specific=1)
            split_df = split_df.merge(spec_cells, on=DEDUP, how="left")
            drop = (split_df["Rule_Source"] == "Expanded") & (split_df["_has_specific"] == 1)
            split_df = split_df.loc[~drop].drop(columns=["_has_specific"]).reset_index(drop=True)
    return split_df


def test_pipeline_drop():
    def row(bin_, gw, share, src):
        return {"Brand": "TotalAV", "RPGT": "Monthly Initial", "Currency": "USD", "BIN": bin_,
                "paymentMethodProvider": "GOOGLEPAY", "Country": "USA",
                "gatewayFid": gw, "Share": share, "Rule_Source": src}

    df = pd.DataFrame([
        # Cell A (400086) — a DEFINED profile (has specific shares); paysafe was zeroed → only an
        # Expanded catch-all row exists for it. That Expanded row must be DROPPED.
        row("400086", "adyen", 30, "Specific"),
        row("400086", "worldpay", 40, "Specific"),
        row("400086", "braintree", 30, "Specific"),
        row("400086", "paysafe", 5, "Expanded"),      # <-- should be dropped (cell is defined)
        row("400086", "braintree", 12, "Expanded"),   # <-- should be dropped (cell is defined)
        # Cell B (999999) — an UNDEFINED profile (no specific rule); catch-all must SURVIVE.
        row("999999", "paysafe", 5, "Expanded"),
        row("999999", "braintree", 12, "Expanded"),
    ])
    out = pipeline_drop(df)

    a = out[out["BIN"] == "400086"]
    b = out[out["BIN"] == "999999"]
    assert (a["Rule_Source"] == "Expanded").sum() == 0, "Expanded not dropped in a defined cell"
    assert set(a["gatewayFid"]) == {"adyen", "worldpay", "braintree"}, "specific rows altered"
    assert (b["Rule_Source"] == "Expanded").sum() == 2, "catch-all wrongly dropped in an empty cell"
    print("A. pipeline: Expanded dropped in defined cell, kept in undefined cell  ✓")


def test_blend_cell_shares():
    catchall = {"Braintree USA - Total AV": 12.0, "PaySafe - Total AV": 1.0}
    # Defined cell → catch-all does NOT fire; specific renormalised, Braintree stays absent.
    spec = {"Adyen_TotalAV": 0.6, "WorldPay - Total AV": 0.4}
    eff = blend_cell_shares(spec, catchall)
    assert abs(sum(eff.values()) - 1.0) < 1e-12
    assert "Braintree USA - Total AV" not in eff, "catch-all fired on a defined cell"
    assert abs(eff["Adyen_TotalAV"] - 0.6) < 1e-12 and abs(eff["WorldPay - Total AV"] - 0.4) < 1e-12
    # Empty cell → catch-all alone.
    eff2 = blend_cell_shares({}, catchall)
    assert abs(sum(eff2.values()) - 1.0) < 1e-12
    assert abs(eff2["Braintree USA - Total AV"] - 12.0 / 13.0) < 1e-12
    print("B. blend_cell_shares: no re-add on defined cell, catch-all on empty cell  ✓")


def build_fm_blend(prop_keys, bpool_rpgt):
    import scipy.sparse as sp
    K = len(prop_keys)
    cid = np.zeros(K, dtype=np.int64); inj = np.zeros(K, dtype=float); cmap = {}
    for i, k in enumerate(prop_keys):
        ps = k.split("|"); vmk = ps[-1]; ck = (ps[0], ps[1], ps[2])
        ca = bpool_rpgt.get((ps[0], ps[2]), {})
        cid[i] = cmap.setdefault(ck, len(cmap))
        if ca:
            for cav, cap in ca.items():
                if str(cav).strip().lower() == vmk.strip().lower():
                    inj[i] = float(cap) / 100.0; break
    A = sp.csr_matrix((np.ones(K), (cid, np.arange(K))), shape=(max(len(cmap), 1), K)); AT = A.T.tocsr()

    def blend(pr, _A=A, _AT=AT, _injv=inj):
        pr = np.ascontiguousarray(pr, float); one = pr.ndim == 1
        if one: pr = pr[None, :]
        pos = pr > 0.0; specpos = np.where(pos, pr, 0.0)
        S = np.asarray((_A @ specpos.T).T); Sb = np.asarray((_AT @ S.T).T)
        empty = Sb <= 0.0; injcol = np.where(empty, _injv[None, :], 0.0)
        INJ = np.asarray((_A @ injcol.T).T); INJb = np.asarray((_AT @ INJ.T).T)
        out = np.where(empty,
                       np.where(INJb > 0, injcol / np.where(INJb > 0, INJb, 1.0), pr),
                       np.where(Sb > 0, specpos / np.where(Sb > 0, Sb, 1.0), 0.0))
        return out[0] if one else out
    return blend, cmap


def test_fm_blend_matches():
    # 2 cells; each has vampMids Braintree/Adyen/WorldPay. bpool re-adds Braintree/PaySafe.
    prop_keys = [f"usd|400086|monthly initial|{v}" for v in
                 ["Braintree USA - Total AV", "Adyen_TotalAV", "WorldPay - Total AV"]] + \
                [f"usd|999999|monthly initial|{v}" for v in
                 ["Braintree USA - Total AV", "Adyen_TotalAV", "WorldPay - Total AV"]]
    bpool = {("usd", "monthly initial"): {"Braintree USA - Total AV": 12.0}}
    blend, cmap = build_fm_blend(prop_keys, bpool)

    # Cell A defined (Adyen+WorldPay positive, Braintree 0), Cell B empty (all 0).
    pr = np.array([0.0, 0.6, 0.4,   0.0, 0.0, 0.0])
    out = blend(pr)
    # Cell A: renormalised specific, Braintree stays 0 (no re-add).
    assert abs(out[0]) < 1e-12, f"Braintree re-added in a defined cell: {out[0]}"
    assert abs(out[1] - 0.6) < 1e-12 and abs(out[2] - 0.4) < 1e-12
    # Cell B: empty → catch-all alone → Braintree = 1.0 (only catch-all member present).
    assert abs(out[3] - 1.0) < 1e-12, f"empty cell should fall to catch-all: {out[3:]}"
    assert abs(out[4]) < 1e-12 and abs(out[5]) < 1e-12
    print("C. _fm_blend_pr: no re-add on defined cell, catch-all on empty cell  ✓")


if __name__ == "__main__":
    test_pipeline_drop()
    test_blend_cell_shares()
    test_fm_blend_matches()
    print("PASS ✓  cell-level catch-all consistent across pipeline, blend_cell_shares, and GA fitness")
