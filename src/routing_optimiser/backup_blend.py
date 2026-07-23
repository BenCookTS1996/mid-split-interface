"""Backup-rule catch-all blend — makes tab 2 (optimiser) and tab 3 (projection) agree
with the REAL VAMP pipeline (tab 5).

WHY THIS EXISTS
---------------
The validation rules folder contains, alongside the newly-exported per-BIN split
(PoolTargeted_*), the pre-existing backup file(s) (mid_split_*_Eff_Backup_*). Those
backups carry a CATCH-ALL row (`BIN = Other`/`All`) that assigns a handful of
incumbent gateways (e.g. braintree-usd-tav = 12/16, bancard-usd-tav = 1) a fixed
share. The pipeline's data_extractor._expand_dynamic_bins EXPANDS that catch-all onto
every historical BIN, and its rule parser DROPS any gateway the split set to exactly 0
(``Share > 0`` filter) — so an explicit 0 cannot override the catch-all. Net effect for
a cell whose split zeroed such a gateway:

    effective = (split shares, summing to 1)  +  (catch-all gateways, as fractions)
                                                  all renormalised

e.g. split braintree 0 (dropped) + catch-all braintree 12 + bancard 1
     ⇒ braintree = 12 / (100 + 12 + 1) = 12/113 = 10.6 %  (matches tab 5 exactly).

Tab 2/3 previously used the raw split (braintree 0 %), so they diverged from tab 5.
This module reproduces the pipeline's blend so both can use the ACTUAL routed shares.

Key subtlety (why the optimiser cares): a specific share OVERRIDES the catch-all
(data_extractor keeps the 'Specific' row over the 'Expanded' one on de-dup). So the
optimiser can dodge the catch-all by assigning a gateway a tiny POSITIVE share instead
of exactly 0 — only an exact 0 triggers the 12 % re-add. Feeding the blend into the
optimiser lets it discover that and make truthful cut decisions.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Tuple

import pandas as pd

__build__ = "2026-07-22-backup-blend+rules-to-split"

# Non-gateway columns in a wide rule sheet (everything else is a gateway weight column).
_ID_COLS = {"go live", "bin group", "brand", "rpgt", "currency", "bin",
            "paymentmethodprovider", "sticky", "country", "check", "dup check"}
_DYNAMIC_BIN = {"other", "all"}
_PMP_ALL = ("non_gp_ap", "googlepay", "applepay")
_COUNTRY_ALL = ("usa", "non-usa")

# The backup catch-all values are on the sheet's 0–100 percentage scale (Check = 100).
_PCT_SCALE = 100.0


def _norm(s) -> str:
    return str(s).strip().lower()


def parse_backup_catchall(backup_dir: str, rpgt_filter: str | None = None) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    """Read every rule file in ``backup_dir`` and return the CATCH-ALL (BIN=Other/All)
    gateway shares, keyed by (currency, rpgt, pmp, country) — all lower-cased — with
    STICKY/'all' pmp/'all' country expanded exactly as data_extractor does.

    Returns {} when the folder is empty/absent (⇒ blend is a no-op, back to raw split).
    Raw sheet percentages are preserved (e.g. braintree 12.0); blend_cell_shares divides
    by 100 to combine with the optimiser's fractional shares.
    """
    out: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    if not backup_dir or not os.path.isdir(backup_dir):
        return out
    files: list[str] = []
    for pat in ("*.xlsx", "*.xls", "*.csv"):
        files += glob.glob(os.path.join(backup_dir, pat))
    for f in files:
        try:
            is_x = f.lower().endswith((".xlsx", ".xls"))
            df = pd.read_excel(f) if is_x else pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        cols = {c: _norm(c) for c in df.columns}
        bin_c = next((c for c, n in cols.items() if n == "bin"), None)
        cur_c = next((c for c, n in cols.items() if n == "currency"), None)
        rp_c = next((c for c, n in cols.items() if n == "rpgt"), None)
        if bin_c is None or cur_c is None or rp_c is None:
            continue
        pmp_c = next((c for c, n in cols.items() if n == "paymentmethodprovider"), None)
        ctry_c = next((c for c, n in cols.items() if n == "country"), None)
        gw_cols = [c for c, n in cols.items() if n not in _ID_COLS]
        # CATCH-ALL rows only (BIN = Other / All).
        sub = df[df[bin_c].astype(str).str.strip().str.lower().isin(_DYNAMIC_BIN)]
        if rpgt_filter is not None:
            sub = sub[sub[rp_c].astype(str).str.strip().str.lower() == _norm(rpgt_filter)]
        for _, r in sub.iterrows():
            cur = _norm(r[cur_c]); rp = _norm(r[rp_c])
            pmp_raw = _norm(r[pmp_c]) if pmp_c else "all"
            ctry_raw = _norm(r[ctry_c]) if ctry_c else "all"
            pmps = _PMP_ALL if pmp_raw in ("all", "", "nan", "0", "0.0") else (pmp_raw,)
            ctries = _COUNTRY_ALL if ctry_raw in ("all", "", "nan", "0", "0.0") else (ctry_raw,)
            gw_share = {}
            for gc in gw_cols:
                try:
                    v = float(str(r[gc]).replace("%", "").replace(",", "").strip())
                except (ValueError, TypeError):
                    v = 0.0
                if v > 0:
                    gw_share[_norm(gc)] = gw_share.get(_norm(gc), 0.0) + v
            if not gw_share:
                continue
            for pmp in pmps:
                for ctry in ctries:
                    key = (cur, rp, pmp, ctry)
                    dst = out.setdefault(key, {})
                    for g, v in gw_share.items():
                        dst[g] = dst.get(g, 0.0) + v   # de-dup across files ⇒ sum (matches groupby-sum)
    return out


def blend_cell_shares(specific: Dict[str, float], catchall: Dict[str, float]) -> Dict[str, float]:
    """Reproduce the pipeline's effective per-cell routing shares.

    ``specific``  : the optimiser/exported split for ONE cell {gatewayFid: share} (any
                    scale; only strictly-positive shares survive, mirroring the parser's
                    ``Share > 0`` drop).
    ``catchall``  : the backup catch-all for that cell {gatewayFid: pct} on the 0–100
                    scale (from parse_backup_catchall).

    Returns {gatewayFid: effective_share} summing to 1.0. A gateway present in BOTH keeps
    its SPECIFIC value (Specific overrides Expanded) and is NOT re-added from the catch-all.
    """
    # Keys are preserved AS-IS (vampMid case matters downstream); the override check
    # (a catch-all gateway already given a positive specific share is NOT re-added) is
    # done case-insensitively.
    spec = {g: float(v) for g, v in (specific or {}).items() if float(v) > 0}
    _stot = sum(spec.values())
    if _stot <= 0:
        # Optimiser gave the cell nothing positive → fall back to the catch-all alone.
        inj = {g: (float(v) / _PCT_SCALE) for g, v in (catchall or {}).items() if float(v) > 0}
        t = sum(inj.values())
        return {g: v / t for g, v in inj.items()} if t > 0 else dict(spec)
    # Normalise the specific split to sum 1 (its own 100 %), then add the catch-all
    # gateways it did NOT already assign a positive share, as fractions (pct / 100).
    spec_n = {g: v / _stot for g, v in spec.items()}
    _spec_low = {str(g).strip().lower() for g in spec_n}
    inj = {g: (float(v) / _PCT_SCALE)
           for g, v in (catchall or {}).items()
           if float(v) > 0 and str(g).strip().lower() not in _spec_low}
    total = 1.0 + sum(inj.values())
    eff = {g: v / total for g, v in spec_n.items()}
    for g, v in inj.items():
        eff[g] = v / total
    return eff


def _catchall_by_vampmid(catchall_cell: Dict[str, float], fid2vamp: Dict[str, str]) -> Dict[str, float]:
    """Map a cell's catch-all {gatewayFid: pct} onto {vampMid: pct} (summing fids that
    share a vampMid), using fid2vamp (lower-cased keys). Fids with no vampMid are dropped."""
    out: Dict[str, float] = {}
    for fid, pct in (catchall_cell or {}).items():
        vm = fid2vamp.get(str(fid).strip().lower())
        if vm is None:
            continue
        out[vm] = out.get(vm, 0.0) + float(pct)
    return out


def blend_prop_items(prop_items, catchall, fid2vamp, by_rpgt=None):
    """Blend the backup catch-all into a projection's prop_items so the projected POST
    matches the deployed pipeline (tab 5).

    prop_items : iterable of tuples at ONE of these grains (the enforced fine grain is the
        7-tuple used by compute_vamp_prepost_granular):
          7-tuple: (Currency, BIN, RPGT, pmp, Country, vampMid, prop_raw)
          5-tuple: (Currency, BIN, RPGT, vampMid, prop_raw)          [pmp/Country pooled]
          4-tuple: (Currency, BIN, vampMid, prop_raw)                [RPGT pooled]
    catchall   : parse_backup_catchall output {(cur,rpgt,pmp,ctry): {gatewayFid: pct}}.
    fid2vamp   : {gatewayFid(lower): vampMid} to map catch-all fids onto vampMids.

    Returns a NEW list of tuples at the SAME arity, with catch-all vampMids injected per
    cell (incl. cells where the split gave them 0/none) and every cell renormalised. If
    catchall is empty the input is returned unchanged (no-op). Only the 7-tuple grain can
    match the catch-all's pmp/Country exactly; coarser grains pool the catch-all (cur,rpgt)
    over pmp/Country using an equal blend — a documented approximation for those callers.
    """
    _pi = [tuple(t) for t in (prop_items or [])]
    if not _pi or not catchall:
        return _pi
    _n = len(_pi[0])
    f2v = {str(k).strip().lower(): v for k, v in (fid2vamp or {}).items()}

    # Pre-pool the catch-all for the coarse (4/5-tuple) callers: average pct across the
    # pmp/Country variants of each (cur, rpgt) — the finest info those grains can carry.
    def _pooled(cur, rpgt):
        acc, cnt = {}, 0
        for (c, r, _p, _ct), gw in catchall.items():
            if c == cur and (rpgt is None or r == rpgt):
                cnt += 1
                for g, v in gw.items():
                    acc[g] = acc.get(g, 0.0) + float(v)
        return {g: v / cnt for g, v in acc.items()} if cnt else {}

    from collections import defaultdict
    cells = defaultdict(dict)          # cell-key -> {vampMid: prop_raw}
    order = []                         # preserve first-seen cell order
    for t in _pi:
        if _n >= 7:
            cur, b, rp, pmp, ctry, vm, s = t[0], t[1], t[2], t[3], t[4], t[5], t[6]
            ck = (cur, b, rp, pmp, ctry)
        elif _n == 5:
            cur, b, rp, vm, s = t
            ck = (cur, b, rp)
        else:  # 4-tuple
            cur, b, vm, s = t
            rp = None
            ck = (cur, b)
        if ck not in cells:
            order.append(ck)
        cells[ck][str(vm)] = cells[ck].get(str(vm), 0.0) + float(s)

    out = []
    for ck in order:
        spec = cells[ck]
        if _n >= 7:
            cur, b, rp, pmp, ctry = ck
            ca = _catchall_by_vampmid(catchall.get((cur, str(rp).strip().lower(),
                                                     str(pmp).strip().lower(),
                                                     str(ctry).strip().lower()), {}), f2v)
        elif _n == 5:
            cur, b, rp = ck
            ca = _catchall_by_vampmid(_pooled(cur, str(rp).strip().lower()), f2v)
        else:
            cur, b = ck
            ca = _catchall_by_vampmid(_pooled(cur, None), f2v)
        eff = blend_cell_shares(spec, ca)
        for vm, s in eff.items():
            if _n >= 7:
                out.append((cur, b, rp, pmp, ctry, vm, s))
            elif _n == 5:
                out.append((cur, b, rp, vm, s))
            else:
                out.append((cur, b, vm, s))
    return out


def parse_rules_to_split(rules_dir: str) -> pd.DataFrame:
    """Reconstruct a proposed-split DataFrame from a folder of exported rule files.

    Reads every wide rule sheet (Currency / RPGT / BIN columns + one weight column per
    gateway), melts the gateway weight columns into rows, and normalises the weights to a
    SHARE summing to 1 within each (rpgt, currency, bank) cell — the shape the Impact tab's
    ``_impact_eval_frame`` expects.

    Returns columns ``[rpgt, currency, bank, gateway, share]``. ``bank`` holds the BIN from
    the sheet (collapsed to a parent bank downstream via bin_to_bank). Returns an EMPTY frame
    (same columns) when the folder has no readable rule files — the caller decides how to flag.
    """
    cols = ["rpgt", "currency", "bank", "gateway", "share"]
    files: list[str] = []
    if rules_dir and os.path.isdir(rules_dir):
        for pat in ("*.xlsx", "*.xls", "*.csv"):
            files += glob.glob(os.path.join(rules_dir, pat))
    recs = []
    for f in files:
        try:
            is_x = f.lower().endswith((".xlsx", ".xls"))
            df = pd.read_excel(f) if is_x else pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        cmap = {c: _norm(c) for c in df.columns}
        rp_c = next((c for c, n in cmap.items() if n == "rpgt"), None)
        cur_c = next((c for c, n in cmap.items() if n == "currency"), None)
        bin_c = next((c for c, n in cmap.items() if n == "bin"), None)
        if rp_c is None or cur_c is None or bin_c is None:
            continue
        gw_cols = [c for c, n in cmap.items() if n not in _ID_COLS]
        for _, r in df.iterrows():
            rp, cur, bnk = str(r[rp_c]).strip(), str(r[cur_c]).strip(), str(r[bin_c]).strip()
            for gc in gw_cols:
                try:
                    w = float(str(r[gc]).replace("%", "").replace(",", "").strip())
                except (ValueError, TypeError):
                    w = 0.0
                if w > 0:
                    recs.append({"rpgt": rp, "currency": cur, "bank": bnk,
                                 "gateway": str(gc).strip(), "weight": w})
    if not recs:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(recs).groupby(
        ["rpgt", "currency", "bank", "gateway"], as_index=False)["weight"].sum()
    _tot = out.groupby(["rpgt", "currency", "bank"])["weight"].transform("sum")
    out["share"] = (out["weight"] / _tot).where(_tot > 0, 0.0)
    return out[cols]
