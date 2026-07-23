"""
Routing eligibility — hard bans and capability restrictions.

Two complementary sources feed one eligibility check:

  * ``routing_restrictions.json`` — a list of ban rules. Each rule says a
    ``target`` (a vampMid or gatewayFid) must receive ZERO traffic whose
    attributes match ALL of the listed field conditions, e.g.::

        {"target": "Merrick - Total AV",
         "match": {"rpgt": ["Annual Sub Sale", "Addon Sale",
                            "Addon Renewal", "P6M Renewal"]}}

  * ``processWallet`` column in Master_MID_List — gatewayFids flagged FALSE
    CANNOT process wallet traffic (paymentMethodProvider GOOGLEPAY / APPLEPAY).

Both are enforced on the proposed split by zeroing the banned (gateway, profile)
shares and redistributing the freed volume to the eligible gateways in the same
routing group (so transactions are conserved). Wallet capability is enforced as
a volume-weighted blend: an incapable gateway keeps only its NON-wallet share.

Enforced at the exploded split grain, which carries rpgt / currency / bank /
gateway. BIN- and country-level bans need finer routing and are not applied here.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

__build__ = "2026-07-16-eligibility-ban-mask-cache"

WALLET_VALUES = {"googlepay", "applepay"}


def load_usa_only(path: str) -> frozenset:
    """Explicit list of gatewayFids that can ONLY process country='USA'.

    Read from the ``usa_only_gateways`` key of routing_restrictions.json. These
    are enforced like wallet capability: the gateway keeps only the USA fraction
    of each cell, the Non-USA portion is redistributed. Missing/invalid -> empty."""
    if not path or not os.path.exists(path):
        return frozenset()
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return frozenset()
    lst = raw.get("usa_only_gateways", []) if isinstance(raw, dict) else []
    return frozenset(str(g).strip().lower() for g in (lst or []) if str(g).strip())


def load_explore_gateways(path: str) -> frozenset:
    """gatewayFids to treat as ELIGIBLE candidates even with no 30-day attempts, so
    capable-but-untested gateways can earn exploration volume (seeded with the pooled
    prior success rate + the exploration floor). Read from the ``explore_untested_
    gateways`` key of routing_restrictions.json. Empty/missing -> no exploration.

    Motivation: eligibility is normally built from OBSERVED 30D attempts, so a brand-
    new gateway (no attempts for a bank) is never a candidate and never gets volume.
    Listing it here forces it into the candidate set for its currency's cells."""
    if not path or not os.path.exists(path):
        return frozenset()
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return frozenset()
    lst = raw.get("explore_untested_gateways", []) if isinstance(raw, dict) else []
    return frozenset(str(g).strip().lower() for g in (lst or []) if str(g).strip())


def load_restrictions(path: str) -> list[dict]:
    """Load and normalise ban rules. Missing/invalid file -> no rules."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return []
    rules = raw.get("rules", []) if isinstance(raw, dict) else raw
    out = []
    for r in (rules or []):
        if not isinstance(r, dict):
            continue
        tgt = str(r.get("target", "")).strip().lower()
        match = r.get("match", {}) or {}
        norm = {}
        for k, vs in match.items():
            vals = vs if isinstance(vs, (list, tuple, set)) else [vs]
            norm[str(k).strip().lower()] = {str(v).strip().lower() for v in vals}
        if tgt and norm:
            out.append({"target": tgt, "match": norm})
    return out


def _resolve_field(field: str, profile: dict):
    """Value for a rule field, aliasing 'bin' onto the 'bank' column (BIN-level
    cells are keyed as 'bank' in this app). Returns None if unavailable."""
    pv = profile.get(field)
    if pv is None and field == "bin":
        pv = profile.get("bank")
    return pv


def _row_banned(gw: str, vmid: str, profile: dict, rules: list[dict]) -> bool:
    """True if any rule bans this gateway/vampMid for this traffic profile.

    A rule fires only when EVERY field it lists is both available at this grain
    AND matches. If a field can't be evaluated (e.g. 'country', which isn't part
    of the routing grain), the rule is treated as unenforceable and does NOT ban
    — otherwise it would silently ban all traffic for the target.
    """
    for r in rules:
        if r["target"] not in (gw, vmid):
            continue
        ok = True
        for field, values in r["match"].items():
            pv = _resolve_field(field, profile)
            if pv is None:
                ok = False  # can't verify this field -> don't ban (safe)
                break
            if str(pv).strip().lower() not in values:
                ok = False
                break
        if ok:
            return True
    return False


# The ban mask depends only on the per-row PROFILE (gateway, vampMid, rpgt/currency/
# bank/bin/country) and the rules — NOT on the split shares. The enforcement runs this
# for every pass/dial on the same rows, so we memoise the mask on a content hash of the
# profile columns + a signature of the rules. Bit-identical: on a hit we return the SAME
# boolean array the loop would have produced; on any content change the hash differs and
# it recomputes. Only the last result is kept (calls alternate over the same one split).
_BAN_MASK_CACHE: dict = {}


def _rules_signature(rules: list[dict]):
    return tuple((r.get("target", ""),
                  tuple(sorted((k, tuple(sorted(v))) for k, v in (r.get("match", {}) or {}).items())))
                 for r in rules)


def _banned_mask_cached(df: pd.DataFrame, rules: list[dict], prof_cols: list[str]) -> np.ndarray:
    cols = ["_gw", "_vm"] + [c for c in prof_cols if c in df.columns]
    try:
        _h = int(pd.util.hash_pandas_object(df[cols].astype(str), index=False).sum() & ((1 << 63) - 1))
        key = (len(df), tuple(cols), _rules_signature(rules), _h)
    except Exception:  # noqa: BLE001 — hashing failed → skip cache, compute directly
        key = None
    if key is not None and _BAN_MASK_CACHE.get("key") == key:
        return _BAN_MASK_CACHE["mask"]
    profiles = (df[prof_cols].astype(str).apply(lambda r: {c: r[c] for c in prof_cols}, axis=1)
                if prof_cols else pd.Series([{}] * len(df), index=df.index))
    mask = np.array([_row_banned(g, v, p, rules)
                     for g, v, p in zip(df["_gw"], df["_vm"], profiles)])
    if key is not None:
        _BAN_MASK_CACHE["key"], _BAN_MASK_CACHE["mask"] = key, mask
    return mask


def unenforceable_fields(rules: list[dict], available_cols) -> set:
    """Match-fields referenced by rules that can't be enforced at this grain
    (after aliasing BIN -> bank). The caller can warn about these (e.g. country)."""
    avail = {str(c).strip().lower() for c in available_cols}
    if "bank" in avail:
        avail.add("bin")
    missing = set()
    for r in rules:
        for f in r["match"]:
            if f not in avail:
                missing.add(f)
    return missing


def _renorm(df: pd.DataFrame, group_keys: list[str], col: str) -> pd.Series:
    """Renormalise `col` to sum 1 within each group (leaves all-zero groups)."""
    s = df.groupby(group_keys, dropna=False)[col].transform("sum")
    return np.where(s > 0, df[col] / s, df[col])


def _capability_blend(df: pd.DataFrame, gk: list[str], incapable, frac_map: dict,
                      default: float) -> np.ndarray:
    """Volume-weighted capability blend, returning the new per-row share array.

    An `incapable` gateway CANNOT serve the `frac` portion of a cell's traffic, so
    it keeps only (1 - frac) of its baseline share and the `frac` portion is
    redistributed to the capable gateways in the cell (transactions conserved).
    Used identically for wallet capability (frac = the cell's wallet share) and for
    country capability (frac = the cell's Non-USA share). `frac_map` is keyed by
    (currency, bank); `default` is used when a cell isn't in the map."""
    incap = (df["_gw"].isin(incapable) | df["_vm"].isin(incapable)).to_numpy()
    new_share = df["share"].to_numpy(float).copy()
    if not (gk and incap.any()):
        return new_share
    has_cur_bank = ("currency" in df.columns and "bank" in df.columns)
    for key, idx in df.groupby(gk, dropna=False).groups.items():
        rows = df.loc[idx]
        base = rows["share"].to_numpy(float)
        if base.sum() <= 0:
            continue
        wf = default
        if has_cur_bank:
            ck = (str(rows["currency"].iloc[0]).strip().lower(),
                  str(rows["bank"].iloc[0]).strip().lower())
            wf = float(frac_map.get(ck, default))
        wf = 0.0 if (wf != wf) else min(max(wf, 0.0), 1.0)
        m = incap[[df.index.get_loc(i) for i in idx]]
        cshare = base.copy()
        cshare[m] = 0.0
        s = cshare.sum()
        cshare = cshare / s if s > 0 else base  # if only incapable exist, no reroute possible
        blended = wf * cshare + (1.0 - wf) * base
        for pos, i in enumerate(idx):
            new_share[df.index.get_loc(i)] = blended[pos]
    return new_share


def apply_restrictions(split: pd.DataFrame, rules: list[dict], fid2vamp: dict,
                       wallet_incapable=frozenset(), wallet_frac: dict | None = None,
                       wallet_default: float = 0.0,
                       usa_only=frozenset(), nonusa_frac: dict | None = None,
                       nonusa_default: float = 0.0,
                       group_keys=("rpgt", "currency", "bank")) -> pd.DataFrame:
    """Return the split with bans + wallet capability + country capability enforced.

    split: rows with at least [gateway, share] and ideally [rpgt, currency, bank].
    rules: from load_restrictions.
    fid2vamp: gatewayFid(lower) -> vampMid(lower).
    wallet_incapable: set of gatewayFids/vampMids (lower) that can't do wallet.
    wallet_frac: {(currency, bank): fraction of the cell that is wallet traffic}.
    usa_only: set of gatewayFids/vampMids (lower) that can ONLY process USA traffic.
    nonusa_frac: {(currency, bank): fraction of the cell that is Non-USA traffic}.
    """
    if split is None or getattr(split, "empty", True):
        return split
    if not rules and not wallet_incapable and not usa_only:
        return split

    df = split.copy()
    df["_gw"] = df["gateway"].astype(str).str.strip().str.lower()
    df["_vm"] = df["_gw"].map(fid2vamp).fillna(df["_gw"])
    gk = [k for k in group_keys if k in df.columns]

    # 1. Hard bans -> share 0, then renormalise within each routing group.
    if rules:
        prof_cols = [c for c in ("rpgt", "currency", "bank", "bin", "country") if c in df.columns]
        banned = _banned_mask_cached(df, rules, prof_cols)
        if banned.any():
            df.loc[banned, "share"] = 0.0
            if gk:
                df["share"] = _renorm(df, gk, "share")

    # 2. Wallet capability — blend: incapable gateways keep only their non-wallet share.
    if wallet_incapable:
        df["share"] = _capability_blend(df, gk, wallet_incapable, wallet_frac or {}, wallet_default)
        if gk:
            df["share"] = _renorm(df, gk, "share")

    # 3. Country capability — blend: USA-only gateways keep only their USA share; the
    #    Non-USA portion of each cell is redistributed to the other gateways. Same
    #    mechanism as wallet, with frac = the cell's Non-USA traffic fraction.
    if usa_only:
        df["share"] = _capability_blend(df, gk, usa_only, nonusa_frac or {}, nonusa_default)
        if gk:
            df["share"] = _renorm(df, gk, "share")

    if "cell_volume" in df.columns:
        df["volume"] = df["cell_volume"] * df["share"]
    return df.drop(columns=[c for c in ["_gw", "_vm"] if c in df.columns])
