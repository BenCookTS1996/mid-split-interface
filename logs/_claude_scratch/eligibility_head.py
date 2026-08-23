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

__build__ = ("2026-08-18-eligibility-ban-mask-cache+population-operator+fid-grain-capability"
             "+exact-subcell-capability")

WALLET_VALUES = {"googlepay", "applepay"}


# [FN-053]
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


# [FN-054]
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


# [FN-055]
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


# [FN-056]
def _resolve_field(field: str, profile: dict):
    """Value for a rule field, aliasing 'bin' onto the 'bank' column (BIN-level
    cells are keyed as 'bank' in this app). Returns None if unavailable."""
    pv = profile.get(field)
    if pv is None and field == "bin":
        pv = profile.get("bank")
    return pv


# [FN-057]
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


# [FN-058]
def _rules_signature(rules: list[dict]):
    return tuple((r.get("target", ""),
                  tuple(sorted((k, tuple(sorted(v))) for k, v in (r.get("match", {}) or {}).items())))
                 for r in rules)


# [FN-059]
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


# [FN-060]
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


# [FN-061]
def _renorm(df: pd.DataFrame, group_keys: list[str], col: str) -> pd.Series:
    """Renormalise `col` to sum 1 within each group (leaves all-zero groups)."""
    s = df.groupby(group_keys, dropna=False)[col].transform("sum")
    return np.where(s > 0, df[col] / s, df[col])


# [FN-062]
# Sub-cell purity. The blend's `frac` is "how much of this cell is wallet / Non-USA traffic".
# At sub-cell grain that is not a proportion at all — the cell IS one or the other, so the
# answer is 0 or 1. Same test as the scaffold's _T0_emask_a, so one rule holds everywhere.
_WALLET_PMP = frozenset({"googlepay", "applepay"})
_USA_CTRY = frozenset({"usa", "us"})
_MIXED = frozenset({"", "_all_", "all", "nan", "none"})


# [FN-062a]
def _exact_wallet_frac(pmp):
    """1.0 in a wallet sub-cell, 0.0 in a card sub-cell, None when the value is mixed."""
    p = str(pmp).strip().lower()
    if p in _MIXED:
        return None
    return 1.0 if p in _WALLET_PMP else 0.0


# [FN-062b]
def _exact_nonusa_frac(ctry):
    """1.0 in a Non-USA sub-cell, 0.0 in a USA sub-cell, None when the value is mixed."""
    c = str(ctry).strip().lower()
    if c in _MIXED:
        return None
    return 0.0 if c in _USA_CTRY else 1.0


# [FN-062c]
def _subcell_col(df, kind):
    """The column carrying this restriction's sub-cell identity, or None at cell grain."""
    if kind == "wallet":
        return "pmp" if "pmp" in df.columns else None
    if kind == "nonusa":
        for _c in ("ctry", "country"):
            if _c in df.columns:
                return _c
    return None


def _capability_blend(df: pd.DataFrame, group_cols: list[str], incapable, frac_map: dict,
                      default: float, kind: str | None = None) -> np.ndarray:
    """Volume-weighted capability blend, returning the new per-row share array.

    ANALOGY: an `incapable` gateway is like a vendor that can't take a certain payment type.
    It keeps only the (1 − frac) share it CAN serve; the `frac` portion of each cell is handed
    to the vendors that CAN (renormalised among themselves), so no transactions are lost. Used
    identically for wallet capability (frac = the cell's wallet share) and country capability
    (frac = the cell's Non-USA share). `frac_map` is keyed by (currency, bank); `default` is
    used when a cell isn't in the map.
    """
    # FID-GRAIN capability (2026-08-17). `incapable` carries BOTH gatewayFids and their
    # rolled-up vampMids, so ALSO testing `_vm` over-blocks every fid of a vampMid whose
    # capability varies by fid: PaySafe - Total AV is wallet-capable on paysafe-usd-tav but
    # not on paysafe-eur-tav / -gbp-tav, and the roll-up barred the USD fid from wallets too.
    # `_gw` alone is exact for BOTH framings — a fid-keyed frame matches its own fid, and a
    # vampMid-keyed frame matches the vampMid, because the set holds both. The `_vm` term
    # contributed nothing except the roll-up, so it is removed.
    incapable_mask = df["_gw"].isin(incapable).to_numpy()
    result_share = df["share"].to_numpy(float).copy()
    if not (group_cols and incapable_mask.any()):
        return result_share
    has_cur_bank = ("currency" in df.columns and "bank" in df.columns)
    _sc_col = _subcell_col(df, kind) if kind else None
    _pos_of = {lbl: p for p, lbl in enumerate(df.index)}   # label -> positional (unique index)
    for _grp_key, row_idx in df.groupby(group_cols, dropna=False).groups.items():
        group_rows = df.loc[row_idx]
        base = group_rows["share"].to_numpy(float)
        if base.sum() <= 0:
            continue
        reroute_frac = default
        if has_cur_bank:
            cur_bank_key = (str(group_rows["currency"].iloc[0]).strip().lower(),
                            str(group_rows["bank"].iloc[0]).strip().lower())
            reroute_frac = float(frac_map.get(cur_bank_key, default))
        # EXACT at sub-cell grain: when the group is pure (one pmp / one Country), the
        # fraction is not an estimate — it is 0 or 1. `_sc_col` is None at cell grain, so
        # the fraction map is used exactly as before.
        if _sc_col is not None:
            _ex = (_exact_wallet_frac(group_rows[_sc_col].iloc[0]) if kind == "wallet"
                   else _exact_nonusa_frac(group_rows[_sc_col].iloc[0]))
            if _ex is not None:
                reroute_frac = _ex
        reroute_frac = 0.0 if (reroute_frac != reroute_frac) else min(max(reroute_frac, 0.0), 1.0)
        incap_in_cell = incapable_mask[[_pos_of[i] for i in row_idx]]
        capable_share = base.copy()
        capable_share[incap_in_cell] = 0.0
        capable_total = capable_share.sum()
        # if only incapable gateways exist, no reroute is possible → keep the baseline
        capable_share = capable_share / capable_total if capable_total > 0 else base
        blended = reroute_frac * capable_share + (1.0 - reroute_frac) * base
        for pos, i in enumerate(row_idx):
            result_share[_pos_of[i]] = blended[pos]
    return result_share


# [FN-063]
def apply_restrictions(split: pd.DataFrame, rules: list[dict], fid2vamp: dict,
                       wallet_incapable=frozenset(), wallet_frac: dict | None = None,
                       wallet_default: float = 0.0,
                       usa_only=frozenset(), nonusa_frac: dict | None = None,
                       nonusa_default: float = 0.0,
                       group_keys=("rpgt", "currency", "bank", "pmp", "ctry")) -> pd.DataFrame:
    """Return the split with bans + wallet capability + country capability enforced.

    split: rows with at least [gateway, share] and ideally [rpgt, currency, bank].
    rules: from load_restrictions.
    fid2vamp: gatewayFid(lower) -> vampMid(lower).
    wallet_incapable: set of gatewayFids/vampMids (lower) that can't do wallet.
    wallet_frac: {(currency, bank): fraction of the cell that is wallet traffic}.
    usa_only: set of gatewayFids/vampMids (lower) that can ONLY process USA traffic.
    nonusa_frac: {(currency, bank): fraction of the cell that is Non-USA traffic}.
    wallet_default / nonusa_default: reroute fraction for a cell absent from wallet_frac /
        nonusa_frac (default 0.0 — no reroute).
    group_keys: cell grouping for the capability blend (default (rpgt, currency, bank)).
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
        df["share"] = _capability_blend(df, gk, wallet_incapable, wallet_frac or {},
                                        wallet_default, kind="wallet")
        if gk:
            df["share"] = _renorm(df, gk, "share")

    # 3. Country capability — blend: USA-only gateways keep only their USA share; the
    #    Non-USA portion of each cell is redistributed to the other gateways. Same
    #    mechanism as wallet, with frac = the cell's Non-USA traffic fraction.
    if usa_only:
        df["share"] = _capability_blend(df, gk, usa_only, nonusa_frac or {}, nonusa_default,
                                        kind="nonusa")
        if gk:
            df["share"] = _renorm(df, gk, "share")

    if "cell_volume" in df.columns:
        df["volume"] = df["cell_volume"] * df["share"]
    return df.drop(columns=[c for c in ["_gw", "_vm"] if c in df.columns])


# ---------------------------------------------------------------------------
# POPULATION OPERATOR — the SAME eligibility maths as `apply_restrictions`, but
# precomputed ONCE for a fixed (cell, gateway) layout and then applied to a whole
# population of share vectors with pure numpy (no per-candidate DataFrame / groupby).
#
# Purpose: let a search (e.g. the genetic engine) SCORE the actually-routable shares —
# bans zeroed + renormalised, wallet / USA-only capability blended — inside its hot loop,
# so it optimises what will really be routed instead of a split that eligibility later
# perturbs. It is a fixed piecewise-linear transform of the share vector (masks + per-cell
# fractions are static), so it needs no projection and costs ~two segment-sums per stage.
#
# `build_elig_operator` returns the static arrays; `apply_elig_pop(X, op)` applies them.
# Proven row-for-row identical to `apply_restrictions` (see the backend equivalence test).
# ---------------------------------------------------------------------------
# [FN-064]
def build_elig_operator(cells: pd.DataFrame, rules: list[dict], fid2vamp: dict, *,
                        wallet_incapable=frozenset(), wallet_frac: dict | None = None,
                        wallet_default: float = 0.0,
                        usa_only=frozenset(), nonusa_frac: dict | None = None,
                        nonusa_default: float = 0.0) -> dict:
    """Precompute static per-row eligibility arrays for a FIXED layout.

    `cells`: one row per (cell, gateway) in the search's EXACT row order, rows CONTIGUOUS
    per cell, with columns at least [cell, gateway, currency, bank] (+ optional rpgt / bin /
    country, used only for ban matching). The cell segments this derives must equal the
    (rpgt, currency, bank) groups `apply_restrictions` renormalises within, so make `cell`
    that composite key. Returns a dict consumed by `apply_elig_pop`."""
    df = cells.reset_index(drop=True)
    n = len(df)
    _gw = df["gateway"].astype(str).str.strip().str.lower().to_numpy()
    _vm = pd.Series(_gw).map(fid2vamp).fillna(pd.Series(_gw)).to_numpy()
    _cell = df["cell"].astype(str).to_numpy()
    # contiguous cell segments (bit-for-bit the reduceat layout the caller's decode uses)
    starts = [0] + [i for i in range(1, n) if _cell[i] != _cell[i - 1]]
    cell_starts = np.asarray(starts, dtype=np.intp)
    cell_counts = np.diff(np.append(cell_starts, n)).astype(np.intp)

    prof_cols = [c for c in ("rpgt", "currency", "bank", "bin", "country") if c in df.columns]
    if rules:
        if prof_cols:
            _p = df[prof_cols].astype(str)
            profiles = [{c: _p.iat[i, j] for j, c in enumerate(prof_cols)} for i in range(n)]
        else:
            profiles = [{}] * n
        ban = np.fromiter((_row_banned(_gw[i], _vm[i], profiles[i], rules) for i in range(n)),
                          dtype=bool, count=n)
    else:
        ban = np.zeros(n, dtype=bool)

    _cur = (df["currency"].astype(str).str.strip().str.lower().to_numpy()
            if "currency" in df.columns else np.array([""] * n))
    _bnk = (df["bank"].astype(str).str.strip().str.lower().to_numpy()
            if "bank" in df.columns else np.array([""] * n))

    # [FN-065]
    def _incap_mask(incapable):
        if not incapable:
            return np.zeros(n, dtype=bool)
        _inc = frozenset(incapable)
        # FID-GRAIN (2026-08-17): `_gw` only — see the note in _capability_blend. Testing
        # `_vm` too rolled a vampMid's capability onto every one of its fids, barring
        # sibling fids that CAN serve (PaySafe - Total AV: USD capable, EUR/GBP not).
        # This operator is the population twin of apply_restrictions, so it must agree.
        return np.fromiter(((_gw[i] in _inc) for i in range(n)),
                           dtype=bool, count=n)

    # Sub-cell identity, when the caller supplies it. Without these the operator applies the
    # GLOBAL wallet / Non-USA fraction to every row — correct at cell grain, plainly wrong at
    # sub-cell grain where each cell is purely wallet or purely card, purely USA or purely not.
    _pmp_a = (df["pmp"].astype(str).str.strip().str.lower().to_numpy()
              if "pmp" in df.columns else None)
    _cty_a = None
    for _cc in ("ctry", "country"):
        if _cc in df.columns:
            _cty_a = df[_cc].astype(str).str.strip().str.lower().to_numpy()
            break
    _n_exact = {"wallet": 0, "nonusa": 0}

    # [FN-066]
    def _wf(frac_map, default, kind=None):
        fm = frac_map or {}
        _col = _pmp_a if kind == "wallet" else (_cty_a if kind == "nonusa" else None)
        _fn = _exact_wallet_frac if kind == "wallet" else _exact_nonusa_frac
        wf = np.empty(n, dtype=float)
        for i in range(n):
            _ex = _fn(_col[i]) if _col is not None else None
            if _ex is None:
                wf[i] = float(fm.get((_cur[i], _bnk[i]), default))
            else:
                wf[i] = _ex
                if kind:
                    _n_exact[kind] += 1
        wf = np.where(np.isnan(wf), 0.0, np.clip(wf, 0.0, 1.0))
        return wf

    return {
        "cell_starts": cell_starts, "cell_counts": cell_counts,
        "ban": ban, "has_ban": bool(rules) and bool(ban.any()),
        "w_incap": _incap_mask(wallet_incapable),
        "w_wf": _wf(wallet_frac, wallet_default, kind="wallet"),
        "has_w": bool(wallet_incapable),
        "u_incap": _incap_mask(usa_only),
        "u_wf": _wf(nonusa_frac, nonusa_default, kind="nonusa"),
        "has_u": bool(usa_only),
        # how many rows got the EXACT 0/1 factor instead of the global fraction (0 = cell grain)
        "n_rows": int(n), "w_exact": int(_n_exact["wallet"]), "u_exact": int(_n_exact["nonusa"]),
    }


# [FN-067]
def _renorm_pop(X: np.ndarray, cs: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Per-cell renormalise to sum 1, leaving all-zero cells (matches `_renorm`)."""
    s = np.repeat(np.add.reduceat(X, cs, axis=1), cc, axis=1)
    return np.where(s > 0, X / np.where(s > 0, s, 1.0), X)


# [FN-068]
def _blend_pop(X: np.ndarray, incap: np.ndarray, wf: np.ndarray,
               cs: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Vectorised twin of `_capability_blend` + its trailing `_renorm`, over a population.
    An incapable gateway keeps (1-wf) of its share; the wf portion redistributes to the
    capable gateways in the cell (renormalised among themselves). Cells with zero total,
    or with no capable gateway, are left unchanged — exactly as the scalar version."""
    base = X
    base_sum = np.repeat(np.add.reduceat(base, cs, axis=1), cc, axis=1)
    capX = base * (~incap)[None, :]
    s_cap = np.repeat(np.add.reduceat(capX, cs, axis=1), cc, axis=1)
    cshare = np.where(s_cap > 0, capX / np.where(s_cap > 0, s_cap, 1.0), base)
    wfb = wf[None, :]
    blended = wfb * cshare + (1.0 - wfb) * base
    out = np.where(base_sum > 0, blended, base)      # skip zero-total cells (the `continue`)
    return _renorm_pop(out, cs, cc)


# [FN-069]
def apply_elig_pop(X: np.ndarray, op: dict) -> np.ndarray:
    """Apply the prebuilt eligibility operator to shares X ((N,) or (P, N)). Reproduces
    `apply_restrictions` (bans -> 0 + renorm; wallet blend + renorm; USA blend + renorm),
    in the SAME order, as a pure-numpy population transform. Returns the same shape as X."""
    Xa = np.asarray(X, dtype=float)
    single = Xa.ndim == 1
    if single:
        Xa = Xa[None, :]
    cs, cc = op["cell_starts"], op["cell_counts"]
    if op.get("has_ban"):
        Xa = Xa * (~op["ban"])[None, :]
        Xa = _renorm_pop(Xa, cs, cc)
    if op.get("has_w"):
        Xa = _blend_pop(Xa, op["w_incap"], op["w_wf"], cs, cc)
    if op.get("has_u"):
        Xa = _blend_pop(Xa, op["u_incap"], op["u_wf"], cs, cc)
    return Xa[0] if single else Xa
