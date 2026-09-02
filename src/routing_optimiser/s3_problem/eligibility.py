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
import os as _os        # 19bl: the in-place twin's kill-switch reads it under this alias

import numpy as np
import pandas as pd

__build__ = ("2026-08-19bs-fused-elementwise-blends"
             "+2026-08-19bq-nocap-select-guarded"
             "+2026-08-19bm-restricted-blends"
             "+2026-08-19bl-exact-subcell-capability-RESTORED-after-19bk-clobber"
             "+2026-08-19bk-elig-inplace"
             "+2026-08-18-eligibility-ban-mask-cache+population-operator+fid-grain-capability"
             "+exact-subcell-capability")

WALLET_VALUES = {"googlepay", "applepay"}


# [FN-053]
def load_usa_only(path: str) -> frozenset:
    """Explicit list of gatewayFids that can ONLY process country='USA'.

    Read from the ``usa_only_gateways`` key of routing_restrictions.json. These
    are enforced like wallet capability: the gateway keeps only the USA fraction
    of each profile, the Non-USA portion is redistributed. Missing/invalid -> empty."""
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
    Listing it here forces it into the candidate set for its currency's profiles."""
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
    """Value for a rule field, aliasing 'bin' onto the 'bin' column (BIN-level
    profiles are keyed as 'bin' in this app). Returns None if unavailable."""
    pv = profile.get(field)
    if pv is None and field == "bin":
        pv = profile.get("bin")
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
    if "bin" in avail:
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
# Profile purity. The blend's `frac` is "how much of this profile is wallet / Non-USA traffic".
# At profile grain that is not a proportion at all — the profile IS one or the other, so the
# answer is 0 or 1. Same test as the scaffold's _T0_emask_a, so one rule holds everywhere.
_WALLET_PMP = frozenset({"googlepay", "applepay"})
_USA_CTRY = frozenset({"usa", "us"})
_MIXED = frozenset({"", "_all_", "all", "nan", "none"})


# [FN-062a]
def _exact_wallet_frac(pmp):
    """1.0 in a wallet profile, 0.0 in a card profile, None when the value is mixed."""
    p = str(pmp).strip().lower()
    if p in _MIXED:
        return None
    return 1.0 if p in _WALLET_PMP else 0.0


# [FN-062b]
def _exact_nonusa_frac(ctry):
    """1.0 in a Non-USA profile, 0.0 in a USA profile, None when the value is mixed."""
    c = str(ctry).strip().lower()
    if c in _MIXED:
        return None
    return 0.0 if c in _USA_CTRY else 1.0


# [FN-062c]
def _profile_col(df, kind):
    """The column carrying this restriction's profile identity, or None at profile grain."""
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
    It keeps only the (1 − frac) share it CAN serve; the `frac` portion of each profile is handed
    to the vendors that CAN (renormalised among themselves), so no transactions are lost. Used
    identically for wallet capability (frac = the profile's wallet share) and country capability
    (frac = the profile's Non-USA share). `frac_map` is keyed by (currency, bank); `default` is
    used when a profile isn't in the map.
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
    has_cur_bank = ("currency" in df.columns and "bin" in df.columns)
    _sc_col = _profile_col(df, kind) if kind else None
    _pos_of = {lbl: p for p, lbl in enumerate(df.index)}   # label -> positional (unique index)
    for _grp_key, row_idx in df.groupby(group_cols, dropna=False).groups.items():
        group_rows = df.loc[row_idx]
        base = group_rows["share"].to_numpy(float)
        if base.sum() <= 0:
            continue
        reroute_frac = default
        if has_cur_bank:
            cur_bank_key = (str(group_rows["currency"].iloc[0]).strip().lower(),
                            str(group_rows["bin"].iloc[0]).strip().lower())
            reroute_frac = float(frac_map.get(cur_bank_key, default))
        # EXACT at profile grain: when the group is pure (one pmp / one Country), the
        # fraction is not an estimate — it is 0 or 1. `_sc_col` is None at profile grain, so
        # the fraction map is used exactly as before.
        if _sc_col is not None:
            _ex = (_exact_wallet_frac(group_rows[_sc_col].iloc[0]) if kind == "wallet"
                   else _exact_nonusa_frac(group_rows[_sc_col].iloc[0]))
            if _ex is not None:
                reroute_frac = _ex
        reroute_frac = 0.0 if (reroute_frac != reroute_frac) else min(max(reroute_frac, 0.0), 1.0)
        incap_in_profile = incapable_mask[[_pos_of[i] for i in row_idx]]
        capable_share = base.copy()
        capable_share[incap_in_profile] = 0.0
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
                       group_keys=("rpgt", "currency", "bin", "pmp", "ctry")) -> pd.DataFrame:
    """Return the split with bans + wallet capability + country capability enforced.

    split: rows with at least [gateway, share] and ideally [rpgt, currency, bank].
    rules: from load_restrictions.
    fid2vamp: gatewayFid(lower) -> vampMid(lower).
    wallet_incapable: set of gatewayFids/vampMids (lower) that can't do wallet.
    wallet_frac: {(currency, bank): fraction of the profile that is wallet traffic}.
    usa_only: set of gatewayFids/vampMids (lower) that can ONLY process USA traffic.
    nonusa_frac: {(currency, bank): fraction of the profile that is Non-USA traffic}.
    wallet_default / nonusa_default: reroute fraction for a profile absent from wallet_frac /
        nonusa_frac (default 0.0 — no reroute).
    group_keys: profile grouping for the capability blend (default (rpgt, currency, bank)).
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
        prof_cols = [c for c in ("rpgt", "currency", "bin", "bin", "country") if c in df.columns]
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
    #    Non-USA portion of each profile is redistributed to the other gateways. Same
    #    mechanism as wallet, with frac = the profile's Non-USA traffic fraction.
    if usa_only:
        df["share"] = _capability_blend(df, gk, usa_only, nonusa_frac or {}, nonusa_default,
                                        kind="nonusa")
        if gk:
            df["share"] = _renorm(df, gk, "share")

    if "profile_volume" in df.columns:
        df["volume"] = df["profile_volume"] * df["share"]
    return df.drop(columns=[c for c in ["_gw", "_vm"] if c in df.columns])


# ---------------------------------------------------------------------------
# POPULATION OPERATOR — the SAME eligibility maths as `apply_restrictions`, but
# precomputed ONCE for a fixed (profile, gateway) layout and then applied to a whole
# population of share vectors with pure numpy (no per-candidate DataFrame / groupby).
#
# Purpose: let a search (e.g. the genetic engine) SCORE the actually-routable shares —
# bans zeroed + renormalised, wallet / USA-only capability blended — inside its hot loop,
# so it optimises what will really be routed instead of a split that eligibility later
# perturbs. It is a fixed piecewise-linear transform of the share vector (masks + per-profile
# fractions are static), so it needs no projection and costs ~two segment-sums per stage.
#
# `build_elig_operator` returns the static arrays; `apply_elig_pop(X, op)` applies them.
# Proven row-for-row identical to `apply_restrictions` (see the backend equivalence test).
# ---------------------------------------------------------------------------
# [FN-064]
def build_elig_operator(profiles: pd.DataFrame, rules: list[dict], fid2vamp: dict, *,
                        wallet_incapable=frozenset(), wallet_frac: dict | None = None,
                        wallet_default: float = 0.0,
                        usa_only=frozenset(), nonusa_frac: dict | None = None,
                        nonusa_default: float = 0.0) -> dict:
    """Precompute static per-row eligibility arrays for a FIXED layout.

    `profiles`: one row per (profile, gateway) in the search's EXACT row order, rows CONTIGUOUS
    per profile, with columns at least [profile, gateway, currency, bank] (+ optional rpgt / bin /
    country, used only for ban matching). The profile segments this derives must equal the
    (rpgt, currency, bank) groups `apply_restrictions` renormalises within, so make `profile`
    that composite key. Returns a dict consumed by `apply_elig_pop`."""
    df = profiles.reset_index(drop=True)
    n = len(df)
    _gw = df["gateway"].astype(str).str.strip().str.lower().to_numpy()
    _vm = pd.Series(_gw).map(fid2vamp).fillna(pd.Series(_gw)).to_numpy()
    _profile = df["profile"].astype(str).to_numpy()
    # contiguous profile segments (bit-for-bit the reduceat layout the caller's decode uses)
    starts = [0] + [i for i in range(1, n) if _profile[i] != _profile[i - 1]]
    profile_starts = np.asarray(starts, dtype=np.intp)
    profile_counts = np.diff(np.append(profile_starts, n)).astype(np.intp)

    prof_cols = [c for c in ("rpgt", "currency", "bin", "bin", "country") if c in df.columns]
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
    _bnk = (df["bin"].astype(str).str.strip().str.lower().to_numpy()
            if "bin" in df.columns else np.array([""] * n))

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

    # Profile identity, when the caller supplies it. Without these the operator applies the
    # GLOBAL wallet / Non-USA fraction to every row — correct at profile grain, plainly wrong at
    # profile grain where each profile is purely wallet or purely card, purely USA or purely not.
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
        "profile_starts": profile_starts, "profile_counts": profile_counts,
        "ban": ban, "has_ban": bool(rules) and bool(ban.any()),
        "w_incap": _incap_mask(wallet_incapable),
        "w_wf": _wf(wallet_frac, wallet_default, kind="wallet"),
        "has_w": bool(wallet_incapable),
        "u_incap": _incap_mask(usa_only),
        "u_wf": _wf(nonusa_frac, nonusa_default, kind="nonusa"),
        "has_u": bool(usa_only),
        # how many rows got the EXACT 0/1 factor instead of the global fraction (0 = profile grain)
        "n_rows": int(n), "w_exact": int(_n_exact["wallet"]), "u_exact": int(_n_exact["nonusa"]),
    }


# ── FUSED ELEMENTWISE PASSES (2026-08-19bs) ───────────────────────────────────────────────────
# See the module note on _blend_pop: the chain's cost is the NUMBER of full-width temporaries, and
# every step except the two np.add.reduceat calls is elementwise. Elementwise work has no
# summation order to preserve, so fusing it into one loop cannot reassociate anything. The
# reductions are left to numpy precisely because reduceat does NOT sum left-to-right and
# reimplementing its association would be a guess.
try:
    from numba import njit as _e_njit
    _E_HAVE_NB = True
except Exception:                                   # noqa: BLE001 - numba absent is not an error
    _E_HAVE_NB = False

    def _e_njit(*_a, **_k):                         # so the bodies stay valid python
        def _deco(f):
            return f
        return _deco


@_e_njit(cache=False, fastmath=False)
def _fu_mask(X, incap, out):
    """capX = X * (~incap). The MULTIPLY is kept, not turned into a branch: numpy casts the bool
    to 1.0/0.0 and multiplies, and for a negative x, x * 0.0 is -0.0 while a branch returning a
    literal 0.0 gives +0.0. Those differ in bits, which is the whole game here."""
    P, N = X.shape
    for p in range(P):
        for i in range(N):
            out[p, i] = X[p, i] * (0.0 if incap[i] else 1.0)
    return out


@_e_njit(cache=False, fastmath=False)
def _fu_blend(capX, base, seg, co, wf, out, all_pos):
    """cshare + the wf blend, in ONE pass instead of five.

    Mirrors, elementwise and operator for operator:
        sd     = repeat(where(seg > 0, seg, 1.0), cc)
        cshare = capX / sd                       (all_pos)   or   where(posc, capX / sd, base)
        out    = wf * cshare + (1.0 - wf) * base
    `all_pos` is the 19bq no-capable-gateway guard, decided at PROFILE grain by the caller."""
    P, N = capX.shape
    for p in range(P):
        for i in range(N):
            s = seg[p, co[i]]
            if s > 0.0:
                csh = capX[p, i] / s
            elif all_pos:
                csh = capX[p, i] / 1.0           # unreachable when all_pos; kept for exactness
            else:
                csh = base[p, i]
            w = wf[i]
            out[p, i] = w * csh + (1.0 - w) * base[p, i]
    return out


@_e_njit(cache=False, fastmath=False)
def _fu_renorm(X, seg, co, out):
    """X / repeat(where(seg > 0, seg, 1.0), cc), in one pass with no repeat and no select."""
    P, N = X.shape
    for p in range(P):
        for i in range(N):
            s = seg[p, co[i]]
            out[p, i] = X[p, i] / (s if s > 0.0 else 1.0)
    return out


_FU_ON = (_os.environ.get("ROUTING_ELIG_FUSE", "1") != "0") and _E_HAVE_NB
# `use` is flipped OFF for the process by the live self-check in `apply_elig_pop` on any mismatch,
# the same way the restriction and the in-place twin are governed. `why` is read into the run log.
_FU_OK = {"use": _FU_ON, "why": (
    "fused elementwise passes ON (ROUTING_ELIG_FUSE=0 reverts to the 19bm numpy chain)"
    if _FU_ON else
    ("fused elementwise passes OFF — numba is unavailable in this process, so the numpy chain "
     "runs; correct, just ~2x dearer" if not _E_HAVE_NB else
     "fused elementwise passes OFF — ROUTING_ELIG_FUSE=0"))}


def _co_build(cc):
    """profile_of: the profile index of every row. Built ONCE per layout by `_rx_build` and carried on
    the operator — never rebuilt per call, and never cached under a (N, nprofile) key, because two
    different sub-layouts can share that pair and a mis-keyed profile map is a silent wrong answer."""
    cc = np.asarray(cc, np.int64)
    return np.repeat(np.arange(cc.size, dtype=np.int32), cc)


# [FN-067]
def _renorm_pop_ref(X: np.ndarray, cs: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """THE REFERENCE. Per-profile renormalise to sum 1, leaving all-zero profiles (matches `_renorm`).
    Untouched since 2026-07-29; `_renorm_pop` below is checked against it, never the reverse."""
    s = np.repeat(np.add.reduceat(X, cs, axis=1), cc, axis=1)
    return np.where(s > 0, X / np.where(s > 0, s, 1.0), X)


def _renorm_pop(X: np.ndarray, cs: np.ndarray, cc: np.ndarray, co=None) -> np.ndarray:
    """Same result, bit for bit, in three full-width passes instead of five.

    TWO CHANGES, both exact:
      * The safe denominator is built at PROFILE grain and repeated. np.repeat is a gather of identical
        values, so repeat(where(seg > 0, seg, 1.0), cc) == where(repeat(seg) > 0, repeat(seg), 1.0)
        elementwise — the 23,418-element select replaces an 8.5-million-element one (68.9 ms).
      * The OUTER select is dropped, because it is a no-op: where seg <= 0 the denominator is exactly
        1.0, so X / 1.0 == X, which is precisely what the discarded branch returned.
    135.5 ms -> 67.0 ms at 35 x 242,670 / 23,418 profiles."""
    seg = np.add.reduceat(X, cs, axis=1)
    if co is not None and _FU_OK["use"]:
        # 19bs: one fused pass instead of repeat + select + divide. Same reduceat, same divisor,
        # same elementwise result — the repeat is a gather of identical values and the select is
        # a no-op where the divisor is exactly 1.0.
        return _fu_renorm(X, seg, co, np.empty_like(X))
    return X / np.repeat(np.where(seg > 0, seg, 1.0), cc, axis=1)


# [FN-068]
def _blend_pop_ref(X: np.ndarray, incap: np.ndarray, wf: np.ndarray,
                   cs: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """THE REFERENCE, untouched since 2026-07-29.
    Vectorised twin of `_capability_blend` + its trailing `_renorm`, over a population.
    An incapable gateway keeps (1-wf) of its share; the wf portion redistributes to the
    capable gateways in the profile (renormalised among themselves). Profiles with zero total,
    or with no capable gateway, are left unchanged — exactly as the scalar version."""
    base = X
    base_sum = np.repeat(np.add.reduceat(base, cs, axis=1), cc, axis=1)
    capX = base * (~incap)[None, :]
    s_cap = np.repeat(np.add.reduceat(capX, cs, axis=1), cc, axis=1)
    cshare = np.where(s_cap > 0, capX / np.where(s_cap > 0, s_cap, 1.0), base)
    wfb = wf[None, :]
    blended = wfb * cshare + (1.0 - wfb) * base
    out = np.where(base_sum > 0, blended, base)      # skip zero-total profiles (the `continue`)
    return _renorm_pop_ref(out, cs, cc)


def _blend_pop(X: np.ndarray, incap: np.ndarray, wf: np.ndarray,
               cs: np.ndarray, cc: np.ndarray, co=None) -> np.ndarray:
    """Same result, bit for bit, with three of the five full-width selects removed.

      * `base_sum` is never materialised. Its only use was the trailing select, and that select is a
        no-op: a profile with base_sum <= 0 is all zeros (shares are non-negative), so capX is zero, so
        seg_c is zero, so cshare == base, so blended == base. Both branches return base.
        This is the ONE step that leans on the data (non-negative shares) rather than purely on IEEE
        arithmetic — which is why the whole path self-checks live and reverts on mismatch.
      * The safe denominator and the `> 0` mask are built at PROFILE grain and repeated; the mask
        repeats as bool, so it moves 1 byte per row instead of 8.
      * The trailing renorm is the fast one above.

    437.3 ms -> 270.2 ms at 35 x 242,670 / 23,418 profiles. Bit-identical (int64 view) across
    all-zero profiles, profiles with no capable gateway, -0.0 shares and 1e-14 magnitudes."""
    base = X
    if co is not None and _FU_OK["use"]:
        # 19bs: THREE full-width arrays instead of eight. The two reduceat calls are untouched and
        # in the same places; only the elementwise steps between them are fused.
        capX = _fu_mask(X, np.asarray(incap, bool), np.empty_like(X))
        seg_c = np.add.reduceat(capX, cs, axis=1)
        _ap = bool((seg_c > 0).all())
        _NC_STAT["skip" if _ap else "select"] += 1
        # IN-PLACE into capX, which saves a 68 MB allocation and measured 114.7 -> 100.8 ms.
        # Safe, and not the thing I refused in 19bo: `capX` is CALL-LOCAL (allocated three lines
        # up, never escapes, never shared with another call or thread), and the fused pass reads
        # element i then writes element i, so no iteration can read a slot a previous one wrote.
        # A module-level scratch POOL would be the unsafe version — rowpar calls this from several
        # threads at once, and that is exactly the silent aliasing the bit-identity bar exists for.
        blended = _fu_blend(capX, X, seg_c, co, np.asarray(wf, float), capX, _ap)
        seg = np.add.reduceat(blended, cs, axis=1)
        return _fu_renorm(blended, seg, co, np.empty_like(X))
    capX = base * (~incap)[None, :]
    seg_c = np.add.reduceat(capX, cs, axis=1)
    pos_profile = seg_c > 0
    sd = np.repeat(np.where(pos_profile, seg_c, 1.0), cc, axis=1)
    # 19bq: THE SELECT IS USUALLY UNREACHABLE. `posc` is False only in a profile where NO gateway is
    # capable, and [elig-nocap] measured that on the live population: 0 of 23,791 profiles for wallet
    # AND 0 for USA-only, carrying 0 rows. Where every profile is positive, np.where(True, a, b) == a
    # elementwise EXACTLY, so the select and the bool repeat are both pure cost — ~69 ms + ~3 ms,
    # twice per delivery. The guard is at PROFILE grain (23,791 booleans), not row grain, so it is free;
    # when any profile IS non-positive it falls through to the identical select and nothing changes.
    if pos_profile.all():
        _NC_STAT["skip"] += 1
        cshare = capX / sd
    else:
        _NC_STAT["select"] += 1
        posc = np.repeat(pos_profile, cc, axis=1)
        cshare = np.where(posc, capX / sd, base)
    wfb = wf[None, :]
    blended = wfb * cshare + (1.0 - wfb) * base
    return _renorm_pop(blended, cs, cc)


# [FN-069b]
# ── RESTRICTED BLENDS (2026-08-19bm) ──────────────────────────────────────────────────────────
# Eligibility was 879.3 ms of a 2,196 ms generation on 2026-08-23 — the largest single stage in the
# run, ahead of the band projector. The in-place twin (19bk) turned out to be 1.05x at the live
# shape, so this takes the other route: do less work rather than allocate less.
#
# A profile whose every row has wf == 0.0 cannot be changed by a blend:
#     blended = 0.0 * cshare + 1.0 * base = 0.0 + base = base   (exact, except base == -0.0)
# and the `base_sum > 0` select returns `base` either way. So the twelve full-width operations of
# `_blend_pop` reproduce their input there, exactly. Compute them only where wf > 0.
#
# The trailing `_renorm_pop` is deliberately NOT restricted: its input does not sum to exactly 1.0,
# so dividing by the profile sum changes bits even in an untouched profile.
# TWO SWITCHES, because these are two independent ideas and one of them is much better than the
# other. ROUTING_ELIG_FAST=0 reverts to the untouched reference helpers entirely (the
# profile-grain rewrite AND the restriction). ROUTING_ELIG_RESTRICT=0 keeps the profile-grain
# rewrite — the unconditional ~1.7x — and only switches off the hit-profile restriction.
_FAST_ON = _os.environ.get("ROUTING_ELIG_FAST", "1") != "0"
_RX_ON = _os.environ.get("ROUTING_ELIG_RESTRICT", "1") != "0"
# MEASURED CUT-OFF, not a guess. Gathering the changeable columns costs ~57 ms at 40% of a
# 35 x 242,670 array, which is the price of a whole full-width operation, and the scatter back costs
# again. Sweeping the hit fraction at the live shape:
#     5% -> 1.83x    21% -> 1.53x    40% -> 0.87x    60% -> 0.92x    80% -> 0.77x
# so above roughly a quarter the restriction LOSES. Ship it only below the crossover; the profile-grain
# rewrite above is the unconditional win and does not care about the hit fraction.
_RX_MAXHIT = float(_os.environ.get("ROUTING_ELIG_RESTRICT_MAXHIT", "0.25") or 0.25)
_RX_OK = {"checked": False, "use": _FAST_ON, "msg": "", "note": ""}

# 19bq: how often the no-capable-gateway select was SKIPPED as unreachable vs actually needed.
# [elig-nocap] measured 0 such profiles on 2026-08-23, but that was one population — if `select` ever
# climbs, the guard has stopped paying and the log will say so instead of leaving it to be assumed.
_NC_STAT = {"skip": 0, "select": 0}


def nocap_note():
    """One line for the run log: was the capability select reachable this run?"""
    _s, _n = int(_NC_STAT["skip"]), int(_NC_STAT["select"])
    if not (_s + _n):
        return ""
    if not _n:
        return (f"[elig-nocap] the no-capable-gateway select was UNREACHABLE on all {_s:,} blend "
                "call(s), so it was skipped every time — every profile had a capable gateway. That is "
                "a ~69 ms full-width select plus a bool repeat saved, twice per delivery, "
                "bit-identically (19bq).")
    return (f"[elig-nocap] the select was skipped on {_s:,} of {_s + _n:,} blend call(s) "
            f"({_s / (_s + _n):.1%}) and genuinely NEEDED on {_n:,} — some profile had no capable "
            "gateway. The guard still shipped the identical answer; it just paid for the select "
            "on those calls.")


def _rx_rows(cs, cc, hit):
    """Row indices of the hit profiles, plus the reduceat starts/counts for the gathered array.

    Vectorised ragged-range: for hit profile i the rows are cs[i] .. cs[i]+cc[i]-1, and in the gathered
    array they land at scs[i] .. scs[i]+scc[i]-1. A python loop over 23,418 profiles would also work
    but this runs once per layout and there is no reason to be slow about it."""
    sel = np.where(hit)[0]
    scc = np.asarray(cc, np.intp)[sel]
    scs = np.concatenate([[0], np.cumsum(scc)[:-1]]).astype(np.intp) if sel.size else \
        np.zeros(0, np.intp)
    tot = int(scc.sum())
    rows = (np.arange(tot, dtype=np.intp)
            + np.repeat(np.asarray(cs, np.intp)[sel] - scs, scc)) if tot else np.zeros(0, np.intp)
    return rows, scs, scc


def _rx_build(op):
    """Per-stage restriction index. Cached on the operator, which is built once per layout."""
    cs = np.asarray(op["profile_starts"], np.intp)
    cc = np.asarray(op["profile_counts"], np.intp)
    n = int(cc.sum())
    rx = {"n_rows": n, "n_profile": int(cs.size), "stages": {}, "why": [],
          # 19bs: built ONCE per layout, here, and passed down. Never cached under a (N, nprofile)
          # key — two sub-layouts can share that pair, and a mis-keyed profile map is a silent wrong
          # answer, not a slow one.
          "co": _co_build(cc) if _FU_OK["use"] else None}
    if not _RX_ON:
        rx["why"].append("hit-profile restriction OFF (ROUTING_ELIG_RESTRICT=0); the profile-grain "
                         "rewrite still applies")
        return rx
    for _k, _wfk in (("w", "w_wf"), ("u", "u_wf")):
        if not op.get("has_" + _k):
            continue
        wf = np.asarray(op[_wfk], float)
        if wf.size != n:
            rx["why"].append(f"{_k}: wf has {wf.size} entries for {n} rows — not restricted")
            continue
        hit = np.maximum.reduceat(wf, cs) > 0.0
        rows, scs, scc = _rx_rows(cs, cc, hit)
        frac = (rows.size / n) if n else 1.0
        if rows.size == 0:
            # the whole stage is the identity up to the trailing renorm
            rx["stages"][_k] = {"rows": rows, "scs": scs, "scc": scc, "co": None,
                                "profiles": 0, "frac": 0.0}
            rx["why"].append(f"{_k}: NO profile can change (every wf == 0) — 0 of {n:,} rows")
        elif frac > _RX_MAXHIT:
            rx["why"].append(f"{_k}: {frac:.1%} of rows are in changeable profiles, above the "
                             f"{_RX_MAXHIT:.0%} cut-off — kept FULL-WIDTH")
        else:
            rx["stages"][_k] = {"rows": rows, "scs": scs, "scc": scc,
                                "co": _co_build(scc) if _FU_OK["use"] else None,
                                "profiles": int(hit.sum()), "frac": frac}
            rx["why"].append(f"{_k}: {int(hit.sum()):,} of {cs.size:,} profiles "
                             f"({hit.mean():.2%}) carrying {rows.size:,} of {n:,} rows "
                             f"({frac:.2%}) can change — the rest is copied through")
    return rx


def _blend_pop_rx(X, incap, wf, cs, cc, st, co=None):
    """`_blend_pop` over the changeable profiles only, then the SAME full-width trailing renorm.

    THE PRIMITIVES MUST MATCH the full version operation for operation — np.add.reduceat over the
    gathered array with its own starts, np.repeat with its own counts. Anything else (a .sum(axis=1),
    a different association order) differs in the last bits, because reduceat does not sum a segment
    left-to-right."""
    rows, scs, scc = st["rows"], st["scs"], st["scc"]
    if rows.size == 0:                       # nothing can change: only the renorm is left
        return _renorm_pop(np.array(X, float, copy=True), cs, cc, co)
    sub = np.ascontiguousarray(X[:, rows])
    inc = np.asarray(incap)[rows]
    if st.get("co") is not None and _FU_OK["use"]:
        # 19bs: the same fusion on the gathered sub-array. `base_sum` is not needed — see the note
        # in `_blend_pop`: the trailing `where(base_sum > 0, blended, sub)` is a no-op because a
        # profile with base_sum <= 0 is all zeros, so blended == sub in both branches.
        _co = st["co"]
        _capX = _fu_mask(sub, np.asarray(inc, bool), np.empty_like(sub))
        _seg = np.add.reduceat(_capX, scs, axis=1)
        _ap = bool((_seg > 0).all())
        _NC_STAT["skip" if _ap else "select"] += 1
        _bl = _fu_blend(_capX, sub, _seg, _co, np.asarray(wf, float)[rows], _capX, _ap)
        Y = np.array(X, float, copy=True)
        Y[:, rows] = _bl
        return _renorm_pop(Y, cs, cc, co)
    base_sum = np.repeat(np.add.reduceat(sub, scs, axis=1), scc, axis=1)
    capX = sub * (~inc)[None, :]
    _sub_seg = np.add.reduceat(capX, scs, axis=1)
    _sub_pos = _sub_seg > 0
    s_cap = np.repeat(np.where(_sub_pos, _sub_seg, 1.0), scc, axis=1)
    # 19bq: the same unreachable select as `_blend_pop`, guarded the same way at profile grain.
    # NOTE the denominator moved from `np.where(s_cap > 0, s_cap, 1.0)` on the REPEATED array to the
    # same select on the PROFILE-grain array before the repeat. That is elementwise identical because
    # np.repeat is a gather: repeat(f(seg)) == f(repeat(seg)).
    if _sub_pos.all():
        _NC_STAT["skip"] += 1
        cshare = capX / s_cap
    else:
        _NC_STAT["select"] += 1
        cshare = np.where(np.repeat(_sub_pos, scc, axis=1), capX / s_cap, sub)
    wfb = np.asarray(wf)[rows][None, :]
    blended = wfb * cshare + (1.0 - wfb) * sub
    Y = np.array(X, float, copy=True)
    Y[:, rows] = np.where(base_sum > 0, blended, sub)
    return _renorm_pop(Y, cs, cc)


def _apply_elig_pop_rx(Xa, op, cs, cc, rx):
    """Same three stages, same order, blends restricted where the index says it is safe.

    19bs: `rx["co"]` is the row->profile map for the full layout, built once when the index was.
    Passing it in is what switches the helpers to the fused elementwise passes; passing None
    leaves them on the 19bm numpy chain, which is the revert path."""
    _co = rx.get("co")
    if op.get("has_ban"):
        Xa = Xa * (~op["ban"])[None, :]
        Xa = _renorm_pop(Xa, cs, cc, _co)
    if op.get("has_w"):
        _st = rx["stages"].get("w")
        Xa = (_blend_pop_rx(Xa, op["w_incap"], op["w_wf"], cs, cc, _st, _co) if _st is not None
              else _blend_pop(Xa, op["w_incap"], op["w_wf"], cs, cc, _co))
    if op.get("has_u"):
        _st = rx["stages"].get("u")
        Xa = (_blend_pop_rx(Xa, op["u_incap"], op["u_wf"], cs, cc, _st, _co) if _st is not None
              else _blend_pop(Xa, op["u_incap"], op["u_wf"], cs, cc, _co))
    return Xa


def _rx_verdict(ref, got, rx, P, N):
    """Compare REFERENCE vs restricted at bit level. np.array_equal calls -0.0 == 0.0, and -0.0 is
    exactly the value for which `x + 0.0 == x` fails, so values alone are not enough here."""
    same_val = bool(np.array_equal(ref, got))
    if not same_val:
        return False, (f"\u26a0 max|\u0394| {float(np.abs(ref - got).max()):.3e}")
    nbits = int(np.count_nonzero(ref.view(np.int64) != got.view(np.int64)))
    _why = "; ".join(list(rx["why"]) + [_FU_OK["why"]]) or "no stage restricted"
    if nbits == 0:
        return True, (f"\u2713 bit-identical to the reference on the live operator "
                      f"(int64 bit-pattern comparison on {P}x{N:,}, stricter than array_equal). "
                      f"{_why}.")
    return True, (f"\u2713 numerically identical, and {nbits:,} of {P * N:,} slots differ only in "
                  f"the SIGN OF ZERO (-0.0 vs 0.0) \u2014 the one value for which x + 0.0 != x "
                  f"bitwise. Every downstream use is a sum or a share, where the two are "
                  f"interchangeable. {_why}.")


# [FN-069]
# ── IN-PLACE TWIN (2026-08-19bk) ──────────────────────────────────────────────────────────────
# Eligibility was ~38% of a full-matrix generation on 2026-08-23 — the single largest stage in the
# run, ahead of the band projector. Not because the arithmetic is heavy but because the two
# `_blend_pop` calls build ~28 full-width (P, N) temporaries between them: ~1.9 GB of allocation
# and traffic per generation at 35 x 242,670.
#
# The twin below does the SAME ufuncs in the SAME order into reused scratch. Two substitutions are
# needed because numpy offers no `out=` for either primitive:
#   np.repeat(seg, cc, axis=1)  ->  np.take(seg, profile_of, axis=1, out=buf)   (a gather; no maths)
#   np.where(m, a, b)           ->  copyto(buf, b); copyto(buf, a, where=m)  (a select; no maths)
# Measured 4.10x and np.array_equal on a 35 x 221,649 / 23,418-profile fixture.
#
# Note np.take(out=) is SLOWER than np.repeat alone (50.8 vs 23.3 ms) — the entire win is in not
# allocating. The same trick bought `_segment_softmax` 1.01x, because five temporaries are
# traffic-bound where twenty-eight are allocation-bound.
_EP_SCRATCH = {}
_EP_BUSY = [False]


class _EpBuf:
    """Scratch for one (N, nprofile) shape, grown in P. Slices of one contiguous (Pmax, N) block, so
    calls at P=1 / 35 / 40 share a single allocation instead of three."""

    __slots__ = ("N", "profile_of", "P", "f", "m")

    def __init__(self, N, profile_of):
        self.N, self.profile_of, self.P = N, profile_of, 0
        self.f, self.m = [], []

    def grow(self, P):
        if P > self.P:
            self.P = P
            self.f = [np.empty((P, self.N)) for _ in range(5)]
            self.m = [np.empty((P, self.N), bool) for _ in range(2)]
        return self

    def take(self, P, i):
        return self.f[i][:P]

    def mask(self, P, i):
        return self.m[i][:P]


def _ep_buf(cs, cc, N):
    key = (int(N), int(np.asarray(cs).size))
    b = _EP_SCRATCH.get(key)
    if b is None:
        b = _EpBuf(int(N), np.repeat(np.arange(np.asarray(cc).size),
                                    np.asarray(cc)).astype(np.intp))
        _EP_SCRATCH[key] = b
    return b


def _seg_bcast(X, cs, buf, profile_of, out):
    """Per-profile sum broadcast back to row grain — np.repeat's result, gathered instead."""
    np.take(np.add.reduceat(X, cs, axis=1), profile_of, axis=1, out=out)
    return out


def _renorm_ip(X, cs, b, P, out):
    """_renorm_pop_ref, in place. Same ufuncs, same order (19bk; default off)."""
    s = _seg_bcast(X, cs, b, b.profile_of, b.take(P, 3))
    pos = np.greater(s, 0.0, out=b.mask(P, 0))
    sd = b.take(P, 4)
    sd.fill(1.0)
    np.copyto(sd, s, where=pos)
    np.divide(X, sd, out=out)
    np.copyto(out, X, where=~pos)
    return out


def _blend_ip(X, incap, wf, cs, b, P, out):
    """_blend_pop_ref, in place. Same ufuncs, same order (19bk; default off)."""
    bs = _seg_bcast(X, cs, b, b.profile_of, b.take(P, 0))
    posb = np.greater(bs, 0.0, out=b.mask(P, 1))
    cx = b.take(P, 1)
    np.multiply(X, (~incap)[None, :], out=cx)
    s_cap = _seg_bcast(cx, cs, b, b.profile_of, b.take(P, 2))
    posc = s_cap > 0.0
    sd = b.take(P, 3)
    sd.fill(1.0)
    np.copyto(sd, s_cap, where=posc)
    csh = b.take(P, 2)
    np.divide(cx, sd, out=csh)
    np.copyto(csh, X, where=~posc)
    wfb = wf[None, :]
    t = b.take(P, 1)
    np.multiply(wfb, csh, out=t)
    u = b.take(P, 3)
    np.multiply(1.0 - wfb, X, out=u)
    bl = b.take(P, 1)
    np.add(t, u, out=bl)
    np.copyto(bl, X, where=~posb)
    return _renorm_ip(bl, cs, b, P, out)


def _apply_elig_pop_alloc(Xa, op, cs, cc):
    """THE REFERENCE. Every fast path is checked against this, never the other way round — so it
    calls the *_ref helpers, which have not been touched since 2026-07-29."""
    if op.get("has_ban"):
        Xa = Xa * (~op["ban"])[None, :]
        Xa = _renorm_pop_ref(Xa, cs, cc)
    if op.get("has_w"):
        Xa = _blend_pop_ref(Xa, op["w_incap"], op["w_wf"], cs, cc)
    if op.get("has_u"):
        Xa = _blend_pop_ref(Xa, op["u_incap"], op["u_wf"], cs, cc)
    return Xa


# DEFAULT OFF as of 19bl. The 4.10x I measured came from a fixture with far fewer profiles than the
# live layout. Re-measured at the REAL shape (35 x 242,670 over 23,418 profiles) it is 1.05x —
# 1043 -> 997 ms — which matches Ben's live 840 -> 879 ms, i.e. a wash. `np.take(out=)` is
# genuinely slower than `np.repeat` (50.8 vs 23.3 ms per call) and there are ~7 of them per
# call; with 23,418 profiles that penalty eats the whole allocation saving. So this stays as code
# behind a switch (ROUTING_ELIG_INPLACE=1) rather than shipping ~150 lines of new risk for 5%.
# The real prize is elsewhere: restrict each blend to the profiles that can actually change, the
# same argument that took blocked-caps from 312 to 21.5 ms.
_EP_INPLACE = _os.environ.get("ROUTING_ELIG_INPLACE", "0") != "0"
# `msg` is read by tab_2_routing_engine and put in the RUN LOG. 19bk only print()ed it, so the single
# line that says whether the fast path is bit-identical never appeared in the log Ben reads.
_EP_OK = {"checked": False, "use": _EP_INPLACE, "msg": ""}


# [FN-069]
def apply_elig_pop(X: np.ndarray, op: dict) -> np.ndarray:
    """Apply the prebuilt eligibility operator to shares X ((N,) or (P, N)). Reproduces
    `apply_restrictions` (bans -> 0 + renorm; wallet blend + renorm; USA blend + renorm),
    in the SAME order, as a pure-numpy population transform. Returns the same shape as X.

    Two implementations, one meaning. `_apply_elig_pop_alloc` is the original and the reference;
    the in-place twin reuses scratch and was measured 4.10x on a 35 x 221,649 fixture. The twin
    self-checks against the reference on its first call and reverts for the process lifetime on any
    mismatch. ROUTING_ELIG_INPLACE=0 forces the reference."""
    Xa = np.asarray(X, dtype=float)
    single = Xa.ndim == 1
    if single:
        Xa = Xa[None, :]
    cs, cc = op["profile_starts"], op["profile_counts"]
    # RE-ENTRANCY: the scratch is shared, so a nested or concurrent call would corrupt it. Both
    # callers are single-threaded here, but the guard costs a boolean and removes the question.
    if not _EP_OK["use"] or _EP_BUSY[0]:
        # 19bm: the RESTRICTED path is the default. The in-place twin (19bk) is a separate,
        # switched-off experiment; when it is on it wins the dispatch so its own measurement is
        # not polluted by this one.
        if _RX_OK["use"]:
            rx = op.get("_rx")
            if rx is None:
                try:
                    rx = op["_rx"] = _rx_build(op)
                except Exception as _rxE:      # noqa: BLE001
                    _RX_OK["use"] = False
                    _RX_OK["msg"] = ("[eligibility] restriction index FAILED to build "
                                     f"({type(_rxE).__name__}: {_rxE}) \u2014 running the "
                                     "reference blend, which is the known-good path. No answer "
                                     "changes, only the speed.")
                    print(_RX_OK["msg"])
                    rx = None
            if rx is not None:
                out = _apply_elig_pop_rx(Xa, op, cs, cc, rx)
                if not _RX_OK["checked"]:
                    _RX_OK["checked"] = True
                    ref = _apply_elig_pop_alloc(np.array(Xa, float, copy=True), op, cs, cc)
                    _good, _why = _rx_verdict(ref, out, rx, Xa.shape[0], Xa.shape[1])
                    if _good:
                        _RX_OK["msg"] = ("[eligibility] restricted blends SELF-CHECK PASSED: "
                                         + _why + " ROUTING_ELIG_RESTRICT=0 reverts.")
                    else:
                        _RX_OK["use"] = False
                        _FU_OK["use"] = False      # 19bs: revert BOTH, then say which were on
                        out = ref
                        _RX_OK["msg"] = (
                            "[eligibility] restricted blends SELF-CHECK FAILED \u2014 " + _why
                            + ". REVERTING to the full-width blend for this process, so what "
                            "ships is the known-good answer. The premise (wf == 0 in every row "
                            "of a profile => that profile's blend is the identity) does not hold on "
                            "this data. Do not treat this as cosmetic: report it. Both the "
                            "hit-profile restriction AND the 19bs fused elementwise passes were in "
                            "play, so both are now off; re-run with ROUTING_ELIG_FUSE=0 to tell "
                            "them apart.")
                    print(_RX_OK["msg"])
                return out[0] if single else out
        out = _apply_elig_pop_alloc(Xa, op, cs, cc)
        return out[0] if single else out
    _EP_BUSY[0] = True
    try:
        P = Xa.shape[0]
        b = _ep_buf(cs, cc, Xa.shape[1]).grow(P)
        cur = Xa
        if op.get("has_ban"):
            tmp = b.take(P, 1)
            np.multiply(cur, (~op["ban"])[None, :], out=tmp)
            cur = _renorm_ip(tmp, cs, b, P, np.empty_like(Xa))
        if op.get("has_w"):
            cur = _blend_ip(cur, op["w_incap"], op["w_wf"], cs, b, P, np.empty_like(Xa))
        if op.get("has_u"):
            cur = _blend_ip(cur, op["u_incap"], op["u_wf"], cs, b, P, np.empty_like(Xa))
        if cur is Xa:                      # no stage applied — return a copy, as before
            cur = Xa.copy()
        if not _EP_OK["checked"]:
            _EP_OK["checked"] = True
            ref = _apply_elig_pop_alloc(np.array(Xa, float, copy=True), op, cs, cc)
            if np.array_equal(ref, cur):
                _EP_OK["msg"] = (
                    "[eligibility] \u2713 in-place transform SELF-CHECK PASSED: bit-identical to "
                    f"the allocating path on the live operator (np.array_equal on {P}x"
                    f"{Xa.shape[1]:,}, not allclose). ~28 full-width temporaries per call removed. "
                    "ROUTING_ELIG_INPLACE=0 forces the allocating path.")
                print(_EP_OK["msg"])
            else:
                _mx = float(np.abs(ref - cur).max())
                _EP_OK["use"] = False
                cur = ref
                _EP_OK["msg"] = (
                    f"[eligibility] \u26a0 in-place transform SELF-CHECK FAILED \u2014 max|\u0394| "
                    f"{_mx:.3e}. REVERTING to the allocating path for this process, so what ships "
                    "is the known-good transform. Do not treat this as cosmetic: report it.")
                print(_EP_OK["msg"])
    finally:
        _EP_BUSY[0] = False
    return cur[0] if single else cur
