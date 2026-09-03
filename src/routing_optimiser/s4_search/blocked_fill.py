"""The blocked-row water-fill rule, and the arithmetic that prices it.

WHY THIS MODULE EXISTS
----------------------
Bank auto-block pins a failing (bank, gateway) row to the exploration floor (0.01) and hands
its freed share to the profile's other rows. The 0.97 max-share water-fill then runs, and every
implementation of it in this codebase picks recipients on share alone::

    recip = (share > 1e-12) & (~over) & (share < cap - 1e-12)

A row that was just pinned to the floor satisfies all three, so the water-fill lifts it straight
back off the floor - the engine routes MORE than the floor to a gateway it has just decided is
failing. 19ih traced this and found the LIVE engine (``impact_calcs._cap_rows``) and the search's
mirror (``tab_2._fm_cap``) agree on it, so it is faithful behaviour rather than a search bug.

THE RULE (Ben, 2026-09-03): a blocked gateway "needs to stay at 0.01, the only exception is if
the gateway in that profile is needed in order to keep other gateways under the max cap in that
same profile."

So blocked rows are recipients of LAST RESORT: non-blocked siblings absorb the excess first, and
a blocked row takes only the part that would otherwise leave some row above the cap. That
exception is not a nicety - without it a profile whose only under-cap rows are blocked would have
no legal way to satisfy the cap at all, and ``_cap_rows`` would leave a row above 0.97.

WHAT IS HERE AND WHAT IS NOT
----------------------------
``unavoidable_excess`` prices the rule: how much of what a water-fill put on blocked rows the cap
genuinely could not have held without them. ``split_room`` IS the rule, as a pure function over
the arrays every one of those water-fills already computes.

Nothing here is wired into a shipping path yet, on purpose. FIVE water-fills run after the
blocked flooring - ``_cap_rows`` (what ships), ``_fm_cap`` (the search), ``_max_share_waterfill``
(the VAMP forecast) and the band projector's own cap in two kernels plus its numpy reference -
and they have to agree or GA-fitness stops matching delivered, which is the failure class the
19i* series exists to close. ``_fm_cap`` calls ``unavoidable_excess`` to MEASURE; the behavioural
change lands once that measurement says what it is worth.

Every function takes arrays in the (P, R) / segment-start layout the callers already use:
``room``/``blocked`` are per (candidate, row); ``excess`` is per (candidate, profile);
``starts`` are the profile segment starts into the row axis.
"""
from __future__ import annotations

import numpy as np

__build__ = "2026-09-03-19ii-blocked-fill-rule"


def _seg(x, starts, axis=1):
    return np.add.reduceat(x, np.asarray(starts, np.intp), axis=axis)


def nonblocked_room(room, blocked, starts):
    """Room per (candidate, profile) held by recipients that are NOT bank-blocked.

    `blocked` may be (R,) - the usual case, since blocking is a property of the row, not of
    the candidate - or (P, R).
    """
    _r = np.asarray(room, float)
    _b = np.asarray(blocked, bool)
    if _b.ndim == 1:
        _b = _b[None, :]
    return _seg(np.where(_b, 0.0, _r), starts)


def unavoidable_excess(excess, room, blocked, starts):
    """The part of each profile's excess that CANNOT be placed without a blocked row.

    ``max(0, excess - room held by non-blocked recipients)``, never more than the excess
    itself. This is the rule's own exception, quantified: a caller that water-filled blocked
    rows freely can subtract this from what it actually put on them to get the AVOIDABLE part
    - the share a non-blocked sibling had room for and did not get.
    """
    _e = np.asarray(excess, float)
    _pnb = nonblocked_room(room, blocked, starts)
    return np.minimum(np.maximum(_e - _pnb, 0.0), np.maximum(_e, 0.0))


def split_room(room, blocked, excess, starts, counts):
    """THE RULE. Split the recipient room into a primary pool and a last-resort pool.

    Returns ``(room_primary, room_fallback, excess_primary, excess_fallback)``, all broadcast
    to the (candidate, row) or (candidate, profile) shape the caller needs:

      * ``room_primary``  - room on NON-blocked recipients (blocked rows zeroed).
      * ``room_fallback`` - room on BLOCKED recipients.
      * ``excess_primary``  - per (candidate, profile), the part of the excess the non-blocked
        recipients can absorb: ``min(excess, room_primary_total)``.
      * ``excess_fallback`` - the shortfall, which is what blocked rows may take. Zero whenever
        a non-blocked sibling could have taken it, which is the whole point.

    A caller water-fills ``room_primary`` with ``excess_primary`` and then, only if
    ``excess_fallback`` is non-zero for that profile, ``room_fallback`` with
    ``excess_fallback``. Both distributions stay proportional-to-room, so a profile with no
    blocked rows gets byte-identical behaviour to the unmodified water-fill: ``room_primary``
    is then the whole pool and ``excess_fallback`` is 0.
    """
    _r = np.asarray(room, float)
    _b = np.asarray(blocked, bool)
    if _b.ndim == 1:
        _b = _b[None, :]
    _e = np.asarray(excess, float)
    room_primary = np.where(_b, 0.0, _r)
    room_fallback = np.where(_b, _r, 0.0)
    _pool_p = _seg(room_primary, starts)
    excess_primary = np.minimum(_e, _pool_p)
    excess_fallback = np.maximum(_e - _pool_p, 0.0)
    return room_primary, room_fallback, excess_primary, excess_fallback


def waterfill_once(shares, cap, blocked, starts, counts):
    """One shed-then-fill sweep under THE RULE. Reference implementation, pure numpy.

    Shape (P, R) in, (P, R) out. Over-cap rows are shed to `cap`; the excess goes first to
    non-blocked recipients in proportion to their room, and only the part they cannot hold
    goes to blocked recipients. Recipients are rows with ``share > 1e-12`` and
    ``share < cap`` that are not themselves over - the same test every caller uses.

    With ``blocked`` all-False this is the unmodified water-fill, which is what makes it safe
    to check a caller against: the rule may only ever change profiles that contain a blocked
    row with room.
    """
    _X = np.asarray(shares, float).copy()
    _b = np.asarray(blocked, bool)
    if _b.ndim == 1:
        _b = _b[None, :]
    _o = _X > cap
    if not _o.any():
        return _X
    _exc = _seg(np.where(_o, _X - cap, 0.0), starts)
    _room = np.where((~_o) & (_X > 1e-12) & (_X < cap), cap - _X, 0.0)
    _rp, _rf, _ep, _ef = split_room(_room, _b, _exc, starts, counts)
    _cc = np.asarray(counts, np.intp)
    _out = np.where(_o, cap, _X)
    for _pool_rows, _take in ((_rp, _ep), (_rf, _ef)):
        _pool = _seg(_pool_rows, starts)
        _f = np.repeat(np.where(_pool > 1e-12, _take / np.where(_pool > 1e-12, _pool, 1.0), 0.0),
                       _cc, axis=1)
        _out = _out + _pool_rows * _f
    return _out


# ---------------------------------------------------------------------------
# THE CANONICAL KEY, and why there has to be one
# ---------------------------------------------------------------------------
# The rule has to hold at FIVE water-fills that run after the blocked flooring, or GA-fitness
# stops matching delivered. The obstacle is that they do not all carry the same row identity:
#
#   site                                          bank   gatewayFid   vampMid   currency
#   ------------------------------------------------------------------------------------
#   tab_2._fm_cap                (the search)      yes      YES         yes       yes
#   impact_calcs._cap_rows       (what SHIPS)      yes      YES         -         yes
#   impact_calcs._max_share_waterfill (forecast)   yes      no          YES       YES
#   band_projection kernels x2 + numpy reference   yes      no          YES       YES
#
# `detect_blocked_gateways` flags (bank, gatewayFid) pairs - the fid is what has the failing
# attempt history. Two sites cannot express a fid at all. Blocking the whole vampMid instead is
# not an option: measured on this book, 31 of 37 active TotalAV fids (84%) sit under a vampMid
# that carries more than one, and the multi-fid vampMids are split by CURRENCY -
# Adyen_TotalAV alone spans adyen-{aud,cad,eur,gbp,usd}-tav. Coarsening to vampMid would block
# five currencies because one was flagged, and would then DISAGREE with the two sites that can
# see the fid. That is the scored-vs-delivered divergence this rule is supposed to avoid.
#
# THE RESOLUTION, and it is the same one 19ht reached for the capability mask: (vampMid,
# currency) pins down a unique ACTIVE gatewayFid. Measured on Master_MID_List: 37 (vampMid,
# currency) groups over 37 active TotalAV fids, ZERO ambiguous. So
#
#       (bank, vampMid, currency)   ==   (bank, gatewayFid)
#
# is an identity on the active set, and it is expressible at every one of the five sites. Every
# site builds its mask from that key, so they agree by construction rather than by inspection.
#
# `equivalence_report` PROVES it per run rather than trusting the paragraph above - if a future
# mid list gives one (vampMid, currency) two active fids, the identity breaks and the run has to
# say so before anything ships on it.


def canonical_keys(blocked_pairs, mid_rows):
    """(bank, gatewayFid) pairs -> {(bank, vampMid_lower, currency_lower)}.

    `mid_rows` is an iterable of mappings with `gatewayFid`, `vampMid`, `currency` and
    `IsActive` (the Master_MID_List rows). Inactive rows are ignored: they cannot carry share.
    """
    _f2vc = {}
    for _r in mid_rows:
        if str(_r.get("IsActive", "")).strip().upper() != "TRUE":
            continue
        _f2vc[str(_r.get("gatewayFid", "")).strip().lower()] = (
            str(_r.get("vampMid", "")).strip().lower(),
            str(_r.get("currency", "")).strip().lower())
    _out = set()
    for _bk, _gw in blocked_pairs or ():
        _vc = _f2vc.get(str(_gw).strip().lower())
        if _vc is not None:
            _out.add((str(_bk).strip().lower(), _vc[0], _vc[1]))
    return _out


def equivalence_report(blocked_pairs, mid_rows, in_scope_fids=None):
    """Is (vampMid, currency) still an identity for gatewayFid on the fids THIS RUN can route to?

    `in_scope_fids` is the run's own gateway set. Pass it. Without it this reads the whole mid
    list, and the whole mid list is genuinely ambiguous: measured 2026-09-03, five (vampMid,
    currency) groups carry two or more active fids, and every one is a CROSS-BRAND collision -
    `adyen_totalsecurity/usd` is `adyen-usd-tsc-x-tav` (Total AV) beside `adyen-usd-tsc-x-tab`
    (Total Adblock); `paysafe - total av/eur` is `paysafe-eur-tav` beside `paysafe-eur-tvn`
    (Total VPN). A run is brand-scoped and drops the siblings ("77 other-brand vs 'TotalAV'"),
    so within a run the identity holds - which is the same conclusion 19ht reached for the
    capability mask ("0 among ACTIVE fids"). Scoping to the run's own fids is therefore not a
    convenience, it is what makes the answer true; an unscoped call is reported as such.

    Returns a dict the caller logs. `ambiguous` lists the (vampMid, currency) groups resolving
    to more than one in-scope fid - each is a place the two coarse sites cannot reproduce what
    the two fine sites do. `ambiguous_hit` is the subset a BLOCKED pair actually lands on, and
    that is the one that gates arming: an ambiguous group nothing is blocked in costs nothing.
    """
    _scope = None if in_scope_fids is None else {
        str(_f).strip().lower() for _f in in_scope_fids}
    _groups = {}
    for _r in mid_rows:
        if str(_r.get("IsActive", "")).strip().upper() != "TRUE":
            continue
        _fid = str(_r.get("gatewayFid", "")).strip().lower()
        if _scope is not None and _fid not in _scope:
            continue
        _k = (str(_r.get("vampMid", "")).strip().lower(),
              str(_r.get("currency", "")).strip().lower())
        _groups.setdefault(_k, set()).add(_fid)
    _amb = sorted(k for k, v in _groups.items() if len(v) > 1)
    _keys = canonical_keys(blocked_pairs, mid_rows)
    _hit = sorted(k[1:] for k in _keys if k[1:] in set(_amb))
    _pairs = {(str(b).strip().lower(), str(g).strip().lower())
              for b, g in (blocked_pairs or ())}
    _known = {_f for _g in _groups.values() for _f in _g}
    _unmapped = sorted(p for p in _pairs if p[1] not in _known)
    return {"groups": len(_groups), "ambiguous": _amb, "ambiguous_hit": _hit,
            "keys": _keys, "n_pairs": len(_pairs), "unmapped": _unmapped,
            "scoped": _scope is not None, "n_scope": (0 if _scope is None else len(_scope)),
            "safe": (not _hit) and (not _unmapped)}


# ---------------------------------------------------------------------------
# ARMING: partial is worse than off
# ---------------------------------------------------------------------------
# A rule applied at four of five water-fills is not "mostly done" - it is a guaranteed
# scored-vs-delivered divergence on exactly the rows it touches, which is harder to find than
# the behaviour it replaced. So arming is gated on every site REGISTERING itself, and the gate
# lives here rather than in any one caller.
SITES = ("_fm_cap", "_cap_rows", "_max_share_waterfill", "band_kernel_profile",
         "band_kernel_flat")
_WIRED = set()


def register(site):
    """A site calls this once when it has the mask and applies the rule."""
    if site not in SITES:
        raise ValueError(f"blocked_fill.register: unknown site {site!r}; expected one of {SITES}")
    _WIRED.add(site)


def wired():
    return sorted(_WIRED)


def missing():
    return [s for s in SITES if s not in _WIRED]


def arming_verdict(requested):
    """(armed, message). `armed` is True only when EVERY site is wired.

    A request that cannot be honoured comes back False with the reason, so the caller logs a
    refusal and runs the old behaviour - never half of the new one.
    """
    if not requested:
        return False, ("[blk-fill] rule OFF (ROUTING_BLOCK_NOFILL unset). Blocked rows are "
                       "water-fill recipients like any other, which is what every build before "
                       "19ij did; [blk-fill] prices what that costs.")
    _miss = missing()
    if _miss:
        return False, ("[blk-fill] ⚠ RULE REQUESTED AND REFUSED: "
                       f"{len(_miss)} of {len(SITES)} water-fill(s) are not wired for it "
                       f"({', '.join(_miss)}). Arming a subset would put the rule on some "
                       "stages and not others, so GA-fitness would stop matching delivered on "
                       "exactly the rows the rule touches - a worse failure than leaving it "
                       "off. Running the OLD behaviour on every stage instead. Wired: "
                       + (", ".join(wired()) or "none"))
    return True, ("[blk-fill] RULE ON at all "
                  f"{len(SITES)} water-fill(s) ({', '.join(wired())}): a bank-blocked gateway "
                  "stays at the exploration floor, and receives water-fill ONLY where its "
                  "profile has no other under-cap recipient with room - i.e. only where the "
                  "0.97 cap could not otherwise hold. ROUTING_BLOCK_NOFILL=0 reverts.")
