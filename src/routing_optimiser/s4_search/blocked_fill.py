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
