"""Baseline & Validate → 'Config Validation' sub-tab.

Load a folder of generated ConnectorPool config JSONs and LOOK UP which config(s) would route a
given transaction profile (currency / BIN / country / provider / payment-scheme / RPGT / connector).
For every matching config it shows the linked filename and the raw JSON. Read-only; no pipeline.

Matching evaluates each pool's `selector.expressions` against the entered profile: a blank field
means "any", so a specific-BIN pool only matches that BIN while a catch-all (no BIN expression)
matches any BIN, etc. RPGT is identified by matching a pool's type-selectors against RPGT_MAP.
"""
from __future__ import annotations

import glob
import json
import os

import streamlit as st

try:
    from routing_optimiser.s1_extract.schema import RPGT_MAP
except Exception:  # noqa: BLE001 - schema should import, but never hard-fail the tab
    RPGT_MAP = {}

# Provider dropdown label -> the cell token used in the configs.
_PROVIDERS = {
    "(Any)": None,
    "ApplePay": "APPLEPAY",
    "GooglePay": "GOOGLEPAY",
    "Other (non-wallet)": "non_gp_ap",
}
_WALLET = {"PAYMENT_METHOD_PROVIDER_APPLEPAY", "PAYMENT_METHOD_PROVIDER_GOOGLEPAY"}


# [FN-430]
def _load_configs(folder):
    """Return [(path, pool_dict)] for every ConnectorPool JSON under `folder` (recursive).
    A file may hold one pool (dict) or a list of pools."""
    out = []
    for fp in sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)):
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001 - skip unreadable / non-JSON files
            continue
        for it in (data if isinstance(data, list) else [data]):
            if isinstance(it, dict) and it.get("kind") == "ConnectorPool":
                out.append((fp, it))
    return out


def _exprs(pool):
    return (pool.get("selector") or {}).get("expressions") or []


# [FN-431]
def _profile_of(pool):
    """Extract the profile dimensions a pool's selector CONSTRAINS (empty/None = unconstrained)."""
    cur, bins, country, provider, scheme = set(), set(), None, None, None
    for e in _exprs(pool):
        k, op = e.get("key"), e.get("operator")
        vals = [str(v) for v in (e.get("values") or [])]
        if k == "charge.amount.currency":
            cur |= set(vals)
        elif k == "method.info.card.bin":
            bins |= set(vals)
        elif k == "method.info.country":
            country = "USA" if (op == "Equal" and "US" in vals) else "Non-USA"
        elif k == "method.provider":
            up = set(vals)
            if op in ("NotIn", "NotEqual"):
                provider = "non_gp_ap"
            elif _WALLET <= up and len(up) >= 2:
                provider = "wallet"          # ApplePay OR GooglePay
            elif "PAYMENT_METHOD_PROVIDER_APPLEPAY" in up:
                provider = "APPLEPAY"
            elif "PAYMENT_METHOD_PROVIDER_GOOGLEPAY" in up:
                provider = "GOOGLEPAY"
        elif k == "method.paymentScheme":
            scheme = "visa" if (op == "Equal" and "card_visa" in vals) else "non-visa"
    return {"currency": cur, "bins": bins, "country": country, "provider": provider, "scheme": scheme}


def _rpgt_of(pool):
    """Identify a pool's RPGT by matching its expressions against the RPGT_MAP signatures."""
    present = {(e.get("key"), e.get("operator"), tuple(str(v) for v in (e.get("values") or [])))
               for e in _exprs(pool)}
    for rpgt, (_term, sels) in RPGT_MAP.items():
        sig = {(s["key"], s["operator"], tuple(str(v) for v in s["values"])) for s in sels}
        if sig and sig <= present:
            return rpgt
    return None


def _connectors_of(pool):
    return [c.get("connectorId") for c in ((pool.get("spec") or {}).get("connectors") or []) if c.get("connectorId")]


# [FN-432]
def _matches(pool, want):
    """True if `pool` would ROUTE the wanted profile. Only the fields set in `want` are checked."""
    P = _profile_of(pool)
    if want.get("currency") and P["currency"] and want["currency"] not in P["currency"]:
        return False
    if want.get("bin") and P["bins"] and want["bin"] not in P["bins"]:
        return False
    if want.get("country") and P["country"] and want["country"] != P["country"]:
        return False
    if want.get("scheme") and P["scheme"] and want["scheme"] != P["scheme"]:
        return False
    if want.get("provider") and P["provider"]:
        pv, wv = P["provider"], want["provider"]
        if pv == "wallet":
            if wv not in ("APPLEPAY", "GOOGLEPAY"):
                return False
        elif pv != wv:
            return False
    if want.get("rpgt"):
        r = _rpgt_of(pool)
        if r is not None and r != want["rpgt"]:
            return False
    if want.get("connector") and want["connector"] not in _connectors_of(pool):
        return False
    return True


# [FN-433b]
def render_profile_lookup(named_pools, key_prefix="cfgval_"):
    """Profile-lookup UI over `named_pools` (list of (filename, pool_dict)): profile filters → scatter
    charts of the matches → a dropdown + single JSON viewer. Widget keys are prefixed so it can render
    in more than one place (Config Validation and the '4 · Generate configs' tab)."""
    from app_common import render_config_profile_charts as _rcpc
    if not named_pools:
        st.markdown("<div style='font-size:12px; color:#0B1F3A;'>No configs to look up yet.</div>",
                    unsafe_allow_html=True)
        return
    _curs = sorted({c for _, p in named_pools for c in _profile_of(p)["currency"]})
    _conns = sorted({c for _, p in named_pools for c in _connectors_of(p)})
    _c1, _c2 = st.columns(2)
    _cur = _c1.selectbox("Currency", ["(Any)"] + _curs, key=key_prefix + "cur")
    _bin = (_c2.text_input("BIN", key=key_prefix + "bin",
                           help="Exact card BIN, e.g. 400609. Catch-all pools (no BIN rule) match any BIN.")
            or "").strip()
    _c3, _c4 = st.columns(2)
    _country = _c3.selectbox("Country", ["(Any)", "USA", "Non-USA"], key=key_prefix + "country")
    _provider = _c4.selectbox("Provider", list(_PROVIDERS.keys()), key=key_prefix + "provider")
    _c5, _c6 = st.columns(2)
    _scheme = _c5.selectbox("Payment scheme", ["(Any)", "Visa", "Non-Visa"], key=key_prefix + "scheme")
    _rpgt = _c6.selectbox("RPGT", ["(Any)"] + list(RPGT_MAP.keys()), key=key_prefix + "rpgt")
    _conn = st.selectbox("Connector", ["(Any)"] + _conns, key=key_prefix + "conn")
    want = {
        "currency": None if _cur == "(Any)" else _cur,
        "bin": _bin or None,
        "country": None if _country == "(Any)" else _country,
        "scheme": None if _scheme == "(Any)" else ("visa" if _scheme == "Visa" else "non-visa"),
        "provider": _PROVIDERS[_provider],
        "rpgt": None if _rpgt == "(Any)" else _rpgt,
        "connector": None if _conn == "(Any)" else _conn,
    }
    matches = [(fn, p) for fn, p in named_pools if _matches(p, want)]
    matches.sort(key=lambda t: (-int((t[1].get("selector") or {}).get("priority", 0) or 0),
                                0 if _profile_of(t[1])["bins"] else 1))
    # Charts sit below the inputs and reflect the CURRENT filter (the matching configs).
    _rcpc([(fn, p) for fn, p in matches])
    if not any(want.values()):
        st.info(f"Showing all **{len(matches)}** config(s). Set one or more cell fields to narrow the lookup.")
    else:
        st.markdown(f"**{len(matches)}** config(s) route this profile.")
    if not matches:
        st.warning("No config routes this exact cell. Try relaxing a field to (Any).")
        return
    _sel_fn = st.selectbox("Config filename", [fn for fn, _ in matches], key=key_prefix + "json_fname",
                           help="Pick one of the configs that route this profile to view its JSON.")
    _hit = next(((fn, p) for fn, p in matches if fn == _sel_fn), None)
    if _hit is None:
        return
    _fn, _p = _hit
    # JSON viewer text at 12px, in a fixed-height scroll window ≈ the two stacked charts' combined
    # height (2 × 300px); anything taller scrolls within the window. Tune the px to taste.
    st.markdown("<style>[data-testid=\"stJson\"], [data-testid=\"stJson\"] * "
                "{ font-size: 12px !important; }</style>", unsafe_allow_html=True)
    st.container(height=620).json(_p)


# [FN-433]
def render(ss, PROJECT_ROOT):
    from app_common import green_button_css

    _gen_col, _lookup_col = st.columns(2)

    # LEFT column: the full config generator (which contains the 'Exported rules folder' input).
    # A distinct key_prefix keeps its widgets/results independent of the '4 · Generate configs' tab;
    # its own per-config find panel is suppressed here (the cell lookup on the right replaces it).
    with _gen_col:
        st.markdown("##### Generate Configs")   # in-line with 'Look up configs by cell' (same h5)
        import tab_4_generate_configs
        tab_4_generate_configs.render(ss, PROJECT_ROOT, key_prefix="cv_", show_find=False)

    # RIGHT column: cell lookup over a loaded folder of configs — side by side with the generator.
    with _lookup_col:
        st.markdown("##### Look up configs by profile")
        _fc, _bc = st.columns([5, 1], vertical_alignment="bottom")
        folder = (_fc.text_input(
            "Configs folder", key="cfgval_folder",
            help="Folder of exported ConnectorPool .json configs (searched recursively). Each file "
                 "may hold one pool or a list of pools.") or "").strip()
        green_button_css("cfgval_reload")
        _reload = _bc.button("Load Configs", key="cfgval_reload")

        _ck = ss.get("_cfgval_cache") or {}
        if folder and (_reload or _ck.get("folder") != folder):
            # Resolve the path: use it as-is if it's a folder; otherwise try it relative to the project
            # root (also fixes a stray leading '/', e.g. '/data/...' which Python reads as an absolute
            # filesystem path, not a project-relative one).
            _resolved = folder
            _alt = os.path.join(PROJECT_ROOT, folder.lstrip("/\\"))   # project-relative fallback
            if not os.path.isdir(_resolved) and os.path.isdir(_alt):
                _resolved = _alt
            if not os.path.isdir(_resolved):
                st.error(f"Not a folder: {folder or '(empty)'} "
                         f"(also tried under the project root: {_alt})")
                return
            _ck = {"folder": folder, "resolved": _resolved, "pools": _load_configs(_resolved)}
            ss["_cfgval_cache"] = _ck

        # Source of configs to look up: a loaded folder wins; otherwise the configs just GENERATED on
        # the left (key_prefix 'cv_' → ss['cv_configs']) auto-populate here.
        if folder:
            _pools = _ck.get("pools") or []
            if not _pools:
                st.warning(f"No ConnectorPool .json configs found in: {folder}")
                return
            _named = [(os.path.basename(fp), p) for fp, p in _pools]
            st.caption(f"Loaded **{len(_named)}** config(s) from `{_ck.get('resolved', folder)}`.")
        elif ss.get("cv_configs"):
            _gen = ss["cv_configs"]
            _named = [(f"{_n}.json", _p) for _n, _p in _gen.items()]
            st.caption(f"Showing the **{len(_named)}** config(s) generated on the left.")
        else:
            st.markdown("<div style='font-size:12px; color:#0B1F3A;'>Enter a configs folder and click "
                        "<b>Load Configs</b>, or generate configs on the left.</div>",
                        unsafe_allow_html=True)
            return
        render_profile_lookup(_named, key_prefix="cfgval_")
