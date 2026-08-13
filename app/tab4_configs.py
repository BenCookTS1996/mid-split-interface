"""Tab: Generate ConnectorPool JSON configs from the proposed split.

This is the LAST-MILE tab: it turns the chosen split variation into the actual
ConnectorPool JSON files production deploys — optionally COMPRESSED to a target
number of pools first (fewer, shared rules instead of one bespoke rule per cell),
then zipped for download, with a search box to inspect a single config.

ANALOGY: the split is the "recipe"; this tab prints the "shopping list" the kitchen
(production) actually follows. Call render(ss, PROJECT_ROOT) from inside `with tab_cfg:`.

Originally split out of streamlit_app.py (since evolved) as the first step of the
per-tab split."""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from impact_calcs import build_split_exports

__build__ = "2026-07-29-cfg-json-viewer-bin-filter+single-variation-dial-guard"


# [FN-386]
def render(ss, PROJECT_ROOT):
    """Render the config-generation tab.

    Flow: if no split has been computed yet, show a placeholder. Otherwise pick the split
    variation (brand + go-live come from earlier tabs), optionally compress it to the pool
    target, build the per-Brand×RPGT templates, run the ConnectorPool generator, and offer the
    results as a zip plus a per-config search/download. `ss` is Streamlit's session_state.
    """
    _variations = ss.get("variations")
    if not _variations:
        from app_common import _locked_panel
        _locked_panel("Head to <b>2 · Routing engine</b> and click <b>Compute split "
                      "variations</b> — you can generate connector configs here once a split exists.")
    else:
        from routing_optimiser.connector_pool_configs import (
            BRANDS as _POOL_BRANDS, company_to_brand_key as _co2brand,
            generate_configs as _gen_cfgs, scheme_code as _scheme_code)

        _fs_c = ss.get("forecast_settings", {}) or {}
        # 'Active' scheme = the one you most recently worked with: the Validate Split selection if set,
        # else the Build Baseline scheme. Drives the rules-folder default below; the actual config
        # scheme_filter is RE-DERIVED at generate time from the folder you point at, so a Mastercard
        # split emits Mastercard configs even when the baseline was Visa.
        _active_scheme = str(ss.get("validate_card_scheme") or _fs_c.get("card_scheme", "visa") or "visa").strip().lower()
        # Rules folder defaults to that scheme's subfolder (data/exported_rules/<scheme>). FIRST-RUN
        # default only (setdefault): a programmatic session_state write on every rerun would make the
        # top-level st.tabs lose the active tab. If you switch scheme later, just edit the folder path.
        ss.setdefault("cfg_rules_folder", os.path.join("data", "exported_rules", _active_scheme))
        _company_c = str(_fs_c.get("company", "TotalAV"))
        _def_brand = _co2brand(_company_c)
        _gl_c = ss.get("split_go_live_date", date.today())
        try:
            _def_date = pd.to_datetime(str(_gl_c)).strftime("%y%m%d")
        except Exception:
            _def_date = date.today().strftime("%y%m%d")

        # Brand comes from the forecast (tab 1) and the date from the Split Go Live date (tab 2),
        # so those inputs are removed. Configs always use the COMPRESSED (pool-targeted) rules.
        brand_key = _def_brand if _def_brand in _POOL_BRANDS else list(_POOL_BRANDS.keys())[0]
        date_tag = _def_date
        _weights_c = [v["weight"] for v in _variations]
        _maxN_cfg = int(ss.get("max_configs", 0) or 0)

        # Green styling for the form's submit button (form submits use their own button kind).
        st.markdown("""<style>
            div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
                background:#22C36B !important; border-color:#22C36B !important; border-radius:0 !important; }
            div[data-testid="stForm"] button[kind="primaryFormSubmit"] * { color:#fff !important; }
        </style>""", unsafe_allow_html=True)

        # --- Generation settings live in a FORM: changing the dial / source / folder / priority does
        #     NOT re-run the tab — they apply only when the green 'Generate JSON configs' submit is
        #     clicked. (The Download button and the live-search Find panel can't sit in a form, so
        #     they render below it.) ---
        _prev_wc = ss.get("cfg_variation_sld", ss.get("selected_variation_weight"))
        _def_wc = _prev_wc if _prev_wc in _weights_c else _weights_c[len(_weights_c) // 2]
        with st.form("cfg_gen_form", border=False):
            # Dial (narrow, top-left).
            _sldc, _sldsp = st.columns([0.9, 5.1])
            if len(_weights_c) > 1:
                picked_w_cfg = _sldc.select_slider(
                    "**Risk  ↔  Conversion**", options=_weights_c, value=_def_wc,
                    format_func=lambda w: f"{int(round(w * 100))}", key="cfg_variation_sld",
                    help="Dial: safer routing ↔ more revenue.")
            else:
                # Single variation: a select_slider over ONE option throws a RangeError.
                picked_w_cfg = _weights_c[0]
            # Configs are ALWAYS generated directly from the exported rules folder (reads the rule
            # sheets as-is so per-provider / per-country splits are preserved). The toggle for this
            # was removed — it was always on.
            _from_folder = True
            # Folder input. Width ≈ the Generate button (1.5 of 6.0 ≈ 25%).
            _erfc, _erfsp = st.columns([1.5, 4.5])
            _cfg_folder = _erfc.text_input(
                "Exported rules folder", key="cfg_rules_folder",
                help="Folder of exported rule sheets (.xlsx / .csv), the same ones used in Validate "
                     "Split. Defaults to data/exported_rules/<scheme>.")
            # Extra priority boost (18% width).
            _pbc, _pbsp = st.columns([1.1, 4.9])
            extra_priority = _pbc.number_input(
                "Extra priority boost", 0, 2_000_000, 0, step=50000, key="cfg_extra_priority",
                help="Added to every pool's priority (script's EXTRA_PRIORITY_AMOUNT).")
            # Green submit — applies all the settings above (width ≈ 25%).
            _gbc, _gbsp = st.columns([1.5, 4.5])
            _do_generate = _gbc.form_submit_button(
                "Generate JSON configs", type="primary", use_container_width=True)

        mode = "sales"        # Mode input removed — always 'sales'.
        emit_generic = False  # pool-generic emit removed (full mode only); always compressed 'sales'.
        # Find & download panel is FILLED at the bottom (after generation, so it sees fresh configs)
        # but POSITIONED here — just below the form — via this reserved slot.
        _find_slot = st.container()

        if _do_generate:
            try:
                _brand_name = _POOL_BRANDS[brand_key]["name"]
                _wc_c = ss.get("wallet_ctx") or {}
                _pool_stats_c = None
                _exports_c = None
                _gen_mode = mode
                if _from_folder:
                    # ---- Faithful path: read the exported rule sheets AS-IS (like the Colab
                    #      generator), grouped by RPGT, so per-provider / per-country splits are
                    #      kept. 'full' mode emits BIN-specific + catch-all pools. ----
                    import glob as _glob
                    _files = sorted(_glob.glob(os.path.join(_cfg_folder, "*.xlsx"))
                                    + _glob.glob(os.path.join(_cfg_folder, "*.xls"))
                                    + _glob.glob(os.path.join(_cfg_folder, "*.csv")))
                    if not _files:
                        st.error(f"No rule files (.xlsx/.csv) found in: {_cfg_folder or '(empty)'}")
                    else:
                        _frames = []
                        for _f in _files:
                            try:
                                _frames.append(pd.read_excel(_f) if _f.lower().endswith((".xlsx", ".xls"))
                                               else pd.read_csv(_f))
                            except Exception:  # noqa: BLE001
                                continue
                        _all = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
                        _exports_c = {}
                        if "RPGT" in _all.columns:
                            for _rp, _sub in _all.groupby("RPGT"):
                                _exports_c[(_brand_name, str(_rp))] = _sub.reset_index(drop=True)
                        _gen_mode = "sales"   # BIN-specific pools ONLY — never the catch-all backups
                        _src_lbl = "exported rules folder"
                        _dial_lbl = None
                else:
                    # ---- Engine-split path: pick the variation, optionally compress to the pool
                    #      target, then build the per-Brand×RPGT templates. ----
                    _chosen_c = _variations[_weights_c.index(picked_w_cfg)]
                    _split_sel = _chosen_c["split"].copy()
                    _src_is_comp = _maxN_cfg > 0   # always use the compressed (pool-targeted) rules
                    if _src_is_comp:
                        from impact_calcs import pool_targeted_compression
                        _sig_c = (float(picked_w_cfg), _maxN_cfg, ss.get("variations_engine"),
                                  brand_key, str(_gl_c), str(mode), bool(emit_generic),
                                  round(float(_wc_c.get("max_share", 0.97)), 4))
                        with st.spinner("Trimming the split to your pool target…"):
                            _split_sel, _pool_stats_c = pool_targeted_compression(
                                ss, _split_sel, target_pools=_maxN_cfg, sig=_sig_c,
                                wallet_ctx=_wc_c, brand_name=_brand_name, brand_key=brand_key,
                                go_live=str(_gl_c),
                                mid_list_path=os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                                date_tag=date_tag, mode=mode, emit_generic=bool(emit_generic))
                    _exports_c = build_split_exports(
                        _split_sel, _brand_name, str(_gl_c),
                        wallet_incapable=set(_wc_c.get("incapable", set())),
                        fid2vamp=_wc_c.get("fid2vamp"),
                        mid_list_path=os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                        usa_only=set(_wc_c.get("usa_only", set())),
                        country_pres=_wc_c.get("country_pres", {}),
                        max_share=float(_wc_c.get("max_share", 0.97)))
                    _src_lbl = "pool-targeted" if _src_is_comp else "ideal"
                    _dial_lbl = int(round(picked_w_cfg * 100))

                if _exports_c is not None:
                    # scheme_filter FOLLOWS the rules being generated: from-folder → infer from the folder
                    # name (data/exported_rules/<scheme>); otherwise the active scheme. This is what makes
                    # a Mastercard split emit Mastercard (non-vi) configs instead of Visa.
                    _seg = os.path.basename(os.path.normpath(str(_cfg_folder or ""))).lower()
                    _gen_scheme_name = _seg if (_from_folder and _seg in ("visa", "mastercard")) else _active_scheme
                    _cfg_scheme = _scheme_code(_gen_scheme_name)
                    _pools, _counts = _gen_cfgs(
                        _exports_c, brand_key, date_tag, scheme=_cfg_scheme, mode=_gen_mode,
                        extra_priority_amount=int(extra_priority), emit_generic=bool(emit_generic))
                    ss["configs"] = _pools
                    ss["configs_counts"] = _counts
                    ss["configs_meta"] = {"brand_key": brand_key, "date": date_tag,
                                          "pool_dir": _counts.get("pool_dir", ""),
                                          "rules_source": _src_lbl, "variation": _dial_lbl,
                                          "scheme": _gen_scheme_name, "scheme_filter": _cfg_scheme}
                    st.caption(f"Generated as **{_gen_scheme_name}** (scheme filter `{_cfg_scheme}`) "
                               f"from `{_cfg_folder if _from_folder else _src_lbl}`.")
                    if not _pools:
                        st.warning("No pools generated — check the rules have mapped gateways and "
                                   "recognised RPGTs.")
                    else:
                        if _from_folder:
                            _note = f"Generated {len(_pools)} ConnectorPool config(s) from the **{_src_lbl}**."
                        else:
                            _tgt_note = ""
                            if _maxN_cfg > 0:
                                _tgt_note = f" (target ≤ {_maxN_cfg:,}"
                                if _pool_stats_c and not _pool_stats_c.get("feasible", True):
                                    _tgt_note += " — not reachable, this is the fewest possible"
                                _tgt_note += f"; from {_pool_stats_c.get('raw_pools', '?') if _pool_stats_c else '?'} ideal)"
                            _note = (f"Generated {len(_pools)} ConnectorPool config(s) from "
                                     f"**{_src_lbl}** rules at dial **{_dial_lbl}**{_tgt_note}.")
                        _sbc, _sbsp = st.columns([1.5, 4.5])   # match the Download button's width
                        _sbc.success(_note)
                        if _counts.get("skipped_rpgts"):
                            st.warning("Skipped unrecognised RPGT(s): " + ", ".join(_counts["skipped_rpgts"]))
            except Exception as _ce:  # noqa: BLE001
                import traceback as _ctb
                st.error(f"Config generation failed: {type(_ce).__name__}: {_ce}")
                with st.expander("Traceback"):
                    st.code(_ctb.format_exc())

        if ss.get("configs"):
            _pools = ss["configs"]
            _counts = ss.get("configs_counts", {})
            _meta = ss.get("configs_meta", {})
            _pr = _counts.get("per_rpgt", {})
            # Download configs (.zip) — outside the form (download buttons can't live in a form).
            import io as _io2
            import json as _json2
            import zipfile as _zip2
            _pool_dir = _meta.get("pool_dir") or _POOL_BRANDS[_meta.get("brand_key", "tav")]["pool_dir"]
            _buf = _io2.BytesIO()
            with _zip2.ZipFile(_buf, "w", _zip2.ZIP_DEFLATED) as z:
                for _name, _pool in _pools.items():
                    z.writestr(f"{_pool_dir}/{_name}.json",
                               _json2.dumps(_pool, indent=2, ensure_ascii=False) + "\n")
            _buf.seek(0)
            _dbc, _dbsp = st.columns([1.5, 4.5])   # same width as the Generate submit
            _dbc.download_button(
                "⬇ Download configs (.zip)", _buf,
                file_name=f"ConnectorPool_configs_{_meta.get('brand_key', 'tav')}_{_meta.get('date', '')}.zip",
                mime="application/zip", type="primary", key="dl_configs_btn", use_container_width=True)
            # RPGT Pools table (just below the Download button).
            if _pr:
                _rows_html, _tot = [], 0
                for _k, _v in _pr.items():
                    _tot += int(_v)
                    _rows_html.append(
                        f'<tr><td style="padding:3px 8px; text-align:left; white-space:nowrap; '
                        f'color:var(--tav-ink); border-bottom:1px solid var(--tav-line);">{_k}</td>'
                        f'<td style="padding:3px 8px; text-align:right; '
                        f'color:var(--tav-ink); border-bottom:1px solid var(--tav-line);">{int(_v):,}</td></tr>')
                _rows_html.append(
                    f'<tr><td style="padding:3px 8px; text-align:left; font-weight:bold; color:var(--tav-ink);">TOTAL</td>'
                    f'<td style="padding:3px 8px; text-align:right; font-weight:bold; color:var(--tav-ink);">{_tot:,}</td></tr>')
                _tbl_html = (
                    '<div style="display:inline-block; box-shadow:0 4px 12px rgba(0,0,0,0.08); '
                    'border-radius:0; overflow:auto; max-height:320px; background-color:var(--tav-card); '
                    'border:1px solid var(--tav-line);">'
                    '<table style="width:auto; border-collapse:collapse; font-family:inherit; font-size:12px; '
                    'line-height:1.15;"><tr>'
                    '<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; font-size:12px; '
                    'padding:3px 8px; text-align:left; position:sticky; top:0;">RPGT</th>'
                    '<th style="background-color:var(--tav-red); color:#FFF; font-weight:bold; font-size:12px; '
                    'padding:3px 8px; text-align:right; position:sticky; top:0;">Pools</th>'
                    '</tr>' + "".join(_rows_html) + '</table></div>')
                st.markdown(_tbl_html, unsafe_allow_html=True)

        # ---- Find & download a single config (outside the form so its BIN / name filters update
        #      live as you type). Rendered into the slot just below the form (reserved above). ----
        with _find_slot:
            _pools_f = ss.get("configs") or {}
            _, _hc = st.columns([3, 3])   # header aligned with the "BIN contains" column (right half)
            _hc.markdown("**Find & download a config**")

            # [FN-387]
            def _pool_bins(_pool):
                """All BINs referenced by a pool's card.bin matching-rule expressions."""
                _found = set()

                # [FN-388]
                def _walk(o):
                    if isinstance(o, dict):
                        if o.get("key") == "method.info.card.bin":
                            for v in (o.get("values") or []):
                                _found.add(str(v).strip())
                        for v in o.values():
                            _walk(v)
                    elif isinstance(o, list):
                        for v in o:
                            _walk(v)
                _walk(_pool)
                return _found

            # Inputs at half width, pushed to the RIGHT: a 3-unit leading spacer + three 1-unit columns.
            _, _f1, _f2, _f3 = st.columns([3, 1, 1, 1])
            _bin_q = _f1.text_input("BIN contains", key="cfg_bin_filter",
                                    help="Show only configs whose card.bin matching rule includes this BIN "
                                         "(partial match, e.g. '470793' or just '4707').")
            _name_q = _f2.text_input("Name contains", key="cfg_name_filter",
                                     help="Filter by config file name (RPGT / currency / provider are in the name).")
            if _pools_f:
                _bq = _bin_q.strip()
                _nq = _name_q.strip().lower()
                _matches = []
                for _nm, _pool in _pools_f.items():
                    if _nq and _nq not in _nm.lower():
                        continue
                    if _bq and not any(_bq in _b for _b in _pool_bins(_pool)):
                        continue
                    _matches.append(_nm)
                _, _mc = st.columns([1, 1])   # caption at half width (right half)
                _mc.caption(f"{len(_matches)} of {len(_pools_f)} configs match.")
                if _matches:
                    # Dropdown sits in the third column, to the right of "Name contains".
                    _sel = _f3.selectbox("Config file", _matches, key="cfg_json_sel")
                    _pool_sel = _pools_f[_sel]
                    _bins_sel = sorted(_pool_bins(_pool_sel), key=lambda x: (len(x), x))
                    if _bins_sel:
                        _shown = ", ".join(_bins_sel[:80]) + (" …" if len(_bins_sel) > 80 else "")
                        _, _bc = st.columns([1, 1])   # caption at half width (right half)
                        _bc.caption(f"BINs in this config ({len(_bins_sel)}): {_shown}")
                    import json as _js3
                    _one = _js3.dumps(_pool_sel, indent=2, ensure_ascii=False) + "\n"
                    # Download button aligned with the "BIN contains" column (right half).
                    _, _dc = st.columns([3, 3])
                    _dc.download_button("⬇ Download this config (.json)", _one, file_name=f"{_sel}.json",
                                        mime="application/json", type="primary", key="cfg_dl_one")
                    # JSON at half width, pushed to the RIGHT (left half is a spacer).
                    _, _jc = st.columns([1, 1])
                    _jc.json(_pool_sel)
                else:
                    st.info("No configs match these filters.")
