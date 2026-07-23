"""Tab: Generate ConnectorPool JSON configs from the proposed split.

Extracted from streamlit_app.py (behaviour unchanged) as the first step of the
per-tab split. Call render(ss, PROJECT_ROOT) from inside `with tab_cfg:`."""
from __future__ import annotations

import os
from datetime import date

import pandas as pd
import streamlit as st

from impact_calcs import build_split_exports

__build__ = "2026-07-22-cfg-json-viewer-bin-filter"


def render(ss, PROJECT_ROOT):
    _variations = ss.get("variations")
    if not _variations:
        st.info("Compute split variations in tab 3 (Routing engine) first.")
    else:
        from routing_optimiser.connector_pool_configs import (
            BRANDS as _POOL_BRANDS, company_to_brand_key as _co2brand,
            generate_configs as _gen_cfgs)

        _fs_c = ss.get("forecast_settings", {}) or {}
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

        # Dial at the top-left, same placement / width / size as tab 3.
        _sldc, _sldsp = st.columns([0.9, 5.1])
        _prev_wc = ss.get("cfg_variation_sld", ss.get("selected_variation_weight"))
        _def_wc = _prev_wc if _prev_wc in _weights_c else _weights_c[len(_weights_c) // 2]
        picked_w_cfg = _sldc.select_slider(
            "**Risk  ↔  Conversion**", options=_weights_c, value=_def_wc,
            format_func=lambda w: f"{int(round(w * 100))}", key="cfg_variation_sld",
            help="Dial: safer routing ↔ more revenue.")

        mode = "sales"   # Mode input removed — always 'sales'.
        emit_generic = False   # pool-generic emit removed (full mode only); always compressed 'sales'.
        extra_priority = st.number_input("Extra priority boost", 0, 2_000_000, 0, step=50000,
                                         help="Added to every pool's priority (script's EXTRA_PRIORITY_AMOUNT).")

        # Two green buttons side by side, each the same width as tab 3's Export button
        # (weight 1.1 of a 6.0 row ≈ 18%).
        _btn1, _btn2, _btnsp = st.columns([1.1, 1.1, 3.8])
        _dl_slot = _btn2.container()
        if _btn1.button("Generate JSON configs", type="primary", key="gen_configs_btn",
                        use_container_width=True):
            try:
                _brand_name = _POOL_BRANDS[brand_key]["name"]
                _wc_c = ss.get("wallet_ctx") or {}
                # Resolve the chosen variation + rules source AT CLICK TIME (nothing ran
                # when the slider / selectbox changed).
                _chosen_c = _variations[_weights_c.index(picked_w_cfg)]
                _split_sel = _chosen_c["split"].copy()
                _src_is_comp = _maxN_cfg > 0   # always use the compressed (pool-targeted) rules
                _pool_stats_c = None
                if _src_is_comp:
                    from impact_calcs import pool_targeted_compression
                    # Signature MUST include mode + emit_generic (both change the pool count)
                    # so tab-6 results don't collide with tab-4's 'sales' cards.
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
                # Build the SAME per-Brand×RPGT templates the export produces, then run
                # the script-faithful ConnectorPool generator over them.
                _exports_c = build_split_exports(
                    _split_sel, _brand_name, str(_gl_c),
                    wallet_incapable=set(_wc_c.get("incapable", set())),
                    fid2vamp=_wc_c.get("fid2vamp"),
                    mid_list_path=os.path.join(PROJECT_ROOT, "data", "mappings", "Master_MID_List.csv"),
                    usa_only=set(_wc_c.get("usa_only", set())),
                    country_pres=_wc_c.get("country_pres", {}),
                    max_share=float(_wc_c.get("max_share", 0.97)))
                _pools, _counts = _gen_cfgs(
                    _exports_c, brand_key, date_tag, scheme="vi", mode=mode,
                    extra_priority_amount=int(extra_priority), emit_generic=bool(emit_generic))
                _src_lbl = "pool-targeted" if _src_is_comp else "ideal"
                _dial_lbl = int(round(picked_w_cfg * 100))
                ss["configs"] = _pools
                ss["configs_counts"] = _counts
                ss["configs_meta"] = {"brand_key": brand_key, "date": date_tag,
                                      "pool_dir": _counts.get("pool_dir", ""),
                                      "rules_source": _src_lbl, "variation": _dial_lbl}
                if not _pools:
                    st.warning("No pools generated — check the split has mapped gateways and "
                               "recognised RPGTs.")
                else:
                    if _src_is_comp:
                        _tgt_note = f" (target ≤ {_maxN_cfg:,}"
                        if _pool_stats_c and not _pool_stats_c.get("feasible", True):
                            _tgt_note += " — not reachable, this is the fewest possible"
                        _tgt_note += f"; from {_pool_stats_c.get('raw_pools', '?') if _pool_stats_c else '?'} ideal)"
                    else:
                        _tgt_note = ""
                    st.success(f"Generated {len(_pools)} ConnectorPool config(s) from "
                               f"**{_src_lbl}** rules at dial **{_dial_lbl}**{_tgt_note}.")
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
            if _pr:
                # HTML table styled like the rest of the app (red sticky header, card bg).
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
                # Hug-content table (grows to fit the 12px text) in a bounded left column.
                _pcol, _psp = st.columns([1, 3])
                _pcol.markdown(_tbl_html, unsafe_allow_html=True)
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
            # Rendered beside the Generate button (top), via the slot created above.
            _dl_slot.download_button(
                "⬇ Download configs (.zip)", _buf,
                file_name=f"ConnectorPool_configs_{_meta.get('brand_key', 'tav')}_{_meta.get('date', '')}.zip",
                mime="application/zip", type="primary", key="dl_configs_btn", use_container_width=True)
            # ---- Find & download a single config: filter by BIN / name / RPGT, view the JSON,
            #      download that one file. Live (not in a form) so the list updates as you type. ----
            st.markdown("**Find & download a config**")

            def _pool_bins(_pool):
                """All BINs referenced by a pool's card.bin matching-rule expressions."""
                _found = set()

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

            _f1, _f2, _f3 = st.columns([1, 1, 1])
            _bin_q = _f1.text_input("BIN contains", key="cfg_bin_filter",
                                    help="Show only configs whose card.bin matching rule includes this BIN "
                                         "(partial match, e.g. '470793' or just '4707').")
            _name_q = _f2.text_input("Name contains", key="cfg_name_filter",
                                     help="Filter by config file name (RPGT / currency / provider are in the name).")
            # Build the matching set.
            _bq = _bin_q.strip()
            _nq = _name_q.strip().lower()
            _matches = []
            for _nm, _pool in _pools.items():
                if _nq and _nq not in _nm.lower():
                    continue
                if _bq and not any(_bq in _b for _b in _pool_bins(_pool)):
                    continue
                _matches.append(_nm)
            st.caption(f"{len(_matches)} of {len(_pools)} configs match.")

            if _matches:
                _sel = st.selectbox("Config file", _matches, key="cfg_json_sel")
                _pool_sel = _pools[_sel]
                _bins_sel = sorted(_pool_bins(_pool_sel), key=lambda x: (len(x), x))
                if _bins_sel:
                    _shown = ", ".join(_bins_sel[:80]) + (" …" if len(_bins_sel) > 80 else "")
                    st.caption(f"BINs in this config ({len(_bins_sel)}): {_shown}")
                import json as _js3
                _one = _js3.dumps(_pool_sel, indent=2, ensure_ascii=False) + "\n"
                st.download_button("⬇ Download this config (.json)", _one, file_name=f"{_sel}.json",
                                   mime="application/json", type="primary", key="cfg_dl_one")
                st.json(_pool_sel)
            else:
                st.info("No configs match these filters." if (_bq or _nq)
                        else "Generate configs above, then search here.")
