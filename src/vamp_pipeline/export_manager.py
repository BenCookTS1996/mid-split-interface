import os
import gc
import re
import calendar
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from .utils import setup_logger

logger = setup_logger(__name__)

__build__ = "2026-07-19-prorata-export-fcp1-per-subcell"


class ExportManager:
    """
    Handles memory-safe chunked exporting of the massive VAMP matrix, 
    MID mapping, and generation of the final FP&A summary reports.
    """

    def __init__(self, config: Dict[str, Any], mid_df: pd.DataFrame, attempts_df: pd.DataFrame,
                 mr_weights: Dict[Any, Any] = None):
        self.config = config
        self.company = str(self.config['run_settings']['company']).strip()
        self.month_var = str(self.config['run_settings']['month_var']).strip()

        # Monthly-renewal daily weights + go-live date drive the ADDITIVE pro-rata
        # export only. They do not touch any forecasting/allocation logic.
        self.mr_weights = mr_weights or {}
        self._month_0 = pd.to_datetime(self.config['run_settings'].get('month_0_start_date'), errors='coerce')
        self._go_live = pd.to_datetime(self.config['run_settings'].get('split_go_live_date'), errors='coerce')

        # We need these from the DataExtractor to build the historical summaries
        self.mid_df = mid_df.copy()
        self.attempts_df = attempts_df.copy()
        
        # 🟢 FIX: Dynamically construct the output directory based on Run Settings
        self.output_dir = self.config['paths'].get('output_dir', 'data/outputs/').format(month_var=self.month_var, company=self.company)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.main_export_file = os.path.join(self.output_dir, 'vamp_t_period_export.csv')

    # =========================================================================
    # === 1. MID MAPPING & MEMORY COMPRESSION
    # =========================================================================

    def _map_mids_and_compress(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Maps raw gateways to final Accounting MIDs and compresses text columns to save RAM."""
        self.mid_df.columns = [str(c).strip().lower().replace(" ", "") for c in self.mid_df.columns]
        mid_map = pd.Series(self.mid_df['vampmid'].values, index=self.mid_df['gatewayfid'].astype(str).str.lower().str.strip()).to_dict()

        post_df['finalGateway'] = post_df['finalGateway'].astype(str).str.lower().str.strip()
        pre_df['finalGateway'] = pre_df['finalGateway'].astype(str).str.lower().str.strip()

        post_df['vampMid'] = post_df['finalGateway'].map(mid_map).fillna('Unmapped')
        pre_df['vampMid'] = pre_df['finalGateway'].map(mid_map).fillna('Unmapped')

        unmapped_gws = set(post_df.loc[post_df['vampMid'] == 'Unmapped', 'finalGateway']).union(set(pre_df.loc[pre_df['vampMid'] == 'Unmapped', 'finalGateway']))
        if unmapped_gws:
            logger.warning(f"Found {len(unmapped_gws)} raw gateways missing from the 'MIDList' sheet!")
            for gw in sorted(unmapped_gws): 
                logger.warning(f"   - {gw}")

        post_df.drop(columns=['finalGateway'], inplace=True)
        pre_df.drop(columns=['finalGateway'], inplace=True)

        grp_keys = ['Company', 'vampMid', 'RPGT', 'BIN', 'Currency']

        # ADDITIVE: carry paymentMethodProvider through the matrix so the pro-rata
        # export can distinguish wallet (GOOGLEPAY/APPLEPAY) from non-wallet
        # traffic. Only added when the pipeline data actually has the column - it
        # merely disaggregates rows; the VAMP totals are unchanged.
        _pmp_names = ('paymentMethodProvider', 'payment_method_provider', 'paymentmethodprovider')
        def _find_pmp(_df):
            for _c in _pmp_names:
                if _c in _df.columns:
                    return _c
            return None
        pre_pmp, post_pmp = _find_pmp(pre_df), _find_pmp(post_df)
        if pre_pmp or post_pmp:
            if post_pmp and post_pmp != 'paymentMethodProvider':
                post_df.rename(columns={post_pmp: 'paymentMethodProvider'}, inplace=True)
            if pre_pmp and pre_pmp != 'paymentMethodProvider':
                pre_df.rename(columns={pre_pmp: 'paymentMethodProvider'}, inplace=True)
            for _df in (post_df, pre_df):
                if 'paymentMethodProvider' not in _df.columns:
                    _df['paymentMethodProvider'] = 'unknown'
                _df['paymentMethodProvider'] = _df['paymentMethodProvider'].astype(str).fillna('unknown')
            grp_keys = grp_keys + ['paymentMethodProvider']
            logger.info("   > paymentMethodProvider found — carried through the matrix (wallet-aware pro-rata).")
        else:
            logger.warning("   > paymentMethodProvider NOT present in pipeline data — pro-rata export will omit it.")

        # ADDITIVE: carry Country through the matrix too, so the pro-rata export can
        # distinguish USA vs Non-USA baseline volume (USA-only gateways serve USA rows
        # only). Same mechanism as pmp — only disaggregates rows, totals unchanged.
        _ctry_names = ('Country', 'country')
        def _find_ctry(_df):
            for _c in _ctry_names:
                if _c in _df.columns:
                    return _c
            return None
        pre_ctry, post_ctry = _find_ctry(pre_df), _find_ctry(post_df)
        if pre_ctry or post_ctry:
            if post_ctry and post_ctry != 'Country':
                post_df.rename(columns={post_ctry: 'Country'}, inplace=True)
            if pre_ctry and pre_ctry != 'Country':
                pre_df.rename(columns={pre_ctry: 'Country'}, inplace=True)
            for _df in (post_df, pre_df):
                if 'Country' not in _df.columns:
                    _df['Country'] = 'unknown'
                _df['Country'] = _df['Country'].astype(str).fillna('unknown')
            grp_keys = grp_keys + ['Country']
            logger.info("   > Country found — carried through the matrix (USA/Non-USA-aware pro-rata).")
        else:
            logger.warning("   > Country NOT present in pipeline data — pro-rata export will omit it.")

        for c in grp_keys:
            if c in post_df.columns: post_df[c] = post_df[c].astype('category')
            if c in pre_df.columns: pre_df[c] = pre_df[c].astype('category')

        return pre_df, post_df, grp_keys

    # =========================================================================
    # === 2. CHUNKED MATRIX EXPORT (RAM SAFE)
    # =========================================================================

    def _extract_sparse_matrix_slice(self, df: pd.DataFrame, v_col: str, vi_col: str, t: int, grp_keys: List[str], is_pre: bool):
        """Pops columns from the dataframe to extract active volume, destroying them to save RAM."""
        if v_col not in df.columns:
            return None

        v_vals = df.pop(v_col).values.astype(np.float32)
        vi_vals = df.pop(vi_col).values.astype(np.float32) if t == 0 and vi_col in df.columns else np.zeros(len(df), dtype=np.float32)

        mask = (v_vals != 0) | (vi_vals != 0)
        if not mask.any():
            return None

        slice_df = df.loc[mask, grp_keys].copy()

        if is_pre:
            slice_df['VAMP_Pre'], slice_df['VI_Txn_Pre'] = v_vals[mask], vi_vals[mask]
            slice_df['VAMP_Post'], slice_df['VI_Txn_Post'] = 0.0, 0.0
        else:
            slice_df['VAMP_Post'], slice_df['VI_Txn_Post'] = v_vals[mask], vi_vals[mask]
            slice_df['VAMP_Pre'], slice_df['VI_Txn_Pre'] = 0.0, 0.0

        return slice_df

    def _assemble_and_format_chunk(self, chunks: list, grp_keys: List[str], m: int, t: int) -> pd.DataFrame:
        """Merges matrix slices and drops zero-volume profiles."""
        chunk = pd.concat(chunks, ignore_index=True, copy=False).groupby(grp_keys, as_index=False, observed=True).sum()
        chunk = chunk[(chunk['VAMP_Pre'] > 0) | (chunk['VAMP_Post'] > 0) | (chunk['VI_Txn_Pre'] > 0) | (chunk['VI_Txn_Post'] > 0)].copy()

        if not chunk.empty:
            chunk['period'], chunk['t'] = m, t
            return chunk
        return None

    def export_chunked_vamp_matrix(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> int:
        """MANAGER: Orchestrates the 'Pop & Stack' export of the massive VAMP matrix."""
        logger.info("   > Assigning MIDs and compressing text keys...")
        pre_df, post_df, grp_keys = self._map_mids_and_compress(pre_df, post_df)

        logger.info(f"   > Exporting to CSV using Pop & Stack compression: {self.main_export_file}")
        header, row_count = True, 0

        with open(self.main_export_file, 'w') as f:
            for m in range(6):
                for t in range(10):
                    chunks = []

                    post_slice = self._extract_sparse_matrix_slice(post_df, f'Reallocated_t{t}_fcast_m{m}', f'Reallocated_fc_vi_trx_m{m}', t, grp_keys, is_pre=False)
                    if post_slice is not None: chunks.append(post_slice)

                    pre_slice = self._extract_sparse_matrix_slice(pre_df, f'PreSim_t{t}_fcast_m{m}', f'PreSim_fc_vi_trx_m{m}', t, grp_keys, is_pre=True)
                    if pre_slice is not None: chunks.append(pre_slice)

                    if chunks:
                        final_chunk = self._assemble_and_format_chunk(chunks, grp_keys, m, t)
                        if final_chunk is not None:
                            export_cols = grp_keys + ['period', 't', 'VAMP_Pre', 'VAMP_Post', 'VI_Txn_Pre', 'VI_Txn_Post']
                            final_chunk[export_cols].to_csv(f, header=header, index=False)
                            header = False
                            row_count += len(final_chunk)

                    gc.collect()

        return row_count

    # =========================================================================
    # === 3. SUMMARY REPORTS (RELOADING FROM CSV)
    # =========================================================================

    def _load_and_filter_export_data(self) -> pd.DataFrame:
        """Loads the massive CSV back into memory with 32-bit float precision."""
        export_dtypes = {
            'Company': 'category', 'vampMid': 'category', 'RPGT': 'category', 'BIN': 'category', 'Currency': 'category', 
            'period': 'int8', 't': 'int8', 
            'VAMP_Pre': 'float32', 'VAMP_Post': 'float32', 'VI_Txn_Pre': 'float32', 'VI_Txn_Post': 'float32',
            'paymentMethodProvider': 'category',  # present only when the pipeline data carries it
        }
        if os.path.exists(self.main_export_file) and os.path.getsize(self.main_export_file) > 0: 
            t_data = pd.read_csv(self.main_export_file, dtype=export_dtypes)
        else: 
            return pd.DataFrame(columns=list(export_dtypes.keys()))

        t_data.columns = [str(c).strip() for c in t_data.columns]
        if 'rpgt' in t_data.columns: t_data.rename(columns={'rpgt': 'RPGT'}, inplace=True)

        if 'Company' in t_data.columns:
            if t_data['Company'].dtype.name != 'category': t_data['Company'] = t_data['Company'].astype('category')
            invalid_companies = [c for c in t_data['Company'].cat.categories if str(c).lower().strip() != self.company.lower().strip()]
            t_data.drop(t_data.index[t_data['Company'].isin(invalid_companies)], inplace=True)
            t_data['Company'] = t_data['Company'].cat.remove_unused_categories()
            
        return t_data

    # 🟢 NEW: Highly aggregated RPGT-level matrix export
    def _generate_rpgt_level_export(self, t_data: pd.DataFrame) -> None:
        """Exports a highly summarized version of the VAMP matrix at the RPGT level."""
        grp_keys = ['Company', 'vampMid', 'RPGT', 'period', 't']
        
        for col in grp_keys:
            if col not in t_data.columns:
                t_data[col] = pd.Series('Unknown', index=t_data.index, dtype='category')
                
        rpgt_df = t_data.groupby(grp_keys, observed=True)[['VAMP_Pre', 'VAMP_Post', 'VI_Txn_Pre', 'VI_Txn_Post']].sum().reset_index()
        
        # Drop rows where everything is exactly zero
        mask_active = (rpgt_df['VAMP_Pre'] > 0) | (rpgt_df['VAMP_Post'] > 0) | (rpgt_df['VI_Txn_Pre'] > 0) | (rpgt_df['VI_Txn_Post'] > 0)
        rpgt_df = rpgt_df[mask_active].copy()
        
        output_path = os.path.join(self.output_dir, 'vamp_t_period_rpgt_export.csv')
        rpgt_df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved '{output_path}' ({len(rpgt_df)} rows).")

    def _generate_mid_level_summary(self, t_data: pd.DataFrame) -> None:
        """Generates the main FP&A reporting matrix (mid_level.csv)."""
        mid_grp = t_data.groupby(['vampMid', 'period'], observed=True)[['VAMP_Pre', 'VAMP_Post', 'VI_Txn_Pre', 'VI_Txn_Post']].sum().reset_index()
        mid_pivot = mid_grp.pivot(index='vampMid', columns='period', values=['VAMP_Pre', 'VAMP_Post', 'VI_Txn_Pre', 'VI_Txn_Post'])
        mid_pivot.columns = [f'{col[0]}_m{col[1]}' for col in mid_pivot.columns]

        mid_pivot = mid_pivot.reset_index()
        mid_pivot['vampMid'] = mid_pivot['vampMid'].astype(str)
        mid_pivot = mid_pivot.fillna(0)

        rename_map = {}
        for m in range(6):
            rename_map.update({
                f'VAMP_Pre_m{m}': f'FC_VAMP_Month_{m}',
                f'VAMP_Post_m{m}': f'FC_VAMP_Month_{m}_Post',
                f'VI_Txn_Pre_m{m}': f'FC_VI_Txn_Month_{m}',
                f'VI_Txn_Post_m{m}': f'FC_VI_Txn_Month_{m}_Post'
            })
        mid_pivot = mid_pivot.rename(columns=rename_map)

        for m in range(6):
            if f'FC_VAMP_Month_{m}' in mid_pivot.columns and f'FC_VI_Txn_Month_{m}' in mid_pivot.columns:
                mid_pivot[f'FC_VAMP_%_Month_{m}'] = (mid_pivot[f'FC_VAMP_Month_{m}'] / mid_pivot[f'FC_VI_Txn_Month_{m}']).replace([np.inf, -np.inf], 0).fillna(0)

        hist_stats = pd.DataFrame()
        if not self.attempts_df.empty:
            att_clean = self.attempts_df[self.attempts_df['Company'].astype(str).str.lower().str.strip() == self.company.lower().strip()].copy()
            mid_map = pd.Series(self.mid_df.iloc[:, 1].values, index=self.mid_df.iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
            gw_col = next((c for c in att_clean.columns if c.lower() == 'gatewayfid'), None)

            if gw_col:
                att_clean['vampMid'] = att_clean[gw_col].astype(str).str.lower().str.strip().map(mid_map).fillna('Unmapped')
                hist_cols_to_sum = ['attemptCount', 'successCount']
                if 'vamp_count' in att_clean.columns: hist_cols_to_sum.append('vamp_count')

                hist_stats = att_clean.groupby('vampMid', observed=True)[hist_cols_to_sum].sum().reset_index()
                if 'vamp_count' not in hist_stats.columns: hist_stats['vamp_count'] = 0.0
                hist_stats.rename(columns={'attemptCount': 'attemptsPre', 'successCount': 'transactionsPre', 'vamp_count': 'vampPre'}, inplace=True)
                hist_stats['vampMid'] = hist_stats['vampMid'].astype(str)

        mid_summ = pd.merge(mid_pivot, hist_stats, on='vampMid', how='outer').fillna(0) if not hist_stats.empty else mid_pivot.assign(attemptsPre=0.0, transactionsPre=0.0, vampPre=0.0)

        mid_summ['Company'] = self.company
        mid_summ['successRatePre'] = np.where(mid_summ['attemptsPre'] > 0, mid_summ['transactionsPre'] / mid_summ['attemptsPre'], 0.0)
        mid_summ['vampRatioPre'] = np.where(mid_summ['transactionsPre'] > 0, mid_summ['vampPre'] / mid_summ['transactionsPre'], 0.0)

        cols_export = ['Company', 'vampMid', 'attemptsPre', 'transactionsPre', 'successRatePre', 'vampPre', 'vampRatioPre']
        for m in range(6):
            cols_export.extend([f'FC_VAMP_Month_{m}', f'FC_VI_Txn_Month_{m}', f'FC_VAMP_%_Month_{m}', f'FC_VAMP_Month_{m}_Post', f'FC_VI_Txn_Month_{m}_Post'])

        for c in cols_export:
            if c not in mid_summ.columns: mid_summ[c] = 0.0

        mask_active = (mid_summ['FC_VAMP_Month_0_Post'] > 0) | (mid_summ['transactionsPre'] > 0) | (mid_summ['FC_VI_Txn_Month_0_Post'] > 0)
        mid_summ = mid_summ[mask_active].copy()

        output_path = os.path.join(self.output_dir, 'mid_level.csv')
        mid_summ[cols_export].to_csv(output_path, index=False)
        logger.info(f"✅ Saved '{output_path}' ({len(mid_summ)} rows).")

    def _generate_granular_impact_export(self, t_data: pd.DataFrame) -> None:
        rpgt_col = 'RPGT' if 'RPGT' in t_data.columns else 'rpgt'
        
        impact_df = t_data.groupby(['vampMid', rpgt_col, 'BIN', 'Currency', 'period'], observed=True)[['VAMP_Pre', 'VAMP_Post', 'VI_Txn_Pre', 'VI_Txn_Post']].sum().reset_index()
        impact_df.rename(columns={'VI_Txn_Pre': 'Txn_Pre', 'VI_Txn_Post': 'Txn_Post'}, inplace=True)
        
        impact_df['VAMP_Diff'] = impact_df['VAMP_Post'] - impact_df['VAMP_Pre']
        impact_df['Txn_Diff'] = impact_df['Txn_Post'] - impact_df['Txn_Pre']
        
        impact_df = impact_df[(impact_df['VAMP_Pre'] > 0) | (impact_df['VAMP_Post'] > 0) | (impact_df['Txn_Pre'] > 0) | (impact_df['Txn_Post'] > 0)].copy()
        impact_df = impact_df.sort_values(['vampMid', 'period', 'VAMP_Diff'], ascending=[True, True, False])
        
        output_path = os.path.join(self.output_dir, 'bin_rpgt_impact_export.csv')
        impact_df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved '{output_path}' ({len(impact_df)} rows).")

    def _generate_effective_rate_export(self, t_data: pd.DataFrame) -> None:
        if 'RPGT' in t_data.columns: t_data = t_data.rename(columns={'RPGT': 'rpgt'})
        
        grp_keys = ['vampMid', 'period', 'rpgt', 'BIN', 'Currency']
        
        for col in grp_keys:
            if col not in t_data.columns: t_data[col] = pd.Series('Unknown', index=t_data.index, dtype='category')

        eff_df = t_data.groupby(grp_keys, observed=True)[['VAMP_Pre', 'VAMP_Post', 'VI_Txn_Pre', 'VI_Txn_Post']].sum().reset_index()
        eff_df['Rate_Pre_Pct'] = (eff_df['VAMP_Pre'] / eff_df['VI_Txn_Pre']).replace([np.inf, -np.inf], 0).fillna(0)
        eff_df['Rate_Post_Pct'] = (eff_df['VAMP_Post'] / eff_df['VI_Txn_Post']).replace([np.inf, -np.inf], 0).fillna(0)

        m0_final = eff_df[(eff_df['period'] == 0) & ((eff_df['VI_Txn_Post'] > 50) | (eff_df['VAMP_Post'] > 1))].copy()
        m0_final = m0_final.sort_values('VAMP_Post', ascending=False)
        
        rename_map = {'VI_Txn_Post': 'Forecast_Sales', 'VAMP_Post': 'Forecast_VAMPs', 'Rate_Post_Pct': 'Forecast_Rate', 'VI_Txn_Pre': 'Sim_Sales', 'VAMP_Pre': 'Sim_VAMPs', 'Rate_Pre_Pct': 'Sim_Rate'}
        cols_out = ['vampMid', 'rpgt', 'BIN', 'Currency'] + list(rename_map.keys())

        m0_export = m0_final[cols_out].rename(columns=rename_map)
        output_path = os.path.join(self.output_dir, 'effective_rate_impact.csv')
        m0_export.to_csv(output_path, index=False)
        logger.info(f"✅ Saved '{output_path}' ({len(m0_export)} rows).")

    # =========================================================================
    # === 4. ORCHESTRATOR
    # =========================================================================

    # =========================================================================
    # === ADDITIVE: PROPOSED-SPLIT GO-LIVE PRO-RATA EXPORT
    #   Surfaces, per (vampMid, RPGT, BIN, Currency, period, t), the baseline
    #   counts plus the fraction of the ORIGINATION month (period - t) that falls
    #   on/after the Split Go Live date, using the SAME RPGT-aware weighting the
    #   AllocationEngine uses (monthly-renewal daily CDF; linear otherwise). This
    #   does NOT change any forecasting/allocation output.
    # =========================================================================

    def _month_prorata(self, month_dt: pd.Timestamp, is_mr: bool) -> float:
        """Weighted fraction of a calendar month on/after the go-live date."""
        if pd.isna(self._go_live) or pd.isna(month_dt):
            return 0.0
        days_in_mo = calendar.monthrange(month_dt.year, month_dt.month)[1]
        month_start = pd.Timestamp(month_dt.year, month_dt.month, 1)
        month_end = month_start + pd.DateOffset(months=1)
        gl = self._go_live.normalize()
        if gl <= month_start:
            return 1.0
        if gl >= month_end:
            return 0.0
        start_day = int((gl - month_start).days) + 1   # 1-based first active day
        end_day = days_in_mo
        if is_mr and self.mr_weights and month_dt.month in self.mr_weights:
            mr_w = np.array([self.mr_weights[month_dt.month].get(d, 0.0) for d in range(1, days_in_mo + 1)], dtype=float)
            if mr_w.sum() <= 0:
                mr_w = np.ones(days_in_mo)
            cdf = np.insert(np.cumsum(mr_w / mr_w.sum()), 0, 0.0)
        else:
            cdf = np.linspace(0.0, 1.0, days_in_mo + 1)
        return float(np.clip(cdf[end_day] - cdf[start_day - 1], 0.0, 1.0))

    def _prorata_lookup(self) -> pd.DataFrame:
        """Pre-computed pro-rata by (is_mr, orig_period) for orig_period in -9..5."""
        rows = []
        if not (pd.isna(self._go_live) or pd.isna(self._month_0)):
            for orig in range(-9, 6):
                month_dt = self._month_0 + pd.DateOffset(months=orig)
                for is_mr in (False, True):
                    rows.append({"is_mr": is_mr, "orig_period": orig,
                                 "pro_rata": self._month_prorata(month_dt, is_mr)})
        return pd.DataFrame(rows, columns=["is_mr", "orig_period", "pro_rata"])

    def _compute_fcp1_frac(self) -> pd.DataFrame:
        """Fraction of each cell's volume the AllocationEngine will actually reroute:
        fcpNumber == 1, AND attemptNumber == 1 for the restricted RPGTs (monthly initial /
        annual sub sale / upgrades). Mirrors allocation_engine._map_and_filter_cohorts.
        Computed at the FULL (vampMid, RPGT, BIN, Currency, paymentMethodProvider, Country)
        sub-cell grain when those columns exist — so each sub-cell moves EXACTLY its own
        eligible volume (matching the pipeline's per-row gating) rather than a cell average.
        Returns empty if the attempts frame lacks fcp data (projection defaults fcp1_frac=1.0).
        """
        _EMPTY = pd.DataFrame(columns=["vampMid", "RPGT", "BIN", "Currency", "fcp1_frac"])
        a = self.attempts_df
        if a is None or a.empty:
            return _EMPTY
        rpgt_c = "RPGT" if "RPGT" in a.columns else ("rpgt" if "rpgt" in a.columns else None)
        gw_c = "gatewayFid" if "gatewayFid" in a.columns else ("gatewayfid" if "gatewayfid" in a.columns else None)
        if rpgt_c is None or "fcpNumber" not in a.columns or gw_c is None:
            logger.info("   > fcp1_frac skipped (attempts frame has no fcpNumber/RPGT/gatewayFid); "
                        "export gets fcp1_frac=1.0.")
            return _EMPTY
        cnt = next((c for c in ("successCount", "visa_trx_count", "trx_count", "trx total")
                    if c in a.columns), None)
        if cnt is None:
            _vi = [c for c in a.columns if re.match(r'^fc_vi_trx_m\d+$', str(c))]
            cnt = _vi[0] if _vi else None
        if cnt is None:
            logger.info("   > fcp1_frac skipped (no volume column in attempts frame); export gets fcp1_frac=1.0.")
            return _EMPTY
        # Optional finer sub-cell dims (present in the export → gate exactly per sub-cell).
        pmp_c = next((c for c in ("paymentMethodProvider", "paymentmethodprovider") if c in a.columns), None)
        ctry_c = next((c for c in ("Country", "country") if c in a.columns), None)
        g2v = {}
        _md = self.mid_df
        if "gatewayfid" in _md.columns and "vampmid" in _md.columns:
            g2v = dict(zip(_md["gatewayfid"].astype(str).str.strip().str.lower(),
                           _md["vampmid"].astype(str).str.strip()))
        _use = [gw_c, rpgt_c, "BIN", "Currency", "fcpNumber", cnt] + \
               ([pmp_c] if pmp_c else []) + ([ctry_c] if ctry_c else []) + \
               (["attemptNumber"] if "attemptNumber" in a.columns else [])
        a = a[_use].copy()
        a["vampMid"] = a[gw_c].astype(str).str.strip().str.lower().map(g2v)
        a = a[a["vampMid"].notna()].copy()
        if a.empty:
            logger.info("   > fcp1_frac skipped (no gatewayFid mapped to a vampMid); export gets fcp1_frac=1.0.")
            return _EMPTY
        a["_cnt"] = pd.to_numeric(a[cnt], errors="coerce").fillna(0.0)
        a["_fcp"] = a["fcpNumber"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        a["_att"] = (a["attemptNumber"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                     if "attemptNumber" in a.columns else "1")
        _restr = a[rpgt_c].astype(str).str.strip().str.lower().isin(
            ["monthly initial", "annual sub sale", "upgrades"])
        _elig = (a["_fcp"] == "1") & (~_restr | (a["_att"] == "1"))
        a["_elig_cnt"] = np.where(_elig, a["_cnt"], 0.0)
        _keys = ["vampMid", rpgt_c, "BIN", "Currency"] + \
                ([pmp_c] if pmp_c else []) + ([ctry_c] if ctry_c else [])
        g = (a.groupby(_keys, observed=True)
             .agg(_tot=("_cnt", "sum"), _el=("_elig_cnt", "sum")).reset_index())
        g["fcp1_frac"] = np.where(g["_tot"] > 0, (g["_el"] / g["_tot"]).clip(0.0, 1.0), 1.0)
        _ren = {rpgt_c: "RPGT"}
        if pmp_c and pmp_c != "paymentMethodProvider":
            _ren[pmp_c] = "paymentMethodProvider"
        if ctry_c and ctry_c != "Country":
            _ren[ctry_c] = "Country"
        g = g.rename(columns=_ren)
        _out = ["vampMid", "RPGT", "BIN", "Currency"] + \
               (["paymentMethodProvider"] if pmp_c else []) + (["Country"] if ctry_c else []) + ["fcp1_frac"]
        return g[_out]

    def _generate_prorata_export(self, t_data: pd.DataFrame) -> None:
        if pd.isna(self._go_live) or pd.isna(self._month_0):
            logger.info("   > pro-rata export skipped (no Split Go Live / month_0 date).")
            return
        rpgt_col = 'RPGT' if 'RPGT' in t_data.columns else 'rpgt'
        base_cols = ['vampMid', rpgt_col, 'BIN', 'Currency', 'period', 't', 'VAMP_Pre', 'VI_Txn_Pre']
        if 'paymentMethodProvider' in t_data.columns:
            base_cols.insert(4, 'paymentMethodProvider')  # after Currency
        if 'Country' in t_data.columns:
            base_cols.insert(4, 'Country')                # after Currency (USA/Non-USA baseline)
        df = t_data[base_cols].copy()
        df = df.rename(columns={rpgt_col: 'RPGT', 'VAMP_Pre': 'vampCount', 'VI_Txn_Pre': 'VI_Txn_Count'})
        df['orig_period'] = df['period'].astype(int) - df['t'].astype(int)
        df['is_mr'] = df['RPGT'].astype(str).str.lower().str.strip() == 'monthly renewal'
        lut = self._prorata_lookup()
        df = df.merge(lut, on=['is_mr', 'orig_period'], how='left')
        # Transactions originated before month 0 pre-date the forecast window, so
        # the proposed split cannot affect them: pro_rata = 0.
        df['pro_rata'] = np.where(df['orig_period'] < 0, 0.0, df['pro_rata'].fillna(0.0))
        df = df.drop(columns=['orig_period', 'is_mr'])

        # ADDITIVE: fcp1_frac = fraction of each (RPGT, BIN, Currency) cell that the
        # AllocationEngine actually reroutes (fcpNumber==1, attemptNumber==1 for restricted
        # RPGTs). The routing optimiser multiplies pro_rata × fcp1_frac so its pre/post
        # impact only moves the same cohort the pipeline forecasts; the rest stays baseline.
        _ff = self._compute_fcp1_frac()
        if not _ff.empty:
            # Join on whatever grain fcp1_frac was computed at — including pmp / Country when
            # present, so each sub-cell gets its EXACT eligible fraction (not a cell average).
            _keys = [k for k in ["vampMid", "RPGT", "BIN", "Currency",
                                 "paymentMethodProvider", "Country"]
                     if k in _ff.columns and k in df.columns]
            for _k in _keys:                     # case/space-insensitive join
                df["_j_" + _k] = df[_k].astype(str).str.strip().str.lower()
                _ff["_j_" + _k] = _ff[_k].astype(str).str.strip().str.lower()
            df = df.merge(_ff[["_j_" + k for k in _keys] + ["fcp1_frac"]],
                          on=["_j_" + k for k in _keys], how="left")
            df = df.drop(columns=["_j_" + k for k in _keys])
            _covered = float(df['fcp1_frac'].notna().mean()) * 100.0
            df['fcp1_frac'] = pd.to_numeric(df['fcp1_frac'], errors='coerce').fillna(1.0).clip(0.0, 1.0)
            logger.info(f"   > fcp1_frac merged at {len(_keys)}-key grain "
                        f"({_covered:.0f}% of rows matched; unmatched default 1.0).")
        else:
            df['fcp1_frac'] = 1.0

        out = os.path.join(self.output_dir, 'vamp_t_period_prorata_export.csv')
        df.to_csv(out, index=False)
        logger.info(f"   > pro-rata export saved ({len(df)} rows, go-live {self._go_live.date()}) -> {out}")

    def run_all_exports(self, pre_df: pd.DataFrame, post_df: pd.DataFrame):
        logger.info("📦 Generating the Massive Export Matrix (Pop & Stack)...")
        row_count = self.export_chunked_vamp_matrix(pre_df, post_df)
        logger.info(f"   > Matrix successfully saved to disk ({row_count} rows).")

        logger.info("📊 Reloading data for Summary Dashboards...")
        t_data = self._load_and_filter_export_data()

        logger.info("📝 Building Final FP&A Reports...")
        self._generate_rpgt_level_export(t_data)  # 🟢 NEW
        self._generate_mid_level_summary(t_data)
        self._generate_granular_impact_export(t_data)
        self._generate_effective_rate_export(t_data)
        self._generate_prorata_export(t_data)  # additive: proposed-split go-live pro-rata

        logger.info("✅ Export Suite Complete. All files saved to data/outputs/")