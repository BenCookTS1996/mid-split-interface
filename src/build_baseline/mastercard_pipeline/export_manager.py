import os
import gc
import re
import calendar
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from .utils import setup_logger

logger = setup_logger(__name__)

__build__ = "2026-08-17-mastercard-initial+reconcile-guard+prorata-export"


def reconcile_granular_to_mid_level(output_dir, *, id_col, vamp_metric, txn_metric,
                                    g_vamp_pre, g_vamp_post, g_txn_pre, g_txn_post,
                                    logger, tol=1.0):
    """RECONCILIATION GUARD: verify bin_rpgt_impact_export.csv aggregates (per id × period) to
    mid_level.csv EXACTLY. Both are built from the same t_data, so they must tie — Tab 3 reads the
    granular file and the Validate Split table reads mid_level, so this proves the two views agree.
    Logs a ✅ certificate or a ⚠️ naming the worst-offending id/period/metric. Never raises.

    Reconciles on the MIDs mid_level reports (it applies an M0 active-mask that drops M0-inactive
    MIDs); any extra MID in the granular file is noted as expected, not a failure.
    """
    try:
        mid_p = os.path.join(output_dir, 'mid_level.csv')
        bin_p = os.path.join(output_dir, 'bin_rpgt_impact_export.csv')
        if not (os.path.exists(mid_p) and os.path.exists(bin_p)):
            logger.info("   > reconciliation skipped (mid_level.csv / bin_rpgt_impact_export.csv missing).")
            return
        mid = pd.read_csv(mid_p)
        binr = pd.read_csv(bin_p)
        if id_col not in mid.columns or id_col not in binr.columns or 'period' not in binr.columns:
            logger.info("   > reconciliation skipped (expected columns not present).")
            return
        g = binr.groupby([id_col, 'period'], as_index=False)[
            [g_vamp_pre, g_vamp_post, g_txn_pre, g_txn_post]].sum()
        g['_id'] = g[id_col].astype(str).str.strip()
        recs = []
        for m in range(6):
            cmap = {f'FC_{vamp_metric}_Month_{m}': 'v_pre',
                    f'FC_{vamp_metric}_Month_{m}_Post': 'v_post',
                    f'FC_{txn_metric}_Month_{m}': 't_pre',
                    f'FC_{txn_metric}_Month_{m}_Post': 't_post'}
            if not all(c in mid.columns for c in cmap):
                continue
            sub = mid[[id_col] + list(cmap)].rename(columns=cmap)
            sub['period'] = m
            recs.append(sub)
        if not recs:
            logger.info("   > reconciliation skipped (mid_level month columns not found).")
            return
        ml = pd.concat(recs, ignore_index=True)
        ml['_id'] = ml[id_col].astype(str).str.strip()
        merged = ml.merge(g, on=['_id', 'period'], how='left').fillna(0.0)
        pairs = [(f'{vamp_metric} pre', g_vamp_pre, 'v_pre'),
                 (f'{vamp_metric} post', g_vamp_post, 'v_post'),
                 (f'{txn_metric} pre', g_txn_pre, 't_pre'),
                 (f'{txn_metric} post', g_txn_post, 't_post')]
        worst = ('', 0.0, '', 0, 0.0, 0.0)
        for label, gc, mc in pairs:
            d = (merged[gc] - merged[mc]).abs()
            if len(d):
                i = int(np.asarray(d.values).argmax()); md = float(d.iloc[i])
                if md > worst[1]:
                    worst = (label, md, str(merged['_id'].iloc[i]), int(merged['period'].iloc[i]),
                             float(merged[gc].iloc[i]), float(merged[mc].iloc[i]))
        if worst[1] <= tol:
            logger.info("   > ✅ RECONCILIATION OK: bin_rpgt_impact_export aggregates to mid_level "
                        f"exactly (max abs diff {worst[1]:.4g} ≤ {tol}). Tab-3 tables will tie to the "
                        "Validate Split table.")
        else:
            logger.warning("   > ⚠️ RECONCILIATION MISMATCH: bin_rpgt_impact_export does NOT aggregate "
                           f"to mid_level. Worst: {worst[0]} for '{worst[2]}' period {worst[3]} — granular "
                           f"{worst[4]:,.2f} vs mid_level {worst[5]:,.2f} (diff {worst[1]:,.2f}). Tab-3 and "
                           "the Validate Split table may not tie for this MID.")
        _extra = sorted(set(g['_id']) - set(ml['_id']))
        if _extra:
            logger.info(f"   > (note: {len(_extra)} MID(s) in bin_rpgt not in mid_level — dropped by its "
                        "M0 active-mask; expected.)")
    except Exception as _e:  # noqa: BLE001 - a guard must never break the export
        logger.info(f"   > reconciliation check skipped ({type(_e).__name__}: {_e}).")


class ExportManager:
    """
    MASTERCARD export manager (parallels the Visa/VAMP ExportManager).

    Streams the massive pre/post allocation matrices to disk via "Pop & Stack"
    compression, then builds the downstream summary CSVs. The risk metric is the
    Mastercard CHARGEBACK (CB), and the transaction metric is MC_Txn.

    Outputs (per the source notebook Sections 9-13):
      * cb_t_period_export.csv    - the full period x t chargeback / txn matrix
      * mid_level.csv             - per-MID pre/post summary over the 6 forecast months
      * bin_rpgt_impact_export.csv- per BIN x RPGT impact (all periods)
      * effective_rate_impact.csv - month-1 effective CB-rate impact (period 1, because
                                    period 0 is the injected real history)
    """

    def __init__(self, config: Dict[str, Any], mid_df: pd.DataFrame, attempts_df: pd.DataFrame,
                 mr_weights: Dict[Any, Any] = None):
        self.config = config
        rs = config['run_settings']
        self.company = str(rs['company']).strip()
        self.month_var = str(rs['month_var']).strip()
        self.mr_weights = mr_weights or {}  # MR daily weights: now used by the pro-rata export
        # Split Go-Live + month-0 drive the additive pro-rata export (parity with the visa pipeline).
        self._month_0 = pd.to_datetime(rs.get('month_0_start_date'), errors='coerce')
        self._go_live = pd.to_datetime(rs.get('split_go_live_date'), errors='coerce')

        self.mid_df = mid_df.copy()
        self.attempts_df = attempts_df.copy()

        self.output_dir = config['paths'].get('output_dir', 'data/outputs/').format(
            month_var=self.month_var, company=self.company)
        os.makedirs(self.output_dir, exist_ok=True)
        self.main_export_file = os.path.join(self.output_dir, 'cb_t_period_export.csv')

    # =========================================================================
    # === VALIDATION & MID MAPPING
    # =========================================================================

    @staticmethod
    def validate_volume_integrity(pre_df, post_df) -> float:
        input_vol_cb = sum(pre_df[c].sum() for c in pre_df.columns if 'PreSim_t' in c)
        output_vol_cb = sum(post_df[c].sum() for c in post_df.columns if 'Reallocated_t' in c)
        diff = output_vol_cb - input_vol_cb
        logger.info(f"   > Input CB:    {input_vol_cb:,.0f}")
        logger.info(f"   > Output CB:   {output_vol_cb:,.0f}")
        logger.info(f"   > Difference:  {diff:,.0f}")
        if abs(diff) > 1:
            logger.warning("   ❌ WARNING: Volume variance detected.")
        else:
            logger.info("   ✅ Perfect Volume Integrity.")
        return diff

    def _map_mids_and_compress(self, pre_df, post_df) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        mid_df = self.mid_df
        mid_df.columns = [str(c).strip().lower().replace(" ", "") for c in mid_df.columns]
        if 'mastercardmid' not in mid_df.columns:
            raise ValueError(f"❌ ERROR: 'mastercardmid' not found in MID list! Columns: {mid_df.columns.tolist()}.")
        mid_map = pd.Series(mid_df['mastercardmid'].values, index=mid_df['gatewayfid'].astype(str).str.lower().str.strip()).to_dict()

        post_df['finalGateway'] = post_df['finalGateway'].astype(str).str.lower().str.strip()
        pre_df['finalGateway'] = pre_df['finalGateway'].astype(str).str.lower().str.strip()

        post_df['mastercardMid'] = post_df['finalGateway'].map(mid_map).fillna('Unmapped')
        pre_df['mastercardMid'] = pre_df['finalGateway'].map(mid_map).fillna('Unmapped')

        unmapped_gws = set(post_df.loc[post_df['mastercardMid'] == 'Unmapped', 'finalGateway']).union(set(pre_df.loc[pre_df['mastercardMid'] == 'Unmapped', 'finalGateway']))
        if unmapped_gws:
            logger.warning(f"   ⚠️ WARNING: {len(unmapped_gws)} raw gateways missing from the MID list:")
            for gw in sorted(unmapped_gws):
                logger.warning(f"      - {gw}")

        post_df.drop(columns=['finalGateway'], inplace=True)
        pre_df.drop(columns=['finalGateway'], inplace=True)

        grp = ['Company', 'mastercardMid', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number', 'fcpNumber', 'attemptNumber']
        for c in grp:
            if c in post_df.columns:
                post_df[c] = post_df[c].astype('category')
            if c in pre_df.columns:
                pre_df[c] = pre_df[c].astype('category')
        return pre_df, post_df, grp

    # =========================================================================
    # === POP & STACK MATRIX EXPORT
    # =========================================================================

    @staticmethod
    def _extract_sparse_matrix_slice(df, v_col, vi_col, t, grp_keys, is_pre):
        if v_col not in df.columns:
            return None

        v_vals = df.pop(v_col).values.astype(np.float64)
        vi_vals = df.pop(vi_col).values.astype(np.float64) if t == 0 and vi_col in df.columns else np.zeros(len(df), dtype=np.float64)

        mask = (v_vals != 0) | (vi_vals != 0)
        if not mask.any():
            return None

        slice_df = df.loc[mask, grp_keys].copy()

        if is_pre:
            slice_df['CB_Pre'], slice_df['MC_Txn_Pre'] = v_vals[mask], vi_vals[mask]
            slice_df['CB_Post'], slice_df['MC_Txn_Post'] = 0.0, 0.0
        else:
            slice_df['CB_Post'], slice_df['MC_Txn_Post'] = v_vals[mask], vi_vals[mask]
            slice_df['CB_Pre'], slice_df['MC_Txn_Pre'] = 0.0, 0.0

        return slice_df

    @staticmethod
    def _assemble_and_format_chunk(chunks, grp_keys, m, t):
        chunk = pd.concat(chunks, ignore_index=True, copy=False).groupby(grp_keys, as_index=False, observed=True).sum()
        chunk = chunk[(chunk['CB_Pre'] > 0) | (chunk['CB_Post'] > 0) | (chunk['MC_Txn_Pre'] > 0) | (chunk['MC_Txn_Post'] > 0)].copy()
        if not chunk.empty:
            chunk['period'], chunk['t'] = m, t
            return chunk
        return None

    def export_chunked_cb_matrix(self, pre_df, post_df, grp_keys) -> int:
        header, row_count = True, 0
        output_file = self.main_export_file

        with open(output_file, 'w') as f:
            for m in range(6):
                for t in range(10):
                    chunks = []

                    post_slice = self._extract_sparse_matrix_slice(
                        post_df, f'Reallocated_t{t}_fcast_m{m}', f'Reallocated_fc_mc_trx_m{m}', t, grp_keys, is_pre=False)
                    if post_slice is not None:
                        chunks.append(post_slice)

                    pre_slice = self._extract_sparse_matrix_slice(
                        pre_df, f'PreSim_t{t}_fcast_m{m}', f'PreSim_fc_mc_trx_m{m}', t, grp_keys, is_pre=True)
                    if pre_slice is not None:
                        chunks.append(pre_slice)

                    if chunks:
                        final_chunk = self._assemble_and_format_chunk(chunks, grp_keys, m, t)
                        if final_chunk is not None:
                            export_cols = grp_keys + ['period', 't', 'CB_Pre', 'CB_Post', 'MC_Txn_Pre', 'MC_Txn_Post']
                            final_chunk[export_cols].to_csv(f, header=header, index=False)
                            header = False
                            row_count += len(final_chunk)

                    gc.collect()

        return row_count

    # =========================================================================
    # === DOWNSTREAM SUMMARY EXPORTS
    # =========================================================================

    def _load_and_filter_export_data(self) -> pd.DataFrame:
        export_dtypes = {'Company': 'category', 'mastercardMid': 'category', 'RPGT': 'category', 'Currency': 'category', 'BIN': 'category', 'paymentMethodProvider': 'category', 'Country': 'category', 'renewal_number': 'category', 'fcpNumber': 'category', 'attemptNumber': 'category', 'period': 'int8', 't': 'int8', 'CB_Pre': 'float64', 'CB_Post': 'float64', 'MC_Txn_Pre': 'float64', 'MC_Txn_Post': 'float64'}
        file_name = self.main_export_file
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            t_data = pd.read_csv(file_name, dtype=export_dtypes)
        else:
            return pd.DataFrame(columns=list(export_dtypes.keys()))

        t_data.columns = [str(c).strip() for c in t_data.columns]
        if 'rpgt' in t_data.columns:
            t_data.rename(columns={'rpgt': 'RPGT'}, inplace=True)

        target_company = self.company
        if 'Company' in t_data.columns:
            if t_data['Company'].dtype.name != 'category':
                t_data['Company'] = t_data['Company'].astype('category')
            invalid_companies = [c for c in t_data['Company'].cat.categories if str(c).lower().strip() != target_company.lower().strip()]
            t_data.drop(t_data.index[t_data['Company'].isin(invalid_companies)], inplace=True)
            t_data['Company'] = t_data['Company'].cat.remove_unused_categories()
        return t_data

    @staticmethod
    def _build_forecast_pivot(t_data) -> pd.DataFrame:
        mid_grp = t_data.groupby(['mastercardMid', 'period'], observed=True)[['CB_Pre', 'CB_Post', 'MC_Txn_Pre', 'MC_Txn_Post']].sum().reset_index()
        mid_pivot = mid_grp.pivot(index='mastercardMid', columns='period', values=['CB_Pre', 'CB_Post', 'MC_Txn_Pre', 'MC_Txn_Post'])
        mid_pivot.columns = [f'{col[0]}_m{col[1]}' for col in mid_pivot.columns]

        mid_pivot = mid_pivot.reset_index()
        mid_pivot['mastercardMid'] = mid_pivot['mastercardMid'].astype(str)
        mid_pivot = mid_pivot.fillna(0)

        rename_map = {}
        for m in range(6):
            rename_map.update({
                f'CB_Pre_m{m}': f'FC_CB_Month_{m}',
                f'CB_Post_m{m}': f'FC_CB_Month_{m}_Post',
                f'MC_Txn_Pre_m{m}': f'FC_MC_Txn_Month_{m}',
                f'MC_Txn_Post_m{m}': f'FC_MC_Txn_Month_{m}_Post'
            })
        mid_pivot = mid_pivot.rename(columns=rename_map)

        for m in range(6):
            if f'FC_CB_Month_{m}' in mid_pivot.columns and f'FC_MC_Txn_Month_{m}' in mid_pivot.columns:
                mid_pivot[f'FC_CB_%_Month_{m}'] = (mid_pivot[f'FC_CB_Month_{m}'] / mid_pivot[f'FC_MC_Txn_Month_{m}']).replace([np.inf, -np.inf], 0).fillna(0)

        return mid_pivot

    def _aggregate_historical_stats(self, target_company) -> pd.DataFrame:
        attempts_df, mid_df = self.attempts_df, self.mid_df
        hist_stats = pd.DataFrame()
        if attempts_df.empty:
            return hist_stats

        att_clean = attempts_df[attempts_df['Company'].astype(str).str.lower().str.strip() == target_company.lower().strip()].copy()
        mid_map = pd.Series(mid_df.iloc[:, 1].values, index=mid_df.iloc[:, 0].astype(str).str.lower().str.strip()).to_dict()
        gw_col = next((c for c in att_clean.columns if c.lower() == 'gatewayfid'), None)

        if gw_col:
            att_clean['mastercardMid'] = att_clean[gw_col].astype(str).str.lower().str.strip().map(mid_map).fillna('Unmapped')
            hist_cols_to_sum = ['attemptCount', 'successCount']
            if 'cb_count' in att_clean.columns:
                hist_cols_to_sum.append('cb_count')

            hist_stats = att_clean.groupby('mastercardMid', observed=True)[hist_cols_to_sum].sum().reset_index()
            if 'cb_count' not in hist_stats.columns:
                hist_stats['cb_count'] = 0.0

            hist_stats.rename(columns={'attemptCount': 'attemptsPre', 'successCount': 'transactionsPre', 'cb_count': 'cbPre'}, inplace=True)
            hist_stats['mastercardMid'] = hist_stats['mastercardMid'].astype(str)

        return hist_stats

    def _merge_and_format_mid_summary(self, mid_pivot, hist_stats, target_company) -> pd.DataFrame:
        mid_summ = pd.merge(mid_pivot, hist_stats, on='mastercardMid', how='outer').fillna(0) if not hist_stats.empty else mid_pivot.assign(attemptsPre=0.0, transactionsPre=0.0, cbPre=0.0)

        mid_summ['Company'] = target_company
        mid_summ['successRatePre'] = np.where(mid_summ['attemptsPre'] > 0, mid_summ['transactionsPre'] / mid_summ['attemptsPre'], 0.0)
        mid_summ['cbRatioPre'] = np.where(mid_summ['transactionsPre'] > 0, mid_summ['cbPre'] / mid_summ['transactionsPre'], 0.0)

        cols_export = ['Company', 'mastercardMid', 'attemptsPre', 'transactionsPre', 'successRatePre', 'cbPre', 'cbRatioPre']
        for m in range(6):
            cols_export.extend([f'FC_CB_Month_{m}', f'FC_MC_Txn_Month_{m}', f'FC_CB_%_Month_{m}', f'FC_CB_Month_{m}_Post', f'FC_MC_Txn_Month_{m}_Post'])

        for c in cols_export:
            if c not in mid_summ.columns:
                mid_summ[c] = 0.0

        mask_active = (mid_summ['FC_CB_Month_0_Post'] > 0) | (mid_summ['transactionsPre'] > 0) | (mid_summ['FC_MC_Txn_Month_0_Post'] > 0)
        mid_summ = mid_summ[mask_active].copy()

        out_path = os.path.join(self.output_dir, 'mid_level.csv')
        mid_summ[cols_export].to_csv(out_path, index=False)
        return mid_summ

    def _generate_mid_level_summary(self, t_data) -> pd.DataFrame:
        mid_pivot = self._build_forecast_pivot(t_data)
        hist_stats = self._aggregate_historical_stats(self.company)
        return self._merge_and_format_mid_summary(mid_pivot, hist_stats, self.company)

    def _generate_granular_impact_export(self, t_data) -> pd.DataFrame:
        rpgt_col = 'RPGT' if 'RPGT' in t_data.columns else 'rpgt'
        if 'Country' not in t_data.columns:
            t_data['Country'] = pd.Series('Unknown', index=t_data.index, dtype='category')
        # Include Currency in the grain (when present) so downstream can price at RPGT × Currency
        # (Tab 3's ATV). Adding a group key doesn't change the per-MID×period totals, so the
        # reconciliation guard still ties bin_rpgt to mid_level exactly.
        _grp = ['mastercardMid', rpgt_col] + (['Currency'] if 'Currency' in t_data.columns else []) \
            + ['Country', 'BIN', 'period']
        impact_df = t_data.groupby(_grp, observed=True)[['CB_Pre', 'CB_Post', 'MC_Txn_Pre', 'MC_Txn_Post']].sum().reset_index().rename(columns={'MC_Txn_Pre': 'Txn_Pre', 'MC_Txn_Post': 'Txn_Post'})
        impact_df['CB_Diff'] = impact_df['CB_Post'] - impact_df['CB_Pre']
        impact_df['Txn_Diff'] = impact_df['Txn_Post'] - impact_df['Txn_Pre']
        impact_df = impact_df[(impact_df['CB_Pre'] > 0) | (impact_df['CB_Post'] > 0) | (impact_df['Txn_Pre'] > 0) | (impact_df['Txn_Post'] > 0)].copy().sort_values(['mastercardMid', 'period', 'CB_Diff'], ascending=[True, True, False])
        out_path = os.path.join(self.output_dir, 'bin_rpgt_impact_export.csv')
        impact_df.to_csv(out_path, index=False)
        return impact_df

    def _generate_effective_rate_export(self, t_data) -> pd.DataFrame:
        """🟢 MASTERCARD: filters to period == 1 because period 0 is the injected history."""
        if 'RPGT' in t_data.columns:
            t_data = t_data.rename(columns={'RPGT': 'rpgt'})
        grp_keys = ['mastercardMid', 'period', 'rpgt', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number']
        for col in grp_keys:
            if col not in t_data.columns:
                t_data[col] = pd.Series('Unknown', index=t_data.index, dtype='category')

        eff_df = t_data.groupby(grp_keys, observed=True)[['CB_Pre', 'CB_Post', 'MC_Txn_Pre', 'MC_Txn_Post']].sum().reset_index()
        eff_df['Rate_Pre_Pct'] = (eff_df['CB_Pre'] / eff_df['MC_Txn_Pre']).replace([np.inf, -np.inf], 0).fillna(0)
        eff_df['Rate_Post_Pct'] = (eff_df['CB_Post'] / eff_df['MC_Txn_Post']).replace([np.inf, -np.inf], 0).fillna(0)

        # 🟢 M1 FILTER FOR MASTERCARD (period 0 is injected real history)
        m1_rates = eff_df[eff_df['period'] == 1].copy()
        mask_vol = (m1_rates['MC_Txn_Post'] > 50) | (m1_rates['CB_Post'] > 1)
        m1_final = m1_rates[mask_vol].copy().sort_values('CB_Post', ascending=False)

        rename_map = {'MC_Txn_Post': 'Forecast_Sales', 'CB_Post': 'Forecast_CBs', 'Rate_Post_Pct': 'Forecast_Rate', 'MC_Txn_Pre': 'Sim_Sales', 'CB_Pre': 'Sim_CBs', 'Rate_Pre_Pct': 'Sim_Rate'}
        cols_out = ['mastercardMid', 'rpgt', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number'] + list(rename_map.keys())

        m1_export = m1_final[cols_out].rename(columns=rename_map)
        out_path = os.path.join(self.output_dir, 'effective_rate_impact.csv')
        m1_export.to_csv(out_path, index=False)
        return m1_export

    # =========================================================================
    # === PRO-RATA EXPORT (mastercard parity with the visa/vamp pipeline)
    # =========================================================================
    def _month_prorata(self, month_dt, is_mr: bool) -> float:
        """Weighted fraction of a calendar month on/after the Split Go-Live date.
        Scheme-agnostic — identical to the visa pipeline (depends only on go-live / month-0 /
        MR daily weights)."""
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
        start_day = int((gl - month_start).days) + 1
        end_day = days_in_mo
        if is_mr and self.mr_weights and month_dt.month in self.mr_weights:
            mr_w = np.array([self.mr_weights[month_dt.month].get(d, 0.0)
                             for d in range(1, days_in_mo + 1)], dtype=float)
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

    def _generate_prorata_export(self, t_data: pd.DataFrame) -> None:
        """Mastercard equivalent of the visa pro-rata export → 'mc_cb_t_period_prorata_export.csv':
        the go-live-pro-rated period×t baseline of chargebacks (CB) and mastercard txn (MC).

        Column schema mirrors the visa export (id + period/t + count columns + pro_rata + fcp1_frac)
        but carries CB / MC semantics: CB_Pre → cbCount, MC_Txn_Pre → MC_Txn_Count.

        Caveats: (1) fcp1_frac defaults to 1.0 — the visa cohort-gating port (fcpNumber==1 etc.)
        is a follow-up; (2) this file is NOT yet consumed by the routing/impact loaders, which read
        the visa-named 'vamp_t_period_prorata_export.csv' — it's an inspection/parity artifact until
        those loaders are made scheme-aware."""
        if pd.isna(self._go_live) or pd.isna(self._month_0):
            logger.info("   > pro-rata export skipped (no Split Go Live / month_0 date).")
            return
        rpgt_col = 'RPGT' if 'RPGT' in t_data.columns else 'rpgt'
        _need = ['mastercardMid', rpgt_col, 'BIN', 'Currency', 'period', 't', 'CB_Pre', 'MC_Txn_Pre']
        _missing = [c for c in _need if c not in t_data.columns]
        if _missing:
            logger.warning(f"   > pro-rata export skipped (t_data missing {_missing}).")
            return
        base_cols = list(_need)
        if 'paymentMethodProvider' in t_data.columns:
            base_cols.insert(4, 'paymentMethodProvider')
        if 'Country' in t_data.columns:
            base_cols.insert(4, 'Country')
        # 19ep: carry POST as well as PRE, mirroring the visa export. APPENDED, not inserted,
        # so the Country / paymentMethodProvider positions above are untouched.
        _post_src = [c for c in ('CB_Post', 'MC_Txn_Post') if c in t_data.columns]
        base_cols = base_cols + _post_src
        df = t_data[base_cols].copy()
        df = df.rename(columns={rpgt_col: 'RPGT', 'CB_Pre': 'cbCount', 'MC_Txn_Pre': 'MC_Txn_Count',
                                'CB_Post': 'cbCount_Post', 'MC_Txn_Post': 'MC_Txn_Count_Post'})
        if len(_post_src) < 2:
            logger.warning("   > pro-rata export: t_data carries %s of the two POST columns, so "
                           "tab 3's Validate table will keep reading bin_rpgt_impact_export.csv.",
                           len(_post_src))
        df['orig_period'] = df['period'].astype(int) - df['t'].astype(int)
        df['is_mr'] = df['RPGT'].astype(str).str.lower().str.strip() == 'monthly renewal'
        lut = self._prorata_lookup()
        df = df.merge(lut, on=['is_mr', 'orig_period'], how='left')
        # Transactions originated before month 0 pre-date the forecast window → pro_rata = 0.
        df['pro_rata'] = np.where(df['orig_period'] < 0, 0.0, df['pro_rata'].fillna(0.0))
        df = df.drop(columns=['orig_period', 'is_mr'])
        df['fcp1_frac'] = 1.0   # TODO: port the visa cohort-gating fcp1_frac for mastercard
        out = os.path.join(self.output_dir, 'mc_cb_t_period_prorata_export.csv')
        df.to_csv(out, index=False)
        logger.info(f"   > pro-rata export saved ({len(df)} rows, go-live {self._go_live.date()}) -> {out}")

    # =========================================================================
    # === PUBLIC RUN ENTRYPOINT
    # =========================================================================

    def run_all_exports(self, pre_df, post_df) -> None:
        logger.info("📦 SECTION 9: VALIDATION, MAPPING & EXPORT (Mastercard)")
        self.validate_volume_integrity(pre_df, post_df)

        logger.info("   > Assigning MIDs and compressing text keys...")
        pre_df, post_df, grp_keys = self._map_mids_and_compress(pre_df, post_df)

        logger.info("   > Exporting to CSV using Pop & Stack compression...")
        row_count = self.export_chunked_cb_matrix(pre_df, post_df, grp_keys)
        logger.info(f"✅ Exported '{self.main_export_file}' with {row_count:,.0f} rows.")

        del post_df, pre_df
        gc.collect()

        logger.info("📊 Loading export data (single source of truth)...")
        t_data = self._load_and_filter_export_data()
        logger.info(f"   > Filtered t_data to {len(t_data):,.0f} rows for '{self.company}'.")

        logger.info("📊 Generating Mid Level Summary...")
        mid_summ = self._generate_mid_level_summary(t_data)
        logger.info(f"✅ Saved 'mid_level.csv' ({len(mid_summ)} rows).")

        logger.info("📊 Generating Granular Impact Export (all periods)...")
        impact_df = self._generate_granular_impact_export(t_data)
        logger.info(f"✅ Saved 'bin_rpgt_impact_export.csv' with {len(impact_df):,.0f} rows.")

        logger.info("📊 Generating Effective Rate Export (M1)...")
        m1_export = self._generate_effective_rate_export(t_data)
        logger.info(f"✅ Saved 'effective_rate_impact.csv' with {len(m1_export):,.0f} rows.")
        if not m1_export.empty:
            logger.info(f"   > Total Forecast Sales (M1): {m1_export['Forecast_Sales'].sum():,.0f}")
            logger.info(f"   > Total Forecast CBs (M1): {m1_export['Forecast_CBs'].sum():,.0f}")

        # Additive pro-rata export (parity with visa). Guarded so this new, not-yet-live-validated
        # artifact can never break the (validated) rest of the mastercard export run.
        try:
            logger.info("📊 Generating Pro-Rata Export (go-live baseline)...")
            self._generate_prorata_export(t_data)
        except Exception as _e:  # noqa: BLE001
            logger.warning(f"   ⚠️ pro-rata export failed (skipped, non-fatal): "
                           f"{type(_e).__name__}: {_e}")

        # GUARD: confirm the granular bin_rpgt export ties exactly to mid_level (Tab 3 reads the
        # former, the Validate Split table reads the latter — this proves the two views reconcile).
        reconcile_granular_to_mid_level(
            self.output_dir, id_col='mastercardMid', vamp_metric='CB', txn_metric='MC_Txn',
            g_vamp_pre='CB_Pre', g_vamp_post='CB_Post', g_txn_pre='Txn_Pre', g_txn_post='Txn_Post',
            logger=logger)
