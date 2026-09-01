import os
import pickle
import calendar
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from .utils import setup_logger

logger = setup_logger(__name__)

__build__ = "2026-08-12-mastercard-initial"


class ActuarialEngine:
    """
    MASTERCARD actuarial engine (parallels the Visa/VAMP ActuarialEngine).

    Builds the chargeback (CB) "thermometer" decay curves, extrapolates lifetime CB
    magnitudes for each micro-cohort, distributes them down the waterfall onto the
    historical gateway mix, and finally applies the MASTERCARD SHIFT:
      * offset FP&A forecast sales forward one month (fc_mc_trx_m{m-1} -> fc_mc_trx_m{m});
      * inject the last completed month's REAL Mastercard transactions as an unaltered
        Month 0 baseline (fc_mc_trx_m0), which the allocator later leaves untouched.

    Risk metric = Mastercard chargebacks (cb_count / cb_fcast), not Visa VAMPs.
    """

    def __init__(self, config: Dict[str, Any], fcast_data: pd.DataFrame,
                 mapping_data: pd.DataFrame, longterm_fcast_pre: pd.DataFrame,
                 attempts_df: pd.DataFrame):
        self.config = config
        self.thermo_config = config.get('thermometer_config', {}) or {}

        self.fcast_data = fcast_data.copy()
        self.mapping_data = mapping_data.copy()
        self.longterm_fcast = longterm_fcast_pre.copy()
        self.attempts_df = attempts_df.copy()

        rs = config['run_settings']
        act = config.get('actuarial_settings', {})
        self.m0_start_dt = pd.to_datetime(rs['month_0_start_date'])
        self.company = str(rs['company']).strip()
        self.t0_lookback_months = int(act.get('t0_lookback_months', 0))
        self.decay_factor = float(act.get('decay_factor', 0.5))
        self.sample_months = int(act.get('thermometer_sample_months', 1))
        self.load_curves_from_cache = bool(rs.get('load_curves_from_cache', True))

        cache_dir = config['paths']['cache_path'].format(
            month_var=str(rs['month_var']).strip(), company=self.company)
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'reference_curves_cache_mc_v6.pkl')

        self.profile_keys = ['Company', 'paymentMethodProvider', 'Country', 'rpgt', 'Currency', 'BIN', 'renewal_number', 'fcpNumber']
        self.m_days = {m: calendar.monthrange((self.m0_start_dt + pd.DateOffset(months=m)).year,
                                              (self.m0_start_dt + pd.DateOffset(months=m)).month)[1] for m in range(6)}

    # =========================================================================
    # === KEY / COLUMN HELPERS
    # =========================================================================

    def _clean_col_names(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {'riskDefinedProductSubscriptionType': 'rpgt', 'risk_defined_subscription_product_type': 'rpgt', 'RPGT': 'rpgt', 'company': 'Company', 'Brand': 'Company', 'brand': 'Company', 'gateway_fid': 'gatewayFid', 'Gateway_Src': 'gatewayFid', 'Gateway': 'gatewayFid', 'paymentmethodprovider': 'paymentMethodProvider', 'country': 'Country', 'Country': 'Country', 'bin': 'BIN', 'Bin': 'BIN', 'currency': 'Currency', 'fcpnumber': 'fcpNumber', 'attemptnumber': 'attemptNumber'}
        rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        if 'BIN' in df.columns:
            df['BIN'] = df['BIN'].astype(str).str.split('.').str[0].str.strip()
        df['attemptNumber'] = df['attemptNumber'].astype(str).str.lower().str.strip() if 'attemptNumber' in df.columns else '1'
        return df

    def _fast_apply_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._clean_col_names(df)
        key_series = df[self.profile_keys[0]].astype(str).str.lower().str.strip()
        df[self.profile_keys[0]] = key_series.astype('category')
        for c in self.profile_keys[1:]:
            if c not in df.columns:
                df[c] = '1' if c == 'fcpNumber' else 'unknown'
            clean_col = df[c].astype(str).str.lower().str.strip()
            df[c] = clean_col.astype('category')
            key_series = key_series.str.cat(clean_col, sep='|')
        df['prof_key'] = key_series.astype('category')
        return df

    def _generate_poly_key_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groups erratic individual BINs into larger smoothing buckets; builds base_key and poly_key."""
        config = self.thermo_config
        default_reqs = config.get('DEFAULT', ['Company', 'rpgt'])
        df['base_key'] = df[default_reqs[0]].astype(str)
        for col in default_reqs[1:]:
            df['base_key'] = df['base_key'].str.cat(df[col].astype(str), sep='|')
        df['poly_key'] = df['base_key'].copy()

        for specific_rpgt, settings in config.items():
            if specific_rpgt == 'DEFAULT' or 'inherit_curve' in settings:
                continue
            level_cols = settings.get('level', default_reqs)
            groups = settings.get('groups', [])
            bin_to_group = {}
            for group in groups:
                g_name = "|" + group['name']
                for v in group['values']:
                    bin_to_group[str(v).split('.')[0].strip().lower()] = g_name

            mask_rpgt = (df['rpgt'].astype(str) == specific_rpgt.lower().strip())
            if not mask_rpgt.any() or not bin_to_group:
                continue

            matched_groups = df.loc[mask_rpgt, 'BIN'].astype(str).map(bin_to_group)
            valid_match = matched_groups.notna()

            if valid_match.any():
                final_idx = matched_groups.index[valid_match]
                subset_key = df.loc[final_idx, level_cols[0]].astype(str)
                for col in level_cols[1:]:
                    subset_key = subset_key.str.cat(df.loc[final_idx, col].astype(str), sep='|')
                df.loc[final_idx, 'poly_key'] = subset_key + matched_groups.loc[final_idx]

        df['base_key'] = df['base_key'].astype('category')
        df['poly_key'] = df['poly_key'].astype('category')
        return df

    def _map_prof_to_poly_strings(self) -> Dict[str, str]:
        prof_to_poly = pd.concat([self.longterm_fcast[['prof_key', 'poly_key']], self.fcast_data[['prof_key', 'poly_key']]], ignore_index=True).drop_duplicates(subset=['prof_key'])
        return {str(k): str(v) for k, v in prof_to_poly.set_index('prof_key')['poly_key'].items()}

    def _align_historical_periods(self) -> None:
        if 'period' in self.fcast_data.columns and self.fcast_data['period'].min() > 0:
            self.fcast_data['period'] -= self.fcast_data['period'].min()
        if 'period' in self.mapping_data.columns and self.mapping_data['period'].min() > 0:
            self.mapping_data['period'] -= self.mapping_data['period'].min()

    # =========================================================================
    # === REFERENCE CURVES (thermometer geometry, base rate, extrapolation factors)
    # =========================================================================

    def _get_curves(self, df: pd.DataFrame, key_col: str) -> Tuple[Dict[Any, Dict[int, float]], Dict[Any, float]]:
        mask = df['period'] <= self.sample_months
        df['time_to_event_months'] = df['time_to_event_months'].fillna(0).astype(int)
        raw = df[mask].groupby([key_col, 'time_to_event_months'], observed=True)['cb_count'].sum().reset_index()
        tot = df[mask].groupby([key_col], observed=True)['cb_count'].sum().reset_index()
        merged = pd.merge(raw, tot, on=key_col, suffixes=('', '_tot'))
        merged['pct'] = (merged['cb_count'] / merged['cb_count_tot']).fillna(0)

        curve_dict, vol_dict = {}, {}
        for k in merged[key_col].unique():
            subset = merged[merged[key_col] == k]
            curve_dict[k] = dict(zip(subset['time_to_event_months'], subset['pct']))
            vol_dict[k] = subset['cb_count_tot'].iloc[0]
        return curve_dict, vol_dict

    def _build_thermo_map(self) -> Dict[str, Dict[int, float]]:
        poly_curves, poly_vols = self._get_curves(self.fcast_data, 'poly_key_str')
        base_curves, _ = self._get_curves(self.fcast_data, 'base_key_str')

        final_thermo_map = {}
        for _, row in self.longterm_fcast[['poly_key_str', 'base_key_str', 'rpgt']].drop_duplicates().iterrows():
            pk, bk, current_rpgt = row['poly_key_str'], row['base_key_str'], str(row['rpgt']).strip().lower()
            borrowed_pk, borrowed_bk = pk, bk

            config_val = next((val for key, val in self.thermo_config.items() if key.lower() == current_rpgt), None)
            if isinstance(config_val, dict) and 'inherit_curve' in config_val:
                target_rpgt = config_val['inherit_curve'].strip().lower()
                borrowed_pk = "|".join([target_rpgt if p == current_rpgt else p for p in pk.split('|')])
                borrowed_bk = "|".join([target_rpgt if p == current_rpgt else p for p in bk.split('|')])

            curve = poly_curves.get(borrowed_pk, {})
            if poly_vols.get(borrowed_pk, 0) < 50 and borrowed_bk in base_curves:
                curve = base_curves[borrowed_bk]
            final_thermo_map[pk] = {t: curve.get(t, 0.0) for t in range(10)}

        return final_thermo_map

    def _calculate_rate_map(self) -> Dict[str, float]:
        lookback_t0 = self.t0_lookback_months
        decay_factor = self.decay_factor
        cb_t0_df = self.fcast_data[(self.fcast_data['period'] <= lookback_t0) & (self.fcast_data['time_to_event_months'] == 0)].copy()
        trx_t0_df = self.mapping_data[self.mapping_data['period'] <= lookback_t0].copy()
        trx_t0_df['poly_key_str'] = trx_t0_df['poly_key'].astype(str)

        cb_t0_df['decayed_cb'] = cb_t0_df['cb_count'] * (decay_factor ** cb_t0_df['period'])
        trx_t0_df['decayed_trx'] = trx_t0_df['mastercard_trx_count'] * (decay_factor ** trx_t0_df['period'])

        cb_t0_sum = cb_t0_df.groupby('poly_key_str')['decayed_cb'].sum()
        trx_sum = trx_t0_df.groupby('poly_key_str')['decayed_trx'].sum()

        return {pk: (cb_t0_sum.get(pk, 0) / trx_sum.get(pk, 0) if trx_sum.get(pk, 0) > 0 else 0.0)
                for pk in self.longterm_fcast['poly_key_str'].unique()}

    def _calculate_hist_extrap_map(self, final_thermo_map: Dict[str, Dict[int, float]], prof_to_poly_str_dict: Dict[str, str]) -> Dict[str, Dict[int, float]]:
        sample_months = self.sample_months
        hist_extrap_map = {}
        self.fcast_data['period_int'] = self.fcast_data['period'].fillna(0).astype(int)
        self.fcast_data['tte_int'] = self.fcast_data['time_to_event_months'].fillna(0).astype(int)
        cb_lookup = self.fcast_data[self.fcast_data['period_int'] <= sample_months].groupby(['prof_key_str', 'period_int', 'tte_int'])['cb_count'].sum().to_dict()

        for prf_str, pk_str in prof_to_poly_str_dict.items():
            curve = final_thermo_map.get(pk_str, {})
            extrap_factors = {}
            for T in range(1, 10):
                cb_sum, thermo_sum = 0.0, 0.0
                for i in range(T + 1):
                    if i <= sample_months:
                        cb_sum += cb_lookup.get((prf_str, i, T - i), 0.0)
                        thermo_sum += curve.get(T - i, 0.0)
                extrap_factors[T] = (cb_sum / thermo_sum) if thermo_sum > 0 else 0.0
            hist_extrap_map[prf_str] = extrap_factors

        return hist_extrap_map

    def _build_reference_curves(self, prof_to_poly_str_dict: Dict[str, str]):
        if self.load_curves_from_cache and os.path.exists(self.cache_file):
            logger.info("   > Loading reference curves from cache...")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
            return cached['final_thermo_map'], cached['rate_map'], cached['hist_extrap_map']

        final_thermo_map = self._build_thermo_map()
        rate_map = self._calculate_rate_map()
        hist_extrap_map = self._calculate_hist_extrap_map(final_thermo_map, prof_to_poly_str_dict)

        with open(self.cache_file, 'wb') as f:
            pickle.dump({'final_thermo_map': final_thermo_map, 'rate_map': rate_map, 'hist_extrap_map': hist_extrap_map}, f)

        return final_thermo_map, rate_map, hist_extrap_map

    # =========================================================================
    # === CB MAGNITUDE EXTRAPOLATION
    # =========================================================================

    def _calculate_actuarial_tails(self, fc_pivot, rate0_arr, curve_arrs, extrap_arrs, N) -> Dict[str, np.ndarray]:
        new_biz_tails, res_arrays, carryover_arrays = {}, {}, {}
        c0_arr = curve_arrs[0]

        for m_origin in range(6):
            vol = fc_pivot[m_origin].values.astype(np.float64) if m_origin in fc_pivot.columns else np.zeros(N, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                total_est = np.where(c0_arr > 0, (vol * rate0_arr) / c0_arr, vol * rate0_arr)
            new_biz_tails[m_origin] = {t: total_est * curve_arrs[t] for t in range(10)}

        for m_target in range(6):
            flex_ratio = np.float64(self.m_days[m_target] / 30.4167)

            for T in range(1, 10):
                if T + m_target <= 9:
                    raw_val = extrap_arrs[T] * curve_arrs[T + m_target]
                    res_arrays[f't{T + m_target}_fcast_m{m_target}'] = res_arrays.get(f't{T + m_target}_fcast_m{m_target}', np.zeros(N, dtype=np.float64)) + (raw_val * flex_ratio)
                    if m_target + 1 < 6 and T + m_target + 1 <= 9:
                        carryover_arrays[(m_target + 1, T + m_target + 1)] = carryover_arrays.get((m_target + 1, T + m_target + 1), np.zeros(N, dtype=np.float64)) + (raw_val - (raw_val * flex_ratio))

            for t in range(10):
                if m_target - t >= 0:
                    raw_val = new_biz_tails[m_target - t][t]
                    res_arrays[f't{t}_fcast_m{m_target}'] = res_arrays.get(f't{t}_fcast_m{m_target}', np.zeros(N, dtype=np.float64)) + (raw_val * flex_ratio)
                    if m_target + 1 < 6 and t + 1 <= 9:
                        carryover_arrays[(m_target + 1, t + 1)] = carryover_arrays.get((m_target + 1, t + 1), np.zeros(N, dtype=np.float64)) + (raw_val - (raw_val * flex_ratio))

            for t_age in range(10):
                if (m_target, t_age) in carryover_arrays:
                    res_arrays[f't{t_age}_fcast_m{m_target}'] = res_arrays.get(f't{t_age}_fcast_m{m_target}', np.zeros(N, dtype=np.float64)) + carryover_arrays[(m_target, t_age)]

        return res_arrays

    def _extrapolate_cb_magnitudes(self, prof_to_poly_str_dict, final_thermo_map, rate_map, hist_extrap_map) -> Tuple[pd.DataFrame, List[str]]:
        fc_pivot = self.longterm_fcast.groupby(['prof_key'] + self.profile_keys + ['month_offset'], observed=True)['forecasted_trx'].sum().unstack(fill_value=0).reset_index()
        fc_pivot.columns.name = None
        hist_profiles = self.fcast_data[['prof_key'] + self.profile_keys].drop_duplicates()
        fc_pivot = pd.concat([fc_pivot, hist_profiles], ignore_index=True).groupby(['prof_key'] + self.profile_keys, as_index=False, observed=True).sum()

        num_cols = [c for c in fc_pivot.columns if c not in ['prof_key'] + self.profile_keys]
        fc_pivot[num_cols] = fc_pivot[num_cols].fillna(0)

        N = len(fc_pivot)
        prof_keys_s = fc_pivot['prof_key'].astype(str)
        poly_keys_s = prof_keys_s.map(prof_to_poly_str_dict).fillna('unknown')
        rate0_arr = poly_keys_s.map(rate_map).fillna(0).values.astype(np.float64)

        curve_df = pd.DataFrame.from_dict(final_thermo_map, orient='index')
        curve_arrs = {t: poly_keys_s.map(curve_df[t]).fillna(0).values.astype(np.float64) if t in curve_df.columns else np.zeros(N, dtype=np.float64) for t in range(10)}
        extrap_df = pd.DataFrame.from_dict(hist_extrap_map, orient='index')
        extrap_arrs = {T: prof_keys_s.map(extrap_df[T]).fillna(0).values.astype(np.float64) if T in extrap_df.columns else np.zeros(N, dtype=np.float64) for T in range(1, 10)}

        res_arrays = self._calculate_actuarial_tails(fc_pivot, rate0_arr, curve_arrs, extrap_arrs, N)
        profile_cbs = pd.concat([fc_pivot[['prof_key']], pd.DataFrame(res_arrays, index=fc_pivot.index)], axis=1)

        cb_dist_cols = []
        for m in range(6):
            m_col = 'cb_fcast' if m == 0 else f'cb_fcast_m{m}'
            cols = [f't{t}_fcast_m{m}' for t in range(10) if f't{t}_fcast_m{m}' in profile_cbs.columns]
            profile_cbs[m_col] = sum([profile_cbs[c].values for c in cols]) if cols else 0.0
            cb_dist_cols.extend(cols + [m_col])

        return profile_cbs, cb_dist_cols

    # =========================================================================
    # === WATERFALL ROUTING (distribute granular CBs onto historical gateway mix)
    # =========================================================================

    def _build_waterfall_baselines(self, dist_keys: List[str]) -> pd.DataFrame:
        granular_pivot = self.longterm_fcast.groupby(dist_keys + ['month_offset'], observed=True)['forecasted_trx'].sum().unstack(fill_value=0).reset_index()
        granular_pivot.columns.name = None
        granular_pivot = granular_pivot.rename(columns={m: f'fc_mc_trx_m{m}' for m in range(6) if m in granular_pivot.columns})
        for m in range(6):
            if f'fc_mc_trx_m{m}' in granular_pivot.columns:
                granular_pivot[f'fc_mc_trx_m{m}'] = granular_pivot[f'fc_mc_trx_m{m}'].astype(np.float64)

        dist_base_hist = self.attempts_df.groupby(dist_keys, as_index=False, observed=True)['successCount'].sum()
        dist_base_hist['successCount'] = dist_base_hist['successCount'].astype(np.float64)

        hist_pivot = self.mapping_data.groupby(dist_keys + ['period'], observed=True)['mastercard_trx_count'].sum().unstack(fill_value=0).reset_index()
        hist_pivot.columns.name = None
        p_cols = [c for c in hist_pivot.columns if isinstance(c, (int, float))]
        hist_pivot = hist_pivot.rename(columns={p: f'p{int(p)}' for p in p_cols})
        for p in p_cols:
            hist_pivot[f'p{int(p)}'] = hist_pivot[f'p{int(p)}'].astype(np.float64)

        dist_base = pd.concat([granular_pivot, dist_base_hist, hist_pivot], ignore_index=True).groupby(dist_keys, as_index=False, observed=True).sum()
        num_cols_dist = [c for c in dist_base.columns if c not in dist_keys]
        dist_base[num_cols_dist] = dist_base[num_cols_dist].fillna(0)
        return dist_base

    def _determine_historical_baselines(self, dist_base: pd.DataFrame) -> pd.DataFrame:
        future_cols = [f'fc_mc_trx_m{m}' for m in range(6) if f'fc_mc_trx_m{m}' in dist_base.columns]
        dist_base['total_future_trx'] = dist_base[future_cols].sum(axis=1)

        p_str_cols = [c for c in dist_base.columns if re.match(r'^p\d+$', c)]
        dist_base['deep_hist_trx'] = dist_base[p_str_cols].sum(axis=1)

        dist_base['total_hist_baseline'] = np.where(
            dist_base['successCount'] > 0, dist_base['successCount'],
            np.where('p0' in dist_base.columns, dist_base['p0'], dist_base['deep_hist_trx'])
        )

        dist_base['prof_future_sum'] = dist_base.groupby('prof_key', observed=True)['total_future_trx'].transform('sum')
        dist_base['prof_hist_base_sum'] = dist_base.groupby('prof_key', observed=True)['total_hist_baseline'].transform('sum')
        dist_base['prof_deep_sum'] = dist_base.groupby('prof_key', observed=True)['deep_hist_trx'].transform('sum')
        return dist_base

    def _compute_forward_and_historical_shares(self, dist_base: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        share_cols = []
        dist_base['share_future_raw'] = np.where(
            dist_base['prof_future_sum'] > 0, dist_base['total_future_trx'],
            np.where(dist_base['prof_hist_base_sum'] > 0, dist_base['total_hist_baseline'], 1.0)
        )
        dist_base['share_future'] = dist_base['share_future_raw'] / dist_base.groupby('prof_key', observed=True)['share_future_raw'].transform('sum')
        share_cols.append('share_future')

        p_cols_exist = [f'p{p}' for p in range(10) if f'p{p}' in dist_base.columns]
        if p_cols_exist:
            prof_p_sums = dist_base.groupby('prof_key', observed=True)[p_cols_exist].transform('sum')
            raw_share_cols = []

            for p_col in p_cols_exist:
                raw_col = f'share_raw_{p_col}'
                dist_base[raw_col] = np.where(
                    prof_p_sums[p_col] > 0, dist_base[p_col],
                    np.where(dist_base['prof_deep_sum'] > 0, dist_base['deep_hist_trx'], 1.0)
                )
                raw_share_cols.append(raw_col)

            raw_share_sums = dist_base.groupby('prof_key', observed=True)[raw_share_cols].transform('sum')
            for p_col, raw_col in zip(p_cols_exist, raw_share_cols):
                share_col = f'share_{p_col}'
                dist_base[share_col] = dist_base[raw_col] / raw_share_sums[raw_col]
                share_cols.append(share_col)
            dist_base.drop(columns=raw_share_cols, inplace=True)
        return dist_base, share_cols

    def _finalize_share_normalization(self, dist_base: pd.DataFrame, share_cols: List[str]) -> pd.DataFrame:
        dist_base.drop(columns=['share_future_raw', 'total_future_trx', 'deep_hist_trx', 'total_hist_baseline', 'prof_future_sum', 'prof_hist_base_sum', 'prof_deep_sum'], errors='ignore', inplace=True)
        for sc in share_cols:
            dist_base[sc] = dist_base[sc] / dist_base.groupby('prof_key', observed=True)[sc].transform('sum')
            dist_base[sc] = dist_base[sc].fillna(0).astype(np.float64)
        return dist_base

    def _calculate_waterfall_shares(self, dist_base: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        dist_base = self._determine_historical_baselines(dist_base)
        dist_base, share_cols = self._compute_forward_and_historical_shares(dist_base)
        return self._finalize_share_normalization(dist_base, share_cols), share_cols

    def _rescue_unmapped_waterfall(self, dist_base, profile_cbs, share_cols) -> pd.DataFrame:
        unmapped_mask = ~profile_cbs['prof_key'].isin(dist_base['prof_key'])
        if unmapped_mask.any():
            unmapped_rows = profile_cbs.loc[unmapped_mask, ['prof_key']].copy()
            split_cols = unmapped_rows['prof_key'].astype(str).str.split('|', expand=True)
            for i, col in enumerate(self.profile_keys):
                unmapped_rows[col] = split_cols[i]
            unmapped_rows['attemptNumber'] = '1'

            valid_gw_hist = self.mapping_data.dropna(subset=['gatewayFid'])
            valid_gw_hist = valid_gw_hist[~valid_gw_hist['gatewayFid'].astype(str).str.lower().isin(['unknown', 'unmapped', 'nan'])]
            gw_recovery_map = valid_gw_hist.groupby('prof_key', observed=True)['gatewayFid'].first().to_dict()

            valid_gw_attempts = self.attempts_df.dropna(subset=['gatewayFid'])
            valid_gw_attempts = valid_gw_attempts[~valid_gw_attempts['gatewayFid'].astype(str).str.lower().isin(['unknown', 'unmapped', 'nan'])]
            gw_recovery_map.update(valid_gw_attempts.groupby('prof_key', observed=True)['gatewayFid'].first().to_dict())

            unmapped_rows['gatewayFid'] = unmapped_rows['prof_key'].map(gw_recovery_map)
            still_unmapped = unmapped_rows['gatewayFid'].isna()
            if still_unmapped.any():
                bin_to_gw = valid_gw_hist.groupby('BIN', observed=True)['gatewayFid'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).to_dict()
                unmapped_rows.loc[still_unmapped, 'gatewayFid'] = unmapped_rows.loc[still_unmapped, 'BIN'].map(bin_to_gw)

            unmapped_rows['gatewayFid'] = unmapped_rows['gatewayFid'].fillna('unmapped')
            unmapped_rows['share_future'] = 1.0
            for sc in share_cols:
                if sc != 'share_future':
                    unmapped_rows[sc] = 1.0
            dist_base = pd.concat([dist_base, unmapped_rows], ignore_index=True)
        return dist_base

    def _apply_magic_lock(self, dist_base, profile_cbs) -> pd.DataFrame:
        merged_profile = pd.merge(dist_base, profile_cbs, on=['prof_key'], how='left')
        matrix_cols = [f't{t}_fcast_m{m}' for m in range(6) for t in range(10) if f't{t}_fcast_m{m}' in merged_profile.columns]
        merged_profile[matrix_cols] = merged_profile[matrix_cols].fillna(0.0)

        for m in range(6):
            for t in range(10):
                c = f't{t}_fcast_m{m}'
                if c in merged_profile.columns:
                    origin_m = m - t
                    if origin_m >= 0:
                        base_share_col = 'share_future'
                    else:
                        period = abs(origin_m) - 1
                        base_share_col = f'share_p{period}' if f'share_p{period}' in merged_profile.columns else 'share_future'
                    merged_profile[c] = (merged_profile[c].values * merged_profile[base_share_col].values).astype(np.float64)

        for m in range(6):
            m_col = 'cb_fcast' if m == 0 else f'cb_fcast_m{m}'
            t_cols_for_m = [f't{t}_fcast_m{m}' for t in range(10) if f't{t}_fcast_m{m}' in merged_profile.columns]
            merged_profile[m_col] = sum([merged_profile[c].values for c in t_cols_for_m]) if t_cols_for_m else 0.0
        return merged_profile

    def _execute_waterfall_routing(self, profile_cbs, cb_dist_cols) -> pd.DataFrame:
        dist_keys = ['prof_key'] + self.profile_keys + ['gatewayFid']
        if 'attemptNumber' in self.longterm_fcast.columns:
            dist_keys.append('attemptNumber')

        dist_base = self._build_waterfall_baselines(dist_keys)
        dist_base, share_cols = self._calculate_waterfall_shares(dist_base)
        dist_base = self._rescue_unmapped_waterfall(dist_base, profile_cbs, share_cols)
        merged_profile = self._apply_magic_lock(dist_base, profile_cbs)

        vi_cols = [f'fc_mc_trx_m{m}' for m in range(6)]
        forecast_cols = cb_dist_cols + vi_cols
        for c in forecast_cols:
            if c not in merged_profile.columns:
                merged_profile[c] = 0.0

        cols_to_keep = [c for c in self.attempts_df.columns if c not in forecast_cols and c not in ['base_key', 'poly_key', 'prof_key']]
        clean_history = self.attempts_df[cols_to_keep].copy()

        dist_export_keys = [k for k in self.profile_keys]
        for extra_key in ['gatewayFid', 'attemptNumber', 'Country']:
            if (extra_key in merged_profile.columns or extra_key in self.attempts_df.columns) and extra_key not in dist_export_keys:
                dist_export_keys.append(extra_key)

        forecast_subset = merged_profile[dist_export_keys + forecast_cols].copy()
        for c in dist_export_keys:
            if c in clean_history.columns:
                clean_history[c] = clean_history[c].astype(object)
            if c in forecast_subset.columns:
                forecast_subset[c] = forecast_subset[c].astype(object)
        for c in forecast_cols:
            if c in forecast_subset.columns:
                forecast_subset[c] = forecast_subset[c].astype(np.float64)

        attempts_df = pd.concat([clean_history, forecast_subset], ignore_index=True, copy=False)
        for c in dist_export_keys:
            attempts_df[c] = attempts_df[c].astype('category')
        num_cols_to_sum = [c for c in attempts_df.columns if pd.api.types.is_numeric_dtype(attempts_df[c])]

        return attempts_df.groupby(dist_export_keys, observed=True, as_index=False)[num_cols_to_sum].sum()

    # =========================================================================
    # === THE MASTERCARD SHIFT
    # =========================================================================

    def _apply_mastercard_shift(self, attempts_df: pd.DataFrame, raw_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        🟢 THE MASTERCARD SHIFT: offset FP&A sales forward 1 month & inject real history as M0.
        Shifts fc_mc_trx_m{m-1} -> fc_mc_trx_m{m}, then replaces Month 0 with the last completed
        month's REAL Mastercard transactions (un-normalised back to raw counts).
        """
        logger.info("🔄 Shifting FP&A Transactions Forward 1 Month (Mastercard Logic)...")
        for m in range(5, 0, -1):
            old_col = f'fc_mc_trx_m{m-1}'
            new_col = f'fc_mc_trx_m{m}'
            attempts_df[new_col] = attempts_df[old_col] if old_col in attempts_df.columns else 0.0

        logger.info("📥 Injecting Historical Period 0 Sales as new Month 0...")
        hist_p1 = raw_mapping[raw_mapping['period'] == 0].copy()

        m0_dt = self.m0_start_dt
        p1_dt = m0_dt - pd.DateOffset(months=1)
        days_in_p1 = calendar.monthrange(p1_dt.year, p1_dt.month)[1]

        # Un-normalise Period 0 (last completed month) back to raw transactions
        hist_p1['mastercard_trx_count'] = (hist_p1['mastercard_trx_count'] * days_in_p1) / 30.4167

        target_comp_lower = str(self.company).strip().lower()
        if 'Company' in hist_p1.columns:
            mask_comp_p0 = hist_p1['Company'].astype(str).str.lower().str.strip() == target_comp_lower
            raw_company_vol = hist_p1.loc[mask_comp_p0, 'mastercard_trx_count'].sum()
            logger.info(f"      🔍 RAW PERIOD 0 MASTERCARD TRANSACTIONS FOR '{self.company.upper()}': {raw_company_vol:,.0f}")

        match_keys = ['Company', 'rpgt', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number', 'gatewayFid', 'fcpNumber', 'attemptNumber']
        grp_cols = [c for c in match_keys if c in hist_p1.columns and c in attempts_df.columns]

        def safe_clean_keys(df, cols):
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.lower().str.strip()
                    df[c] = df[c].str.replace(r'\.0$', '', regex=True)
            return df

        hist_p1 = safe_clean_keys(hist_p1, grp_cols)
        attempts_df = safe_clean_keys(attempts_df, grp_cols)

        p1_agg = hist_p1.groupby(grp_cols, as_index=False)['mastercard_trx_count'].sum()
        p1_agg.rename(columns={'mastercard_trx_count': 'new_m0'}, inplace=True)

        attempts_df = attempts_df.drop(columns=['fc_mc_trx_m0'], errors='ignore')
        attempts_df = pd.merge(attempts_df, p1_agg, on=grp_cols, how='outer')
        attempts_df['fc_mc_trx_m0'] = attempts_df['new_m0'].fillna(0).astype(np.float64)
        attempts_df.drop(columns=['new_m0'], inplace=True)

        num_cols_att = attempts_df.select_dtypes(include=['number']).columns
        attempts_df[num_cols_att] = attempts_df[num_cols_att].fillna(0)

        return attempts_df

    def _print_forecast_breakdown(self, attempts_df: pd.DataFrame) -> None:
        target_company_str = str(self.company).lower().strip()
        logger.info("=" * 50)
        logger.info(f"📈 TXN FORECAST BREAKDOWN FOR '{self.company}' (M0)")
        if attempts_df is not None and not attempts_df.empty and 'fc_mc_trx_m0' in attempts_df.columns:
            mask_comp = attempts_df['Company'].astype(str).str.lower().str.strip() == target_company_str
            if not mask_comp.any():
                logger.info(f"❌ No volume found in attempts_df for Company '{self.company}'.")
            else:
                rpgt_col = 'RPGT' if 'RPGT' in attempts_df.columns else 'rpgt'
                rpgt_breakdown = attempts_df.loc[mask_comp, [rpgt_col, 'fc_mc_trx_m0']].groupby(rpgt_col, observed=True)['fc_mc_trx_m0'].sum().reset_index()
                rpgt_breakdown = rpgt_breakdown[rpgt_breakdown['fc_mc_trx_m0'] > 0].sort_values('fc_mc_trx_m0', ascending=False)
                for _, row in rpgt_breakdown.iterrows():
                    logger.info(f"   > {str(row[rpgt_col]).title():<20} : {row['fc_mc_trx_m0']:,.0f}")
                logger.info(f"   🎯 TOTAL FOR '{self.company.upper()}' : {rpgt_breakdown['fc_mc_trx_m0'].sum():,.0f}")
        logger.info("=" * 50)

    # =========================================================================
    # === PUBLIC RUN ENTRYPOINT
    # =========================================================================

    def run_engine(self) -> pd.DataFrame:
        logger.info("🌡️ Generating pooled poly keys (fast vectorized mode)...")
        self.fcast_data = self._generate_poly_key_fast(self.fcast_data)
        self.mapping_data = self._generate_poly_key_fast(self.mapping_data)
        self.attempts_df = self._generate_poly_key_fast(self.attempts_df)
        self.longterm_fcast = self._fast_apply_keys(self.longterm_fcast)
        self.longterm_fcast = self._generate_poly_key_fast(self.longterm_fcast)

        # Keep an untouched copy of the historical mapping for the Mastercard Shift
        raw_mapping = self.mapping_data.copy()

        prof_to_poly_str_dict = self._map_prof_to_poly_strings()
        self._align_historical_periods()

        self.fcast_data['poly_key_str'], self.fcast_data['base_key_str'], self.fcast_data['prof_key_str'] = self.fcast_data['poly_key'].astype(str), self.fcast_data['base_key'].astype(str), self.fcast_data['prof_key'].astype(str)
        self.longterm_fcast['poly_key_str'], self.longterm_fcast['base_key_str'] = self.longterm_fcast['poly_key'].astype(str), self.longterm_fcast['base_key'].astype(str)

        logger.info("   ... Calculating reference curves & extrapolating CB magnitudes ...")
        final_thermo_map, rate_map, hist_extrap_map = self._build_reference_curves(prof_to_poly_str_dict)

        profile_cbs, cb_dist_cols = self._extrapolate_cb_magnitudes(prof_to_poly_str_dict, final_thermo_map, rate_map, hist_extrap_map)
        del self.fcast_data

        logger.info("   ... Distributing granular CBs to waterfall matrices ...")
        attempts_df = self._execute_waterfall_routing(profile_cbs, cb_dist_cols)

        self._print_forecast_breakdown(attempts_df)

        # 🟢 THE MASTERCARD SHIFT (offset sales & inject real history as M0)
        attempts_df = self._apply_mastercard_shift(attempts_df, raw_mapping)

        return attempts_df
