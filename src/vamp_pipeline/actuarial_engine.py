import os
import pickle
import calendar
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from .utils import setup_logger

logger = setup_logger(__name__)

class ActuarialEngine:
    """
    Handles Poly-Key generation, Thermometer Decay Curves, VAMP Extrapolation,
    and Waterfall Distribution linking future forecasts to historical bases.
    """

    def __init__(self, config: Dict[str, Any], fcast_data: pd.DataFrame, 
                 mapping_data: pd.DataFrame, longterm_fcast_pre: pd.DataFrame, 
                 attempts_df: pd.DataFrame):
        
        self.config = config
        self.thermo_config = config.get('thermometer_config', {})
        
        # State DataFrames
        self.fcast_data = fcast_data.copy()
        self.mapping_data = mapping_data.copy()
        self.longterm_fcast = longterm_fcast_pre.copy()
        self.attempts_df = attempts_df.copy()
        
        # Actuarial Constants
        self.m0_start_dt = pd.to_datetime(config['run_settings']['month_0_start_date'])
        self.t0_lookback_months = config.get('actuarial_settings', {}).get('t0_lookback_months', 0)
        self.decay_factor = config.get('actuarial_settings', {}).get('decay_factor', 0.5)
        self.sample_months = config.get('actuarial_settings', {}).get('thermometer_sample_months', 1)
        self.cache_file = os.path.join(
            config['paths']['cache_path'].format(
                month_var=config['run_settings']['month_var'], 
                company=config['run_settings']['company']
            ), 
            'reference_curves_cache.pkl'
        )
        
        # Core Keys
        self.profile_keys = ['Company', 'paymentMethodProvider', 'Country', 'rpgt', 'Currency', 'BIN', 'renewal_number', 'fcpNumber']
        self.m_days = {m: calendar.monthrange((self.m0_start_dt + pd.DateOffset(months=m)).year, 
                                              (self.m0_start_dt + pd.DateOffset(months=m)).month)[1] for m in range(6)}

    # =========================================================================
    # === 1. POLY KEYS & PERIOD ALIGNMENT
    # =========================================================================

    def _generate_poly_key_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groups erratic individual BINs into larger, smoother 'Buckets' for stable decay curves."""
        default_reqs = self.thermo_config.get('DEFAULT', ['Company', 'rpgt'])
        df['base_key'] = df[default_reqs[0]].astype(str)
        
        for col in default_reqs[1:]: 
            df['base_key'] = df['base_key'].str.cat(df[col].astype(str), sep='|')
        df['poly_key'] = df['base_key'].copy()

        for specific_rpgt, settings in self.thermo_config.items():
            if specific_rpgt == 'DEFAULT' or 'inherit_curve' in settings: 
                continue
                
            level_cols = settings.get('level', default_reqs)
            groups = settings.get('groups', [])
            bin_to_group = {"|" + group['name']: [str(v).split('.')[0].strip().lower() for v in group['values']] for group in groups}
            flattened_map = {v: k for k, values in bin_to_group.items() for v in values}

            mask_rpgt = (df['rpgt'].astype(str) == specific_rpgt.lower().strip())
            if mask_rpgt.any():
                matched_groups = df.loc[mask_rpgt, 'BIN'].astype(str).map(flattened_map)
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

    def _align_historical_periods(self) -> None:
        """Auto-shifts the period index so the most recently completed month reliably anchors at exactly 0."""
        if 'period' in self.fcast_data.columns and self.fcast_data['period'].min() > 0: 
            self.fcast_data['period'] -= self.fcast_data['period'].min()
        if 'period' in self.mapping_data.columns and self.mapping_data['period'].min() > 0: 
            self.mapping_data['period'] -= self.mapping_data['period'].min()

    # =========================================================================
    # === 2. REFERENCE CURVES & BASE RATES
    # =========================================================================

    def _get_curves(self, df: pd.DataFrame, key_col: str) -> Tuple[Dict[str, Dict[int, float]], Dict[str, float]]:
        """Builds the raw historical VAMP decay curve."""
        mask = df['period'] <= self.sample_months
        df['time_to_event_months'] = df['time_to_event_months'].fillna(0).astype(int)
        
        raw = df[mask].groupby([key_col, 'time_to_event_months'], observed=True)['vamp_count'].sum().reset_index()
        tot = df[mask].groupby([key_col], observed=True)['vamp_count'].sum().reset_index()
        merged = pd.merge(raw, tot, on=key_col, suffixes=('','_tot'))
        merged['pct'] = (merged['vamp_count'] / merged['vamp_count_tot']).fillna(0)

        curve_dict, vol_dict = {}, {}
        for k in merged[key_col].unique():
            subset = merged[merged[key_col] == k]
            curve_dict[k] = dict(zip(subset['time_to_event_months'], subset['pct']))
            vol_dict[k] = subset['vamp_count_tot'].iloc[0]
            
        return curve_dict, vol_dict

    def _build_thermo_map(self) -> Dict[str, Dict[int, float]]:
        """Constructs the final decay curves dictionary, handling proxy curve inheritance."""
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
        """Calculates the baseline VAMP-to-Transaction ratio at t=0, applying an exponential time-decay weight."""
        vamp_t0_df = self.fcast_data[(self.fcast_data['period'] <= self.t0_lookback_months) & (self.fcast_data['time_to_event_months'] == 0)].copy()
        trx_t0_df = self.mapping_data[self.mapping_data['period'] <= self.t0_lookback_months].copy()
        trx_t0_df['poly_key_str'] = trx_t0_df['poly_key'].astype(str)

        vamp_t0_df['decayed_vamp'] = vamp_t0_df['vamp_count'] * (self.decay_factor ** vamp_t0_df['period'])
        trx_t0_df['decayed_trx'] = trx_t0_df['visa_trx_count'] * (self.decay_factor ** trx_t0_df['period'])

        vamp_t0_sum = vamp_t0_df.groupby('poly_key_str')['decayed_vamp'].sum()
        trx_sum = trx_t0_df.groupby('poly_key_str')['decayed_trx'].sum()
        
        return {pk: (vamp_t0_sum.get(pk, 0) / trx_sum.get(pk, 0) if trx_sum.get(pk, 0) > 0 else 0.0) 
                for pk in self.longterm_fcast['poly_key_str'].unique()}

    def _calculate_hist_extrap_map(self, final_thermo_map: Dict[str, Any], prof_to_poly_str_dict: Dict[str, str]) -> Dict[str, Dict[int, float]]:
        """Generates extrapolation factors by comparing historical tails to predicted tails."""
        hist_extrap_map = {}
        self.fcast_data['period_int'] = self.fcast_data['period'].fillna(0).astype(int)
        self.fcast_data['tte_int'] = self.fcast_data['time_to_event_months'].fillna(0).astype(int)
        vamp_lookup = self.fcast_data[self.fcast_data['period_int'] <= self.sample_months].groupby(['prof_key_str', 'period_int', 'tte_int'])['vamp_count'].sum().to_dict()

        for prf_str, pk_str in prof_to_poly_str_dict.items():
            curve = final_thermo_map.get(pk_str, {})
            extrap_factors = {}
            for T in range(1, 10):
                vamp_sum, thermo_sum = 0.0, 0.0
                for i in range(T + 1):
                    if i <= self.sample_months:
                        vamp_sum += vamp_lookup.get((prf_str, i, T - i), 0.0)
                        thermo_sum += curve.get(T - i, 0.0)
                extrap_factors[T] = (vamp_sum / thermo_sum) if thermo_sum > 0 else 0.0
            hist_extrap_map[prf_str] = extrap_factors
            
        return hist_extrap_map

    # =========================================================================
    # === 3. ACTUARIAL MATH & EXTRAPOLATION
    # =========================================================================

    def _calculate_actuarial_tails(self, fc_pivot: pd.DataFrame, rate0_arr: np.ndarray, curve_arrs: Dict[int, np.ndarray], extrap_arrs: Dict[int, np.ndarray], N: int) -> Dict[str, np.ndarray]:
        """Calculates expected VAMPs for both New Business and Historical Carryover."""
        new_biz_tails, res_arrays, carryover_arrays = {}, {}, {}
        c0_arr = curve_arrs[0]

        for m_origin in range(6):
            # 🟢 UPGRADED TO FLOAT64
            vol = fc_pivot[m_origin].values.astype(np.float64) if m_origin in fc_pivot.columns else np.zeros(N, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                total_est = np.where(c0_arr > 0, (vol * rate0_arr) / c0_arr, vol * rate0_arr)
            new_biz_tails[m_origin] = {t: total_est * curve_arrs[t] for t in range(10)}

        for m_target in range(6):
            # 🟢 UPGRADED TO FLOAT64
            flex_ratio = np.float64(self.m_days[m_target] / 30.4167)
            
            # Historical Carryover
            for T in range(1, 10):
                if T + m_target <= 9:
                    raw_val = extrap_arrs[T] * curve_arrs[T + m_target]
                    # 🟢 UPGRADED TO FLOAT64
                    res_arrays[f't{T + m_target}_fcast_m{m_target}'] = res_arrays.get(f't{T + m_target}_fcast_m{m_target}', np.zeros(N, dtype=np.float64)) + (raw_val * flex_ratio)
                    if m_target + 1 < 6 and T + m_target + 1 <= 9:
                        carryover_arrays[(m_target + 1, T + m_target + 1)] = carryover_arrays.get((m_target + 1, T + m_target + 1), np.zeros(N, dtype=np.float64)) + (raw_val - (raw_val * flex_ratio))

            # New Business
            for t in range(10):
                if m_target - t >= 0:
                    raw_val = new_biz_tails[m_target - t][t]
                    # 🟢 UPGRADED TO FLOAT64
                    res_arrays[f't{t}_fcast_m{m_target}'] = res_arrays.get(f't{t}_fcast_m{m_target}', np.zeros(N, dtype=np.float64)) + (raw_val * flex_ratio)
                    if m_target + 1 < 6 and t + 1 <= 9:
                        carryover_arrays[(m_target + 1, t + 1)] = carryover_arrays.get((m_target + 1, t + 1), np.zeros(N, dtype=np.float64)) + (raw_val - (raw_val * flex_ratio))

            # Absorb Carryover
            for t_age in range(10):
                if (m_target, t_age) in carryover_arrays:
                    # 🟢 UPGRADED TO FLOAT64
                    res_arrays[f't{t_age}_fcast_m{m_target}'] = res_arrays.get(f't{t_age}_fcast_m{m_target}', np.zeros(N, dtype=np.float64)) + carryover_arrays[(m_target, t_age)]
                    
        return res_arrays

    def _extrapolate_vamp_magnitudes(self, prof_to_poly_str_dict: Dict[str, str], final_thermo_map: Dict[str, Any], rate_map: Dict[str, float], hist_extrap_map: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Drops FP&A sales down the decay curves to calculate exact Lifetime VAMPs."""
        fc_pivot = self.longterm_fcast.groupby(['prof_key'] + self.profile_keys + ['month_offset'], observed=True)['forecasted_trx'].sum().unstack(fill_value=0).reset_index()
        fc_pivot.columns.name = None
        hist_profiles = self.fcast_data[['prof_key'] + self.profile_keys].drop_duplicates()
        fc_pivot = pd.concat([fc_pivot, hist_profiles], ignore_index=True).groupby(['prof_key'] + self.profile_keys, as_index=False, observed=True).sum()

        num_cols = [c for c in fc_pivot.columns if c not in ['prof_key'] + self.profile_keys]
        fc_pivot[num_cols] = fc_pivot[num_cols].fillna(0)

        N = len(fc_pivot)
        prof_keys_s = fc_pivot['prof_key'].astype(str)
        poly_keys_s = prof_keys_s.map(prof_to_poly_str_dict).fillna('unknown')
        
        # 🟢 UPGRADED TO FLOAT64
        rate0_arr = poly_keys_s.map(rate_map).fillna(0).values.astype(np.float64)

        curve_df = pd.DataFrame.from_dict(final_thermo_map, orient='index')
        # 🟢 UPGRADED TO FLOAT64
        curve_arrs = {t: poly_keys_s.map(curve_df[t]).fillna(0).values.astype(np.float64) if t in curve_df.columns else np.zeros(N, dtype=np.float64) for t in range(10)}
        extrap_df = pd.DataFrame.from_dict(hist_extrap_map, orient='index')
        extrap_arrs = {T: prof_keys_s.map(extrap_df[T]).fillna(0).values.astype(np.float64) if T in extrap_df.columns else np.zeros(N, dtype=np.float64) for T in range(1, 10)}

        res_arrays = self._calculate_actuarial_tails(fc_pivot, rate0_arr, curve_arrs, extrap_arrs, N)
        profile_vamps = pd.concat([fc_pivot[['prof_key']], pd.DataFrame(res_arrays, index=fc_pivot.index)], axis=1)

        vamp_dist_cols = []
        for m in range(6):
            m_col = 'vamp_fcast' if m == 0 else f'vamp_fcast_m{m}'
            cols = [f't{t}_fcast_m{m}' for t in range(10) if f't{t}_fcast_m{m}' in profile_vamps.columns]
            profile_vamps[m_col] = sum([profile_vamps[c].values for c in cols]) if cols else 0.0
            vamp_dist_cols.extend(cols + [m_col])
            
        return profile_vamps, vamp_dist_cols

    # =========================================================================
    # === 4. WATERFALL ROUTING (HISTORICAL DISTRIBUTION)
    # =========================================================================

    def _build_waterfall_baselines(self, dist_keys: List[str]) -> pd.DataFrame:
        """Pivots granular forecast and historical actuals into a unified baseline."""
        granular_pivot = self.longterm_fcast.groupby(dist_keys + ['month_offset'], observed=True)['forecasted_trx'].sum().unstack(fill_value=0).reset_index()
        granular_pivot.columns.name = None
        granular_pivot = granular_pivot.rename(columns={m: f'fc_vi_trx_m{m}' for m in range(6) if m in granular_pivot.columns})
        
        # 🟢 UPGRADED TO FLOAT64
        for m in range(6):
            if f'fc_vi_trx_m{m}' in granular_pivot.columns: 
                granular_pivot[f'fc_vi_trx_m{m}'] = granular_pivot[f'fc_vi_trx_m{m}'].astype(np.float64)

        dist_base_hist = self.attempts_df.groupby(dist_keys, as_index=False, observed=True)['successCount'].sum()
        dist_base_hist['successCount'] = dist_base_hist['successCount'].astype(np.float64)

        hist_pivot = self.mapping_data.groupby(dist_keys + ['period'], observed=True)['visa_trx_count'].sum().unstack(fill_value=0).reset_index()
        hist_pivot.columns.name = None
        p_cols = [c for c in hist_pivot.columns if isinstance(c, (int, float))]
        hist_pivot = hist_pivot.rename(columns={p: f'p{int(p)}' for p in p_cols})
        for p in p_cols: 
            hist_pivot[f'p{int(p)}'] = hist_pivot[f'p{int(p)}'].astype(np.float64)

        dist_base = pd.concat([granular_pivot, dist_base_hist, hist_pivot], ignore_index=True).groupby(dist_keys, as_index=False, observed=True).sum()
        num_cols_dist = [c for c in dist_base.columns if c not in dist_keys]
        dist_base[num_cols_dist] = dist_base[num_cols_dist].fillna(0)
        return dist_base

    def _calculate_waterfall_shares(self, dist_base: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Determines what percentage of volume belongs to each historical gateway."""
        future_cols = [f'fc_vi_trx_m{m}' for m in range(6) if f'fc_vi_trx_m{m}' in dist_base.columns]
        dist_base['total_future_trx'] = dist_base[future_cols].sum(axis=1)
        p_str_cols = [c for c in dist_base.columns if re.match(r'^p\d+$', c)]
        dist_base['deep_hist_trx'] = dist_base[p_str_cols].sum(axis=1)

        dist_base['total_hist_baseline'] = np.where(dist_base['successCount'] > 0, dist_base['successCount'], 
                                                    np.where('p0' in dist_base.columns, dist_base['p0'], dist_base['deep_hist_trx']))
        
        dist_base['prof_future_sum'] = dist_base.groupby('prof_key', observed=True)['total_future_trx'].transform('sum')
        dist_base['prof_hist_base_sum'] = dist_base.groupby('prof_key', observed=True)['total_hist_baseline'].transform('sum')
        dist_base['prof_deep_sum'] = dist_base.groupby('prof_key', observed=True)['deep_hist_trx'].transform('sum')

        share_cols = []
        dist_base['share_future_raw'] = np.where(dist_base['prof_future_sum'] > 0, dist_base['total_future_trx'], 
                                                 np.where(dist_base['prof_hist_base_sum'] > 0, dist_base['total_hist_baseline'], 1.0))
        dist_base['share_future'] = dist_base['share_future_raw'] / dist_base.groupby('prof_key', observed=True)['share_future_raw'].transform('sum')
        share_cols.append('share_future')

        p_cols_exist = [f'p{p}' for p in range(10) if f'p{p}' in dist_base.columns]
        if p_cols_exist:
            prof_p_sums = dist_base.groupby('prof_key', observed=True)[p_cols_exist].transform('sum')
            raw_share_cols = []
            for p_col in p_cols_exist:
                raw_col = f'share_raw_{p_col}'
                dist_base[raw_col] = np.where(prof_p_sums[p_col] > 0, dist_base[p_col], 
                                              np.where(dist_base['prof_deep_sum'] > 0, dist_base['deep_hist_trx'], 1.0))
                raw_share_cols.append(raw_col)

            raw_share_sums = dist_base.groupby('prof_key', observed=True)[raw_share_cols].transform('sum')
            for p_col, raw_col in zip(p_cols_exist, raw_share_cols):
                share_col = f'share_{p_col}'
                dist_base[share_col] = dist_base[raw_col] / raw_share_sums[raw_col]
                share_cols.append(share_col)
            dist_base.drop(columns=raw_share_cols, inplace=True)

        dist_base.drop(columns=['share_future_raw', 'total_future_trx', 'deep_hist_trx', 'total_hist_baseline', 'prof_future_sum', 'prof_hist_base_sum', 'prof_deep_sum'], errors='ignore', inplace=True)

        for sc in share_cols:
            dist_base[sc] = dist_base[sc] / dist_base.groupby('prof_key', observed=True)[sc].transform('sum')
            # 🟢 UPGRADED TO FLOAT64
            dist_base[sc] = dist_base[sc].fillna(0).astype(np.float64)
            
        return dist_base, share_cols

    def _rescue_unmapped_waterfall(self, dist_base: pd.DataFrame, profile_vamps: pd.DataFrame, share_cols: List[str]) -> pd.DataFrame:
        """Catches purely new cohorts and maps them using BIN fallbacks."""
        unmapped_mask = ~profile_vamps['prof_key'].isin(dist_base['prof_key'])
        if unmapped_mask.any():
            unmapped_rows = profile_vamps.loc[unmapped_mask, ['prof_key']].copy()
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
                if sc != 'share_future': unmapped_rows[sc] = 1.0
            dist_base = pd.concat([dist_base, unmapped_rows], ignore_index=True)
            
        return dist_base

    def _apply_magic_lock(self, dist_base: pd.DataFrame, profile_vamps: pd.DataFrame) -> pd.DataFrame:
        """Enforces chronological integrity, pinning past VAMPs to past routing."""
        merged_profile = pd.merge(dist_base, profile_vamps, on=['prof_key'], how='left')
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
                    # 🟢 UPGRADED TO FLOAT64
                    merged_profile[c] = (merged_profile[c].values * merged_profile[base_share_col].values).astype(np.float64)

        for m in range(6):
            m_col = 'vamp_fcast' if m == 0 else f'vamp_fcast_m{m}'
            t_cols_for_m = [f't{t}_fcast_m{m}' for t in range(10) if f't{t}_fcast_m{m}' in merged_profile.columns]
            merged_profile[m_col] = sum([merged_profile[c].values for c in t_cols_for_m]) if t_cols_for_m else 0.0
            
        return merged_profile

    def _execute_waterfall_routing(self, profile_vamps: pd.DataFrame, vamp_dist_cols: List[str]) -> pd.DataFrame:
        """Links VAMP forecasts to historical profiles and rescues unmapped rows."""
        dist_keys = ['prof_key'] + self.profile_keys + ['gatewayFid']
        if 'attemptNumber' in self.longterm_fcast.columns: 
            dist_keys.append('attemptNumber')

        dist_base = self._build_waterfall_baselines(dist_keys)
        dist_base, share_cols = self._calculate_waterfall_shares(dist_base)
        dist_base = self._rescue_unmapped_waterfall(dist_base, profile_vamps, share_cols)
        merged_profile = self._apply_magic_lock(dist_base, profile_vamps)

        vi_cols = [f'fc_vi_trx_m{m}' for m in range(6)]
        forecast_cols = vamp_dist_cols + vi_cols
        for c in forecast_cols:
            if c not in merged_profile.columns: merged_profile[c] = 0.0

        cols_to_keep = [c for c in self.attempts_df.columns if c not in forecast_cols and c not in ['base_key', 'poly_key', 'prof_key']]
        clean_history = self.attempts_df[cols_to_keep].copy()

        dist_export_keys = [k for k in self.profile_keys]
        for extra_key in ['gatewayFid', 'attemptNumber', 'Country']:
            if (extra_key in merged_profile.columns or extra_key in self.attempts_df.columns) and extra_key not in dist_export_keys:
                dist_export_keys.append(extra_key)

        forecast_subset = merged_profile[dist_export_keys + forecast_cols].copy()
        for c in dist_export_keys:
            if c in clean_history.columns: clean_history[c] = clean_history[c].astype(object)
            if c in forecast_subset.columns: forecast_subset[c] = forecast_subset[c].astype(object)
        for c in forecast_cols:
            # 🟢 UPGRADED TO FLOAT64
            if c in forecast_subset.columns: forecast_subset[c] = forecast_subset[c].astype(np.float64)

        final_attempts = pd.concat([clean_history, forecast_subset], ignore_index=True, copy=False)
        for c in dist_export_keys: 
            final_attempts[c] = final_attempts[c].astype('category')
            
        num_cols_to_sum = [c for c in final_attempts.columns if pd.api.types.is_numeric_dtype(final_attempts[c])]
        return final_attempts.groupby(dist_export_keys, observed=True, as_index=False)[num_cols_to_sum].sum()

    # =========================================================================
    # === 5. ORCHESTRATOR
    # =========================================================================

    def run_engine(self) -> pd.DataFrame:
        """
        MANAGER: Orchestrates the entire Section 7 logic. Generates Poly Keys, 
        Builds Actuarial Curves, Extrapolates Lifetime VAMPs, and routes them 
        into the final Waterfall matrix.
        """
        logger.info("Generating Pooled Poly Keys (Fast Vectorized Mode)...")
        self.fcast_data = self._generate_poly_key_fast(self.fcast_data)
        self.mapping_data = self._generate_poly_key_fast(self.mapping_data)
        self.attempts_df = self._generate_poly_key_fast(self.attempts_df)
        self.longterm_fcast = self._generate_poly_key_fast(self.longterm_fcast)
        
        # Link profiles and align periods
        prof_to_poly = pd.concat([
            self.longterm_fcast[['prof_key', 'poly_key']], 
            self.fcast_data[['prof_key', 'poly_key']]
        ], ignore_index=True).drop_duplicates(subset=['prof_key'])
        prof_to_poly_str_dict = {str(k): str(v) for k, v in prof_to_poly.set_index('prof_key')['poly_key'].items()}
        
        self._align_historical_periods()

        logger.info("Calculating Reference Curves & Extrapolating Magnitudes...")
        self.fcast_data['poly_key_str'] = self.fcast_data['poly_key'].astype(str)
        self.fcast_data['base_key_str'] = self.fcast_data['base_key'].astype(str)
        self.fcast_data['prof_key_str'] = self.fcast_data['prof_key'].astype(str)
        self.longterm_fcast['poly_key_str'] = self.longterm_fcast['poly_key'].astype(str)
        self.longterm_fcast['base_key_str'] = self.longterm_fcast['base_key'].astype(str)

        if self.config['run_settings'].get('load_curves_from_cache', True) and os.path.exists(self.cache_file):
            logger.info("   > Loaded Actuarial Curves from Cache.")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
            final_thermo_map, rate_map, hist_extrap_map = cached['final_thermo_map'], cached['rate_map'], cached['hist_extrap_map']
        else:
            final_thermo_map = self._build_thermo_map()
            rate_map = self._calculate_rate_map()
            hist_extrap_map = self._calculate_hist_extrap_map(final_thermo_map, prof_to_poly_str_dict)
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'final_thermo_map': final_thermo_map, 'rate_map': rate_map, 'hist_extrap_map': hist_extrap_map}, f)

        # Extrapolate and Route
        profile_vamps, vamp_dist_cols = self._extrapolate_vamp_magnitudes(prof_to_poly_str_dict, final_thermo_map, rate_map, hist_extrap_map)
        
        logger.info("Distributing Granular VAMPs to Waterfall Matrices...")
        final_attempts_df = self._execute_waterfall_routing(profile_vamps, vamp_dist_cols)
        
        logger.info("✅ Actuarial Engine Complete.")
        
        return final_attempts_df