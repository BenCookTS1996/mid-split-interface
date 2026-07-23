import pandas as pd
import numpy as np
import re
import calendar
from typing import Dict, Any, Tuple, List, Optional

from .utils import setup_logger, clean_key_col

logger = setup_logger(__name__)

__build__ = "2026-07-21-alloc-trace-rawsplit+profile-samples"


def _doomed_keys(pre_totals: pd.Series, pre_deads: pd.Series) -> pd.Index:
    """MultiIndex keys whose ENTIRE volume is dead (total count == dead count).

    Avoids ``pre_deads.reindex(pre_totals.index)`` on the MultiIndex: older pandas
    (the Python 3.8 build) raises ``ValueError: Buffer dtype mismatch, expected
    'const int8_t' but got 'short'`` when aligning two MultiIndexes whose level
    codes have different integer widths (int8 when a level has <=128 categories,
    int16/'short' otherwise). This shows up once switch-offs create dead gateways
    and the two groupby indexes end up with different level cardinalities. A plain
    dict lookup keyed on the index tuples sidesteps that hashtable path entirely
    and is exactly equivalent to the old reindex(fill_value=0) comparison."""
    dead_map = pre_deads.to_dict()
    tot = pre_totals.to_numpy()
    dead = np.fromiter((dead_map.get(k, 0) for k in pre_totals.index),
                       dtype=tot.dtype, count=len(pre_totals))
    return pre_totals.index[tot == dead]


class AllocationEngine:
    """
    Handles Time-Aware Routing, Mid-Month Snapshots, Volume Stealing, 
    and Lossless VAMP Redistribution.
    """

    def __init__(self, config: Dict[str, Any], attempts_df: pd.DataFrame, split_df: pd.DataFrame, mr_weights: Optional[Dict[int, Dict[int, float]]] = None):
        self.config = config
        self.attempts_df = attempts_df
        self.split_df = split_df
        self.mr_daily_weights = mr_weights or {}
        
        # System State
        self.overrides = config.get('gateway_volume_overrides', {})
        self.m0_start = pd.to_datetime(config['run_settings']['month_0_start_date'])
        self.target_dates = {m: self.m0_start + pd.DateOffset(months=m) for m in range(6)}

        # Determine structural columns automatically
        self.t_cols = [c for c in attempts_df.columns if re.match(r'^t\d+_fcast_m\d+$', c)]
        self.vi_cols = [c for c in attempts_df.columns if re.match(r'^fc_vi_trx_m\d+$', c)]
        self.valid_agg_cols = self.t_cols + self.vi_cols

        self.map_std = {
            'company': 'Company', 'Brand': 'Company', 'brand': 'Company', 
            'gateway_fid': 'gatewayFid', 'Gateway': 'gatewayFid', 'gateway': 'gatewayFid', 
            'riskDefinedProductSubscriptionType': 'RPGT', 'rpgt': 'RPGT', 
            'paymentmethodprovider': 'paymentMethodProvider', 'country': 'Country', 
            'bin': 'BIN', 'currency': 'Currency', 'STICKY': 'renewal_number', 'sticky': 'renewal_number'
        }
        self.join_keys = ['Company', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number']

    # =========================================================================
    # === 1. MATRIX PREPARATION & TIMELINES
    # =========================================================================

    def _prepare_allocation_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans and standardizes the historical attempts and the Google Sheets rules."""
        cols_to_extract = list(set(list(self.map_std.keys()) + list(self.map_std.values()) + self.join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber'] + self.valid_agg_cols).intersection(self.attempts_df.columns))
        df_in = self.attempts_df[cols_to_extract].rename(columns=self.map_std)

        for c in self.valid_agg_cols:
            if c in df_in.columns: 
                # 🟢 UPGRADED TO FLOAT64
                df_in[c] = df_in[c].astype(np.float64)

        split_work = self.split_df.rename(columns=self.map_std).copy() if not self.split_df.empty else pd.DataFrame(columns=self.join_keys + ['GO LIVE', 'gatewayFid', 'Share'])
        
        if 'Share' not in split_work.columns: split_work['Share'] = 0.0
        if 'GO LIVE' not in split_work.columns: split_work['GO LIVE'] = pd.to_datetime('2020-01-01')
        if 'gatewayFid' not in split_work.columns: split_work['gatewayFid'] = 'unmapped'
        
        for c in self.join_keys:
            if c not in split_work.columns: split_work[c] = 'unknown'

        for col in self.join_keys:
            df_in[col] = clean_key_col(df_in[col], remove_dot_zero=True)
            split_work[col] = clean_key_col(split_work[col], remove_dot_zero=True)

        df_in['gatewayFid'] = df_in['gatewayFid'].astype('category')
        split_work['gatewayFid'] = split_work['gatewayFid'].astype('category')
        
        df_in['fcpNumber'] = clean_key_col(df_in['fcpNumber']) if 'fcpNumber' in df_in.columns else '1'
        df_in['attemptNumber'] = clean_key_col(df_in['attemptNumber']) if 'attemptNumber' in df_in.columns else '1'

        split_work['GO LIVE'] = pd.to_datetime(split_work['GO LIVE'], errors='coerce', dayfirst=True).fillna(pd.Timestamp('2020-01-01'))
        split_work['Share'] = pd.to_numeric(split_work['Share'], errors='coerce').fillna(0)
        split_work = split_work.groupby(self.join_keys + ['GO LIVE', 'gatewayFid'], observed=True)['Share'].sum().reset_index()
        split_work['Share_Vamp'] = split_work['Share'].copy()
        
        return df_in, split_work

    def _inject_dynamic_snapshots(self, split_work: pd.DataFrame) -> pd.DataFrame:
        """Handles mid-month 'death' rules for gateways, creating dynamic date-based snapshots."""
        if not self.overrides: 
            return self._normalize_shares(split_work)

        immediate_trx, immediate_vamp, future_kills = [], [], []
        for fid, cfg in self.overrides.items():
            if isinstance(cfg, dict) and cfg.get('target', 0) == 0:
                gw, app, eff_date = str(fid).strip().lower(), cfg.get('apply_to', 'both'), cfg.get('effective_date')
                if eff_date: 
                    future_kills.append((gw, app, pd.to_datetime(eff_date)))
                else:
                    if app in ['trx', 'both']: immediate_trx.append(gw)
                    if app in ['vamp', 'both']: immediate_vamp.append(gw)

        if immediate_trx: 
            split_work.loc[split_work['gatewayFid'].astype(str).str.lower().isin(immediate_trx), 'Share'] = 0.0
        if immediate_vamp: 
            split_work.loc[split_work['gatewayFid'].astype(str).str.lower().isin(immediate_vamp), 'Share_Vamp'] = 0.0

        if future_kills:
            for eff_dt in sorted(list(set([d for _, _, d in future_kills]))):
                split_work = split_work.sort_values(self.join_keys + ['GO LIVE'])
                split_work['Next_GO_LIVE'] = split_work.groupby(self.join_keys, observed=True)['GO LIVE'].shift(-1)
                
                gw_list = [k[0] for k in [(gw, app) for gw, app, d in future_kills if d == eff_dt]]
                mask_active = (split_work['GO LIVE'] <= eff_dt) & (split_work['Next_GO_LIVE'].isna() | (split_work['Next_GO_LIVE'] > eff_dt))
                profiles_affected = split_work.loc[mask_active & split_work['gatewayFid'].astype(str).str.lower().isin(gw_list), self.join_keys].drop_duplicates()

                if not profiles_affected.empty:
                    snapshot_df = pd.merge(split_work[mask_active], profiles_affected, on=self.join_keys, how='inner')
                    snapshot_df['GO LIVE'] = eff_dt
                    for gw, app in [(gw, app) for gw, app, d in future_kills if d == eff_dt]:
                        mask_target_gw = snapshot_df['gatewayFid'].astype(str).str.lower() == gw
                        if app in ['trx', 'both']: snapshot_df.loc[mask_target_gw, 'Share'] = 0.0
                        if app in ['vamp', 'both']: snapshot_df.loc[mask_target_gw, 'Share_Vamp'] = 0.0
                    split_work = pd.concat([split_work.drop(columns=['Next_GO_LIVE']), snapshot_df], ignore_index=True)
                    split_work = split_work.drop_duplicates(subset=self.join_keys + ['GO LIVE', 'gatewayFid'], keep='last')
                else:
                    split_work = split_work.drop(columns=['Next_GO_LIVE'])

        return self._normalize_shares(split_work)

    def _normalize_shares(self, split_work: pd.DataFrame) -> pd.DataFrame:
        """Recalculates routing fractions so they always equal exactly 1.0 (100%)."""
        for s_col, n_col in [('Share', 'Share_Norm'), ('Share_Vamp', 'Share_Norm_Vamp')]:
            total = split_work.groupby(self.join_keys + ['GO LIVE'], observed=True)[s_col].transform('sum')
            split_work[n_col] = np.where(total > 0, split_work[s_col] / total, 0.0)
        return split_work

    def _stitch_timeline(self, split_work: pd.DataFrame) -> pd.DataFrame:
        """Flawlessly links overlapping Google Sheet rules chronologically."""
        unique_dates = split_work[self.join_keys + ['GO LIVE']].drop_duplicates().sort_values(self.join_keys + ['GO LIVE']).reset_index(drop=True)
        if not unique_dates.empty:
            unique_dates['Next_GO_LIVE'] = unique_dates['GO LIVE'].shift(-1)
            mask_same_group = (unique_dates[self.join_keys] == unique_dates[self.join_keys].shift(-1)).all(axis=1)
            unique_dates.loc[~mask_same_group, 'Next_GO_LIVE'] = pd.NaT
        else:
            unique_dates['Next_GO_LIVE'] = pd.Series(dtype='datetime64[ns]')
        return pd.merge(split_work.drop(columns=['Next_GO_LIVE'], errors='ignore'), unique_dates, on=self.join_keys + ['GO LIVE'], how='left')

    def _map_and_filter_cohorts(self, df_in: pd.DataFrame, split_work: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Explicitly splits the data into mapped profiles and unmapped fallbacks."""
        chunk_agg = df_in.groupby(self.join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber'], as_index=False, observed=True)[self.valid_agg_cols].sum()
        unique_splits = split_work[self.join_keys].drop_duplicates().copy()
        unique_splits['_is_mapped'] = True
        chunk_agg = chunk_agg.merge(unique_splits, on=self.join_keys, how='left')

        mapped_mask = (chunk_agg['fcpNumber'] == '1') & (chunk_agg['_is_mapped'] == True)
        is_restricted_rpgt = chunk_agg['RPGT'].astype(str).str.lower().isin(['monthly initial', 'annual sub sale', 'upgrades'])
        mapped_mask = mapped_mask & (~is_restricted_rpgt | (is_restricted_rpgt & (chunk_agg['attemptNumber'] == '1')))

        mapped_agg = chunk_agg[mapped_mask].drop(columns=['_is_mapped']).copy()
        unmapped_agg = chunk_agg[~mapped_mask].drop(columns=['_is_mapped']).copy()
        return mapped_agg, unmapped_agg

    # =========================================================================
    # === 2. VECTOR MATH & CHUNK PROCESSING
    # =========================================================================

    def _get_weighted_fraction(self, start_dates: pd.Series, end_dates: pd.Series, target_date: pd.Timestamp, rpgt_series: pd.Series) -> np.ndarray:
        """Calculates exactly what percentage of a calendar month a routing rule was alive for."""
        _, days_in_mo = calendar.monthrange(target_date.year, target_date.month)
        month_start = pd.Timestamp(target_date.year, target_date.month, 1)
        month_end = month_start + pd.DateOffset(months=1)

        m_start_v = np.datetime64(month_start, 'D')
        m_end_v = np.datetime64(month_end, 'D')
        s_dt = pd.to_datetime(start_dates).values.astype('datetime64[D]')
        e_dt = pd.to_datetime(end_dates).fillna(month_end).values.astype('datetime64[D]')

        eff_start = np.clip(s_dt, m_start_v, m_end_v)
        eff_end = np.clip(e_dt, m_start_v, m_end_v)

        start_days = (eff_start - m_start_v).astype('timedelta64[D]').astype(int) + 1
        end_days = (eff_end - m_start_v).astype('timedelta64[D]').astype(int)

        target_month_num = target_date.month
        if self.mr_daily_weights and target_month_num in self.mr_daily_weights:
            mr_w = np.array([self.mr_daily_weights[target_month_num].get(d, 0.0) for d in range(1, days_in_mo + 1)])
        else: 
            mr_w = np.ones(days_in_mo)

        mr_cdf = np.insert(np.cumsum(mr_w / mr_w.sum()), 0, 0)
        lin_cdf = np.linspace(0, 1, days_in_mo + 1)

        is_mr = (rpgt_series.astype(str).values == 'monthly renewal')
        valid = (e_dt > m_start_v) & (s_dt < m_end_v) & (start_days <= end_days)

        # 🟢 UPGRADED TO FLOAT64
        weights = np.zeros(len(start_dates), dtype=np.float64)
        ed_idx, sd_idx = end_days[valid], start_days[valid] - 1
        weights[valid] = np.where(is_mr[valid], mr_cdf[ed_idx] - mr_cdf[sd_idx], lin_cdf[ed_idx] - lin_cdf[sd_idx]).astype(np.float64)
        return np.clip(weights, 0.0, 1.0)

    def _process_allocation(self, chunk_merged_df: pd.DataFrame, mapped_agg_df: pd.DataFrame, is_pre: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """MANAGER: Applies time-weights and physically shifts volume between gateways."""
        dest_gw_col = 'gatewayFid_y' if 'gatewayFid_y' in chunk_merged_df.columns else 'Gateway'
        src_gw_col = 'gatewayFid_x' if 'gatewayFid_x' in chunk_merged_df.columns else 'gatewayFid'
        prefix = 'PreSim_' if is_pre else 'Reallocated_'

        idx_src = self.join_keys + [src_gw_col, 'fcpNumber', 'attemptNumber']
        idx_dest = self.join_keys + [dest_gw_col, 'fcpNumber', 'attemptNumber']

        go_live_v, next_gl_v, rpgt_v = chunk_merged_df['GO LIVE'], chunk_merged_df['Next_GO_LIVE'], chunk_merged_df['RPGT']
        
        # 🟢 UPGRADED TO FLOAT64
        share_vamp = chunk_merged_df['Share_Norm_Vamp'].values.astype(np.float64)
        share_norm = chunk_merged_df['Share_Norm'].values.astype(np.float64)
        
        calc_arrays, all_calc_cols = {}, []
        for m in range(6):
            check_date = (self.m0_start - pd.DateOffset(months=1)) if is_pre else self.target_dates[m]
            w_active = self._get_weighted_fraction(go_live_v, next_gl_v, check_date, rpgt_v)

            if f'fc_vi_trx_m{m}' in chunk_merged_df.columns:
                target_col = f"{prefix}fc_vi_trx_m{m}"
                all_calc_cols.append(target_col)
                # 🟢 UPGRADED TO FLOAT64
                calc_arrays[target_col] = chunk_merged_df[f'fc_vi_trx_m{m}'].values.astype(np.float64) * w_active * np.where(share_norm > 0, share_norm, 0)

            for t in range(10):
                if f't{t}_fcast_m{m}' in chunk_merged_df.columns:
                    target_col = f"{prefix}t{t}_fcast_m{m}"
                    all_calc_cols.append(target_col)
                    vamp_move_ratio = share_vamp if t <= m else 0.0
                    # 🟢 UPGRADED TO FLOAT64
                    calc_arrays[target_col] = chunk_merged_df[f't{t}_fcast_m{m}'].values.astype(np.float64) * w_active * np.where(vamp_move_ratio > 0, vamp_move_ratio, 0)

        # Aggregate Moves
        temp_df = pd.DataFrame(calc_arrays)
        mask_has_moves = (temp_df[all_calc_cols] > 0).any(axis=1)
        temp_df = temp_df.loc[mask_has_moves]
        filtered_chunk = chunk_merged_df.loc[mask_has_moves]

        for c in idx_src: temp_df[c] = filtered_chunk[c].values
        moves_src = temp_df.groupby(idx_src, observed=True)[all_calc_cols].sum().reset_index().rename(columns={src_gw_col: 'gatewayFid'})
        temp_df.drop(columns=idx_src, errors='ignore', inplace=True)

        for c in idx_dest: temp_df[c] = filtered_chunk[c].values
        moved_res = temp_df.groupby(idx_dest, observed=True)[all_calc_cols].sum().reset_index().rename(columns={dest_gw_col: 'finalGateway'})

        # Calculate Remainder
        remainder = mapped_agg_df[self.join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber']].copy()
        for c in all_calc_cols:
            orig = c.replace(prefix, '')
            # 🟢 UPGRADED TO FLOAT64
            remainder[c] = mapped_agg_df[orig].values.astype(np.float64) if orig in mapped_agg_df.columns else 0.0

        mask_has_base = (remainder[all_calc_cols] > 0).any(axis=1)
        remainder = pd.merge(remainder.loc[mask_has_base], moves_src, on=self.join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber'], how='left', suffixes=('', '_moved'))

        for c in all_calc_cols:
            moved_col = f"{c}_moved"
            if moved_col in remainder.columns:
                remainder[c] = (remainder[c].fillna(0) - remainder[moved_col].fillna(0)).clip(lower=0)
        
        remain_res = remainder.rename(columns={'gatewayFid': 'finalGateway'})[self.join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber'] + all_calc_cols]
        remain_res = remain_res.loc[(remain_res[all_calc_cols] > 0).any(axis=1)]

        return moved_res, remain_res

    # =========================================================================
    # === 3. DEATH SYNCS & REDISTRIBUTION
    # =========================================================================

    def _ram_safe_redistribute(self, df: pd.DataFrame, col: str, dead_gws: set, group_cols: List[str]):
        """The Lossless Load-Balancer. Pushes stranded volume to surviving gateways."""
        if col not in df.columns: return
        mask_dead = df['finalGateway'].isin(dead_gws)
        if not mask_dead.any(): return

        orphan_totals = df.loc[mask_dead].groupby(group_cols, observed=True)[col].sum().to_dict()
        df.loc[mask_dead, col] = 0.0

        mask_alive = ~mask_dead
        alive_totals = df.loc[mask_alive].groupby(group_cols, observed=True)[col].transform('sum').values
        alive_counts = df.loc[mask_alive].groupby(group_cols, observed=True)[col].transform('count').values
        orphan_mapped = df.loc[mask_alive].set_index(group_cols).index.map(orphan_totals).fillna(0.0).values

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(alive_totals > 0, df.loc[mask_alive, col].values / alive_totals, np.where(alive_counts > 0, 1.0 / alive_counts, 0.0))
            bonus = ratios * orphan_mapped

        # 🟢 UPGRADED TO FLOAT64
        df.loc[mask_alive, col] += bonus.astype(np.float64)

    def _apply_death_syncs(self, pre_df: pd.DataFrame, post_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """MANAGER: The ultimate safety net. Perfectly redistributes orphaned volume."""
        if not self.overrides: return pre_df, post_df
        
        dead_trx, dead_vamp = {m: set() for m in range(6)}, {m: set() for m in range(6)}
        for fid, cfg in self.overrides.items():
            if isinstance(cfg, dict) and cfg.get('target', 0) == 0:
                gw, app, eff_dt = str(fid).strip().lower(), cfg.get('apply_to', 'both'), cfg.get('effective_date')
                for m in range(6):
                    if not eff_dt or pd.to_datetime(eff_dt) <= self.target_dates[m]:
                        if app in ['trx', 'both']: dead_trx[m].add(gw)
                        if app in ['vamp', 'both']: dead_vamp[m].add(gw)

        full_keys = self.join_keys + ['fcpNumber', 'attemptNumber']
        for m in range(6):
            if dead_vamp[m]:
                v_post, v_pre = [f'Reallocated_t{t}_fcast_m{m}' for t in range(10)], [f'PreSim_t{t}_fcast_m{m}' for t in range(10)]
                pre_totals = pre_df.groupby(full_keys, observed=True).size()
                pre_deads = pre_df[pre_df['finalGateway'].isin(dead_vamp[m])].groupby(full_keys, observed=True).size()
                doomed = _doomed_keys(pre_totals, pre_deads)
                if len(doomed) > 0:
                    post_mask = post_df.set_index(full_keys).index.isin(doomed)
                    for col in v_post:
                        if col in post_df.columns: post_df.loc[post_mask, col] = 0.0
                for po_col, pr_col in zip(v_post, v_pre):
                    self._ram_safe_redistribute(post_df, po_col, dead_vamp[m], full_keys)
                    self._ram_safe_redistribute(pre_df, pr_col, dead_vamp[m], full_keys)

            if dead_trx[m]:
                t_post, t_pre = [f'Reallocated_fc_vi_trx_m{m}'], [f'PreSim_fc_vi_trx_m{m}']
                pre_totals = pre_df.groupby(full_keys, observed=True).size()
                pre_deads = pre_df[pre_df['finalGateway'].isin(dead_trx[m])].groupby(full_keys, observed=True).size()
                doomed = _doomed_keys(pre_totals, pre_deads)
                if len(doomed) > 0:
                    post_mask = post_df.set_index(full_keys).index.isin(doomed)
                    if t_post[0] in post_df.columns: post_df.loc[post_mask, t_post[0]] = 0.0
                self._ram_safe_redistribute(post_df, t_post[0], dead_trx[m], full_keys)
                self._ram_safe_redistribute(pre_df, t_pre[0], dead_trx[m], full_keys)

        # 🟢 UPGRADED TO FLOAT64
        num_post = post_df.select_dtypes(include=['number']).columns
        post_df[num_post] = post_df[num_post].fillna(0.0).astype(np.float64)
        num_pre = pre_df.select_dtypes(include=['number']).columns
        pre_df[num_pre] = pre_df[num_pre].fillna(0.0).astype(np.float64)

        return pre_df, post_df

    # =========================================================================
    # === 4. ORCHESTRATOR
    # =========================================================================

    def execute_time_aware_routing(self, batch_size: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        MANAGER: Orchestrates the Time-Aware Engine, chunking massive blocks 
        to preserve RAM and applying Google Sheet Rules to 'Pre' and 'Post' schemas.
        """
        logger.info("Initializing Allocation Matrix (Data Cleaning & Setup)...")

        # DIAGNOSTIC (log-only): dump the RAW parsed routing rules (self.split_df) for the
        # focus cell, straight from data_extractor and BEFORE any engine processing — each
        # gateway's parsed Share + Rule_Source. This pinpoints where a gateway the exported
        # rule scored 0% / omitted (e.g. braintree, bancard) actually enters the split.
        try:
            import os as _os6
            _tr6 = _os6.environ.get("ROUTING_ALLOC_TRACE", "").strip()
            if _tr6 and isinstance(self.split_df, pd.DataFrame) and not self.split_df.empty:
                _p6 = [x.strip().lower() for x in _tr6.split("|")] + ["", "", ""]
                _c6, _b6, _r6 = _p6[0], _p6[1], _p6[2]
                _sd6 = self.split_df.copy()
                def _col6(*names):
                    for _n in names:
                        if _n in _sd6.columns:
                            return _n
                    return None
                _cc, _bc, _rc = _col6("Currency", "currency"), _col6("BIN", "bin"), _col6("RPGT", "rpgt")
                if _cc and _bc and _rc:
                    _sel6 = _sd6[(_sd6[_cc].astype(str).str.strip().str.lower() == _c6) &
                                 (_sd6[_bc].astype(str).str.strip().str.replace(r"\.0$", "", regex=True) == _b6) &
                                 (_sd6[_rc].astype(str).str.strip().str.lower() == _r6)]
                    if _sel6.empty:
                        logger.info(f"ALLOC_TRACE RAW split_df: no parsed-rule rows for {_tr6} "
                                    f"(⇒ these gateways are NOT in the parsed rules — injected later)")
                    else:
                        _sc = _col6("Share")
                        _showcols = [c for c in ("paymentMethodProvider", "Country", "STICKY",
                                                 "GO LIVE", "Rule_Source", "gatewayFid", _sc) if c and c in _sel6.columns]
                        logger.info(f"ALLOC_TRACE RAW split_df (parsed rules, pre-engine) {_tr6}  rows={len(_sel6)}")
                        for _rec in _sel6.sort_values(_sc, ascending=False)[_showcols].head(60).itertuples(index=False) \
                                if _sc else _sel6[_showcols].head(60).itertuples(index=False):
                            logger.info("   " + " | ".join(str(x) for x in _rec))
                        # explicit braintree / bancard presence check
                        _gc = _col6("gatewayFid")
                        if _gc:
                            _gl = _sel6[_gc].astype(str).str.lower()
                            logger.info(f"   → braintree-usd-tav in parsed rules? "
                                        f"{bool(_gl.str.contains('braintree-usd-tav').any())} · "
                                        f"bancard-usd-tav? {bool(_gl.str.contains('bancard-usd-tav').any())}")
        except Exception as _e6:  # noqa: BLE001
            logger.info(f"ALLOC_TRACE RAW split_df failed: {_e6}")

        df_in, split_work = self._prepare_allocation_matrices()
        split_work = self._inject_dynamic_snapshots(split_work)
        split_work = self._stitch_timeline(split_work)
        
        mapped_agg, unmapped_agg = self._map_and_filter_cohorts(df_in, split_work)

        # DIAGNOSTIC (log-only): dump the engine's OWN normalised routing shares (Share_Norm)
        # per gateway for the focus cell, BEFORE any allocation. This is the decisive test:
        # if a gateway the exported rule scores at 0% (e.g. braintree) shows Share_Norm>0 here,
        # the divergence is in rule parsing / normalisation (data_extractor / snapshots); if it
        # is 0 here yet still receives volume in POST, the divergence is a later redistribution.
        try:
            import os as _os3
            _tr3 = _os3.environ.get("ROUTING_ALLOC_TRACE", "").strip()
            if _tr3:
                _p3 = [x.strip().lower() for x in _tr3.split("|")] + ["", "", ""]
                _c3, _b3, _r3 = _p3[0], _p3[1], _p3[2]
                _sw = split_work
                _selm = ((_sw["Currency"].astype(str).str.strip().str.lower() == _c3) &
                         (_sw["BIN"].astype(str).str.strip() == _b3) &
                         (_sw["RPGT"].astype(str).str.strip().str.lower() == _r3))
                _s = _sw[_selm]
                if _s.empty:
                    logger.info(f"ALLOC_TRACE SPLIT_WORK: no rule rows for {_tr3}")
                else:
                    _ex = [c for c in ("paymentMethodProvider", "Country", "STICKY", "GO LIVE") if c in _s.columns]
                    _cols = _ex + ["gatewayFid", "Share_Norm"] + (["Share_Norm_Vamp"] if "Share_Norm_Vamp" in _s.columns else [])
                    logger.info(f"ALLOC_TRACE SPLIT_WORK (normalised routing shares) {_tr3}  rows={len(_s)}")
                    for _rec in (_s.sort_values(_ex + ["Share_Norm"], ascending=False)[_cols]
                                 .head(60).itertuples(index=False)):
                        logger.info("   " + " | ".join(str(x) for x in _rec))
        except Exception as _e3:  # noqa: BLE001
            logger.info(f"ALLOC_TRACE SPLIT_WORK failed: {_e3}")

        # DIAGNOSTIC (log-only, never changes results): env ROUTING_ALLOC_TRACE="cur|bin|rpgt"
        # dumps each gateway's MAPPED (rerouted) vs UNMAPPED (held with incumbent) VI for that
        # cell, so the tab-3-vs-tab-5 held-cohort gap (e.g. Braintree) can be localised exactly —
        # is the incumbent's retained volume the unmapped cohort, or something the projection can't
        # see? Gated so it costs nothing unless enabled.
        import os as _os
        _tr = _os.environ.get("ROUTING_ALLOC_TRACE", "").strip()
        if _tr:
            try:
                _p = [x.strip().lower() for x in _tr.split("|")] + ["", "", ""]
                _tcur, _tbin, _trp = _p[0], _p[1], _p[2]

                def _alloc_trace(_df, _lbl):
                    if _df is None or _df.empty:
                        logger.info(f"ALLOC_TRACE {_lbl}: empty"); return
                    _c = _df
                    _cur = _c["Currency"].astype(str).str.strip().str.lower()
                    _bn = _c["BIN"].astype(str).str.strip()
                    _rp = _c["RPGT"].astype(str).str.strip().str.lower()
                    _sel = _c[(_cur == _tcur) & (_bn == _tbin) & (_rp == _trp)]
                    _col = next((x for x in _sel.columns if str(x).startswith("fc_vi_trx_m1")), None) \
                        or next((x for x in _sel.columns if str(x).startswith("fc_vi_trx")), None)
                    if _col is None or _sel.empty:
                        logger.info(f"ALLOC_TRACE {_lbl}: no rows/VI col for {_tr}"); return
                    _g = _sel.groupby("gatewayFid", observed=True)[_col].sum().sort_values(ascending=False)
                    logger.info(f"ALLOC_TRACE {_lbl} [{_col}] {_tcur}/{_tbin}/{_trp}  Σ={_g.sum():,.1f}")
                    for _fid, _v in _g.head(20).items():
                        logger.info(f"   {_fid}: {_v:,.1f}")
                _alloc_trace(mapped_agg, "MAPPED (rerouted per rule)")
                _alloc_trace(unmapped_agg, "UNMAPPED (held with incumbent)")
            except Exception as _e:  # noqa: BLE001
                logger.info(f"ALLOC_TRACE failed: {_e}")

        logger.info("Executing Micro-Chunked Vector Math...")
        post_chunks, pre_chunks = [], []
        mapped_agg = mapped_agg.reset_index(drop=True)
        total_rows = len(mapped_agg)

        group_cols = self.join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber']

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            sub_mapped = mapped_agg.iloc[start_idx:end_idx].copy()
            sub_merged = pd.merge(sub_mapped, split_work[split_work['RPGT'].isin(sub_mapped['RPGT'].unique())], on=self.join_keys, how='inner')
            
            if sub_merged.empty: continue

            for is_pre_flag, chunks_list in [(False, post_chunks), (True, pre_chunks)]:
                m_res, r_res = self._process_allocation(sub_merged, sub_mapped, is_pre_flag)
                sub_df = pd.concat([m_res, r_res], ignore_index=True)
                num_cols = sub_df.select_dtypes(include='number').columns
                sub_df = sub_df.loc[(sub_df[num_cols] > 0).any(axis=1)]
                chunks_list.append(sub_df.groupby(group_cols, as_index=False, observed=True).sum())

                if len(chunks_list) >= 5:
                    compressed = pd.concat(chunks_list, ignore_index=True).groupby(group_cols, as_index=False, observed=True).sum()
                    chunks_list.clear()
                    chunks_list.append(compressed)

        unmapped_post = unmapped_agg.rename(columns={'gatewayFid': 'finalGateway'})
        unmapped_pre = unmapped_agg.rename(columns={'gatewayFid': 'finalGateway'})
        post_cols = [f'Reallocated_{c}' for c in self.valid_agg_cols]
        pre_cols = [f'PreSim_{c}' for c in self.valid_agg_cols]

        # 🟢 UPGRADED TO FLOAT64
        for c, orig_c in zip(post_cols, self.valid_agg_cols): unmapped_post[c] = unmapped_agg[orig_c].values.astype(np.float64) if orig_c in unmapped_agg.columns else 0.0
        for c, orig_c in zip(pre_cols, self.valid_agg_cols): unmapped_pre[c] = unmapped_agg[orig_c].values.astype(np.float64) if orig_c in unmapped_agg.columns else 0.0
        
        unmapped_post = unmapped_post.loc[(unmapped_post[post_cols] > 0).any(axis=1)]
        unmapped_pre = unmapped_pre.loc[(unmapped_pre[pre_cols] > 0).any(axis=1)]

        post_df = pd.concat(post_chunks + [unmapped_post[group_cols + post_cols]], ignore_index=True).groupby(group_cols, as_index=False, observed=True).sum()
        pre_df = pd.concat(pre_chunks + [unmapped_pre[group_cols + pre_cols]], ignore_index=True).groupby(group_cols, as_index=False, observed=True).sum()

        logger.info("Applying Death Syncs & Lossless Redistribution...")
        pre_df, post_df = self._apply_death_syncs(pre_df, post_df)

        # DIAGNOSTIC (log-only): if ROUTING_ALLOC_TRACE="cur|bin|rpgt" is set, dump the FINAL
        # pre→post VI per finalGateway for that cell (after death-syncs), broken down by
        # paymentMethodProvider × Country when present. Paired with the MAPPED/UNMAPPED dump
        # above, this shows EXACTLY where an incumbent's post volume comes from — held (unmapped)
        # vs received-as-destination vs death-sync — i.e. the source of the tab3-vs-tab5 gap
        # (e.g. Braintree post 440 vs a held-only 95). Never changes results.
        try:
            import os as _os2
            _tr2 = _os2.environ.get("ROUTING_ALLOC_TRACE", "").strip()
            if _tr2:
                _p2 = [x.strip().lower() for x in _tr2.split("|")] + ["", "", ""]
                _tc2, _tb2, _tr2r = _p2[0], _p2[1], _p2[2]
                def _post_trace(_df, _lbl, _pre):
                    if _df is None or _df.empty:
                        logger.info(f"ALLOC_TRACE POST {_lbl}: empty"); return
                    _cur = _df["Currency"].astype(str).str.strip().str.lower()
                    _bn = _df["BIN"].astype(str).str.strip()
                    _rp = _df["RPGT"].astype(str).str.strip().str.lower()
                    _sel = _df[(_cur == _tc2) & (_bn == _tb2) & (_rp == _tr2r)]
                    if _sel.empty:
                        logger.info(f"ALLOC_TRACE POST {_lbl}: no rows for {_tr2}"); return
                    _pfx = "PreSim_" if _pre else "Reallocated_"
                    _col = next((c for c in _sel.columns if str(c).startswith(f"{_pfx}fc_vi_trx_m1")), None) \
                        or next((c for c in _sel.columns if str(c).startswith(f"{_pfx}fc_vi_trx")), None)
                    if _col is None:
                        logger.info(f"ALLOC_TRACE POST {_lbl}: no {_pfx}VI col"); return
                    _extra = [c for c in ("paymentMethodProvider", "Country") if c in _sel.columns]
                    _keys = _extra + ["finalGateway"]
                    _g = _sel.groupby(_keys, observed=True)[_col].sum()
                    _g = _g[_g != 0].sort_values(ascending=False)
                    logger.info(f"ALLOC_TRACE POST {_lbl} [{_col}] {_tc2}/{_tb2}/{_tr2r}  Σ={_g.sum():,.1f}")
                    for _k, _v in _g.head(40).items():
                        logger.info(f"   {_k}: {_v:,.1f}")
                _post_trace(pre_df, "PRE  (final per gateway)", True)
                _post_trace(post_df, "POST (final per gateway)", False)
        except Exception as _e2:  # noqa: BLE001
            logger.info(f"ALLOC_TRACE POST failed: {_e2}")

        # --- GRANULAR PROFILE SAMPLES (tab-5 run log): dump a handful of representative
        #     routed profiles (Currency × BIN × RPGT × pmp × Country) end-to-end — each
        #     gateway's PRE → POST VI (M1) — so the pipeline's ACTUAL per-profile routing is
        #     visible at the same grain as the tab-2/tab-3 samples, for thorough tab3-vs-tab5
        #     debugging. Samples the biggest profiles + the biggest reallocations. Reads the
        #     in-memory frames (no re-read of the export). Set ROUTING_PROFILE_SAMPLES=0 to
        #     disable. Best-effort — never affects results. ---
        try:
            import os as _os5
            if _os5.environ.get("ROUTING_PROFILE_SAMPLES", "1") != "0":
                _pcols = [c for c in ("Currency", "BIN", "RPGT",
                                      "paymentMethodProvider", "Country") if c in post_df.columns]
                _vp = next((c for c in pre_df.columns if str(c).startswith("PreSim_fc_vi_trx_m1")), None)
                _vq = next((c for c in post_df.columns if str(c).startswith("Reallocated_fc_vi_trx_m1")), None)
                if _pcols and _vp and _vq and not post_df.empty:
                    _pg = pre_df.groupby(_pcols + ["finalGateway"], observed=True)[_vp].sum().rename("pre")
                    _qg = post_df.groupby(_pcols + ["finalGateway"], observed=True)[_vq].sum().rename("post")
                    _m = pd.concat([_pg, _qg], axis=1).fillna(0.0).reset_index()
                    _m["absd"] = (_m["post"] - _m["pre"]).abs()
                    _cellst = _m.groupby(_pcols).agg(vol=("post", "sum"), move=("absd", "sum"))
                    _pick, _seen = [], set()
                    for _k in list(_cellst.sort_values("vol", ascending=False).head(3).index) + \
                              list(_cellst.sort_values("move", ascending=False).head(4).index):
                        _kk = _k if isinstance(_k, tuple) else (_k,)
                        if _kk not in _seen:
                            _seen.add(_kk); _pick.append(_kk)
                        if len(_pick) >= 6:
                            break
                    _mg = _m.groupby(_pcols)
                    logger.info(f"── GRANULAR PROFILE SAMPLES (tab-5 actual) · {len(_pick)} of "
                                f"{len(_cellst):,} profiles ({' × '.join(_pcols)}) · M1 VI ──")
                    logger.info("   each row: gateway · PRE → POST · Δ")
                    for _kk in _pick:
                        _rows = _mg.get_group(_kk if len(_kk) > 1 else _kk[0]).copy()
                        _rows = _rows[(_rows["pre"] > 1e-6) | (_rows["post"] > 1e-6)]
                        _rows = _rows.sort_values("post", ascending=False).head(14)
                        _lbl = " / ".join(str(x) for x in _kk)
                        logger.info(f"   • {_lbl}  ·  vol={float(_rows['post'].sum()):,.0f}  ·  "
                                    f"{len(_rows)} active gateway(s)")
                        for _, _r in _rows.iterrows():
                            _pre, _post = float(_r["pre"]), float(_r["post"])
                            logger.info(f"       {str(_r['finalGateway']):<30s} {_pre:>9,.1f} → {_post:>9,.1f}  "
                                        f"({_post - _pre:+9,.1f})")
        except Exception as _e5:  # noqa: BLE001
            logger.info(f"GRANULAR PROFILE SAMPLES failed: {_e5}")

        return pre_df, post_df