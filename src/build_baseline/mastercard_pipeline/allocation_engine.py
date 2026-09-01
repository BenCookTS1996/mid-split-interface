import re
import calendar
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from .utils import setup_logger, clean_key_col

logger = setup_logger(__name__)

__build__ = "2026-08-12-mastercard-initial+dedupe-renamed-columns"


def _doomed_keys(pre_totals: pd.Series, pre_deads: pd.Series) -> pd.Index:
    """Return the MultiIndex keys whose ENTIRE volume is dead (total == dead count).
    Uses a plain dict lookup rather than Series.reindex to sidestep a pandas
    'Buffer dtype mismatch' error on mismatched MultiIndex level-code widths."""
    dead_map = pre_deads.to_dict()
    keep = [k for k, tot in pre_totals.items() if tot == dead_map.get(k, 0)]
    return pd.Index(keep, tupleize_cols=False) if keep else pd.Index([])


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate-named columns into ONE (first non-null value per row).

    `map_std` maps several source aliases onto the same target (e.g. Gateway / gateway /
    gateway_fid → gatewayFid; company / Brand / brand → Company). If an input sheet carries more
    than one of those aliases — or the target name PLUS an alias — the rename produces duplicate
    columns, so df['gatewayFid'] becomes 2-D and the downstream groupby raises
    'Grouper for 'gatewayFid' not 1-dimensional'. This coalesces each duplicated label back to a
    single column (left-to-right first non-null), preserving column order. No-op when there are
    no duplicates."""
    if not df.columns.duplicated().any():
        return df
    seen: Dict[str, List[int]] = {}
    for i, name in enumerate(df.columns):
        seen.setdefault(name, []).append(i)
    data = {}
    for name, idxs in seen.items():
        if len(idxs) == 1:
            data[name] = df.iloc[:, idxs[0]]
        else:
            data[name] = df.iloc[:, idxs].bfill(axis=1).iloc[:, 0]   # first non-null across dupes
    return pd.DataFrame(data, index=df.index)


def _safe_rename_std(df: pd.DataFrame, map_std: Dict[str, str]) -> pd.DataFrame:
    """`df.rename(columns=map_std)` that can NEVER create a duplicate column.

    Several source aliases map to the SAME target (Gateway / gateway / gateway_fid → gatewayFid;
    company / Brand / brand → Company). An alias is renamed ONLY if its target isn't already a real
    column, and only the FIRST alias per target is used. So a canonical column that already exists —
    e.g. the melted 'gatewayFid' — is PRESERVED and a stray alias is left untouched instead of
    colliding onto it (which is what produced the 2-D 'gatewayFid' and the groupby crash)."""
    claimed = set(map(str, df.columns))          # names already taken (targets satisfied as-is)
    ren = {}
    for src, tgt in map_std.items():
        if src in df.columns and src != tgt and tgt not in claimed:
            ren[src] = tgt
            claimed.add(tgt)
    return df.rename(columns=ren) if ren else df


class AllocationEngine:
    """
    MASTERCARD time-aware allocation engine (parallels the Visa/VAMP AllocationEngine).

    Applies the split rules across the six forecast months and the t-period chargeback
    matrix, honouring GO LIVE timelines, mid-month gateway deaths and orphan redistribution.

    Mastercard specifics:
      * only fc_mc_trx_m1..m5 sales are re-routed — fc_mc_trx_m0 is the injected real history
        and is EXCLUDED from allocation, then appended back unaltered afterwards;
      * the risk matrix columns are t{t}_fcast_m{m} chargebacks.
    """

    def __init__(self, config: Dict[str, Any], attempts_df: pd.DataFrame,
                 split_df: pd.DataFrame, mr_weights: Optional[Dict[int, Dict[int, float]]] = None):
        self.config = config
        self.attempts_df = attempts_df
        self.split_df = split_df
        self.mr_daily_weights = mr_weights or {}
        self.overrides = config.get('gateway_volume_overrides', {}) or {}

        self.m0_start = pd.to_datetime(config['run_settings']['month_0_start_date'])
        self.target_dates = {m: self.m0_start + pd.DateOffset(months=m) for m in range(6)}

        # t{t}_fcast_m{m} chargeback columns (all months) + fc_mc_trx_m1..m5 sales columns.
        # 🟢 fc_mc_trx_m0 (injected real history) is intentionally EXCLUDED from valid_agg_cols.
        self.t_cols = [c for c in attempts_df.columns if re.match(r'^t\d+_fcast_m\d+$', str(c))]
        self.mc_cols = [c for c in attempts_df.columns if re.match(r'^fc_mc_trx_m[1-5]$', str(c))]
        self.valid_agg_cols = self.t_cols + self.mc_cols

        self.map_std = {'company': 'Company', 'Brand': 'Company', 'brand': 'Company', 'gateway_fid': 'gatewayFid', 'Gateway': 'gatewayFid', 'gateway': 'gatewayFid', 'riskDefinedProductSubscriptionType': 'RPGT', 'rpgt': 'RPGT', 'paymentmethodprovider': 'paymentMethodProvider', 'country': 'Country', 'bin': 'BIN', 'currency': 'Currency', 'STICKY': 'renewal_number', 'sticky': 'renewal_number'}
        self.join_keys = ['Company', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number']

    # =========================================================================
    # === MATRIX PREPARATION
    # =========================================================================

    def _prepare_allocation_matrices(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        map_std, join_keys, valid_agg_cols = self.map_std, self.join_keys, self.valid_agg_cols
        attempts_df = self.attempts_df
        split_df = self.split_df if self.split_df is not None else pd.DataFrame()

        cols_to_extract = list(set(list(map_std.keys()) + list(map_std.values()) + join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber'] + valid_agg_cols).intersection(attempts_df.columns))
        df_in = _dedupe_columns(_safe_rename_std(attempts_df[cols_to_extract], map_std))

        for c in valid_agg_cols:
            if c in df_in.columns:
                df_in[c] = df_in[c].astype(np.float64)

        split_work = _safe_rename_std(split_df, map_std).copy() if not split_df.empty else pd.DataFrame(columns=join_keys + ['GO LIVE', 'gatewayFid', 'Share'])
        split_work = _dedupe_columns(split_work)   # backstop: collapse any pre-existing duplicate names
        if 'Share' not in split_work.columns:
            split_work['Share'] = 0.0
        if 'GO LIVE' not in split_work.columns:
            split_work['GO LIVE'] = pd.to_datetime('2020-01-01')
        if 'gatewayFid' not in split_work.columns:
            split_work['gatewayFid'] = 'unmapped'
        for c in join_keys:
            if c not in split_work.columns:
                split_work[c] = 'unknown'

        for col in join_keys:
            df_in[col] = clean_key_col(df_in[col], remove_dot_zero=True)
            split_work[col] = clean_key_col(split_work[col], remove_dot_zero=True)

        df_in['gatewayFid'], split_work['gatewayFid'] = df_in['gatewayFid'].astype('category'), split_work['gatewayFid'].astype('category')
        df_in['fcpNumber'] = clean_key_col(df_in['fcpNumber']) if 'fcpNumber' in df_in.columns else '1'
        df_in['attemptNumber'] = clean_key_col(df_in['attemptNumber']) if 'attemptNumber' in df_in.columns else '1'

        split_work['GO LIVE'] = pd.to_datetime(split_work['GO LIVE'], errors='coerce', dayfirst=True).fillna(pd.Timestamp('2020-01-01'))
        split_work['Share'] = pd.to_numeric(split_work['Share'], errors='coerce').fillna(0)
        split_work = split_work.groupby(join_keys + ['GO LIVE', 'gatewayFid'], observed=True)['Share'].sum().reset_index()
        split_work['Share_CB'] = split_work['Share'].copy()
        return df_in, split_work

    # =========================================================================
    # === DYNAMIC OVERRIDE SNAPSHOTS
    # =========================================================================

    def _parse_overrides(self):
        immediate_trx_kills, immediate_cb_kills, future_kills = [], [], []
        for fid, cfg in self.overrides.items():
            if isinstance(cfg, dict) and cfg.get('target', 0) == 0:
                gw, app, eff_date = str(fid).strip().lower(), cfg.get('apply_to', 'both'), cfg.get('effective_date')
                if eff_date:
                    future_kills.append((gw, app, pd.to_datetime(eff_date)))
                else:
                    if app in ['trx', 'both']:
                        immediate_trx_kills.append(gw)
                    if app in ['cb', 'vamp', 'both']:
                        immediate_cb_kills.append(gw)
            elif not isinstance(cfg, dict) and cfg == 0:
                immediate_trx_kills.append(str(fid).strip().lower())
                immediate_cb_kills.append(str(fid).strip().lower())
        return immediate_trx_kills, immediate_cb_kills, future_kills

    @staticmethod
    def _apply_immediate_kills(split_work, immediate_trx_kills, immediate_cb_kills):
        if immediate_trx_kills:
            split_work.loc[split_work['gatewayFid'].astype(str).str.lower().isin(immediate_trx_kills), 'Share'] = 0.0
        if immediate_cb_kills:
            split_work.loc[split_work['gatewayFid'].astype(str).str.lower().isin(immediate_cb_kills), 'Share_CB'] = 0.0
        return split_work

    def _inject_future_snapshots(self, split_work, future_kills, join_keys):
        if not future_kills:
            return split_work

        for eff_dt in sorted(list(set([d for _, _, d in future_kills]))):
            split_work = split_work.sort_values(join_keys + ['GO LIVE'])
            split_work['Next_GO_LIVE'] = split_work.groupby(join_keys, observed=True)['GO LIVE'].shift(-1)

            gw_list = [k[0] for k in [(gw, app) for gw, app, d in future_kills if d == eff_dt]]

            mask_active = (split_work['GO LIVE'] <= eff_dt) & (split_work['Next_GO_LIVE'].isna() | (split_work['Next_GO_LIVE'] > eff_dt))
            profiles_affected = split_work.loc[mask_active & split_work['gatewayFid'].astype(str).str.lower().isin(gw_list), join_keys].drop_duplicates()

            if not profiles_affected.empty:
                snapshot_df = pd.merge(split_work[mask_active], profiles_affected, on=join_keys, how='inner')
                snapshot_df['GO LIVE'] = eff_dt

                for gw, app in [(gw, app) for gw, app, d in future_kills if d == eff_dt]:
                    mask_target_gw = snapshot_df['gatewayFid'].astype(str).str.lower() == gw
                    if app in ['trx', 'both']:
                        snapshot_df.loc[mask_target_gw, 'Share'] = 0.0
                    if app in ['cb', 'vamp', 'both']:
                        snapshot_df.loc[mask_target_gw, 'Share_CB'] = 0.0

                split_work = pd.concat([split_work.drop(columns=['Next_GO_LIVE']), snapshot_df], ignore_index=True)
                split_work = split_work.drop_duplicates(subset=join_keys + ['GO LIVE', 'gatewayFid'], keep='last')
            else:
                split_work = split_work.drop(columns=['Next_GO_LIVE'])

        return split_work

    def _normalize_shares(self, split_work, join_keys):
        for s_col, n_col in [('Share', 'Share_Norm'), ('Share_CB', 'Share_Norm_CB'), ('Share_CB_Restricted', 'Share_Norm_CB_Restricted')]:
            total = split_work.groupby(join_keys + ['GO LIVE'], observed=True)[s_col].transform('sum')
            if s_col == 'Share_CB_Restricted':
                fallback_total = split_work.groupby(join_keys + ['GO LIVE'], observed=True)['Share_CB'].transform('sum')
                split_work[n_col] = np.where(total > 0, split_work[s_col] / total,
                                    np.where(fallback_total > 0, split_work['Share_CB'] / fallback_total, 0.0))
            else:
                split_work[n_col] = np.where(total > 0, split_work[s_col] / total, 0.0)
        return split_work

    def _inject_dynamic_snapshots(self, split_work):
        join_keys = self.join_keys
        if not self.overrides:
            split_work['Share_CB_Restricted'] = split_work['Share_CB'].copy()
            return self._normalize_shares(split_work, join_keys)

        immediate_trx, immediate_cb, future_kills = self._parse_overrides()
        split_work = self._apply_immediate_kills(split_work, immediate_trx, immediate_cb)
        split_work = self._inject_future_snapshots(split_work, future_kills, join_keys)

        split_work['Share_CB_Restricted'] = split_work['Share_CB'].copy()
        age_gated_gws = [str(gw).strip().lower() for gw, cfg in self.overrides.items() if isinstance(cfg, dict) and cfg.get('age_gate_t') == 3]
        mask_age_gate = split_work['gatewayFid'].astype(str).str.lower().isin(age_gated_gws)
        split_work.loc[mask_age_gate, 'Share_CB_Restricted'] = 0.0

        return self._normalize_shares(split_work, join_keys)

    def _stitch_timeline(self, split_work):
        join_keys = self.join_keys
        unique_dates = split_work[join_keys + ['GO LIVE']].drop_duplicates().sort_values(join_keys + ['GO LIVE']).reset_index(drop=True)
        if not unique_dates.empty:
            unique_dates['Next_GO_LIVE'] = unique_dates['GO LIVE'].shift(-1)
            mask_same_group = (unique_dates[join_keys] == unique_dates[join_keys].shift(-1)).all(axis=1)
            unique_dates.loc[~mask_same_group, 'Next_GO_LIVE'] = pd.NaT
        else:
            unique_dates['Next_GO_LIVE'] = pd.Series(dtype='datetime64[ns]')
        return pd.merge(split_work.drop(columns=['Next_GO_LIVE'], errors='ignore'), unique_dates, on=join_keys + ['GO LIVE'], how='left')

    def _map_and_filter_cohorts(self, df_in, split_work) -> Tuple[pd.DataFrame, pd.DataFrame]:
        join_keys, valid_agg_cols = self.join_keys, self.valid_agg_cols
        chunk_agg = df_in.groupby(join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber'], as_index=False, observed=True)[valid_agg_cols].sum()
        unique_splits = split_work[join_keys].drop_duplicates().copy()
        unique_splits['_is_mapped'] = True
        chunk_agg = chunk_agg.merge(unique_splits, on=join_keys, how='left')

        mapped_mask = (chunk_agg['fcpNumber'] == '1') & (chunk_agg['_is_mapped'] == True)
        is_restricted_rpgt = chunk_agg['RPGT'].astype(str).str.lower().isin(['monthly initial', 'annual sub sale', 'upgrades'])
        mapped_mask = mapped_mask & (~is_restricted_rpgt | (is_restricted_rpgt & (chunk_agg['attemptNumber'] == '1')))

        mapped_agg = chunk_agg[mapped_mask].drop(columns=['_is_mapped']).copy()
        unmapped_agg = chunk_agg[~mapped_mask].drop(columns=['_is_mapped']).copy()
        return mapped_agg, unmapped_agg

    # =========================================================================
    # === TIME-WEIGHTED ALLOCATION MATH
    # =========================================================================

    def _get_weighted_fraction(self, start_dates, end_dates, target_date, rpgt_series) -> np.ndarray:
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
        if target_month_num in self.mr_daily_weights:
            mr_w = np.array([self.mr_daily_weights[target_month_num].get(d, 0.0) for d in range(1, days_in_mo + 1)])
        else:
            mr_w = np.ones(days_in_mo)
        if mr_w.sum() <= 0:
            mr_w = np.ones(days_in_mo)

        mr_cdf = np.insert(np.cumsum(mr_w / mr_w.sum()), 0, 0)
        lin_cdf = np.linspace(0, 1, days_in_mo + 1)

        is_mr = (rpgt_series.astype(str).values == 'monthly renewal')
        valid = (e_dt > m_start_v) & (s_dt < m_end_v) & (start_days <= end_days)

        weights = np.zeros(len(start_dates), dtype=np.float64)
        ed_idx, sd_idx = end_days[valid], start_days[valid] - 1
        weights[valid] = np.where(is_mr[valid], mr_cdf[ed_idx] - mr_cdf[sd_idx], lin_cdf[ed_idx] - lin_cdf[sd_idx]).astype(np.float64)
        return np.clip(weights, 0.0, 1.0)

    def _compute_allocation_arrays(self, chunk_merged_df, prefix, is_pre):
        go_live_v = chunk_merged_df['GO LIVE']
        next_gl_v = chunk_merged_df['Next_GO_LIVE']
        rpgt_v = chunk_merged_df['RPGT']
        share_cb = chunk_merged_df['Share_Norm_CB'].values.astype(np.float64)
        share_norm = chunk_merged_df['Share_Norm'].values.astype(np.float64)
        share_cb_restr = chunk_merged_df['Share_Norm_CB_Restricted'].values.astype(np.float64)

        calc_arrays, all_calc_cols = {}, []

        for m in range(6):
            check_date_trx = (self.m0_start - pd.DateOffset(months=1)) if is_pre else self.target_dates[m]
            w_trx = self._get_weighted_fraction(go_live_v, next_gl_v, check_date_trx, rpgt_v)

            col_mc = f'fc_mc_trx_m{m}'
            if col_mc in chunk_merged_df.columns:
                target_col = f"{prefix}{col_mc}"
                all_calc_cols.append(target_col)
                calc_arrays[target_col] = chunk_merged_df[col_mc].values.astype(np.float64) * w_trx * np.where(share_norm > 0, share_norm, 0)

            for t in range(10):
                cohort_date = self.target_dates[m] - pd.DateOffset(months=t)
                check_date_cb = (self.m0_start - pd.DateOffset(months=1)) if (is_pre and t <= m) else cohort_date
                w_cb = self._get_weighted_fraction(go_live_v, next_gl_v, check_date_cb, rpgt_v)
                col_v = f't{t}_fcast_m{m}'

                if col_v in chunk_merged_df.columns:
                    target_col = f"{prefix}{col_v}"
                    all_calc_cols.append(target_col)
                    active_share_cb = share_cb_restr if t > 3 else share_cb
                    cb_w = active_share_cb if t <= m else 0.0
                    calc_arrays[target_col] = chunk_merged_df[col_v].values.astype(np.float64) * w_cb * np.where(cb_w > 0, cb_w, 0)

        return calc_arrays, all_calc_cols

    @staticmethod
    def _aggregate_allocation_moves(calc_arrays, all_calc_cols, chunk_merged_df, idx_src, idx_dest, src_gw_col, dest_gw_col):
        temp_df = pd.DataFrame(calc_arrays)
        mask_has_moves = (temp_df[all_calc_cols] > 0).any(axis=1)
        temp_df = temp_df.loc[mask_has_moves]
        filtered_chunk_merged = chunk_merged_df.loc[mask_has_moves]

        for c in idx_src:
            temp_df[c] = filtered_chunk_merged[c].values
        moves_src = temp_df.groupby(idx_src, observed=True)[all_calc_cols].sum().reset_index()
        moves_src.rename(columns={src_gw_col: 'gatewayFid'}, inplace=True)
        temp_df.drop(columns=idx_src, errors='ignore', inplace=True)

        for c in idx_dest:
            temp_df[c] = filtered_chunk_merged[c].values
        moved_res = temp_df.groupby(idx_dest, observed=True)[all_calc_cols].sum().reset_index()
        moved_res.rename(columns={dest_gw_col: 'finalGateway'}, inplace=True)

        return moved_res, moves_src

    def _calculate_moved_volume(self, chunk_merged_df, prefix, idx_src, idx_dest, src_gw_col, dest_gw_col, join_keys, is_pre):
        calc_arrays, all_calc_cols = self._compute_allocation_arrays(chunk_merged_df, prefix, is_pre)
        moved_res, moves_src = self._aggregate_allocation_moves(calc_arrays, all_calc_cols, chunk_merged_df, idx_src, idx_dest, src_gw_col, dest_gw_col)
        return moved_res, moves_src, all_calc_cols

    @staticmethod
    def _calculate_remainder_volume(mapped_agg_df, moves_src, all_calc_cols, prefix, join_keys):
        remainder = mapped_agg_df[join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber']].copy()
        for c in all_calc_cols:
            orig = c.replace(prefix, '')
            remainder[c] = mapped_agg_df[orig].values.astype(np.float64) if orig in mapped_agg_df.columns else 0.0

        mask_has_base = (remainder[all_calc_cols] > 0).any(axis=1)
        remainder = remainder.loc[mask_has_base].copy()
        remainder = pd.merge(remainder, moves_src, on=join_keys + ['gatewayFid', 'fcpNumber', 'attemptNumber'], how='left', suffixes=('', '_moved'))

        for c in all_calc_cols:
            moved_col = f"{c}_moved"
            if moved_col in remainder.columns:
                remainder[c] = (remainder[c].fillna(0) - remainder[moved_col].fillna(0)).clip(lower=0)
                remainder.drop(columns=[moved_col], inplace=True)

        remainder_res = remainder.rename(columns={'gatewayFid': 'finalGateway'})[join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber'] + all_calc_cols]
        return remainder_res.loc[(remainder_res[all_calc_cols] > 0).any(axis=1)]

    def _process_allocation(self, chunk_merged_df, mapped_agg_df, is_pre):
        join_keys = self.join_keys
        dest_gw_col = 'gatewayFid_y' if 'gatewayFid_y' in chunk_merged_df.columns else 'Gateway'
        src_gw_col = 'gatewayFid_x' if 'gatewayFid_x' in chunk_merged_df.columns else 'gatewayFid'
        prefix = 'PreSim_' if is_pre else 'Reallocated_'

        idx_src = join_keys + [src_gw_col, 'fcpNumber', 'attemptNumber']
        idx_dest = join_keys + [dest_gw_col, 'fcpNumber', 'attemptNumber']

        moved_res, moves_src, all_calc_cols = self._calculate_moved_volume(chunk_merged_df, prefix, idx_src, idx_dest, src_gw_col, dest_gw_col, join_keys, is_pre)
        remainder_res = self._calculate_remainder_volume(mapped_agg_df, moves_src, all_calc_cols, prefix, join_keys)
        return moved_res, remainder_res

    # =========================================================================
    # === DEATH SYNC & ORPHAN REDISTRIBUTION
    # =========================================================================

    @staticmethod
    def _ram_safe_redistribute(df, col, dead_gws, group_cols):
        if col not in df.columns:
            return
        mask_dead = df['finalGateway'].isin(dead_gws)
        if not mask_dead.any():
            return

        orphan_totals = df.loc[mask_dead].groupby(group_cols, observed=True)[col].sum().to_dict()
        df.loc[mask_dead, col] = 0.0

        mask_alive = ~mask_dead
        alive_totals = df.loc[mask_alive].groupby(group_cols, observed=True)[col].transform('sum').values
        alive_counts = df.loc[mask_alive].groupby(group_cols, observed=True)[col].transform('count').values
        orphan_mapped = df.loc[mask_alive].set_index(group_cols).index.map(orphan_totals).fillna(0.0).values

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(alive_totals > 0, df.loc[mask_alive, col].values / alive_totals,
                     np.where(alive_counts > 0, 1.0 / alive_counts, 0.0))
            bonus = ratios * orphan_mapped

        df.loc[mask_alive, col] += bonus.astype(np.float64)

    def _parse_death_schedules(self):
        dead_gws_trx = {m: set() for m in range(6)}
        dead_gws_cb = {m: set() for m in range(6)}
        for fid, cfg in self.overrides.items():
            if isinstance(cfg, dict) and cfg.get('target', 0) == 0:
                gw = str(fid).strip().lower()
                app = cfg.get('apply_to', 'both')
                eff_date = cfg.get('effective_date')
                for m in range(6):
                    if not eff_date or pd.to_datetime(eff_date) <= self.target_dates[m]:
                        if app in ['trx', 'both']:
                            dead_gws_trx[m].add(gw)
                        if app in ['cb', 'vamp', 'both']:
                            dead_gws_cb[m].add(gw)
        return dead_gws_trx, dead_gws_cb

    @staticmethod
    def _sync_doomed_cohorts(pre_df, post_df, dead_gws, full_profile_keys, target_post_cols):
        pre_totals = pre_df.groupby(full_profile_keys, observed=True).size()
        pre_deads = pre_df[pre_df['finalGateway'].isin(dead_gws)].groupby(full_profile_keys, observed=True).size()

        doomed_profiles = _doomed_keys(pre_totals, pre_deads)

        if len(doomed_profiles) > 0:
            post_doomed_mask = post_df.set_index(full_profile_keys).index.isin(doomed_profiles)
            for col in target_post_cols:
                if col in post_df.columns:
                    post_df.loc[post_doomed_mask, col] = 0.0
        return post_df

    def _redistribute_month_volume(self, pre_df, post_df, dead_gws, full_profile_keys, pre_cols, post_cols):
        for post_col, pre_col in zip(post_cols, pre_cols):
            self._ram_safe_redistribute(post_df, post_col, dead_gws, full_profile_keys)
            self._ram_safe_redistribute(pre_df, pre_col, dead_gws, full_profile_keys)
        return pre_df, post_df

    def _apply_death_syncs(self, pre_df, post_df):
        if not self.overrides:
            return pre_df, post_df

        dead_gws_trx, dead_gws_cb = self._parse_death_schedules()
        full_profile_keys = self.join_keys + ['fcpNumber', 'attemptNumber']

        for m in range(6):
            if dead_gws_cb[m]:
                cb_post_cols = [f'Reallocated_t{t}_fcast_m{m}' for t in range(10)]
                cb_pre_cols = [f'PreSim_t{t}_fcast_m{m}' for t in range(10)]
                post_df = self._sync_doomed_cohorts(pre_df, post_df, dead_gws_cb[m], full_profile_keys, cb_post_cols)
                pre_df, post_df = self._redistribute_month_volume(pre_df, post_df, dead_gws_cb[m], full_profile_keys, cb_pre_cols, cb_post_cols)

            if dead_gws_trx[m]:
                trx_post_cols = [f'Reallocated_fc_mc_trx_m{m}']
                trx_pre_cols = [f'PreSim_fc_mc_trx_m{m}']
                post_df = self._sync_doomed_cohorts(pre_df, post_df, dead_gws_trx[m], full_profile_keys, trx_post_cols)
                pre_df, post_df = self._redistribute_month_volume(pre_df, post_df, dead_gws_trx[m], full_profile_keys, trx_pre_cols, trx_post_cols)

        num_cols_post = post_df.select_dtypes(include=['number']).columns
        post_df[num_cols_post] = post_df[num_cols_post].fillna(0.0).astype(np.float64)
        num_cols_pre = pre_df.select_dtypes(include=['number']).columns
        pre_df[num_cols_pre] = pre_df[num_cols_pre].fillna(0.0).astype(np.float64)

        return pre_df, post_df

    # =========================================================================
    # === MICRO-CHUNKED ALLOCATION
    # =========================================================================

    def _process_allocation_batch(self, sub_merged, sub_mapped, is_pre):
        moved_res, remain_res = self._process_allocation(sub_merged, sub_mapped, is_pre)
        sub_df = pd.concat([moved_res, remain_res], ignore_index=True)
        num_cols = sub_df.select_dtypes(include='number').columns
        sub_df = sub_df.loc[(sub_df[num_cols] > 0).any(axis=1)]
        group_cols = self.join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber']
        return sub_df.groupby(group_cols, as_index=False, observed=True).sum()

    def _compress_chunks_if_needed(self, chunks, limit=5):
        if len(chunks) >= limit:
            group_cols = self.join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber']
            condensed = pd.concat(chunks, ignore_index=True).groupby(group_cols, as_index=False, observed=True).sum()
            return [condensed]
        return chunks

    def _format_unmapped_fallbacks(self, unmapped_agg):
        valid_agg_cols = self.valid_agg_cols
        unmapped_post = unmapped_agg.rename(columns={'gatewayFid': 'finalGateway'})
        unmapped_pre = unmapped_agg.rename(columns={'gatewayFid': 'finalGateway'})

        post_cols = [f'Reallocated_{c}' for c in valid_agg_cols]
        pre_cols = [f'PreSim_{c}' for c in valid_agg_cols]

        for c, orig_c in zip(post_cols, valid_agg_cols):
            unmapped_post[c] = unmapped_agg[orig_c].values.astype(np.float64) if orig_c in unmapped_agg.columns else 0.0
        for c, orig_c in zip(pre_cols, valid_agg_cols):
            unmapped_pre[c] = unmapped_agg[orig_c].values.astype(np.float64) if orig_c in unmapped_agg.columns else 0.0

        unmapped_post = unmapped_post.loc[(unmapped_post[post_cols] > 0).any(axis=1)]
        unmapped_pre = unmapped_pre.loc[(unmapped_pre[pre_cols] > 0).any(axis=1)]
        return unmapped_post, unmapped_pre, post_cols, pre_cols

    def _assemble_final_allocation(self, chunks, unmapped_res, target_cols):
        group_cols = self.join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber']
        combined_list = chunks + [unmapped_res[group_cols + target_cols]]
        final_df = pd.concat(combined_list, ignore_index=True)
        return final_df.groupby(group_cols, as_index=False, observed=True).sum()

    def _execute_micro_chunked_allocation(self, mapped_agg, unmapped_agg, split_work, batch_size=50000):
        join_keys = self.join_keys
        post_chunks, pre_chunks = [], []
        mapped_agg = mapped_agg.reset_index(drop=True)
        total_rows = len(mapped_agg)

        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            sub_mapped = mapped_agg.iloc[start_idx:end_idx].copy()
            sub_merged = pd.merge(sub_mapped, split_work[split_work['RPGT'].isin(sub_mapped['RPGT'].unique())], on=join_keys, how='inner')

            if sub_merged.empty:
                continue

            post_chunks.append(self._process_allocation_batch(sub_merged, sub_mapped, False))
            pre_chunks.append(self._process_allocation_batch(sub_merged, sub_mapped, True))

            post_chunks = self._compress_chunks_if_needed(post_chunks)
            pre_chunks = self._compress_chunks_if_needed(pre_chunks)

        unmapped_post, unmapped_pre, post_cols, pre_cols = self._format_unmapped_fallbacks(unmapped_agg)

        post_df = self._assemble_final_allocation(post_chunks, unmapped_post, post_cols)
        pre_df = self._assemble_final_allocation(pre_chunks, unmapped_pre, pre_cols)
        return post_df, pre_df

    # =========================================================================
    # === MONTH-0 HISTORY RE-APPEND (Mastercard)
    # =========================================================================

    def _append_unaltered_month0(self, pre_df, post_df):
        """🟢 Append the unaltered injected Month-0 history back into the pre/post matrices."""
        join_keys = self.join_keys
        cols_to_pull = [c if c != 'RPGT' else 'rpgt' for c in join_keys] + ['gatewayFid', 'fcpNumber', 'attemptNumber', 'fc_mc_trx_m0']
        m0_hist = self.attempts_df[[c for c in cols_to_pull if c in self.attempts_df.columns]].copy()
        if 'rpgt' in m0_hist.columns:
            m0_hist.rename(columns={'rpgt': 'RPGT'}, inplace=True)
        if 'gatewayFid' in m0_hist.columns:
            m0_hist.rename(columns={'gatewayFid': 'finalGateway'}, inplace=True)

        actual_merge_keys = [c for c in join_keys + ['finalGateway', 'fcpNumber', 'attemptNumber'] if c in m0_hist.columns]

        # Match the key normalisation used across the allocation matrices (lower/strip + drop '.0')
        def _norm(df, cols):
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.lower().str.strip().str.replace(r'\.0$', '', regex=True)
            return df
        m0_hist = _norm(m0_hist, actual_merge_keys)

        m0_hist = m0_hist.groupby(actual_merge_keys, as_index=False, observed=True)['fc_mc_trx_m0'].sum()

        post_df = pd.merge(post_df, m0_hist, on=actual_merge_keys, how='outer')
        post_df['Reallocated_fc_mc_trx_m0'] = post_df['fc_mc_trx_m0'].fillna(0.0).astype(np.float64)
        post_df.drop(columns=['fc_mc_trx_m0'], inplace=True, errors='ignore')

        pre_df = pd.merge(pre_df, m0_hist, on=actual_merge_keys, how='outer')
        pre_df['PreSim_fc_mc_trx_m0'] = pre_df['fc_mc_trx_m0'].fillna(0.0).astype(np.float64)
        pre_df.drop(columns=['fc_mc_trx_m0'], inplace=True, errors='ignore')

        post_df[post_df.select_dtypes(include=['number']).columns] = post_df.select_dtypes(include=['number']).fillna(0.0)
        pre_df[pre_df.select_dtypes(include=['number']).columns] = pre_df.select_dtypes(include=['number']).fillna(0.0)
        return pre_df, post_df

    # =========================================================================
    # === PUBLIC RUN ENTRYPOINT
    # =========================================================================

    def execute_time_aware_routing(self, batch_size: int = 50000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("🚀 SECTION 8: TIME-AWARE ALLOCATION ENGINE (Mastercard, sparse matrix optimized)")

        df_in, split_work = self._prepare_allocation_matrices()
        split_work = self._inject_dynamic_snapshots(split_work)
        split_work = self._stitch_timeline(split_work)

        mapped_agg, unmapped_agg = self._map_and_filter_cohorts(df_in, split_work)
        del df_in

        logger.info("   ... Initiating micro-chunked allocation matrix ...")
        post_df, pre_df = self._execute_micro_chunked_allocation(mapped_agg, unmapped_agg, split_work, batch_size=batch_size)

        logger.info("   > 💀 Syncing death penalties and redistributing orphans...")
        pre_df, post_df = self._apply_death_syncs(pre_df, post_df)

        logger.info("   ... Appending unaltered Month 0 history to final matrices ...")
        pre_df, post_df = self._append_unaltered_month0(pre_df, post_df)

        return pre_df, post_df
