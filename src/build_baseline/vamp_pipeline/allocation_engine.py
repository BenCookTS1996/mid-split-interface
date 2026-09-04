import pandas as pd
import numpy as np
import re
import calendar
from typing import Dict, Any, Tuple, List, Optional

from .utils import setup_logger, clean_key_col

# ── 19kg: SETTINGS THAT USED TO BE ENVIRONMENT SWITCHES ──────────────────
# No environment variable changes a run any more. Each name below is frozen at the
# value the shipped run already used - the defaults, because no routing.env exists and
# run.command exports nothing - so what shipped is what these say. They stay NAMES, not
# literals inlined at the use site, for two reasons: a test can still A/B a whole search
# by rebinding one, and a reader can see in one place every decision this module makes.
# Changing behaviour now means editing this block and saying so in a commit.
_SW_DEATHSYNC_AUDIT = True   # was ROUTING_DEATHSYNC_AUDIT, default '1'
_SW_DEATHSYNC_BLOCK = True   # was ROUTING_DEATHSYNC_BLOCK, default '1'
_SW_DEATHSYNC_CASCADE = True   # was ROUTING_DEATHSYNC_CASCADE, default '1'
_SW_DEATHSYNC_DOOMED = False   # was ROUTING_DEATHSYNC_DOOMED, default '0'
_SW_VAMP_ORIGIN_SHARE = True   # was ROUTING_VAMP_ORIGIN_SHARE, default '1'

logger = setup_logger(__name__)

__build__ = "2026-07-21-alloc-trace-rawsplit+profile-samples+vp02-vamp-origin+vp03-deathsync-block+vp04-orphan-cascade+ds05-audit+ds06-doomed-off"


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

    def _vamp_origin_kill(self) -> Dict[str, Tuple[float, float]]:
        """{gatewayFid: (off_rel, frac)} for every gateway switched OFF FOR TRANSACTIONS.

        `off_rel` is the override's effective month relative to month_0 (negative = before the
        forecast window); `frac` is the fraction of that month the gateway was still trading, so
        the switch-off month itself is pro-rated rather than kept or killed whole. A gateway with
        no effective date was never trading in any month the forecast covers, so `off_rel` is set
        below any reachable origin.

        Scope matches vp01 exactly: `target: 0` with `apply_to` in (trx, both). An
        `apply_to: "vamp"` override is the OPPOSITE instruction — that gateway still TAKES
        transactions and is merely barred from holding VAMP — and `_inject_dynamic_snapshots`
        already zeroes its `Share_Vamp` outright.
        """
        if getattr(self, "_vok_cache", None) is not None:
            return self._vok_cache
        out = {}
        for fid, cfg in (self.overrides or {}).items():
            if not isinstance(cfg, dict):
                continue
            if float(cfg.get('target', 0) or 0) != 0.0:
                continue
            if str(cfg.get('apply_to', 'both')).lower() not in ('trx', 'both'):
                continue
            eff = cfg.get('effective_date')
            if eff:
                D = pd.to_datetime(eff)
                off_rel = (D.year - self.m0_start.year) * 12 + (D.month - self.m0_start.month)
                # Transactions ON the effective date are already gone, hence day - 1.
                frac = float(max(D.day - 1, 0)) / float(calendar.monthrange(D.year, D.month)[1])
            else:
                off_rel, frac = -10 ** 6, 0.0
            out[str(fid).strip().lower()] = (float(off_rel), frac)
        self._vok_cache = out
        return out

    def _origin_aware_vamp_shares(self, chunk_merged_df: pd.DataFrame, dest_gw_col: str,
                                  idx_src: List[str], share_vamp: np.ndarray):
        """Per-origin copies of `Share_Norm_Vamp` with switched-off destinations removed.

        Returns `(shares_by_origin, ctx)`, or `(None, None)` when nothing in this chunk is
        affected — in which case the caller keeps the original array and the result is
        bit-identical to the pre-vp02 engine. `ctx` carries only what the log-only accounting
        needs.

        Only origins 0..5 are built: the mover runs on `t <= m` profiles only, and there
        `origin = m - t` is always in [0, m] and so in [0, 5].
        """
        import os as _os_vo
        if (not _SW_VAMP_ORIGIN_SHARE):
            return None, None
        vok = self._vamp_origin_kill()
        if not vok or dest_gw_col not in chunk_merged_df.columns:
            return None, None

        dest_l = chunk_merged_df[dest_gw_col].astype(str).str.strip().str.lower()
        off = dest_l.map({k: v[0] for k, v in vok.items()}).to_numpy(dtype=np.float64)
        frc = dest_l.map({k: v[1] for k, v in vok.items()}).to_numpy(dtype=np.float64)
        aff = ~np.isnan(off)
        if not aff.any():
            return None, None
        # Unaffected destinations get off = +inf, so every origin compares "<" and their scale
        # stays exactly 1.0 without any NaN ever entering a comparison.
        off = np.where(aff, off, np.inf)
        frc = np.where(aff, np.nan_to_num(frc), 0.0)

        # Renormalisation group = one source profile x one timeline slice, i.e. exactly the set of
        # destination rules `_normalize_shares` normalised over.
        gcols = [c for c in (idx_src + ['GO LIVE']) if c in chunk_merged_df.columns]
        gid = chunk_merged_df.groupby(gcols, observed=True, sort=False).ngroup().to_numpy()
        if gid.size == 0:
            return None, None
        if (gid < 0).any():                      # NaN in a group key -> own singleton group
            bad = gid < 0
            gid = gid.copy()
            gid[bad] = (int(gid.max()) + 1) + np.arange(int(bad.sum()))
        ng = int(gid.max()) + 1

        S = np.bincount(gid, weights=share_vamp, minlength=ng)
        out = {}
        for o in range(6):
            scale = np.ones(len(share_vamp), dtype=np.float64)
            scale[o > off] = 0.0
            eq = (o == off)
            scale[eq] = frc[eq]
            w = share_vamp * scale
            Sp = np.bincount(gid, weights=w, minlength=ng)
            with np.errstate(divide='ignore', invalid='ignore'):
                fac = np.where(Sp > 0.0, S / Sp, 0.0)
            out[o] = w * fac[gid]

        ai = np.flatnonzero(aff)
        codes, uniq = pd.factorize(dest_l.to_numpy()[ai])
        return out, (ai, codes.astype(np.intp), [str(u) for u in uniq])

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

        # vp02: a gateway switched off for TRANSACTIONS cannot be the destination for VAMP whose
        # ORIGIN transaction post-dates the switch-off, because it could not have taken that
        # transaction. `_vo_share[o]` is `share_vamp` with those destinations removed for origin
        # month `o`, renormalised onto the survivors in the same profile. None => nothing in this
        # chunk is affected and the original array is used unchanged (bit-identical).
        _vo_share, _vo_ctx = self._origin_aware_vamp_shares(
            chunk_merged_df, dest_gw_col, idx_src, share_vamp)
        
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
                    if t > m:
                        vamp_move_ratio = 0.0            # historic origin - never moved
                    elif _vo_share is None:
                        vamp_move_ratio = share_vamp
                    else:
                        # vp02: origin month of this profile is m - t, always in [0, 5] when t <= m
                        vamp_move_ratio = _vo_share[m - t]
                    # 🟢 UPGRADED TO FLOAT64
                    _vamp_vals = chunk_merged_df[f't{t}_fcast_m{m}'].values.astype(np.float64)
                    if _vo_share is not None and t <= m:
                        self._vo_note(_vo_ctx, _vamp_vals * w_active, share_vamp,
                                      vamp_move_ratio, m, is_pre)
                    calc_arrays[target_col] = _vamp_vals * w_active * np.where(vamp_move_ratio > 0, vamp_move_ratio, 0)

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

    def _ds_ladder(self, full_keys: List[str], df: pd.DataFrame) -> List[List[str]]:
        """Cohort grains from finest to coarsest, giving up the least informative cut first."""
        drops = ([],
                 ['renewal_number'],
                 ['renewal_number', 'BIN'],
                 ['renewal_number', 'BIN', 'fcpNumber', 'attemptNumber'],
                 ['renewal_number', 'BIN', 'fcpNumber', 'attemptNumber',
                  'paymentMethodProvider', 'Country'],
                 ['renewal_number', 'BIN', 'fcpNumber', 'attemptNumber',
                  'paymentMethodProvider', 'Country', 'RPGT'],
                 ['renewal_number', 'BIN', 'fcpNumber', 'attemptNumber',
                  'paymentMethodProvider', 'Country', 'RPGT', 'Currency'])
        out = []
        for d in drops:
            cols = [k for k in full_keys if k not in d and k in df.columns]
            if cols and (not out or cols != out[-1]):
                out.append(cols)
        return out

    def _ds_gid_getter(self, df: pd.DataFrame, ladder: List[List[str]]):
        """Lazy, cached group ids per rung. A rung nothing reaches is never computed."""
        cache = [None] * len(ladder)

        def get(i):
            if cache[i] is None:
                g = df.groupby(ladder[i], observed=True, sort=False).ngroup().to_numpy()
                if g.size and (g < 0).any():            # NaN in a key -> own singleton group
                    g = g.copy()
                    bad = g < 0
                    g[bad] = (int(g.max()) + 1) + np.arange(int(bad.sum()))
                cache[i] = g
            return cache[i]
        return get

    def _ds_book(self, df: pd.DataFrame, cols: List[str]) -> float:
        """Total across `cols`. Read-only — used by the ds05 audit and nothing else."""
        _cs = [c for c in cols if c in df.columns]
        if not _cs:
            return 0.0
        return float(df[_cs].to_numpy(dtype=np.float64).sum())

    def _ds_stat(self, audit_key):
        """Accumulator for the ds05 audit. Log-only; never read by any calculation."""
        if audit_key is None:
            return None
        _a = getattr(self, "_ds_audit", None)
        if _a is None:
            _a = self._ds_audit = {}
        return _a.setdefault(audit_key, {"orphan": 0.0, "rung0": 0.0, "rungs": 0.0,
                                         "lost": 0.0, "nocasc": 0.0})

    def _ram_safe_redistribute(self, df: pd.DataFrame, col: str, dead_gws: set, group_cols: List[str],
                               blocked_gws: Optional[set] = None, gid_get=None,
                               ladder: Optional[List[List[str]]] = None, audit_key=None):
        """The Lossless Load-Balancer. Pushes stranded volume to surviving gateways.

        `dead_gws` are stripped AND excluded from absorbing. `blocked_gws` (vp03) are excluded
        from absorbing ONLY — they keep everything they already hold, because "this gateway
        processes nothing" must not become "confiscate what it earned while it was trading".

        vp04: a blocked gateway is excluded at EVERY grain, with no fallback. Orphaned volume
        that finds no eligible absorber in its own cohort walks `ladder` to coarser cohorts until
        it reaches gateways that actually process volume.
        """
        if col not in df.columns: return
        mask_dead = df['finalGateway'].isin(dead_gws)
        if not mask_dead.any(): return

        orphan_totals = df.loc[mask_dead].groupby(group_cols, observed=True)[col].sum().to_dict()
        dead_vals = df.loc[mask_dead, col].to_numpy(dtype=np.float64).copy()
        df.loc[mask_dead, col] = 0.0

        _st = self._ds_stat(audit_key)                      # ds05: log-only
        if _st is not None:
            _st["orphan"] += float(dead_vals.sum())

        cascade = gid_get is not None and bool(ladder)
        mask_alive = ~mask_dead
        if blocked_gws:
            blk = mask_alive & df['finalGateway'].isin(blocked_gws)
            if blk.any():
                if cascade:
                    # Blocked at EVERY grain. Volume that finds no absorber here walks the ladder
                    # rather than falling back onto a gateway that processes nothing.
                    mask_alive = mask_alive & ~blk
                else:
                    # `_SW_DEATHSYNC_CASCADE = False`: the vp03 fallback, kept verbatim so the switch
                    # reverts exactly one change. Skips the block where it would destroy volume.
                    _sub = df.loc[mask_alive, group_cols].copy()
                    _sub['_elig'] = (~blk.loc[mask_alive]).to_numpy(dtype=np.float64)
                    _n_elig = _sub.groupby(group_cols, observed=True)['_elig'].transform('sum').to_numpy()
                    _drop = blk.loc[mask_alive].to_numpy() & (_n_elig > 0.0)
                    if _drop.any():
                        mask_alive = mask_alive.copy()
                        mask_alive.loc[df.index[mask_alive][_drop]] = False

        # --- rung 0: the pre-vp04 computation, untouched ---------------------------------
        alive_totals = df.loc[mask_alive].groupby(group_cols, observed=True)[col].transform('sum').values
        alive_counts = df.loc[mask_alive].groupby(group_cols, observed=True)[col].transform('count').values
        orphan_mapped = df.loc[mask_alive].set_index(group_cols).index.map(orphan_totals).fillna(0.0).values

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(alive_totals > 0, df.loc[mask_alive, col].values / alive_totals, np.where(alive_counts > 0, 1.0 / alive_counts, 0.0))
            bonus = ratios * orphan_mapped

        # 🟢 UPGRADED TO FLOAT64
        df.loc[mask_alive, col] += bonus.astype(np.float64)

        if gid_get is None or not ladder:
            if _st is not None:
                # No ladder on this call, so rung-0 placement cannot be separated from what the
                # old code silently dropped. Flagged rather than guessed at.
                _st["nocasc"] += 1.0
            return

        # --- what rung 0 could not place ------------------------------------------------
        elig = mask_alive.to_numpy()
        dead = mask_dead.to_numpy()
        g0 = gid_get(0)
        n0 = int(g0.max()) + 1 if g0.size else 0
        if n0 <= 0:
            return
        have0 = np.bincount(g0, weights=elig.astype(np.float64), minlength=n0) > 0.0
        left = np.zeros(len(df), dtype=np.float64)
        left[dead] = np.where(have0[g0[dead]], 0.0, dead_vals)
        if _st is not None:
            _st["rung0"] += float(dead_vals.sum() - left.sum())
        if not left.any():
            return

        stats = getattr(self, "_ds_casc", None)
        if stats is None:
            stats = self._ds_casc = {"levels": [0.0] * len(ladder), "lost": 0.0}

        # --- walk the ladder with what is left ------------------------------------------
        for lvl in range(1, len(ladder)):
            g = gid_get(lvl)
            ng = int(g.max()) + 1 if g.size else 0
            if ng <= 0:
                break
            O = np.bincount(g, weights=left, minlength=ng)
            if not (O > 0.0).any():
                break
            vals = df[col].to_numpy(dtype=np.float64)
            E = np.bincount(g, weights=np.where(elig, vals, 0.0), minlength=ng)
            C = np.bincount(g, weights=elig.astype(np.float64), minlength=ng)
            has = C > 0.0
            with np.errstate(divide='ignore', invalid='ignore'):
                Eg, Cg = E[g], C[g]
                r = np.where(Eg > 0.0, vals / Eg, np.where(Cg > 0.0, 1.0 / Cg, 0.0))
                add = np.where(elig & has[g], r * O[g], 0.0)
            placed = float(np.where(has[g] & (left > 0.0), left, 0.0).sum())
            if placed <= 0.0:
                continue
            df[col] = vals + add
            stats["levels"][lvl] += placed
            if _st is not None:
                _st["rungs"] += placed
            left = np.where(has[g], 0.0, left)
            if not left.any():
                return
        stats["lost"] += float(left.sum())
        if _st is not None:
            _st["lost"] += float(left.sum())

    def _vo_note(self, ctx, weighted_vals, old_ratio, new_ratio, m: int, is_pre: bool) -> None:
        """Accumulate, per switched-off gateway, the VAMP vp02 withheld from it. LOG-ONLY.

        Vectorised on purpose: this runs 21 times per chunk per pre/post pass, so the per-row
        Python loop it replaces would have cost tens of millions of iterations on a live book.
        """
        try:
            ai, codes, names = ctx
            if ai.size == 0:
                return
            d = weighted_vals[ai] * (old_ratio[ai] - new_ratio[ai])
            if not d.any():
                return
            sums = np.bincount(codes, weights=d, minlength=len(names))
            rep = getattr(self, "_vo_rep", None)
            if rep is None:
                rep = self._vo_rep = {}
            for _i, _nm in enumerate(names):
                if sums[_i] != 0.0:
                    rep.setdefault((_nm, bool(is_pre)), [0.0] * 6)[m] += float(sums[_i])
        except Exception as _e_vo:  # noqa: BLE001
            logger.info(f"   > vamp-origin accounting skipped ({_e_vo})")

    def _vo_report(self) -> None:
        """Print what vp02 withheld and where it went instead. LOG-ONLY."""
        rep = getattr(self, "_vo_rep", None)
        if not rep:
            if self._vamp_origin_kill():
                logger.info("   > VAMP ORIGIN SHARE (vp02): nothing withheld - no switched-off "
                            "gateway was a destination for forecast-window VAMP.")
            return
        logger.info("   > VAMP ORIGIN SHARE (vp02): a gateway switched off for transactions "
                    "cannot be the destination for VAMP whose ORIGIN transaction post-dates the "
                    "switch-off. Withheld below and renormalised onto the live destinations in "
                    "the same profile - each profile's VAMP TOTAL IS UNCHANGED, only who holds it:")
        for _pre in (True, False):
            rows = [(k[0], v) for k, v in rep.items() if k[1] is _pre and abs(sum(v)) > 1e-9]
            if not rows:
                continue
            logger.info(f"        [{'PRE ' if _pre else 'POST'}]      "
                        + "  ".join(f"{'M%d' % m:>7s}" for m in range(6)) + "     total")
            for fid, v in sorted(rows, key=lambda r: -sum(r[1])):
                logger.info(f"        {fid:26s} " + " ".join(f"{x:7.2f}" for x in v)
                            + f"  {sum(v):9.2f}")
        logger.info("     `_SW_VAMP_ORIGIN_SHARE = False` reverts this entirely.")

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

        import os as _os_ds
        _ds_block_on = _SW_DEATHSYNC_BLOCK
        _ds_blocked_any = sorted(set().union(*[set(dead_trx[m]) - set(dead_vamp[m])
                                               for m in range(6)])) if _ds_block_on else []

        # vp04: orphaned VAMP that finds no eligible absorber in its own cohort widens the search
        # instead of falling back onto a gateway that processes nothing. Group ids per rung are
        # built lazily and cached for the whole phase, so an unreached rung costs nothing.
        _casc_on = _SW_DEATHSYNC_CASCADE
        _ds_ladder = self._ds_ladder(full_keys, post_df) if _casc_on else None
        _gid_post = self._ds_gid_getter(post_df, _ds_ladder) if _casc_on else None
        _gid_pre = self._ds_gid_getter(pre_df, _ds_ladder) if _casc_on else None

        _aud_on = _SW_DEATHSYNC_AUDIT
        # ds06: default OFF. =1 restores the pre-ds06 guard exactly.
        _dm_on = _SW_DEATHSYNC_DOOMED
        _aud_rows = []

        for m in range(6):
            if dead_vamp[m]:
                v_post, v_pre = [f'Reallocated_t{t}_fcast_m{m}' for t in range(10)], [f'PreSim_t{t}_fcast_m{m}' for t in range(10)]
                # ds05 (log-only): the VAMP on the way IN, before anything in this block runs.
                _b_post = self._ds_book(post_df, v_post) if _aud_on else 0.0
                _b_pre = self._ds_book(pre_df, v_pre) if _aud_on else 0.0
                # ds06: the `doomed` guard is OFF. It zeroed POST for cohorts whose every row
                # is a gateway barred from holding VAMP — before the redistribution ran, and to
                # POST only. That was a duplicate of the redistribution's own first move
                # (`df.loc[mask_dead, col] = 0.0`) and was harmless while neither frame could
                # place the volume. Once vp04 gave PRE somewhere to put it, this became the only
                # thing destroying VAMP: measured at 3,851.39 units, exactly the POST shortfall,
                # with resid 0.00 on every row of the ds05 audit.
                _doomed_zeroed = 0.0
                _doomed_skipped = 0.0
                if _dm_on or _aud_on:
                    pre_totals = pre_df.groupby(full_keys, observed=True).size()
                    pre_deads = pre_df[pre_df['finalGateway'].isin(dead_vamp[m])].groupby(full_keys, observed=True).size()
                    doomed = _doomed_keys(pre_totals, pre_deads)
                    if len(doomed) > 0:
                        post_mask = post_df.set_index(full_keys).index.isin(doomed)
                        _amt = float(sum(
                            float(post_df.loc[post_mask, col].sum())
                            for col in v_post if col in post_df.columns)) if _aud_on else 0.0
                        if _dm_on:
                            _doomed_zeroed = _amt
                            for col in v_post:
                                if col in post_df.columns: post_df.loc[post_mask, col] = 0.0
                        else:
                            # Not deleted — reported so the audit can show what it would have cost.
                            _doomed_skipped = _amt
                # vp03: a gateway switched off for TRANSACTIONS may keep what it earned but
                # must not absorb anyone else's orphaned VAMP - it processes nothing, so nothing
                # new can originate on it. Excluded from the absorbing pool only, never stripped.
                _blocked = (set(dead_trx[m]) - set(dead_vamp[m])) if _ds_block_on else None
                _k_post = (m, "POST") if _aud_on else None
                _k_pre = (m, "PRE ") if _aud_on else None
                for po_col, pr_col in zip(v_post, v_pre):
                    self._ram_safe_redistribute(post_df, po_col, dead_vamp[m], full_keys,
                                                _blocked, _gid_post, _ds_ladder, _k_post)
                    self._ram_safe_redistribute(pre_df, pr_col, dead_vamp[m], full_keys,
                                                _blocked, _gid_pre, _ds_ladder, _k_pre)

                if _aud_on:
                    # The VAMP on the way OUT. `delta` must be zero — this is a redistribution.
                    _aud_rows.append((m, "POST", _b_post, _doomed_zeroed,
                                      self._ds_book(post_df, v_post), _k_post, _doomed_skipped))
                    _aud_rows.append((m, "PRE ", _b_pre, 0.0,
                                      self._ds_book(pre_df, v_pre), _k_pre, 0.0))

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

        if _ds_blocked_any:
            logger.info("   > DEATH-SYNC BLOCK (vp03): these gateways are switched off for "
                        "TRANSACTIONS, so they no longer absorb VAMP orphaned from gateways that "
                        "may not hold any, at ANY cohort grain. They keep everything they earned "
                        "- this only stops NEW attribution: " + ", ".join(_ds_blocked_any))
        if _aud_rows:
            _A = getattr(self, "_ds_audit", None) or {}
            logger.info("   > DEATH-SYNC CONSERVATION AUDIT (ds05, log-only — changes no number). "
                        "VAMP through _apply_death_syncs. `delta` MUST be 0: this is a "
                        "redistribution. `resid` = delta + doomed + lost; if resid is non-zero the "
                        "loss is somewhere this audit does not yet name.")
            logger.info(f"        {'m':>2s} {'frame':5s} {'before':>11s} {'orphan':>9s} "
                        f"{'doomed':>9s} {'rung0':>9s} {'rung1+':>9s} {'lost':>8s} "
                        f"{'after':>11s} {'delta':>9s} {'resid':>9s}")
            _tot = {k: 0.0 for k in ("orphan", "doomed", "rung0", "rungs", "lost", "delta", "resid")}
            _skipped = sum(_r[6] for _r in _aud_rows)
            for _m, _fr, _b, _dz, _af, _k, _sk in _aud_rows:
                _s = _A.get(_k, {"orphan": 0.0, "rung0": 0.0, "rungs": 0.0, "lost": 0.0,
                                 "nocasc": 0.0})
                _d = _af - _b
                _r = _d + _dz + _s["lost"]
                logger.info(f"        {_m:2d} {_fr:5s} {_b:11.2f} {_s['orphan']:9.2f} "
                            f"{_dz:9.2f} {_s['rung0']:9.2f} {_s['rungs']:9.2f} {_s['lost']:8.2f} "
                            f"{_af:11.2f} {_d:9.2f} {_r:9.2f}"
                            + ("   (no cascade on this call)" if _s["nocasc"] else ""))
                _tot["orphan"] += _s["orphan"]; _tot["doomed"] += _dz
                _tot["rung0"] += _s["rung0"]; _tot["rungs"] += _s["rungs"]
                _tot["lost"] += _s["lost"]; _tot["delta"] += _d; _tot["resid"] += _r
            logger.info(f"        TOTAL      orphan {_tot['orphan']:.2f} · doomed "
                        f"{_tot['doomed']:.2f} · rung0 {_tot['rung0']:.2f} · rung1+ "
                        f"{_tot['rungs']:.2f} · lost {_tot['lost']:.2f} · delta "
                        f"{_tot['delta']:.2f} · resid {_tot['resid']:.2f}")
            if _dm_on:
                logger.info("        doomed guard is ON (`_SW_DEATHSYNC_DOOMED = True`) — it is "
                            "deleting the volume above from POST and not from PRE, so the two "
                            "books will not agree. Set it False at the top of allocation_engine.py "
                            "to let the redistribution place it.")
            elif _skipped > 0.0:
                logger.info(f"        doomed guard OFF (ds06): {_skipped:.2f} unit(s) that the old "
                            "guard would have deleted from POST are now redistributed exactly as "
                            "PRE does. `doomed` and `delta` above should both read 0.00.")
            if abs(_tot["resid"]) > 1e-6:
                logger.warning(f"   > DEATH-SYNC AUDIT: {_tot['resid']:.2f} unit(s) of VAMP are "
                               "unaccounted for — the book does not balance and the cause is NOT "
                               "the doomed pre-zeroing or an unplaceable orphan. Do not fix "
                               "either of those on the strength of this run.")
            logger.info("     `_SW_DEATHSYNC_AUDIT = False` silences this block.")

        _cs = getattr(self, "_ds_casc", None)
        if _cs and _ds_ladder:
            _tot = sum(_cs["levels"])
            if _tot > 0.0 or _cs["lost"] > 0.0:
                logger.info("   > ORPHAN CASCADE (vp04): VAMP with no eligible absorber in its own "
                            "cohort was retried against progressively coarser cohorts. Where each "
                            "rung caught it:")
                for _i, _cols in enumerate(_ds_ladder):
                    _v = _cs["levels"][_i]
                    if _i == 0 or _v <= 0.0:
                        continue
                    logger.info(f"        rung {_i}: {_v:10.2f} unit(s)  ->  "
                                + " x ".join(_cols))
                logger.info(f"        TOTAL {_tot:.2f} unit(s) re-homed this way. Read the rungs: "
                            "volume caught at a coarse rung is being spread across the company "
                            "rather than attributed to a comparable gateway, and DELETING it may "
                            "be the more honest treatment.")
            if _cs["lost"] > 1e-9:
                logger.warning(f"   > ORPHAN CASCADE: {_cs['lost']:.2f} unit(s) could not be "
                               "placed at ANY grain and have been dropped from the book - nothing "
                               "eligible exists to hold them. This needs looking at.")
            logger.info("     `_SW_DEATHSYNC_CASCADE = False` reverts to the vp03 fallback; "
                        "`_SW_DEATHSYNC_BLOCK = False` reverts both.")

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


        df_in, split_work = self._prepare_allocation_matrices()
        split_work = self._inject_dynamic_snapshots(split_work)
        split_work = self._stitch_timeline(split_work)
        
        mapped_agg, unmapped_agg = self._map_and_filter_cohorts(df_in, split_work)



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

        self._vo_report()

        logger.info("Applying Death Syncs & Lossless Redistribution...")
        pre_df, post_df = self._apply_death_syncs(pre_df, post_df)


        # 19kg: the ROUTING_ALLOC_TRACE / ROUTING_PROFILE_SAMPLES diagnostic dumps are DELETED
        # with their switches (171 lines over five blocks). Every one was armed by an env
        # string - "currency|bin|rpgt" - so none of them ever ran in a real pipeline; what they
        # were built to localise, the tab-3 vs tab-5 held-cohort gap, is reported every run by
        # the baseline reconciliation table and [rung] on the split that actually ships.
        return pre_df, post_df