import os
import calendar
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from google.cloud import bigquery

from .utils import setup_logger

logger = setup_logger(__name__)

# Bump this when data_extractor.py changes; the adapter logs it so a stale
# .pyc (old code) is immediately obvious in the run log.
__build__ = "2026-08-16-mastercard-initial+cell-level-catchall"


class DataExtractor:
    """
    Handles all external data fetching from BigQuery, local Excel/CSV files, and cache
    for the MASTERCARD pipeline. Also handles the initial transformation of the FP&A
    macro forecast into micro-cohorts.

    Differences vs the Visa/VAMP DataExtractor:
      * the historical transaction denominator is the MASTERCARD transaction count;
      * mapping percentages are computed on the 'mastercard' account_type slice;
      * a "Visa kill switch" zeroes forecast volume on Visa (BIN 4xx) cohorts;
      * MIDs map raw gateways to the mastercardMid used in the final export.
    """

    def __init__(self, config: Dict[str, Any], bq_client: bigquery.Client):
        self.config = config
        self.bq = bq_client

        self.company = str(self.config['run_settings']['company']).strip()
        self.month_var = str(self.config['run_settings']['month_var']).strip()
        self.m0_start_date = str(self.config['run_settings']['month_0_start_date']).strip()

        self.actuals_start = str(self.config['run_settings'].get('actuals_start_date', self.m0_start_date)).strip()
        self.actuals_end = str(self.config['run_settings'].get('actuals_end_date')).strip()

        # Mastercard-specific switches
        self.kill_4_bins = bool(self.config['run_settings'].get('kill_4_bins', True))
        self.actuals_end_day = int(self.config['run_settings'].get('actuals_end_day', 30))

        self.cache_path = self.config['paths']['cache_path'].format(month_var=self.month_var, company=self.company)
        self.chunked_dir = self.config['paths']['chunked_files_dir'].format(month_var=self.month_var, company=self.company)
        self.output_dir = self.config['paths'].get('output_dir', 'data/outputs/').format(month_var=self.month_var, company=self.company)

        self.split_rules_path = self.config['paths'].get('split_rules_file', '')
        self.mid_list_path = self.config['paths'].get('mid_list_file', '')

        self.profile_keys = ['Company', 'paymentMethodProvider', 'Country', 'rpgt', 'Currency', 'BIN', 'renewal_number', 'fcpNumber']

        self.fcast_data_df = pd.DataFrame()
        self.gw_mapping_df = pd.DataFrame()
        self.lt_fcast_mapping_df = pd.DataFrame()
        self.attempts_df = pd.DataFrame()
        self.mid_df = pd.DataFrame()
        self.split_df = pd.DataFrame()

        self.longterm_fcast_df = pd.DataFrame()

    def _query_to_df(self, query: str) -> pd.DataFrame:
        """Run a query and materialise a DataFrame.

        Tries the fast BigQuery Storage download first, and on the specific
        'readsessions.create' permission denial falls back to the slower REST
        download (which only needs ordinary query permissions). Results get cached
        to parquet, so the slower path is a one-off per cache-miss.
        """
        from google.api_core.exceptions import Forbidden, PermissionDenied
        job = self.bq.query(query)
        try:
            return job.to_dataframe()
        except (Forbidden, PermissionDenied) as e:  # noqa: BLE001
            if "readsessions.create" not in str(e):
                raise  # a real access problem — surface it loudly
            logger.warning(
                "   > ⚠️  No 'bigquery.readsessions.create' permission — the fast BigQuery "
                "Storage download is unavailable. Falling back to the slower REST download "
                "(affects speed only, not the result). Ask an admin for role "
                f"'roles/bigquery.readSessionUser' on project '{getattr(self.bq, 'project', '?')}'.")
            return job.to_dataframe(create_bqstorage_client=False)

    # =========================================================================
    # === 1. DATAFRAME FORMATTING HELPERS
    # =========================================================================

    def _clean_col_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names from disparate sources into one unified convention."""
        col_map = {
            'riskDefinedProductSubscriptionType': 'rpgt', 'risk_defined_subscription_product_type': 'rpgt',
            'RPGT': 'rpgt', 'company': 'Company', 'Brand': 'Company', 'brand': 'Company',
            'gateway_fid': 'gatewayFid', 'Gateway_Src': 'gatewayFid', 'Gateway': 'gatewayFid',
            'paymentmethodprovider': 'paymentMethodProvider', 'country': 'Country', 'bin': 'BIN',
            'Bin': 'BIN', 'currency': 'Currency', 'fcpnumber': 'fcpNumber', 'attemptnumber': 'attemptNumber'
        }
        rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)

        if 'BIN' in df.columns:
            df['BIN'] = df['BIN'].astype(str).str.split('.').str[0].str.strip()

        df['attemptNumber'] = df['attemptNumber'].astype(str).str.lower().str.strip() if 'attemptNumber' in df.columns else '1'
        return df

    def _fast_apply_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Concatenates demographic columns into a single 'prof_key' string per micro-cohort."""
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

    # =========================================================================
    # === 2. BIGQUERY EXTRACTION
    # =========================================================================

    def _read_sql_file(self, filename: str) -> str:
        """
        Reads a raw .sql file and injects the target boundary dates from
        settings_mastercard.yaml so BigQuery queries the correct time window.
        Replaces {MONTH_0_START_DATE}, {ACTUALS_START_DATE}, {ACTUALS_END_DATE}.
        """
        cfg_dir = self.config.get('paths', {}).get('queries_dir')
        # 19fr: FOUR dirnames, not three. This file moved from src/<pkg>/ to
        # src/build_baseline/<pkg>/, so the walk to the project root gained a level.
        # Three would have resolved to src/ and the queries/ fallback would have
        # looked in src/queries — a path that does not exist, reported as a missing
        # SQL file rather than as a moved package.
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        # 19fh: queries/ now has per-scheme subfolders. Scheme-specific SQL lives in
        # queries/mastercard/, and SQL used by BOTH schemes stays in queries/ itself. Each root is
        # tried at its SCHEME subfolder FIRST and then bare, so a scheme-specific file wins over
        # a same-named shared one and neither pipeline can pick up the other's query.
        _roots = ([cfg_dir] if cfg_dir else []) + ['queries',
                                                   os.path.join(pkg_root, 'queries')]
        candidates = []
        for _r in _roots:
            candidates.append(os.path.join(_r, 'mastercard', filename))
            candidates.append(os.path.join(_r, filename))
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if path is None:
            qd = os.path.join(pkg_root, 'queries')
            listing = ((sorted(os.listdir(qd)) +
                        [os.path.join('mastercard', _x)
                         for _x in sorted(os.listdir(os.path.join(qd, 'mastercard')))]
                        if os.path.isdir(os.path.join(qd, 'mastercard')) else sorted(os.listdir(qd)))
                       if os.path.isdir(qd) else "(dir does not exist)")
            detail = [
                f"SQL file '{filename}' could not be found.",
                f"  running module : {os.path.abspath(__file__)}",
                f"  module build   : {__build__}",
                f"  inferred root  : {pkg_root}",
                f"  working dir    : {os.getcwd()}",
                "  paths tried    :",
            ] + [f"     - {os.path.abspath(c)}  (exists={os.path.isfile(c)})"
                 for c in candidates] + [
                f"  contents of {qd}: {listing}",
                "",
                "If the queries/ files ARE present, you are running STALE cached bytecode. "
                "Delete every __pycache__ folder and fully restart:",
                "    find . -name __pycache__ -type d -exec rm -rf {} +",
            ]
            raise FileNotFoundError("\n".join(detail))
        logger.info(f"      - reading SQL: {path}")
        with open(path, 'r') as file:
            query = file.read()

        query = query.replace('{MONTH_0_START_DATE}', self.m0_start_date)
        query = query.replace('{ACTUALS_START_DATE}', self.actuals_start)
        query = query.replace('{ACTUALS_END_DATE}', self.actuals_end)
        return query

    @staticmethod
    def _txn_total(df) -> str:
        """Sum any transaction-count column(s) in df so each loaded source reports its total volume."""
        try:
            import re as _re
            _cols = [c for c in df.columns
                     if str(c).strip().lower() in ("mastercard_trx_count", "mc_trx_count",
                                                   "trx_count", "trx total", "trx_total")
                     or _re.match(r'^(presim_|reallocated_)?fc_mc_trx_m\d+$', str(c).strip().lower())]
            if not _cols or len(df) == 0:
                return ""
            _tot = float(sum(float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in _cols))
            return f" · Σ txn = {_tot:,.0f} [{', '.join(map(str, _cols[:4]))}{' …' if len(_cols) > 4 else ''}]"
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _log_head(df, label: str, n: int = 5) -> None:
        """Log df.head(n) so the CONTENT of each loaded/exported file is visible in the run log."""
        try:
            import pandas as _pd
            with _pd.option_context("display.max_columns", 20, "display.width", 200,
                                    "display.max_colwidth", 24):
                _txt = df.head(n).to_string(max_cols=20)
            logger.info(f"      head({n}) of {label}"
                        + (f"  [{len(df.columns)} cols, showing ≤20]:" if len(df.columns) > 20 else ":"))
            for _ln in _txt.splitlines():
                logger.info("        " + _ln)
        except Exception:  # noqa: BLE001
            pass

    def _fetch_bq_data(self, cache_filename: str, sql_filename: str, apply_keys: bool = True) -> pd.DataFrame:
        """Checks the local parquet cache; on miss, fetches from BigQuery, caches, and applies profile keys."""
        full_path = os.path.join(self.cache_path, cache_filename)
        if os.path.exists(full_path):
            _df_cached = pd.read_parquet(full_path)
            try:
                import datetime as _dt
                _sz_mb = os.path.getsize(full_path) / 1e6
                _mt = _dt.datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M")
                logger.info(f"   > Loading {cache_filename} from cache "
                            f"[{len(_df_cached):,} rows, {_sz_mb:.1f} MB, mtime {_mt}]"
                            f"{self._txn_total(_df_cached)}")
            except Exception:  # noqa: BLE001
                logger.info(f"   > Loading {cache_filename} from cache...")
            self._log_head(_df_cached, cache_filename)
            return _df_cached

        logger.info(f"   > 📡 Cache miss. Running BigQuery for {cache_filename}...")
        query = self._read_sql_file(sql_filename)
        df = self._query_to_df(query)
        logger.info(f"   > BigQuery returned {cache_filename}: {len(df):,} rows{self._txn_total(df)}")
        self._log_head(df, cache_filename)

        if apply_keys:
            logger.info(f"      - Generating Profile Keys before caching...")
            df = self._fast_apply_keys(df)

        os.makedirs(self.cache_path, exist_ok=True)
        df.to_parquet(full_path, index=False)
        return df

    # =========================================================================
    # === 3. HISTORICAL BASELINE GENERATION
    # =========================================================================

    def _derive_historical_attempts(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates the most recent two months of historical MASTERCARD transactions to
        establish a baseline of which gateway processed what volume (successCount).
        """
        logger.info("   > Deriving 'attempts_df' directly from historical mapping cache...")
        recent_history = mapping_df[mapping_df['period'].isin([0, 1])].copy()
        group_keys = ['Company', 'rpgt', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'renewal_number', 'fcpNumber', 'attemptNumber', 'gatewayFid', 'prof_key']
        actual_keys = [c for c in group_keys if c in recent_history.columns]

        attempts = recent_history.groupby(actual_keys, observed=True, as_index=False)['mastercard_trx_count'].sum()
        attempts = attempts.rename(columns={'mastercard_trx_count': 'successCount'})
        attempts['attemptCount'] = attempts['successCount']
        return attempts

    def _flex_attempts_day_count(self, attempts_df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes historical base volume when the target month has a different day-count than the source month."""
        target_dt = pd.to_datetime(self.m0_start_date)
        _, target_days = calendar.monthrange(target_dt.year, target_dt.month)
        source_dt = target_dt - pd.DateOffset(months=1)
        _, source_days = calendar.monthrange(source_dt.year, source_dt.month)
        day_flex_factor = target_days / source_days
        if day_flex_factor != 1.0:
            for col in ['successCount', 'attemptCount', 'cb_count']:
                if col in attempts_df.columns:
                    attempts_df[col] = pd.to_numeric(attempts_df[col], errors='coerce').fillna(0) * day_flex_factor
        return attempts_df

    # =========================================================================
    # === 4. LOCAL/URL EXTRACTION
    # =========================================================================

    def _fetch_local_mid_list(self) -> pd.DataFrame:
        """
        Loads the master MID mapping list from a local CSV, translating raw technical
        gateway names into the clean mastercardMid used in the final export.
        Expected columns: gatewayFid (or gateway_fid), mastercardMid (or mastercard_mid).
        """
        # 19fr: FOUR dirnames, not three. This file moved from src/<pkg>/ to
        # src/build_baseline/<pkg>/, so the walk to the project root gained a level.
        # Three would have resolved to src/ and the queries/ fallback would have
        # looked in src/queries — a path that does not exist, reported as a missing
        # SQL file rather than as a moved package.
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        candidates = [self.mid_list_path,
                      os.path.join(pkg_root, self.mid_list_path) if self.mid_list_path else None,
                      os.path.join(pkg_root, "data", "mappings", "Master_MID_List_Mastercard.csv")]
        path = next((c for c in candidates if c and os.path.isfile(c)), None)
        if path is None:
            logger.warning("MID list not found (looked in: %s); mastercardMid mapping will be empty.",
                           " ; ".join(os.path.abspath(c) for c in candidates if c))
            return pd.DataFrame(columns=['gatewayFid', 'mastercardMid'])
        try:
            logger.info(f"      - reading MID list: {path}")
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.rename(columns={'gateway_fid': 'gatewayFid', 'gatewayfid': 'gatewayFid',
                                    'mastercard_mid': 'mastercardMid', 'mastercardmid': 'mastercardMid'})
            if 'gatewayFid' not in df.columns or 'mastercardMid' not in df.columns:
                logger.error("MID list missing gatewayFid/mastercardMid columns; found %s", df.columns.tolist())
                return pd.DataFrame(columns=['gatewayFid', 'mastercardMid'])
            return df[['gatewayFid', 'mastercardMid']]
        except Exception as e:
            logger.error(f"MID list read failed: {e}")
            return pd.DataFrame(columns=['gatewayFid', 'mastercardMid'])

    def _fetch_mr_daily_weights(self) -> dict:
        """
        Fetches front-loaded MASTERCARD monthly-renewal daily billing weights from BigQuery
        to prorate mid-month kills. Returns {month:int -> {day:int -> weight:float}}.
        """
        cache_file = os.path.join(self.cache_path, 'mr_daily_weights.pkl')
        if os.path.exists(cache_file):
            logger.info("   > Loading MR Daily Weights from cache...")
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info("   > 📡 Cache miss. Running BigQuery for MR Daily Weights (mastercard_mr_weights.sql)...")
        query = self._read_sql_file('mastercard_mr_weights.sql')
        df = self._query_to_df(query)

        mr_daily_weights = {}
        for _, row in df.iterrows():
            raw_m = str(row['calendarMonth']).strip()
            m = int(raw_m.split('-')[1]) if '-' in raw_m else int(float(raw_m))
            d = int(float(row['recordDOM']))
            w = float(row['avg_daily_share'])
            if m not in mr_daily_weights:
                mr_daily_weights[m] = {}
            mr_daily_weights[m][d] = w

        import pickle
        os.makedirs(self.cache_path, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(mr_daily_weights, f)

        return mr_daily_weights

    def _fetch_chunked_rules(self) -> pd.DataFrame:
        """Loads routing rules from individual local .xlsx files, applying the future go-live threshold."""
        logger.info("   > [CHUNKED MODE] Loading rules from local .xlsx files...")
        if not os.path.exists(self.chunked_dir):
            logger.warning("   > ⚠️  SPLIT-RULES DIRECTORY NOT FOUND: %s — the forecast has NO routing "
                           "rules, so every MC-Txn output will be ZERO. Place this month's split-rule "
                           ".xlsx files there and re-run.", self.chunked_dir)
            return pd.DataFrame()

        excel_files = [f for f in os.listdir(self.chunked_dir) if f.endswith('.xlsx')]
        if not excel_files:
            logger.warning("   > ⚠️  NO .xlsx SPLIT-RULE FILES in %s — every MC-Txn output will be ZERO. "
                           "Add this month's rule files and re-run.", self.chunked_dir)
            return pd.DataFrame()
        df_list = []
        cat_cols = ['Company', 'Brand', 'company', 'brand', 'riskDefinedProductSubscriptionType', 'RPGT', 'rpgt', 'currency', 'Currency', 'paymentMethodProvider', 'paymentmethodprovider', 'STICKY', 'sticky', 'Country', 'country']
        blend_future = self.config['run_settings'].get('blend_future_sheet_rules', False)
        future_anchor = self.config['run_settings'].get('future_anchor_date')
        future_threshold = pd.to_datetime(future_anchor).normalize() if future_anchor else pd.to_datetime('today').normalize()

        for file in excel_files:
            try:
                temp_df = pd.read_excel(os.path.join(self.chunked_dir, file), sheet_name=0)
                temp_df.columns = temp_df.columns.astype(str).str.strip()
                temp_df = temp_df.dropna(how='all', axis=1).dropna(how='all', axis=0)

                if blend_future and 'GO LIVE' in temp_df.columns:
                    temp_df['GO_LIVE_DT'] = pd.to_datetime(temp_df['GO LIVE'], errors='coerce', dayfirst=True)
                    temp_df = temp_df[temp_df['GO_LIVE_DT'] >= future_threshold].drop(columns=['GO_LIVE_DT'])

                for c in cat_cols:
                    if c in temp_df.columns:
                        temp_df[c] = temp_df[c].astype('category')
                if not temp_df.empty:
                    df_list.append(temp_df)
            except Exception as e:
                logger.warning(f"Could not open {file}. Error: {e}")

        if not df_list:
            logger.warning("   > ⚠️  SPLIT-RULE files in %s were all empty / filtered out — MC-Txn volume "
                           "will be ZERO. Check the .xlsx contents and GO LIVE dates.", self.chunked_dir)
            return pd.DataFrame()
        return pd.concat(df_list, ignore_index=True)

    def _fetch_direct_rules(self) -> pd.DataFrame:
        """Extracts routing split percentages from a single company tab on a master Excel matrix."""
        if not self.split_rules_path:
            return pd.DataFrame()
        try:
            return pd.read_excel(self.split_rules_path, sheet_name=self.company)
        except Exception:
            return pd.DataFrame()

    # =========================================================================
    # === 5. MACRO FORECAST MATRIX BUILDER
    # =========================================================================

    def _apply_forecast_volume_overrides(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Applies manual risk overrides (scaling down or killing volume) to base transactions per gateway_volume_overrides."""
        overrides_dict = self.config.get('gateway_volume_overrides', {})
        if not overrides_dict:
            return mapping_df

        mapping_df['trx_count'] = pd.to_numeric(mapping_df['trx_count'], errors='coerce').fillna(0)
        m0_start_dt = pd.to_datetime(self.m0_start_date)

        for fid, cfg in overrides_dict.items():
            target_vol = cfg.get('target', 0) if isinstance(cfg, dict) else cfg
            apply_to = cfg.get('apply_to', 'both') if isinstance(cfg, dict) else 'both'
            eff_date = cfg.get('effective_date') if isinstance(cfg, dict) else None

            if apply_to in ['trx', 'both'] and (not eff_date or pd.to_datetime(eff_date) <= m0_start_dt):
                clean_fid = str(fid).strip().lower()
                mask_fc = mapping_df['gatewayFid'].astype(str).str.strip().str.lower() == clean_fid
                if mask_fc.any():
                    current_vol = mapping_df.loc[mask_fc, 'trx_count'].sum()
                    if current_vol > 0:
                        mapping_df.loc[mask_fc, 'trx_count'] *= (target_vol / current_vol)
                    elif target_vol == 0:
                        mapping_df.loc[mask_fc, 'trx_count'] = 0

        mapping_df['trx_count'] = mapping_df['trx_count'].round(5).astype(np.float64)
        return mapping_df

    def _calculate_mastercard_mapping_pct(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the historical baseline to pure MASTERCARD volume and calculates
        exactly what percentage of the total RPGT pie each micro-cohort represents.
        """
        mapping_df = mapping_df[mapping_df['Company'].astype(str).str.lower().str.strip() == self.company.lower().strip()].copy()
        mapping_df = mapping_df[mapping_df['account_type'].astype(str).str.lower().str.strip() == 'mastercard'].copy()
        mapping_df['trx total'] = mapping_df.groupby('rpgt')['trx_count'].transform('sum')
        mapping_df['mapping_pct'] = (mapping_df['trx_count'] / mapping_df['trx total']).replace([np.inf, -np.inf], 0).fillna(0)
        return mapping_df

    def _build_future_forecast_matrix(self, mapping_df: pd.DataFrame, fcast_df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Pivots the BigQuery macro forecast and multiplies by the mapping percentages to
        determine how many future MASTERCARD transactions belong to each micro-cohort.
        """
        m0_start_dt = pd.to_datetime(self.m0_start_date)
        target_months = [m0_start_dt + pd.DateOffset(months=i) for i in range(6)]

        fcast_df['fcast_units'] = fcast_df['fcast_units'].fillna(0).astype(float)
        fcast_df['period'] = pd.to_datetime(fcast_df['period'])
        fcast_df['rpgt'] = fcast_df['rpgt'].astype(str).str.lower().str.strip()

        pivoted_fcast = fcast_df[fcast_df['period'].isin(target_months)].copy()
        pivoted_fcast['period_str'] = pivoted_fcast['period'].dt.strftime('%Y-%m-%d')
        pivoted_fcast = pivoted_fcast.pivot(index='rpgt', columns='period_str', values='fcast_units').reset_index()

        final_df = pd.merge(mapping_df, pivoted_fcast, on='rpgt', how='left')
        trx_columns_to_keep = []

        for period_col in [dt.strftime('%Y-%m-%d') for dt in target_months]:
            if period_col in final_df.columns:
                new_col_name = f"{period_col}_trx"
                final_df[new_col_name] = (final_df[period_col] * final_df['mapping_pct']).fillna(0).round(5).astype(np.float64)
                trx_columns_to_keep.append(new_col_name)

        base_cols = ['prof_key', 'rpgt', 'account_type', 'BIN', 'Company', 'Currency', 'gatewayFid', 'paymentMethodProvider', 'Country', 'renewal_number', 'fcpNumber', 'attemptNumber']
        longterm_fcast_df = final_df[[col for col in base_cols + trx_columns_to_keep if col in final_df.columns]].copy()
        return longterm_fcast_df[longterm_fcast_df['account_type'] == 'mastercard'], trx_columns_to_keep

    def _apply_visa_kill_switch(self, longterm_fcast_df: pd.DataFrame) -> pd.DataFrame:
        """
        🟢 THE VISA KILL SWITCH (Mastercard-specific).
        Zeroes all forecast transaction columns for Visa BINs (starting with '4') so the
        Mastercard pipeline never routes Visa volume. Gated by run_settings.kill_4_bins.
        """
        if not self.kill_4_bins:
            return longterm_fcast_df
        trx_cols = [c for c in longterm_fcast_df.columns if '_trx' in c]
        if not trx_cols:
            return longterm_fcast_df
        mask_4_bins = longterm_fcast_df['BIN'].astype(str).str.startswith('4')
        n = int(mask_4_bins.sum())
        longterm_fcast_df.loc[mask_4_bins, trx_cols] = 0.0
        logger.info(f"   > 🟢 Mastercard filter: removed {n:,} Visa (BIN 4xx) cohort rows from the forecast.")
        return longterm_fcast_df

    def _melt_and_apply_targets(self, fcast_df: pd.DataFrame, trx_columns_to_keep: list) -> pd.DataFrame:
        """
        Melts the wide forecast matrix into long format (month_offset), merges the MID list,
        and forces Month 0 to match the FP&A macro target volume / RPGT mix.
        """
        fcast_df = pd.merge(fcast_df, self.mid_df, how='left', on='gatewayFid')
        id_vars = [col for col in fcast_df.columns if col not in trx_columns_to_keep]
        for col in id_vars:
            if fcast_df[col].dtype == 'object':
                fcast_df[col] = fcast_df[col].astype('category')

        melted_df = pd.melt(fcast_df, id_vars=id_vars, value_vars=trx_columns_to_keep, var_name='month_offset', value_name='forecasted_trx')
        melted_df['month_offset'] = melted_df['month_offset'].map({col: i for i, col in enumerate(trx_columns_to_keep)})

        targets_config = self.config.get('targets', {})
        target_volume = targets_config.get('company_target_volume')
        rpgt_mix_dict = targets_config.get('company_rpgt_target_volumes', {})

        mask_m0 = melted_df['month_offset'] == 0
        current_company_m0_volume = melted_df.loc[mask_m0, 'forecasted_trx'].sum()

        if current_company_m0_volume > 0 and target_volume:
            melted_df['forecasted_trx'] = melted_df['forecasted_trx'] * (target_volume / current_company_m0_volume)

        if rpgt_mix_dict:
            total_input_rpgt_vol = sum(rpgt_mix_dict.values())
            rpgt_mix = {k.strip().lower(): (v / total_input_rpgt_vol) * 100 for k, v in rpgt_mix_dict.items()} if total_input_rpgt_vol > 0 else {}

            final_company_m0_volume = melted_df.loc[mask_m0, 'forecasted_trx'].sum()
            melted_df['rpgt_lower'] = melted_df['rpgt'].astype(str).str.lower().str.strip()
            current_rpgt_vols_m0 = melted_df[mask_m0].groupby('rpgt_lower')['forecasted_trx'].sum()

            for rpgt, current_vol_m0 in current_rpgt_vols_m0.items():
                if rpgt in rpgt_mix:
                    target_vol_m0 = final_company_m0_volume * (rpgt_mix[rpgt] / 100.0)
                    if current_vol_m0 > 0:
                        melted_df.loc[melted_df['rpgt_lower'] == rpgt, 'forecasted_trx'] *= (target_vol_m0 / current_vol_m0)
                else:
                    melted_df.loc[melted_df['rpgt_lower'] == rpgt, 'forecasted_trx'] = 0
            melted_df = melted_df.drop(columns=['rpgt_lower'])

        return melted_df

    # =========================================================================
    # === 6. SPLIT RULE PROCESSING (Wide-to-Long)
    # =========================================================================

    def _clean_google_sheets_rules(self, sheet_df: pd.DataFrame) -> pd.DataFrame:
        """Melts the wide Excel routing matrix into a long engine-ready format, parsing percentage strings into floats."""
        if sheet_df.empty:
            return pd.DataFrame()
        sheet_df.columns = sheet_df.columns.astype(str).str.strip()
        sheet_df = sheet_df.loc[:, ~sheet_df.columns.duplicated()]

        cols_to_drop = [c for c in sheet_df.columns if (c.upper() in ['BIN GROUP', 'DUP CHECK'] or 'DUP CHECK' in c.upper()) and c != 'Check'] + ['Share_Str', 'share_str', 'Share']
        sheet_df.drop(columns=[c for c in cols_to_drop if c in sheet_df.columns], inplace=True)

        rule_col_map = {'Company': 'Brand', 'company': 'Brand', 'riskDefinedProductSubscriptionType': 'RPGT', 'rpgt': 'RPGT', 'currency': 'Currency', 'bin': 'BIN', 'paymentmethodprovider': 'paymentMethodProvider', 'country': 'Country'}
        sheet_df = sheet_df.rename(columns=rule_col_map)

        for c in ['Brand', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country']:
            if c in sheet_df.columns:
                sheet_df[c] = sheet_df[c].astype(str).str.lower().str.strip()
                if c == 'BIN':
                    sheet_df[c] = sheet_df[c].str.replace(r'\.0$', '', regex=True)

        id_vars = [c for c in ['Brand', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'Check', 'STICKY', 'GO LIVE'] if c in sheet_df.columns]

        split_df = sheet_df.melt(id_vars=id_vars, var_name='gatewayFid', value_name='Share')
        split_df['Share'] = pd.to_numeric(split_df['Share'].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip(), errors='coerce').fillna(0)
        split_df = split_df[split_df['Share'] > 0].copy()
        split_df['Rule_Source'] = 'Specific'

        force_actuals_rpgts = self.config.get('filters', {}).get('force_actuals_for_rpgts', [])
        if force_actuals_rpgts:
            mask_override = split_df['RPGT'].astype(str).str.lower().isin([str(r).strip().lower() for r in force_actuals_rpgts])
            if mask_override.any():
                split_df = split_df[~mask_override].copy()

        return split_df

    def _expand_categorical_rules(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Explicitly expands high-level categorical rules (STICKY/paymentMethodProvider/Country = Both/All) into literal rows."""
        if split_df.empty:
            return split_df

        sticky_col = 'STICKY' if 'STICKY' in split_df.columns else ('sticky' if 'sticky' in split_df.columns else None)
        if sticky_col:
            mask_both = split_df[sticky_col].astype(str).str.lower() == 'both'
            if mask_both.any():
                r_sticky, r_non = split_df[mask_both].copy(), split_df[mask_both].copy()
                r_sticky[sticky_col], r_non[sticky_col] = 'sticky', 'non_sticky'
                split_df = pd.concat([split_df[~mask_both], r_sticky, r_non], ignore_index=True)

        if 'paymentMethodProvider' in split_df.columns:
            mask_all_pm = split_df['paymentMethodProvider'].astype(str).str.lower() == 'all'
            if mask_all_pm.any():
                r_non_gp, r_google, r_apple = split_df[mask_all_pm].copy(), split_df[mask_all_pm].copy(), split_df[mask_all_pm].copy()
                r_non_gp['paymentMethodProvider'], r_google['paymentMethodProvider'], r_apple['paymentMethodProvider'] = 'non_gp_ap', 'googlepay', 'applepay'
                split_df = pd.concat([split_df[~mask_all_pm], r_non_gp, r_google, r_apple], ignore_index=True)

        country_col = 'Country' if 'Country' in split_df.columns else 'country'
        if country_col not in split_df.columns:
            split_df[country_col] = 'all'
        split_df[country_col] = split_df[country_col].astype(str).replace({'0': 'all', '0.0': 'all', 'nan': 'all', '': 'all'})
        mask_all_country = split_df[country_col].str.lower() == 'all'
        if mask_all_country.any():
            r_usa, r_non_usa = split_df[mask_all_country].copy(), split_df[mask_all_country].copy()
            r_usa[country_col], r_non_usa[country_col] = 'usa', 'non-usa'
            split_df = pd.concat([split_df[~mask_all_country], r_usa, r_non_usa], ignore_index=True)

        split_df[country_col] = split_df[country_col].astype(str).str.lower().str.strip()
        return split_df

    def _expand_dynamic_bins(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Explodes 'All'/'Other' BIN rules into distinct rows using historical actuals."""
        if split_df.empty or 'BIN' not in split_df.columns:
            return split_df

        hist_cols = ['Company', 'rpgt', 'Currency', 'BIN', 'paymentMethodProvider', 'Country']
        unique_hist = self.attempts_df[[c for c in hist_cols if c in self.attempts_df.columns]].drop_duplicates().rename(columns={'Company': 'Brand', 'rpgt': 'RPGT'})

        mask_dynamic = split_df['BIN'].astype(str).str.lower().isin(['other', 'all'])
        rules_dynamic = split_df[mask_dynamic].copy()
        if not rules_dynamic.empty:
            rules_dynamic = rules_dynamic.drop(columns=['BIN'])
            join_cols = [c for c in ['Brand', 'RPGT', 'Currency', 'paymentMethodProvider', 'Country'] if c in rules_dynamic.columns and c in unique_hist.columns]
            for c in join_cols:
                rules_dynamic[c], unique_hist[c] = rules_dynamic[c].astype('category'), unique_hist[c].astype('category')
            df_expanded = pd.merge(rules_dynamic, unique_hist, on=join_cols, how='inner')
            df_expanded['Rule_Source'] = 'Expanded'
            return pd.concat([split_df[~mask_dynamic], df_expanded], ignore_index=True)
        return split_df

    def _apply_chronological_deduplication(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """THE SCALPEL: filters overlapping Excel rules, preserving GO LIVE timelines so future rules replace older ones."""
        if split_df.empty:
            return split_df

        dedup_cols = ['Brand', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country']
        if 'STICKY' in split_df.columns:
            dedup_cols.append('STICKY')

        # CELL-LEVEL CATCH-ALL: the 'Other'/'All' fallback is a safety net for profiles that have NO
        # specific rule — it must NOT be injected into a cell that is already explicitly routed.
        # Drop every Expanded (catch-all) row whose full-grain cell already has >=1 Specific row, so
        # the catch-all only fires where the profile is genuinely undefined. (Mirrors the same fix in
        # vamp_pipeline/data_extractor.py — see that build marker.)
        if 'Rule_Source' in split_df.columns and (split_df['Rule_Source'] == 'Expanded').any():
            _spec_cells = split_df.loc[split_df['Rule_Source'] == 'Specific', dedup_cols].drop_duplicates()
            if not _spec_cells.empty:
                _spec_cells = _spec_cells.assign(_has_specific=1)
                split_df = split_df.merge(_spec_cells, on=dedup_cols, how='left')
                _drop_expanded = (split_df['Rule_Source'] == 'Expanded') & (split_df['_has_specific'] == 1)
                _n_dropped = int(_drop_expanded.sum())
                split_df = split_df.loc[~_drop_expanded].drop(columns=['_has_specific']).reset_index(drop=True)
                logger.info("cell-level catch-all: dropped %d Expanded catch-all row(s) in cells that "
                            "already carry a Specific rule.", _n_dropped)

        if not split_df.empty:
            if 'GO LIVE' not in split_df.columns:
                split_df['GO LIVE'] = pd.to_datetime('2020-01-01')
            else:
                split_df['GO LIVE'] = pd.to_datetime(split_df['GO LIVE'], errors='coerce', dayfirst=True).fillna(pd.Timestamp('2020-01-01'))

            sheet_rules = split_df[split_df['Rule_Source'].isin(['Specific', 'Expanded'])]
            if not sheet_rules.empty:
                min_dates = sheet_rules.groupby(dedup_cols, observed=True)['GO LIVE'].min().reset_index().rename(columns={'GO LIVE': 'Min_Sheet_Date'})
                merged_check = pd.merge(split_df, min_dates, on=dedup_cols, how='left')
            else:
                merged_check = split_df.copy()
                merged_check['Min_Sheet_Date'] = pd.NaT

            mask_keep = (merged_check['Rule_Source'].isin(['Specific', 'Expanded']) | ((merged_check['Rule_Source'] == 'Actuals') & (merged_check['Min_Sheet_Date'].isna() | (merged_check['Min_Sheet_Date'] > merged_check['GO LIVE']))))
            split_df = merged_check[mask_keep].copy().drop(columns=['Min_Sheet_Date'], errors='ignore')
            split_df = split_df.drop_duplicates(subset=[c for c in dedup_cols + ['gatewayFid', 'GO LIVE'] if c in split_df.columns])

        cat_cols = ['Brand', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'gatewayFid', 'Rule_Source'] + (['STICKY'] if 'STICKY' in split_df.columns else [])
        for c in cat_cols:
            if c in split_df.columns:
                split_df[c] = split_df[c].astype('category')
        return split_df

    def _generate_sql_actuals_rules(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts pure MTD SQL MASTERCARD transaction data into a normalized 'Share' routing
        matrix (Rule_Source='Actuals'), inflating shares for mid-month deaths (actuals_end_day)
        and scrubbing test gateways. Mirrors the notebook's generate_sql_actuals_rules.
        """
        actuals = mapping_df.copy()
        actuals.columns = actuals.columns.astype(str).str.strip().str.lower()
        rename_map = {
            'risk_defined_subscription_product_type': 'RPGT', 'rpgt': 'RPGT', 'company': 'Brand',
            'brand': 'Brand', 'currency': 'Currency', 'bin': 'BIN', 'gateway_fid': 'Gateway',
            'gatewayfid': 'Gateway', 'paymentmethodprovider': 'paymentMethodProvider',
            'country': 'Country', 'renewal_number': 'STICKY', 'sticky': 'STICKY'
        }
        actuals = actuals.rename(columns=rename_map)

        sql_profile_keys = [c for c in ['Brand', 'RPGT', 'Currency', 'BIN', 'paymentMethodProvider', 'Country', 'STICKY'] if c in actuals.columns]
        for c in sql_profile_keys + ['Gateway']:
            if c in actuals.columns:
                actuals[c] = actuals[c].astype(str).str.lower().str.strip()
                if c == 'BIN':
                    actuals[c] = actuals[c].str.replace(r'\.0$', '', regex=True)

        actuals['Profile_Total'] = actuals.groupby(sql_profile_keys, observed=True)['trx_count'].transform('sum')
        actuals['Share'] = np.where(actuals['Profile_Total'] > 0, actuals['trx_count'] / actuals['Profile_Total'], 0.0)

        invalid_gws = ['', 'nan', 'none', 'null', 'unmapped']
        actuals = actuals[(actuals['Share'] > 0) & (~actuals['Gateway'].isin(invalid_gws))].copy()
        actuals['Profile_Total'] = actuals.groupby(sql_profile_keys, observed=True)['Share'].transform('sum')
        actuals['Share'] = np.where(actuals['Profile_Total'] > 0, actuals['Share'] / actuals['Profile_Total'], 0.0)
        actuals = actuals[actuals['Share'] > 0].copy()

        actuals['Rule_Source'], actuals['GO LIVE'], actuals['Check'] = 'Actuals', pd.to_datetime('2020-01-01'), True

        # Inflate historical shares for mid-month deaths (effective_date before actuals_end_day)
        overrides_dict = self.config.get('gateway_volume_overrides', {}) if self.config['run_settings'].get('use_live_actuals', True) else {}
        m0_dt = pd.to_datetime(self.m0_start_date)
        if overrides_dict:
            for fid, cfg in overrides_dict.items():
                if isinstance(cfg, dict) and cfg.get('target', 0) == 0 and cfg.get('effective_date'):
                    eff_dt = pd.to_datetime(cfg.get('effective_date'))
                    if eff_dt.month == m0_dt.month and eff_dt.year == m0_dt.year and eff_dt.day < self.actuals_end_day:
                        mask_gw = actuals['Gateway'] == str(fid).strip().lower()
                        if mask_gw.any():
                            actuals.loc[mask_gw, 'Share'] *= (self.actuals_end_day / eff_dt.day)

        scrub_list = self.config.get('filters', {}).get('test_gateways_to_scrub', [])
        if scrub_list:
            actuals = actuals[~actuals['Gateway'].isin([str(g).strip().lower() for g in scrub_list])].copy()

        return actuals.drop(columns=['trx_count', 'Profile_Total', 'fcpNumber', 'attemptNumber'], errors='ignore')

    # =========================================================================
    # === 7. ORCHESTRATOR
    # =========================================================================

    def extract_all(self) -> None:
        """MANAGER: orchestrates the entire Mastercard data extraction, mapping, rules parsing, and matrix assembly."""
        logger.info("📥 Starting BigQuery Extraction Phase (Mastercard)...")

        self.fcast_data_df = self._fetch_bq_data(f'thermometer_data_{self.month_var}_mc_fcp_v3.parquet', 'mastercard_fcast_query.sql')
        self.gw_mapping_df = self._fetch_bq_data(f'gateway_mapping_data_{self.month_var}_mc_fcp_v3.parquet', 'mastercard_gatewayfid_trx_mapping.sql')

        lt_mapping = self._fetch_bq_data(f'lt_fc_mapping_data_{self.month_var}_mc_fcp_v3.parquet', 'mastercard_longterm_fcast.sql')
        self.lt_fcast_mapping_df = lt_mapping[lt_mapping['rpgt'] != 'Other'].copy()

        # Scrub test gateways from the historical baselines BEFORE deriving attempts
        scrub_list = self.config.get('filters', {}).get('test_gateways_to_scrub', [])
        if scrub_list:
            scrub_lower = [str(g).strip().lower() for g in scrub_list]
            self.lt_fcast_mapping_df = self.lt_fcast_mapping_df[~self.lt_fcast_mapping_df['gatewayFid'].astype(str).str.lower().str.strip().isin(scrub_lower)].copy()
            self.gw_mapping_df = self.gw_mapping_df[~self.gw_mapping_df['gatewayFid'].astype(str).str.lower().str.strip().isin(scrub_lower)].copy()

        logger.info("   > Loading FP&A Macro Forecast from BigQuery...")
        raw_macro_fcast = self._fetch_bq_data('macro_forecast.parquet', 'mastercard_macro_forecast.sql', apply_keys=False)
        self.mid_df = self._fetch_local_mid_list()

        self._fetch_mr_daily_weights()

        raw_attempts = self._derive_historical_attempts(self.gw_mapping_df)
        self.attempts_df = self._flex_attempts_day_count(raw_attempts)

        logger.info("📊 Fetching Expected Rules (Local Excel)...")
        if self.config['run_settings'].get('use_chunked_csv_files', False):
            raw_rules = self._fetch_chunked_rules()
        else:
            raw_rules = self._fetch_direct_rules()

        # === SECTION 5: apply overrides & MASTERCARD mapping % (must precede SQL actuals) ===
        logger.info("📐 Sizing the MASTERCARD long-term mapping (overrides & mapping %)...")
        sql_rename_map = {'risk_defined_subscription_product_type': 'rpgt', 'gateway_fid': 'gatewayFid', 'bin': 'BIN', 'company': 'Company', 'currency': 'Currency', 'country': 'Country'}
        self.lt_fcast_mapping_df = self.lt_fcast_mapping_df.rename(columns=sql_rename_map)
        self.lt_fcast_mapping_df = self._apply_forecast_volume_overrides(self.lt_fcast_mapping_df)
        self.lt_fcast_mapping_df = self._calculate_mastercard_mapping_pct(self.lt_fcast_mapping_df)

        os.makedirs(self.output_dir, exist_ok=True)
        mapping_export_cols = [c for c in self.profile_keys + ['gatewayFid', 'attemptNumber', 'trx_count', 'trx total', 'mapping_pct'] if c in self.lt_fcast_mapping_df.columns]
        mapping_path = os.path.join(self.output_dir, 'mapping_pct_export.csv')
        self.lt_fcast_mapping_df[mapping_export_cols].to_csv(mapping_path, index=False)
        logger.info(f"   > Exported '{mapping_path}' ({len(self.lt_fcast_mapping_df)} rows).")

        # === SECTION 6: split rules = GSheet rules + SQL actuals, expanded & deduplicated ===
        logger.info("🛠️ Processing Split Rules (GSheet + SQL Actuals merge & Deduplication)...")
        split_df = self._clean_google_sheets_rules(raw_rules)
        actuals_base = self._generate_sql_actuals_rules(self.lt_fcast_mapping_df)
        split_df = pd.concat([split_df, actuals_base], ignore_index=True)
        split_df = self._expand_categorical_rules(split_df)
        split_df = self._expand_dynamic_bins(split_df)
        self.split_df = self._apply_chronological_deduplication(split_df)

        # === Build the future forecast matrix, apply the Visa kill switch, then melt & target ===
        logger.info("📐 Building Final Long-Term Forecast Matrix...")
        wide_fcast, trx_cols = self._build_future_forecast_matrix(self.lt_fcast_mapping_df, raw_macro_fcast)
        wide_fcast = self._apply_visa_kill_switch(wide_fcast)  # 🟢 THE VISA KILL SWITCH (before melting)
        self.longterm_fcast_df = self._melt_and_apply_targets(wide_fcast, trx_cols)

        logger.info("✅ Mastercard Data Extraction & Matrix Assembly Complete.")
