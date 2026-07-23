"""
Run .sql files and cache the results.

In production each file in sql/ is executed against BigQuery and cached to
parquet, so re-running the forecast reuses the cache instead of re-querying.
When BigQuery (or credentials) isn't available - e.g. local dev - callers can
pass a fallback CSV so the app still runs on sample data.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os

logger = logging.getLogger(__name__)

# Bumped when this file's API changes; if a traceback shows an unexpected keyword,
# check that __build__ matches — a mismatch means stale __pycache__.
__build__ = "2026-07-04-templated-params"


def list_sql_files(sql_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(sql_dir, "*.sql")))


def cache_path_for(sql_path: str, cache_dir: str,
                   params: dict | None = None) -> str:
    """Cache filename includes a short hash of params so different runs cache
    separately (e.g. different date ranges or companies)."""
    stem = os.path.splitext(os.path.basename(sql_path))[0]
    if params:
        key = "|".join(f"{k}={params[k]}" for k in sorted(params))
        digest = hashlib.md5(key.encode()).hexdigest()[:10]
        stem = f"{stem}_{digest}"
    return os.path.join(cache_dir, f"{stem}.parquet")


def _substitute_params(sql: str, params: dict | None) -> str:
    if not params:
        return sql
    for k, v in params.items():
        sql = sql.replace("{" + k + "}", str(v))
    return sql


def run_sql_file(sql_path: str, cache_dir: str, use_cache: bool = True,
                 fallback_csv: str | None = None,
                 project: str | None = None,
                 params: dict | None = None) -> tuple[str, str]:
    """
    Return (data_path, source) where source is one of:
      'cache'     - reused a previously cached parquet
      'bigquery'  - freshly queried and cached
      'fallback'  - BigQuery unavailable, used the fallback CSV

    `params` are substituted into the SQL (e.g. {"START_DATE": "2026-01-01"}
    replaces {START_DATE}), and change the cache filename so different runs
    don't collide. Raises if BigQuery fails and no fallback is provided.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = cache_path_for(sql_path, cache_dir, params)
    name = os.path.basename(sql_path)
    if use_cache and os.path.exists(cache_file):
        logger.info(f"   > SQL {name}: loading cached result from {cache_file}")
        return cache_file, "cache"

    logger.info(f"   > SQL {name}: running on BigQuery (project={project})"
                f"{'; params=' + str(params) if params else ''}...")
    try:
        from google.cloud import bigquery  # type: ignore

        with open(sql_path) as f:
            sql = _substitute_params(f.read(), params)
        client = bigquery.Client(project=project) if project else bigquery.Client()
        df = client.query(sql).to_dataframe()
        logger.info(f"   > SQL {name}: returned {len(df):,} rows")
        try:
            df.to_parquet(cache_file)
            logger.info(f"   > SQL {name}: cached to {cache_file}")
            return cache_file, "bigquery"
        except Exception:  # parquet engine missing -> csv cache
            csv_file = cache_file.replace(".parquet", ".csv")
            df.to_csv(csv_file, index=False)
            return csv_file, "bigquery"
    except Exception as exc:  # noqa: BLE001
        logger.error(f"   > SQL {name}: FAILED — {type(exc).__name__}: {exc}")
        if fallback_csv and os.path.exists(fallback_csv):
            logger.warning(f"   > SQL {name}: falling back to {fallback_csv}")
            return fallback_csv, f"fallback: {type(exc).__name__}: {exc}"[:300]
        raise
