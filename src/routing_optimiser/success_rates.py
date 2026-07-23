"""
Estimate the expected success (authorisation) rate for every
RPGT x Currency x Bank x Gateway cell from the attempts/success data.

Many cells have tiny sample sizes, so a raw success/attempts ratio is noisy
(one gateway looks "100%" off two transactions). We shrink each cell's rate
towards a sensible prior (the pooled rate for its RPGT x Currency) using
empirical-Bayes shrinkage, so small cells lean on the group average and only
break away when they have real evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SCENARIO_TO_RPGT, SUCCESS_DATA_COLUMNS as C


def load_success_data(source) -> pd.DataFrame:
    """Load the attempts/success data from a DataFrame, or a CSV/parquet path.

    Accepts either the new query shape (with an ``rpgt`` column already) or the
    older shape with ``transactionScenario`` — both are normalised to ``rpgt``.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif str(source).endswith(".parquet"):
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source)

    # 1. Normalise to a single 'rpgt' column safely
    if "transactionScenario" in df.columns and "rpgt" not in df.columns:
        df = df.rename(columns={"transactionScenario": "rpgt"})
        
    if "rpgt" not in df.columns:
        raise KeyError(
            "Attempts/success data has neither 'rpgt' nor 'transactionScenario' "
            f"column. Got: {sorted(df.columns.tolist())[:20]}"
        )

    # Standardize RPGT strings using the schema mapping
    df["rpgt"] = df["rpgt"].map(SCENARIO_TO_RPGT).fillna(df["rpgt"])

    # 2. Safely extract values from the schema mapping using .get() to avoid KeyErrors
    rename_map = {
        C.get("currency", "currency"): "currency",
        C.get("bank_name", "bankName"): "bank",
        C.get("processor", "processor"): "processor",
        C.get("gateway_fid", "gatewayFid"): "gateway",
        C.get("initial_attempt", "initialattempt"): "attempts",
        C.get("success", "initialSuccess"): "success",
        C.get("amount", "amount"): "amount",
    } if hasattr(C, "get") else {}

    # Apply only the renames where the source column actually exists
    valid_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_renames)

    # 3. Explicit fallback for the new SQL logic (in case the schema dict was incomplete)
    if "bank" not in df.columns and "bankName" in df.columns:
        df = df.rename(columns={"bankName": "bank"})
    if "gateway" not in df.columns and "gatewayFid" in df.columns:
        df = df.rename(columns={"gatewayFid": "gateway"})
        
    if "success" not in df.columns and "initialSuccess" in df.columns:
        df["success"] = df["initialSuccess"]
    if "attempts" not in df.columns and "initialattempt" in df.columns:
        df["attempts"] = df["initialattempt"]

    # Canonicalise gateway names so any deprecated '-x' MID collapses onto its
    # non-'-x' sibling. Matches the same rule used for the pipeline forecast,
    # so the two datasets join cleanly per (rpgt, currency, bank, gateway).
    if "gateway" in df.columns:
        from .forecast_pipeline import _canonical_gateway
        df["gateway"] = df["gateway"].map(_canonical_gateway)

    return df


def _apply_time_decay(df: pd.DataFrame, half_life_days: float | None,
                      date_col: str = "date") -> pd.DataFrame:
    """
    Apply an exponential half-life weight to each row so recent attempts count
    more than old ones. Weight = 0.5 ** (age_days / half_life_days).
    `attempts` and `success` are scaled by that weight; downstream aggregations
    then act on decayed counts.
    """
    if half_life_days is None or half_life_days <= 0 or date_col not in df.columns:
        return df
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    ref = d[date_col].max()
    age_days = (ref - d[date_col]).dt.total_seconds() / 86400.0
    w = 0.5 ** (age_days.clip(lower=0) / float(half_life_days))
    w = w.fillna(1.0)
    d["attempts"] = d["attempts"].astype(float) * w
    d["success"] = d["success"].astype(float) * w
    return d


def _empirical_bayes_kappa(grp: pd.DataFrame, scope: list[str],
                           fallback: float, kmax: float = 5_000.0) -> pd.DataFrame:
    """Method-of-moments Beta-Binomial concentration (kappa) per prior_scope group.

    Model each group's gateway success rates as draws from Beta(mean=mu, conc=kappa),
    so var_beta = mu(1-mu)/(kappa+1). Estimate the TRUE between-gateway variance as
    (observed weighted variance of rates) - (mean binomial sampling variance), then
    kappa = mu(1-mu)/true_var - 1. Tight spread -> large kappa (trust the pool);
    wide spread -> small kappa (trust each gateway). Needs >=2 gateways; else uses
    `fallback`. Returns a frame of scope keys + 'kappa'.
    """
    rows = []
    for key, g in grp.groupby(scope):
        key = key if isinstance(key, tuple) else (key,)
        n = g["attempts"].to_numpy(float)
        x = g["success"].to_numpy(float)
        m = n > 0
        n, x = n[m], x[m]
        if len(n) < 2 or n.sum() <= 0:
            rows.append((*key, float(fallback)))
            continue
        p = x / n
        mu = x.sum() / n.sum()
        if mu <= 0 or mu >= 1:
            # A group where every gateway succeeded (or all failed) — usually a thin-
            # sample artefact, not truth. Use the modest `fallback` kappa (NOT kmax), so
            # per-gateway evidence still shows through rather than being erased to 0/1. (F1)
            rows.append((*key, float(fallback)))
            continue
        obs_var = float((n * (p - mu) ** 2).sum() / n.sum())      # attempt-weighted
        samp_var = mu * (1.0 - mu) * len(n) / n.sum()             # mean binomial noise
        true_var = obs_var - samp_var
        if true_var <= 1e-9:
            # Gateways look alike: shrink hard, but only up to the (now sane) kmax cap so
            # some between-gateway signal survives instead of collapsing to the pool. (F1/F2)
            rows.append((*key, kmax))
            continue
        kap = mu * (1.0 - mu) / true_var - 1.0
        rows.append((*key, float(min(max(kap, 1.0), kmax))))
    return pd.DataFrame(rows, columns=scope + ["kappa"])


def gateway_success_rates(
    df: pd.DataFrame,
    gateway_col: str = "gateway",
    shrink_strength: float = 12.0,
    time_decay_half_life_days: float | None = None,
    prior_scope: tuple[str, ...] = ("rpgt", "currency"),
    empirical_bayes: bool = False,
) -> pd.DataFrame:
    """
    Returns one row per (rpgt, currency, bank, gateway) with:
      attempts, success, raw_rate, prior_rate, kappa, success_rate (shrunk).

    `shrink_strength` (kappa) is the number of "prior transactions" mixed in when
    `empirical_bayes` is False. When `empirical_bayes` is True, kappa is estimated
    per `prior_scope` group from the spread of gateway rates (method of moments);
    `shrink_strength` is then only the fallback for groups with too few gateways.
    `time_decay_half_life_days`, if set, exponentially down-weights older attempts.
    `prior_scope` sets the grouping the shrinkage prior is pooled over.
    """
    df = _apply_time_decay(df, time_decay_half_life_days)
    grp = (
        df.groupby(["rpgt", "currency", "bank", gateway_col], as_index=False)
        .agg(attempts=("attempts", "sum"), success=("success", "sum"))
    )
    grp = grp.rename(columns={gateway_col: "gateway"})
    grp["raw_rate"] = np.where(grp["attempts"] > 0,
                               grp["success"] / grp["attempts"], np.nan)

    scope = list(prior_scope)
    prior = (
        df.groupby(scope, as_index=False)
        .agg(p_success=("success", "sum"), p_attempts=("attempts", "sum"))
    )
    prior["prior_rate"] = np.where(prior["p_attempts"] > 0,
                                   prior["p_success"] / prior["p_attempts"],
                                   np.nan)
    global_rate = df["success"].sum() / max(df["attempts"].sum(), 1)
    prior["prior_rate"] = prior["prior_rate"].fillna(global_rate)

    out = grp.merge(prior[scope + ["prior_rate"]], on=scope, how="left")
    out["prior_rate"] = out["prior_rate"].fillna(global_rate)

    if empirical_bayes:
        kap_df = _empirical_bayes_kappa(grp, scope, fallback=float(shrink_strength))
        out = out.merge(kap_df, on=scope, how="left")
        out["kappa"] = out["kappa"].fillna(float(shrink_strength))
    else:
        out["kappa"] = float(shrink_strength)

    out["success_rate"] = (out["success"] + out["kappa"] * out["prior_rate"]) / (out["attempts"] + out["kappa"])
    return out


def risk_rates_from_forecast(forecast: pd.DataFrame | None,
                             gateways: list[str],
                             default: float = 0.006,
                             shrink: float = 500.0) -> dict[str, float]:
    """
    Expected chargeback/VAMP rate per gateway. In production this comes from
    the "post" numbers of your VAMP pipeline (VAMPs / sales per gateway).

    The raw ratio is SHRUNK toward the pooled VAMP rate with `shrink` pseudo-sales
    (Empirical-Bayes style): rate = (vamps + shrink·pooled) / (sales + shrink). A
    thin gateway (e.g. 1 VAMP on 3 sales) is pulled to the pool instead of reporting
    a wild 33%; a high-volume gateway is essentially unchanged. This stops noisy
    thin-gateway risk from dominating cap enforcement and portfolio CVaR. (F3)
    If no forecast is supplied we fall back to a flat default so the app runs.
    """
    if forecast is None or "gateway" not in forecast.columns:
        return {g: default for g in gateways}
    if {"vamps", "sales"}.issubset(forecast.columns):
        agg = forecast.groupby("gateway").agg(_v=("vamps", "sum"), _s=("sales", "sum"))
        tot_v = float(agg["_v"].sum()); tot_s = float(agg["_s"].sum())
        pooled = (tot_v / tot_s) if tot_s > 0 else default
        r = ((agg["_v"] + shrink * pooled) / (agg["_s"] + shrink)).to_dict()
        return {g: float(r.get(g, pooled)) for g in gateways}
    return {g: default for g in gateways}