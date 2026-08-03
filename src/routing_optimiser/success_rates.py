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

import warnings

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
        # numerator must be the INITIAL-stage success so it matches the
        # 'initialattempt' denominator (same funnel stage); using the final
        # 'success' column here mixed funnel stages and inflated the ratio.
        C.get("initial_success", "initialSuccess"): "success",
        C.get("amount", "amount"): "amount",
    } if hasattr(C, "get") else {}

    # Apply only the renames where the source column actually exists
    valid_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=valid_renames)

    # (The previous "explicit fallback" block that re-renamed bankName/gatewayFid
    # and copied initialSuccess/initialattempt was removed: rename_map above already
    # covers those exact source columns, so the block was redundant and — since it
    # copied the INITIAL success — contradicted the old final-'success' mapping.)

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
    # Rows whose date failed to parse (NaT -> NaN age) get ZERO weight, not full
    # most-recent weight: a bad/unparseable date must not silently count as fresh.
    w = w.fillna(0.0)
    d["attempts"] = d["attempts"].astype(float) * w
    d["success"] = d["success"].astype(float) * w
    return d


def _empirical_bayes_kappa(grp: pd.DataFrame, scope: list[str],
                           fallback: float, kmax: float = 5_000.0) -> pd.DataFrame:
    """Method-of-moments Beta-Binomial concentration (kappa) per prior_scope group.

    ANALOGY: kappa asks "do the gateways in this group behave alike?" If their success rates
    cluster tightly (small true spread), trust the pooled average heavily → big kappa (shrink
    hard). If they're all over the place, trust each gateway's own data → small kappa (shrink
    little). It's LEARNED from the data rather than being a fixed dial.

    Model each group's gateway success rates as draws from Beta(mean=mu, conc=kappa),
    so var_beta = mu(1-mu)/(kappa+1). Estimate the TRUE between-gateway variance as
    (observed weighted variance of rates) - (mean binomial sampling variance), then
    kappa = mu(1-mu)/true_var - 1. Tight spread -> large kappa (trust the pool);
    wide spread -> small kappa (trust each gateway). Needs >=2 gateways; else uses
    `fallback`. Returns a frame of scope keys + 'kappa'.
    """
    rows = []
    # Gateways must be compared only against OTHER GATEWAYS IN THE SAME BANK,
    # otherwise cross-bank differences leak into the "between-gateway" spread and
    # bias kappa. So `bank` is always part of the per-cell grouping (added here if
    # it isn't already in `scope`); we measure the spread WITHIN each
    # (scope, bank) cell, then attempt-weight-pool those cells up to `scope`.
    cell_extra = [] if "bank" in scope else ["bank"]
    for key, g in grp.groupby(scope):
        key = key if isinstance(key, tuple) else (key,)
        n_all = g["attempts"].to_numpy(float)
        x_all = g["success"].to_numpy(float)
        m_all = n_all > 0
        if not m_all.any() or n_all[m_all].sum() <= 0:
            rows.append((*key, float(fallback)))
            continue
        mu = x_all[m_all].sum() / n_all[m_all].sum()               # pooled scope mean
        # Accumulate the attempt-weighted within-bank observed & sampling variances.
        num_obs = num_samp = wsum = 0.0
        cells = [g] if not cell_extra else [gb for _, gb in g.groupby(cell_extra)]
        for gb in cells:
            n = gb["attempts"].to_numpy(float)
            x = gb["success"].to_numpy(float)
            m = n > 0
            n, x = n[m], x[m]
            if len(n) < 2 or n.sum() <= 0:
                # single-gateway bank cell carries no between-gateway signal — skip it
                continue
            p = x / n
            mu_b = x.sum() / n.sum()                               # this bank's own mean
            obs_var_b = float((n * (p - mu_b) ** 2).sum() / n.sum())   # attempt-weighted
            samp_var_b = mu_b * (1.0 - mu_b) * len(n) / n.sum()        # mean binomial noise
            wt = float(n.sum())
            num_obs += wt * obs_var_b
            num_samp += wt * samp_var_b
            wsum += wt
        if wsum <= 0 or mu <= 0 or mu >= 1:
            # No bank had >=2 gateways to compare (or every gateway succeeded/failed) —
            # usually a thin-sample artefact, not truth. Use the modest `fallback` kappa
            # (NOT kmax), so per-gateway evidence still shows through. (F1)
            rows.append((*key, float(fallback)))
            continue
        obs_var = num_obs / wsum
        samp_var = num_samp / wsum
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
    # Defensive: a raw_rate above 1.0 means success exceeded attempts, which can
    # only happen from a funnel column-mapping mismatch (e.g. a final-stage
    # 'success' numerator paired with the 'initialattempt' denominator). Warn so
    # the mapping gets fixed, then clamp so the shrinkage maths can't be poisoned.
    if bool((grp["raw_rate"] > 1.0).any()):
        warnings.warn(
            "raw_rate > 1.0 detected (success exceeds attempts): likely a funnel "
            "column-mapping mismatch, e.g. the final 'success' column mapped to the "
            "numerator while 'initialattempt' is the denominator. Clamping to 1.0.",
            stacklevel=2,
        )
    grp["raw_rate"] = grp["raw_rate"].clip(upper=1.0)

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

    # Empirical-Bayes shrinkage — ANALOGY: grading a gateway on limited evidence. With only a
    # few attempts you don't fully trust its raw rate, so you blend it toward the pooled prior;
    # `kappa` behaves like "pseudo-attempts" of that prior. The more real attempts a gateway has,
    # the less the prior matters and the closer the result sits to its own observed rate.
    out["success_rate"] = (out["success"] + out["kappa"] * out["prior_rate"]) / (out["attempts"] + out["kappa"])
    return out


def detect_blocked_gateways(adf, min_consecutive: float, date_col: str = "date"):
    """Flag (bank, gateway) pairs the acquiring bank appears to have BLOCKED us on.

    ANALOGY: like spotting a vendor whose card terminal has declined EVERY transaction for days
    straight — most likely the bank cut them off, so we stop throwing traffic at a dead route
    (the caller caps that gateway to the exploration floor) instead of bleeding conversions.

    Looks at the MOST-RECENT consecutive run of daily attempts that ALL failed (a day counts
    as failed only if it had zero successes); a day with any success breaks the run. If the
    attempts in that leading all-failed run reach `min_consecutive`, the pair is flagged
    `blocked` — the caller then caps that gateway's share (for that bank) to the exploration
    floor. Vectorised (one groupby-cummax, no per-pair Python loop).

    Returns a DataFrame: bank, gateway, consec_failed (attempts in the leading failed run),
    last_success_date, blocked (bool), sorted by consec_failed descending. Empty if the inputs
    lack the needed columns or `min_consecutive` <= 0.
    """
    cols = ["bank", "gateway", "consec_failed", "last_success_date", "blocked"]
    need = {"bank", "gateway", "attempts", "success"}
    if (adf is None or not need.issubset(getattr(adf, "columns", [])) or date_col not in
            getattr(adf, "columns", []) or float(min_consecutive) <= 0):
        return pd.DataFrame(columns=cols)
    d = adf[["bank", "gateway", "attempts", "success", date_col]].copy()
    d["_day"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d = d.dropna(subset=["_day"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["_bank"] = d["bank"].astype(str)
    d["_gw"] = d["gateway"].astype(str)
    d["att"] = pd.to_numeric(d["attempts"], errors="coerce").fillna(0.0)
    d["suc"] = pd.to_numeric(d["success"], errors="coerce").fillna(0.0)
    g = (d.groupby(["_bank", "_gw", "_day"], as_index=False)
         .agg(att=("att", "sum"), suc=("suc", "sum")))
    # Sort each pair's days MOST-RECENT first; the leading all-failed run is where the running
    # max of successes (from the top) is still 0. Sum its attempts.
    g = g.sort_values(["_bank", "_gw", "_day"], ascending=[True, True, False])
    g["_cmax"] = g.groupby(["_bank", "_gw"])["suc"].cummax()
    _run = (g[g["_cmax"] <= 0].groupby(["_bank", "_gw"], as_index=False)["att"].sum()
            .rename(columns={"att": "consec_failed"}))
    out = g[["_bank", "_gw"]].drop_duplicates().merge(_run, on=["_bank", "_gw"], how="left")
    out["consec_failed"] = out["consec_failed"].fillna(0.0)
    _succ = (g[g["suc"] > 0].groupby(["_bank", "_gw"], as_index=False)["_day"].max()
             .rename(columns={"_day": "last_success_date"}))
    out = out.merge(_succ, on=["_bank", "_gw"], how="left")
    out["blocked"] = out["consec_failed"] >= float(min_consecutive)
    out = out.rename(columns={"_bank": "bank", "_gw": "gateway"})
    return out[cols].sort_values("consec_failed", ascending=False).reset_index(drop=True)


def rpgt_gateway_sensitivity(sr_df, avg_ticket: float = 1.0, min_attempts: float = 1.0):
    """How sensitive is each RPGT to WHERE its traffic is routed.

    Consumes the output of ``gateway_success_rates`` (one row per rpgt×currency×bank×gateway
    with the EMPIRICAL-BAYES-shrunk ``success_rate`` — thin gateways already pulled toward
    the pooled rate, so they can't masquerade as best/worst). For each (rpgt, currency, bank)
    cell we take the best−worst gap of the shrunk gateway rates: that is the success-rate swing
    reachable by rerouting within that cell. Volume-weighting those gaps up to the RPGT gives
    its sensitivity (percentage points), and Σ(gap × volume × avg_ticket) is the revenue at
    stake between best- and worst-case routing.

    Returns one row per rpgt: ``volume`` (30-day attempts), ``sensitivity_pp``,
    ``dollars_at_stake``, ``cells`` (number of routable cells), sorted by dollars.
    """
    need = {"rpgt", "currency", "bank", "gateway", "attempts", "success_rate"}
    d = sr_df.copy()
    missing = need - set(d.columns)
    if missing:
        raise KeyError(f"rpgt_gateway_sensitivity needs columns {sorted(missing)} in sr_df")
    d["attempts"] = pd.to_numeric(d["attempts"], errors="coerce")
    d["success_rate"] = pd.to_numeric(d["success_rate"], errors="coerce")
    d = d[(d["attempts"] >= float(min_attempts)) & d["success_rate"].notna()]
    cols = ["rpgt", "volume", "sensitivity_pp", "dollars_at_stake", "cells"]
    if d.empty:
        return pd.DataFrame(columns=cols)
    cell = d.groupby(["rpgt", "currency", "bank"], as_index=False).agg(
        vol=("attempts", "sum"), rmax=("success_rate", "max"),
        rmin=("success_rate", "min"), ngw=("gateway", "nunique"))
    # Only cells with ≥2 eligible gateways are reroutable; single-gateway cells
    # contribute NOTHING — not to the gap, and not to the volume/cells denominators
    # either — so they can't dilute sensitivity_pp or dollars_at_stake.
    routable = cell["ngw"] >= 2
    cell["gap"] = np.where(routable, (cell["rmax"] - cell["rmin"]).clip(lower=0.0), 0.0)
    cell["rvol"] = np.where(routable, cell["vol"], 0.0)      # routable volume only
    cell["gapvol"] = cell["gap"] * cell["rvol"]
    cell["rcell"] = routable.astype(int)                    # routable-cell counter
    rp = cell.groupby("rpgt", as_index=False).agg(
        volume=("rvol", "sum"), gapvol=("gapvol", "sum"), cells=("rcell", "sum"))
    rp["sensitivity_pp"] = np.where(rp["volume"] > 0, rp["gapvol"] / rp["volume"] * 100.0, 0.0)
    rp["dollars_at_stake"] = rp["gapvol"] * float(avg_ticket)
    return (rp[cols].sort_values("dollars_at_stake", ascending=False)
            .reset_index(drop=True))


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