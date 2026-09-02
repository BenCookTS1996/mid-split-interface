"""
Load the inputs the optimiser needs and turn them into CellProblems.

Two inputs:
  1. The "pre" forecast (baseline volumes + current split) from the VAMP
     pipeline. We accept a tidy CSV/parquet with columns:
        rpgt, currency, bin, gateway, volume, baseline_share [, risk_rate]
     If you don't have that shape yet, `synthesise_forecast_from_success`
     builds a stand-in from the attempts data so the app runs end to end.
  2. The success/attempts data (for success rates).

The output is a list of CellProblem objects, one per RPGT x Currency x Bank.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from routing_optimiser.engines import ProfileProblem
from routing_optimiser.s1_extract.success_rates import gateway_success_rates, load_success_data


# [FN-048]
def synthesise_forecast_from_success(success_df: pd.DataFrame,
                                     default_risk: float = 0.006) -> pd.DataFrame:
    """Build a plausible baseline forecast from the attempts data.

    Used when a real VAMP 'pre' export isn't wired in yet. Volume = observed
    attempts; baseline_share = observed share of that gateway within the profile.
    """
    g = (success_df.groupby(["rpgt", "currency", "bin", "gateway"], as_index=False)
         .agg(volume=("attempts", "sum")))
    tot = g.groupby(["rpgt", "currency", "bin"])["volume"].transform("sum")
    g["baseline_share"] = np.where(tot > 0, g["volume"] / tot, 0.0)
    # crude per-gateway risk: higher-volume processors slightly riskier, just
    # so the sample has variation. Replace with real VAMP 'post' numbers.
    rng = np.random.default_rng(7)
    per_gw = {gw: float(np.clip(default_risk + rng.normal(0, 0.002), 0.001, 0.02))
              for gw in g["gateway"].unique()}
    g["risk_rate"] = g["gateway"].map(per_gw)
    return g


# [FN-049]
def load_forecast(path: str | None, success_df: pd.DataFrame) -> pd.DataFrame:
    """Load the baseline 'pre' forecast — from a file, a pipeline output directory, or (when
    no path is given) a stand-in synthesised from the attempts data so the app still runs.
    Also normalises the pipeline's effective-rate export into the optimiser's forecast contract
    when that's the shape it's handed."""
    if path is None:
        return synthesise_forecast_from_success(success_df)
    if os.path.isdir(path):  # a pipeline output directory
        from routing_optimiser.s2_forecast.vamp_forecast_pipeline import load_pre_forecast
        return load_pre_forecast(path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # If this is the VAMP pipeline's effective_rate_impact.csv, normalise its
    # baseline ('Sim_*') columns into the optimiser's forecast contract.
    from routing_optimiser.s2_forecast.vamp_forecast_pipeline import (looks_like_effective_rate,
                                    normalise_pre_from_effective_rate)
    if looks_like_effective_rate(df):
        return normalise_pre_from_effective_rate(df)
    return df


# [FN-050]
def build_profile_problems(
    forecast: pd.DataFrame,
    success_rates: pd.DataFrame,
    default_risk: float = 0.006,
) -> list[ProfileProblem]:
    """Join forecast volume + baseline split with success/risk rates per profile.

    ANALOGY: assembling each profile's "briefing pack". For every RPGT×Currency×Bank profile we pull
    the forecast's volume + current split together with each gateway's success rate, risk rate
    and evidence, and hand the engine one CellProblem it can solve. Gateways with no per-profile
    attempts fall back to the pooled prior (flagged so the UI can show which are educated guesses
    rather than measured rates).
    """
    # Normalise the join keys (strip + case-fold) on BOTH sides so a casing/whitespace
    # difference between the success data (bankName) and the forecast (pipeline) doesn't
    # silently miss and dump every gateway onto the pooled prior. (If the two sides key on
    # fundamentally different values — e.g. BIN vs bank-name — normalisation can't help, but
    # the fallback-rate warning below makes that visible instead of silent.)
    # [FN-051]
    def _nk(x):
        return str(x).strip().casefold()

    _srn = success_rates.copy()
    for _c in ("rpgt", "currency", "bin", "gateway"):
        if _c in _srn.columns:
            _srn[_c] = _srn[_c].map(_nk)
    sr = _srn.set_index(["rpgt", "currency", "bin", "gateway"])
    # Normalising the keys can collapse case/whitespace-variant rows onto the same key; drop
    # the resulting duplicate index entries (keep first) so `sr.loc[key]` returns exactly one
    # row (a Series) rather than a multi-row DataFrame — otherwise float(row[...]) raises.
    if sr.index.has_duplicates:
        sr = sr[~sr.index.duplicated(keep="first")]
    # Attempts-WEIGHTED pooled prior (an unweighted mean over profiles would let tiny, noisy
    # profiles count as much as huge ones).
    if len(sr) and {"success", "attempts"}.issubset(sr.columns) and float(sr["attempts"].sum()) > 0:
        _global_rate = float(sr["success"].sum() / sr["attempts"].sum())
    elif len(sr):
        _global_rate = float(sr["success_rate"].mean())
    else:
        _global_rate = 0.85
    _has_prior = "prior_rate" in sr.columns
    _has_kappa = "kappa" in sr.columns
    problems: list[ProfileProblem] = []
    _n_gw = _n_pool = 0

    for (rpgt, currency, bin_), profile in forecast.groupby(["rpgt", "currency", "bin"]):
        gateways = list(profile["gateway"])
        vol = float(profile["volume"].sum())
        base = profile["baseline_share"].to_numpy(float)
        base = base / base.sum() if base.sum() > 0 else np.full(len(gateways), 1 / len(gateways))

        succ, obs_s, obs_a, is_pool = [], [], [], []
        prior_r, kap = [], []
        for gw in gateways:
            key = (_nk(rpgt), _nk(currency), _nk(bin_), _nk(gw))
            _n_gw += 1
            if key in sr.index:
                row = sr.loc[key]
                succ.append(float(row["success_rate"]))
                obs_s.append(float(row["success"]))
                obs_a.append(float(row["attempts"]))
                prior_r.append(float(row["prior_rate"]) if _has_prior else float(row["success_rate"]))
                kap.append(float(row["kappa"]) if _has_kappa else 0.0)
                is_pool.append(False)
            else:
                # No per-profile attempts data for this gateway: fall back to the
                # pooled mean. Flag it so the UI can show which profiles are on
                # the pooled prior rather than real per-profile evidence.
                _n_pool += 1
                succ.append(_global_rate)
                obs_s.append(0.0)
                obs_a.append(0.0)
                prior_r.append(_global_rate)
                kap.append(0.0)
                is_pool.append(True)

        if "risk_rate" in profile.columns:
            risk = profile["risk_rate"].to_numpy(float)
        else:
            risk = np.full(len(gateways), default_risk)

        # Risk-rate sample size = the transaction/sales count the VAMP rate was
        # measured over. Prefer an explicit 'risk_n' column; else fall back to the
        # profile's routing volume (which, on the granular path, IS the Txn count).
        if "risk_n" in profile.columns:
            risk_n = pd.to_numeric(profile["risk_n"], errors="coerce").fillna(0.0).to_numpy(float)
        elif "volume" in profile.columns:
            risk_n = pd.to_numeric(profile["volume"], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            risk_n = None

        problem = ProfileProblem(
            rpgt=str(rpgt), currency=str(currency), bin=str(bin_),
            gateways=gateways,
            success_rates=np.array(succ, float),
            risk_rates=np.array(risk, float),
            volume=vol,
            baseline_shares=base,
            obs_success=np.array(obs_s, float),
            obs_attempts=np.array(obs_a, float),
            prior_rate=np.array(prior_r, float),
            kappa=np.array(kap, float),
            risk_n=risk_n,
        )
        # Attach a diagnostic array (which gateways used the pooled fallback).
        # Not on the dataclass so it doesn't force a schema change downstream.
        problem.pooled_fallback = np.array(is_pool, bool)  # type: ignore[attr-defined]
        # Attach which gateways are auto-explore (capable-but-untested) candidates.
        # Non-Thompson engines cap the COMBINED explore share per cell (and each
        # individually) so unproven gateways can't dilute proven volume; Thompson
        # ignores the flag (its wide posterior self-limits). Same attach-not-schema
        # pattern as pooled_fallback so nothing downstream needs to change.
        if "is_explore" in profile.columns:
            _expl = profile["is_explore"].fillna(False).to_numpy(bool)
        else:
            _expl = np.zeros(len(gateways), bool)
        problem.is_explore = _expl  # type: ignore[attr-defined]
        problems.append(problem)
    # Surface silent join misses: if most gateways found no per-profile success data, the
    # forecast/success-rate keys probably don't line up (e.g. BIN vs bankName) and every
    # rate is really the pooled prior — a real, otherwise-invisible data bug.
    if _n_gw and _n_pool / _n_gw > 0.5:
        _matched = _n_gw - _n_pool
        _pct = 100 * _n_pool / _n_gw
        if _matched == 0:
            # NOTHING joined → the two sides key on genuinely different values: a real bug.
            warnings.warn(
                f"build_profile_problems: 0/{_n_gw} gateways matched a per-cell success rate — the "
                f"forecast and success-rate join keys don't line up AT ALL (likely a BIN-vs-bankName "
                f"mismatch on 'bin'); every rate is the pooled prior.", stacklevel=2)
        else:
            # Keys ALIGN (some matched) but per-profile data is sparse — EXPECTED on a granular
            # BIN-level forecast, where most BIN×gateway combos have no direct attempts and
            # correctly inherit the pooled prior. Informational, not a key mismatch.
            logger.info(
                f"   build_profile_problems: {_n_pool}/{_n_gw} ({_pct:.0f}%) gateway-profiles on the pooled "
                f"prior (sparse per-profile attempts); {_matched} matched, so the join keys ARE aligned — "
                f"expected at BIN grain, not a mismatch.")
    return problems


# [FN-050b]
def build_profile_problems(
    forecast: pd.DataFrame,
    success_rates: pd.DataFrame,
    default_risk: float = 0.006,
) -> list[ProfileProblem]:
    """PROFILE variant of :func:`build_profile_problems` — one CellProblem per
    (rpgt × currency × bin × pmp × Country) profile.

    Design (locked): the DECISION grain is the profile, but the SCORING (success-rate) grain
    stays at PROFILE — so success rates are joined on the PROFILE key (rpgt,currency,bin,gateway) and
    BROADCAST onto each profile (no pmp/Country split of the thin conversion data). `forecast`
    must already carry `pmp` and `ctry` columns with the volume apportioned to profiles (see
    `routing_optimiser.s3_problem.profile.expand_forecast_to_profiles`). `bin` is the raw BIN and
    the profile identity is carried on `CellProblem.pmp` / `.ctry`, so the band projector's
    profile scaffold (keyed bin/pmp/ctry) still aligns.

    `build_profile_problems` is left byte-identical; this is a separate, gated path.
    """
    def _nk(x):
        return str(x).strip().casefold()

    _srn = success_rates.copy()
    for _c in ("rpgt", "currency", "bin", "gateway"):
        if _c in _srn.columns:
            _srn[_c] = _srn[_c].map(_nk)
    sr = _srn.set_index(["rpgt", "currency", "bin", "gateway"])
    if sr.index.has_duplicates:
        sr = sr[~sr.index.duplicated(keep="first")]
    if len(sr) and {"success", "attempts"}.issubset(sr.columns) and float(sr["attempts"].sum()) > 0:
        _global_rate = float(sr["success"].sum() / sr["attempts"].sum())
    elif len(sr):
        _global_rate = float(sr["success_rate"].mean())
    else:
        _global_rate = 0.85
    _has_prior = "prior_rate" in sr.columns
    _has_kappa = "kappa" in sr.columns

    _fc = forecast.copy()
    if "pmp" not in _fc.columns:
        _fc["pmp"] = "_all_"
    if "ctry" not in _fc.columns:
        _fc["ctry"] = "_all_"

    problems: list[ProfileProblem] = []
    _n_gw = _n_pool = 0
    for (rpgt, currency, bin_, pmp, ctry), profile in _fc.groupby(
            ["rpgt", "currency", "bin", "pmp", "ctry"]):
        gateways = list(profile["gateway"])
        vol = float(profile["volume"].sum())
        base = profile["baseline_share"].to_numpy(float)
        base = base / base.sum() if base.sum() > 0 else np.full(len(gateways), 1 / len(gateways))

        succ, obs_s, obs_a, is_pool, prior_r, kap = [], [], [], [], [], []
        for gw in gateways:
            key = (_nk(rpgt), _nk(currency), _nk(bin_), _nk(gw))   # PROFILE-grain rate (broadcast)
            _n_gw += 1
            if key in sr.index:
                row = sr.loc[key]
                succ.append(float(row["success_rate"]))
                obs_s.append(float(row["success"])); obs_a.append(float(row["attempts"]))
                prior_r.append(float(row["prior_rate"]) if _has_prior else float(row["success_rate"]))
                kap.append(float(row["kappa"]) if _has_kappa else 0.0)
                is_pool.append(False)
            else:
                _n_pool += 1
                succ.append(_global_rate); obs_s.append(0.0); obs_a.append(0.0)
                prior_r.append(_global_rate); kap.append(0.0); is_pool.append(True)

        risk = (profile["risk_rate"].to_numpy(float) if "risk_rate" in profile.columns
                else np.full(len(gateways), default_risk))
        if "risk_n" in profile.columns:
            risk_n = pd.to_numeric(profile["risk_n"], errors="coerce").fillna(0.0).to_numpy(float)
        elif "volume" in profile.columns:
            risk_n = pd.to_numeric(profile["volume"], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            risk_n = None

        problem = ProfileProblem(
            rpgt=str(rpgt), currency=str(currency), bin=str(bin_),
            gateways=gateways, success_rates=np.array(succ, float),
            risk_rates=np.array(risk, float), volume=vol, baseline_shares=base,
            obs_success=np.array(obs_s, float), obs_attempts=np.array(obs_a, float),
            prior_rate=np.array(prior_r, float), kappa=np.array(kap, float), risk_n=risk_n,
            pmp=str(pmp), ctry=str(ctry))
        problem.pooled_fallback = np.array(is_pool, bool)  # type: ignore[attr-defined]
        _expl = (profile["is_explore"].fillna(False).to_numpy(bool)
                 if "is_explore" in profile.columns else np.zeros(len(gateways), bool))
        problem.is_explore = _expl  # type: ignore[attr-defined]
        problems.append(problem)
    return problems


# [FN-052]
def prepare_inputs(success_source, forecast_path: str | None = None,
                   shrink_strength: float = 12.0):
    """Convenience: load everything and return (problems, success_rates, forecast).

    success_source may be a CSV/parquet path or an already-loaded DataFrame.
    """
    sdf = load_success_data(success_source)
    sr = gateway_success_rates(sdf, shrink_strength=shrink_strength)
    forecast = load_forecast(forecast_path, sdf)
    problems = build_profile_problems(forecast, sr)
    return problems, sr, forecast
