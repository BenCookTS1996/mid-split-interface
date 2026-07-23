"""
Hard and soft constraints for the split optimiser.

Hard constraints  -> a split is only a valid candidate if ALL are satisfied.
Soft constraints  -> a split MAY break these, but pays a penalty in the score.

The UI exposes these directly: hard constraints are the "must obey" inputs,
soft constraints are the "prefer, but bendable" inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HardConstraints:
    """Feasibility rules. Any candidate split violating one is rejected."""

    # Max fraction of a single cell's volume any one gateway may carry (0-1).
    # Keeping this < 1 forces at least some diversification. Mirrors the
    # MAX_GATEWAY_CAP = 97% "always keep a backup" idea in your k-means script.
    max_gateway_share: float = 0.97

    # VAMP / chargeback ceiling for the cell, as a rate (e.g. 0.009 = 0.9%).
    # Enforced as: sum_g(share_g * expected_risk_rate_g) <= vamp_cap.
    # None disables the per-cell cap (e.g. when you enforce risk only at the
    # portfolio level in a later aggregation step).
    vamp_cap: float | None = 0.009

    # Gateways that are hard-banned for this cell (e.g. currency not supported).
    banned_gateways: tuple[str, ...] = ()

    # Gateways that MUST receive traffic if eligible (rarely needed, but here
    # for "this bank must always keep some volume on gateway X" style rules).
    forced_gateways: tuple[str, ...] = ()


@dataclass
class SoftConstraints:
    """Preferences. Broken freely but penalised in the objective."""

    # Minimum share you'd LIKE every eligible gateway to keep, so you never
    # lose visibility of how a gateway is performing (the exploration floor).
    # Soft: the optimiser is nudged, not forced, to honour it.
    exploration_floor: float = 0.03

    # Penalty weight applied when a gateway falls below the exploration floor.
    exploration_penalty: float = 1.0

    # Penalty weight for drifting away from the current baseline split
    # (operational stability - big reroutes are disruptive).
    stability_penalty: float = 0.0

    # Preferred gateways and the reward for using them (e.g. cheaper fees).
    preferred_gateways: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimiserSettings:
    """Top-level knobs for a whole optimisation run."""

    # Conversion<->risk slider in [0, 1].
    #   1.0 = maximise conversion, no regard for risk
    #   0.0 = minimise risk, no regard for conversion
    risk_conversion_weight: float = 0.7

    # Which engine to use (key into engines.ENGINES).
    engine: str = "entropy"

    # Engine-specific extra parameters (e.g. softmax temperature, entropy lambda).
    engine_params: dict = field(default_factory=dict)

    hard: HardConstraints = field(default_factory=HardConstraints)
    soft: SoftConstraints = field(default_factory=SoftConstraints)
