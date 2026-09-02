"""Engine registry. The UI dropdown is built straight from ENGINES."""
from __future__ import annotations

from .base import BaseEngine, ProfileProblem, ProfileSolution
from .genetic_ref import GeneticRefEngine

# RETIRED FROM THE UI (2026-08-31), still imported on purpose. These four moved to
# routing_optimiser.legacy_engines because the dropdown now offers only the full-matrix GA.
# They must remain REGISTERED, not just hidden:
#   * OptimiserSettings.engine still defaults to "entropy" (constraints.py:75), so
#     get_engine() must be able to resolve it for any caller that does not pass one;
#   * ThompsonEngine and PortfolioEngine both subclass SoftmaxEngine, so softmax cannot
#     be dropped without taking the other two with it.
# base.py and genetic_ref.py deliberately did NOT move: the whole pipeline uses ProfileProblem /
# BaseEngine, and the full-matrix GA dispatches its own revenue reference through
# get_engine("genetic_ref") (tab_2_routing_engine.py, `optimise_split(agg_problems, ref_settings)`).
from ..legacy_engines.entropy import EntropyEngine
from ..legacy_engines.portfolio import PortfolioEngine
from ..legacy_engines.softmax import SoftmaxEngine
from ..legacy_engines.thompson import ThompsonEngine

# NOTE: the "genetic" option in the UI is served by the CROSS-PROFILE tilt GA
# (routing_optimiser.legacy_engines.midtilt_cmaes.run_midtilt_ga, RETIRED 19gd and no longer
# reachable), which the app dispatched
# directly — it is NOT a registry engine. The old per-profile GeneticEngine was
# removed; the dropdown injects the "genetic" option itself (see streamlit_app).
ENGINES: dict[str, type[BaseEngine]] = {
    e.key: e for e in [
        SoftmaxEngine, EntropyEngine, ThompsonEngine,
        PortfolioEngine, GeneticRefEngine,
    ]
}


# [FN-399]
def get_engine(key: str, weight: float, hard, soft, **params) -> BaseEngine:
    if key not in ENGINES:
        raise KeyError(f"Unknown engine '{key}'. Options: {list(ENGINES)}")
    return ENGINES[key](weight=weight, hard=hard, soft=soft, **params)


# [FN-400]
def engine_choices() -> list[tuple[str, str]]:
    """(key, label) pairs for building a dropdown."""
    return [(k, e.label) for k, e in ENGINES.items()]


__all__ = [
    "ENGINES", "get_engine", "engine_choices",
    "BaseEngine", "ProfileProblem", "ProfileSolution",
]
