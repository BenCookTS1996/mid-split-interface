"""Engine registry. The UI dropdown is built straight from ENGINES."""
from __future__ import annotations

from .base import BaseEngine, CellProblem, CellSolution
from .entropy import EntropyEngine
from .genetic_ref import GeneticRefEngine
from .portfolio import PortfolioEngine
from .softmax import SoftmaxEngine
from .thompson import ThompsonEngine

# NOTE: the "genetic" option in the UI is served by the CROSS-CELL tilt GA
# (routing_optimiser.genetic_global.run_midtilt_ga), which the app dispatches
# directly — it is NOT a registry engine. The old per-cell GeneticEngine was
# removed; the dropdown injects the "genetic" option itself (see streamlit_app).
ENGINES: dict[str, type[BaseEngine]] = {
    e.key: e for e in [
        SoftmaxEngine, EntropyEngine, ThompsonEngine,
        PortfolioEngine, GeneticRefEngine,
    ]
}


def get_engine(key: str, weight: float, hard, soft, **params) -> BaseEngine:
    if key not in ENGINES:
        raise KeyError(f"Unknown engine '{key}'. Options: {list(ENGINES)}")
    return ENGINES[key](weight=weight, hard=hard, soft=soft, **params)


def engine_choices() -> list[tuple[str, str]]:
    """(key, label) pairs for building a dropdown."""
    return [(k, e.label) for k, e in ENGINES.items()]


__all__ = [
    "ENGINES", "get_engine", "engine_choices",
    "BaseEngine", "CellProblem", "CellSolution",
]
