"""Engines retired from the UI on 2026-08-31.

The Routing Engine dropdown now offers ONLY the full-matrix GA. These four modules are no
longer selectable, but they are still imported and still registered in
`routing_optimiser.engines.ENGINES`, for three reasons:

  * `OptimiserSettings.engine` defaults to "entropy" (constraints.py), so `get_engine()` has
    to resolve it for any caller that does not pass an engine explicitly;
  * `ThompsonEngine` and `PortfolioEngine` both subclass `SoftmaxEngine`, so softmax cannot
    be retired without taking the other two with it;
  * several `engine_key == "softmax" / "thompson" / "portfolio"` branches survive in
    tab2_engine.py and tab3_impact.py and would raise if the classes disappeared.

`base.py` and `genetic_ref.py` deliberately stayed in `engines/`: the whole pipeline uses
`CellProblem` / `BaseEngine`, and the full-matrix GA dispatches its own revenue reference
through `get_engine("genetic_ref")`.

`from .base import ...` was rewritten to `from ..engines.base import ...` on the move; nothing
else in these files changed.
"""
