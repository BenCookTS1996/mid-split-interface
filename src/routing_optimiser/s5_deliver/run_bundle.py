"""Reproducible run bundles + cooperative graceful stop (OPT-IN, additive).

Adapted from the co-worker pipeline's timestamped `runs/` folders. Nothing here runs
unless the caller calls it; importing this module has no side effects.

  write_run_bundle(runs_dir, config, ...) -> folder
      Writes a timestamped folder holding the EXACT config used, a log, and any numpy
      artifacts (e.g. the winning genome / diversity archive / share vector), plus a
      meta.json with the build marker. Auto-prunes the oldest runs beyond `keep`. Makes
      every optimisation reproducible and auditable — matching the project's
      build-marker / crash-loudly / no-silent-fallback philosophy.

  make_stop_check(stop_file) -> callable
      Returns a zero-arg predicate the GAs can poll at each generation. Dropping the
      `_stop` file on disk (request_stop) makes a long run exit gracefully at the next
      generation boundary, keeping the best-so-far. clear_stop removes the flag.

Config is written as YAML when PyYAML is importable, else JSON — no hard dependency.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import numpy as np

__build__ = "2026-07-24-run-bundle+graceful-stop+fitness-field"

STOP_FILENAME = "_stop"


# [FN-209]
def _write_config(folder: str, config: dict) -> str:
    """Write config as YAML if available, else JSON. Returns the path written."""
    try:
        import yaml  # optional
        path = os.path.join(folder, "config.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        return path
    except Exception:  # noqa: BLE001 - PyYAML missing or unserialisable -> JSON fallback
        path = os.path.join(folder, "config.json")
        with open(path, "w") as fh:
            json.dump(config, fh, indent=2, default=str)
        return path


# [FN-210]
def prune_old_runs(runs_dir: str, keep: int = 20) -> list[str]:
    """Keep the `keep` most-recent run folders under `runs_dir`; delete the rest.
    Returns the list of removed folder paths. Never raises if a folder is already gone."""
    if keep is None or keep <= 0 or not os.path.isdir(runs_dir):
        return []
    subs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))]
    subs.sort(key=lambda p: os.path.getmtime(p))          # oldest first
    removed = []
    for p in subs[:-keep] if len(subs) > keep else []:
        try:
            shutil.rmtree(p)
            removed.append(p)
        except OSError:
            pass
    return removed


# [FN-211]
def write_run_bundle(runs_dir: str, config: dict, *, log="", artifacts: dict | None = None,
                     name: str | None = None, keep: int = 20) -> str:
    """Create runs_dir/<timestamp[_name]>/ with config, log.txt, artifacts.npz, meta.json.
    Prunes old runs to `keep`. Returns the created folder path.

    artifacts: {str: array-like} saved via np.savez (e.g. {'genome':..., 'archive':...,
    'shares':...}). Non-array values are coerced with np.asarray."""
    os.makedirs(runs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = os.path.join(runs_dir, f"{ts}_{name}" if name else ts)
    os.makedirs(folder, exist_ok=True)

    _write_config(folder, config)
    with open(os.path.join(folder, "log.txt"), "w") as fh:
        fh.write(log if isinstance(log, str) else "\n".join(str(x) for x in log))
    if artifacts:
        np.savez(os.path.join(folder, "artifacts.npz"),
                 **{k: np.asarray(v) for k, v in artifacts.items()})
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump({"created": ts, "__build__": __build__,
                   "artifacts": sorted((artifacts or {}).keys())}, fh, indent=2)

    prune_old_runs(runs_dir, keep=keep)
    return folder


# --------------------------------------------------------------------------- graceful stop
# [FN-212]
def _stop_path(target: str) -> str:
    """A directory -> its _stop file; a file path -> itself."""
    return os.path.join(target, STOP_FILENAME) if os.path.isdir(target) else target


# [FN-213]
def request_stop(target: str) -> str:
    """Drop the stop flag (target may be a runs dir or an explicit file path)."""
    p = _stop_path(target)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as fh:
        fh.write(datetime.now().isoformat())
    return p


# [FN-214]
def clear_stop(target: str) -> None:
    p = _stop_path(target)
    try:
        os.remove(p)
    except OSError:
        pass


# [FN-215]
def stop_requested(target: str) -> bool:
    return os.path.exists(_stop_path(target))


class _StopCheck:
    """Picklable zero-arg predicate: True once the stop-flag file exists.

    A plain module-level callable CLASS (not a lambda/closure) so it survives being pickled
    into `loky`/process workers — a lambda cannot be pickled, which would force joblib's
    process backend to fall back to the slower sequential path."""
    __slots__ = ("path",)

    # [FN-216]
    def __init__(self, path: str):
        self.path = path

    # [FN-217]
    def __call__(self) -> bool:
        return os.path.exists(self.path)


# [FN-218]
def make_stop_check(target: str):
    """Return a zero-arg predicate for a GA's `stop_check` param: True once the flag exists.
    Usage:  run_midtilt_ga(ctx, lam, stop_check=make_stop_check(runs_dir))."""
    return _StopCheck(_stop_path(target))


class _ProgressWriter:
    """Picklable per-seed progress reporter for the GA's `progress_cb` param.

    Called once per generation with the number of candidate splits evaluated that generation
    (λ); it accumulates a running per-seed total and writes it to `path`. Because it's a plain
    module-level class (not a closure/lambda) it survives pickling into loky workers, so each
    parallel seed reports its own live count to its own file — the main process sums the files
    to show an aggregate "candidate splits evaluated so far" while the search runs. Best-effort:
    any write failure is swallowed so progress reporting can NEVER break a run."""
    __slots__ = ("path", "total", "best", "best_fit", "best_nv")

    # [FN-219]
    def __init__(self, path: str):
        self.path = path
        self.total = 0
        self.best = None                  # best-so-far engine score for this seed (higher = better)
        self.best_fit = None              # FITNESS (revenue-ish objective) of that best-so-far split
        self.best_nv = None               # # vampMids with an unmet per-MID band on that best split

    # [FN-220]
    def __call__(self, inc: int, score=None, fitness=None, nviol=None) -> None:
        try:
            self.total += int(inc)
            if score is not None:
                _s = float(score)
                # Track fitness + unmet-MID count ALONGSIDE the best score (not their own max/min):
                # we want the revenue and constraint state of the split that OWNS the current best
                # score, so they stay a matched triple.
                if self.best is None or _s > self.best:
                    self.best = _s
                    if fitness is not None:
                        self.best_fit = float(fitness)
                    if nviol is not None:
                        self.best_nv = int(nviol)
            _tmp = self.path + ".tmp"
            with open(_tmp, "w") as _f:
                # "total|best|fit|nviol" — trailing fields blank until reported (back-compatible: the
                # poller reads field 0 as the count and treats missing fields as None).
                _f.write("{}|{}|{}|{}".format(
                    self.total,
                    "" if self.best is None else repr(self.best),
                    "" if self.best_fit is None else repr(self.best_fit),
                    "" if self.best_nv is None else int(self.best_nv)))
            os.replace(_tmp, self.path)   # atomic swap so the poller never reads a torn record
        except Exception:  # noqa: BLE001
            pass
