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

__build__ = "2026-07-24-run-bundle+graceful-stop"

STOP_FILENAME = "_stop"


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
def _stop_path(target: str) -> str:
    """A directory -> its _stop file; a file path -> itself."""
    return os.path.join(target, STOP_FILENAME) if os.path.isdir(target) else target


def request_stop(target: str) -> str:
    """Drop the stop flag (target may be a runs dir or an explicit file path)."""
    p = _stop_path(target)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as fh:
        fh.write(datetime.now().isoformat())
    return p


def clear_stop(target: str) -> None:
    p = _stop_path(target)
    try:
        os.remove(p)
    except OSError:
        pass


def stop_requested(target: str) -> bool:
    return os.path.exists(_stop_path(target))


class _StopCheck:
    """Picklable zero-arg predicate: True once the stop-flag file exists.

    A plain module-level callable CLASS (not a lambda/closure) so it survives being pickled
    into `loky`/process workers — a lambda cannot be pickled, which would force joblib's
    process backend to fall back to the slower sequential path."""
    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path

    def __call__(self) -> bool:
        return os.path.exists(self.path)


def make_stop_check(target: str):
    """Return a zero-arg predicate for a GA's `stop_check` param: True once the flag exists.
    Usage:  run_midtilt_ga(ctx, lam, stop_check=make_stop_check(runs_dir))."""
    return _StopCheck(_stop_path(target))
