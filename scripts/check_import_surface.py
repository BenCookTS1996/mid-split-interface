#!/usr/bin/env python3
"""Resolve EVERY `routing_optimiser` import target in the repo, and fail if one does not exist.

WHY THIS EXISTS (19fu). The 19fp restructure moved 27 modules into numbered step packages
(s1_extract / s2_forecast / s3_problem / s4_search / s5_deliver). It was verified by importing
all 37 modules in a fresh interpreter -- and that verification PASSED while the app was broken,
because it tested the wrong surface:

    verified:  every module inside the package imports cleanly
    NOT tested: whether the app's own import STATEMENTS still resolve

Those are different questions. `routing_optimiser/__init__.py` re-exports SYMBOLS
(`optimise_split`, `HardConstraints`, ...) but never binds the SUBMODULE objects, so

    from routing_optimiser import optimiser as _optmod        # tab 2, line 2914

kept parsing, kept compiling, and raised ImportError at RUN TIME -- inside a `render()` a user
only reaches after clicking Compute. py_compile cannot catch it (it is a name lookup, not
syntax) and importing the package cannot catch it (the failing statement lives in the app).

Eight sites were broken this way across three modules. This script closes that gap for good:
it parses every .py in the repo, collects every `from routing_optimiser... import X` and
`import routing_optimiser...`, and actually resolves each target.

It FLAGS ONLY -- it imports nothing into the app and changes no behaviour. Exit 0 = clean,
exit 1 = at least one target does not resolve, with the file:line of every site.

    python3 scripts/check_import_surface.py          # from the repo root
"""
from __future__ import annotations

import ast
import importlib
import os
import sys

PKG = "routing_optimiser"
SKIP_DIRS = {"__pycache__", ".git", ".venv", "data", "runs", "logs", "node_modules"}
# Walk OUR code only. A whitelist, not a blacklist: the repo also carries a vendored
# google-cloud-sdk/ full of Python-2 files that do not parse, and their noise would bury the
# one line that matters. Root-level .py files (main.py, ...) are picked up separately below.
CODE_ROOTS = ("app", "src", "scripts")


def _py_files(repo_root):
    """Every project .py file: the CODE_ROOTS trees plus the repo root's own scripts."""
    for fn in sorted(os.listdir(repo_root)):
        if fn.endswith(".py"):
            yield os.path.join(repo_root, fn)
    for root in CODE_ROOTS:
        base = os.path.join(repo_root, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    yield os.path.join(dirpath, fn)


def collect(repo_root):
    """(module, name) -> ["file:line", ...] for every PKG import statement in project code."""
    found: dict[tuple[str, str], list[str]] = {}
    n_files = 0
    if True:
        for path in _py_files(repo_root):
            n_files += 1
            rel = os.path.relpath(path, repo_root)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read(), rel)
            except (SyntaxError, UnicodeDecodeError) as exc:
                # Our own file failing to parse IS a finding -- loud, not swallowed.
                print(f"  !! could not parse {rel}: {exc}", file=sys.stderr)
                continue
            for node in ast.walk(tree):
                # `from routing_optimiser.x import y` -- y may be a symbol OR a submodule
                if isinstance(node, ast.ImportFrom) and node.module \
                        and (node.module == PKG or node.module.startswith(PKG + ".")):
                    for alias in node.names:
                        if alias.name != "*":
                            found.setdefault((node.module, alias.name), []).append(
                                f"{rel}:{node.lineno}")
                # `import routing_optimiser.x`
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == PKG or alias.name.startswith(PKG + "."):
                            found.setdefault(("<import>", alias.name), []).append(
                                f"{rel}:{node.lineno}")
    print(f"[check_import_surface] scanned {n_files} project .py file(s) "
          f"({', '.join(CODE_ROOTS)} + repo root)")
    return found


def resolve(module, name):
    """None if the target resolves, else the reason it does not."""
    try:
        if module == "<import>":
            importlib.import_module(name)
            return None
        mod = importlib.import_module(module)
        if hasattr(mod, name):
            return None
        # not an attribute -- the only other legal shape is `from pkg import submodule`
        importlib.import_module(f"{module}.{name}")
        return None
    except Exception as exc:  # noqa: BLE001 -- any failure is a failure worth reporting
        return f"{type(exc).__name__}: {exc}"


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for extra in (os.path.join(repo_root, "src"), os.path.join(repo_root, "app")):
        if extra not in sys.path:
            sys.path.insert(0, extra)   # mirrors how Streamlit sees the app

    targets = collect(repo_root)
    broken = []
    for (module, name), sites in sorted(targets.items()):
        why = resolve(module, name)
        if why:
            broken.append((module, name, why, sites))

    print(f"[check_import_surface] {len(targets)} distinct {PKG} import target(s) checked")
    if not broken:
        print("[check_import_surface] OK — every target resolves")
        return 0

    print(f"[check_import_surface] {len(broken)} BROKEN target(s):\n")
    for module, name, why, sites in broken:
        stmt = f"import {name}" if module == "<import>" else f"from {module} import {name}"
        print(f"  {stmt}")
        print(f"      {why}")
        for site in sites:
            print(f"      at {site}")
        print()
    print("These parse and compile but raise ImportError at RUN TIME. Fix the statement or the "
          "package, then re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
