#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-click launcher for the Routing Optimiser.
#
#   • macOS: double-click this file in Finder, OR run  ./run.command  in a terminal.
#   • It bootstraps EVERYTHING a fresh clone needs, then opens the app:
#       1. finds Python 3 (prefers 3.8 to match the pinned environment),
#       2. creates a local virtual environment (.venv) on first run,
#       3. installs / updates the pinned dependencies (idempotent — fast after run 1),
#       4. checks for the gcloud CLI (needed for live BigQuery) and tells you if it's missing,
#       5. launches Streamlit (a browser tab opens at http://localhost:8501).
#
# Safe to run every time — pip skips anything already satisfied. The .venv folder is
# git-ignored, so this never touches the repo or your global Python.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# Keep the window open on any error so a double-click user can read what happened.
trap 'echo; echo "❌ Something went wrong (see the message above)."; read -r -p "Press Enter to close…" _; exit 1' ERR

# 1. Locate a Python interpreter (prefer 3.8 → pinned target; else any python3).
PY="$(command -v python3.8 || command -v python3 || true)"
if [ -z "$PY" ]; then
  echo "❌ Python 3 was not found."
  echo "   Install Python 3.8+ from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close…" _; exit 1
fi
echo "Using Python: $PY ($("$PY" --version 2>&1))"

# 2. Create the virtual environment on first run.
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv)…"
  "$PY" -m venv .venv
fi
VENV_PY="./.venv/bin/python"

# 3. Install / update dependencies (idempotent).
echo "Installing / updating dependencies — first run takes a few minutes, later runs are quick…"
"$VENV_PY" -m pip install --upgrade pip >/dev/null
"$VENV_PY" -m pip install -r requirements.txt

# 4. Heads-up about the gcloud CLI (needed for LIVE BigQuery runs; not a pip package).
if ! command -v gcloud >/dev/null 2>&1; then
  echo ""
  echo "⚠️  The 'gcloud' CLI was not found on PATH."
  echo "    Live BigQuery runs need it — install from https://cloud.google.com/sdk/docs/install"
  echo "    (You can still use cached / previously-run outputs without it. Once gcloud is"
  echo "     installed, the app's Environment-check panel gives you a one-click sign-in.)"
  echo ""
fi

# 5. Launch the app.
echo "✅ Ready — starting the app. A browser tab will open at http://localhost:8501"
echo "   (Leave this window open while you use the app; close it or press Ctrl+C to stop.)"
exec "./.venv/bin/streamlit" run app/streamlit_app.py
