"""Mastercard pipeline runner — a SHIM. The runner is main.py (19fk).

    python main_mastercard.py       ==  python main.py --scheme mastercard

Kept only so an existing habit, script or cron line does not break. It carries
no logic: if it ever seems to behave differently from `main.py --scheme
mastercard`, that is a bug in this file, not a Mastercard difference. Safe to
delete once nothing calls it.
"""
import sys

from main import main

if __name__ == "__main__":
    main(["--scheme", "mastercard", *sys.argv[1:]])
