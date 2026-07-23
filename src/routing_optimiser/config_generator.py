"""
Generate JSON routing configs from the compressed split.

Adapted from your config_generator.py, made self-contained (no Colab/gspread).
Each compressed rule becomes a weighted connector "pool" with the RPGT type
selectors, currency filter, and country filter your production configs use.
"""
from __future__ import annotations

import json
import os
import uuid

import pandas as pd

# RPGT -> (term, type selectors) ported from your config generator.
RPGT_MAP = {
    "Monthly Initial": ("p1m-ini", [
        {"key": "charge.meta.item.duration", "operator": "Lt", "conversion": "", "values": ["3456000"]},
        {"key": "charge.renewalNumber", "operator": "Equal", "conversion": "", "values": ["0"]},
        {"key": "charge.meta.item.skuType", "operator": "Equal", "conversion": "", "values": ["SKU_TYPE_PRIMARY"]},
    ]),
    "Annual Sub Sale": ("p1y-ini", [
        {"key": "charge.meta.item.duration", "operator": "Gt", "conversion": "", "values": ["3456000"]},
        {"key": "charge.renewalNumber", "operator": "Equal", "conversion": "", "values": ["0"]},
    ]),
    "Addon Sale": ("addon-ini", [
        {"key": "charge.meta.item.skuType", "operator": "Equal", "conversion": "", "values": ["SKU_TYPE_ADDON"]},
        {"key": "charge.renewalNumber", "operator": "Equal", "conversion": "", "values": ["0"]},
    ]),
    "Upgrades": ("upgrade-ini", [
        {"key": "charge.meta.item.name", "operator": "InLike", "conversion": "", "values": ["Modify", "Upgrade"]},
        {"key": "charge.renewalNumber", "operator": "Equal", "conversion": "", "values": ["0"]},
    ]),
    "Monthly Renewal": ("p1m-ren", [
        {"key": "charge.meta.item.duration", "operator": "Lt", "conversion": "", "values": ["3456000"]},
        {"key": "charge.renewalNumber", "operator": "Gt", "conversion": "", "values": ["0"]},
    ]),
    "Annual Sub Renewal": ("p1y-ren", [
        {"key": "charge.meta.item.duration", "operator": "Gt", "conversion": "", "values": ["25920000"]},
        {"key": "charge.renewalNumber", "operator": "Gt", "conversion": "", "values": ["0"]},
    ]),
    "P6M Renewals": ("p6m-ren", [
        {"key": "charge.meta.item.duration", "operator": "Gt", "conversion": "", "values": ["3456000"]},
        {"key": "charge.meta.item.duration", "operator": "Lt", "conversion": "", "values": ["25920000"]},
        {"key": "charge.renewalNumber", "operator": "Gt", "conversion": "", "values": ["0"]},
    ]),
    "Addon Renewal": ("addon-ren", [
        {"key": "charge.meta.item.skuType", "operator": "Equal", "conversion": "", "values": ["SKU_TYPE_ADDON"]},
        {"key": "charge.renewalNumber", "operator": "Gt", "conversion": "", "values": ["0"]},
    ]),
}


def _currency_expr(currency: str) -> list[dict]:
    return [{"key": "charge.amount.currency", "operator": "Equal",
             "conversion": "", "values": [str(currency)]}]


def _scheme_expr(scheme: str) -> list[dict]:
    if scheme == "vi":
        return [{"key": "method.paymentScheme", "operator": "Equal", "conversion": "", "values": ["card_visa"]}]
    if scheme == "non-vi":
        return [{"key": "method.paymentScheme", "operator": "NotEqual", "conversion": "", "values": ["card_visa"]}]
    return []


# Production match key + values for the wallet (paymentMethodProvider) selector.
# The data has three values: non_gp_ap (non-wallet), GOOGLEPAY, APPLEPAY (wallet).
# Uses only Equal/NotEqual (already used for scheme) to avoid an unsupported operator.
# NOTE: confirm WALLET_KEY matches the production schema (e.g. a "method." prefix).
WALLET_KEY = "paymentMethodProvider"
NONWALLET_VALUE = "non_gp_ap"


def _pmp_expr(pmp) -> list[dict]:
    pmp = str(pmp).strip().lower()
    if pmp == "wallet":               # GOOGLEPAY or APPLEPAY -> everything that isn't non_gp_ap
        return [{"key": WALLET_KEY, "operator": "NotEqual", "conversion": "", "values": [NONWALLET_VALUE]}]
    if pmp in ("non_gp_ap", "nonwallet", "non-wallet"):
        return [{"key": WALLET_KEY, "operator": "Equal", "conversion": "", "values": [NONWALLET_VALUE]}]
    return []


def _make_pool(rule: pd.Series, gateway_cols: list[str], brand: str,
               scheme: str) -> dict | None:
    rpgt = rule["rpgt"]
    if rpgt not in RPGT_MAP:
        return None
    term, selectors = RPGT_MAP[rpgt]

    connectors = {}
    for gc in gateway_cols:
        val = float(rule.get(gc, 0) or 0)
        if val > 0:
            connectors[gc] = round(val, 4)  # weight as a percentage
    if not connectors:
        return None

    pmp = rule.get("pmp")
    pmp_expr = _pmp_expr(pmp) if pmp is not None else []
    expressions = _currency_expr(rule["currency"]) + _scheme_expr(scheme) + pmp_expr + selectors
    _pmp_tag = f"-{str(pmp).strip().lower()}" if (pmp is not None and str(pmp).strip()) else ""
    return {
        "id": str(uuid.uuid4()),
        "name": f"{brand}-{term}-{str(rule['currency']).lower()}{_pmp_tag}",
        "term": term,
        "rpgt": rpgt,
        "currency": rule["currency"],
        "paymentMethodProvider": (None if pmp is None else str(pmp)),
        "coversBanks": rule.get("banks", []),
        "match": {"operator": "And", "expressions": expressions},
        "connectorPool": [
            {"connector": k, "weight": v} for k, v in connectors.items()
        ],
    }


def build_configs(compressed: pd.DataFrame, brand: str = "tdr",
                  scheme: str = "vi") -> dict[str, list[dict]]:
    """Return {rpgt: [pool, ...]} ready to serialise."""
    meta = {"rpgt", "currency", "banks", "n_cells", "volume", "pmp"}
    gateway_cols = [c for c in compressed.columns if c not in meta]
    configs: dict[str, list[dict]] = {}
    for _, rule in compressed.iterrows():
        pool = _make_pool(rule, gateway_cols, brand, scheme)
        if pool is None:
            continue
        configs.setdefault(rule["rpgt"], []).append(pool)
    return configs


def write_configs(configs: dict[str, list[dict]], outdir: str,
                  brand: str = "tdr", date: str = "260629") -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for rpgt, pools in configs.items():
        term = RPGT_MAP.get(rpgt, (rpgt.lower().replace(" ", "-"),))[0]
        fname = f"{brand}_{term}_{date}.json"
        path = os.path.join(outdir, fname)
        with open(path, "w") as f:
            json.dump({"brand": brand, "rpgt": rpgt, "date": date, "pools": pools},
                      f, indent=2)
        paths.append(path)
    # combined manifest
    manifest = os.path.join(outdir, f"{brand}_all_configs_{date}.json")
    with open(manifest, "w") as f:
        json.dump(configs, f, indent=2)
    paths.append(manifest)
    return paths
