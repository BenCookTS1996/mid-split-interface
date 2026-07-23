"""Generate wallet-aware ConnectorPool JSON configs from the proposed split.

A faithful, in-process adaptation of the original Colab script
(`config-generator.ipynb`). The ONLY behavioural change is the INPUT: instead of
reading Compressed_Rules Google Sheets from Drive, it reads the SAME template as
pandas DataFrames — exactly the ones produced by `build_split_exports` in the app
(one DataFrame per Brand × RPGT, identical column layout to the .gsheet/.xlsx
templates). All the pool-building logic (grouping, provider combination, priority,
weight normalisation, selectors, naming) is preserved verbatim.

(The older `config_generator.build_configs` — a simpler per-cell pooler used by the
retired k-means tab — is left untouched; this module is the script-faithful path.)

Entry point: ``generate_configs(exports, brand_key, date, ...)`` → returns
``(pools, counts)`` where ``pools`` is ``{pool_name: pool_dict}`` (ready to write
as ``<name>.json``) and ``counts`` is a small summary dict. No Drive / gspread /
Colab dependencies; nothing is written to disk (the caller zips/downloads).
"""
from __future__ import annotations

import uuid
from collections import defaultdict

__build__ = "2026-07-15-connector-pool-configs-from-split-templates"

# --- Brand-specific constants (verbatim from the script) --------------------
BRANDS = {
    "tav": {"project_id": "totalav",       "name": "TotalAV",       "prefix": "tav", "pool_dir": "totalav/ConnectorPool"},
    "tab": {"project_id": "adblocker-prod", "name": "Total Adblock", "prefix": "tab", "pool_dir": "adblocker-prod/ConnectorPool"},
    "tdr": {"project_id": "totaldrive",     "name": "Total Drive",   "prefix": "tdr", "pool_dir": "totaldrive/ConnectorPool"},
}

GENERIC_SKIP = {"tdr": {"pool-paypal"}}

SALE_TERMS = ("p1m-ini", "p1y-ini", "addon-ini", "upgrade-ini")
RENEWAL_TERMS = ("p1m-ren", "p1y-ren", "p6m-ren", "addon-ren")

RPGT_MAP = {
    "Monthly Initial": ("p1m-ini", [
        {"key": "charge.meta.item.duration", "operator": "Lt", "conversion": "", "values": ["3456000"]},
        {"key": "charge.renewalNumber", "operator": "Equal", "conversion": "", "values": ["0"]},
        {"key": "charge.meta.item.skuType", "operator": "Equal", "conversion": "", "values": ["SKU_TYPE_PRIMARY"]},
    ]),
    "Annual Sub Sale": ("p1y-ini", [
        {"key": "charge.meta.item.duration", "operator": "Gt", "conversion": "", "values": ["3456000"]},
        {"key": "charge.renewalNumber", "operator": "Equal", "conversion": "", "values": ["0"]},
        {"key": "charge.meta.item.skuType", "operator": "Equal", "conversion": "", "values": ["SKU_TYPE_PRIMARY"]},
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

PROVIDER_TAG = {"non_gp_ap": "", "APPLEPAY": "-ap", "GOOGLEPAY": "-gp"}


def company_to_brand_key(company: str) -> str:
    """Map a company display name to a BRANDS key (defaults to 'tav')."""
    c = str(company or "").strip().lower().replace(" ", "")
    if c in ("totaladblock", "adblock", "tab"):
        return "tab"
    if c in ("totaldrive", "drive", "tdr"):
        return "tdr"
    return "tav"


def get_priority(term, is_apgp, has_bins, extra_priority_amount=200000):
    is_special = term in ("upgrade-ini", "addon-ren")
    base = 500000 if is_special else 100000
    if not has_bins:
        calculated_priority = base + (100000 if is_apgp else 0)
    else:
        calculated_priority = base + 200000 + (100000 if is_apgp else 0)
    return calculated_priority + extra_priority_amount


def get_combinable_provider_sets(providers):
    """Determines which providers can be safely combined into a single pool."""
    p_set = set(providers)
    if p_set == {"APPLEPAY", "GOOGLEPAY", "non_gp_ap"}:
        return [["APPLEPAY", "GOOGLEPAY", "non_gp_ap"]]
    elif {"APPLEPAY", "GOOGLEPAY"}.issubset(p_set):
        sets = [["APPLEPAY", "GOOGLEPAY"]]
        for p in p_set - {"APPLEPAY", "GOOGLEPAY"}:
            sets.append([p])
        return sets
    else:
        return [[p] for p in providers]


def rows_from_dataframe(df, brand_name):
    """Replaces the script's ``parse_sheet``: turn a Compressed_Rules template
    DataFrame into the same list-of-dict rows, using the identical column layout
    (Brand=2, RPGT=3, Currency=4, BIN=5, provider=6, 'Check' marks the start of
    connector columns, 'Country' optional)."""
    if df is None or getattr(df, "empty", True):
        return []
    headers = [str(c) for c in df.columns]
    data = df.fillna("").astype(str).values.tolist()
    if "Check" not in headers:
        return []
    check_idx = headers.index("Check")
    connector_cols = [h for h in headers[check_idx + 1:] if h and h != "DUP CHECK"]
    connector_indices = [i for i, h in enumerate(headers) if h in connector_cols]
    try:
        country_idx = headers.index("Country")
    except ValueError:
        country_idx = None

    rows = []
    for row_data in data:
        row = row_data + [""] * (len(headers) - len(row_data))
        if row[2] != brand_name:
            continue
        country_val = str(row[country_idx]).strip() if country_idx is not None and row[country_idx] else None
        if country_val:
            if country_val.upper() in ("USA", "US"):
                country_val = "USA"
            elif country_val.upper() in ("NON-USA", "NON-US"):
                country_val = "Non-USA"
        bin_val = str(row[5]) if row[5] != "" else "Other"
        if bin_val not in ("Other", "All"):
            try:
                bin_val = str(int(float(bin_val)))
            except (ValueError, TypeError):
                continue
        provider = row[6]
        if provider == "All":
            provider = "non_gp_ap"
        connectors = {}
        for ci in connector_indices:
            col_name = headers[ci]
            val = row[ci]
            if val and str(val).strip() != "":
                try:
                    f_val = float(val)
                    if f_val > 0:
                        connectors[col_name] = f_val
                except ValueError:
                    continue
        rows.append({
            "rpgt": row[3], "currency": row[4], "bin": bin_val,
            "provider": provider, "country": country_val, "connectors": connectors,
        })
    return rows


def make_pool(cfg, name, priority, currencies, bins, providers, scheme_filter,
              type_selectors, connectors_weighted, country_label):
    expressions = []
    if currencies and len(currencies) < 5:
        if len(currencies) == 1:
            expressions.append({"key": "charge.amount.currency", "operator": "Equal", "conversion": "", "values": list(currencies)})
        else:
            expressions.append({"key": "charge.amount.currency", "operator": "In", "conversion": "", "values": sorted(currencies)})

    p_set = set(providers)
    if p_set == {"APPLEPAY", "GOOGLEPAY", "non_gp_ap"}:
        pass
    elif p_set == {"APPLEPAY", "GOOGLEPAY"}:
        expressions.append({"key": "method.provider", "operator": "In", "conversion": "",
                            "values": ["PAYMENT_METHOD_PROVIDER_APPLEPAY", "PAYMENT_METHOD_PROVIDER_GOOGLEPAY"]})
    else:
        provider = providers[0]
        if provider == "APPLEPAY":
            expressions.append({"key": "method.provider", "operator": "Equal", "conversion": "", "values": ["PAYMENT_METHOD_PROVIDER_APPLEPAY"]})
        elif provider == "GOOGLEPAY":
            expressions.append({"key": "method.provider", "operator": "Equal", "conversion": "", "values": ["PAYMENT_METHOD_PROVIDER_GOOGLEPAY"]})
        elif provider == "non_gp_ap":
            expressions.append({"key": "method.provider", "operator": "NotIn", "conversion": "",
                                "values": ["PAYMENT_METHOD_PROVIDER_APPLEPAY", "PAYMENT_METHOD_PROVIDER_GOOGLEPAY"]})

    if scheme_filter == "vi":
        expressions.append({"key": "method.paymentScheme", "operator": "Equal", "conversion": "", "values": ["card_visa"]})
    elif scheme_filter == "non-vi":
        expressions.append({"key": "method.paymentScheme", "operator": "NotEqual", "conversion": "", "values": ["card_visa"]})

    expressions.extend(type_selectors)

    if country_label == "USA":
        expressions.append({"key": "method.info.country", "operator": "Equal", "conversion": "", "values": ["US"]})
    elif country_label == "Non-USA":
        expressions.append({"key": "method.info.country", "operator": "NotEqual", "conversion": "", "values": ["US"]})

    if bins:
        if len(bins) == 1:
            expressions.append({"key": "method.info.card.bin", "operator": "Equal", "conversion": "", "values": [bins[0]]})
        else:
            expressions.append({"key": "method.info.card.bin", "operator": "In", "conversion": "", "values": sorted(set(bins))})

    pool_connectors = [
        {"connectorId": cid, "priority": 0, "weighting": w, "uses": 0}
        for cid, w in sorted(connectors_weighted.items())
    ]
    return {
        "kind": "ConnectorPool",
        "metadata": {
            "projectId": cfg["project_id"], "name": name, "uuid": str(uuid.uuid4()),
            "displayName": "", "description": "", "annotations": {},
            "labels": None, "disabled": False,
        },
        "specVersion": "v1",
        "selector": {"priority": priority, "expressions": expressions},
        "spec": {"restriction": "fullCycle", "connectors": pool_connectors},
    }


def normalize_weights(connectors, expected_total):
    weighted = {k: round(v * 10) for k, v in connectors.items()}
    total = sum(weighted.values())
    diff = total - expected_total
    if diff == 0:
        return weighted
    sorted_keys = sorted(weighted.keys(), key=lambda k: -weighted[k])
    if not sorted_keys:
        return weighted
    step = -1 if diff > 0 else 1
    for i in range(abs(diff)):
        weighted[sorted_keys[i % len(sorted_keys)]] += step
    return weighted


def process_compressed_rows(cfg, rows, scheme_filter):
    """BIN-specific pools (bin != 'Other'). Returns {name: pool}."""
    out = {}
    if not rows:
        return out
    rpgt = rows[0]["rpgt"]
    if rpgt not in RPGT_MAP:
        return out
    term, type_selectors = RPGT_MAP[rpgt]

    groups = defaultdict(lambda: {"bins": set(), "raw": {}})
    for r in rows:
        if r["bin"] == "Other":
            continue
        sig = tuple(sorted((k, round(v, 4)) for k, v in r["connectors"].items()))
        key = (r["provider"], r["currency"], r["country"], sig)
        groups[key]["bins"].add(r["bin"])
        groups[key]["raw"] = dict(r["connectors"])

    config_to_pc = defaultdict(list)
    for (provider, currency, country, sig), data in groups.items():
        bins_tuple = tuple(sorted(data["bins"]))
        config_to_pc[(currency, sig, bins_tuple)].append((provider, country))

    tag_n = defaultdict(int)
    for (currency, sig, bins_tuple), pc_list in sorted(config_to_pc.items()):
        prov_to_countries = defaultdict(set)
        for p, c in pc_list:
            prov_to_countries[p].add(c)
        clabel_to_provs = defaultdict(list)
        for p, c_set in prov_to_countries.items():
            if {"USA", "Non-USA"}.issubset(c_set):
                clabel_to_provs["All"].append(p)
            elif "USA" in c_set:
                clabel_to_provs["USA"].append(p)
            elif "Non-USA" in c_set:
                clabel_to_provs["Non-USA"].append(p)
            else:
                clabel_to_provs["All"].append(p)

        for country_label, providers in sorted(clabel_to_provs.items()):
            for p_set in get_combinable_provider_sets(providers):
                orig_country = "USA" if country_label == "All" and "USA" in prov_to_countries[p_set[0]] else country_label
                if orig_country == "All":
                    orig_country = None
                raw = groups[(p_set[0], currency, orig_country, sig)]["raw"]
                weighted = normalize_weights(raw, 1000)
                is_apgp = all(p in ("APPLEPAY", "GOOGLEPAY") for p in p_set)
                priority = get_priority(term, is_apgp, True, cfg.get("extra_priority", 200000))
                if set(p_set) == {"APPLEPAY", "GOOGLEPAY"}:
                    tag = "-apgp"
                elif set(p_set) == {"APPLEPAY", "GOOGLEPAY", "non_gp_ap"}:
                    tag = ""
                else:
                    tag = PROVIDER_TAG[p_set[0]]
                country_tag = "-us" if country_label == "USA" else ("-nonus" if country_label == "Non-USA" else "")
                file_prefix = f"rr-{cfg['prefix']}" if term.endswith("-ren") else cfg["prefix"]
                tag_n[f"{tag}{country_tag}"] += 1
                n = tag_n[f"{tag}{country_tag}"]
                name = f"{file_prefix}-{term}-{cfg['date']}-{scheme_filter}{tag}{country_tag}-bins-{n}"
                out[name] = make_pool(cfg, name, priority, {currency}, list(bins_tuple), p_set,
                                      scheme_filter, type_selectors, weighted, country_label)
    return out


def process_backup_rows(cfg, rows, scheme_filter):
    """Catch-all pools (bin == 'Other'). Returns {name: pool}."""
    out = {}
    if not rows:
        return out
    rpgt = rows[0]["rpgt"]
    if rpgt not in RPGT_MAP:
        return out
    term, type_selectors = RPGT_MAP[rpgt]

    by_pc = defaultdict(lambda: {"raw": {}, "currencies": set()})
    for r in rows:
        if r["bin"] != "Other":
            continue
        key = (r["provider"], r["country"])
        by_pc[key]["raw"].update(r["connectors"])
        by_pc[key]["currencies"].add(r["currency"])

    config_to_pc = defaultdict(list)
    for (provider, country), data in by_pc.items():
        sig = tuple(sorted((k, round(v, 4)) for k, v in data["raw"].items()))
        config_to_pc[(sig, tuple(sorted(data["currencies"])))].append((provider, country))

    for (sig, currencies_tuple), pc_list in sorted(config_to_pc.items()):
        prov_to_countries = defaultdict(set)
        for p, c in pc_list:
            prov_to_countries[p].add(c)
        clabel_to_provs = defaultdict(list)
        for p, c_set in prov_to_countries.items():
            if {"USA", "Non-USA"}.issubset(c_set):
                clabel_to_provs["All"].append(p)
            elif "USA" in c_set:
                clabel_to_provs["USA"].append(p)
            elif "Non-USA" in c_set:
                clabel_to_provs["Non-USA"].append(p)
            else:
                clabel_to_provs["All"].append(p)

        for country_label, providers in sorted(clabel_to_provs.items()):
            for p_set in get_combinable_provider_sets(providers):
                orig_country = "USA" if country_label == "All" and "USA" in prov_to_countries[p_set[0]] else country_label
                if orig_country == "All":
                    orig_country = None
                data = by_pc[(p_set[0], orig_country)]
                raw, currencies = data["raw"], data["currencies"]
                weighted = normalize_weights(raw, len(currencies) * 1000)
                is_apgp = all(p in ("APPLEPAY", "GOOGLEPAY") for p in p_set)
                priority = get_priority(term, is_apgp, False, cfg.get("extra_priority", 200000))
                if set(p_set) == {"APPLEPAY", "GOOGLEPAY"}:
                    tag = "-apgp"
                elif set(p_set) == {"APPLEPAY", "GOOGLEPAY", "non_gp_ap"}:
                    tag = ""
                else:
                    tag = PROVIDER_TAG[p_set[0]]
                country_tag = "-us" if country_label == "USA" else ("-nonus" if country_label == "Non-USA" else "")
                file_prefix = f"rr-{cfg['prefix']}" if term.endswith("-ren") else cfg["prefix"]
                name = f"{file_prefix}-{term}-{cfg['date']}-{scheme_filter}{tag}{country_tag}"
                out[name] = make_pool(cfg, name, priority, currencies, None, p_set,
                                      scheme_filter, type_selectors, weighted, country_label)
    return out


def emit_pool_generic(cfg, pools):
    """pool-generic: every connector seen across `pools`, weight 100, priority 1."""
    skip = GENERIC_SKIP.get(cfg["brand"], set())
    all_connectors = set()
    for name, data in pools.items():
        if data["metadata"]["name"] in skip:
            continue
        for c in data["spec"]["connectors"]:
            all_connectors.add(c["connectorId"])
    return {
        "kind": "ConnectorPool",
        "metadata": {
            "projectId": cfg["project_id"], "name": "pool-generic", "uuid": str(uuid.uuid4()),
            "displayName": "", "description": "", "annotations": {}, "labels": None, "disabled": False,
        },
        "specVersion": "v1",
        "selector": {"priority": 1, "expressions": []},
        "spec": {"restriction": "fullCycle",
                 "connectors": [{"connectorId": cid, "priority": 0, "weighting": 100, "uses": 0}
                                for cid in sorted(all_connectors)]},
    }


def generate_configs(exports, brand_key, date, scheme="vi", mode="sales",
                     extra_priority_amount=200000, emit_generic=False):
    """Generate ConnectorPool configs from the export templates.

    exports: dict{(brand, rpgt): DataFrame} as returned by build_split_exports.
    brand_key: 'tav' | 'tab' | 'tdr'.
    date: YYMMDD tag embedded in pool names.
    mode: 'sales' (BIN-specific compressed only) or 'full' (also catch-all backup).
    Returns (pools, counts): pools = {name: pool_dict}; counts summary dict.
    """
    if brand_key not in BRANDS:
        raise ValueError(f"Unknown brand '{brand_key}' (expected one of {sorted(BRANDS)})")
    cfg = dict(BRANDS[brand_key])
    cfg["brand"] = brand_key
    cfg["date"] = str(date)
    cfg["extra_priority"] = int(extra_priority_amount)

    pools = {}
    per_rpgt = {}
    skipped = []
    for key, df in (exports.items() if hasattr(exports, "items") else []):
        rpgt_lbl = key[1] if isinstance(key, (tuple, list)) and len(key) > 1 else str(key)
        rows = rows_from_dataframe(df, cfg["name"])
        if not rows:
            continue
        if rows[0]["rpgt"] not in RPGT_MAP:
            skipped.append(rows[0]["rpgt"])
            continue
        n_before = len(pools)
        pools.update(process_compressed_rows(cfg, rows, scheme))
        if mode == "full":
            pools.update(process_backup_rows(cfg, rows, scheme))
        per_rpgt[str(rpgt_lbl)] = per_rpgt.get(str(rpgt_lbl), 0) + (len(pools) - n_before)

    generic = 0
    if emit_generic and mode == "full":
        pools["pool-generic"] = emit_pool_generic(cfg, pools)
        generic = 1

    counts = {
        "total": len(pools),
        "per_rpgt": per_rpgt,
        "generic": generic,
        "skipped_rpgts": sorted(set(skipped)),
        "pool_dir": cfg["pool_dir"],
    }
    return pools, counts
