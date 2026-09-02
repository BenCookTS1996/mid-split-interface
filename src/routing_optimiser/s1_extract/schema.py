"""
Shared schema and column contracts for the routing optimiser.

Everything downstream (engines, k-means, config generator, UI) agrees on the
column names defined here so the pieces stay swappable. This is the single
place to change a column name if your upstream data changes.
"""
from __future__ import annotations

# --- Decision granularity ---------------------------------------------------
# Routing decisions are made per PROFILE. A profile is one combination of:
#   RPGT (transaction type) x Currency x Bank
# The optimiser decides, for each profile, what fraction of that profile's forecast
# volume to send to each eligible gateway/MID.
COARSE_PROFILE_KEYS = ["rpgt", "currency", "bin"]

# --- Full profile key -------------------------------------------------------
# The k-means + config generator work at a finer grain (they also carry brand,
# country, BIN and payment-method-provider). We keep those columns flowing
# through so the compressed output matches your existing templates.
PROFILE_KEYS = ["brand", "rpgt", "country", "currency", "bin", "payment_method_provider"]

# --- Success / attempts data columns (from queries/attempts_success.sql) ----
SUCCESS_DATA_COLUMNS = {
    "date": "date",
    "company": "company",
    "payment_method": "paymentMethod",
    "account_type": "accountType",
    "processor": "gatewayPaymentProcessor",   # processor family, e.g. "adyen"
    "scenario": "rpgt",                       # SQL now emits the RPGT directly
    "currency": "currency",
    "amount": "amount",
    "initial_success": "initialSuccess",
    "success": "success",
    "initial_failure": "initialFailure",
    "failure": "failure",
    "initial_attempt": "initialattempt",
    "fcp_number": "FCPnumber",
    "bank_name": "bankName",
    "bin": "bin",
    "gateway_fid": "gatewayFid",              # the MID, e.g. "adyen-usd-tav"
    "country": "country",
    "grouping": "transactionGrouping",
}

# --- Template columns (mid_split_* / Compressed_Rules_* .xlsx) ---------------
# Non-gateway leading columns in the routing template. Everything after "Check"
# and before "DUP CHECK" is treated as a gateway/MID column.
TEMPLATE_META_COLUMNS = [
    "GO LIVE", "BIN GROUP", "Brand", "RPGT", "Currency", "BIN",
    "paymentMethodProvider", "STICKY", "Country", "Check",
]
TEMPLATE_TRAILING_COLUMNS = ["DUP CHECK"]

# Map legacy success-data scenario names onto the canonical RPGT names. NOTE: the current
# attempts_success.sql already emits canonical RPGT values, so most keys below now MISS and are
# restored by the `.fillna` in load_success_data — only the 'Monthly Intiial' typo alias still
# actively fires. Kept for the legacy `transactionScenario` shape and old cached parquet.
SCENARIO_TO_RPGT = {
    "Monthly Sale": "Monthly Initial",
    "Annual Sale": "Annual Sub Sale",
    "Addon Sale": "Addon Sale",
    "Monthly Renewal": "Monthly Renewal",
    "Annual Renewal": "Annual Sub Renewal",
    "Addon Renewal": "Addon Renewal",
    "Upgrade": "Upgrades",
    "P6M Renewal": "P6M Renewals",
    # Legacy alias: attempts_success.sql historically emitted a misspelled
    # 'Monthly Intiial'. The query is now fixed, but old cached parquet / re-loaded
    # outputs may still carry the typo, so canonicalise it here at the single loader
    # chokepoint (load_success_data) rather than relying on per-tab maps.
    "Monthly Intiial": "Monthly Initial",
}

# --- RPGT -> (term, ConnectorPool type selectors) ---------------------------
# SINGLE SOURCE OF TRUTH for the connector-pool selector map. Both the
# script-faithful generator (connector_pool_configs) and the older per-profile
# pooler (config_generator) import THIS map so the deployed selectors can't
# drift. 'Annual Sub Sale' carries skuType==SKU_TYPE_PRIMARY, matching
# 'Monthly Initial' (the other primary/initial-sale RPGT).
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
    "Annual Sub Renewal": ("asr", [
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


# [FN-221]
def gateway_columns(columns) -> list[str]:
    """Return the gateway/MID columns from a template header."""
    cols = list(columns)
    lowered_meta = {c.lower() for c in TEMPLATE_META_COLUMNS}
    lowered_trailing = {c.lower() for c in TEMPLATE_TRAILING_COLUMNS}
    out = []
    for c in cols:
        cl = str(c).strip().lower()
        if cl in lowered_meta or cl in lowered_trailing:
            continue
        if "dup check" in cl or cl == "check":
            continue
        out.append(c)
    return out
