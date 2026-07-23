"""
Shared schema and column contracts for the routing optimiser.

Everything downstream (engines, k-means, config generator, UI) agrees on the
column names defined here so the pieces stay swappable. This is the single
place to change a column name if your upstream data changes.
"""
from __future__ import annotations

# --- Decision granularity ---------------------------------------------------
# Routing decisions are made per CELL. A cell is one combination of:
#   RPGT (transaction type) x Currency x Bank
# The optimiser decides, for each cell, what fraction of that cell's forecast
# volume to send to each eligible gateway/MID.
CELL_KEYS = ["rpgt", "currency", "bank"]

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

# Map the success-data scenario names onto the RPGT names used in the templates
# and the VAMP pipeline. Extend this as your taxonomy grows.
SCENARIO_TO_RPGT = {
    "Monthly Sale": "Monthly Initial",
    "Annual Sale": "Annual Sub Sale",
    "Addon Sale": "Addon Sale",
    "Monthly Renewal": "Monthly Renewal",
    "Annual Renewal": "Annual Sub Renewal",
    "Addon Renewal": "Addon Renewal",
    "Upgrade": "Upgrades",
    "P6M Renewal": "P6M Renewals",
}


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
