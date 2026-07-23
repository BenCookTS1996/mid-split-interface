-- processor_benchmark.sql
-- ============================================================================
-- DRAFT — VALIDATE ON BIGQUERY BEFORE RELYING ON IT.
-- ============================================================================
-- Cross-brand initial-attempt success rate per (gatewayPaymentProcessor, currency),
-- pooled across ALL brands/companies. Used ONLY as a fallback prior for the routing
-- optimiser's untested gatewayFids: when an untested gateway's processor has NO
-- same-brand sibling with data, we borrow this processor-level, cross-brand rate
-- (layer 2 of the sibling prior) instead of the bank×currency average.
--
-- It deliberately DROPS the two brand-scoping filters that attempts_success.sql
-- applies (Company = '{COMPANY}' and gatewayFid IN {GATEWAY_FIDS}) so it sees every
-- brand's traffic on each processor. It keeps the same date window, card scheme and
-- BIN-prefix filters, and the SAME initial-attempt success definition, so the rate
-- is comparable to the main query's.
--
-- Template variables (reuses the attempts_success.sql params):
--   {START_DATE} {END_DATE} {CARD_SCHEME} {BIN_PREFIX}
--
-- Returns one row per (processor, currency): successes, attempts.
-- ============================================================================

WITH base_transactions AS (
  SELECT
    a.date,
    a.gatewayPaymentProcessor,
    a.currency,
    a.paymentMethod,
    a.accountType,
    a.wasSuccess,
    a.fcpNumber,
    a.attemptCount,
    a.attemptNumber,
    LEFT(a.bankName, 6) AS bin
  FROM `fortifi-1985.orgss1618qwk5g.reporting_finance_transactions_*` AS a
  WHERE _table_suffix IN ('2025_q1','2025_q2','2025_q3','2025_q4','2026_q1','2026_q2','2026_q3','2026_q4')
    AND a.companyFid != 'FID:COMP:1543328902:DSeIJLO98yd5'
    AND a.gatewayPaymentProcessor NOT IN ('play_store','accountbalance','kount')
    AND a.accountType != 'unknown'
    AND NOT (a.responseCode = '20' AND a.responseText = 'FRAUD' AND a.gatewayPaymentProcessor = 'adyen')
),

scored AS (
  SELECT
    gatewayPaymentProcessor AS processor,
    currency,
    date,
    accountType,
    bin,
    -- SAME initial-attempt success / attempt definition as attempts_success.sql.
    CASE
      WHEN paymentMethod != 'paypal' AND wasSuccess = true  AND fcpNumber = 1 AND attemptCount = 1  THEN 1
      WHEN paymentMethod  = 'paypal' AND wasSuccess = true  AND fcpNumber = 1 AND attemptNumber < 3 THEN 1
      ELSE 0
    END AS initialSuccess,
    CASE
      WHEN paymentMethod != 'paypal' AND fcpNumber = 1 AND attemptCount = 1  THEN 1
      WHEN paymentMethod  = 'paypal' AND fcpNumber = 1 AND attemptNumber < 3 THEN 1
      ELSE 0
    END AS initialattempt
  FROM base_transactions
)

SELECT
  processor,
  currency,
  SUM(initialSuccess) AS successes,
  SUM(initialattempt) AS attempts
FROM scored
WHERE date >= '{START_DATE}'
  AND date < '{END_DATE}'
  AND (CASE WHEN accountType IN ('mastercard','maestro') THEN 'mastercard' ELSE accountType END) = '{CARD_SCHEME}'
  AND LEFT(CAST(bin AS string), 1) IN ('{BIN_PREFIX}')
GROUP BY processor, currency
