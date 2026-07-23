-- attempts_success.sql
-- Fetches historical initial-attempt success/failure data used to estimate
-- gateway success (auth) rates for the routing optimiser.
--
-- Template variables (substituted by the app before running):
--   {START_DATE}     YYYY-MM-DD, inclusive lower bound on transaction date
--   {END_DATE}       YYYY-MM-DD, exclusive upper bound on transaction date
--   {COMPANY}        e.g. 'TotalAV'
--   {CARD_SCHEME}    'visa' or 'mastercard'
--   {BIN_PREFIX}     leading BIN digit(s) - '4' for visa, '5' for mastercard
--   {GATEWAY_FIDS}   parenthesised, quoted list, e.g. ('adyen-usd-tav','...')

WITH chive_dedup AS (
  SELECT
    connector_transaction_id_0,
    bank_country_0
  FROM `chive-data-1985.ws_39d_9B2SQQ1f1l.TRAN47f28cde`
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY connector_transaction_id_0
    ORDER BY CASE WHEN LENGTH(bank_country_0) > 1 THEN 1 ELSE 0 END DESC
  ) = 1
),

base_transactions AS (
  SELECT
    a.date,
    a.accountType,
    a.paymentMethod,
    CASE WHEN a.paymentMethodProvider NOT IN ('GOOGLEPAY', 'APPLEPAY')
         THEN 'non_gp_ap' ELSE a.paymentMethodProvider END AS paymentMethodProvider,
    a.transactionType,
    a.paymentType,
    a.amount,
    a.productGroupType,
    a.paymentProcessor,
    a.gatewayPaymentProcessor,
    a.currency,
    a.wasSuccess,
    a.fcpNumber,
    a.attemptCount,
    a.attemptNumber,
    a.bankName AS original_bankName,
    a.gatewayFid,
    c.Company AS company_name,
    chive.bank_country_0
  FROM `fortifi-1985.orgss1618qwk5g.reporting_finance_transactions_*` AS a
  LEFT JOIN `sapient-tangent-172609.Mapping.companyFidtoCompany` AS c
    ON a.companyFid = c.companyFid
  LEFT JOIN chive_dedup AS chive
    ON chive.connector_transaction_id_0 = a.gatewayTransactionId
  WHERE _table_suffix IN ('2025_q1','2025_q2','2025_q3','2025_q4','2026_q1','2026_q2','2026_q3','2026_q4')
    AND a.companyFid != 'FID:COMP:1543328902:DSeIJLO98yd5'
    AND a.gatewayPaymentProcessor NOT IN ('play_store','accountbalance','kount')
    AND a.accountType != 'unknown'
    AND NOT (
      a.responseCode = '20' AND
      a.responseText = 'FRAUD' AND
      a.gatewayPaymentProcessor = 'adyen'
    )
),

enriched_logic AS (
  SELECT
    date,
    accountType,
    company_name AS Company,
    paymentMethodProvider,

    CASE
      WHEN transactionType = 'auth' AND paymentType = 'order' AND amount < 12 AND paymentMethod != 'paypal' THEN 'Monthly Intiial'
      WHEN transactionType = 'auth' AND paymentType = 'order' AND amount > 12 AND amount < 37 AND amount != 23 AND paymentMethod != 'paypal' THEN 'Annual Sub Sale'
      WHEN transactionType = 'auth' AND paymentType = 'order' AND amount = 39 AND paymentMethod != 'paypal' THEN 'Upgrade'
      WHEN transactionType = 'capture' AND paymentType = 'order' AND productGroupType = '2' AND paymentMethod != 'paypal' THEN 'Addon Sale'
      WHEN transactionType = 'captureauth' AND paymentType = 'order' AND amount < 12 AND paymentMethod = 'paypal' AND paymentProcessor = 'paypal' THEN 'Monthly Initial'
      WHEN transactionType = 'auth' AND paymentType = 'order' AND amount < 12 AND paymentMethod = 'paypal' AND paymentProcessor = 'chargehive' THEN 'Monthly Initial'
      WHEN transactionType = 'capture' AND paymentType = 'order' AND amount < 12 AND paymentMethod = 'paypal' AND productGroupType = '1' THEN 'Monthly Initial'
      WHEN transactionType = 'captureauth' AND paymentType = 'order' AND amount > 12 AND amount < 37 AND productGroupType != '2' AND paymentMethod = 'paypal' AND paymentProcessor = 'paypal' THEN 'Annual Sub Sale'
      WHEN transactionType = 'auth' AND paymentType = 'order' AND amount > 12 AND amount < 37 AND productGroupType != '2' AND paymentMethod = 'paypal' AND paymentProcessor = 'chargehive' THEN 'Annual Sub Sale'
      WHEN transactionType = 'capture' AND paymentType = 'order' AND amount > 11 AND paymentMethod = 'paypal' AND productGroupType = '1' THEN 'Annual Sub Sale'
      WHEN transactionType = 'captureauth' AND paymentType = 'order' AND amount = 39 AND paymentMethod = 'paypal' AND paymentProcessor = 'paypal' THEN 'Upgrade'
      WHEN transactionType = 'auth' AND paymentType = 'order' AND amount = 39 AND paymentMethod = 'paypal' AND paymentProcessor = 'chargehive' THEN 'Upgrade'
      WHEN transactionType = 'captureauth' AND paymentType = 'order' AND productGroupType = '2' AND paymentMethod = 'paypal' THEN 'Addon Sale'
      WHEN transactionType = 'capture' AND paymentType = 'invoice' AND productGroupType != '2' AND amount < 19 THEN 'Monthly Renewal'
      WHEN transactionType = 'capture' AND paymentType = 'invoice' AND productGroupType != '2' AND amount != 49 AND amount > 18 THEN 'Annual Sub Renewal'
      WHEN transactionType = 'capture' AND paymentType = 'invoice' AND productGroupType != '2' AND gatewayPaymentProcessor = 'vindicia' AND amount = 49 THEN 'Annual Sub Renewal'
      WHEN transactionType = 'capture' AND paymentType = 'invoice' AND productGroupType != '2' AND gatewayPaymentProcessor != 'vindicia' AND (amount = 49 OR amount = 80) THEN 'P6M Renewals'
      WHEN transactionType = 'capture' AND paymentType = 'invoice' AND productGroupType = '2' THEN 'Addon Renewal'
      WHEN transactionType = 'refund' THEN 'Refund'
      WHEN transactionType = 'auth' AND paymentType = 'invoice' AND amount < 19 AND productGroupType != '2' THEN 'Monthly Renewal'
      WHEN transactionType = 'auth' AND paymentType = 'invoice' AND amount > 18 AND productGroupType != '2' THEN 'Annual Sub Renewal'
      WHEN transactionType = 'auth' AND paymentType = 'invoice' AND productGroupType = '2' THEN 'Addon Renewal'
      WHEN transactionType = 'captureauth' AND paymentMethod IN ('creditcard','debitcard','prepaidcard','google_pay','apple_pay') THEN 'Bin'
      WHEN transactionType = 'auth' AND paymentMethod = 'paypal' THEN 'Bin'
      ELSE 'Error'
    END AS rpgt,

    currency,
    amount,

    CASE
      WHEN paymentMethod != 'paypal' AND wasSuccess = true AND fcpNumber = 1 AND attemptCount = 1 THEN 1
      WHEN paymentMethod = 'paypal' AND wasSuccess = true AND fcpNumber = 1 AND attemptNumber < 3 THEN 1
      ELSE 0
    END AS initialSuccess,

    CASE
      WHEN paymentMethod != 'paypal' AND wasSuccess = false AND fcpNumber = 1 AND attemptCount = 1 THEN 1
      WHEN paymentMethod = 'paypal' AND wasSuccess = false AND fcpNumber = 1 AND attemptNumber < 3 THEN 1
      ELSE 0
    END AS initialFailure,

    CASE
      WHEN paymentMethod != 'paypal' AND fcpNumber = 1 AND attemptCount = 1 THEN 1
      WHEN paymentMethod = 'paypal' AND fcpNumber = 1 AND attemptNumber < 3 THEN 1
      ELSE 0
    END AS initialattempt,

    fcpNumber,

    CASE
      WHEN ARRAY_LENGTH(SPLIT(original_bankName, ' - ')) < 2 THEN ''
      ELSE TRIM(UPPER(REGEXP_REPLACE(SPLIT(original_bankName, ' - ')[SAFE_OFFSET(1)], r'[^\w\s]', '')))
    END AS bankName,
    LEFT(original_bankName, 6) AS bin,

    gatewayFid,

    CASE
      WHEN bank_country_0 IN ('US', 'USA') THEN 'USA'
      ELSE 'Non-USA'
    END AS country

  FROM base_transactions
)

SELECT
  date, accountType, Company, paymentMethodProvider, rpgt, currency, amount,
  initialSuccess, initialFailure, initialattempt, fcpNumber,
  bankName, bin, gatewayFid, country
FROM enriched_logic
WHERE rpgt NOT IN ('Bin', 'Refund')
  AND date >= '{START_DATE}'
  AND date < '{END_DATE}'
  AND Company = '{COMPANY}'
  AND (CASE WHEN accountType IN ('mastercard','maestro') THEN 'mastercard' ELSE accountType END) = '{CARD_SCHEME}'
  AND LEFT(CAST(bin AS string), 1) IN ('{BIN_PREFIX}')
  AND fcpNumber = 1
  AND gatewayFid IN {GATEWAY_FIDS}
