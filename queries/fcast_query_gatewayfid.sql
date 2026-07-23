-- COPY of fcast_query.sql with riskdata2025.gatewayFid ADDED to the SELECT/grain.
-- Purpose: produce the actuals thermometer data at gatewayFid level so the tab-3
-- "actuals -> Forecast Post" charts can filter actuals to the SAME gatewayFid grain
-- as the forecast bars. The original fcast_query.sql is UNCHANGED (it is critical to
-- the vamp_pipeline); this is a separate query feeding a separate cache.
-- Only difference vs fcast_query.sql: the `riskdata2025.gatewayFid AS gatewayFid`
-- line below (GROUP BY ALL picks it up automatically).
SELECT
  DATE_DIFF(DATE_TRUNC(DATE_SUB('{MONTH_0_START_DATE}', INTERVAL 1 MONTH), MONTH), DATE_TRUNC(CAST(riskdata2025.recordDate AS DATE), MONTH), MONTH) AS period,

  -- 🟢 UPGRADE: Extrapolating out to 9+ months
  CASE WHEN DATE_DIFF(DATE(CAST(riskdata2025.recordDate AS TIMESTAMP)), DATE(riskdata2025.originDate), MONTH) < 0 THEN 0 WHEN DATE_DIFF(DATE(CAST(riskdata2025.recordDate AS TIMESTAMP)), DATE(riskdata2025.originDate), MONTH) >= 9 THEN 9 ELSE DATE_DIFF(DATE(CAST(riskdata2025.recordDate AS TIMESTAMP)), DATE(riskdata2025.originDate), MONTH) END AS time_to_event_months,

  CASE WHEN riskdata2025.paymentMethodProvider NOT IN ('GOOGLEPAY', 'APPLEPAY') THEN 'non_gp_ap' ELSE riskdata2025.paymentMethodProvider END AS paymentMethodProvider,
  riskdata2025.Company,
  -- 🟢 ADDED: gatewayFid so the actuals can be filtered to the forecast's gateway grain.
  riskdata2025.gatewayFid AS gatewayFid,
  CASE
    WHEN (riskdata2025.rebillNumber = 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M', 'P19D', 'P27D', 'P0Y', 'P7D', 'P30D', 'P1M') AND riskdata2025.action = 'setup' AND riskdata2025.productGroupType = '1') OR (riskdata2025.rebillNumber = 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M', 'P19D', 'P27D', 'P0Y', 'P7D', 'P30D', 'P1M') AND riskdata2025.recordType = 'refund' AND riskdata2025.productGroupType = '1') THEN 'Monthly Initial'
    WHEN (riskdata2025.rebillNumber <> 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M', 'P19D', 'P27D', 'P0Y', 'P7D', 'P30D', 'P1M') AND riskdata2025.action = 'renew' AND riskdata2025.productGroupType = '1') OR (riskdata2025.rebillNumber <> 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M', 'P19D', 'P27D', 'P0Y', 'P7D', 'P30D', 'P1M') AND riskdata2025.recordType = 'refund' AND riskdata2025.productGroupType = '1') THEN 'Monthly Renewal'
    WHEN (riskdata2025.action = 'setup' AND riskdata2025.productGroupType = '2') OR (riskdata2025.rebillNumber = 0 AND riskdata2025.productGroupType = '2' AND riskdata2025.recordType = 'refund') THEN 'Addon Sale'
    WHEN (riskdata2025.action = 'renew' AND riskdata2025.productGroupType = '2') OR (riskdata2025.rebillNumber <> 0 AND riskdata2025.productGroupType = '2' AND riskdata2025.recordType = 'refund') THEN 'Addon Renewal'
    WHEN (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y', 'P12M', 'P2Y', 'P5Y', 'P24M') AND riskdata2025.action = 'setup') OR (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y', 'P12M', 'P2Y', 'P5Y', 'P24M') AND riskdata2025.rebillNumber = 0 AND riskdata2025.recordType = 'refund') THEN 'Annual Sub Sale'
    WHEN (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y', 'P12M', 'P2Y', 'P5Y', 'P24M') AND riskdata2025.action = 'renew') OR (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y', 'P12M', 'P2Y', 'P5Y', 'P24M') AND riskdata2025.rebillNumber <> 0) THEN 'Annual Sub Renewal'
    WHEN riskdata2025.termIso = 'P6M' THEN 'P6M Renewals' WHEN riskdata2025.termIso <> 'P6M' AND riskdata2025.action = 'upgraded' THEN 'Upgrades' ELSE 'Other'
  END AS rpgt,
  riskdata2025.currency AS currency,
  COALESCE(riskdata2025.bin, 'unallocated') AS bin,
  CASE
  WHEN riskdata2025.Company != 'Total Drive' AND cast(chive.renewal_number_1 as string) in ('1','2','3') then 'sticky'
  WHEN riskdata2025.Company != 'Total Drive' AND cast(chive.renewal_number_1 as string) not in ('1','2','3') THEN 'non_sticky'
  ELSE 'non_sticky' END AS renewal_number,

  -- 🟢 UPGRADE: Added FCP tracking to curve math
  case when ftran.fcpNumber != 1 then '2+' else '1' end as fcpNumber,
  case when ftran.attemptNumber != 1 then '2+' else '1' end as attemptNumber,
  case when chive.bank_country_0 in ('US', 'USA') then 'USA' else 'Non-USA' end as country,

  -- 🟢 THE FIX: 30.4167-Day Actuarial Normalization tied strictly to the ORIGIN DATE
  (COUNT(DISTINCT CASE WHEN (CASE WHEN riskdata2025.cardTypeN LIKE '%nyce%' THEN 'nyce' WHEN riskdata2025.cardTypeN LIKE '%pulse%' THEN 'pulse' WHEN riskdata2025.cardTypeN LIKE '%accel%' THEN 'accel' WHEN riskdata2025.cardTypeN LIKE '%star%' THEN 'star' WHEN riskdata2025.cardTypeN LIKE '%mc_google_pay%' THEN 'mastercard' WHEN riskdata2025.cardTypeN LIKE '%mc_applepay%' THEN 'mastercard' WHEN riskdata2025.cardTypeN LIKE '%maestro_usa%' THEN 'mastercard' ELSE riskdata2025.accountType END) = 'visa' AND riskdata2025.isVampTransaction AND NOT COALESCE(LEFT((CASE WHEN riskdata2025.cardTypeN LIKE '%nyce%' THEN 'nyce' WHEN riskdata2025.cardTypeN LIKE '%pulse%' THEN 'pulse' WHEN riskdata2025.cardTypeN LIKE '%accel%' THEN 'accel' WHEN riskdata2025.cardTypeN LIKE '%star%' THEN 'star' WHEN riskdata2025.cardTypeN LIKE '%mc_google_pay%' THEN 'mastercard' WHEN riskdata2025.cardTypeN LIKE '%mc_applepay%' THEN 'mastercard' WHEN riskdata2025.cardTypeN LIKE '%maestro_usa%' THEN 'mastercard' ELSE riskdata2025.accountType END), 6) = 'paypal', FALSE) THEN CONCAT(CAST(riskdata2025.gatewayTransactionId AS STRING), '_', CAST(riskdata2025.recordType AS STRING)) ELSE NULL END) * 30.4167)
  / EXTRACT(DAY FROM LAST_DAY(MAX(DATE(riskdata2025.originDate)))) AS vamp_count

FROM `sapient-tangent-172609.Risk_Data.risk-data-bc-test` AS riskdata2025
LEFT JOIN `sapient-tangent-172609.Mapping.gatewayFidtoGateway` AS gateway_fidto_gateway ON riskdata2025.gatewayFid = gateway_fidto_gateway.gatewayFid

LEFT JOIN (select gatewayTransactionId, fcpNumber, attemptNumber from `sapient-tangent-172609.fortifi_views.reporting_finance_transactions_last_365_days` QUALIFY ROW_NUMBER() OVER(PARTITION BY gatewayTransactionId ORDER BY date ASC) = 1) as ftran ON CAST(ftran.gatewayTransactionId AS STRING) = CAST(riskdata2025.gatewayTransactionId AS STRING)
LEFT JOIN (select connector_transaction_id_0, bank_country_0, renewal_number_1 from `chive-data-1985.ws_39d_9B2SQQ1f1l.TRAN47f28cde`QUALIFY ROW_NUMBER() OVER(PARTITION BY connector_transaction_id_0 ORDER BY date ASC) = 1) AS chive ON CAST(chive.connector_transaction_id_0 AS STRING) = CAST(riskdata2025.gatewayTransactionId AS STRING)



-- 🟢 DYNAMIC WHERE CLAUSE: Exactly 9 months backward from Month 0
WHERE CAST(riskdata2025.recordDate AS DATE) >= DATE_SUB(CAST('{MONTH_0_START_DATE}' AS DATE), INTERVAL 9 MONTH)
  AND CAST(riskdata2025.recordDate AS DATE) < CAST('{MONTH_0_START_DATE}' AS DATE)
  AND riskdata2025.gatewayFid NOT LIKE '%paypal%'
GROUP BY ALL
