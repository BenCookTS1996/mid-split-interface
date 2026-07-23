WITH rawQ AS (
  SELECT
      FORMAT_TIMESTAMP('%Y-%m', CAST(riskdata2025.recordDate AS TIMESTAMP)) AS recordMonth,
      EXTRACT(DAY FROM DATE(CAST(riskdata2025.recordDate AS TIMESTAMP))) AS recordDOM,
      COUNT(DISTINCT CASE WHEN (
          CASE
            WHEN riskdata2025.cardTypeN LIKE '%nyce%' THEN 'nyce'
            WHEN riskdata2025.cardTypeN LIKE '%pulse%' THEN 'pulse'
            WHEN riskdata2025.cardTypeN LIKE '%accel%' THEN 'accel'
            WHEN riskdata2025.cardTypeN LIKE '%star%' THEN 'star'
            WHEN riskdata2025.cardTypeN LIKE '%mc_google_pay%' THEN 'mastercard'
            WHEN riskdata2025.cardTypeN LIKE '%mc_applepay%' THEN 'mastercard'
            WHEN riskdata2025.cardTypeN LIKE '%maestro_usa%' THEN 'mastercard'
            ELSE riskdata2025.accountType
          END = 'visa'
      ) AND (riskdata2025.recordType = 'transaction') 
        AND NOT COALESCE(LEFT((
          CASE
            WHEN riskdata2025.cardTypeN LIKE '%nyce%' THEN 'nyce'
            WHEN riskdata2025.cardTypeN LIKE '%pulse%' THEN 'pulse'
            WHEN riskdata2025.cardTypeN LIKE '%accel%' THEN 'accel'
            WHEN riskdata2025.cardTypeN LIKE '%star%' THEN 'star'
            WHEN riskdata2025.cardTypeN LIKE '%mc_google_pay%' THEN 'mastercard'
            WHEN riskdata2025.cardTypeN LIKE '%mc_applepay%' THEN 'mastercard'
            WHEN riskdata2025.cardTypeN LIKE '%maestro_usa%' THEN 'mastercard'
            ELSE riskdata2025.accountType
          END), 6) = 'paypal', FALSE) 
      THEN riskdata2025.gatewayTransactionId ELSE NULL END) AS visaTrxCount

  FROM `sapient-tangent-172609.Risk_Data.risk-data-bc-test` AS riskdata2025

  -- 🟢 DYNAMIC WHERE CLAUSE: Exactly 12 months backward from Month 0
  WHERE CAST(riskdata2025.recordDate AS DATE) >= DATE_SUB(CAST('{MONTH_0_START_DATE}' AS DATE), INTERVAL 12 MONTH)
    AND CAST(riskdata2025.recordDate AS DATE) < CAST('{MONTH_0_START_DATE}' AS DATE)
    
    AND (
      CASE
        WHEN (riskdata2025.rebillNumber = 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M','P19D', 'P27D', 'P0Y', 'P7D', 'P27D', 'P7D', 'P30D', 'P1M') AND riskdata2025.action = 'setup' AND riskdata2025.productGroupType = '1')
          OR (riskdata2025.rebillNumber = 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M','P19D', 'P27D', 'P0Y', 'P7D', 'P27D', 'P7D', 'P30D', 'P1M') AND riskdata2025.recordType = 'refund' AND riskdata2025.productGroupType = '1') THEN 'Monthly Initial'
        WHEN (riskdata2025.rebillNumber <> 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M','P19D', 'P27D', 'P0Y', 'P7D', 'P27D', 'P7D', 'P30D', 'P1M') AND riskdata2025.action = 'renew' AND riskdata2025.productGroupType = '1')
          OR (riskdata2025.rebillNumber <> 0 AND riskdata2025.termIso IN ('OT', 'P26D', 'P2M', 'P3M','P19D', 'P27D', 'P0Y', 'P7D', 'P27D', 'P7D', 'P30D', 'P1M') AND riskdata2025.recordType = 'refund' AND riskdata2025.productGroupType = '1') THEN 'Monthly Renewal'
        WHEN (riskdata2025.action = 'setup' AND riskdata2025.productGroupType = '2')
          OR (riskdata2025.rebillNumber = 0 AND riskdata2025.productGroupType = '2' AND riskdata2025.recordType = 'refund') THEN 'Addon Sale'
        WHEN (riskdata2025.action = 'renew' AND riskdata2025.productGroupType = '2')
          OR (riskdata2025.rebillNumber <> 0 AND riskdata2025.productGroupType = '2' AND riskdata2025.recordType = 'refund') THEN 'Addon Renewal'
        WHEN (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y','P12M', 'P2Y','P5Y','P24M') AND riskdata2025.action = 'setup')
          OR (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y','P12M', 'P2Y','P5Y','P24M') AND riskdata2025.rebillNumber = 0 AND riskdata2025.recordType = 'refund') THEN 'Annual Sub Sale'
        WHEN (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y','P12M', 'P2Y','P5Y','P24M') AND riskdata2025.action = 'renew')
          OR (riskdata2025.productGroupType = '1' AND riskdata2025.termIso IN ('P1Y','P12M', 'P2Y','P5Y','P24M') AND riskdata2025.rebillNumber <> 0) THEN 'Annual Sub Renewal'
        WHEN riskdata2025.termIso = 'P6M' THEN 'P6M Renewals'
        WHEN riskdata2025.termIso <> 'P6M' AND riskdata2025.action = 'upgraded' THEN 'Upgrades'
        ELSE 'Other' 
      END
    ) = 'Monthly Renewal'
    AND riskdata2025.Company IN ('TotalAV','Total Adblock')
  GROUP BY ALL
  ORDER BY 1, 2
),

ShareData AS (
  SELECT
    rawQ.recordMonth,
    rawQ.recordDOM,
    rawQ.visaTrxCount,
    SUM(rawQ.visaTrxCount) OVER(PARTITION BY rawQ.recordMonth) AS monthTotal,
    rawQ.visaTrxCount / (SUM(rawQ.visaTrxCount) OVER(PARTITION BY rawQ.recordMonth)) AS monthShare,
    CASE
      WHEN rawQ.recordDOM = 1 THEN 'HIGH'
      WHEN rawQ.recordDOM < 29 AND rawQ.recordDOM > 1 THEN 'MID'
      WHEN rawQ.recordDOM >= 29 THEN 'LOW'
      ELSE NULL 
    END AS band
  FROM rawQ
  GROUP BY ALL
)

SELECT
  recordMonth AS calendarMonth,
  recordDOM,
  AVG(monthShare) OVER(PARTITION BY recordMonth, band) AS avg_daily_share
FROM ShareData
ORDER BY recordMonth, recordDOM