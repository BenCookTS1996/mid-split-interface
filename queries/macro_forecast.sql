WITH last_month_actuals AS (
    SELECT
        (CASE
            WHEN (ptran.rebillNumber = 0 AND ptran.termIso IN ("OT", "P26D", "P2M", "P3M","P19D", "P27D", "P0Y", "P7D", "P27D", "P7D", "P30D", "P1M") AND ptran.action = 'setup' AND ptran.productGroupType = '1') THEN 'Monthly Initial'
            WHEN (ptran.rebillNumber <> 0 AND ptran.termIso IN ("OT", "P26D", "P2M", "P3M","P19D", "P27D", "P0Y", "P7D", "P27D", "P7D", "P30D", "P1M") AND ptran.action = 'renew' AND ptran.productGroupType = '1') THEN 'Monthly Renewal'
            WHEN (ptran.action = 'setup' AND ptran.productGroupType = '2') THEN 'Addon Sale'
            WHEN (ptran.action = 'renew' AND ptran.productGroupType = '2') THEN 'Addon Renewal' 
            WHEN (ptran.productGroupType = '1' AND ptran.termIso IN ('P1Y','P12M', 'P2Y','P5Y','P24M') AND ptran.action = 'setup') THEN 'Annual Sub Sale'
            WHEN (ptran.productGroupType = '1' AND ptran.termIso IN ('P1Y','P12M', 'P2Y','P5Y','P24M') AND ptran.action = 'renew') THEN 'Annual Sub Renewal'
            WHEN ptran.termIso = 'P6M' THEN 'P6M Renewals'
            WHEN ptran.termIso <> 'P6M' AND ptran.action = 'upgraded' THEN 'Upgrades'
            ELSE 'Other' 
        END) AS rpgt,
        COUNT(DISTINCT ptran.gatewayTransactionId) AS trxCount,
        COUNT(DISTINCT ptran.purchaseFid) AS pfidCount
    FROM `sapient-tangent-172609.fortifi_views.reporting_finance_purchase_transactions_last_3_months` AS ptran 
    WHERE ptran.usdAmountPaid > 0 
      AND ptran.transactionType IN ('capture','captureauth') 
      AND ptran.accountType NOT LIKE '%play_store%' 
      AND ptran.accountType NOT LIKE '%account_balance%'
      
      -- DYNAMIC DATE FILTER: Always looks at the last fully completed month
      AND CAST(ptran.date AS DATE) >= DATE_TRUNC(DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH), MONTH)
      AND CAST(ptran.date AS DATE) < DATE_TRUNC(CURRENT_DATE(), MONTH)
      
    GROUP BY 1
)

SELECT 
    DATE_TRUNC(CAST(forecast.period AS DATE), MONTH) AS period,
    forecast.units as raw_units,
    forecast.rpgt,
    SAFE_DIVIDE(actuals.trxCount, actuals.pfidCount) AS pfid_per_trx_ratio,
    forecast.units * SAFE_DIVIDE(actuals.trxCount, actuals.pfidCount) as fcast_units

FROM `sapient-tangent-172609.Risk_Data_new.fpa_marco_rpgt_units_forecast` AS forecast
LEFT JOIN last_month_actuals AS actuals
    ON forecast.rpgt = actuals.rpgt