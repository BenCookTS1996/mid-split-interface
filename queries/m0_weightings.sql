-- Projected transaction counts for the CURRENT month for the selected card scheme, split by
-- risk-defined subscription product type (RPGT). Used to auto-fill "2 · M0 Transaction
-- Weightings" on tab 1 and Validate Split. `{company}` and `{CARD_SCHEME}` ('visa' or
-- 'mastercard') are substituted by sql_runner from the app's Company / Card Scheme selectors.
-- Mastercard collapses maestro (mirrors attempts_success.sql).
-- Projection: run-rate of completed days this month, scaled to the full month (>=14
-- completed days -> month-to-date run-rate; <14 -> trailing-14-day run-rate).
SELECT
    (case
          -- my definition of subscription_product type mapping (may change)
            when(riskdata2025.rebillNumber = 0 and riskdata2025.termIso in ("OT", "P26D", "P2M", "P3M","P19D", "P27D", "P0Y", "P7D", "P27D", "P7D", "P30D", "P1M") and riskdata2025.action = 'setup' and riskdata2025.productGroupType = '1')
            or (riskdata2025.rebillNumber = 0 and riskdata2025.termIso in ("OT", "P26D", "P2M", "P3M","P19D", "P27D", "P0Y", "P7D", "P27D", "P7D", "P30D", "P1M") and riskdata2025.recordType = 'refund' and riskdata2025.productGroupType = '1')
            then 'Monthly Initial'
      when (riskdata2025.rebillNumber <> 0 and riskdata2025.termIso in ("OT", "P26D", "P2M", "P3M","P19D", "P27D", "P0Y", "P7D", "P27D", "P7D", "P30D", "P1M") and riskdata2025.action = 'renew' and riskdata2025.productGroupType = '1')
      or (riskdata2025.rebillNumber <> 0 and riskdata2025.termIso in ("OT", "P26D", "P2M", "P3M","P19D", "P27D", "P0Y", "P7D", "P27D", "P7D", "P30D", "P1M") and riskdata2025.recordType = 'refund' and riskdata2025.productGroupType = '1')
      then 'Monthly Renewal'
      when (riskdata2025.action = 'setup' and riskdata2025.productGroupType = '2')
      or (riskdata2025.rebillNumber=0 and riskdata2025.productGroupType = '2' and riskdata2025.recordType="refund")
      then 'Addon Sale'
      when (riskdata2025.action = 'renew' and riskdata2025.productGroupType = '2')
      or (riskdata2025.rebillNumber<>0 and riskdata2025.productGroupType = '2' and riskdata2025.recordType="refund")
      then 'Addon Renewal'
      when (riskdata2025.productGroupType = '1'and riskdata2025.termIso in ('P1Y','P12M', 'P2Y','P5Y','P24M') and riskdata2025.action = 'setup')
      or (riskdata2025.productGroupType = '1'and riskdata2025.termIso in ('P1Y','P12M', 'P2Y','P5Y','P24M') and riskdata2025.rebillNumber = 0 and riskdata2025.recordType = "refund")
      then 'Annual Sub Sale'
      when (riskdata2025.productGroupType = '1'and riskdata2025.termIso in ('P1Y','P12M', 'P2Y','P5Y','P24M') and riskdata2025.action = 'renew')
      or (riskdata2025.productGroupType = '1'and riskdata2025.termIso in ('P1Y','P12M', 'P2Y','P5Y','P24M') and riskdata2025.rebillNumber <> 0)
      then 'Annual Sub Renewal'
      when riskdata2025.termIso = 'P6M' then 'P6M Renewals'
      when riskdata2025.termIso <> 'P6M' and riskdata2025.action = 'upgraded' then 'Upgrades'
      else 'Other' end
) AS riskdata2025_risk_defined_subscription_product_type,
    -- UPDATED: Projections based on the Last Completed Day (Yesterday)
    CASE
        -- CONDITION 1: Last completed day is >= 14
        WHEN EXTRACT(DAY FROM DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)) >= 14 THEN
            ( COUNT(DISTINCT CASE
                WHEN (( (case
                        when riskdata2025.cardTypeN like "%nyce%" then "nyce"
                        when riskdata2025.cardTypeN like "%pulse%" then "pulse"
                        when riskdata2025.cardTypeN like "%accel%" then "accel"
                        when riskdata2025.cardTypeN like "%star%" then "star"
                        when riskdata2025.cardTypeN like '%mc_google_pay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mc_applepay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro_usa%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mastercard%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%visa%' then 'visa'
                        else riskdata2025.accountType end) = '{CARD_SCHEME}'  ))
                AND (( riskdata2025.recordType = "transaction"  ))
                AND (NOT COALESCE(( left((case
                        when riskdata2025.cardTypeN like "%nyce%" then "nyce"
                        when riskdata2025.cardTypeN like "%pulse%" then "pulse"
                        when riskdata2025.cardTypeN like "%accel%" then "accel"
                        when riskdata2025.cardTypeN like "%star%" then "star"
                        when riskdata2025.cardTypeN like '%mc_google_pay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mc_applepay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro_usa%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mastercard%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%visa%' then 'visa'
                        else riskdata2025.accountType end),6) = "paypal"  ), FALSE))

                -- Ensure it is for the month/year of the last completed day
                AND EXTRACT(MONTH FROM cast(riskdata2025.recordDate as timestamp)) = EXTRACT(MONTH FROM DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))
                AND EXTRACT(YEAR FROM cast(riskdata2025.recordDate as timestamp)) = EXTRACT(YEAR FROM DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))

                -- EXCLUDE today's partial data from the count
                AND cast(riskdata2025.recordDate as timestamp) < TIMESTAMP(CURRENT_DATE())

                THEN riskdata2025.gatewayTransactionId
                ELSE NULL END)
            / EXTRACT(DAY FROM DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)) ) * EXTRACT(DAY FROM LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)))
        -- CONDITION 2: Last completed day is < 14
        ELSE
            ( COUNT(DISTINCT CASE
                WHEN (( (case
                        when riskdata2025.cardTypeN like "%nyce%" then "nyce"
                        when riskdata2025.cardTypeN like "%pulse%" then "pulse"
                        when riskdata2025.cardTypeN like "%accel%" then "accel"
                        when riskdata2025.cardTypeN like "%star%" then "star"
                        when riskdata2025.cardTypeN like '%mc_google_pay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mc_applepay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro_usa%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mastercard%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%visa%' then 'visa'
                        else riskdata2025.accountType end) = '{CARD_SCHEME}'  ))
                AND (( riskdata2025.recordType = "transaction"  ))
                AND (NOT COALESCE(( left((case
                        when riskdata2025.cardTypeN like "%nyce%" then "nyce"
                        when riskdata2025.cardTypeN like "%pulse%" then "pulse"
                        when riskdata2025.cardTypeN like "%accel%" then "accel"
                        when riskdata2025.cardTypeN like "%star%" then "star"
                        when riskdata2025.cardTypeN like '%mc_google_pay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mc_applepay%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro_usa%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%mastercard%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%maestro%' then 'mastercard'
                        when riskdata2025.cardTypeN like '%visa%' then 'visa'
                        else riskdata2025.accountType end),6) = "paypal"  ), FALSE))

                -- Isolate to exactly the last 14 completed days (excluding today)
                AND cast(riskdata2025.recordDate as timestamp) >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY))
                AND cast(riskdata2025.recordDate as timestamp) < TIMESTAMP(CURRENT_DATE())

                THEN riskdata2025.gatewayTransactionId
                ELSE NULL END)
            / 14 ) * EXTRACT(DAY FROM LAST_DAY(DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)))
    END AS riskdata2025_scheme_trx_count
FROM `sapient-tangent-172609.Risk_Data.risk-data-bc-test`  AS riskdata2025
WHERE ((( cast(riskdata2025.recordDate as timestamp)  ) >= ((TIMESTAMP_ADD(TIMESTAMP_TRUNC(TIMESTAMP(FORMAT_TIMESTAMP('%F %H:%M:%E*S', CURRENT_TIMESTAMP())), DAY), INTERVAL -31 DAY))) AND ( cast(riskdata2025.recordDate as timestamp)  ) < ((TIMESTAMP_ADD(TIMESTAMP_ADD(TIMESTAMP_TRUNC(TIMESTAMP(FORMAT_TIMESTAMP('%F %H:%M:%E*S', CURRENT_TIMESTAMP())), DAY), INTERVAL -31 DAY), INTERVAL 31 DAY))))) AND (case
            when riskdata2025.Company like "%TotalAV%" then "TotalAV"
            when riskdata2025.Company like "%Total Adblock%" then "Total Adblock"
            when riskdata2025.Company like "%Total Drive%" then "Total Drive"
            when riskdata2025.Company like "%Total VPN%" then "Total VPN"
            when riskdata2025.Company like "%Total Cleaner%" then "Total Cleaner"
          else "Other Brands" end) = '{company}'
GROUP BY
    1
ORDER BY
    2 DESC
