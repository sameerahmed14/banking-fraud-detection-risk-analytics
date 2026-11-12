
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

if not CSV.exists():
    raise FileNotFoundError(CSV)

csv_path = CSV.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT *
FROM read_csv_auto('{csv_path}', header=true)
""")

# 1. Fraud risk by transaction amount
amount_risk = con.execute("""
WITH binned AS (
    SELECT
        "type" AS transaction_type,
        amount,
        isFraud,
        CASE
            WHEN amount < 1000 THEN 'Under 1k'
            WHEN amount < 10000 THEN '1k-10k'
            WHEN amount < 100000 THEN '10k-100k'
            ELSE '100k+'
        END AS amount_band,
        CASE
            WHEN amount < 1000 THEN 1
            WHEN amount < 10000 THEN 2
            WHEN amount < 100000 THEN 3
            ELSE 4
        END AS sort_order
    FROM transactions
    WHERE "type" IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    transaction_type,
    amount_band,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_cases,
    ROUND(
        100.0 * SUM(isFraud) / COUNT(*), 4
    ) AS fraud_rate_pct,
    ROUND(
        SUM(
            CASE WHEN isFraud = 1
            THEN amount ELSE 0 END
        ), 2
    ) AS fraud_transaction_value
FROM binned
GROUP BY transaction_type, amount_band, sort_order
ORDER BY transaction_type, sort_order
""").df()

amount_risk.to_csv(
    OUT / "fraud_by_amount_band.csv", index=False
)

print("\n=== FRAUD RISK BY AMOUNT ===")
print(amount_risk.to_string(index=False))

# 2. Fraud distribution over simulated days
daily_risk = con.execute("""
SELECT
    CAST(
        FLOOR((step - 1) / 24) + 1 AS INTEGER
    ) AS simulated_day,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_cases,
    ROUND(
        100.0 * SUM(isFraud) / COUNT(*), 4
    ) AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY 1
""").df()

daily_risk.to_csv(
    OUT / "fraud_by_day.csv", index=False
)

print("\n=== TOP 10 DAYS BY FRAUD COUNT ===")
print(
    daily_risk.sort_values(
        "fraud_cases", ascending=False
    ).head(10).to_string(index=False)
)

print("\nFiles saved to outputs/")
con.close()
