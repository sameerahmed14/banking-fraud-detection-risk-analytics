
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError(CSV)

con = duckdb.connect()

csv_path = CSV.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT *
FROM read_csv_auto('{csv_path}', header=true)
""")

results = con.execute("""
WITH balance_groups AS (
    SELECT
        "type" AS transaction_type,
        amount,
        isFraud,
        CASE
            WHEN oldbalanceOrg IS NULL
                THEN 'Unknown prior balance'
            WHEN oldbalanceOrg = 0
                THEN 'Zero prior balance'
            WHEN ABS(amount - oldbalanceOrg) <= 0.01
                THEN 'Entire prior balance'
            WHEN amount > oldbalanceOrg
                THEN 'Exceeds prior balance'
            ELSE 'Partial prior balance'
        END AS balance_pattern
    FROM transactions
    WHERE "type" IN ('TRANSFER', 'CASH_OUT')
)
SELECT
    transaction_type,
    balance_pattern,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_cases,
    ROUND(
        100.0 * SUM(isFraud) / COUNT(*), 4
    ) AS fraud_rate_pct,
    ROUND(
        SUM(CASE
            WHEN isFraud = 1 THEN amount
            ELSE 0
        END), 2
    ) AS fraud_transaction_value
FROM balance_groups
GROUP BY transaction_type, balance_pattern
ORDER BY transaction_type, fraud_cases DESC
""").df()

results.to_csv(
    OUTPUT / "fraud_balance_patterns.csv",
    index=False
)

print("\n=== FRAUD BY SENDER BALANCE BEHAVIOUR ===")
print(results.to_string(index=False))

print("\nReport saved:")
print(OUTPUT / "fraud_balance_patterns.csv")

con.close()
