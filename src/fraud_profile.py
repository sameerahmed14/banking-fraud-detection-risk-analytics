
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

con = duckdb.connect()
path = CSV.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT *
FROM read_csv_auto('{path}', header=true)
""")

# Investigation 1: Fraud risk by transaction type
by_type = con.execute("""
SELECT
    "type" AS transaction_type,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END)
        AS fraud_cases,
    ROUND(
        100.0 * SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END)
        / COUNT(*), 4
    ) AS fraud_rate_pct,
    ROUND(
        SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2
    ) AS fraud_transaction_value
FROM transactions
GROUP BY "type"
ORDER BY fraud_rate_pct DESC
""").df()

print("\n=== FRAUD RISK BY TRANSACTION TYPE ===")
print(by_type.to_string(index=False))

by_type.to_csv(OUT / "fraud_by_type.csv", index=False)

# Investigation 2: Existing fraud flag vs actual label
flags = con.execute("""
SELECT
    isFraud,
    isFlaggedFraud,
    COUNT(*) AS transactions
FROM transactions
GROUP BY isFraud, isFlaggedFraud
ORDER BY isFraud DESC, isFlaggedFraud DESC
""").df()

print("\n=== EXISTING FRAUD FLAG VALIDATION ===")
print(flags.to_string(index=False))

flags.to_csv(OUT / "fraud_flag_validation.csv", index=False)

print("\nReports saved in outputs/")
con.close()
