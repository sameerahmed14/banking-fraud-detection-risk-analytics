
from pathlib import Path
import json
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
OUTPUT = ROOT / "outputs" / "fraud_data_audit.json"

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

con = duckdb.connect()
csv_path = DATA.as_posix().replace("'", "''")

con.execute(f"""
    CREATE VIEW transactions AS
    SELECT * FROM read_csv_auto('{csv_path}', header=true)
""")

columns = [
    row[0]
    for row in con.execute("DESCRIBE transactions").fetchall()
]

total, fraud, flagged, invalid_amount, first_step, last_step = con.execute("""
    SELECT
        COUNT(*),
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN isFlaggedFraud = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN amount IS NULL OR amount < 0 THEN 1 ELSE 0 END),
        MIN(step),
        MAX(step)
    FROM transactions
""").fetchone()

fraud_rate = round(100 * fraud / total, 4)

by_type = con.execute("""
    SELECT
        "type",
        COUNT(*) AS transactions,
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_cases
    FROM transactions
    GROUP BY "type"
    ORDER BY transactions DESC
""").fetchall()

report = {
    "total_transactions": total,
    "fraud_transactions": fraud,
    "legitimate_transactions": total - fraud,
    "fraud_rate_pct": fraud_rate,
    "flagged_fraud_transactions": flagged,
    "invalid_amount_rows": invalid_amount,
    "first_step": first_step,
    "last_step": last_step,
    "columns": columns,
    "by_transaction_type": [
        {"type": t, "transactions": n, "fraud_cases": f}
        for t, n, f in by_type
    ]
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n=== BANKING FRAUD DATA AUDIT ===")
print(f"Total transactions: {total:,}")
print(f"Fraud transactions: {fraud:,}")
print(f"Legitimate transactions: {total - fraud:,}")
print(f"Fraud rate: {fraud_rate:.4f}%")
print(f"Flagged transactions: {flagged:,}")
print(f"Invalid amount rows: {invalid_amount:,}")
print(f"Time steps: {first_step} to {last_step}")
print(f"Columns: {len(columns)}")

print("\nTransaction types:")
for t, n, f in by_type:
    print(f"{t}: {n:,} transactions | {f:,} fraud")

print(f"\nAudit saved: {OUTPUT}")
con.close()
