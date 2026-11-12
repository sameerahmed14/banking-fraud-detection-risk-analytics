
from pathlib import Path
import csv
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
OUTPUT = ROOT / "outputs/chronological_validation.csv"

if not DATA.exists():
    raise FileNotFoundError(DATA)

con = duckdb.connect()

path = DATA.as_posix().replace("'", "''")
con.execute(f"""
    CREATE VIEW transactions AS
    SELECT *
    FROM read_csv_auto('{path}', header=true)
""")

# Find the simulated-hour cutoff covering ~80% of transactions.
steps = con.execute("""
    SELECT step, COUNT(*) AS transactions
    FROM transactions
    GROUP BY step
    ORDER BY step
""").fetchall()

total = sum(count for _, count in steps)
target = total * 0.80
cumulative = 0
cutoff = None

for step, count in steps:
    cumulative += count
    cutoff = step
    if cumulative >= target:
        break

print("\n=== CHRONOLOGICAL SPLIT ===")
print(f"Total transactions: {total:,}")
print(f"Training period: step <= {cutoff}")
print(f"Testing period: step > {cutoff}")
print(f"Training share: {cumulative / total:.2%}")

# Compare two detectors in both time periods.
results = con.execute("""
WITH predictions AS (
    SELECT
        CASE WHEN step <= ? THEN 'Train' ELSE 'Test'
        END AS split,
        isFraud AS actual,
        isFlaggedFraud AS existing_flag,
        CASE
            WHEN "type" IN ('TRANSFER', 'CASH_OUT')
             AND oldbalanceOrg > 0
             AND ABS(amount - oldbalanceOrg) <= 0.01
            THEN 1 ELSE 0
        END AS balance_rule
    FROM transactions
),
comparison AS (
    SELECT split, actual,
           existing_flag AS predicted,
           'Existing flag' AS detector
    FROM predictions

    UNION ALL

    SELECT split, actual,
           balance_rule AS predicted,
           'Full-balance rule' AS detector
    FROM predictions
)
SELECT
    split,
    detector,
    SUM(CASE WHEN actual=1 AND predicted=1
        THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN actual=0 AND predicted=1
        THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN actual=1 AND predicted=0
        THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN actual=0 AND predicted=0
        THEN 1 ELSE 0 END) AS tn
FROM comparison
GROUP BY split, detector
ORDER BY split, detector
""", [cutoff]).fetchall()

records = []

print("\n=== DETECTOR PERFORMANCE BY TIME PERIOD ===")

for split, detector, tp, fp, fn, tn in results:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    print(f"\n{split} | {detector}")
    print(f"Transactions: {tp + fp + fn + tn:,}")
    print(f"Fraud cases: {tp + fn:,}")
    print(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"Precision: {precision:.2%}" if precision is not None
          else "Precision: N/A (no alerts)")
    print(f"Recall: {recall:.2%}" if recall is not None
          else "Recall: N/A (no fraud cases)")

    records.append({
        "split": split,
        "detector": detector,
        "cutoff_step": cutoff,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved: {OUTPUT}")
print("WARNING: Rule was discovered using the full dataset.")
print("This is a temporal stability check, not independent validation.")

con.close()
