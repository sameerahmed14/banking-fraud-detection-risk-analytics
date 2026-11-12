
from pathlib import Path
import csv
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
OUTPUT = ROOT / "outputs/rule_baseline_results.csv"

if not DATA.exists():
    raise FileNotFoundError(DATA)

con = duckdb.connect()
path = DATA.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT
    "type", amount, oldbalanceOrg,
    isFraud, isFlaggedFraud
FROM read_csv_auto('{path}', header=true)
""")

results = con.execute("""
WITH predictions AS (
    SELECT
        isFraud AS actual,
        isFlaggedFraud AS existing_alert,
        CASE
            WHEN "type" IN ('TRANSFER', 'CASH_OUT')
             AND oldbalanceOrg > 0
             AND ABS(amount - oldbalanceOrg) <= 0.01
            THEN 1 ELSE 0
        END AS balance_alert
    FROM transactions
),
comparison AS (
    SELECT
        'Existing PaySim flag' AS detector,
        actual,
        existing_alert AS predicted
    FROM predictions

    UNION ALL

    SELECT
        'Full-balance exploratory rule' AS detector,
        actual,
        balance_alert AS predicted
    FROM predictions
)
SELECT
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
GROUP BY detector
ORDER BY detector
""").fetchall()

print("\n=== EXPLORATORY FRAUD DETECTION BASELINES ===")

records = []

for name, tp, fp, fn, tn in results:
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(f"\nDetector: {name}")
    print(f"True positives:  {tp:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True negatives:  {tn:,}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")

    records.append({
        "detector": name,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6)
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"\nSaved: {OUTPUT}")
print("Note: In-sample exploratory results, not a holdout test.")

con.close()
