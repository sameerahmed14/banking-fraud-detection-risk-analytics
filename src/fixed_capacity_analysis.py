
from pathlib import Path
import math

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
MODEL = ROOT / "outputs/fraud_ml_baseline.joblib"
OUTPUT = ROOT / "outputs/fixed_capacity_analysis.csv"

for file in [CSV, MODEL]:
    if not file.exists():
        raise FileNotFoundError(file)

saved = joblib.load(MODEL)
model = saved["model"]
train_cutoff = saved["train_cutoff"]
test_cutoff = saved["test_cutoff"]

con = duckdb.connect()
csv_path = CSV.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT *
FROM read_csv_auto('{csv_path}', header=true)
""")

def load_period(condition, params):
    return con.execute(f"""
        SELECT
            step,
            "type" AS tx_type,
            amount,
            isFraud AS label
        FROM transactions
        WHERE {condition}
        ORDER BY step, nameOrig, nameDest, amount
    """, params).df()

def predict_scores(frame):
    scores = np.zeros(len(frame), dtype=np.float64)

    mask = frame["tx_type"].isin(
        ["TRANSFER", "CASH_OUT"]
    ).to_numpy()

    subset = frame.loc[mask]

    step = subset["step"].to_numpy(dtype=np.int64)
    amount = subset["amount"].to_numpy(dtype=np.float64)

    features = np.column_stack([
        np.log1p(np.maximum(amount, 0)),
        step.astype(np.float64),
        (subset["tx_type"] == "TRANSFER")
            .to_numpy(dtype=np.float64),
        ((step - 1) % 24).astype(np.float64)
    ])

    if len(subset):
        scores[mask] = model.predict_proba(features)[:, 1]

    return scores

BUDGETS = [0.001, 0.0025, 0.005, 0.01]
records = []

def evaluate_period(split, condition, params):
    print(f"\nScoring {split.lower()} data...")

    frame = load_period(condition, params)
    scores = predict_scores(frame)
    actual = frame["label"].to_numpy() == 1

    n = len(frame)
    fraud_count = int(actual.sum())

    # Highest scores first. Stable sort provides a
    # deterministic tie break after the SQL ordering.
    ranking = np.argsort(-scores, kind="stable")

    ap = average_precision_score(actual, scores)

    print(f"\n=== {split.upper()} FIXED-CAPACITY RESULTS ===")
    print(f"Transactions: {n:,}")
    print(f"Fraud cases: {fraud_count:,}")
    print(f"Average precision: {ap:.4f}")

    for budget in BUDGETS:
        capacity = min(n, math.ceil(n * budget))

        selected = ranking[:capacity]
        tp = int(actual[selected].sum())
        fp = capacity - tp
        fn = fraud_count - tp
        tn = n - capacity - fn

        precision = tp / capacity if capacity else 0
        recall = tp / fraud_count if fraud_count else 0

        records.append({
            "split": split,
            "capacity_pct": budget * 100,
            "transactions": n,
            "fraud_cases": fraud_count,
            "alerts": capacity,
            "actual_alert_rate_pct": 100 * capacity / n,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision_pct": 100 * precision,
            "recall_pct": 100 * recall,
            "average_precision": ap
        })

        print(
            f"\nCapacity: {budget:.2%}"
            f"\nAlerts: {capacity:,}"
            f"\nTP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}"
            f"\nPrecision: {precision:.2%}"
            f"\nRecall: {recall:.2%}"
        )

    del frame, scores, actual, ranking

evaluate_period(
    "Validation",
    "step > ? AND step <= ?",
    [train_cutoff, test_cutoff]
)

evaluate_period(
    "Test",
    "step > ?",
    [test_cutoff]
)

result = pd.DataFrame(records)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print(f"\nResults saved: {OUTPUT}")
print("Exact alert capacity is enforced for each period.")
print("This is retrospective batch prioritisation.")
print("It is not a live, real-time fraud detection system.")

con.close()
