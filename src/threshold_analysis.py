
from pathlib import Path
import math

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
MODEL_PATH = ROOT / "outputs/fraud_ml_baseline.joblib"
OUTPUT = ROOT / "outputs/threshold_analysis.csv"

if not CSV.exists():
    raise FileNotFoundError(CSV)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

saved = joblib.load(MODEL_PATH)
model = saved["model"]
train_cutoff = saved["train_cutoff"]
test_cutoff = saved["test_cutoff"]

con = duckdb.connect()
path = CSV.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT
    step,
    "type" AS tx_type,
    amount,
    isFraud AS label
FROM read_csv_auto('{path}', header=true)
""")

def load_period(where_sql, params):
    return con.execute(f"""
        SELECT *
        FROM transactions
        WHERE {where_sql}
    """, params).df()

validation = load_period(
    "step > ? AND step <= ?",
    [train_cutoff, test_cutoff]
)

testing = load_period(
    "step > ?",
    [test_cutoff]
)

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

    scores[mask] = model.predict_proba(features)[:, 1]

    return scores

print("\nScoring validation transactions...")
val_scores = predict_scores(validation)

print("Scoring testing transactions...")
test_scores = predict_scores(testing)

BUDGETS = [0.001, 0.0025, 0.005, 0.01]
records = []

for budget in BUDGETS:

    # Select each operating threshold using validation only.
    n_alerts = max(
        1, math.ceil(len(val_scores) * budget)
    )

    threshold = float(
        np.partition(
            val_scores,
            len(val_scores) - n_alerts
        )[len(val_scores) - n_alerts]
    )

    for split, frame, scores in [
        ("Validation", validation, val_scores),
        ("Test", testing, test_scores)
    ]:
        actual = frame["label"].to_numpy() == 1
        predicted = scores >= threshold

        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))
        tn = int(np.sum(~actual & ~predicted))

        alerts = tp + fp

        precision = tp / alerts if alerts else 0
        recall = tp / (tp + fn) if tp + fn else 0

        records.append({
            "split": split,
            "validation_budget_pct": budget * 100,
            "threshold": threshold,
            "transactions": len(frame),
            "alerts": alerts,
            "actual_alert_rate_pct": (
                100 * alerts / len(frame)
            ),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision_pct": precision * 100,
            "recall_pct": recall * 100
        })

result = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("\n=== FRAUD ALERT CAPACITY ANALYSIS ===")

for split in ["Validation", "Test"]:
    print(f"\n{split.upper()}")

    cols = [
        "validation_budget_pct",
        "alerts",
        "actual_alert_rate_pct",
        "true_positives",
        "false_positives",
        "precision_pct",
        "recall_pct"
    ]

    print(
        result[result["split"] == split][cols]
        .round(3)
        .to_string(index=False)
    )

print(f"\nResults saved: {OUTPUT}")
print("Thresholds selected using validation scores only.")
print("Actual alert rates may differ due to tied scores.")
print("Test comparisons are diagnostic, not a basis for")
print("retuning the already-examined testing period.")

con.close()
