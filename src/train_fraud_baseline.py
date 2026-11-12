
from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

if not CSV.exists():
    raise FileNotFoundError(CSV)

con = duckdb.connect()
path = CSV.as_posix().replace("'", "''")

con.execute(f"""
CREATE VIEW transactions AS
SELECT *
FROM read_csv_auto('{path}', header=true)
""")

# Preserve our previous chronological test boundary.
TEST_CUTOFF = 355

# Reserve the final 25% of the earlier-period volume
# for validation, keeping simulated hours intact.
steps = con.execute("""
    SELECT step, COUNT(*)
    FROM transactions
    WHERE step <= ?
    GROUP BY step
    ORDER BY step
""", [TEST_CUTOFF]).fetchall()

target = sum(n for _, n in steps) * 0.75
running = 0
TRAIN_CUTOFF = None

for step, count in steps:
    running += count
    TRAIN_CUTOFF = step
    if running >= target:
        break

print("\n=== TIME PERIODS ===")
print(f"Training:   step <= {TRAIN_CUTOFF}")
print(f"Validation: {TRAIN_CUTOFF + 1} to {TEST_CUTOFF}")
print(f"Testing:    step > {TEST_CUTOFF}")

# Training uses all labelled fraud in the training period,
# plus a deterministic 15% sample of legitimate candidates.
# This reduces memory usage without touching validation/test.
train = con.execute("""
SELECT
    step,
    "type" AS tx_type,
    amount,
    isFraud AS label
FROM transactions
WHERE step <= ?
  AND "type" IN ('TRANSFER', 'CASH_OUT')
  AND (
      isFraud = 1
      OR hash(nameOrig) % 100 < 15
  )
""", [TRAIN_CUTOFF]).df()

# Validation and testing include ALL transaction types
# and ALL records, without class resampling.
def load_period(where_clause, cutoff):
    return con.execute(f"""
        SELECT
            step,
            "type" AS tx_type,
            amount,
            isFraud AS label,
            isFlaggedFraud
        FROM transactions
        WHERE {where_clause}
    """, cutoff).df()

val = load_period(
    "step > ? AND step <= ?",
    [TRAIN_CUTOFF, TEST_CUTOFF]
)

test = load_period(
    "step > ?",
    [TEST_CUTOFF]
)

def make_features(frame):
    step = frame["step"].to_numpy(dtype=np.int64)
    amount = frame["amount"].to_numpy(dtype=np.float64)

    return np.column_stack([
        np.log1p(np.maximum(amount, 0)),
        step.astype(np.float64),
        (frame["tx_type"] == "TRANSFER")
            .to_numpy(dtype=np.float64),
        ((step - 1) % 24).astype(np.float64)
    ])

print("\n=== TRAINING DATA ===")
print(f"Sampled training rows: {len(train):,}")
print(f"Training fraud cases: {int(train['label'].sum()):,}")
print(f"Validation rows: {len(val):,}")
print(f"Testing rows: {len(test):,}")

model = HistGradientBoostingClassifier(
    max_iter=120,
    max_leaf_nodes=15,
    learning_rate=0.08,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42
)

model.fit(
    make_features(train),
    train["label"].to_numpy(dtype=np.int8)
)

# No fraud was labelled in other transaction types
# in the dataset explored earlier. This baseline
# assigns those types a zero score.
def predict_scores(frame):
    scores = np.zeros(len(frame), dtype=np.float64)
    candidates = frame["tx_type"].isin(
        ["TRANSFER", "CASH_OUT"]
    ).to_numpy()

    scores[candidates] = model.predict_proba(
        make_features(frame.loc[candidates])
    )[:, 1]

    return scores

val_scores = predict_scores(val)

# Select a threshold using ONLY validation scores,
# targeting roughly 0.5% of validation transactions.
ALERT_RATE = 0.005
alert_count = max(1, int(np.ceil(len(val) * ALERT_RATE)))

threshold = float(
    np.partition(
        val_scores,
        len(val_scores) - alert_count
    )[len(val_scores) - alert_count]
)

print(f"\nValidation-selected threshold: {threshold:.6f}")
print("Target validation alert rate: 0.50%")

records = []

def evaluate(split_name, frame, scores):
    actual = frame["label"].to_numpy(dtype=np.int8)

    detectors = {
        "ML baseline": scores >= threshold,
        "Existing PaySim flag":
            frame["isFlaggedFraud"].to_numpy() == 1
    }

    ap = average_precision_score(actual, scores)

    print(f"\n=== {split_name.upper()} ===")
    print(f"Fraud cases: {int(actual.sum()):,}")
    print(f"ML average precision (AP): {ap:.4f}")

    for name, predictions in detectors.items():
        tp = int(np.sum((actual == 1) & predictions))
        fp = int(np.sum((actual == 0) & predictions))
        fn = int(np.sum((actual == 1) & ~predictions))
        tn = int(np.sum((actual == 0) & ~predictions))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        print(f"\n{name}")
        print(f"TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"Alerts: {tp + fp:,}")

        records.append({
            "split": split_name,
            "detector": name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "average_precision": (
                ap if name == "ML baseline" else None
            )
        })

evaluate("Validation", val, val_scores)
evaluate("Test", test, predict_scores(test))

pd.DataFrame(records).to_csv(
    OUT / "ml_baseline_results.csv",
    index=False
)

joblib.dump(
    {
        "model": model,
        "threshold": threshold,
        "train_cutoff": TRAIN_CUTOFF,
        "test_cutoff": TEST_CUTOFF,
        "features": [
            "log_amount",
            "step",
            "is_transfer",
            "simulated_hour"
        ]
    },
    OUT / "fraud_ml_baseline.joblib"
)

print("\nSaved model and evaluation CSV to outputs/")
print("Scores are not calibrated fraud probabilities.")
print("This dataset was explored before model development.")

con.close()
