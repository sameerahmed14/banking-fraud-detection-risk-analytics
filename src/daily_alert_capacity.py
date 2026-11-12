
from pathlib import Path
import math

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/raw/PS_20174392719_1491204439457_log.csv"
MODEL_PATH = ROOT / "outputs/fraud_ml_baseline.joblib"
OUT = ROOT / "outputs"

if not CSV.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError("Dataset or trained model missing")

OUT.mkdir(parents=True, exist_ok=True)

saved = joblib.load(MODEL_PATH)
model = saved["model"]
test_cutoff = saved["test_cutoff"]

con = duckdb.connect()
path = CSV.as_posix().replace("'", "''")

print("Loading test transactions...")

frame = con.execute(f"""
    SELECT
        step,
        "type" AS tx_type,
        amount,
        isFraud AS label
    FROM read_csv_auto('{path}', header=true)
    WHERE step > ?
    ORDER BY step, nameOrig, nameDest, amount
""", [test_cutoff]).df()

con.close()

# PaySim steps represent simulated hours.
frame["simulated_day"] = (
    (frame["step"] - 1) // 24 + 1
).astype(int)

def calculate_scores(data):
    scores = np.zeros(len(data), dtype=np.float64)

    mask = data["tx_type"].isin(
        ["TRANSFER", "CASH_OUT"]
    ).to_numpy()

    subset = data.loc[mask]
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
        scores[mask] = model.predict_proba(
            features
        )[:, 1]

    return scores

print("Calculating fraud risk scores...")

scores = calculate_scores(frame)
actual = frame["label"].to_numpy(dtype=np.int8)

BUDGETS = [0.001, 0.0025, 0.005, 0.01]
records = []

# Rank transactions independently within each day.
for day, indices in frame.groupby(
    "simulated_day", sort=True
).indices.items():

    indices = np.asarray(indices)
    n = len(indices)
    fraud_count = int(actual[indices].sum())

    # Stable sorting ensures reproducible tie handling.
    ranking = indices[
        np.argsort(-scores[indices], kind="stable")
    ]

    for budget in BUDGETS:
        capacity = min(n, math.ceil(n * budget))

        selected = ranking[:capacity]

        tp = int(actual[selected].sum())
        fp = capacity - tp
        fn = fraud_count - tp
        tn = n - capacity - fn

        records.append({
            "simulated_day": int(day),
            "capacity_pct": budget * 100,
            "transactions": n,
            "fraud_cases": fraud_count,
            "alerts": capacity,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision_pct": (
                100 * tp / capacity if capacity else 0
            ),
            "recall_pct": (
                100 * tp / fraud_count
                if fraud_count else 0
            )
        })

daily = pd.DataFrame(records)

daily.to_csv(
    OUT / "daily_alert_results.csv",
    index=False
)

# Aggregate daily decisions for each capacity.
summary = daily.groupby(
    "capacity_pct", as_index=False
)[
    ["transactions", "fraud_cases", "alerts",
     "tp", "fp", "fn", "tn"]
].sum()

summary["precision_pct"] = (
    100 * summary["tp"] / summary["alerts"]
)

summary["recall_pct"] = (
    100 * summary["tp"] / summary["fraud_cases"]
)

summary.to_csv(
    OUT / "daily_alert_summary.csv",
    index=False
)

print("\n=== DAILY FRAUD ALERT CAPACITY ===")
print(
    summary[
        ["capacity_pct", "alerts", "tp", "fp",
         "fn", "precision_pct", "recall_pct"]
    ].round(2).to_string(index=False)
)

print("\n=== DAILY DETAIL: 0.50% CAPACITY ===")

detail = daily[
    np.isclose(daily["capacity_pct"], 0.5)
]

print(
    detail[
        ["simulated_day", "transactions",
         "fraud_cases", "alerts", "tp", "fp",
         "recall_pct"]
    ].round(2).to_string(index=False)
)

print("\nSaved:")
print(OUT / "daily_alert_results.csv")
print(OUT / "daily_alert_summary.csv")

print("\nIMPORTANT:")
print("Retrospective end-of-day batch ranking.")
print("Not real-time or independent test validation.")
print("Boundary days may contain partial-day data.")
