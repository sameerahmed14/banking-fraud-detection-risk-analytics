
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

daily = pd.read_csv(
    OUT / "daily_alert_results.csv"
)

# Validate the saved confusion-matrix counts
assert (
    daily["tp"] + daily["fn"]
    == daily["fraud_cases"]
).all()

assert (
    daily["tp"] + daily["fp"]
    == daily["alerts"]
).all()

assert (
    daily["tp"] + daily["fp"]
    + daily["fn"] + daily["tn"]
    == daily["transactions"]
).all()

scenarios = {
    "All test days": daily,
    "Interior days 16-30": daily[
        daily["simulated_day"].between(16, 30)
    ]
}

results = []

for scope, data in scenarios.items():
    summary = data.groupby(
        "capacity_pct", as_index=False
    )[[
        "transactions", "fraud_cases",
        "alerts", "tp", "fp", "fn", "tn"
    ]].sum()

    summary["scope"] = scope
    summary["precision_pct"] = (
        100 * summary["tp"] / summary["alerts"]
    )
    summary["recall_pct"] = (
        100 * summary["tp"] / summary["fraud_cases"]
    )
    results.append(summary)

combined = pd.concat(results, ignore_index=True)

combined.to_csv(
    OUT / "daily_robustness_summary.csv",
    index=False
)

print("\n=== DAILY ALERT ROBUSTNESS CHECK ===")

for scope in scenarios:
    print(f"\n{scope.upper()}")

    view = combined[
        combined["scope"] == scope
    ][[
        "capacity_pct", "transactions",
        "fraud_cases", "alerts",
        "tp", "fp", "precision_pct",
        "recall_pct"
    ]]

    print(view.round(2).to_string(index=False))

# Inspect the risk distribution by simulated day
day_quality = daily[
    np.isclose(daily["capacity_pct"], 0.5)
][[
    "simulated_day",
    "transactions",
    "fraud_cases"
]].copy()

day_quality["fraud_rate_pct"] = (
    100 * day_quality["fraud_cases"]
    / day_quality["transactions"]
)

day_quality["boundary_day"] = (
    day_quality["simulated_day"].isin([15, 31])
)

day_quality.to_csv(
    OUT / "daily_data_quality.csv",
    index=False
)

print("\n=== BOUNDARY DAY QUALITY CHECK ===")
print(
    day_quality[
        day_quality["boundary_day"]
    ].round(2).to_string(index=False)
)

print("\nSaved robustness and quality reports.")
