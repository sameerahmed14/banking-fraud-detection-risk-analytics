
from pathlib import Path
import csv
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs"
DEST = ROOT / "dashboards" / "powerbi_data"

DEST.mkdir(parents=True, exist_ok=True)

# Validated outputs from our previous investigations.
FILES = {
    "fraud_by_type.csv": "fraud_by_type.csv",
    "fraud_by_amount_band.csv": "fraud_by_amount_band.csv",
    "fraud_by_day.csv": "fraud_by_day.csv",
    "fraud_balance_patterns.csv": "fraud_balance_patterns.csv",
    "ml_baseline_results.csv": "model_performance.csv",
    "fixed_capacity_analysis.csv": "batch_capacity.csv",
    "daily_alert_results.csv": "daily_alert_detail.csv",
    "daily_alert_summary.csv": "daily_capacity.csv",
    "daily_robustness_summary.csv": "robustness_results.csv",
    "daily_data_quality.csv": "day_quality.csv",
}

print("\n=== PREPARING POWER BI DATA ===")

missing = [
    name for name in FILES
    if not (SOURCE / name).is_file()
]

audit_path = SOURCE / "fraud_data_audit.json"
if not audit_path.is_file():
    missing.append("fraud_data_audit.json")

if missing:
    raise FileNotFoundError(
        "Missing analysis outputs:\n" + "\n".join(missing)
    )

for source_name, destination_name in FILES.items():
    shutil.copy2(
        SOURCE / source_name,
        DEST / destination_name
    )
    print(f"Created: {destination_name}")

# Create a compact executive KPI table.
audit = json.loads(
    audit_path.read_text(encoding="utf-8")
)

# Fraud transaction amounts are simulated values,
# not verified losses or a stated real-world currency.
fraud_value = 0.0

with (SOURCE / "fraud_by_type.csv").open(
    newline="", encoding="utf-8"
) as f:
    for row in csv.DictReader(f):
        fraud_value += float(
            row["fraud_transaction_value"]
        )

kpis = {
    "total_transactions": audit["total_transactions"],
    "fraud_transactions": audit["fraud_transactions"],
    "legitimate_transactions": audit["legitimate_transactions"],
    "fraud_rate_pct": audit["fraud_rate_pct"],
    "existing_flagged_transactions":
        audit["flagged_fraud_transactions"],
    "fraud_transaction_value": round(fraud_value, 2),
}

with (DEST / "executive_kpis.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=kpis.keys())
    writer.writeheader()
    writer.writerow(kpis)

print("Created: executive_kpis.csv")

# Quick consistency checks
assert kpis["total_transactions"] == (
    kpis["fraud_transactions"]
    + kpis["legitimate_transactions"]
)

assert kpis["fraud_transactions"] == 8213
assert kpis["total_transactions"] == 6362620

print("\nSUCCESS: Power BI datasets prepared")
print(f"Destination: {DEST}")
print(f"Files created: {len(FILES) + 1}")
print("Validation: PASSED")
