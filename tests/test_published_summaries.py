"""Smoke checks for the published Power BI summary outputs.

These checks validate committed exports, not reproducibility from raw PaySim.
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "dashboards" / "powerbi_data"


def load(name):
    with (DATA / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def test_executive_totals():
    rows = load("executive_kpis.csv")
    assert len(rows) == 1
    row = rows[0]
    assert int(row["total_transactions"]) == 6_362_620
    assert int(row["fraud_transactions"]) == 8_213
    assert int(row["legitimate_transactions"]) == 6_354_407
    assert int(row["fraud_transactions"]) + int(row["legitimate_transactions"]) == int(row["total_transactions"])


def test_fraud_by_type_reconciles_to_executive():
    rows = load("fraud_by_type.csv")
    assert sum(int(x["total_transactions"]) for x in rows) == 6_362_620
    assert sum(int(float(x["fraud_cases"])) for x in rows) == 8_213
    counts = {x["transaction_type"]: int(float(x["fraud_cases"])) for x in rows}
    assert counts["CASH_OUT"] == 4_116
    assert counts["TRANSFER"] == 4_097


def test_model_confusion_matrices_reconcile():
    rows = load("model_performance.csv")
    assert len(rows) == 4
    for row in rows:
        tp, fp, fn, tn = [int(row[key]) for key in ("tp", "fp", "fn", "tn")]
        expected = 1_248_736 if row["split"] == "Test" else 1_262_167
        assert tp + fp + fn + tn == expected
        expected_fraud = 4_250 if row["split"] == "Test" else 752
        assert tp + fn == expected_fraud
    model = next(r for r in rows if r["split"] == "Test" and r["detector"] == "ML baseline")
    assert (int(model["tp"]), int(model["fp"]), int(model["fn"])) == (2_169, 8_437, 2_081)


def test_daily_capacity_confusion_matrices():
    rows = load("daily_capacity.csv")
    assert len(rows) == 4
    for row in rows:
        tp, fp, fn, tn = [int(row[key]) for key in ("tp", "fp", "fn", "tn")]
        assert tp + fp == int(row["alerts"])
        assert tp + fn == int(row["fraud_cases"]) == 4_250
        assert tp + fp + fn + tn == int(row["transactions"]) == 1_248_736
    half = next(row for row in rows if float(row["capacity_pct"]) == 0.5)
    assert (int(half["alerts"]), int(half["tp"]), int(half["fp"])) == (6_251, 1_227, 5_024)


def test_robustness_reconciles_to_daily_detail():
    rows = load("daily_alert_detail.csv")
    robust = load("robustness_results.csv")
    for scenario in robust:
        matching = [
            row for row in rows
            if float(row["capacity_pct"]) == float(scenario["capacity_pct"])
            and (
                scenario["scope"] == "All test days"
                or 16 <= int(row["simulated_day"]) <= 30
            )
        ]
        for column in ("transactions", "fraud_cases", "alerts", "tp", "fp", "fn", "tn"):
            assert sum(int(row[column]) for row in matching) == int(scenario[column])
