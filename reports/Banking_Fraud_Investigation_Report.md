# Banking Fraud Detection & Risk Analytics
## Business Investigation Report

**Dataset:** PaySim synthetic mobile-money transactions  
**Analysis:** independent analytics portfolio study, November 2026  
**Tools:** Python, DuckDB SQL, pandas, scikit-learn and Power BI  
**Evidence status:** Retrospective exploratory and chronological evaluation on synthetic data. No live bank deployment or independently untouched prospective test.

## Executive summary

This investigation analysed **6,362,620** simulated financial transactions and **8,213** labelled fraud cases, an overall fraud prevalence of **0.1291%**. Most transactions were legitimate, so a simple accuracy percentage would be misleading.

All labelled fraud cases occurred in **TRANSFER** (4,097 cases, **0.7688%** of transfers) and **CASH_OUT** (4,116 cases, **0.1840%** of cash-outs). Some transaction groups had high fraud *counts* but lower fraud *rates*. The dataset contains approximately **12.06 billion simulated monetary units** of transactions associated with fraud labels; this is neither verified loss nor a real-world currency figure.

A previously observed full-balance signature matched **8,018** fraud cases, with **97.63% in-sample recall** and zero in-sample false positives. Because this pattern was discovered by examining all available data and likely reflects simulation mechanics, it must not be reported as an independently validated real-world detection rule.

A separate histogram gradient-boosting model, deliberately excluding balance fields, achieved **51.04% recall and 20.45% precision** on later-period transactions at a validation-selected fixed score threshold. It detected **2,169** of **4,250** labelled frauds but generated **8,437 false positives**, demonstrating a material operational cost. In contrast, the pre-existing PaySim fraud flag detected only **13** frauds in that period with no false positives (**0.31% recall**, 100% observed precision based on only 13 alerts).

Investigating the top-scored **0.50% of transactions separately per simulated day** generated **6,251 alerts**, detected **1,227** fraud cases, and yielded **19.63% precision / 28.87% recall**. This is more operationally constrained than ranking transactions across the entire later period, but it is still **retrospective end-of-day selection**, not a live fraud-prevention service.

**Recommendation:** Treat the project as a proof of analytical workflow. For a hypothetical financial-risk team, evaluate precision/recall against fixed investigation capacity, inspect major changes in fraud prevalence, and establish trustworthy data and model-testing practices before proposing deployment.

## 1. Problem and scope

**Business question:** How effectively can fraud-risk signals prioritise suspicious transactions under limited investigation capacity?

The analysis aimed to:
- Identify labelled fraud concentrations by transaction type, value and simulated time.
- Investigate account balance behaviours while distinguishing simulator signatures from plausible generalisable signals.
- Benchmark an existing fraud flag, a transparent exploratory rule and a machine-learning classifier.
- Quantify false-positive workload, missed fraud and the impact of allocating investigator capacity across days.
- Communicate uncertainty, data limitations and proposed next steps through a four-page Power BI report.

**Dataset context:** PaySim is a synthetic **mobile-money** simulation; it does not contain transactions from the user's or an actual bank's customer accounts. The field `step` represents a simulated hour. Amounts have no independently verified physical currency in this investigation.

## 2. Data audit and definitions

| Audit measure | Result |
|---|---:|
| Rows | 6,362,620 |
| Labelled fraud (`isFraud = 1`) | 8,213 |
| Labelled legitimate (`isFraud = 0`) | 6,354,407 |
| Fraud prevalence | 0.1291% |
| `isFlaggedFraud = 1` | 16 |
| Null or negative amount rows | 0 |
| Simulated steps | 1–743 |
| Dataset columns | 11 |

**Definitions:** Fraud rate = labelled fraud transactions / total transactions in the relevant group. Fraud-associated value = sum of simulated transaction amounts where `isFraud = 1`. Precision = true fraud alerts / all alerts. Recall = true fraud alerts / all labelled fraud. False positives are legitimate transactions incorrectly alerted; false negatives are labelled fraud not alerted.

## 3. Exploratory findings

### 3.1 Transaction types and exposure

| Type | Transactions | Labelled fraud | Fraud rate |
|---|---:|---:|---:|
| TRANSFER | 532,909 | 4,097 | 0.7688% |
| CASH_OUT | 2,237,500 | 4,116 | 0.1840% |
| PAYMENT | 2,151,495 | 0 | 0% |
| CASH_IN | 1,399,284 | 0 | 0% |
| DEBIT | 41,432 | 0 | 0% |

TRANSFER has approximately **4.18×** the fraud *rate* of CASH_OUT, even though CASH_OUT has 19 more labelled fraud *cases*. Both measures are useful and answer different questions.

Fraud-associated transaction amounts total **12,056,415,427.84** synthetic units: approximately **6.07 billion** in TRANSFER and **5.99 billion** in CASH_OUT. This amount is **not** proven financial loss.

### 3.2 Fraud counts versus fraud rates by amount

| Band | TRANSFER cases / rate | CASH_OUT cases / rate |
|---|---:|---:|
| Under 1k | 21 / 3.6395% | 37 / 0.5122% |
| 1k–10k | 110 / 1.9975% | 110 / 0.1679% |
| 10k–100k | 711 / 1.2706% | 718 / 0.1024% |
| 100k+ | 3,255 / 0.6913% | 3,251 / 0.2221% |

The 100k+ group contains **6,506 of 8,213 frauds (79.2%)**, yet transfers below 1,000 show the highest *observed* fraud rate. That smallest TRANSFER group contains only **577 transactions**; small-denominator uncertainty matters.

### 3.3 Sender's pre-transaction balance

| Balance pattern | CASH_OUT fraud cases | TRANSFER fraud cases |
|---|---:|---:|
| Entire recorded prior balance | 4,075 | 3,943 |
| Partial prior balance | 1 | 149 |
| Zero prior balance | 37 | 4 |
| Exceeds prior balance | 3 | 1 |

All **8,018** eligible full-balance-pattern transactions in the dataset are labelled fraud. A simple in-sample rule therefore has **8,018 TP, 0 FP, 195 FN and 6,354,407 TN**, or **97.63% recall**. This near-perfect rule was developed **after exploring all the data** and may reproduce the simulator's built-in fraudulent-agent pattern, rather than generalisable bank behaviour. It is **not** an unbiased benchmark.

### 3.4 Simulated-time patterns and boundary anomaly

Fraud cases are unevenly distributed through the simulation. Day 17 contains 320 fraud cases among 425,766 transactions, whereas simulated day 3 contains 310 among only 1,070. The final simulated day (31) contains 272 transactions, all 272 labelled fraud. Day 31 is anomalous and may reflect simulation-end behaviour or data-boundary effects, not a real-world event.

## 4. Machine-learning evaluation

### 4.1 Model design

The baseline is a scikit-learn **HistGradientBoostingClassifier** using only:
- log1p(transaction amount),
- simulated step,
- TRANSFER indicator, and
- simulated hour-of-day.

Balance columns, IDs, target labels and the existing fraud flag are excluded as **predictive features**. The training dataset contains **all 3,211 training-period frauds** and a deterministic 15% sample of eligible legitimate transfer/cash-out transactions (256,162 training rows total). Validation and test datasets are **not downsampled**.

Period boundaries are:
- **Training:** steps ≤282;
- **Validation:** steps 283–355, 1,262,167 transactions and 752 labelled frauds;
- **Test:** steps >355, 1,248,736 transactions and 4,250 labelled frauds.

**Important evaluation caveats:** The full dataset was examined during project exploration **before** these features and baselines were finalised. Consequently, the later-period test is chronologically separate from model fitting but **not a pristine prospectively untouched test**. The model's scores are also **not calibrated fraud probabilities** because the legitimate class was subsampled for training.

### 4.2 Fixed score threshold

A fixed score threshold (**0.093134**) was chosen from validation scores to target roughly 0.50% validation alerts.

| Outcome | Validation ML | Test ML | Test existing flag |
|---|---:|---:|---:|
| True positives | 382 | 2,169 | 13 |
| False positives | 6,157 | 8,437 | 0 |
| False negatives | 370 | 2,081 | 4,237 |
| True negatives | 1,255,258 | 1,236,049 | 1,244,486 |
| Precision | 5.84% | **20.45%** | **100.00%** |
| Recall | 50.80% | **51.04%** | **0.31%** |
| Average precision (ML ranking) | 0.4165 | **0.2970** | — |

The test period has **higher fraud prevalence (0.3403%)** than the earlier 80% of transactions (**0.0775%**), which affects precision. A 0.50% target in validation resulted in **0.849% actual test alerts**—illustrating why an unchanged model-score cutoff does not guarantee stable staffing demand.

### 4.3 Whole-period top-K alert allocation

| Test-period capacity | Alerts | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.10% | 1,249 | 724 | 525 | 57.97% | 17.04% |
| 0.25% | 3,122 | 1,423 | 1,699 | 45.58% | 33.48% |
| 0.50% | 6,244 | 1,961 | 4,283 | 31.41% | 46.14% |
| 1.00% | 12,488 | 2,202 | 10,286 | 17.63% | 51.81% |

These quotas require ranking **all** transactions from the full later evaluation period before picking a subset. They are retrospective, not possible as instantaneous fraud-prevention decisions.

## 5. Daily investigation-capacity analysis

We next allocated each day's own alert quota, using the model's risk-score ranking for that day.

| Daily quota | Total alerts | Detected fraud | False positives | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 0.10% | 1,256 | 543 | 713 | 43.23% | 12.78% |
| 0.25% | 3,130 | 867 | 2,263 | 27.70% | 20.40% |
| **0.50%** | **6,251** | **1,227** | **5,024** | **19.63%** | **28.87%** |
| 1.00% | 12,493 | 1,605 | 10,888 | 12.85% | 37.76% |

At the illustrative 0.50% quota, daily selection captured **734 fewer** frauds than whole-period top-K selection, despite **7 more** alerts. This is an *allocation* difference, not a new model.

Each day's quota uses **ceil(daily transaction count × target fraction)**; totals can slightly exceed the exact fraction of the full sample. Decisions are made retrospectively using the **full day's** scores: the experiment does not demonstrate online latency, real-time detection or blocked transactions.

### Sensitivity to dataset boundaries

| Evaluation scope at 0.50% daily capacity | Transactions | Frauds | Alerts | TP | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| All test days | 1,248,736 | 4,250 | 6,251 | 1,227 | 19.63% | 28.87% |
| Interior days 16–30 | 1,176,291 | 3,924 | 5,888 | 1,209 | 20.53% | 30.81% |

Day 15 is partial because the testing cutoff falls within that simulated day. Day 31 has just **272** records, all labelled fraudulent. Omitting both improves recall by approximately **1.94 percentage points** and precision by **0.90 percentage points**. This is a **sensitivity analysis**, not a new independent test.

## 6. Proposed actions for a hypothetical risk team

| Priority | Action | Validation needed before implementation |
|---|---|---|
| 1 | Base staffing discussions on **alert capacity, precision, recall, and false-positive workload together**. | Estimate actual investigator throughput, cost per review and the harm of fraud missed versus customers wrongly flagged. |
| 2 | Set **day-specific or time-window alert limits** rather than relying entirely on a fixed model cutoff. | Conduct forward-looking evaluation at the actual decision time; compare daily quotas and rolling/hourly budgets. |
| 3 | Monitor **score distributions and fraud prevalence** for drift and unusual boundary conditions. | Track reference cohorts, record data provenance and set quality alerts; audit shifts like the 100%-fraud synthetic final day. |
| 4 | Evaluate fraud amounts alongside fraud counts **without equating gross transaction amount with loss avoided**. | Obtain confirmed losses, recoveries, false-positive customer harm and realistic intervention success rates. |
| 5 | Rebuild and retest models on a **new, truly unexamined holdout** using prospective design decisions. | Freeze features and alert thresholds before accessing the new period; evaluate model calibration and stability. |
| 6 | Treat highly predictive full-balance rules as **simulation-specific hypotheses**. | Validate any analogous signal on independently sourced operational data; check transaction timing, account balances, fraud definition and spurious generation artefacts. |

An **illustrative 0.50% daily quota** is a clear way to present the trade-off, but available evidence does **not** show it is financially optimal. No realised costs, savings or fraud-prevention benefits have been established.

## 7. Limitations

1. **Synthetic simulation, not real bank data.** PaySim's fraud-generating mechanisms may create perfect balance signatures not found in real operations.
2. **Prior exploratory exposure.** The later-period set was examined before model development and is not independently untouched.
3. **Uncalibrated model scores.** Training sampled legitimate records; a score cannot simply be interpreted as fraud probability.
4. **No operational timing.** Whole-period and end-of-day ranking are offline retrospective choices.
5. **Unspecified monetary unit and financial impact.** Neither real currency nor verified fraud loss, recoveries and investigation costs are established.
6. **Incomplete boundary periods and prevalence drift.** The final simulated day is especially atypical; period choice can change metric estimates.
7. **No customer-level intervention evaluation.** The study does not quantify blocking, friction, fairness, live transaction timing, or investigator outcomes.
8. **Possible aggregation pitfalls.** Dashboard source tables represent different grouping levels; rates must be calculated from relevant numerators and denominators, not added across scenarios.

## 8. Reproducibility and deliverables

The [project README](../README.md) provides the Kaggle data source, install commands, script execution order, GitHub folder guide and screenshots. The raw dataset and model binary are intentionally omitted from GitHub; all **11 compact Power BI CSV exports** are included.

Delivered:
- Python/DuckDB data-quality audit and exploratory investigations;
- baseline fraud flag and transparent full-balance exploratory rule;
- chronological ML baseline with validation/test results;
- threshold, fixed whole-period quota, daily quota and boundary robustness analysis;
- four-page Power BI report and screenshot previews;
- documented limitations and business recommendations.

## Conclusion

This project shows why practical fraud analytics requires more than a high fraud-detection score. Fraud is rare, cost and capacity are constrained, scores change across time, and some very strong synthetic patterns may not generalise. A transparent approach that reports **precision, recall, alert volume, false positives and testing limits** offers more trustworthy support for operational decisions than claiming an unrealistically perfect detector.

---

*All figures refer to PaySim simulated transactions and the published retrospective outputs. This is an independent portfolio case study, not a commissioned banking deployment.*
