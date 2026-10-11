# Streamlit Community Cloud Deployment Readiness Report

**Project:** Financial Fraud Pattern Discovery  
**Target:** Streamlit Community Cloud  
**Status:** Deployment preparation started; application is not yet deployed.

## Executive summary

The dashboard entry point is [app/dashboard.py](./app/dashboard.py). The repository has the core Streamlit and pandas dependencies, and `scikit-learn` has been added because the dashboard uses it for evaluation metrics and threshold selection. The dashboard passes a Python syntax check.

The main deployment blocker is the generated `outputs/fraud_risk_scores.csv` file. It is approximately **188 MB** and is not tracked in Git. Therefore, a Community Cloud deployment from the current GitHub repository will not receive this file. The dashboard can still show the tracked association-rule results, but risk-score and evaluation sections will report that their data is unavailable.

No cloud app was created or published in this session. Publishing requires a GitHub-backed Community Cloud app setup and making the required score data available to that deployed app.

## Application overview

- **Framework:** Streamlit
- **Entry point:** `app/dashboard.py`
- **Dependencies:** Streamlit, pandas, and scikit-learn for evaluation. The dashboard also imports NumPy.
- **Generated outputs:** Association rules, frequent itemsets, fraud-focused rules, risk scores, high-risk transactions, and model comparison data.
- **Data source:** Synthetic PaySim financial transaction data and generated CSV outputs.

The dashboard resolves the `outputs` directory relative to the repository root. This works when the entry point is launched as `streamlit run app/dashboard.py` from the project root and when Community Cloud runs the configured entry point from the repository.

## Changes and checks completed

- Added `scikit-learn` to `requirements.txt` for precision/recall and average-precision evaluation.
- Updated the dashboard threshold evaluation so it selects a threshold on a reproducible stratified calibration split and reports metrics on separate rows from the scored CSV.
- Ran `python -m py_compile app\dashboard.py`; the syntax check passed.
- Confirmed the key score file is about 188 MB and is not tracked by Git.

The syntax check verifies parsing only; it does not establish that a deployed app can load every dataset or that all dashboard interactions have been tested in the cloud.

## Data availability

| Output | Approximate size | Tracked in Git | Cloud impact if absent |
|---|---:|:---:|---|
| `outputs/fraud_rules.csv` | 1 KB | Yes | Core association-rule view remains available |
| `outputs/frequent_itemsets.csv` | 67 KB | Yes | Core overview remains available |
| `outputs/fraud_focused_rules.csv` | 34 KB | No | Fraud-focused extension is unavailable |
| `outputs/fraud_risk_scores.csv` | 188 MB | No | Risk scoring and evaluation data are unavailable |
| `outputs/high_risk_transactions.csv` | 858 KB | No | High-risk transaction table is unavailable |
| `outputs/model_comparison.csv` | <1 KB | No | Comparison data is unavailable if referenced by the app |

Only the two core association-rule outputs are currently tracked. The other files are present in the local working tree but would not be included in a deployment sourced from the tracked GitHub files.

## Deployment blocker and options

The large risk-score file should not be added as a regular GitHub file: it is larger than GitHub's normal 100 MB per-file limit. The deployment needs a deliberate data-delivery approach before it can provide the complete dashboard.

1. **Recommended for a lightweight demo:** Reduce the data to a representative, clearly labeled sample that is small enough to track and deploy. Keep the full evaluation results separate so the demo sample is not presented as full-dataset evaluation.
2. **For complete output data:** Store the CSV in an approved external object store and update the app to load it securely at runtime. This requires choosing and configuring a hosting location and access method.
3. **Git LFS:** Consider only after confirming the repository and hosting workflow support LFS files and accounting for its storage and bandwidth limits. It should not be assumed to solve the Community Cloud data-delivery requirement.

No data has been reduced, uploaded, or moved to external storage as part of this preparation.

## Steps to publish after resolving data delivery

1. Commit and push the intended app, dependency, and data-loading changes to the GitHub repository. Review the existing working-tree changes first; there are other modified and untracked analysis files in the workspace.
2. Open Streamlit Community Cloud and create a new app connected to the repository and desired branch.
3. Set the main file path to `app/dashboard.py`.
4. Confirm the app is using the repository's `requirements.txt`.
5. Deploy and review the startup logs. Verify the sidebar data status, overview metrics, risk-scoring tab, evaluation tab, and high-risk transaction tab against the data expected to be available.
6. If any data is externalized, configure credentials through Community Cloud secrets rather than committing credentials to the repository.

## Readiness assessment

- **App entry point identified:** Yes
- **Dependency declaration updated:** Yes
- **Python syntax verified:** Yes
- **Complete dashboard tested in Community Cloud:** No
- **Required large score data delivered to the cloud:** No
- **Community Cloud app published:** No

**Conclusion:** The application is partially prepared, but it is not ready for a full-featured cloud deployment until a data-delivery option is selected for the untracked outputs, especially the 188 MB risk-score CSV.
