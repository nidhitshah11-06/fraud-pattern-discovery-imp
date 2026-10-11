# Streamlit Community Cloud Deployment Readiness Report

**Project:** Financial Fraud Pattern Discovery  
**Target:** Streamlit Community Cloud  
**Status:** The app is deployed by the project owner. A missing risk-score file was identified and is being fixed.

## Executive summary

The dashboard entry point is [app/dashboard.py](./app/dashboard.py). The app owner reported that the Risk Scoring tab showed an unavailable-data message. The cause was that `outputs/fraud_risk_scores.csv` was not present in the GitHub repository used by Streamlit Cloud.

The local source score file is approximately **188 MB**, above GitHub's regular per-file limit. It compresses to approximately **4.96 MB** as `outputs/fraud_risk_scores.csv.gz`. The dashboard loader has been updated to use this gzip file when the uncompressed CSV is absent. This keeps all scored rows while allowing the artifact to be committed normally.

The fix must be pushed to the repository and the Streamlit deployment must restart or pull the latest `main` commit before the Risk Scoring tab can load the file.

## Application overview

- **Framework:** Streamlit
- **Entry point:** `app/dashboard.py`
- **Dependencies:** Streamlit, pandas, and scikit-learn for evaluation. The dashboard also imports NumPy.
- **Generated outputs:** Association rules, frequent itemsets, fraud-focused rules, compressed risk scores, high-risk transactions, and model comparison data.
- **Data source:** Synthetic PaySim financial transaction data and generated CSV outputs.

The dashboard resolves the `outputs` directory relative to the repository root. This works when the entry point is launched as `streamlit run app/dashboard.py` from the project root and when Community Cloud runs the configured entry point from the repository.

## Changes and checks completed

- Added `scikit-learn` to `requirements.txt` for precision/recall and average-precision evaluation.
- Updated the dashboard threshold evaluation so it selects a threshold on a reproducible stratified calibration split and reports metrics on separate rows from the scored CSV.
- Compressed the complete 188 MB risk-score CSV to `outputs/fraud_risk_scores.csv.gz` (approximately 4.96 MB).
- Updated `load_csv` to transparently load the `.csv.gz` file when the plain `.csv` is absent.
- Ran `python -m py_compile app\dashboard.py`; the syntax check passed.
- The compressed file remains the full score dataset; it is not a downsample.

The syntax check verifies parsing only. The fix must still be verified in the deployed Streamlit app after the commit is available.

## Data availability

| Output | Approximate size | Tracked in Git | Cloud impact if absent |
|---|---:|:---:|---|
| `outputs/fraud_rules.csv` | 1 KB | Yes | Core association-rule view remains available |
| `outputs/frequent_itemsets.csv` | 67 KB | Yes | Core overview remains available |
| `outputs/fraud_focused_rules.csv` | 34 KB | Yes | Fraud-focused extension is available |
| `outputs/fraud_risk_scores.csv.gz` | 4.96 MB | To be pushed | Risk scoring and evaluation data load through the compressed-file fallback |
| `outputs/high_risk_transactions.csv` | 858 KB | Yes | High-risk transaction table is available |
| `outputs/model_comparison.csv` | <1 KB | Yes | Comparison data is available if referenced by the app |

The raw 493 MB dataset and uncompressed 188 MB risk-score CSV are excluded; the complete risk-score data is included as gzip instead.

## Deployment blocker and options

The original uncompressed score file exceeds GitHub's regular per-file limit. Compression resolves this without reducing the data.

1. Push the gzip artifact and loader update to the same GitHub repository and branch configured in Streamlit Cloud.
2. Confirm `app/dashboard.py` is still set as the app's main file.
3. Restart or reboot the app so Streamlit Cloud pulls the latest commit.
4. Check the sidebar data status, then open Risk Scoring and Evaluation Metrics.

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
- **Risk score data compressed without downsampling:** Yes
- **Dashboard compressed-file fallback implemented:** Yes
- **Fix verified in Community Cloud:** Pending push and app restart

**Conclusion:** The reported error is caused by the missing untracked score CSV, not by the risk scoring calculation itself. The full score data now has a GitHub-compatible compressed representation, and the dashboard can load it transparently after the fix is pushed and the app restarts.
