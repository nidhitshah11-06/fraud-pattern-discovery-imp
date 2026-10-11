# Financial Fraud Pattern Discovery Using Association Rule Mining

## Academic Project Report

**Project area:** Data mining and financial fraud analytics  
**Primary method:** FP-Growth association rule mining  
**Application:** Interactive Streamlit dashboard  
**Dataset:** Synthetic PaySim financial transactions  
**Student:** [Add name]  
**Course / institution:** [Add course and institution]  
**Submission date:** [Add date]

## Abstract

This project explores how frequent-pattern mining can help identify combinations of transaction characteristics associated with financial fraud. Using the synthetic PaySim transaction dataset, the analysis prepares categorical transaction features, mines frequent itemsets with FP-Growth, and examines fraud-associated rules using support, confidence, and lift. A complementary rule-based risk score ranks scored transactions for review. A Streamlit dashboard presents the mined patterns, risk summaries, high-risk transactions, and evaluation measures.

The saved analysis outputs contain 670 frequent itemsets, 10 standard fraud-associated rules, 220 fraud-focused patterns, and risk scores for 554,082 transactions. The dashboard’s current evaluation logic selects an F1 threshold on a reproducible stratified calibration split and reports results on separate held-out rows. Recalculation from the saved risk-score file yields a threshold of 99.75 and held-out precision of 1.000, recall of 0.966, F1 of 0.983, and average precision of 0.985. These figures describe this saved synthetic-data evaluation only; they do not establish performance on real banking activity. The repository also contains a model-comparison CSV whose average-precision values do not match this evaluation, so those values are not treated as verified results in this report.

**Keywords:** fraud analytics, FP-Growth, association rules, imbalanced classification, risk scoring, Streamlit

## 1. Introduction

Financial transaction systems generate large volumes of records, making it challenging to inspect every transaction manually. Data mining can help analysts identify recurring feature combinations that occur disproportionately often among transactions labelled fraudulent.

This project uses association rule mining to make such combinations interpretable. It also provides a dashboard to explore the resulting patterns and review a heuristic transaction risk score. The system is an academic analytical prototype: it is intended to support investigation and understanding, not to make autonomous decisions about customers or transactions.

## 2. Problem statement

The project addresses the question:

> Can frequent transaction patterns and association rules be used to identify interpretable combinations of transaction attributes associated with the fraud labels in a synthetic financial dataset?

The project further considers whether those patterns can contribute to a score that ranks transactions by similarity to fraud-associated patterns.

## 3. Objectives and scope

The project objectives are to:

1. Prepare transaction characteristics for categorical pattern mining.
2. Discover frequent itemsets using FP-Growth.
3. Generate and inspect fraud-associated association rules.
4. Rank scored transactions using a pattern-based heuristic.
5. Evaluate the score using fraud-detection measures appropriate to an imbalanced dataset.
6. Present results through an interactive Streamlit dashboard.

The analysis is based on synthetic PaySim data. It does not use real customer information, connect to a bank, or perform live transaction monitoring.

## 4. Dataset and preparation

The project documentation identifies PaySim as the data source and describes a dataset of approximately 6.3 million simulated financial transactions. The raw CSV is not included in the repository; the repository’s dataset instructions ask users to obtain it separately. Saved outputs are included locally, but some are not tracked in Git.

The documented analysis focuses on `TRANSFER` and `CASH_OUT` transactions. Transaction attributes are represented as categorical items, including:

- transaction type;
- amount category (`LOW`, `MEDIUM`, `HIGH`, or `VERY_HIGH`);
- balance-change category;
- simulation-time category (`MORNING`, `AFTERNOON`, `EVENING`, or `NIGHT`); and
- fraud label.

The notebook also contains engineered balance-related indicators, including whether an origin account was emptied and whether an origin balance check was consistent. The selected categorical features support interpretable itemset and rule generation. Because the raw dataset and a full fresh notebook run are not part of this report’s validation, preprocessing details and the final analysis population should be confirmed against the complete notebook and source data before formal replication.

## 5. Methodology

### 5.1 Frequent itemset mining

FP-Growth is used to identify combinations of categorical items that occur frequently without explicitly enumerating candidate itemsets in the manner of Apriori. The project README documents a minimum support value of 0.0005 for the standard mining workflow.

Support measures the proportion of transactions containing an itemset:

\[
\operatorname{support}(X \cup Y) =
\frac{\#(X \text{ and } Y)}{\#(\text{all transactions})}
\]

### 5.2 Association rule evaluation

A rule has the form \(X \rightarrow Y\), where \(X\) is an antecedent pattern and \(Y\) is a consequent, such as the fraud label. The project evaluates rules with:

- **Support:** how often the antecedent and consequent occur together.
- **Confidence:** the proportion of transactions matching the antecedent that also match the consequent.
- **Lift:** how much more often the consequent occurs with the antecedent than its overall baseline rate.

\[
\operatorname{confidence}(X \rightarrow Y) =
\frac{\operatorname{support}(X \cup Y)}{\operatorname{support}(X)}
\]

\[
\operatorname{lift}(X \rightarrow Y) =
\frac{\operatorname{confidence}(X \rightarrow Y)}{\operatorname{support}(Y)}
\]

Lift above 1 indicates a positive association in the analysed data; it does not establish causation.

### 5.3 Fraud-focused patterns and heuristic risk score

In addition to the standard association-rule output, the saved fraud-focused results contain 220 patterns with transaction counts, fraud counts, support, confidence, and lift. The notebook also builds a heuristic score from matches to fraud-associated rules. This score is used to rank transactions; it is not inherently a calibrated probability.

### 5.4 Evaluation procedure

The saved `fraud_risk_scores.csv` contains 554,082 rows and 1,642 fraud labels, corresponding to a fraud prevalence of approximately 0.296%. This severe class imbalance makes accuracy alone an inadequate summary.

The dashboard uses a stratified, reproducible 50/50 split of the saved scored rows (`random_state=42`). It chooses the F1-maximising threshold on the calibration half, then calculates precision, recall, F1, average precision, and confusion counts on the separate held-out half. The threshold procedure is intended to reduce optimistic evaluation caused by tuning and measuring on the same rows. It is an additional split of the saved scored output, not a fresh end-to-end retraining of the notebook.

The metrics are:

\[
\operatorname{precision} = \frac{TP}{TP+FP}
\qquad
\operatorname{recall} = \frac{TP}{TP+FN}
\]

\[
F1 = 2 \cdot \frac{\operatorname{precision}\cdot\operatorname{recall}}
{\operatorname{precision}+\operatorname{recall}}
\]

Average precision summarizes the precision-recall ranking over score thresholds and is more informative than accuracy for this highly imbalanced dataset.

## 6. Implementation

The analysis is implemented in the Jupyter notebook [notebooks/fraud_analysis.ipynb](./notebooks/fraud_analysis.ipynb). The dashboard is [app/dashboard.py](./app/dashboard.py), built with Streamlit and pandas, with NumPy for data operations and scikit-learn for evaluation functions.

The dashboard provides:

- an overview of rule and itemset counts;
- association-rule tables, lift visualizations, and a rule explorer;
- fraud-focused pattern analysis;
- risk-score summaries and risk-level distributions;
- a high-risk transaction explorer with filters and CSV download; and
- evaluation metrics, a confusion matrix, threshold comparison, and a precision-recall curve.

The Streamlit entry point is run locally with:

```bash
streamlit run app/dashboard.py
```

## 7. Results

### 7.1 Saved mining outputs

The currently available output files contain:

| Result | Value |
|---|---:|
| Frequent itemsets | 670 |
| Standard fraud-associated rules | 10 |
| Fraud-focused patterns | 220 |
| Scored transactions | 554,082 |
| Fraud labels in scored transactions | 1,642 |
| High-risk transaction rows in the saved output | 1,597 |

The highest-lift pattern in the current `fraud_focused_rules.csv` is `origin_emptied=ORIGIN_EMPTIED AND orig_balance_check=ORIG_OK`. The output records 4,811 transactions for this pattern, all labelled fraud, with confidence 1.0 and lift approximately 337.31. This striking association should be interpreted cautiously: it may reflect how the synthetic data’s fraud labels and balance-related attributes are generated, rather than a relationship that generalizes to real financial systems.

The README describes a different standard fraud-rule result: 10 rules and a strongest rule involving very high balance change and night-time simulation. The 220-row fraud-focused output is a separate analysis. These outputs should not be conflated.

### 7.2 Reproducible held-out score evaluation

Using the current saved `fraud_risk_scores.csv` and the dashboard’s stratified calibration/held-out split, the results are:

| Measure | Held-out result |
|---|---:|
| Selected threshold | 99.75 |
| Held-out transactions | 277,041 |
| Held-out fraud cases | 821 |
| Precision | 1.000 |
| Recall | 0.966 |
| F1-score | 0.983 |
| Average precision | 0.985 |
| Transactions flagged | 793 |
| True positives | 793 |
| False positives | 0 |
| False negatives | 28 |

The associated held-out fraud prevalence is approximately 0.296%. These results indicate strong ranking and classification performance on this particular saved score file and random split. Perfect observed precision means no false positives occurred among the 793 flagged held-out rows; it does not guarantee zero false alarms on future data.

### 7.3 Result consistency note

The repository’s `outputs/model_comparison.csv` records average precision values of 0.9898 for a rule-based score, 0.9933 for logistic regression, and 0.9276 for gradient boosting. Those numbers do not match the 0.9847 held-out average precision calculated from the current risk-score file. The project notebook contains multiple experimental sections and saved outputs, and the provenance of this comparison file is not established here. Accordingly, these comparison values are disclosed but excluded from the primary findings above until the exact data split, scoring implementation, and evaluation procedure are verified and regenerated consistently.

## 8. Work completed in the project

The project work represented by the current repository and recent changes includes:

1. An FP-Growth workflow for mining frequent transaction itemsets.
2. Generation of standard fraud-associated association rules and a separate fraud-focused pattern output.
3. Engineering of categorical transaction features and balance-related indicators for pattern analysis.
4. A heuristic, pattern-based transaction risk score and associated risk-level outputs.
5. A Streamlit dashboard for exploring association patterns, risk scores, high-risk transactions, and evaluation measures.
6. A dashboard runtime fix and follow-on evaluation changes, including a reproducible stratified calibration split separated from held-out reporting.
7. A declaration of scikit-learn in `requirements.txt` for the dashboard’s evaluation functions.
8. A deployment-readiness report documenting that the application has not been published and that the large risk-score output is not tracked in Git.

The dashboard Python file passed a syntax compilation check during the work. This check does not replace a full notebook rerun, comprehensive dashboard testing, or deployment validation.

## 9. Limitations and responsible interpretation

1. **Synthetic data:** PaySim is simulated and cannot establish effectiveness on real customer transactions.
2. **Potential simulator-specific signals:** Balance-related variables may be closely tied to the synthetic fraud-generation process. A rule with very high confidence may be learning that simulation mechanism.
3. **Association is not causation:** High lift describes co-occurrence, not a causal fraud indicator.
4. **Class imbalance:** With less than 0.3% fraud in the scored output, threshold changes strongly affect the number of alerts. Evaluation should report precision, recall, flagged volume, and the confusion matrix together.
5. **Risk score interpretation:** The score is a heuristic ranking value, not a calibrated probability of fraud.
6. **Evaluation scope:** The metrics in this report come from a reproducible split of a saved scored output. They do not demonstrate performance across time, institutions, or independent real-world datasets.
7. **Artifact consistency:** Multiple notebook experiments and outputs exist. The model-comparison CSV requires provenance and reproducibility checks before its values can support a direct model comparison.
8. **Deployment readiness:** The full dashboard has not been deployed to Streamlit Community Cloud. Its approximately 188 MB risk-score CSV is not tracked in Git, so the complete risk and evaluation features will require a suitable data-delivery approach.

The prototype should not be used to automatically block transactions, deny services, or make decisions about individuals. Any operational use would require external validation, governance, monitoring, privacy review, and human oversight.

## 10. Conclusion

This project demonstrates an interpretable fraud-pattern analysis workflow using FP-Growth and association rules, complemented by a heuristic risk score and an interactive dashboard. The saved outputs show substantial pattern associations, while the separately evaluated score file produces strong precision-recall results on the held-out portion of the scored sample.

The most important academic conclusion is bounded: these are results on synthetic data and saved artifacts, not proof of real-world fraud-detection performance. Future work should regenerate the full pipeline from a documented dataset version, reconcile the saved model-comparison results, test temporal or external validation splits, assess feature leakage, and document threshold trade-offs before drawing stronger conclusions.

## 11. Suggested future work

- Record the exact dataset version, source, preprocessing steps, and train/validation/test split in the notebook.
- Re-run the complete notebook from a clean environment and regenerate all outputs together.
- Investigate whether balance-derived variables leak information about the synthetic label-generation process.
- Compare the association-rule heuristic with supervised baselines using identical data partitions and metrics.
- Validate by time period or on an independent dataset, rather than relying only on a random split.
- Report confidence intervals and operational alert volumes for selected thresholds.
- Establish a suitable data-storage method before deploying the complete dashboard.

## Project files referenced

- [README.md](./README.md) — project overview, documented methodology, and dashboard instructions.
- [notebooks/fraud_analysis.ipynb](./notebooks/fraud_analysis.ipynb) — analysis and experiments.
- [app/dashboard.py](./app/dashboard.py) — interactive dashboard and evaluation display.
- [outputs/fraud_rules.csv](./outputs/fraud_rules.csv) — standard fraud-associated rules.
- [outputs/frequent_itemsets.csv](./outputs/frequent_itemsets.csv) — mined frequent itemsets.
- [outputs/fraud_focused_rules.csv](./outputs/fraud_focused_rules.csv) — fraud-focused patterns.
- [outputs/fraud_risk_scores.csv](./outputs/fraud_risk_scores.csv) — transaction scores and labels used for evaluation.
- [outputs/high_risk_transactions.csv](./outputs/high_risk_transactions.csv) — saved high-risk transaction subset.
- [outputs/model_comparison.csv](./outputs/model_comparison.csv) — comparison artifact requiring reproducibility verification.
