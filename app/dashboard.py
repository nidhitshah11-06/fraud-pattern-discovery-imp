# ============================================================
# FINANCIAL FRAUD ASSOCIATION RULE & RISK ANALYSIS DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Pattern Discovery",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# 2. FILE LOCATIONS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

FILES = {
    "fraud_rules.csv": OUTPUT_DIR / "fraud_rules.csv",
    "frequent_itemsets.csv": OUTPUT_DIR / "frequent_itemsets.csv",
    "fraud_focused_rules.csv": OUTPUT_DIR / "fraud_focused_rules.csv",
    "fraud_risk_scores.csv": OUTPUT_DIR / "fraud_risk_scores.csv",
    "high_risk_transactions.csv": OUTPUT_DIR / "high_risk_transactions.csv",
}

LEVEL_ORDER = ["Low", "Medium", "High"]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def show_df(df, **kwargs):
    """Display a dataframe full-width across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(
            df, use_container_width=True, hide_index=True, **kwargs
        )


@st.cache_data(show_spinner=False)
def load_csv(path_str):
    """Return (dataframe or None, error message or None)."""
    path = Path(path_str)
    if not path.exists():
        compressed_path = Path(f"{path}.gz")
        if compressed_path.exists():
            path = compressed_path
        else:
            return None, "file not found"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, f"could not be read: {exc}"


def find_col(df, options):
    """Find the first column whose name matches any option."""
    lookup = {c.lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lookup:
            return lookup[option.lower()]
    return None


def prepare_scores(df):
    """
    Standardise the risk-score table.
    Returns (clean dataframe or None, list of notes).
    """
    notes = []
    if df is None:
        return None, notes

    df = df.copy()

    score_col = find_col(df, ["risk_score", "fraud_risk_score", "score"])
    if score_col is None:
        notes.append("No risk_score column was found.")
        return None, notes
    if score_col != "risk_score":
        df = df.rename(columns={score_col: "risk_score"})

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df = df.dropna(subset=["risk_score"])

    if len(df) and df["risk_score"].max() <= 1.0:
        df["risk_score"] = df["risk_score"] * 100
        notes.append("Risk scores looked like 0–1 values and were "
                     "rescaled to 0–100.")

    label_col = find_col(df, ["isFraud", "is_fraud", "fraud", "label"])
    if label_col is not None:
        if label_col != "isFraud":
            df = df.rename(columns={label_col: "isFraud"})
        df["isFraud"] = (
            pd.to_numeric(df["isFraud"], errors="coerce")
            .fillna(0).astype(int)
        )

    level_col = find_col(
        df, ["risk_level", "risk_category", "risk_band"]
    )
    if level_col is not None:
        if level_col != "risk_level":
            df = df.rename(columns={level_col: "risk_level"})
        df["risk_level"] = df["risk_level"].astype(str).str.title()
    else:
        df["risk_level"] = pd.cut(
            df["risk_score"],
            bins=[-np.inf, 40, 70, np.inf],
            labels=LEVEL_ORDER,
            right=False
        ).astype(str)
        notes.append("No risk_level column was found, so levels were "
                     "derived from the score (Low < 40, Medium 40–69, "
                     "High ≥ 70).")

    return df, notes


def metrics_at(y_true, scores, threshold):
    """Precision, recall, F1 and confusion counts at a threshold."""
    pred = scores >= threshold
    actual = y_true == 1

    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    tn = int((~pred & ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged": tp + fp,
    }


def best_threshold_for_f1(y_true, scores):
    """Choose the threshold that maximises F1 on the observed score values."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    best = None
    for threshold in np.unique(scores):
        result = metrics_at(y_true, scores, float(threshold))
        candidate = {
            "threshold": float(threshold),
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
        }
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate
        elif (
            abs(candidate["f1"] - best["f1"]) < 1e-12
            and candidate["precision"] > best["precision"]
        ):
            best = candidate

    if best is None:
        return 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return best["threshold"], {
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
    }


# ============================================================
# 4. LOAD FILES
# ============================================================

loaded = {}
errors = {}
for name, path in FILES.items():
    loaded[name], errors[name] = load_csv(str(path))

rules = loaded["fraud_rules.csv"]
itemsets = loaded["frequent_itemsets.csv"]
focused_rules = loaded["fraud_focused_rules.csv"]

risk_scores, score_notes = prepare_scores(loaded["fraud_risk_scores.csv"])
high_risk_file, _ = prepare_scores(loaded["high_risk_transactions.csv"])

# Core files are required.
if rules is None or itemsets is None:
    st.error(
        "Required files are missing from the outputs folder: "
        "fraud_rules.csv and frequent_itemsets.csv."
    )
    st.code(f"Looking in: {OUTPUT_DIR}")
    st.stop()

required_cols = {"rule", "support", "confidence", "lift"}
missing_cols = required_cols - set(rules.columns)
if missing_cols:
    st.error(
        f"fraud_rules.csv is missing columns: {sorted(missing_cols)}. "
        f"Columns found: {rules.columns.tolist()}"
    )
    st.stop()

for col in ["support", "confidence", "lift"]:
    rules[col] = pd.to_numeric(rules[col], errors="coerce")
rules = rules.dropna(subset=["support", "confidence", "lift"])
rules = rules.reset_index(drop=True)


# ============================================================
# 5. SIDEBAR — DATA STATUS
# ============================================================

with st.sidebar:
    st.header("Data status")
    st.caption(f"Outputs folder: `{OUTPUT_DIR}`")

    for name in FILES:
        df_status = loaded[name]
        if df_status is not None:
            st.success(f"{name}: {len(df_status):,} rows")
        else:
            st.warning(f"{name}: {errors[name]}")

    if risk_scores is not None:
        st.divider()
        st.write("**Risk-score columns found**")
        st.write(
            "risk_score ✅  "
            f"| risk_level ✅  "
            f"| isFraud {'✅' if 'isFraud' in risk_scores.columns else '❌'}"
        )
        for note in score_notes:
            st.info(note)

    if st.button("Reload data from disk"):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# 6. HEADER
# ============================================================

st.title("Financial Fraud Pattern Discovery")

st.write(
    "FP-Growth based analysis of transaction patterns "
    "associated with financial fraud in banking transactions, "
    "extended with fraud-focused pattern mining and risk scoring."
)

st.caption(
    "Academic prototype using the synthetic PaySim financial "
    "transaction dataset."
)


# ============================================================
# 7. TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Association Rules",
    "Risk Scoring",
    "High-Risk Transactions",
    "Evaluation Metrics",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Fraud-Associated Rules", len(rules))
    with c2:
        st.metric("Frequent Itemsets", len(itemsets))
    with c3:
        st.metric("Highest Lift", f"{rules['lift'].max():.2f}")
    with c4:
        st.metric(
            "Strongest Confidence",
            f"{rules['confidence'].max() * 100:.2f}%"
        )

    if risk_scores is not None:
        st.subheader("Risk Scoring Summary")

        r1, r2, r3, r4 = st.columns(4)
        n_total = len(risk_scores)
        n_high = int((risk_scores["risk_level"] == "High").sum())

        with r1:
            st.metric("Scored Transactions", f"{n_total:,}")
        with r2:
            st.metric(
                "High-Risk Transactions",
                f"{n_high:,}",
                f"{n_high / n_total * 100:.2f}% of scored"
                if n_total else None,
                delta_color="off"
            )
        if "isFraud" in risk_scores.columns:
            total_fraud = int(risk_scores["isFraud"].sum())
            caught = int(
                risk_scores.loc[
                    risk_scores["risk_level"] == "High", "isFraud"
                ].sum()
            )
            with r3:
                st.metric("Actual Fraud Cases", f"{total_fraud:,}")
            with r4:
                share = caught / total_fraud * 100 if total_fraud else 0
                st.metric("Fraud Found in High Group", f"{share:.1f}%")

    st.subheader("Key Finding")

    strongest_rule = rules.loc[rules["lift"].idxmax()]

    st.success(
        f"The strongest fraud-associated pattern in the analysed "
        f"dataset was **{strongest_rule['rule']}**, with a confidence "
        f"of **{strongest_rule['confidence'] * 100:.2f}%** and a lift "
        f"of **{strongest_rule['lift']:.2f}**."
    )

    st.subheader("What This Means for Fraud Analytics")

    st.write(
        "The discovered rules identify combinations of transaction "
        "characteristics that occur disproportionately often among "
        "transactions carrying the fraud label."
    )
    st.write(
        "These patterns can support fraud analysts by highlighting "
        "behavioural combinations that may deserve additional review."
    )
    st.write(
        "For example, combinations involving very high balance changes "
        "and specific simulation-time periods showed substantially "
        "higher association with the fraud label."
    )

    st.subheader("⚠️ Important Limitations")

    st.info(
        "The dataset used in this project is synthetic. Association "
        "rules represent statistical relationships and do not prove "
        "that a transaction is fraudulent or that one behaviour causes "
        "fraud. The discovered patterns should therefore be treated as "
        "analytical signals rather than standalone fraud decisions."
    )


# ============================================================
# TAB 2 — ASSOCIATION RULES
# ============================================================

with tab2:

    # ---------------- Top patterns ----------------
    st.subheader("Top Fraud-Associated Patterns")

    top_n = 8

    top_rules = rules.sort_values(
        by="lift", ascending=False
    ).head(top_n)

    show_df(top_rules[["rule", "support", "confidence", "lift"]])

    st.subheader("Fraud-Associated Patterns by Lift")
    st.bar_chart(top_rules.set_index("rule")["lift"])

    # ---------------- Rule explorer ----------------
    st.subheader("Fraud Rule Explorer")

    rule_search = st.text_input(
        "Filter the rule list (optional)", key="rule_search"
    )
    rule_options = rules["rule"].astype(str)
    if rule_search:
        rule_options = rule_options[
            rule_options.str.contains(rule_search, case=False, na=False)
        ]

    if rule_options.empty:
        st.warning("No rules match that text.")
    else:
        selected_rule = st.selectbox(
            "Select a fraud-associated pattern:",
            rule_options.tolist()
        )
        selected = rules[
            rules["rule"].astype(str) == selected_rule
        ].iloc[0]

        support = selected["support"]
        confidence = selected["confidence"]
        lift = selected["lift"]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Support", f"{support:.4f}")
        with m2:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        with m3:
            st.metric("Lift", f"{lift:.2f}")

        st.subheader("Rule Interpretation")
        st.write(
            f"**Support:** This pattern occurs in approximately "
            f"**{support * 100:.4f}%** of the analysed transactions."
        )
        st.write(
            f"**Confidence:** When this pattern occurs, approximately "
            f"**{confidence * 100:.2f}%** of those transactions are "
            f"associated with the fraud label."
        )
        st.write(
            f"**Lift:** A lift of **{lift:.2f}** indicates that this "
            f"pattern is associated with the fraud label substantially "
            f"more often than the overall fraud baseline in this "
            f"dataset."
        )

    # ========================================================
    # FRAUD-FOCUSED RULE DISCOVERY
    # ========================================================
    st.divider()
    st.subheader("Fraud-Focused Rule Discovery")

    if focused_rules is None:
        st.warning(
            "fraud_focused_rules.csv was not found in the outputs "
            "folder. Run the fraud-focused rule step in the notebook "
            "to enable this section."
        )
    else:
        fr = focused_rules.copy()

        rule_col = find_col(fr, ["rule", "pattern", "antecedents"])
        fraud_count_col = find_col(
            fr, ["fraud_count", "fraud_cases", "fraud"]
        )
        lift_col = find_col(fr, ["lift"])

        if lift_col is not None:
            fr[lift_col] = pd.to_numeric(fr[lift_col], errors="coerce")
        if fraud_count_col is not None:
            fr[fraud_count_col] = pd.to_numeric(
                fr[fraud_count_col], errors="coerce"
            )

        if lift_col is not None:
            fr = fr.sort_values(by=lift_col, ascending=False)

        st.write(f"**{len(fr):,}** fraud-focused patterns shown.")
        show_df(fr.head(50))

        if rule_col is not None and lift_col is not None and len(fr):
            st.bar_chart(fr.head(10).set_index(rule_col)[lift_col])

    # ========================================================
    # ADVANCED FRAUD-FOCUSED PATTERN ANALYSIS
    # ========================================================
    st.divider()
    st.subheader("Advanced Fraud-Focused Pattern Analysis")

    st.write(
        "This enhancement extends the original FP-Growth project "
        "with targeted fraud-pattern discovery and rule-based "
        "transaction risk scoring."
    )

    st.markdown("#### What is a fraud association rule?")
    st.markdown(
        "A rule such as "
        "`BALANCE_CHANGE_VERY_HIGH + TIME_NIGHT → FRAUD` means that "
        "transactions matching both conditions were labelled fraudulent "
        "more often than the overall fraud baseline in the dataset."
    )
    st.markdown(
        "- **Support:** fraction of all transactions that satisfy the "
        "whole rule, including the fraud label.\n"
        "- **Confidence:** fraction of transactions matching the "
        "conditions that are labelled fraudulent.\n"
        "- **Lift:** how many times the rule's confidence exceeds the "
        "overall fraud rate (lift > 1 means positive association)."
    )

    st.markdown("#### How does the enhancement work?")
    st.markdown(
        "1. **Feature combination discovery:** transactions are grouped "
        "by combinations of transaction type, amount category, "
        "balance-change category and time category.\n"
        "2. **Fraud-focused filtering:** combinations with at least "
        "five fraud cases in the training data are kept.\n"
        "3. **Rule ranking:** candidate patterns are ranked by fraud "
        "confidence and by lift relative to the training-set fraud "
        "rate.\n"
        "4. **Risk scoring:** the discovered patterns contribute to a "
        "heuristic score from 0 to 100, helping prioritise "
        "transactions for review."
    )

    st.markdown("#### Standard FP-Growth vs. our enhancement")

    comparison = pd.DataFrame({
        "Aspect": [
            "Mining focus",
            "Rule selection",
            "Output",
            "Purpose",
        ],
        "Standard FP-Growth (original)": [
            "All frequent itemsets and association rules",
            "Minimum support, then confidence and lift",
            "Association rules",
            "Discover recurring relationships",
        ],
        "Fraud-focused enhancement (this project)": [
            "Feature combinations associated with fraud",
            "Minimum fraud count, confidence and lift against the "
            "training-set fraud rate",
            "Fraud-focused rules and transaction risk scores",
            "Prioritise transactions for fraud review",
        ],
    })
    show_df(comparison)

    st.warning(
        "Methodological note: this is an enhancement built on top of "
        "the FP-Growth results, not a new implementation of an advanced "
        "FP-Growth algorithm such as Guided FP-Growth or CFP-Growth. "
        "The risk score is a heuristic and not a trained classifier."
    )


# ============================================================
# TAB 3 — RISK SCORING
# ============================================================

with tab3:

    st.subheader("Fraud Risk Scoring")

    if risk_scores is None or risk_scores.empty:
        st.warning(
            "Risk scores are not available. Run the notebook's "
            "risk-scoring step and save fraud_risk_scores.csv inside "
            "the outputs folder. The sidebar shows which files were "
            "found."
        )
    else:
        has_label = "isFraud" in risk_scores.columns

        with st.expander("What is the risk score?"):
            st.write(
                "A heuristic from 0 to 100 built from the discovered "
                "fraud-focused patterns. A higher score means the "
                "transaction resembles known fraud patterns more "
                "closely. It is used to rank transactions for review "
                "and is not proof of fraud."
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Scored Transactions", f"{len(risk_scores):,}")
        with m2:
            st.metric(
                "Average Risk Score",
                f"{risk_scores['risk_score'].mean():.1f}"
            )
        with m3:
            st.metric(
                "High-Risk Transactions",
                f"{int((risk_scores['risk_level'] == 'High').sum()):,}"
            )

        counts = (
            risk_scores["risk_level"].value_counts()
            .reindex(LEVEL_ORDER).fillna(0)
        )

        left, right = st.columns(2)
        with left:
            st.markdown("**Transactions by risk level**")
            st.bar_chart(counts)

        with right:
            if has_label:
                st.markdown("**Actual fraud rate by risk level (%)**")
                fraud_rate = (
                    risk_scores.groupby("risk_level")["isFraud"].mean()
                    .reindex(LEVEL_ORDER).fillna(0) * 100
                )
                st.bar_chart(fraud_rate)
            else:
                st.info(
                    "The isFraud column is missing, so actual fraud "
                    "rates cannot be shown."
                )

        st.markdown("**Risk score distribution**")
        hist, edges = np.histogram(
            risk_scores["risk_score"], bins=20, range=(0, 100)
        )
        st.bar_chart(pd.DataFrame(
            {"transactions": hist},
            index=[f"{int(e):02d}-{int(e + 5)}" for e in edges[:-1]]
        ))

        # ---------------- Review budget ----------------
        if has_label and risk_scores["isFraud"].sum() > 0:
            st.divider()
            st.subheader("Review Budget: Checking Only the Top-Scored Transactions")
            st.write(
                "If analysts review only the highest-scoring "
                "transactions, this shows how much fraud they would "
                "catch."
            )

            ordered_all = risk_scores.sort_values(
                "risk_score", ascending=False
            )
            total_fraud = int(risk_scores["isFraud"].sum())

            budget_rows = []
            for n in [100, 500, 1000, 5000]:
                if n > len(ordered_all):
                    continue
                caught = int(ordered_all.head(n)["isFraud"].sum())
                budget_rows.append({
                    "Transactions reviewed": n,
                    "Fraud cases caught": caught,
                    "Precision (%)": round(caught / n * 100, 1),
                    "Share of all fraud found (%)": round(
                        caught / total_fraud * 100, 1
                    ),
                })
            if budget_rows:
                show_df(pd.DataFrame(budget_rows))

        for note in score_notes:
            st.caption(note)


# ============================================================
# TAB 4 — HIGH-RISK TRANSACTION EXPLORER
# ============================================================

with tab4:

    st.subheader("High-Risk Transaction Explorer")

    source = risk_scores if risk_scores is not None else high_risk_file

    if source is None or source.empty:
        st.warning(
            "No transaction-level results found. Save "
            "fraud_risk_scores.csv in the outputs folder."
        )
    else:
        data = source.copy()
        has_label = "isFraud" in data.columns

        f1, f2 = st.columns(2)

        with f1:
            levels = [l for l in LEVEL_ORDER if l in data["risk_level"].unique()]
            default_levels = ["High"] if "High" in levels else levels
            chosen_levels = st.multiselect(
                "Risk level", levels, default=default_levels
            )
            data = data[data["risk_level"].isin(chosen_levels)]

        with f2:
            if "type" in data.columns:
                types = sorted(data["type"].astype(str).unique())
                chosen_types = st.multiselect(
                    "Transaction type", types, default=types
                )
                data = data[data["type"].astype(str).isin(chosen_types)]

        if has_label:
            outcome = st.radio(
                "Outcome (compares the score with the actual label)",
                [
                    "All flagged transactions",
                    "Caught fraud (actual label = 1)",
                    "False alarms (actual label = 0)",
                ],
                horizontal=True
            )
            if outcome.startswith("Caught"):
                data = data[data["isFraud"] == 1]
            elif outcome.startswith("False"):
                data = data[data["isFraud"] == 0]

        search = st.text_input("Search any column (text match)")
        if search:
            mask = data.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            data = data[mask]

        n_rows = 100

        summary = f"**{len(data):,}** transactions match the filters."
        if has_label and len(data):
            summary += (
                f" Of these, **{int(data['isFraud'].sum()):,}** are "
                f"actual fraud ({data['isFraud'].mean() * 100:.1f}%)."
            )
        st.write(summary)

        display = data.sort_values("risk_score", ascending=False).head(n_rows)
        if has_label:
            display = display.rename(
                columns={"isFraud": "isFraud (actual label)"}
            )
        show_df(display)

        st.download_button(
            "Download filtered results (CSV)",
            data.to_csv(index=False).encode("utf-8"),
            file_name="filtered_high_risk_transactions.csv",
            mime="text/csv"
        )

        st.caption(
            "The risk score never reads the actual label, so a flagged "
            "table naturally contains both real fraud (1) and false "
            "alarms (0)."
        )


# ============================================================
# TAB 5 — EVALUATION METRICS
# ============================================================

with tab5:

    st.subheader("Evaluation Metrics")

    if (
        risk_scores is None
        or "isFraud" not in risk_scores.columns
    ):
        st.warning(
            "Evaluation requires fraud_risk_scores.csv with "
            "risk_score and isFraud columns."
        )
    else:
        y_true = risk_scores["isFraud"].astype(int).to_numpy()
        scores = risk_scores["risk_score"].to_numpy()

        if y_true.sum() == 0:
            st.error(
                "The scored data contains no fraud cases, so "
                "precision, recall and average precision cannot be "
                "computed."
            )
        elif y_true.sum() == len(y_true):
            st.error("Every scored transaction is fraud; evaluation "
                     "needs both classes.")
        else:
            try:
                from sklearn.metrics import (
                    average_precision_score,
                    precision_recall_curve,
                )
                from sklearn.model_selection import train_test_split
            except ImportError:
                st.error("scikit-learn is required: pip install "
                         "scikit-learn")
                st.stop()

            calibration_idx, evaluation_idx = train_test_split(
                np.arange(len(y_true)),
                test_size=0.5,
                random_state=42,
                stratify=y_true,
            )
            calibration_y = y_true[calibration_idx]
            calibration_scores = scores[calibration_idx]
            evaluation_y = y_true[evaluation_idx]
            evaluation_scores = scores[evaluation_idx]
            threshold, calibration_metrics = best_threshold_for_f1(
                calibration_y, calibration_scores
            )

            res = metrics_at(evaluation_y, evaluation_scores, threshold)
            ap = average_precision_score(evaluation_y, evaluation_scores)
            st.caption(
                f"Threshold {threshold:.2f} was selected on a stratified "
                f"calibration split (F1={calibration_metrics['f1']:.3f}) "
                "and evaluated on separate held-out rows. The split is "
                "fixed so reruns are reproducible."
            )

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Precision", f"{res['precision']:.3f}")
            with e2:
                st.metric("Recall", f"{res['recall']:.3f}")
            with e3:
                st.metric("F1-score", f"{res['f1']:.3f}")
            with e4:
                st.metric("Average Precision", f"{ap:.3f}")

            if res["flagged"] == 0:
                st.info(
                    "No transactions reach this threshold, so "
                    "precision is shown as 0. Lower the threshold."
                )

            st.markdown("**Confusion matrix at this threshold**")
            show_df(pd.DataFrame(
                [[res["tn"], res["fp"]], [res["fn"], res["tp"]]],
                index=["Actual: Not fraud", "Actual: Fraud"],
                columns=["Predicted: Not fraud", "Predicted: Fraud"]
            ).reset_index().rename(columns={"index": ""}))

            # ---------------- Threshold table ----------------
            st.markdown("**How the metrics change with the threshold**")
            rows = []
            for t in range(0, 101, 10):
                r = metrics_at(calibration_y, calibration_scores, t)
                rows.append({
                    "Threshold": t,
                    "Flagged": r["flagged"],
                    "Precision": round(r["precision"], 3),
                    "Recall": round(r["recall"], 3),
                    "F1": round(r["f1"], 3),
                })
            table = pd.DataFrame(rows)
            show_df(table)

            st.caption("Threshold comparison above uses calibration rows only.")

            # ---------------- PR curve ----------------
            st.markdown("**Precision–recall curve**")
            p_curve, r_curve, _ = precision_recall_curve(
                evaluation_y, evaluation_scores
            )
            curve = pd.DataFrame({"recall": r_curve, "precision": p_curve})
            if len(curve) > 500:
                curve = curve.iloc[
                    np.linspace(0, len(curve) - 1, 500).astype(int)
                ]
            st.line_chart(curve.set_index("recall"))

            baseline = evaluation_y.mean()
            st.caption(
                f"Baseline fraud rate in the scored data: "
                f"{baseline * 100:.3f}%. Compare average precision with "
                f"this baseline: fraud is highly imbalanced, so "
                f"accuracy would be misleading."
            )


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander("About this project"):

    st.write("**Algorithm:** FP-Growth, plus fraud-focused pattern mining")
    st.write("**Dataset:** PaySim synthetic financial transaction dataset")
    st.write(
        "**Analysis:** Support, Confidence, Lift, heuristic risk "
        "scoring, Precision, Recall, F1 and Average Precision"
    )
    st.write(
        "**Purpose:** Discover recurring transaction patterns "
        "associated with financial fraud and prioritise transactions "
        "for review."
    )
    st.write(
        "**Technology:** Python, Pandas, mlxtend, scikit-learn and "
        "Streamlit"
    )


# ============================================================
# GITHUB LINK
# ============================================================

st.markdown("---")

st.markdown(
    "[🔗 View the project source code on GitHub]"
    "(https://github.com/nidhitshah11-06/financial-fraud-arm)"
)