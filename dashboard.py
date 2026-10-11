
import ast
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud Risk Prioritization",
    page_icon="🔍",
    layout="wide"
)

st.title("Fraud-Guided Risk Prioritization")
st.write("Identify suspicious digital payment transactions using association rules.")

# -------------------------------
# 1. Upload transaction dataset
# -------------------------------
data_file = st.file_uploader(
    "Upload PaySim dataset (CSV)",
    type=["csv"]
)

if data_file is None:
    st.info("Upload your PaySim CSV file to continue.")
    st.stop()

df = pd.read_csv(data_file)

required_columns = [
    "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "step", "isFraud"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# -------------------------------
# 2. Prepare transaction features
# -------------------------------
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

df["balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["amount_category"] = pd.cut(
    df["amount"],
    bins=[-np.inf, 10_000, 100_000, 1_000_000, np.inf],
    labels=["low", "medium", "high", "very_high"],
    include_lowest=True
).astype("string").fillna("unknown")

df["balance_change_category"] = pd.cut(
    df["balance_change"],
    bins=[-np.inf, 0, 10_000, 100_000, 1_000_000, np.inf],
    labels=[
        "negative_or_zero", "small", "medium",
        "large", "very_large"
    ],
    include_lowest=True
).astype("string").fillna("unknown")

df["time_category"] = pd.cut(
    df["step"],
    bins=[-np.inf, 24, 96, 168, np.inf],
    labels=["early", "mid", "late", "very_late"],
    include_lowest=True
).astype("string").fillna("unknown")

# -------------------------------
# 3. Main transaction summary
# -------------------------------
total = len(df)
frauds = int(df["isFraud"].sum())
legitimate = total - frauds

c1, c2, c3 = st.columns(3)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Transactions", f"{frauds:,}")
c3.metric(
    "Fraud Rate",
    f"{frauds / total:.3%}" if total else "0.000%"
)

# -------------------------------
# 4. Upload trained fraud rules
# -------------------------------
st.divider()
st.header("Transaction Risk Scoring")

rules_file = st.file_uploader(
    "Upload fraud_rules_train.csv",
    type=["csv"]
)

def parse_antecedent(value):
    raw = value if isinstance(value, list) else ast.literal_eval(str(value))

    return [
        (str(item[0]), str(item[1]))
        for item in raw
        if isinstance(item, (list, tuple)) and len(item) == 2
    ]

def score_transactions(data, rules):
    scores = np.zeros(len(data), dtype=float)

    for _, rule in rules.iterrows():
        try:
            pairs = parse_antecedent(rule["antecedent"])
            weight = np.log1p(max(float(rule["lift"]), 0.0))

            mask = np.ones(len(data), dtype=bool)

            for column, value in pairs:
                if column not in data.columns:
                    mask[:] = False
                    break

                mask &= (
                    data[column].astype(str).eq(value).to_numpy()
                )

            scores[mask] += weight

        except (ValueError, SyntaxError, TypeError, KeyError):
            continue

    return scores

if rules_file is not None:
    try:
        rules = pd.read_csv(rules_file)

        if not {"antecedent", "lift"}.issubset(rules.columns):
            st.error(
                "The rules CSV must contain 'antecedent' and 'lift' columns."
            )
            st.stop()

        df["risk_score"] = score_transactions(df, rules)

        threshold = st.number_input(
            "Risk score threshold",
            min_value=0.0,
            value=17.98,
            step=0.5
        )

        df["risk_flag"] = df["risk_score"] >= threshold

        flagged = df[df["risk_flag"]]
        flagged_frauds = int((flagged["isFraud"] == 1).sum())
        flagged_legitimate = int((flagged["isFraud"] == 0).sum())

        c1, c2, c3 = st.columns(3)

        c1.metric("Flagged Transactions", f"{len(flagged):,}")
        c2.metric("Flagged Frauds", f"{flagged_frauds:,}")
        c3.metric("Flagged Legitimate", f"{flagged_legitimate:,}")

        st.subheader("Highest-Risk Transactions")

        display_columns = [
            "step", "type", "amount",
            "amount_category", "balance_change_category",
            "time_category", "isFraud",
            "risk_score", "risk_flag"
        ]

        st.dataframe(
            flagged.sort_values(
                "risk_score", ascending=False
            )[display_columns].head(100),
            use_container_width=True
        )

        st.download_button(
            "Download Risk Results",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="fraud_risk_results.csv",
            mime="text/csv"
        )

    except Exception as exc:
        st.error(f"Could not process the rules file: {exc}")

else:
    st.info("Upload your trained fraud rules CSV to calculate risk scores.")

st.caption(
    "Research prototype: risk flags prioritize transactions for review "
    "and do not prove that a transaction is fraudulent."
)
