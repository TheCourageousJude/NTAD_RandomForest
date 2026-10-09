"""Streamlit app: Network Traffic Anomaly Detection Using Random Forest
CCE105 - University of Mindanao, College of Computing Education
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
DATA = BASE / "networkanomalydataset_2000.csv"
FEATURES = [
    "Inbound Rate(bit/s)",
    "Outbound Rate(bit/s)",
    "Inbound Bandwidth Utilization(%)",
    "Outbound Bandwidth Utilization(%)",
]

st.set_page_config(page_title="Network Anomaly Detector", page_icon="🛡️", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def train(n_estimators: int, max_depth):
    df = load_data()
    X, y = df[FEATURES], df["Label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    ).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1-Score": f1_score(y_te, pred),
    }
    return model, metrics, confusion_matrix(y_te, pred)


df = load_data()

# ---------------- Sidebar ----------------
st.sidebar.header("Model configuration")
choice = st.sidebar.radio(
    "Random Forest version",
    ["Tuned (GridSearchCV): 25 trees, depth 3", "Baseline: 100 trees, no depth limit"],
)
if choice.startswith("Tuned"):
    n_est, depth = 25, 3
else:
    n_est, depth = 100, None
model, metrics, cm = train(n_est, depth)
st.sidebar.caption("Trained on 1,600 records (80% stratified split), tested on 400.")
st.sidebar.markdown("**Authors**  \nJude Emmanuel Corage  \nNoel Anthony Mamitez  \n\nCCE105, University of Mindanao")

# ---------------- Header ----------------
st.title("🛡️ Network Traffic Anomaly Detection")
st.write("A Random Forest classifier that labels network traffic as **Normal** or **Anomaly** "
         "from four standardized traffic measurements.")

tab_pred, tab_batch, tab_perf, tab_data = st.tabs(
    ["Single prediction", "Batch (CSV)", "Model performance", "Dataset & figures"]
)

# ---------------- Single prediction ----------------
with tab_pred:
    st.subheader("Enter a traffic record")
    st.caption("Values are standardized (z-scores), matching the dataset. Typical range is about -2 to +3.")
    c1, c2 = st.columns(2)
    vals = {}
    for i, f in enumerate(FEATURES):
        col = c1 if i % 2 == 0 else c2
        vals[f] = col.slider(f, -3.0, 4.0, 0.0, 0.05)

    if st.button("Classify traffic", type="primary"):
        row = pd.DataFrame([vals])[FEATURES]
        pred = int(model.predict(row)[0])
        proba = model.predict_proba(row)[0]
        if pred == 1:
            st.error(f"🚨 ANOMALY detected (confidence {proba[1]:.1%})")
        else:
            st.success(f"✅ Normal traffic (confidence {proba[0]:.1%})")
        st.progress(float(proba[1]), text=f"Anomaly probability: {proba[1]:.1%}")

    with st.expander("Try an example from the dataset"):
        ex_label = st.selectbox("Example type", ["Normal (0)", "Anomaly (1)"])
        lbl = 0 if ex_label.startswith("Normal") else 1
        ex = df[df["Label"] == lbl].sample(1, random_state=7)[FEATURES]
        st.dataframe(ex, hide_index=True)
        st.write("Model says:", "**Anomaly**" if model.predict(ex)[0] == 1 else "**Normal**")

# ---------------- Batch ----------------
with tab_batch:
    st.subheader("Classify many records at once")
    st.write("Upload a CSV with these columns (a `Label` column is optional):")
    st.code(", ".join(FEATURES))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            new = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in new.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                out = new.copy()
                out["Predicted"] = model.predict(new[FEATURES])
                out["Predicted"] = out["Predicted"].map({0: "Normal", 1: "Anomaly"})
                out["Anomaly probability"] = model.predict_proba(new[FEATURES])[:, 1].round(3)
                st.write(f"**{(out['Predicted'] == 'Anomaly').sum()}** of **{len(out)}** records flagged as anomalous.")
                st.dataframe(out, width="stretch")
                st.download_button("Download results", out.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read that file: {e}")

# ---------------- Performance ----------------
with tab_perf:
    st.subheader(f"Test-set results ({choice.split(':')[0]})")
    cols = st.columns(4)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.2%}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        st.dataframe(
            pd.DataFrame(cm, index=["Actual Normal", "Actual Anomaly"],
                         columns=["Predicted Normal", "Predicted Anomaly"])
        )
        st.caption(f"Missed anomalies (false negatives): {cm[1][0]} of {cm[1].sum()}")
    with right:
        st.markdown("**Feature importance**")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        st.bar_chart(imp)

    for img, cap in [("gridsearch_heatmap.png", "GridSearchCV results"),
                     ("learning_curve_before_after.png", "Learning curve before vs. after tuning")]:
        if (BASE / img).exists():
            st.image(str(BASE / img), caption=cap, width="stretch")

# ---------------- Dataset ----------------
with tab_data:
    st.subheader("Dataset")
    st.write(f"{len(df):,} records ({(df.Label == 0).sum()} Normal / {(df.Label == 1).sum()} Anomaly). "
             "Base: Kaggle Network Anomaly Dataset (1,654 records), expanded with class-wise synthetic augmentation.")
    st.dataframe(df.head(50), width="stretch")
    gal = [("label_distribution.png", "Class distribution"),
           ("scatter_top_features.png", "Inbound Rate vs. Inbound Bandwidth Utilization"),
           ("boxplots_by_class.png", "Feature distributions by class"),
           ("decision_boundary.png", "Decision boundary"),
           ("pipeline_diagram.png", "Pipeline")]
    for i in range(0, len(gal), 2):
        cols = st.columns(2)
        for col, (img, cap) in zip(cols, gal[i:i + 2]):
            if (BASE / img).exists():
                col.image(str(BASE / img), caption=cap, width="stretch")
