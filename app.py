"""Streamlit app: scores saved models on an uploaded test CSV."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics import compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_FILE = ROOT / "test_data.csv"
TARGET = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Steel Plate Fault Inspector", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def evaluate(name, data):
    model = load_model(name)
    X, y = data.drop(columns=TARGET), data[TARGET]
    X = X[list(model.feature_names_in_)]
    pred, proba = model.predict(X), model.predict_proba(X)
    return y, pred, compute_metrics(y, pred, proba, model.classes_), list(model.classes_)


def read_input():
    st.sidebar.header("Test data")
    upload = st.sidebar.file_uploader("Upload test CSV", type="csv")
    use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=upload is None)
    if upload is not None:
        return pd.read_csv(upload)
    if use_sample and SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE)
    return None


def check_columns(data):
    expected = set(load_model("KNN").feature_names_in_) | {TARGET}
    missing = sorted(expected - set(data.columns))
    if missing:
        st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
        return False
    if data[list(expected)].isna().any().any():
        st.error("Uploaded file contains empty cells; please remove or fill them.")
        return False
    return True


def draw_confusion(y, pred, classes):
    cm = confusion_matrix(y, pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def main():
    st.title("Steel Plate Fault Inspector")
    st.caption(
        "Seven kinds of surface fault, 27 sensor and geometry readings per plate. "
        "Pick a saved model and see how it scores on your test CSV."
    )
    data = read_input()
    if data is None:
        st.info("Upload a test CSV (features + `Class` column) in the sidebar to begin.")
        return
    if not check_columns(data):
        return

    st.sidebar.header("Model")
    choice = st.sidebar.selectbox("Classifier", list(MODEL_FILES))
    st.sidebar.success(f"{len(data):,} rows loaded")

    y, pred, scores, classes = evaluate(choice, data)

    st.subheader(f"{choice}: evaluation metrics")
    for col, (label, value) in zip(st.columns(len(scores)), scores.items()):
        col.metric(label, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(draw_confusion(y, pred, classes))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

    st.subheader("All models on this data")
    table = pd.DataFrame(
        [{"ML Model Name": name, **evaluate(name, data)[2]} for name in MODEL_FILES]
    ).set_index("ML Model Name")
    st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf"), width="stretch")


main()
