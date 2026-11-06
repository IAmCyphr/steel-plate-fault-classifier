"""Streamlit app: scores saved models on an uploaded test CSV."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

import theme
from charts import class_counts_figure, confusion_figure, metric_bars_figure
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
MODEL_NOTES = {
    "Logistic Regression": "Linear model on standardised features.",
    "Decision Tree": "Depth-limited tree (max depth 8) on raw features.",
    "KNN": "7 nearest neighbours on standardised features.",
    "Naive Bayes": "Gaussian Naive Bayes, assumes independent features.",
    "Support Vector Machine": "RBF kernel with probability estimates, standardised features.",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
RANK_WORDS = {1: "Best", 2: "2nd", 3: "3rd"}

st.set_page_config(page_title="Steel Plate Fault Inspector", layout="wide")
theme.apply()


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data(show_spinner=False)
def evaluate(name, data):
    model = load_model(name)
    X, y = data.drop(columns=TARGET), data[TARGET]
    X = X[list(model.feature_names_in_)]
    pred, proba = model.predict(X), model.predict_proba(X)
    scores = compute_metrics(y, pred, proba, model.classes_)
    return y, pred, scores, list(model.classes_)


def rank_label(table, metric, name):
    rank = int(table[metric].rank(ascending=False, method="min")[name])
    return f"{RANK_WORDS.get(rank, f'{rank}th')} of {len(table)}", rank == 1


def metric_cards(table, name):
    cards = []
    for metric in METRICS:
        text, leads = rank_label(table, metric, name)
        css = "metric-rank lead" if leads else "metric-rank"
        cards.append(
            f'<div class="metric-card"><div class="metric-label">{metric}</div>'
            f'<div class="metric-value">{table.loc[name, metric]:.4f}</div>'
            f'<div class="{css}">{text}</div></div>'
        )
    st.markdown(f'<div class="metric-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def section(label, note=None):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def read_input():
    st.sidebar.markdown('<div class="section-label">Test data</div>', unsafe_allow_html=True)
    upload = st.sidebar.file_uploader("Upload test CSV", type="csv", label_visibility="collapsed")
    use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=upload is None)
    if upload is not None:
        return pd.read_csv(upload), upload.name
    if use_sample and SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE), SAMPLE_FILE.name
    return None, None


def check_columns(data):
    expected = set(load_model("KNN").feature_names_in_) | {TARGET}
    missing = sorted(expected - set(data.columns))
    if missing:
        st.error(f"The file is missing required columns: {', '.join(missing)}")
        return False
    if data[list(expected)].isna().any().any():
        st.error("The file has empty cells. Remove or fill them and upload again.")
        return False
    return True


def header(data, source):
    st.markdown('<h1 class="page-title">Steel Plate Fault Inspector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Pre-trained classifiers that predict one of seven surface fault types '
        "for a steel plate from 27 sensor and geometry readings. Choose a model, and the page "
        "scores it on your test file without retraining.</p>",
        unsafe_allow_html=True,
    )
    if data is None:
        return
    facts = [
        ("Source", source),
        ("Rows", f"{len(data):,}"),
        ("Features", data.shape[1] - 1),
        ("Fault classes", data[TARGET].nunique()),
    ]
    chips = "".join(f'<span class="fact">{k} <b>{v}</b></span>' for k, v in facts)
    st.markdown(f'<div class="fact-row">{chips}</div>', unsafe_allow_html=True)


def evaluation_tab(data, choice, table):
    y, pred, _, classes = evaluate(choice, data)
    metric_cards(table, choice)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        section("Confusion matrix")
        view = st.radio("Cell values", ["Counts", "Row %"], horizontal=True, label_visibility="collapsed")
        st.pyplot(confusion_figure(y, pred, classes, view == "Row %"), width="stretch")
    with right:
        section("Per-class report", "Precision, recall and F1 for each fault type.")
        report = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
        report.pop("accuracy", None)
        frame = pd.DataFrame(report).T.rename(columns={"f1-score": "f1"})
        frame["support"] = frame["support"].astype(int)
        frame.index = [name.replace("_", " ") for name in frame.index]
        st.dataframe(
            frame.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
            width="stretch",
            height=330,
        )
        out = data.assign(Predicted=pred)
        st.download_button(
            "Download predictions (CSV)", out.to_csv(index=False), f"predictions_{choice}.csv", "text/csv"
        )


def comparison_tab(table, choice):
    section("Compare models", "Same test file, every saved model. The selected model is shaded darker.")
    metric = st.segmented_control("Metric", METRICS, default="Accuracy", label_visibility="collapsed")
    metric = metric or "Accuracy"
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.pyplot(metric_bars_figure(table, metric, choice), width="stretch")
    with right:
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#cde2fb"),
            width="stretch",
        )
        st.caption(
            "Precision, recall and F1 are macro averages over the seven classes. "
            "AUC is one-vs-rest macro. Highlighted cells are the best score per metric."
        )


def data_tab(data):
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        section("Class balance")
        counts = data[TARGET].value_counts()
        st.pyplot(class_counts_figure(list(counts.index), list(counts.values)), width="stretch")
    with right:
        section("Preview", "First 100 rows of the loaded file.")
        st.dataframe(data.head(100), width="stretch", height=340)
    section("Feature summary")
    st.dataframe(data.drop(columns=TARGET).describe().T.round(3), width="stretch")


def main():
    data, source = read_input()
    header(data, source)
    if data is None:
        st.info("Upload a test CSV in the sidebar. It needs the 27 feature columns and a Class column.")
        return
    if not check_columns(data):
        return

    st.sidebar.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    choice = st.sidebar.selectbox("Classifier", list(MODEL_FILES), label_visibility="collapsed")
    st.sidebar.markdown(f'<div class="model-note">{MODEL_NOTES[choice]}</div>', unsafe_allow_html=True)

    table = pd.DataFrame({name: evaluate(name, data)[2] for name in MODEL_FILES}).T
    table.index.name = "ML Model Name"

    evaluation, comparison, dataset = st.tabs(["Evaluation", "Model comparison", "Data"])
    with evaluation:
        evaluation_tab(data, choice, table)
    with comparison:
        comparison_tab(table, choice)
    with dataset:
        data_tab(data)


main()
