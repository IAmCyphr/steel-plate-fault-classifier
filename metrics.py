"""Shared metric computation, used by both train.py and the Streamlit app."""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROUND_TO = 4


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Multi-class metrics. Precision/Recall/F1 are macro-averaged; AUC is one-vs-rest macro."""
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=list(classes)
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return {k: round(float(v), ROUND_TO) for k, v in scores.items()}
