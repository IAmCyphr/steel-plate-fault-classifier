"""Shared metric computation, used by both train.py and the Streamlit app."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROUND_TO = 4


def one_vs_rest_auc(y_true, y_proba, classes, present):
    """Macro AUC over the classes that occur in y_true; NaN when it cannot be computed."""
    scores = []
    for label in present:
        is_label = y_true == label
        if is_label.all():
            continue
        scores.append(roc_auc_score(is_label, y_proba[:, classes.index(label)]))
    return float(np.mean(scores)) if scores else float("nan")


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Multi-class metrics. Precision, recall and F1 are macro averages and AUC is
    one-vs-rest macro, both over the classes present in y_true."""
    y_true = np.asarray(y_true)
    classes = list(classes)
    present = [c for c in classes if (y_true == c).any()]
    macro = {"average": "macro", "labels": present, "zero_division": 0}
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": one_vs_rest_auc(y_true, y_proba, classes, present),
        "Precision": precision_score(y_true, y_pred, **macro),
        "Recall": recall_score(y_true, y_pred, **macro),
        "F1": f1_score(y_true, y_pred, **macro),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return {k: round(float(v), ROUND_TO) for k, v in scores.items()}
