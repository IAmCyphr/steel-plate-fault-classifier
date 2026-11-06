"""Trains the five classifiers and saves them to model/."""
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics import compute_metrics

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "steel_plates_faults.csv"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
TARGET = "Class"
SEED = 42
TEST_FRACTION = 0.20


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=5, probability=True, random_state=SEED)
        ),
    }


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = pd.read_csv(RAW_FILE).drop_duplicates().reset_index(drop=True)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.concat([X_te, y_te], axis=1).to_csv(ROOT / "test_data.csv", index=False)
    print(f"rows={len(df)}  train={len(X_tr)}  test={len(X_te)}")

    rows = []
    for name, model in build_models().items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        joblib.dump(model, MODEL_DIR / file_name(name))
        scores = compute_metrics(y_te, model.predict(X_te), model.predict_proba(X_te), model.classes_)
        rows.append({"ML Model Name": name, **scores, "Train time (s)": round(fit_s, 2)})
        print(f"{name:24s} fit {fit_s:6.2f}s  acc={scores['Accuracy']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
