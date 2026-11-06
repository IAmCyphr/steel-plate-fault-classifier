# Steel Plate Fault Inspector

Multi-class classification of steel plate surface faults with five classical ML models, evaluated through a Streamlit app.

- **GitHub repository:** https://github.com/IAmCyphr/steel-plate-fault-classifier
- **Live app:** https://steel-plate-fault-classifier-gz6crdafengjsxp8qyzmx4.streamlit.app/

## a. Problem statement

A steel mill inspects plates after rolling. Each plate is described by 27 numeric readings (geometry of the defect region, luminosity, steel type and thickness, and derived shape indices). The task is to predict which of **seven fault types** a plate has, so defective plates can be routed to the right corrective process. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same split and compared on six metrics.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, *Steel Plates Faults* (Research Center of Sciences of Communication, Rome) |
| Instances | 1,941 (no missing values, no duplicates) |
| Features | 27 numeric (e.g. `X_Minimum`, `Pixels_Areas`, `Sum_of_Luminosity`, `Steel_Plate_Thickness`, `Edges_Index`, `LogOfAreas`, `SigmoidOfAreas`) |
| Target | `Class`, one of 7 faults. The original 7 one-hot columns were merged into one label (exactly one is set per row) |
| Class counts | Other_Faults 673, Bumps 402, K_Scatch 391, Z_Scratch 190, Pastry 158, Stains 72, Dirtiness 55 |

Preprocessing: stratified 80/20 split (`random_state=42`), giving 1,552 training rows and 389 test rows. `StandardScaler` is applied inside the pipeline for Logistic Regression, KNN and SVM, so scaling is saved with each model. Decision Tree and Naive Bayes use raw features. The 389-row test split is saved as `test_data.csv`.

Metric averaging: Precision, Recall and F1 are **macro** averages; AUC is **one-vs-rest macro** from predicted probabilities; MCC is the multi-class generalisation. Macro averaging is used because the rare classes (Stains, Dirtiness) matter as much as the common ones.

## c. GitHub Repository Link

https://github.com/IAmCyphr/steel-plate-fault-classifier

Repository layout:

```
├── app.py               Streamlit app (loads saved models, never retrains)
├── charts.py            confusion matrix and bar chart figures
├── theme.py             colour tokens and stylesheet loader
├── assets/style.css     app styling
├── .streamlit/config.toml  theme settings
├── train.py             trains and saves all five models, writes test_data.csv
├── metrics.py           metric code shared by train.py and app.py
├── requirements.txt
├── test_data.csv        held-out 389 rows for the app
├── data/steel_plates_faults.csv
├── model/               5 x .joblib pipelines
└── results/metrics.csv
```

Run locally: `pip install -r requirements.txt`, then `python train.py` (about 5 seconds in total), then `streamlit run app.py`.

## d. Models used

Test-set results (389 rows). Total training time for all five models is under 5 seconds.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7172 | 0.9120 | 0.7854 | 0.7450 | 0.7596 | 0.6368 |
| KNN | 0.7429 | 0.9218 | 0.7389 | 0.7573 | 0.7442 | 0.6732 |
| Naive Bayes | 0.4524 | 0.8386 | 0.4315 | 0.4795 | 0.3830 | 0.3789 |
| Support Vector Machine | 0.7506 | 0.9554 | 0.7971 | 0.7492 | 0.7667 | 0.6777 |

Settings: Logistic Regression `max_iter=1000`; Decision Tree `max_depth=8`; KNN `k=7`; Gaussian NB defaults; SVM RBF kernel `C=5, probability=True`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid linear baseline (accuracy 0.7275, AUC 0.9379). Strong on the distinct faults K_Scatch (recall 0.91) and Z_Scratch (0.87), but weakest on Bumps (0.57), which it confuses with Other_Faults (30 of 81 Bumps). A linear boundary cannot separate those overlapping groups. |
| Decision Tree | Lowest accuracy of the four non-NB models (0.7172) and lowest AUC (0.9120), because tree probabilities are coarse. Its macro precision (0.7854) and recall (0.7450) are good, and it is the best at rare classes: Dirtiness recall 0.82 against 0.64 for LR and SVM, since splits can isolate small groups. Max depth 8 keeps it from overfitting. |
| KNN | Second-best accuracy (0.7429) and best on K_Scatch (recall 0.96) once features are scaled, but worst at Pastry (recall 0.38), a mid-sized class whose neighbours are mostly Bumps. Training is instant, but the saved model stores the whole training set, and prediction is the slow part. |
| Naive Bayes | Clearly the worst (accuracy 0.4524, MCC 0.3789). Many features are strongly correlated (min/max coordinates, area, log-area, sigmoid-area), which breaks the independence assumption. It labels most Other_Faults as Bumps (78 of 135) and recalls only 10% of Other_Faults and 3% of Pastry. It does well on Stains (0.86) and Bumps (0.74). |
| Support Vector Machine | Best on accuracy (0.7506), AUC (0.9554), precision (0.7971) and MCC (0.6777). The RBF kernel handles the non-linear overlap between classes, and its probabilities rank classes best. Remaining errors are mostly Bumps and Other_Faults confused both ways (24 and 23 cases). Training takes about 0.6 s. |
| **Overall winner** | **Support Vector Machine.** It leads on Accuracy, AUC, Precision, F1 and MCC; only Recall goes to KNN (0.7573 vs 0.7492). KNN is a close second, and Naive Bayes is clearly unsuitable for this correlated feature set. |

## e. Streamlit app

- **Sidebar:** CSV upload (or the bundled `test_data.csv`) and the model dropdown.
- **Evaluation tab:** the six metrics for the selected model with its rank among the five, a confusion matrix (counts or row percentages), a per-class classification report and a predictions download.
- **Model comparison tab:** a bar chart of any one metric across all five models, plus the full comparison table for the uploaded data.
- **Data tab:** class balance, a preview of the loaded rows and a feature summary.

The uploaded file must contain the 27 feature columns and `Class`. Models are loaded from `model/` and no training happens in the app.
