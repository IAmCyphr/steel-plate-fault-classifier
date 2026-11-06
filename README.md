# Steel Plate Fault Inspector

Classifying steel plate surface faults with five ML models and a Streamlit app.

- GitHub repository: https://github.com/IAmCyphr/steel-plate-fault-classifier
- Live app: https://steel-plate-fault-classifier-gz6crdafengjsxp8qyzmx4.streamlit.app/

## a. Problem statement

Steel plates are checked for surface faults after rolling. Each plate in this dataset comes with 27 numeric measurements: the size and position of the defect, brightness readings, the steel type and thickness, and a few shape indices. The goal is to predict which of seven fault types a plate has, so that it can be sent to the right corrective process. I trained five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and SVM) on the same train/test split and compared them on six metrics.

## b. Dataset description

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, *Steel Plates Faults* (Research Center of Sciences of Communication, Rome) |
| Instances | 1,941 (no missing values, no duplicates) |
| Features | 27 numeric, for example `X_Minimum`, `Pixels_Areas`, `Sum_of_Luminosity`, `Steel_Plate_Thickness`, `Edges_Index`, `LogOfAreas`, `SigmoidOfAreas` |
| Target | `Class`, one of 7 faults. The original 7 one-hot columns were merged into a single label (exactly one is set per row) |
| Class counts | Other_Faults 673, Bumps 402, K_Scatch 391, Z_Scratch 190, Pastry 158, Stains 72, Dirtiness 55 |

The data was split 80/20 with stratification (`random_state=42`), which gives 1,552 training rows and 389 test rows. Logistic Regression, KNN and SVM are wrapped in a pipeline with `StandardScaler`, so the scaling is saved inside each model file. Decision Tree and Naive Bayes use the raw values. The 389 test rows are saved as `test_data.csv`.

Precision, Recall and F1 are macro averages, AUC is one-vs-rest macro computed from the predicted probabilities, and MCC is the multi-class version. I used macro averaging because the small classes (Stains, Dirtiness) should count as much as the big ones.

## c. GitHub Repository Link

https://github.com/IAmCyphr/steel-plate-fault-classifier

Files in the repository:

- `app.py`: the Streamlit app, which loads the saved models and never retrains
- `charts.py`: confusion matrix and bar chart figures
- `theme.py` and `assets/style.css`: colours and styling for the app
- `.streamlit/config.toml`: theme settings
- `train.py`: trains and saves the five models and writes `test_data.csv`
- `metrics.py`: metric code shared by `train.py` and `app.py`
- `requirements.txt`
- `test_data.csv`: the 389 held-out rows
- `data/steel_plates_faults.csv`: the full dataset
- `model/`: the five saved `.joblib` pipelines
- `results/metrics.csv`: the metrics table from `train.py`

To run it locally: `pip install -r requirements.txt`, then `python train.py` (about 5 seconds), then `streamlit run app.py`.

## d. Models used

Results on the 389 test rows. Training all five models takes under 5 seconds.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7172 | 0.9120 | 0.7854 | 0.7450 | 0.7596 | 0.6368 |
| KNN | 0.7429 | 0.9218 | 0.7389 | 0.7573 | 0.7442 | 0.6732 |
| Naive Bayes | 0.4524 | 0.8386 | 0.4315 | 0.4795 | 0.3830 | 0.3789 |
| Support Vector Machine | 0.7506 | 0.9554 | 0.7971 | 0.7492 | 0.7667 | 0.6777 |

Settings: Logistic Regression `max_iter=1000`, Decision Tree `max_depth=8`, KNN `k=7`, Gaussian Naive Bayes with defaults, SVM with RBF kernel and `C=5`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A decent linear baseline: accuracy 0.7275 and AUC 0.9379. It does well on K_Scatch (recall 0.91) and Z_Scratch (0.87) but badly on Bumps (recall 0.57). 30 of the 81 Bumps were predicted as Other_Faults, because a straight boundary cannot separate those two groups. |
| Decision Tree | Accuracy 0.7172 and AUC 0.9120, the lowest of the four models other than Naive Bayes. A tree gives coarse probabilities, which hurts AUC. Precision (0.7854) and recall (0.7450) are still good, and it handles the small classes best: Dirtiness recall is 0.82 against 0.64 for Logistic Regression and SVM. The depth is limited to 8 to avoid overfitting. |
| KNN | Second best accuracy (0.7429) and the best recall overall (0.7573). After scaling it is very good on K_Scatch (recall 0.96) but weak on Pastry (recall 0.38), where many plates are predicted as Bumps or Other_Faults. It trains instantly, but the saved model keeps the whole training set and prediction is the slower part. |
| Naive Bayes | By far the worst: accuracy 0.4524 and MCC 0.3789. Many features are highly correlated (the min and max coordinates, area, log of area and so on), while Naive Bayes assumes they are independent. 78 of the 135 Other_Faults plates were predicted as Bumps, so recall is only 0.10 on Other_Faults and 0.03 on Pastry. It does fine on Stains (0.86) and Bumps (0.74). |
| Support Vector Machine | Best accuracy (0.7506), AUC (0.9554), precision (0.7971), F1 (0.7667) and MCC (0.6777). The RBF kernel can model the overlap between classes, which a linear model cannot. Most of the remaining mistakes are Bumps and Other_Faults mixed up in both directions (24 and 23 plates). Training takes under a second. |
| Overall winner | Support Vector Machine. It has the best Accuracy, AUC, Precision, F1 and MCC, and only loses Recall to KNN (0.7573 against 0.7492). KNN comes second. Naive Bayes is not suitable for this dataset because the features are so correlated. |

## e. Streamlit app

- Sidebar: upload a CSV (or tick the bundled `test_data.csv`) and pick a model from the dropdown.
- Evaluation tab: the six metrics for the chosen model, a confusion matrix, a per-class classification report and a button to download the predictions.
- Model comparison tab: a bar chart of one metric across all five models, and a table with every metric.
- Data tab: class balance, a preview of the rows and a feature summary.

The uploaded CSV needs the 27 feature columns (numeric) and a `Class` column with the seven fault names. Extra columns and a different column order are fine. Any other file shows an error message. The test file I used is `test_data.csv` in the repository root, and the sidebar has a button to download it. If the uploaded file has only some of the classes, precision, recall, F1 and AUC are averaged over the classes that are present, and AUC shows n/a when there is only one class. The app only loads the saved models from `model/`. It never trains.
