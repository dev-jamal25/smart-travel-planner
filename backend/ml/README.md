# Travel Style Destination Classifier

## Dataset source and transformation

The original dataset was `Tourist_Destinations.csv`. It contained **2,000 tourist destinations** with these columns:

- `Destination Name`
- `Country`
- `Continent`
- `Type`
- `Avg Cost (USD/day)`
- `Best Season`
- `Avg Rating`
- `Annual Visitors (M)`
- `UNESCO Site`

The original dataset in data/raw/Tourist_Destinations.csv did **not** contain the required target column, `Travel Style`. To match the assignment, a curated  dataset was created: "data/processed/travel_data_labeled.csv"

This new dataset keeps the original destination attributes and adds one target column:

```text
Travel Style
```

The data were selected from the original dataset and labelled using a fixed rule-based scoring rubric. The model therefore learns the documented labelling logic, not an objective universal truth about tourism.

---

## Target definition and class distribution

Target column: `Travel Style`  
Problem type: multiclass classification  
Rows used: 2000

| Travel Style | Rows |
|---|---:|
| Family | 449 |
| Culture | 408 |
| Luxury | 395 |
| Budget | 372 |
| Adventure | 235 |
| Relaxation | 141 |

The dataset is mildly imbalanced, especially for `Adventure` and `Relaxation`, so macro F1 and per-class metrics are more important than accuracy alone.

---

## Labelling method

The original dataset did not provide travel-style labels. Each destination was scored across the six possible styles, then the class with the highest score became the final label.

If multiple classes tied, this fixed priority order was used:

```text
Budget > Luxury > Culture > Adventure > Relaxation > Family
```

The rubric used thresholds calculated from the original dataset distribution, including low/median/high daily cost, median annual visitors, and median rating.

### Label rules summary

- **Adventure**: favoured destinations marked as adventure/nature, less mass-tourism destinations, and outdoor-friendly seasons.
- **Relaxation**: favoured beach or nature destinations, especially in summer or spring.
- **Culture**: favoured historical/religious destinations, UNESCO sites, and highly rated city destinations.
- **Budget**: favoured destinations in the lowest daily-cost range, while still keeping acceptable quality signals.
- **Luxury**: favoured high-cost destinations with strong rating, popularity, or UNESCO/premium appeal.
- **Family**: treated as a proxy label because the source data has no direct family-safety or child-friendly variables. It favoured popular, accessible, well-rated destinations with mid-range cost.

---

## Feature choices

The model uses these predictors:

| Feature | Reason |
|---|---|
| `Country` | Captures geographic and tourism context. |
| `Continent` | Adds broader regional travel patterns. |
| `Type` | Represents destination category, such as Beach, City, Nature, Historical, Religious, or Adventure. |
| `Avg Cost (USD/day)` | Important for Budget and Luxury distinctions. |
| `Best Season` | Helps represent seasonal suitability. |
| `Avg Rating` | Proxy for traveller satisfaction and destination quality. |
| `Annual Visitors (M)` | Proxy for popularity and accessibility. |
| `UNESCO Site` | Strong signal for cultural or heritage appeal. |

`Destination Name` is removed from training because it is an identifier-like column and could encourage memorisation.

### Note on `Type`

`Type` is kept as a feature because it describes the destination category, while `Travel Style` describes the traveller preference/style. They are related but not identical. For example, a beach destination can be classified as Relaxation, Budget, Luxury, or Family depending on cost, rating, popularity, and season.

Because `Type` was also considered during label creation, it is acknowledged as a strong predictor and possible target-proxy risk. It is kept because it is realistic information that would be known about a destination at prediction time, and the final label was not based on `Type` alone.

---

## Modelling workflow

The notebook follows this workflow:

1. Load and inspect the dataset.
2. Clean text/numeric columns and remove duplicates.
3. Remove `Destination Name` from the feature matrix.
4. Split the data into train/val/test using a stratified 60/20/20 split.
5. Build preprocessing inside a `ColumnTransformer`:
   - numeric: median imputation + scaling
   - categorical: most-frequent imputation + one-hot encoding
6. Train all models inside scikit-learn `Pipeline` objects.
7. Compare models using the same split and 5-fold `StratifiedKFold` cross-validation.
8. Tune one promising model with `GridSearchCV`.
9. Evaluate the selected model once on the untouched test set.
10. Save results and the final fitted pipeline.

---

## Models compared

The notebook compares:

- Logistic Regression with `class_weight='balanced'`
- Random Forest with `class_weight='balanced'`
- Extra Trees with `class_weight='balanced'`
- Histogram Gradient Boosting with `class_weight='balanced'`

The main selection metric is **macro F1**, because the classes are not perfectly balanced.

---

## Tuning strategy

Random Forest was tuned using `GridSearchCV` with macro F1 scoring.

The searched parameters were:

- `n_estimators`: number of trees
- `max_depth`: controls tree complexity and overfitting
- `min_samples_leaf`: regularises leaf size
- `class_weight`: keeps class imbalance handling active

---

## Quick evaluation summary

In the current run, Histogram Gradient Boosting gave the strongest validation result.

| Model | Val Accuracy | Val Macro F1 |
|---|---:|---:|
| HistGradientBoosting_balanced | 0.980 | 0.978 |
| RandomForest_balanced | 0.915 | 0.897 |
| LogisticRegression_balanced | 0.775 | 0.796 |
| ExtraTrees_balanced | 0.625 | 0.683 |
| RandomForest_tuned | 0.915 | 0.897 |

The selected final model was:

```text
HistGradientBoosting_balanced
```

Final untouched test-set metrics:

| Metric | Score |
|---|---:|
| Accuracy | 0.980 |
| Macro F1 | 0.982 |
| Macro Precision | 0.986 |
| Macro Recall | 0.979 |

The lower test score compared with validation shows that the dataset is small and the model may still be sensitive to the split. This is why cross-validation, macro metrics, and per-class reports are included.

---

## Outputs

Running the script creates JSON files for all structured outputs (preferred format for programmatic access) and keeps `results.csv` for assignment compliance.

**Tabular outputs (JSON format):**
- `outputs/results.json` — all model results (validation and tuning)
- `outputs/results.csv` — same as results.json, kept for assignment compliance
- `outputs/results_validation.json` — baseline model validation metrics
- `outputs/results_cv.json` — 5-fold cross-validation results
- `outputs/tuning_results.json` — GridSearchCV hyperparameter tuning details
- `outputs/final_test_metrics.json` — final model performance on test set
- `outputs/final_classification_report.json` — per-class precision/recall/F1 from test set
- `outputs/final_confusion_matrix.json` — confusion matrix (test set)
- `outputs/logistic_regression_top_coefficients.json` — interpretability: top feature weights
- `outputs/permutation_importance_best_validation_model.json` — interpretability: feature importance
- `outputs/model_metadata.json` — dataset info, selected model, feature names, class distribution

**Model artifact:**
- `models/final_travel_style_pipeline.joblib` — fully fitted scikit-learn Pipeline (preprocessing + classifier)

**Note:** No PNG graphs are generated. All results are stored as structured data (JSON/CSV) for reproducibility and integration.

---

## Reproducibility

A fixed seed is used throughout the workflow:

```python
SEED = 42
```

Dependencies are pinned in `requirements.txt`.

---

## Limitations

- The labels are rule-based, so the model learns the labelling rubric rather than a perfect real-world truth.
- `Family` is a proxy label because the original dataset has no direct family-specific features.
- `Type` is a strong domain feature and possible target-proxy risk, but it is kept and documented because it represents destination category rather than the final style label.
- `Destination Name` is excluded to reduce memorisation.
