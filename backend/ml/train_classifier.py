"""
Structured sklearn workflow for Travel Style classification.

Run:
    python train_structured_workflow.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_PATH = Path(__file__).resolve().parent  # backend/ml/
BACKEND_PATH = ROOT_PATH.parent  # backend/
RANDOM_STATE = 42
DATASET_PATH = ROOT_PATH / "data" / "processed" / "travel_data.csv"
TARGET_COLUMN = "Travel Style"
ID_COLUMNS = ["Destination Name"]
LEAKAGE_COLUMNS: list[str] = []
PRIMARY_METRIC = "f1_macro"


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def safe_convert_numeric_objects(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Convert object columns to numeric only when most non-null values are numeric-looking."""
    df = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null_original = df[col].notna().sum()
            non_null_converted = converted.notna().sum()
            if non_null_original > 0 and (non_null_converted / non_null_original) >= 0.90:
                df[col] = converted
    return df


def class_distribution(name: str, y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    pct = (counts / len(y) * 100).round(2)
    dist = pd.DataFrame({"count": counts, "percentage": pct})
    print(f"\n{name} class distribution:")
    print(dist)
    return dist


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features, categorical_features


def make_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Create comparable model pipelines with cloned preprocessing."""
    return {
        "LogisticRegression_balanced": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest_balanced": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTrees_balanced": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting_balanced": Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),   
    }


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
) -> tuple[dict, Pipeline]:
    """Fit a pipeline and return validation metrics."""
    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)

    train_pred = fitted.predict(X_train)
    val_pred = fitted.predict(X_val)

    train_accuracy = accuracy_score(y_train, train_pred)
    train_macro_f1 = f1_score(y_train, train_pred, average="macro", zero_division=0)

    val_accuracy = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_precision_macro = precision_score(y_val, val_pred, average="macro", zero_division=0)
    val_recall_macro = recall_score(y_val, val_pred, average="macro", zero_division=0)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "params": str(fitted.named_steps["classifier"].get_params()),
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_precision_macro": val_precision_macro,
        "val_recall_macro": val_recall_macro,
        "train_val_accuracy_gap": train_accuracy - val_accuracy,
        "train_val_macro_f1_gap": train_macro_f1 - val_macro_f1,
    }

    return result, fitted


def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    metric_cols = [
        "val_accuracy",
        "val_macro_f1",
        "val_precision_macro",
        "val_recall_macro",
    ]
    ax = results_df.set_index("model")[metric_cols].plot(kind="bar", figsize=(12, 6))
    ax.set_title("val Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "val_metric_comparison.png", dpi=150)
    plt.close()

    gap_cols = ["train_val_accuracy_gap", "train_val_macro_f1_gap"]
    ax = results_df.set_index("model")[gap_cols].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Train–Validation Gap")
    ax.set_ylabel("Train score minus validation score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "train_val_gap.png", dpi=150)
    plt.close()


def save_confusion_matrix(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    predictions = estimator.predict(X)
    cm = confusion_matrix(y, predictions, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, values_format="d", xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = BACKEND_PATH
    data_path = DATASET_PATH
    output_dir = ROOT_PATH / "outputs"
    model_dir = BACKEND_PATH / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    print_header("1. Load and inspect dataset")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    print_header("2. Clean dataset")
    df = df.copy()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype(str).str.strip()

    before_duplicates = len(df)
    df = df.drop_duplicates()
    print(f"Dropped duplicate rows: {before_duplicates - len(df)}")

    df = safe_convert_numeric_objects(df, TARGET_COLUMN)

    columns_to_drop = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    print(f"Columns removed from features: {columns_to_drop}")

    print_header("3. Create X and y")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN] + columns_to_drop)

    class_distribution("Full dataset", y)

    print(f"\nFeature columns: {X.columns.tolist()}")

    print_header("4. Stratified 60/20/20 split")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows: {len(X_test)}")
    class_distribution("Train", y_train)
    class_distribution("Validation", y_val)
    class_distribution("Test", y_test)

    print_header("5. Build preprocessing")
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    print_header("6. Train/validation model comparison")
    models = make_models(preprocessor)
    val_results = []
    fitted_models = {}

    for name, pipe in models.items():
        result, fitted = evaluate_model(pipe, X_train, y_train, X_val, y_val, name)
        val_results.append(result)
        fitted_models[name] = fitted
        print(f"{name}: validation macro F1 = {result['val_macro_f1']:.4f}")

    results_df = pd.DataFrame(val_results).sort_values("val_macro_f1", ascending=False)
    print("\nValidation results:")
    print(results_df[[
        "model",
        "val_accuracy",
        "val_macro_f1",
        "val_precision_macro",
        "val_recall_macro",
        "train_val_macro_f1_gap",
    ]])

    results_df.to_csv(output_dir / "results_validation.csv", index=False)
    results_df.to_csv(output_dir / "results.csv", index=False)
    plot_model_comparison(results_df, output_dir)

    labels = sorted(y.unique().tolist())
    for name, fitted in fitted_models.items():
        safe_name = name.replace(" ", "_").replace("/", "_")
        save_confusion_matrix(
            fitted,
            X_val,
            y_val,
            labels,
            f"Validation Confusion Matrix — {name}",
            output_dir / f"confusion_matrix_validation_{safe_name}.png",
        )

    print_header("7. K-fold cross-validation on training data")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    cv_rows = []
    for name, pipe in models.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=None,
        )
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
            "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_train_macro_f1_mean": scores["train_macro_f1"].mean(),
            "cv_train_val_macro_f1_gap": scores["train_macro_f1"].mean() - scores["test_macro_f1"].mean(),
        }
        cv_rows.append(row)
        print(
            f"{name}: CV accuracy {row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}, "
            f"CV macro F1 {row['cv_macro_f1_mean']:.4f} ± {row['cv_macro_f1_std']:.4f}"
        )

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_macro_f1_mean", ascending=False)
    cv_df.to_csv(output_dir / "results_cv.csv", index=False)

    print_header("8. Interpretability")
    # Logistic Regression coefficients
    if "LogisticRegression_balanced" in fitted_models:
        logreg_pipe = fitted_models["LogisticRegression_balanced"]
        feature_names = logreg_pipe.named_steps["preprocessor"].get_feature_names_out()
        clf = logreg_pipe.named_steps["classifier"]
        coef_rows = []
        for class_label, coefs in zip(clf.classes_, clf.coef_, strict=False):
            top_pos_idx = np.argsort(coefs)[-10:][::-1]
            top_neg_idx = np.argsort(coefs)[:10]
            for idx in top_pos_idx:
                coef_rows.append({
                    "class": class_label,
                    "direction": "positive",
                    "feature": feature_names[idx],
                    "coefficient": coefs[idx],
                })
            for idx in top_neg_idx:
                coef_rows.append({
                    "class": class_label,
                    "direction": "negative",
                    "feature": feature_names[idx],
                    "coefficient": coefs[idx],
                })
        pd.DataFrame(coef_rows).to_csv(output_dir / "logistic_regression_top_coefficients.csv", index=False)
        print("Saved Logistic Regression coefficient interpretation.")

    # Permutation importance for best validation model
    best_val_model_name = results_df.iloc[0]["model"]
    best_val_model = fitted_models[best_val_model_name]
    perm = permutation_importance(
        best_val_model,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=None,
    )
    perm_df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(output_dir / "permutation_importance_best_validation_model.csv", index=False)
    print(f"Saved permutation importance for {best_val_model_name}.")

    print_header("9. Hyperparameter tuning")
    # Tune Random Forest because it handles mixed feature spaces well and supports class_weight.
    rf_tuning_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [None, 8],
        "classifier__min_samples_leaf": [1, 3],
        "classifier__class_weight": ["balanced"],
    }

    grid_search = GridSearchCV(
        estimator=rf_tuning_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=None,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    tuning_df = pd.DataFrame(grid_search.cv_results_).sort_values("rank_test_score")
    tuning_df.to_csv(output_dir / "tuning_results.csv", index=False)

    tuned_model = grid_search.best_estimator_
    tuned_val_pred = tuned_model.predict(X_val)
    tuned_result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "RandomForest_tuned",
        "params": str(grid_search.best_params_),
        "train_accuracy": accuracy_score(y_train, tuned_model.predict(X_train)),
        "train_macro_f1": f1_score(y_train, tuned_model.predict(X_train), average="macro", zero_division=0),
        "val_accuracy": accuracy_score(y_val, tuned_val_pred),
        "val_macro_f1": f1_score(y_val, tuned_val_pred, average="macro", zero_division=0),
        "val_precision_macro": precision_score(y_val, tuned_val_pred, average="macro", zero_division=0),
        "val_recall_macro": recall_score(y_val, tuned_val_pred, average="macro", zero_division=0),
    }
    tuned_result["train_val_accuracy_gap"] = tuned_result["train_accuracy"] - tuned_result["val_accuracy"]
    tuned_result["train_val_macro_f1_gap"] = tuned_result["train_macro_f1"] - tuned_result["val_macro_f1"]

    print(f"Best RF params: {grid_search.best_params_}")
    print(f"Best RF CV macro F1: {grid_search.best_score_:.4f}")
    print(f"Tuned RF validation macro F1: {tuned_result['val_macro_f1']:.4f}")

    all_results_df = pd.concat([results_df, pd.DataFrame([tuned_result])], ignore_index=True)
    all_results_df = all_results_df.sort_values("val_macro_f1", ascending=False)
    all_results_df.to_csv(output_dir / "results.csv", index=False)

    print_header("10. Final model selection and untouched test evaluation")
    selected_name = all_results_df.iloc[0]["model"]
    print(f"Selected model based on validation macro F1: {selected_name}")

    if selected_name == "RandomForest_tuned":
        final_pipeline = clone(tuned_model)
    else:
        final_pipeline = clone(models[selected_name])

    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    final_pipeline.fit(X_train_val, y_train_val)
    test_pred = final_pipeline.predict(X_test)

    test_metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "selected_model": selected_name,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "test_precision_macro": precision_score(y_test, test_pred, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, test_pred, average="macro", zero_division=0),
    }

    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    report_dict = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(output_dir / "final_classification_report.csv")

    final_cm = confusion_matrix(y_test, test_pred, labels=labels)
    pd.DataFrame(final_cm, index=labels, columns=labels).to_csv(output_dir / "final_confusion_matrix.csv")
    save_confusion_matrix(
        final_pipeline,
        X_test,
        y_test,
        labels,
        f"Final Test Confusion Matrix — {selected_name}",
        output_dir / "final_confusion_matrix.png",
    )

    with open(output_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    model_path = model_dir / "final_travel_style_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    metadata = {
        "dataset_path": str(DATASET_PATH),
        "target_column": TARGET_COLUMN,
        "problem_type": "multiclass classification",
        "random_state": RANDOM_STATE,
        "id_columns_removed": ID_COLUMNS,
        "leakage_columns_removed": LEAKAGE_COLUMNS,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": labels,
        "class_distribution_full": y.value_counts().to_dict(),
        "selected_model": selected_name,
        "model_path": str(model_path.relative_to(project_root)),
        "primary_metric": "macro F1",
        "note": "Destination Name excluded to reduce memorisation. Test set evaluated only once after model selection.",
    }

    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved final pipeline to: {model_path}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
