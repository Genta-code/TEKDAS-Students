"""Train dan evaluasi model Machine Learning prediksi risiko Dropout mahasiswa.

Alur:
    student_train.csv & student_test.csv -> Pipeline (ColumnTransformer + RandomForest) -> Metrik & Feature Importance

Dijalankan dengan:
    python ml_model.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
)

BASE_DIR = Path(__file__).resolve().parent


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLUMNS),
        ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
    ])

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", classifier),
    ])


def train_and_save_model():
    train_path = BASE_DIR / "student_train.csv"
    test_path = BASE_DIR / "student_test.csv"

    if not train_path.exists() or not test_path.exists():
        from data_prep import main as run_data_prep
        print("Menjalankan data_prep.py terlebih dahulu...")
        run_data_prep()

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET]

    pipeline = build_pipeline()
    print("Melatih model Random Forest Dropout Classifier...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_dropout": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_dropout": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_dropout": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_dropout_rate": float(y_test.mean()),
    }

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

    # Simpan pipeline model
    model_output_path = BASE_DIR / "student_dropout_pipeline.pkl"
    joblib.dump(pipeline, model_output_path)
    print(f"Model tersimpan di: {model_output_path}")

    # Simpan metrik
    metrics_path = BASE_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrik tersimpan di: {metrics_path}")

    # Feature Importance
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    fi_df = pd.DataFrame({
        "feature": [str(f).replace("num__", "").replace("cat__", "") for f in feature_names],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi_path = BASE_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importance tersimpan di: {fi_path}")

    return pipeline, metrics, fi_df


if __name__ == "__main__":
    train_and_save_model()
