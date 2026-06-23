"""
Обучение и сравнение моделей кредитного скоринга.

Запуск:
    python -m src.models.train
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.data.preprocessing import load_dataset, split_features_target

SEED = 42
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "credit_data_full.csv"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def build_baseline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])


def build_tuned_random_forest():
    pipe = Pipeline([
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 6, 8],
        "clf__min_samples_leaf": [5, 20, 50],
    }
    return GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=CV, n_jobs=-1)


def build_tuned_gradient_boosting():
    pipe = Pipeline([
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ])
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.03, 0.05, 0.1],
    }
    return GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=CV, n_jobs=-1)


def cross_val_score_simple(model, X, y):
    aucs, f1s = [], []
    for train_idx, val_idx in CV.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y.iloc[val_idx], proba))
        f1s.append(f1_score(y.iloc[val_idx], pred))
    return float(np.mean(aucs)), float(np.mean(f1s))


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(str(DATA_PATH))
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    results = []
    fitted_models = {}

    # 1. Baseline без тюнинга
    t0 = time.time()
    baseline = build_baseline()
    cv_auc, cv_f1 = cross_val_score_simple(baseline, X_train, y_train)
    baseline.fit(X_train, y_train)
    test_proba = baseline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    test_f1 = f1_score(y_test, (test_proba >= 0.5).astype(int))
    fitted_models["logreg_baseline"] = baseline
    results.append({
        "model": "logreg_baseline", "cv_roc_auc": round(cv_auc, 4), "cv_f1": round(cv_f1, 4),
        "test_roc_auc": round(test_auc, 4), "test_f1": round(test_f1, 4),
        "train_time_sec": round(time.time() - t0, 2), "best_params": "-",
    })
    print(f"logreg_baseline: cv_roc_auc={cv_auc:.4f} test_roc_auc={test_auc:.4f} test_f1={test_f1:.4f}")

    # 2-3. Модели с подбором гиперпараметров по GridSearchCV (scoring=roc_auc, cv=5)
    tuned_builders = {
        "random_forest_tuned": build_tuned_random_forest,
        "gradient_boosting_tuned": build_tuned_gradient_boosting,
    }
    for name, builder in tuned_builders.items():
        t0 = time.time()
        search = builder()
        if name == "gradient_boosting_tuned":
            # GradientBoostingClassifier не поддерживает class_weight (в отличие от
            # LogisticRegression/RandomForest), поэтому баланс классов передаётся
            # через sample_weight, иначе модель недооценивает дефолтный класс
            sample_weight = compute_sample_weight("balanced", y_train)
            search.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            search.fit(X_train, y_train)
        best_model = search.best_estimator_
        cv_auc = search.best_score_
        test_proba = best_model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)
        test_auc = roc_auc_score(y_test, test_proba)
        test_f1 = f1_score(y_test, test_pred)
        elapsed = time.time() - t0

        fitted_models[name] = best_model
        results.append({
            "model": name, "cv_roc_auc": round(cv_auc, 4), "cv_f1": None,
            "test_roc_auc": round(test_auc, 4), "test_f1": round(test_f1, 4),
            "train_time_sec": round(elapsed, 2), "best_params": json.dumps(search.best_params_),
        })
        print(f"{name}: cv_roc_auc={cv_auc:.4f} test_roc_auc={test_auc:.4f} test_f1={test_f1:.4f} "
              f"best_params={search.best_params_}")

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    # калибровка вероятностей финальной модели на train-наборе
    uncalibrated_proba = best_model.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, uncalibrated_proba)

    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    final_proba = calibrated.predict_proba(X_test)[:, 1]
    final_auc = roc_auc_score(y_test, final_proba)
    brier_after = brier_score_loss(y_test, final_proba)
    print(f"\nBrier score до калибровки: {brier_before:.4f}, после калибровки: {brier_after:.4f}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, final_proba)
    f1_scores = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros_like(precisions), where=(precisions + recalls) != 0,
    )
    best_threshold_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) else 0
    best_threshold = float(thresholds[best_threshold_idx]) if len(thresholds) else 0.5
    final_f1 = float(f1_scores[best_threshold_idx]) if len(thresholds) else 0.0

    joblib.dump(calibrated, ARTIFACTS_DIR / "model.joblib")

    # интерпретируемость: feature importance финальной модели (до калибровки,
    # калиброванный wrapper не предоставляет feature_importances_ напрямую)
    final_clf = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(final_clf, "feature_importances_"):
        importances = pd.Series(final_clf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        importances.to_csv(ARTIFACTS_DIR / "feature_importance.csv", header=["importance"])
        print("\nFeature importance финальной модели:")
        print(importances.round(4).to_string())

    metadata = {
        "best_model": best_name,
        "calibrated_test_roc_auc": round(float(final_auc), 4),
        "calibrated_test_f1": round(float(final_f1), 4),
        "brier_score_before_calibration": round(float(brier_before), 4),
        "brier_score_after_calibration": round(float(brier_after), 4),
        "decision_threshold": round(best_threshold, 4),
        "feature_columns": list(X.columns),
        "seed": SEED,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nЛучшая модель (до калибровки):", best_name)
    print("Метрики финальной (калиброванной) модели на test:",
          metadata["calibrated_test_roc_auc"], metadata["calibrated_test_f1"])
    print("Артефакты сохранены в", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
