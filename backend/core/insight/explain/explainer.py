# Libraries
import os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence

# Logic Import
from backend.core.model_training.preprocess.pipeline import preprocess
from backend.core.model_training.trainer.train_model import get_model_map

# Functions
_N_JOBS = min(4, os.cpu_count() or 1)

# Retrain best model ด้วย params ที่ดีที่สุด คืน (model, X_train, X_test, y_train, y_test, task_type) ใช้ใน insight_page
def get_fitted_model(df, target_col, best_key, best_params, trans_summary,
                     missing_rules=None, outlier_rules=None):
    scaling_method     = trans_summary.get("scaling_method", "standard_scaler")
    encoding_decisions = trans_summary.get("encoding_decisions") or None
    X_train, X_test, y_train, y_test, task_type = preprocess(
        df, target_col,
        scaling_method=scaling_method,
        encoding_decisions=encoding_decisions,
        missing_rules=missing_rules,
        outlier_rules=outlier_rules,
    )
    m = get_model_map()[best_key]()
    if best_params:
        try:
            m.set_params(**best_params)
        except Exception:
            pass
    m.fit(X_train, y_train)
    return m, X_train, X_test, y_train, y_test, task_type


# คำนวณ Permutation Importance คืน DataFrame (Feature/Importance/Std) เรียงลำดับ ใช้ใน Feature Importance tab
def compute_permutation_importance(model, X_test, y_test, task_type, n_repeats=10):
    scoring = "f1_macro" if task_type == "classification" else "r2"
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, scoring=scoring, random_state=42, n_jobs=_N_JOBS
    )
    return (pd.DataFrame({
        "Feature":    X_test.columns,
        "Importance": result.importances_mean,
        "Std":        result.importances_std,
    }).sort_values("Importance", ascending=False).reset_index(drop=True))


# คำนวณ Partial Dependence สำหรับ feature เดียว คืน (x_vals, y_avg, error|None) ใช้ใน Data Visualization tab
def compute_pdp(model, X_train, feature_col, task_type, n_points=30):
    try:
        result = partial_dependence(
            model, X_train, features=[feature_col],
            kind="average", grid_resolution=n_points
        )
        x_vals  = result["grid_values"][0]
        avg_raw = result["average"]
        # avg_raw shape varies: (1, n) binary clf, (n_classes, n) multiclass, (1, n) regression
        if avg_raw.ndim == 3:
            avg = avg_raw[:, 0, :].mean(axis=0)
        else:
            avg = avg_raw[0]
        return x_vals, np.asarray(avg), None
    except Exception as e:
        return None, None, str(e)


# คืน scalar score (max proba สำหรับ clf หรือ predicted value สำหรับ reg) ใช้ภายใน explain_single_row
def _safe_score(model, X, task_type):
    if task_type == "classification":
        try:
            return float(model.predict_proba(X)[0].max())
        except AttributeError:
            pass
    return float(model.predict(X)[0])


# Perturbation-based local explanation คืน (contrib_df, base_score, row_score) ใช้ใน Feature Importance tab แถว single row
def explain_single_row(model, X_train, row: pd.Series, task_type: str, max_features: int = 15):
    baseline   = X_train.mean()
    base_input = baseline.to_frame().T.reset_index(drop=True)
    row_input  = row.to_frame().T.reset_index(drop=True)

    base_score = _safe_score(model, base_input, task_type)
    row_score  = _safe_score(model, row_input, task_type)

    contribs = []
    for feat in X_train.columns[:max_features]:
        perturbed       = row_input.copy()
        perturbed[feat] = float(baseline[feat])
        p               = _safe_score(model, perturbed, task_type)
        contribs.append({
            "Feature":      feat,
            "Value":        round(float(row[feat]), 4),
            "BaseValue":    round(float(baseline[feat]), 4),
            "Contribution": round(row_score - p, 6),
        })

    df = (pd.DataFrame(contribs)
            .sort_values("Contribution", key=abs, ascending=False)
            .reset_index(drop=True))
    df["Direction"] = df["Contribution"].apply(lambda x: "positive" if x >= 0 else "negative")
    return df, base_score, row_score
