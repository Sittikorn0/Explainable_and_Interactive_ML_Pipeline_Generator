"""
ml_process/runner.py
Model Competition + Auto Hyperparameter Tuning
ไม่มี Streamlit ในไฟล์นี้ → test ได้อิสระ
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_process.config import MODELS_CLASSIFICATION, MODELS_REGRESSION, PARAM_GRIDS, SLOW_MODELS, MAX_ROWS_SLOW

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# ── Optional libraries ────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ── Model registry ────────────────────────────────────────────
def get_model_map() -> dict:
    """
    คืน dict ของ model factories ทั้งหมดที่พร้อมใช้งาน
    XGBoost / LightGBM / CatBoost จะถูกเพิ่มอัตโนมัติถ้าติดตั้งแล้ว
    """
    model_map = {
        # classification
        "logistic_regression":  lambda: LogisticRegression(max_iter=300, solver="lbfgs"),
        "decision_tree":        lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
        "random_forest":        lambda: RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "gradient_boosting":    lambda: HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42),
        "svm":                  lambda: SGDClassifier(loss="hinge", max_iter=500, random_state=42),
        "knn":                  lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "naive_bayes":          lambda: GaussianNB(),
        # regression
        "linear_regression":           lambda: LinearRegression(),
        "decision_tree_regressor":     lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest_regressor":     lambda: RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        "gradient_boosting_regressor": lambda: HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=42),
        "knn_regressor":               lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }

    if _HAS_XGB:
        model_map["xgboost"]           = lambda: XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42, verbosity=0)
        model_map["xgboost_regressor"] = lambda: XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)

    if _HAS_LGB:
        model_map["lightgbm"]           = lambda: LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
        model_map["lightgbm_regressor"] = lambda: LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)

    if _HAS_CAT:
        model_map["catboost"]           = lambda: CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0)
        model_map["catboost_regressor"] = lambda: CatBoostRegressor(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0)

    return model_map


def get_available_models(task_type: str) -> dict:
    """
    คืน display names ของ model ที่พร้อมใช้สำหรับ task นั้น
    รวม XGBoost/LightGBM/CatBoost ถ้าติดตั้งแล้ว
    """
    base = dict(MODELS_CLASSIFICATION if task_type == "classification" else MODELS_REGRESSION)

    if task_type == "classification":
        if _HAS_XGB: base["xgboost"]   = "XGBoost"
        if _HAS_LGB: base["lightgbm"]  = "LightGBM"
        if _HAS_CAT: base["catboost"]  = "CatBoost"
    else:
        if _HAS_XGB: base["xgboost_regressor"]   = "XGBoost"
        if _HAS_LGB: base["lightgbm_regressor"]  = "LightGBM"
        if _HAS_CAT: base["catboost_regressor"]  = "CatBoost"

    return base


# ── Helpers ───────────────────────────────────────────────────
def _safe_cv(y_train, task_type: str) -> int:
    """cv ที่ปลอดภัย — ไม่เกิน min class count"""
    if task_type == "classification":
        min_count = int(pd.Series(y_train).value_counts().min())
        return max(2, min(3, min_count))
    return max(2, min(3, len(y_train) // 2))


def _grid_size(grid: dict) -> int:
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def _sample(X, y, max_rows: int):
    if len(X) <= max_rows:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


# ── Competition ───────────────────────────────────────────────
def run_competition(X_train, X_test, y_train, y_test,
                    task_type: str,
                    on_progress=None) -> dict:
    """
    Train ทุก model พร้อม Auto Hyperparameter Tuning
    เลือก best จาก CV score แล้ว predict บน test set

    Args:
        on_progress: callback(model_label: str, i: int, total: int)
                     ใช้อัปเดต progress bar ใน UI

    Returns: {
        "competition":  {key: {label, cv_score, cv_std, best_params, error}},
        "best_key":     str,
        "best_label":   str,
        "best_params":  dict,
        "y_test":       Series,
        "y_pred":       ndarray,
        "task_type":    str,
    }
    """
    scorer    = "f1_weighted" if task_type == "classification" else "r2"
    cv        = _safe_cv(y_train, task_type)
    models    = get_available_models(task_type)
    model_map = get_model_map()

    competition: dict = {}
    best_key    = ""
    best_score  = -float("inf")
    best_model  = None

    total = len(models)
    for i, (key, label) in enumerate(models.items()):

        if on_progress:
            on_progress(label, i, total)

        # slow model cap
        X_tr, y_tr = X_train, y_train
        if key in SLOW_MODELS:
            X_tr, y_tr = _sample(X_train, y_train, MAX_ROWS_SLOW)

        try:
            factory = model_map.get(key)
            if factory is None:
                raise ImportError(f"{key} ไม่พร้อมใช้งาน")

            m    = factory()
            grid = PARAM_GRIDS.get(key, {})
            best_params: dict = {}

            if grid:
                search = RandomizedSearchCV(
                    m, grid,
                    n_iter       = min(8, _grid_size(grid)),
                    cv           = cv,
                    scoring      = scorer,
                    random_state = 42,
                    n_jobs       = -1,
                    refit        = True,
                    error_score  = "raise",
                )
                search.fit(X_tr, y_tr)
                m           = search.best_estimator_
                best_params = search.best_params_
                cv_mean     = float(search.best_score_)
                cv_std      = float(search.cv_results_["mean_test_score"].std())
            else:
                m.fit(X_tr, y_tr)
                scores  = cross_val_score(m, X_tr, y_tr, cv=cv, scoring=scorer, n_jobs=-1)
                cv_mean = float(scores.mean())
                cv_std  = float(scores.std())
                m.fit(X_tr, y_tr)

            competition[key] = {
                "label":       label,
                "cv_score":    round(cv_mean, 4),
                "cv_std":      round(cv_std, 4),
                "best_params": best_params,
                "error":       None,
            }

            if cv_mean > best_score:
                best_score = cv_mean
                best_key   = key
                best_model = m

        except Exception as ex:
            competition[key] = {
                "label":       label,
                "cv_score":    None,
                "cv_std":      None,
                "best_params": {},
                "error":       str(ex),
            }

    if best_model is None:
        errors = [f"{v['label']}: {v['error']}" for v in competition.values() if v["error"]]
        raise ValueError("ทุก model ล้มเหลว:\n" + "\n".join(errors))

    y_pred = best_model.predict(X_test)

    return {
        "competition": competition,
        "best_key":    best_key,
        "best_label":  models[best_key],
        "best_params": competition[best_key]["best_params"],
        "y_test":      y_test,
        "y_pred":      y_pred,
        "task_type":   task_type,
    }