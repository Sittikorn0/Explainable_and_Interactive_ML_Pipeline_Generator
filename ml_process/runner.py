"""ml_process/runner.py — model registry + competition"""
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

from ml_process.config import MODELS_CLF, MODELS_REG, PARAM_GRIDS, SLOW_MODELS, MAX_ROWS_SLOW

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


def get_model_map() -> dict:
    m = {
        "logistic_regression":         lambda: LogisticRegression(max_iter=300, solver="lbfgs"),
        "decision_tree":               lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
        "random_forest":               lambda: RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "gradient_boosting":           lambda: HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42),
        "svm":                         lambda: SGDClassifier(loss="hinge", max_iter=500, random_state=42),
        "knn":                         lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "naive_bayes":                 lambda: GaussianNB(),
        "linear_regression":           lambda: LinearRegression(),
        "decision_tree_regressor":     lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest_regressor":     lambda: RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        "gradient_boosting_regressor": lambda: HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=42),
        "knn_regressor":               lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }
    if _HAS_XGB:
        m["xgboost"]           = lambda: XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42, verbosity=0)
        m["xgboost_regressor"] = lambda: XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
    if _HAS_LGB:
        m["lightgbm"]           = lambda: LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
        m["lightgbm_regressor"] = lambda: LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    if _HAS_CAT:
        m["catboost"]           = lambda: CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0)
        m["catboost_regressor"] = lambda: CatBoostRegressor(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0)
    return m


def get_available_models(task_type: str) -> dict:
    base = dict(MODELS_CLF if task_type == "classification" else MODELS_REG)
    if task_type == "classification":
        if _HAS_XGB: base["xgboost"]  = "XGBoost"
        if _HAS_LGB: base["lightgbm"] = "LightGBM"
        if _HAS_CAT: base["catboost"] = "CatBoost"
    else:
        if _HAS_XGB: base["xgboost_regressor"]  = "XGBoost"
        if _HAS_LGB: base["lightgbm_regressor"]  = "LightGBM"
        if _HAS_CAT: base["catboost_regressor"]  = "CatBoost"
    return base


def _safe_cv(y_train, task_type: str) -> int:
    if task_type == "classification":
        return max(2, min(3, int(pd.Series(y_train).value_counts().min())))
    return max(2, min(3, len(y_train) // 2))


def _grid_size(grid): 
    total = 1
    for v in grid.values(): total *= len(v)
    return total


def _sample(X, y, max_rows):
    if len(X) <= max_rows: return X, y
    idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def run_competition(X_train, X_test, y_train, y_test,
                    task_type: str, on_progress=None) -> dict:
    scorer    = "f1_macro" if task_type == "classification" else "r2"
    cv        = _safe_cv(y_train, task_type)
    models    = get_available_models(task_type)
    model_map = get_model_map()

    competition = {}
    best_key, best_score, best_model = "", -float("inf"), None

    for i, (key, label) in enumerate(models.items()):
        if on_progress:
            on_progress(label, i, len(models))
        X_tr, y_tr = (_sample(X_train, y_train, MAX_ROWS_SLOW)
                      if key in SLOW_MODELS else (X_train, y_train))
        try:
            m    = model_map[key]()
            grid = PARAM_GRIDS.get(key, {})
            best_params = {}
            if grid:
                search = RandomizedSearchCV(m, grid, n_iter=min(8, _grid_size(grid)),
                                            cv=cv, scoring=scorer, random_state=42,
                                            n_jobs=-1, refit=True, error_score="raise")
                search.fit(X_tr, y_tr)
                m, best_params = search.best_estimator_, search.best_params_
                cv_mean = float(search.best_score_)
                cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
            else:
                scores  = cross_val_score(m, X_tr, y_tr, cv=cv, scoring=scorer, n_jobs=-1)
                cv_mean, cv_std = float(scores.mean()), float(scores.std())
                m.fit(X_tr, y_tr)

            competition[key] = {"label": label, "cv_score": round(cv_mean, 4),
                                 "cv_std": round(cv_std, 4), "best_params": best_params, "error": None}
            if cv_mean > best_score:
                best_score, best_key, best_model = cv_mean, key, m

        except Exception as ex:
            competition[key] = {"label": label, "cv_score": None,
                                 "cv_std": None, "best_params": {}, "error": str(ex)}

    if best_model is None:
        error_lines = "\n".join(
            f"  • {v['label']}: {v['error']}"
            for v in competition.values()
            if v.get("error")
        )
        raise ValueError(f"ทุก model ล้มเหลว:\n{error_lines}")

    return {"competition": competition, "best_key": best_key,
            "best_label": models[best_key], "best_params": competition[best_key]["best_params"],
            "y_test": y_test, "y_pred": best_model.predict(X_test), "task_type": task_type}