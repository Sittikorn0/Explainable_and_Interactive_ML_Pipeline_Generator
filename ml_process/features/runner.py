"""ml_process/runner.py — model registry + competition"""
import os
import warnings
import numpy as np
import pandas as pd

# กำหนดจำนวน CPU Cores ที่ใช้ (ไม่ให้ดึงไปหมดจนเครื่องค้าง)
# เช่น i7-12700H (20 threads) จะใช้แค่ 6 threads เพื่อให้เครื่องยังเล่นเน็ตหรือทำงานอื่นได้ลื่นๆ
_N_JOBS = min(6, max(1, os.cpu_count() // 2)) if os.cpu_count() else 2
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from ml_process.features.config import MODELS_CLF, MODELS_REG, PARAM_GRIDS, SLOW_MODELS, MAX_ROWS_SLOW

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
        "logistic_regression":         lambda: LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
        "decision_tree":               lambda: DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "random_forest":               lambda: RandomForestClassifier(n_estimators=50, n_jobs=_N_JOBS, random_state=42, class_weight="balanced"),
        "gradient_boosting":           lambda: HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42),
        "svm":                         lambda: SGDClassifier(loss="hinge", max_iter=500, random_state=42, class_weight="balanced"),
        "knn":                         lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=_N_JOBS),
        "naive_bayes":                 lambda: GaussianNB(),
        "linear_regression":           lambda: LinearRegression(),
        "decision_tree_regressor":     lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest_regressor":     lambda: RandomForestRegressor(n_estimators=50, n_jobs=_N_JOBS, random_state=42),
        "gradient_boosting_regressor": lambda: HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=42),
        "knn_regressor":               lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=_N_JOBS),
    }
    if _HAS_XGB:
        m["xgboost"]           = lambda: XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42, verbosity=0)
        m["xgboost_regressor"] = lambda: XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
    if _HAS_LGB:
        m["lightgbm"]           = lambda: LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1, class_weight="balanced")
        m["lightgbm_regressor"] = lambda: LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    if _HAS_CAT:
        m["catboost"]           = lambda: CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0, auto_class_weights="Balanced")
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


def _safe_cv(y_train, task_type: str):
    if task_type == "classification":
        min_class_count = int(pd.Series(y_train).value_counts().min())
        cv = max(2, min(5, min_class_count))
        # ถ้ามี class ที่มีข้อมูลไม่พอสำหรับ StratifiedKFold (ต้องมี >= cv)
        # ให้เปลี่ยนไปใช้ KFold ธรรมดาเพื่อป้องกัน Error
        if min_class_count < cv:
            return KFold(n_splits=cv, shuffle=True, random_state=42)
        return cv
    return max(2, min(5, len(y_train) // 2))


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

    target_le = None
    if task_type == "classification" and y_train.dtype == object:
        target_le = LabelEncoder()
        y_train = pd.Series(target_le.fit_transform(y_train), index=y_train.index)
        
        known = set(target_le.classes_)
        # ใช้ mode เป็น fallback ปลอดภัยกว่าตัวแรกสุด
        fallback = target_le.transform([pd.Series(target_le.classes_).mode()[0]])[0] 
        y_test = pd.Series(
            y_test.apply(lambda x: target_le.transform([x])[0] if x in known else fallback),
            index=y_test.index
        )

    for i, (key, label) in enumerate(models.items()):
        if on_progress:
            on_progress(label, i, len(models))
        X_tr, y_tr = (_sample(X_train, y_train, MAX_ROWS_SLOW)
                      if key in SLOW_MODELS else (X_train, y_train))
        try:
            m    = model_map[key]()
            grid = PARAM_GRIDS.get(key, {})
            best_params = {}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message="The least populated class")
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        message=".*n_jobs.*no effect")
                if grid:
                    search = RandomizedSearchCV(m, grid, n_iter=min(25, _grid_size(grid)),
                                                cv=cv, scoring=scorer, random_state=42,
                                                n_jobs=_N_JOBS, refit=False, error_score="raise")
                    search.fit(X_tr, y_tr)
                    best_params = search.best_params_
                    cv_mean = float(search.best_score_)
                    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
                    
                    # ตั้งค่า best parameter ให้โมเดล
                    m.set_params(**best_params)
                else:
                    scores  = cross_val_score(m, X_tr, y_tr, cv=cv, scoring=scorer, n_jobs=_N_JOBS)
                    cv_mean, cv_std = float(scores.mean()), float(scores.std())
                    
                # Retrain โมเดลด้วยพารามิเตอร์ที่ดีที่สุดบนชุดข้อมูล X_train เต็มเสมอ (ไม่ใช้ตัวที่สุ่มมาทำ Grid Search)
                # HistGradientBoosting ไม่มี class_weight → ใช้ sample_weight แทน
                fit_kw = {}
                if task_type == "classification" and key in ("gradient_boosting",):
                    fit_kw["sample_weight"] = compute_sample_weight("balanced", y_train)
                m.fit(X_train, y_train, **fit_kw)

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

    y_pred = best_model.predict(X_test)
    
    # ถ้ามีการเข้ารหัส Target ให้แปลงกลับเป็น String เพื่อให้กราฟและ UI อ่านรู้เรื่อง
    if target_le is not None:
        y_test = pd.Series(target_le.inverse_transform(y_test), index=y_test.index)
        y_pred = target_le.inverse_transform(y_pred)

    return {"competition": competition, "best_key": best_key,
            "best_label": models[best_key], "best_params": competition[best_key]["best_params"],
            "y_test": y_test, "y_pred": y_pred, "task_type": task_type}