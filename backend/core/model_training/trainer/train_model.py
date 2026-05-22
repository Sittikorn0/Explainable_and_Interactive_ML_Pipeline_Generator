# Libraries
import os
import warnings
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
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Logic Import
from backend.core.model_training.config.config_ml import *

# กำหนดจำนวน CPU Cores ที่ใช้
N_JOBS_LIMIT = min(6, max(1, os.cpu_count() // 2)) if os.cpu_count() else 2


try:
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# คืน dict mapping model_key → lambda factory สำหรับ instantiate model ใช้ใน run_competition และ get_fitted_model
def get_model_map() -> dict:
    model_mapping = {
        "logistic_regression":         lambda: LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced"),
        "decision_tree":               lambda: DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "random_forest":               lambda: RandomForestClassifier(n_estimators=50, n_jobs=N_JOBS_LIMIT, random_state=42, class_weight="balanced"),
        "gradient_boosting":           lambda: HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42),
        "svm":                         lambda: SGDClassifier(loss="hinge", max_iter=500, random_state=42, class_weight="balanced"),
        "knn":                         lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS_LIMIT),
        "naive_bayes":                 lambda: GaussianNB(),
        "linear_regression":           lambda: LinearRegression(),
        "decision_tree_regressor":     lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest_regressor":     lambda: RandomForestRegressor(n_estimators=50, n_jobs=N_JOBS_LIMIT, random_state=42),
        "gradient_boosting_regressor": lambda: HistGradientBoostingRegressor(max_iter=50, max_depth=4, random_state=42),
        "knn_regressor":               lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=N_JOBS_LIMIT),
    }
    if HAS_XGBOOST:
        model_mapping["xgboost"]           = lambda: XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42, verbosity=0)
        model_mapping["xgboost_regressor"] = lambda: XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
    if HAS_LIGHTGBM:
        model_mapping["lightgbm"]           = lambda: LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1, class_weight="balanced")
        model_mapping["lightgbm_regressor"] = lambda: LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    if HAS_CATBOOST:
        model_mapping["catboost"]           = lambda: CatBoostClassifier(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0, auto_class_weights="Balanced", train_dir="cache/catboost_info")
        model_mapping["catboost_regressor"] = lambda: CatBoostRegressor(iterations=50, depth=4, learning_rate=0.1, random_seed=42, verbose=0, train_dir="cache/catboost_info")
        
    return model_mapping

# คืน dict model_key→label ที่ใช้งานได้จริงตาม task_type และ installed libraries ใช้ใน run_competition
def get_available_models(task_type: str) -> dict:
    available_models = dict(MODELS_CLF if task_type == "classification" else MODELS_REG)
    if task_type == "classification":
        if HAS_XGBOOST: available_models["xgboost"]  = "XGBoost"
        if HAS_LIGHTGBM: available_models["lightgbm"] = "LightGBM"
        if HAS_CATBOOST: available_models["catboost"] = "CatBoost"
    else:
        if HAS_XGBOOST: available_models["xgboost_regressor"]  = "XGBoost"
        if HAS_LIGHTGBM: available_models["lightgbm_regressor"]  = "LightGBM"
        if HAS_CATBOOST: available_models["catboost_regressor"]  = "CatBoost"
    return available_models

# คำนวณ cv splits ที่ปลอดภัย (2-5) ไม่เกิน minority class count ใช้ใน run_competition
def calculate_safe_cv(target_train, task_type: str):
    if task_type == "classification":
        min_class_count = int(pd.Series(target_train).value_counts().min())
        cv_splits = max(2, min(5, min_class_count))
        if min_class_count < cv_splits:
            return KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        return cv_splits
    return max(2, min(5, len(target_train) // 2))

# คำนวณจำนวน combinations ทั้งหมดของ param_grid ใช้ใน run_competition เพื่อกำหนด n_iter
def calculate_grid_size(param_grid: dict) -> int:
    total_combinations = 1
    for param_values in param_grid.values(): 
        total_combinations *= len(param_values)
    return total_combinations

# สุ่มลด training data สำหรับ slow models (knn/svm) ใช้ใน run_competition
def sample_data(features, target, max_rows: int):
    if len(features) <= max_rows: 
        return features, target
    sampled_indices = np.random.RandomState(42).choice(len(features), max_rows, replace=False)
    return features.iloc[sampled_indices].reset_index(drop=True), target.iloc[sampled_indices].reset_index(drop=True)

# train ทุก model ด้วย RandomizedSearchCV เลือก best ตาม CV score คืน competition dict ใช้ใน model_process_page
def run_competition(features_train, features_test, target_train, target_test,
                    task_type: str, on_progress=None) -> dict:
                        
    scorer_metric = "f1_macro" if task_type == "classification" else "r2"
    cv_strategy   = calculate_safe_cv(target_train, task_type)
    model_configs = get_available_models(task_type)
    model_mapping = get_model_map()

    competition_results = {}
    best_model_key = ""
    best_model_score = -float("inf")
    best_model_instance = None

    target_label_encoder = None
    if task_type == "classification":
        target_label_encoder = LabelEncoder()
        target_train = pd.Series(target_label_encoder.fit_transform(target_train), index=target_train.index)
        known_categories = set(target_label_encoder.classes_)
        fallback_category = target_label_encoder.transform([pd.Series(target_label_encoder.classes_).mode()[0]])[0] 
        target_test = pd.Series(
            target_test.apply(lambda val: target_label_encoder.transform([val])[0] if val in known_categories else fallback_category),
            index=target_test.index
        )

    for index, (model_key, model_label) in enumerate(model_configs.items()):
        if on_progress:
            on_progress(model_label, index, len(model_configs))
            
        sampled_features_train, sampled_target_train = (
            sample_data(features_train, target_train, MAX_ROWS_SLOW)
            if model_key in SLOW_MODELS else (features_train, target_train)
        )
        
        try:
            model_instance = model_mapping[model_key]()
            param_grid = PARAM_GRIDS.get(model_key, {})
            best_parameters = {}
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="The least populated class")
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*n_jobs.*no effect")
                
                if param_grid:
                    random_search = RandomizedSearchCV(
                        model_instance, param_grid, 
                        n_iter=min(25, calculate_grid_size(param_grid)),
                        cv=cv_strategy, scoring=scorer_metric, random_state=42,
                        n_jobs=N_JOBS_LIMIT, refit=False, error_score="raise"
                    )
                    random_search.fit(sampled_features_train, sampled_target_train)
                    best_parameters = random_search.best_params_
                    cv_mean_score = float(random_search.best_score_)
                    cv_std_score = float(random_search.cv_results_["std_test_score"][random_search.best_index_])
                    model_instance.set_params(**best_parameters)
                else:
                    cross_val_scores = cross_val_score(model_instance, sampled_features_train, sampled_target_train, cv=cv_strategy, scoring=scorer_metric, n_jobs=N_JOBS_LIMIT)
                    cv_mean_score, cv_std_score = float(cross_val_scores.mean()), float(cross_val_scores.std())
                    
                fit_keywords = {}
                if task_type == "classification" and model_key in ("gradient_boosting", "xgboost"):
                    # HistGradientBoosting และ XGBoost ไม่รองรับ class_weight ใน constructor
                    # → ใช้ sample_weight ใน fit แทน เพื่อ balance imbalanced classes
                    fit_keywords["sample_weight"] = compute_sample_weight("balanced", target_train)
                model_instance.fit(features_train, target_train, **fit_keywords)

            competition_results[model_key] = {
                "label": model_label,
                "cv_score": round(cv_mean_score, 4),
                "cv_score_raw": cv_mean_score,
                "cv_std": round(cv_std_score, 4),
                "best_params": best_parameters,
                "error": None
            }
            
            if cv_mean_score > best_model_score:
                best_model_score = cv_mean_score
                best_model_key = model_key
                best_model_instance = model_instance

        except Exception as exception:
            competition_results[model_key] = {
                "label": model_label, "cv_score": None, "cv_std": None, "best_params": {}, "error": str(exception)
            }

    if best_model_instance is None:
        raise ValueError("ทุก model ล้มเหลว")

    predicted_test_target = best_model_instance.predict(features_test)
    if target_label_encoder is not None:
        target_test = pd.Series(target_label_encoder.inverse_transform(target_test), index=target_test.index)
        predicted_test_target = target_label_encoder.inverse_transform(predicted_test_target)

    return {
        "competition": competition_results, 
        "best_key": best_model_key,
        "best_label": model_configs[best_model_key], 
        "best_params": competition_results[best_model_key]["best_params"],
        "y_test": target_test, 
        "y_pred": predicted_test_target, 
        "task_type": task_type
    }