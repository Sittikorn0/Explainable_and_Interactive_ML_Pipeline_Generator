"""
ml_process/config.py
Constants, model list, hyperparameter grids
แก้ที่นี่ที่เดียวเมื่อต้องการเพิ่ม/ลบ model หรือปรับ param
"""

# ── Row caps ──────────────────────────────────────────────────
MAX_ROWS_TRAIN = 5_000   # cap ก่อน train ทุก model
MAX_ROWS_SLOW  = 2_000   # cap เพิ่มสำหรับ model ที่ช้า (kNN, SVM)

# ── Models ที่ช้ากับข้อมูลใหญ่ ────────────────────────────────
SLOW_MODELS = {"knn", "knn_regressor", "svm"}

# ── Optional library flags (set ใน runner.py) ─────────────────
# ใช้ try/import ใน runner.py แล้วส่งมา config ผ่าน flag

# ── Model display names ───────────────────────────────────────
MODELS_CLASSIFICATION = {
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "svm":                 "SVM",
    "knn":                 "kNN",
    "naive_bayes":         "Naive Bayes",
    # XGBoost / LightGBM / CatBoost เพิ่มใน runner.py ถ้าติดตั้งแล้ว
}

MODELS_REGRESSION = {
    "random_forest_regressor":     "Random Forest",
    "gradient_boosting_regressor": "Gradient Boosting",
    "linear_regression":           "Linear Regression",
    "decision_tree_regressor":     "Decision Tree",
    "knn_regressor":               "kNN",
    # XGBoost / LightGBM / CatBoost เพิ่มใน runner.py ถ้าติดตั้งแล้ว
}

# ── Hyperparameter search spaces ─────────────────────────────
# RandomizedSearchCV จะสุ่มจาก list นี้ (n_iter ≤ 8)
PARAM_GRIDS = {
    # classification
    "logistic_regression":  {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "decision_tree":        {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]},
    "random_forest":        {"n_estimators": [30, 50, 100], "max_depth": [4, 6, None]},
    "gradient_boosting":    {"max_iter": [30, 50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "svm":                  {"alpha": [0.0001, 0.001, 0.01]},
    "knn":                  {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "naive_bayes":          {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    "xgboost":              {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "lightgbm":             {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "catboost":             {"iterations": [50, 100], "depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    # regression
    "linear_regression":           {},
    "decision_tree_regressor":     {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]},
    "random_forest_regressor":     {"n_estimators": [30, 50, 100], "max_depth": [4, 6, None]},
    "gradient_boosting_regressor": {"max_iter": [30, 50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "knn_regressor":               {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "xgboost_regressor":           {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "lightgbm_regressor":          {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "catboost_regressor":          {"iterations": [50, 100], "depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
}