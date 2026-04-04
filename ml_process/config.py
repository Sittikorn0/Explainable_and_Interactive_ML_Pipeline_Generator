"""ml_process/config.py — constants ทั้งหมด"""

MAX_ROWS_TRAIN = 5_000
MAX_ROWS_SLOW  = 2_000
SLOW_MODELS    = {"knn", "knn_regressor", "svm"}

MODELS_CLF = {
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "svm":                 "SVM",
    "knn":                 "kNN",
    "naive_bayes":         "Naive Bayes",
}
MODELS_REG = {
    "random_forest_regressor":     "Random Forest",
    "gradient_boosting_regressor": "Gradient Boosting",
    "linear_regression":           "Linear Regression",
    "decision_tree_regressor":     "Decision Tree",
    "knn_regressor":               "kNN",
}
MODEL_DESC = {
    "random_forest":               "ทนทานต่อ noise ดี เหมาะเป็น anchor",
    "gradient_boosting":           "แก้ error ทีละขั้น มักให้ accuracy สูงสุด",
    "logistic_regression":         "เร็ว อธิบายได้ง่าย ช่วย balance ensemble",
    "decision_tree":               "เห็น decision boundary ชัด เพิ่ม diversity",
    "svm":                         "แข็งแกร่งกับ high-dimensional data",
    "knn":                         "ใช้ k เพื่อนบ้านที่ใกล้ที่สุดในการตัดสินใจ",
    "naive_bayes":                 "เร็วมาก เหมาะเป็น baseline",
    "xgboost":                     "Boosting ที่แม่นยำสูง ชนะ Kaggle หลายรายการ",
    "lightgbm":                    "เร็วกว่า XGBoost เหมาะกับข้อมูลใหญ่",
    "catboost":                    "จัดการ categorical ได้ดีโดยไม่ต้อง encode",
    "linear_regression":           "เร็ว stable เหมาะกับข้อมูล linear",
    "decision_tree_regressor":     "จับ non-linear ได้ เพิ่ม diversity",
    "random_forest_regressor":     "ทนทาน เหมาะเป็น anchor",
    "gradient_boosting_regressor": "accuracy สูงสุดสำหรับ regression",
    "knn_regressor":               "ใช้ k เพื่อนบ้านในการประมาณค่า",
    "xgboost_regressor":           "Boosting แม่นยำสูงสำหรับ regression",
    "lightgbm_regressor":          "เร็วกว่า XGBoost สำหรับ regression",
    "catboost_regressor":          "จัดการ categorical ได้ดีสำหรับ regression",
}
MODEL_WHY = {
    "Random Forest":
        "เป็น Ensemble ของ Decision Tree หลายต้น แต่ละต้น train บน subset ของข้อมูล ทำให้ทนทานต่อ noise และ overfitting",
    "Gradient Boosting":
        "สร้าง model ทีละตัวโดยแก้ error ของตัวก่อนหน้า มักให้ผลดีที่สุดกับ tabular data",
    "Logistic Regression":
        "เรียนรู้ decision boundary เป็นเส้นตรง เร็วและ interpretable เหมาะเมื่อมี linear relationship",
    "Decision Tree":
        "สร้าง decision rules แบบ if-then-else เหมาะเมื่อข้อมูลมี pattern ชัดเจน",
    "SVM":
        "หา hyperplane ที่แยก class ได้ดีที่สุด เหมาะกับ high-dimensional data",
    "kNN":
        "classify โดยดูจาก k เพื่อนบ้านที่ใกล้ที่สุด เหมาะเมื่อข้อมูลมี local pattern",
    "Naive Bayes":
        "ใช้ probability theorem เร็วมาก เหมาะกับ feature ที่เป็นอิสระจากกัน",
    "XGBoost":
        "Gradient Boosting ที่ optimize แล้ว มี regularization ป้องกัน overfitting",
    "LightGBM":
        "Gradient Boosting ที่เร็วที่สุด ใช้ histogram-based algorithm เหมาะข้อมูลใหญ่",
    "CatBoost":
        "จัดการ categorical features โดยตรงโดยไม่ต้อง encode",
    "Linear Regression":
        "fit เส้นตรงผ่านข้อมูล เร็วและ interpretable เหมาะเมื่อ target มี linear relationship",
}
PARAM_GRIDS = {
    "logistic_regression":         {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "decision_tree":               {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]},
    "random_forest":               {"n_estimators": [30, 50, 100], "max_depth": [4, 6, None]},
    "gradient_boosting":           {"max_iter": [30, 50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "svm":                         {"alpha": [0.0001, 0.001, 0.01]},
    "knn":                         {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "naive_bayes":                 {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    "xgboost":                     {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "lightgbm":                    {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "catboost":                    {"iterations": [50, 100], "depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "linear_regression":           {},
    "decision_tree_regressor":     {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]},
    "random_forest_regressor":     {"n_estimators": [30, 50, 100], "max_depth": [4, 6, None]},
    "gradient_boosting_regressor": {"max_iter": [30, 50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "knn_regressor":               {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
    "xgboost_regressor":           {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "lightgbm_regressor":          {"n_estimators": [50, 100], "max_depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
    "catboost_regressor":          {"iterations": [50, 100], "depth": [3, 4, 6], "learning_rate": [0.05, 0.1, 0.2]},
}