"""ml_process/config.py — constants ทั้งหมด"""

MAX_ROWS_TRAIN = 50_000
MAX_ROWS_SLOW  = 5_000
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
    "random_forest":               "ใช้ต้นไม้ตัดสินใจหลายต้นร่วมกันทำนาย (Ensemble) เพื่อให้ได้ผลลัพธ์ที่แม่นยำและเสถียรสูง",
    "gradient_boosting":           "เรียนรู้แบบลำดับขั้น โดยแต่ละโมเดลจะแก้ไขข้อผิดพลาดจากตัวก่อนหน้าเพื่อความแม่นยำ",
    "logistic_regression":         "หาความน่าจะเป็นด้วยความสัมพันธ์เชิงเส้น เหมาะเป็นพื้นฐานการเรียนรู้โมเดลจำแนกประเภท",
    "decision_tree":               "จำลองตรรกะการตัดสินใจเป็นลำดับขั้น (Hierarchical Rules) ที่มนุษย์สามารถตีความได้ง่าย",
    "svm":                         "หาเส้นแบ่งข้อมูลที่มีช่องว่างกว้างที่สุด (Maximum Margin) เพื่อแยกประเภทข้อมูลที่มีความซับซ้อน",
    "knn":                         "จำแนกข้อมูลตามความเหมือน โดยอ้างอิงจากกลุ่มข้อมูลตัวอย่างที่อยู่ใกล้เคียงที่สุด",
    "naive_bayes":                 "ใช้หลักความน่าจะเป็นพื้นฐาน (Bayes' Theorem) ทำงานได้เร็วและมีประสิทธิภาพกับข้อมูลขนาดใหญ่",
    "xgboost":                     "อัลกอริทึมขั้นสูงที่ใช้เทคนิค Gradient Boosting แบบเพิ่มประสิทธิภาพเพื่อความแม่นยำสูงสุด",
    "lightgbm":                    "เทคนิค Boosting ที่เน้นความรวดเร็วในการฝึกฝนโมเดล เหมาะสำหรับใช้สอนกับชุดข้อมูลขนาดใหญ่",
    "catboost":                    "โดดเด่นในการจัดการตัวแปรเชิงหมวดหมู่ (Categorical Features) โดยไม่ต้องเตรียมข้อมูลซับซ้อน",
    "linear_regression":           "การพยากรณ์ค่าด้วยสมการเชิงเส้น เป็นโมเดลพื้นฐานที่สำคัญที่สุดในงาน Regression",
    "decision_tree_regressor":     "พยากรณ์ค่าตัวเลขตามลำดับเงื่อนไข ช่วยให้เห็นโครงสร้างการตัดสินใจของข้อมูลได้ชัดเจน",
    "random_forest_regressor":     "ใช้ต้นไม้ตัดสินใจหลายต้นรวมกันเพื่อพยากรณ์ค่า ช่วยให้ผลลัพธ์มีความเสถียรและทนทานต่อสัญญาณรบกวน",
    "gradient_boosting_regressor": "พัฒนาความแม่นยำในการพยากรณ์ตัวเลขผ่านการเรียนรู้จากข้อผิดพลาดแบบวนซ้ำ",
    "knn_regressor":               "ประมาณค่าตัวเลขโดยอ้างอิงจากค่าเฉลี่ยของกลุ่มตัวอย่างที่มีลักษณะใกล้เคียงกันที่สุด",
    "xgboost_regressor":           "โมเดลพยากรณ์เชิงตัวเลขประสิทธิภาพสูง ที่มีการควบคุมความซับซ้อนเพื่อป้องกันโมเดลจำจำข้อมูลเกิน",
    "lightgbm_regressor":          "ใช้โครงสร้างการเรียนรู้แบบพิเศษที่ช่วยให้ประมวลผลงานพยากรณ์เชิงตัวเลขได้รวดเร็วเป็นพิเศษ",
    "catboost_regressor":          "จัดการตัวแปรเชิงกลุ่มได้อย่างมีประสิทธิภาพในงานพยากรณ์ ช่วยลดขั้นตอนการเตรียมข้อมูลลง",
}
MODEL_WHY = {
    "Random Forest":
        "ใช้วิธีการ Ensemble Learning โดยรวมผลลัพธ์จากต้นไม้หลายต้นเข้าด้วยกันเพื่อหาค่าเฉลี่ยหรือโหวต ช่วยเพิ่มความแม่นยำและทนทานต่อข้อมูลที่ผิดปกติ (Noise)",
    "Gradient Boosting":
        "สร้างโมเดลแบบ Iterative โดยแต่ละขั้นตอนจะมุ่งเน้นการลดส่วนตกค้าง (Residuals) หรือข้อผิดพลาดของโมเดลก่อนหน้าเพื่อให้ได้ความแม่นยำสูงสุด",
    "Logistic Regression":
        "คำนวณหาความสัมพันธ์เชิงเส้นเพื่อระบุความน่าจะเป็นในการแยกประเภทข้อมูล เป็นพื้นฐานสำคัญในการเรียนรู้เรื่องการแบ่งกลุ่มข้อมูล",
    "Decision Tree":
        "สร้างลำดับกฎการตัดสินใจในรูปแบบโครงสร้างต้นไม้ (Tree Structure) ที่เลียนแบบตรรกะของมนุษย์ ทำให้ง่ายต่อการตีความและวิเคราะห์เงื่อนไข",
    "SVM":
        "หาขอบเขตการตัดสินใจที่กว้างที่สุด (Maximum Margin) เพื่อแยกประเภทข้อมูลที่ซับซ้อน เหมาะสำหรับใช้สอนเรื่องการแบ่งแยกข้อมูลในมิติสูง",
    "kNN":
        "ใช้หลักการเปรียบเทียบความคล้ายคลึงของข้อมูล (Instance-based Learning) โดยอาศัยกลุ่มตัวอย่างรอบข้างในการตัดสินใจ",
    "Naive Bayes":
        "ใช้อัลกอริทึมที่อ้างอิงจากทฤษฎีความน่าจะเป็นของเบย์ (Bayes' Theorem) มีประสิทธิภาพสูงมากสำหรับชุดข้อมูลที่มีฟีเจอร์เป็นอิสระต่อกัน",
    "XGBoost":
        "ระบบ Gradient Boosting ที่ผ่านการปรับปรุงประสิทธิภาพ (Optimization) และมีกระบวนการควบคุมความซับซ้อน (Regularization) เพื่อความแม่นยำ",
    "LightGBM":
        "ใช้สถาปัตยกรรมที่เน้นความเร็วและประหยัดทรัพยากร เหมาะสำหรับการสาธิตกับชุดข้อมูลขนาดใหญ่ในระดับอุตสาหกรรม",
    "CatBoost":
        "นวัตกรรมการจัดการตัวแปรชนิดกลุ่ม (Categorical Features) โดยตรง ช่วยลดภาระในการทำ Feature Engineering ขั้นพื้นฐาน",
    "Linear Regression":
        "พื้นฐานสำคัญของสถิติและการเรียนรู้ของเครื่อง ใช้สมการเชิงเส้นในการอธิบายความสัมพันธ์ระหว่างตัวแปรต้นและตัวแปรตามที่เป็นตัวเลข",
}

# หลักการออกแบบ PARAM_GRIDS
# 1. ทุก grid มี ≤ 12 combinations → n_iter=10 ครอบคลุมได้ดี
# 2. ค่า default ที่ดีอยู่กลาง grid เสมอ (ไม่ใช่ขอบ)
# 3. ใช้ log scale สำหรับ C, alpha, learning_rate
# 4. liblinear ถูกตัดออก เพราะไม่รองรับ multiclass (n_classes ≥ 3)
PARAM_GRIDS = {
    # Classification

    # C: log scale 0.01→100, lbfgs รองรับ multiclass + L2 (default)
    # ตัด saga ออก: saga ต้องการ max_iter สูงมากและมักเกิด ConvergenceWarning
    # combinations: 5
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10, 100],
    },

    # max_depth: None=unbound ไว้เป็น ceiling, min_samples_split ป้องกัน overfit
    # combinations: 4×3 = 12
    "decision_tree": {
        "max_depth":         [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
    },

    # n_estimators: 50-200 (มากกว่า 200 ให้ผลดีขึ้นน้อยมาก)
    # combinations: 3×3 = 9
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth":    [4, 6, None],
    },

    # learning_rate+max_iter มี trade-off กัน rate ต่ำต้องการ iter มาก
    # combinations: 3×3×3 = 27 → n_iter จะสุ่ม 10 ชุด
    "gradient_boosting": {
        "max_iter":     [50, 100, 200],
        "max_depth":    [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },

    # alpha log scale: ค่าน้อย=regularize มาก
    # combinations: 4
    "svm": {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
    },

    # n_neighbors เลขคี่เสมอ ป้องกัน tie ใน binary
    # combinations: 4×2 = 8
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights":     ["uniform", "distance"],
    },

    # var_smoothing: ป้องกัน log(0), log scale
    # combinations: 3
    "naive_bayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },

    # XGBoost/LightGBM/CatBoost: combinations: 2×3×3 = 18
    "xgboost":  {
        "n_estimators": [50, 100],
        "max_depth":    [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },
    "lightgbm": {
        "n_estimators": [50, 100],
        "max_depth":    [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },
    "catboost": {
        "iterations":   [50, 100],
        "depth":        [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },

    # Regression

    # Linear Regression: closed-form solution ไม่มี hyperparameter
    "linear_regression": {},

    # Regression trees: เหมือน classification แต่ใช้ MSE แทน gini
    # combinations: 4×3 = 12
    "decision_tree_regressor": {
        "max_depth":         [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
    },

    # combinations: 3×3 = 9
    "random_forest_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth":    [4, 6, None],
    },

    # combinations: 3×3×3 = 27
    "gradient_boosting_regressor": {
        "max_iter":     [50, 100, 200],
        "max_depth":    [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },

    # combinations: 4×2 = 8
    "knn_regressor": {
        "n_neighbors": [3, 5, 7, 11],
        "weights":     ["uniform", "distance"],
    },

    "xgboost_regressor": {
        "n_estimators": [50, 100],
        "max_depth":    [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },
    "lightgbm_regressor": {
        "n_estimators": [50, 100],
        "max_depth":    [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },
    "catboost_regressor": {
        "iterations":   [50, 100],
        "depth":        [3, 4, 6],
        "learning_rate":[0.05, 0.1, 0.2],
    },
}