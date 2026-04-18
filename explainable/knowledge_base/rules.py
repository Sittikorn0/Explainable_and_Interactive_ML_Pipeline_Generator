"""
knowledge_base/rules.py
Rule definitions สำหรับทุก domain — pure data, ไม่มี logic

Condition format:
  "key": ["a","b"]           → facts[key] must be in list
  "key": {"min": x}          → facts[key] >= x
  "key": {"max": x}          → facts[key] <= x
  "key": {"min": x, "max":y} → x <= facts[key] <= y
  "key": True / False        → exact boolean match
  (missing key in conditions = don't-care / skip check)

Priority: ตัวเลขน้อย = เช็คก่อน (override rule สำคัญกว่า)
"""

RULES: list[dict] = [

    # ══════════════════════════════════════════════════════════════
    # Domain: missing_value  (อ้างอิง Topic 7 — Missing Data)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "MISS_001",
        "domain": "missing_value",
        "priority": 1,
        "conditions": {"missing_pct": {"min": 0.5}},
        "action": "drop_column",
        "explanation": (
            "คอลัมน์นี้ขาดข้อมูลมากกว่า 50% — ข้อมูลที่หายไปมากขนาดนี้ทำให้ imputation "
            "ไม่น่าเชื่อถือ เพราะ 'แต่งเติม' ข้อมูลมากกว่าที่มีจริง การ drop column "
            "จึงปลอดภัยกว่าสำหรับ model"
        ),
        "reference": "Topic 7 — Missing Data: Listwise Deletion",
        "confidence": 0.95,
    },
    {
        "id": "MISS_002",
        "domain": "missing_value",
        "priority": 5,
        "conditions": {"dtype": ["datetime"]},
        "action": "forward_fill",
        "explanation": (
            "ข้อมูล datetime มักเรียงตามเวลา — Forward Fill ใช้ค่าแถวก่อนหน้ามาแทน "
            "ซึ่ง consistent กับธรรมชาติของ time-series data มากกว่าการ impute ด้วย mean"
        ),
        "reference": "Topic 7 — Missing Data / Topic 10 — Time Series",
        "confidence": 0.9,
    },
    {
        "id": "MISS_003",
        "domain": "missing_value",
        "priority": 10,
        "conditions": {"dtype": ["string", "bool", "category"]},
        "action": "most_frequent",
        "explanation": (
            "ข้อมูล categorical ไม่มี numerical mean/median — Most Frequent (Mode) "
            "เป็นตัวแทนที่เหมาะสมที่สุด เพราะสะท้อนค่าที่พบบ่อยสุดใน dataset"
        ),
        "reference": "Topic 7 — Missing Data: Mode Imputation",
        "confidence": 0.92,
    },
    {
        "id": "MISS_004",
        "domain": "missing_value",
        "priority": 20,
        "conditions": {"dtype": ["float"], "skewness_abs": {"max": 0.5}},
        "action": "mean",
        "explanation": (
            "ข้อมูล continuous กระจายใกล้ Normal (|skewness| ≤ 0.5) — "
            "Mean เป็นตัวแทนที่ดีที่สุดสำหรับ symmetric distribution "
            "เพราะ minimize squared error จากทุก data point"
        ),
        "reference": "Topic 7 — Missing Data: Mean Imputation / Topic 2 — Basic Statistical Description",
        "confidence": 0.88,
    },
    {
        "id": "MISS_005",
        "domain": "missing_value",
        "priority": 25,
        "conditions": {"dtype": ["float"], "skewness_abs": {"min": 0.5}},
        "action": "median",
        "explanation": (
            "ข้อมูล continuous มี skewness สูง (|skewness| > 0.5) — "
            "Median ทนทานต่อ outlier ดีกว่า Mean เพราะ Mean ถูกดึงไปทาง tail ที่ยาว "
            "ทำให้ค่าไม่ represent ข้อมูลส่วนใหญ่"
        ),
        "reference": "Topic 7 — Missing Data: Median Imputation / Topic 8 — Outlier Detection",
        "confidence": 0.9,
    },
    {
        "id": "MISS_006",
        "domain": "missing_value",
        "priority": 30,
        "conditions": {"dtype": ["int"]},
        "action": "median (rounded)",
        "explanation": (
            "ข้อมูล integer ต้องการค่าที่เป็นจำนวนเต็มเท่านั้น — "
            "Median (Rounded) รักษา data type เดิมและทนทานต่อ outlier "
            "เหมาะกับข้อมูลเช่น อายุ จำนวนสินค้า คะแนน"
        ),
        "reference": "Topic 7 — Missing Data: Median Imputation",
        "confidence": 0.85,
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: outlier  (อ้างอิง Topic 8 — Outlier Detection)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "OUT_001",
        "domain": "outlier",
        "priority": 5,
        "conditions": {"outlier_pct": {"min": 20.0}},
        "action": "clip",
        "explanation": (
            "พบ outlier มากกว่า 20% ของข้อมูล — อาจไม่ใช่ข้อผิดพลาด "
            "แต่เป็น natural distribution ของข้อมูลชุดนี้ "
            "แนะนำ Clip (จำกัดค่า) แทน Drop Rows เพื่อไม่สูญเสียข้อมูลมากเกินไป"
        ),
        "reference": "Topic 8 — Outlier Detection: Handling Strategy",
        "confidence": 0.75,
    },
    {
        "id": "OUT_002",
        "domain": "outlier",
        "priority": 10,
        "conditions": {"skewness_abs": {"max": 0.5}},
        "action": "clip",
        "explanation": (
            "ข้อมูลกระจายใกล้ Normal (|skewness| ≤ 0.5) — "
            "ใช้ Z-Score ตรวจจับ outlier (ค่านอกช่วง ±3 SD = 3-sigma rule) "
            "Clip ปรับค่าให้อยู่ในขอบเขตโดยไม่ลบแถว เหมาะเมื่อ outlier อาจเป็นข้อมูลจริง"
        ),
        "reference": "Topic 8 — Outlier Detection: Z-Score Method / Topic 2 — Normal Distribution",
        "confidence": 0.88,
    },
    {
        "id": "OUT_003",
        "domain": "outlier",
        "priority": 15,
        "conditions": {"skewness_abs": {"min": 0.5}},
        "action": "clip",
        "explanation": (
            "ข้อมูลมี skewness สูง (|skewness| > 0.5) — "
            "ใช้ IQR ตรวจจับ outlier (ค่านอกช่วง Q1−1.5×IQR ถึง Q3+1.5×IQR) "
            "เพราะข้อมูลไม่กระจายแบบ Normal ทำให้ Z-Score ไม่น่าเชื่อถือ"
        ),
        "reference": "Topic 8 — Outlier Detection: IQR Method",
        "confidence": 0.9,
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: encoding  (อ้างอิง Topic 9 — Data Transformation)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "ENC_001",
        "domain": "encoding",
        "priority": 5,
        "conditions": {"cardinality_ratio": {"min": 0.5}},
        "action": "drop_column",
        "explanation": (
            "คอลัมน์นี้มี unique value มากกว่า 50% ของจำนวนแถว — "
            "แทบทุก row มีค่าต่างกัน (เช่น ID, ชื่อ, URL) "
            "model ไม่สามารถเรียนรู้ pattern ได้ การ One-hot จะสร้างคอลัมน์มหาศาล "
            "ที่ไม่มีประโยชน์"
        ),
        "reference": "Topic 9 — Data Transformation: Feature Engineering",
        "confidence": 0.95,
    },
    {
        "id": "ENC_002",
        "domain": "encoding",
        "priority": 10,
        "conditions": {"cardinality": {"max": 2}},
        "action": "label_encoding",
        "explanation": (
            "คอลัมน์นี้มีเพียง 2 categories (binary) — "
            "Label Encoding (0/1) เพียงพอและไม่สิ้นเปลืองคอลัมน์ "
            "One-hot สำหรับ binary จะสร้างคอลัมน์ซ้ำซ้อน (dummy variable trap)"
        ),
        "reference": "Topic 9 — Data Transformation: Label Encoding",
        "confidence": 0.97,
    },
    {
        "id": "ENC_003",
        "domain": "encoding",
        "priority": 20,
        "conditions": {"cardinality": {"min": 3, "max": 10}},
        "action": "one_hot_encoding",
        "explanation": (
            "คอลัมน์นี้มี 3–10 categories (low cardinality) — "
            "One-hot Encoding เหมาะที่สุดเพราะไม่สร้าง ordinal relationship ที่ไม่มีอยู่จริง "
            "เช่น Label Encoding ทำให้ 'Bangkok=1, Chiang Mai=2' ซึ่ง model อาจเข้าใจผิดว่า "
            "Chiang Mai 'มากกว่า' Bangkok"
        ),
        "reference": "Topic 9 — Data Transformation: One-Hot Encoding",
        "confidence": 0.93,
    },
    {
        "id": "ENC_004",
        "domain": "encoding",
        "priority": 30,
        "conditions": {"cardinality": {"min": 11, "max": 20}},
        "action": "one_hot_encoding",
        "explanation": (
            "คอลัมน์นี้มี 11–20 categories — One-hot ยังใช้ได้ "
            "แต่จะเพิ่มจำนวนคอลัมน์มากขึ้น หากต้องการลด dimensionality "
            "อาจพิจารณา Label Encoding หรือ Target Encoding แทน"
        ),
        "reference": "Topic 9 — Data Transformation: One-Hot Encoding",
        "confidence": 0.75,
    },
    {
        "id": "ENC_005",
        "domain": "encoding",
        "priority": 40,
        "conditions": {"cardinality": {"min": 21}},
        "action": "label_encoding",
        "explanation": (
            "คอลัมน์นี้มี categories มากกว่า 20 (high cardinality) — "
            "One-hot จะสร้างคอลัมน์จำนวนมาก ทำให้เกิด Curse of Dimensionality "
            "Label Encoding ลด dimensionality แต่สร้าง ordinal relationship โดยปริยาย "
            "ควรพิจารณาว่า model ที่ใช้ sensitive ต่อ ordinal หรือไม่"
        ),
        "reference": "Topic 9 — Data Transformation: Encoding High-Cardinality",
        "confidence": 0.78,
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: scaling  (อ้างอิง Topic 9 — Data Transformation)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "SCL_001",
        "domain": "scaling",
        "priority": 5,
        "conditions": {"no_numeric": True},
        "action": "no_scaling",
        "explanation": "ไม่มี numeric feature — ไม่จำเป็นต้องทำ scaling",
        "reference": "Topic 9 — Data Transformation",
        "confidence": 1.0,
    },
    {
        "id": "SCL_002",
        "domain": "scaling",
        "priority": 10,
        "conditions": {"has_outliers": True},
        "action": "robust_scaler",
        "explanation": (
            "พบ outlier ใน dataset — Robust Scaler ใช้ Median และ IQR แทน Mean/Std "
            "ทำให้ค่า extreme ไม่ดึง scale ให้เบี้ยว "
            "Standard Scaler จะถูก outlier กดดันทำให้ค่าปกติส่วนใหญ่ถูกบีบให้อยู่ในช่วงแคบ"
        ),
        "reference": "Topic 9 — Data Transformation: Robust Scaling / Topic 8 — Outlier Detection",
        "confidence": 0.92,
    },
    {
        "id": "SCL_003",
        "domain": "scaling",
        "priority": 20,
        "conditions": {"has_heavy_skew": True, "has_outliers": False},
        "action": "log_transform",
        "explanation": (
            "ข้อมูลมี skewness รุนแรง (|skew| > 2) และไม่มี outlier — "
            "Log Transform (log1p) ลด skewness ก่อน แล้วตาม Standard Scaler "
            "เหมาะกับข้อมูล long-tail เช่น รายได้ ราคา จำนวน transaction"
        ),
        "reference": "Topic 9 — Data Transformation: Log Transformation",
        "confidence": 0.87,
    },
    {
        "id": "SCL_004",
        "domain": "scaling",
        "priority": 30,
        "conditions": {"is_skewed": True, "has_outliers": False, "has_heavy_skew": False},
        "action": "minmax_scaler",
        "explanation": (
            "ข้อมูลมี skewness ปานกลาง (1 < |skew| ≤ 2) — "
            "MinMax Scaler แปลงค่าให้อยู่ในช่วง [0, 1] "
            "เหมาะกับข้อมูลที่ไม่กระจายแบบ Normal เพราะ Standard Scaler "
            "สมมติ Normal distribution"
        ),
        "reference": "Topic 9 — Data Transformation: Min-Max Normalization",
        "confidence": 0.83,
    },
    {
        "id": "SCL_005",
        "domain": "scaling",
        "priority": 40,
        "conditions": {"has_outliers": False, "is_skewed": False},
        "action": "standard_scaler",
        "explanation": (
            "ข้อมูลกระจายใกล้ Normal และไม่มี outlier — "
            "Standard Scaler (Z-score normalization) เหมาะที่สุด "
            "แปลงให้ mean=0, std=1 รักษา relative distance ระหว่าง data points"
        ),
        "reference": "Topic 9 — Data Transformation: Standardization / Topic 2 — Normal Distribution",
        "confidence": 0.9,
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: model_selection
    # ══════════════════════════════════════════════════════════════

    {
        "id": "MDL_001",
        "domain": "model_selection",
        "priority": 5,
        "conditions": {"n_samples": {"max": 500}},
        "action": "prefer_simple",
        "explanation": (
            "Dataset เล็กมาก (< 500 rows) — model ซับซ้อน (Ensemble, XGBoost) เสี่ยง overfit "
            "เพราะไม่มีข้อมูลเพียงพอให้เรียนรู้ pattern ทั่วไป "
            "Decision Tree หรือ Logistic Regression interpretable กว่าและ generalize ดีกว่า"
        ),
        "reference": "Topic 2 — Getting to Know Your Data: Sample Size",
        "confidence": 0.85,
    },
    {
        "id": "MDL_002",
        "domain": "model_selection",
        "priority": 10,
        "conditions": {"n_samples": {"min": 10000}},
        "action": "prefer_boosting",
        "explanation": (
            "Dataset ใหญ่ (≥ 10,000 rows) — Gradient Boosting / LightGBM / XGBoost "
            "ให้ผลดีที่สุดบน tabular data ขนาดใหญ่ "
            "LightGBM เร็วที่สุดในกลุ่มนี้เพราะใช้ histogram-based algorithm"
        ),
        "reference": "Topic 2 — Getting to Know Your Data",
        "confidence": 0.88,
    },
    {
        "id": "MDL_003",
        "domain": "model_selection",
        "priority": 15,
        "conditions": {"class_imbalance_ratio": {"min": 3.0}, "task_type": ["classification"]},
        "action": "warn_imbalance",
        "explanation": (
            "พบ class imbalance สูง (ratio ≥ 3:1) — "
            "Accuracy จะสูงเทียมโดยที่ model แค่ทำนาย majority class เสมอ "
            "ควรดู F1 (Macro) และ Recall ของ minority class เป็นหลัก "
            "พิจารณาใช้ class_weight='balanced' หรือ SMOTE oversampling"
        ),
        "reference": "Topic 2 — Getting to Know Your Data: Class Distribution",
        "confidence": 0.92,
    },
    {
        "id": "MDL_004",
        "domain": "model_selection",
        "priority": 20,
        "conditions": {"n_features": {"min": 50}},
        "action": "prefer_regularized",
        "explanation": (
            "Dataset มี feature จำนวนมาก (≥ 50) — "
            "Random Forest และ Gradient Boosting จัดการ high-dimensional data ได้ดี "
            "เพราะ built-in feature selection ผ่าน information gain "
            "Logistic Regression กับ Regularization (L1/L2) ก็เหมาะเมื่อต้องการ interpretability"
        ),
        "reference": "Topic 9 — Data Transformation: Dimensionality",
        "confidence": 0.82,
    },
    {
        "id": "MDL_005",
        "domain": "model_selection",
        "priority": 30,
        "conditions": {"task_type": ["regression"]},
        "action": "use_r2_rmse",
        "explanation": (
            "Task เป็น Regression — วัดผลด้วย R² Score และ RMSE "
            "R² อธิบายว่า model อธิบาย variance ของ target ได้กี่ % (1.0 = perfect) "
            "RMSE วัด error เฉลี่ยในหน่วยเดียวกับ target"
        ),
        "reference": "Topic 2 — Getting to Know Your Data: Regression Metrics",
        "confidence": 0.95,
    },
]
