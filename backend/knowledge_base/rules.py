"""
knowledge_base/rules.py
Rule definitions สำหรับทุก domain  pure data, ไม่มี logic

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
    # Domain: missing_value
    # ══════════════════════════════════════════════════════════════

    {
        "id": "MIS_001",
        "domain": "missing_value",
        "priority": 5,
        "conditions": {"missing_ratio": {"max": 0.0}},
        "action": "no_action",
        "explanation": "ไม่มีค่าที่หายไปในคอลัมน์นี้  ไม่จำเป็นต้องทำ imputation",
    },
    {
        "id": "MIS_002",
        "domain": "missing_value",
        "priority": 10,
        "conditions": {"missing_ratio": {"min": 0.5}},
        "action": "drop_column",
        "explanation": (
            "คอลัมน์นี้มีค่าหายไปมากกว่า 50%  การ impute จะสร้างข้อมูลสังเคราะห์มากเกินไป "
            "ซึ่งอาจบิดเบือน pattern จริงในข้อมูล การลบคอลัมน์ทิ้งปลอดภัยกว่า"
        ),
    },
    {
        "id": "MIS_003",
        "domain": "missing_value",
        "priority": 20,
        "conditions": {"dtype": ["numeric"], "has_outliers": True},
        "action": "median_impute",
        "explanation": (
            "คอลัมน์ตัวเลขนี้มี outlier  Median ทนต่อ outlier ดีกว่า Mean "
            "เพราะ outlier จะดึง Mean ให้เบี้ยว ทำให้ค่าที่ใช้ impute ไม่ represent ข้อมูลส่วนใหญ่"
        ),
    },
    {
        "id": "MIS_004",
        "domain": "missing_value",
        "priority": 30,
        "conditions": {"dtype": ["numeric"], "has_outliers": False, "is_skewed": False},
        "action": "mean_impute",
        "explanation": (
            "คอลัมน์ตัวเลขนี้ไม่มี outlier และกระจายใกล้ Normal  "
            "Mean Imputation เหมาะที่สุดเพราะรักษา distribution ของข้อมูล "
            "และ Mean เป็นตัวแทนที่ดีเมื่อข้อมูลไม่มี extreme values"
        ),
    },
    {
        "id": "MIS_005",
        "domain": "missing_value",
        "priority": 35,
        "conditions": {"dtype": ["numeric"], "has_outliers": False, "is_skewed": True},
        "action": "median_impute",
        "explanation": (
            "คอลัมน์ตัวเลขนี้มี skewness สูงแต่ไม่มี outlier  "
            "Median ดีกว่า Mean สำหรับข้อมูล skewed เพราะ Median ไม่ถูกดึงโดย tail "
            "ทำให้ค่าที่ impute อยู่ใกล้กับ 'กลาง' ของข้อมูลจริงๆ"
        ),
    },
    {
        "id": "MIS_006",
        "domain": "missing_value",
        "priority": 40,
        "conditions": {"dtype": ["categorical"]},
        "action": "mode_impute",
        "explanation": (
            "คอลัมน์ categorical นี้ไม่สามารถใช้ Mean/Median ได้  "
            "Mode (ค่าที่พบบ่อยที่สุด) เป็นตัวเลือกที่สมเหตุสมผลที่สุด "
            "เพราะแทนค่าด้วยสิ่งที่มีโอกาสเกิดขึ้นมากที่สุดในข้อมูล"
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: outlier  (treatment strategy: clip vs drop rows)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "OUT_001",
        "domain": "outlier",
        "priority": 5,
        "conditions": {"outlier_ratio": {"max": 0.0}},
        "action": "no_action",
        "explanation": "ไม่พบ outlier ในคอลัมน์นี้  ไม่จำเป็นต้องทำการจัดการ",
    },
    {
        "id": "OUT_002",
        "domain": "outlier",
        "priority": 10,
        "conditions": {"outlier_ratio": {"min": 0.001, "max": 0.01}},
        "action": "clip",
        "explanation": (
            "พบ outlier น้อยมาก (< 1% ของข้อมูล)  จำนวนน้อยเกินไปที่จะ drop โดยไม่เสียใจ "
            "Clipping ปรับค่าให้อยู่ในขอบเขตโดยไม่สูญเสียแถวข้อมูล"
        ),
    },
    {
        "id": "OUT_003",
        "domain": "outlier",
        "priority": 20,
        "conditions": {"outlier_ratio": {"min": 0.01, "max": 0.05}},
        "action": "drop rows",
        "explanation": (
            "พบ outlier น้อย (1-5% ของข้อมูล)  มักเป็นข้อมูลที่บันทึกผิดพลาดหรือ noise "
            "การลบแถวเหล่านี้ทิ้งสูญเสียข้อมูลน้อยมากและทำให้ dataset สะอาดขึ้น"
        ),
    },
    {
        "id": "OUT_004",
        "domain": "outlier",
        "priority": 30,
        "conditions": {"outlier_ratio": {"min": 0.05}},
        "action": "clip",
        "explanation": (
            "พบ outlier มาก (> 5% ของข้อมูล)  การลบแถวจะสูญเสียข้อมูลมากเกินไป "
            "Clipping บีบค่าที่เกินขอบเขตให้อยู่ในเกณฑ์ รักษาจำนวนแถวไว้ครบ "
            "และยังคงสะท้อน pattern จริงของข้อมูล"
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: outlier_detection  (วิธีตรวจจับ: Z-Score vs IQR)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "ODT_001",
        "domain": "outlier_detection",
        "priority": 5,
        "conditions": {"skewness_abs": {"min": 0.5}},
        "action": "iqr",
        "explanation": (
            "ข้อมูลเบ้ (|Skew| ≥ 0.5)  ใช้ IQR Method (Q1 - 1.5×IQR, Q3 + 1.5×IQR) "
            "เพราะ IQR อิงค่า Median และ Quartile ซึ่งไม่ถูกรบกวนโดย tail ที่ยาว "
            "Z-Score ใช้ Mean/Std ซึ่งถูกดึงโดย skewness ทำให้ขอบเขตคลาดเคลื่อน"
        ),
    },
    {
        "id": "ODT_002",
        "domain": "outlier_detection",
        "priority": 10,
        "conditions": {"skewness_abs": {"max": 0.5}},
        "action": "zscore",
        "explanation": (
            "ข้อมูลกระจายสมมาตร (|Skew| < 0.5)  ใช้ Z-Score Method (mean ± 3σ) "
            "เพราะข้อมูลใกล้ Normal Distribution ทำให้ Mean และ Std เป็นตัวแทนที่ดี "
            "ค่าที่ห่างจาก mean เกิน 3 standard deviation ถือเป็น outlier"
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: column_drop  (Drop Column suggestion ในหน้า Cleaning)
    # ══════════════════════════════════════════════════════════════

    {
        "id": "CDR_001",
        "domain": "column_drop",
        "priority": 5,
        "conditions": {"is_constant": True},
        "action": "drop",
        "explanation": (
            "คอลัมน์นี้มีค่าเดียวตลอด (Constant Column)  "
            "โมเดลไม่สามารถเรียนรู้ Pattern ใดได้จากข้อมูลที่ไม่มีความแตกต่างกันเลย "
            "การเก็บคอลัมน์นี้ไว้จะสิ้นเปลืองหน่วยความจำโดยไม่มีประโยชน์"
        ),
    },
    {
        "id": "CDR_002",
        "domain": "column_drop",
        "priority": 10,
        "conditions": {"missing_ratio": {"min": 0.8}},
        "action": "drop",
        "explanation": (
            "คอลัมน์นี้มีค่าว่างมากกว่า 80%  การเติมค่า (Imputation) จะสร้างข้อมูลสังเคราะห์ "
            "มากเกินไปซึ่งอาจบิดเบือน Pattern จริง การตัดคอลัมน์ทิ้งปลอดภัยกว่า"
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: feature_selection
    # ══════════════════════════════════════════════════════════════

    {
        "id": "FSL_001",
        "domain": "feature_selection",
        "priority": 10,
        "conditions": {"corr_value": {"min": 0.85}},
        "action": "drop_high_correlation",
        "explanation": (
            "คอลัมน์คู่นี้มี Correlation ≥ 0.85 (Multicollinearity)  "
            "ทั้งสองคอลัมน์มีข้อมูลซ้ำซ้อนกันมาก การเก็บทั้งคู่ไว้จะทำให้โมเดลให้น้ำหนัก "
            "เกินความเป็นจริงและแปลผลได้ยาก แนะนำตัดคอลัมน์ที่สองออก"
        ),
    },
    {
        "id": "FSL_002",
        "domain": "feature_selection",
        "priority": 20,
        "conditions": {"cv_abs": {"max": 0.01}},
        "action": "drop_low_variance",
        "explanation": (
            "คอลัมน์นี้มี Coefficient of Variation < 1%  "
            "ค่าเกือบทั้งหมดเหมือนกัน โมเดลไม่สามารถเรียนรู้ Pattern ได้ "
            "จากข้อมูลที่แทบไม่มีความแตกต่างกัน"
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: encoding
    # ══════════════════════════════════════════════════════════════

    {
        "id": "ENC_001",
        "domain": "encoding",
        "priority": 5,
        "conditions": {"cardinality_ratio": {"min": 0.5}},
        "action": "drop_column",
        "explanation": (
            "คอลัมน์นี้มี unique value มากกว่า 50% ของจำนวนแถว  "
            "แทบทุก row มีค่าต่างกัน (เช่น ID, ชื่อ, URL) "
            "model ไม่สามารถเรียนรู้ pattern ได้ การ One-hot จะสร้างคอลัมน์มหาศาล "
            "ที่ไม่มีประโยชน์"
        ),
    },
    {
        "id": "ENC_002",
        "domain": "encoding",
        "priority": 10,
        "conditions": {"cardinality": {"max": 2}},
        "action": "label_encoding",
        "explanation": (
            "คอลัมน์นี้มีเพียง 2 categories (binary)  "
            "Label Encoding (0/1) เพียงพอและไม่สิ้นเปลืองคอลัมน์ "
            "One-hot สำหรับ binary จะสร้างคอลัมน์ซ้ำซ้อน (dummy variable trap)"
        ),
    },
    {
        "id": "ENC_003",
        "domain": "encoding",
        "priority": 15,
        "conditions": {"looks_ordinal": True, "cardinality": {"min": 3, "max": 10}},
        "action": "ordinal_encoding",
        "explanation": (
            "ค่าของคอลัมน์นี้ตรงกับ pattern ที่มีลำดับที่ทราบได้ เช่น Low/Medium/High, "
            "Small/Medium/Large หรือ rating ที่ชัดเจน  "
            "Ordinal Encoding รักษา ordinal relationship ไว้ให้ model ใช้งานได้ถูกต้อง "
            "กรุณายืนยันว่า order ที่ระบบเรียงให้ตรงกับความเป็นจริง"
        ),
    },
    {
        "id": "ENC_004",
        "domain": "encoding",
        "priority": 16,
        "conditions": {"looks_ordinal": True, "cardinality": {"min": 11, "max": 20}},
        "action": "ordinal_encoding",
        "explanation": (
            "ค่าของคอลัมน์นี้ดูมีลำดับชัดเจน และมี 11–20 categories  "
            "Ordinal Encoding เหมาะกว่า One-hot เพราะไม่ทำให้ dimensionality สูงเกินไป "
            "และยังรักษา ordinal relationship ไว้ได้  "
            "กรุณายืนยัน order ที่ถูกต้องก่อนใช้งาน"
        ),
    },
    {
        "id": "ENC_005",
        "domain": "encoding",
        "priority": 20,
        "conditions": {"cardinality": {"min": 3, "max": 10}},
        "action": "one_hot_encoding",
        "explanation": (
            "คอลัมน์นี้มี 3–10 categories (low cardinality)  "
            "One-hot Encoding เหมาะที่สุดเพราะไม่สร้าง ordinal relationship ที่ไม่มีอยู่จริง "
            "เช่น Label Encoding ทำให้ 'Bangkok=1, Chiang Mai=2' ซึ่ง model อาจเข้าใจผิดว่า "
            "Chiang Mai 'มากกว่า' Bangkok"
        ),
    },
    {
        "id": "ENC_006",
        "domain": "encoding",
        "priority": 30,
        "conditions": {"cardinality": {"min": 11, "max": 20}},
        "action": "one_hot_encoding",
        "explanation": (
            "คอลัมน์นี้มี 11–20 categories  One-hot ยังใช้ได้ "
            "แต่จะเพิ่มจำนวนคอลัมน์มากขึ้น หากต้องการลด dimensionality "
            "อาจพิจารณา Label Encoding หรือ Target Encoding แทน"
        ),
    },
    {
        "id": "ENC_007",
        "domain": "encoding",
        "priority": 40,
        "conditions": {"cardinality": {"min": 21}},
        "action": "label_encoding",
        "explanation": (
            "คอลัมน์นี้มี categories มากกว่า 20 (high cardinality)  "
            "One-hot จะสร้างคอลัมน์จำนวนมาก ทำให้เกิด Curse of Dimensionality "
            "Label Encoding ลด dimensionality แต่สร้าง ordinal relationship โดยปริยาย "
            "ควรพิจารณาว่า model ที่ใช้ sensitive ต่อ ordinal หรือไม่"
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # Domain: scaling
    # ══════════════════════════════════════════════════════════════

    {
        "id": "SCL_001",
        "domain": "scaling",
        "priority": 5,
        "conditions": {"no_numeric": True},
        "action": "no_scaling",
        "explanation": "ไม่มี numeric feature  ไม่จำเป็นต้องทำ scaling",
    },
    {
        "id": "SCL_002",
        "domain": "scaling",
        "priority": 10,
        "conditions": {"has_outliers": True},
        "action": "robust_scaler",
        "explanation": (
            "พบ outlier ใน dataset  Robust Scaler ใช้ Median และ IQR แทน Mean/Std "
            "ทำให้ค่า extreme ไม่ดึง scale ให้เบี้ยว "
            "Standard Scaler จะถูก outlier กดดันทำให้ค่าปกติส่วนใหญ่ถูกบีบให้อยู่ในช่วงแคบ"
        ),
    },
    {
        "id": "SCL_003",
        "domain": "scaling",
        "priority": 20,
        "conditions": {"has_heavy_skew": True, "has_outliers": False},
        "action": "log_transform",
        "explanation": (
            "ข้อมูลมี skewness รุนแรง (|skew| > 2) และไม่มี outlier  "
            "Log Transform (log1p) ลด skewness ก่อน แล้วตาม Standard Scaler "
            "เหมาะกับข้อมูล long-tail เช่น รายได้ ราคา จำนวน transaction"
        ),
    },
    {
        "id": "SCL_004",
        "domain": "scaling",
        "priority": 30,
        "conditions": {"is_skewed": True, "has_outliers": False, "has_heavy_skew": False},
        "action": "minmax_scaler",
        "explanation": (
            "ข้อมูลมี skewness ปานกลาง (1 < |skew| ≤ 2)  "
            "MinMax Scaler แปลงค่าให้อยู่ในช่วง [0, 1] "
            "เหมาะกับข้อมูลที่ไม่กระจายแบบ Normal เพราะ Standard Scaler "
            "สมมติ Normal distribution"
        ),
    },
    {
        "id": "SCL_005",
        "domain": "scaling",
        "priority": 40,
        "conditions": {"has_outliers": False, "is_skewed": False},
        "action": "standard_scaler",
        "explanation": (
            "ข้อมูลกระจายใกล้ Normal และไม่มี outlier  "
            "Standard Scaler (Z-score normalization) เหมาะที่สุด "
            "แปลงให้ mean=0, std=1 รักษา relative distance ระหว่าง data points"
        ),
    },

]
