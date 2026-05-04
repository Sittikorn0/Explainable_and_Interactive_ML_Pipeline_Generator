# Explainable & Interactive ML Pipeline Generator

ระบบ AutoML Web Application สำหรับการศึกษา ที่ไม่ได้แค่สร้าง Model ให้อัตโนมัติ แต่อธิบายด้วยว่า "ทำไม" ระบบถึงตัดสินใจแบบนั้นในแต่ละขั้นตอน

Senior Project — Data Science 1312414

---

## สารบัญ

- [วิธีติดตั้ง](#วิธีติดตั้ง)
- [ไฟล์ที่รองรับ](#ไฟล์ที่รองรับ)
- [ขั้นตอนการทำงานของ Pipeline](#ขั้นตอนการทำงานของ-pipeline)
  - [1. Upload Dataset](#1-upload-dataset)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
  - [4. Data Transformation](#4-data-transformation)
  - [5. ML Process &amp; Leaderboard](#5-ml-process--leaderboard)
  - [6. Explainable &amp; Insights](#6-explainable--insights)
- [โมเดลที่รองรับ](#โมเดลที่รองรับ)
- [โครงสร้างโปรเจค](#โครงสร้างโปรเจค)
- [เทคโนโลยีที่ใช้](#เทคโนโลยีที่ใช้)

---

## วิธีติดตั้ง

ต้องการ Python 3.13+ และ [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone https://github.com/Sittikorn0/Senior_Project_Explainable_Pipeline.git
cd Senior_Project_Explainable_Pipeline
uv sync
uv run streamlit run app.py
```

แอปจะเปิดที่ `http://localhost:8501`

หรือใช้ pip:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ไฟล์ที่รองรับ

| รูปแบบ | นามสกุล      | หมายเหตุ                                 |
| ------------ | ------------------- | ------------------------------------------------ |
| CSV          | `.csv`            | รองรับ UTF-8, UTF-8 BOM, TIS-620           |
| Excel        | `.xlsx`, `.xls` | อ่านผ่าน openpyxl/xlrd                   |
| JSON         | `.json`           | รองรับ records, columns, index orientation |

ขนาดสูงสุด 200 MB

---

## ขั้นตอนการทำงานของ Pipeline

### 1. Upload Dataset

ผู้ใช้อัปโหลดไฟล์ ระบบทำ 3 อย่างหลัก:

**ตรวจจับประเภทข้อมูล** — ระบบดู dtype ของแต่ละคอลัมน์แล้วจำแนกเป็น int, float, string, bool, datetime โดยมี logic พิเศษ เช่น คอลัมน์ที่เป็น object แต่ค่าทั้งหมดแปลงเป็นตัวเลขได้ก็จะถูกแปลงเป็น numeric dtype อัตโนมัติ

**แนะนำ Target Column** — ระบบให้คะแนนทุกคอลัมน์ตามเกณฑ์ต่อไปนี้:

| เกณฑ์                          | คะแนน | เหตุผล                                                |
| ----------------------------------- | ---------- | ----------------------------------------------------------- |
| คอลัมน์สุดท้าย        | +1.0       | convention ทั่วไปของ ML dataset                    |
| มีแค่ 2 ค่า (binary)        | +3.0       | ลักษณะเฉพาะของ binary classification target   |
| Unique ≤ 5% ของจำนวนแถว | +2.0       | cardinality ต่ำ เหมาะเป็น classification target |
| Unique > 90% ของจำนวนแถว | −3.0      | น่าจะเป็น ID ไม่ใช่ target                   |
| Missing > 10%                       | −1.5      | target ที่ดีมักมีข้อมูลครบ               |
| เป็น datetime                   | −5.0      | ไม่เหมาะเป็น target                             |

คอลัมน์ที่ได้คะแนนสูงสุดจะถูกแนะนำ ผู้ใช้เปลี่ยนได้

**ระบุประเภทงาน** — ถ้า target เป็นข้อความ → classification เสมอ ถ้าเป็นตัวเลขจะดู unique values: ≤15 → classification, >100 → regression, 16–100 → ดูอัตราส่วน unique/rows ถ้า ≥5% ถือว่า continuous (regression) ถ้า <5% ถือว่า discrete class (classification)

### 2. Data Cleaning

ระบบสแกนปัญหาคุณภาพข้อมูลแล้วให้ผู้ใช้จัดการทีละรายการ:

**Missing Values** — แนะนำวิธีเติมตาม data type:

- ตัวเลขที่ skewness > 1 → ใช้ Median เพราะ Mean จะถูกดึงไปตามค่าสุดโต่ง
- ตัวเลขที่ skewness ≤ 1 → ใช้ Mean เพราะข้อมูลกระจายค่อนข้างสมมาตร
- ข้อความ → ใช้ Mode (ค่าที่พบบ่อยสุด) เพราะ Mean/Median ใช้กับข้อความไม่ได้

**Outlier Detection** — ใช้ IQR method: คำนวณ Q1, Q3 แล้วหาค่า IQR = Q3−Q1 ค่าที่ต่ำกว่า Q1−1.5×IQR หรือสูงกว่า Q3+1.5×IQR ถือว่าเป็น outlier ขอบเขตนี้จะถูก lock ไว้ตั้งแต่ครั้งแรกที่คำนวณ เพื่อให้การ clean ซ้ำ (เช่น clip แล้ว clip อีก) ไม่ขยับขอบเขตไปเรื่อยๆ ผู้ใช้เลือกได้ระหว่าง Clip (ตัดค่าให้อยู่ในขอบเขต) กับ Drop (ลบแถว)

**Duplicates** — ตรวจแถวที่ค่าทุกคอลัมน์เหมือนกัน ลบออกได้

**Drop Columns** — ลบคอลัมน์ที่ไม่ต้องการ เช่น ID หรือคอลัมน์ที่มีค่าเดียว

การเปลี่ยนแปลงทั้งหมดสะสมใน `working_df` แยกจาก `main_df` ต้นฉบับ ผู้ใช้ดู preview ก่อนกด Confirm & Save ถ้า Reset จะกลับไปเป็น `main_df` เดิม

### 3. Exploratory Data Analysis

หน้า EDA ใช้ข้อมูลหลัง clean (ถ้า confirm แล้ว) มี 3 แท็บ:

**Profile** — ตารางสรุปทุกคอลัมน์: data type, ML category (Numeric/Categorical/Datetime/Target), missing count, outlier count, unique count ใช้ outlier bounds ชุดเดียวกับหน้า Cleaning เพื่อให้ตัวเลขตรงกัน

**Distributions** — เลือกคอลัมน์แล้วดูกราฟ:

- Numeric → Histogram + Box plot พร้อมคำนวณ skewness ถ้า |skew| ≥ 1 จะแนะนำ transform (Log, Box-Cox, Yeo-Johnson) ถ้าค่าต่ำสุด ≤ 0 จะแนะนำเฉพาะ Yeo-Johnson เพราะ Log รับค่า ≤ 0 ไม่ได้
- Categorical → Bar chart พร้อมตรวจ class imbalance (ถ้าเป็น target และ max/min > 3 เท่า จะแจ้งเตือน)
- Datetime → Line chart แบ่ง granularity ได้ (Year/Month/Day)

**Relationships** — วิเคราะห์ความสัมพันธ์:

- Feature vs Target: เลือก feature แล้วแสดงกราฟตามประเภทคู่ — Numeric vs Numeric ใช้ Scatter + OLS trendline, Numeric vs Categorical ใช้ Box plot, Categorical vs Categorical ใช้ Grouped bar
- Feature-Target Correlation: ถ้า target เป็นตัวเลข แสดง Pearson r เป็น bar chart ถ้าเป็นข้อความ แสดง mean ของ numeric features แยกตาม class
- Correlation Heatmap: ตรวจคู่ที่ |r| > 0.8 แล้วแจ้งเตือน multicollinearity

### 4. Data Transformation

เตรียมข้อมูลให้ model ใช้ได้ มี 3 ส่วน:

**Encoding** — แปลงข้อความเป็นตัวเลข:

- Unique ≤ 15 ค่า → One-Hot (สร้างคอลัมน์ใหม่ต่อค่า ใช้ `drop_first=True` เพื่อลด dimension) เพราะ One-Hot ไม่สร้างลำดับปลอมให้ข้อมูลที่ไม่มีลำดับ
- Unique > 15 ค่า → Label Encoding (แปลงเป็น 0, 1, 2, ...) เพราะ One-Hot จะสร้างคอลัมน์เยอะเกินไป

Encoding ทำบน train set ก่อนแล้ว transform test set ด้วย schema เดียวกัน ถ้า test set มีค่าที่ train ไม่เคยเห็น จะ fallback เป็น mode ของ train (ไม่ใช่ 0 เพราะ 0 คือ category จริงตัวแรก)

**Scaling** — ปรับขนาดตัวเลข:

- Standard Scaler → ปรับให้ mean=0, std=1 เหมาะกับข้อมูลที่กระจายใกล้ normal
- MinMax Scaler → ย่อเข้า [0, 1] เหมาะเมื่อไม่มี outlier รุนแรง
- Robust Scaler → ใช้ median/IQR แทน mean/std ทนต่อ outlier ได้ดีกว่า
- Log Transform → ลดความเบ้ เหมาะกับข้อมูล right-skewed
- No Scaling → สำหรับ tree-based model (Random Forest, XGBoost ฯลฯ) ที่ตัดสินใจจากการเปรียบเทียบค่า ไม่สนใจ scale

**Drop Features** — ตัดคอลัมน์ที่ไม่เป็นประโยชน์ เช่น ID ที่ unique ทุกแถว

### 5. ML Process & Leaderboard

ฝึกโมเดลทั้งหมดที่เหมาะกับประเภทงาน แล้วจัดอันดับ:

**Data Split** — แบ่ง 80% train / 20% test ถ้าข้อมูลเกิน 50,000 แถวจะ random sample ลงก่อน สำหรับ slow models (kNN, SVM) จะ sample เหลือ 5,000 แถวเพื่อไม่ให้ใช้เวลานานเกินไป

**Cross-Validation** — ใช้ 5-Fold CV บน train set: แบ่งข้อมูลเป็น 5 ส่วน วนสลับฝึก 4 ส่วนทดสอบ 1 ส่วน แล้วเอาคะแนนมาเฉลี่ย ได้ทั้ง mean score และ std (บอกความเสถียร)

**Hyperparameter Tuning** — ใช้ `RandomizedSearchCV` กับ search space ที่กำหนดไว้ล่วงหน้าต่อโมเดล ระบบเลือก parameter set ที่ให้ CV score สูงสุด

**Class Imbalance** — สำหรับ classification ระบบจัดการอัตโนมัติ: model ที่รองรับ `class_weight="balanced"` (เช่น Random Forest, Logistic Regression, SVM, LightGBM) จะเปิดไว้เลย model ที่ไม่รองรับ (เช่น HistGradientBoosting) จะใช้ `compute_sample_weight("balanced")` แทน

**Leaderboard** — จัดอันดับโมเดลตาม CV score เฉลี่ย โมเดลที่ดีที่สุดจะถูก evaluate บน test set แยกต่างหาก:

- Classification: Accuracy, Precision, Recall, F1 Macro, F1 Weighted
- Regression: MAE, RMSE, R²

**Data Leakage Detection** — ก่อนฝึก ระบบตรวจทุกคอลัมน์ด้วย 4 วิธี:

1. Pearson correlation กับ target ≥ 0.85 → แจ้งเตือน
2. Mutual Information ≥ 0.85 → จับ non-linear relationship ที่ correlation จับไม่ได้
3. Bijective mapping → ตรวจว่า feature กับ target มี one-to-one mapping (อาจเป็น encoded version)
4. ชื่อคล้าย target → เช่น `target_encoded` กับ `target`

**Feature Importance** — ใช้ Permutation Importance: สลับค่าใน feature ทีละตัวแล้วดูว่า score ตกเท่าไหร่ ถ้าตกมาก = feature นั้นสำคัญ ถ้ามี `feature_importances_` (tree-based) หรือ `coef_` (linear) ก็ใช้ได้เลย

### 6. Explainable & Insights

หน้าสุดท้ายรวมคำอธิบายทั้งหมดไว้ 3 แท็บ:

**Feature Importance** — กราฟแท่งแนวนอน แสดงว่า feature ไหนสำคัญแค่ไหน พร้อมตาราง

**Model Guide** — อธิบายโมเดลที่ชนะ:

- วิธีทำงาน (เช่น Random Forest = สร้าง Decision Tree หลายต้น แต่ละต้นฝึกบน subset ของข้อมูล แล้วโหวตผล)
- ข้อดี ข้อเสีย
- ตาราง metric เปรียบเทียบกับโมเดลอื่น

**Pipeline Trace** — log ตามลำดับเวลาของทุกการตัดสินใจตลอด pipeline แต่ละ entry มี:

- **What**: สรุปข้อเท็จจริง เช่น "3,000 แถว × 14 คอลัมน์, ไม่มี missing"
- **Why**: เหตุผลเป็นภาษาไทย เช่น "ทำไมเป็น Classification? เพราะ target เป็นข้อความ มี 5 กลุ่ม จึงต้องแยกกลุ่ม ไม่ใช่ทำนายตัวเลข"

ครอบคลุม: เหตุผลเลือก target, ประเภทงาน, วิธี clean, วิธี scale/encode, เหตุผลเลือกโมเดล, ทำไมใช้ CV, ทำไมแบ่ง 80/20, และการตีความคะแนน

**Export** — ดาวน์โหลด HTML Report หรือ ZIP (predictions.csv, metrics.csv, feature_importance.csv)

---

## โมเดลที่รองรับ

### Classification

| โมเดล          | วิธีทำงาน                                                                                             |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| Random Forest       | สร้าง Decision Tree หลายต้นบน random subset แล้วโหวตผล                                 |
| Gradient Boosting   | สร้าง tree ทีละต้น แต่ละต้นแก้ error ของต้นก่อนหน้า (ใช้ HistGradient) |
| XGBoost             | Gradient Boosting + regularization (L1/L2) ป้องกัน overfitting                                          |
| LightGBM            | Gradient Boosting แบบ histogram-based ฝึกเร็วกว่า XGBoost                                        |
| CatBoost            | Gradient Boosting ที่ encode categorical ภายในตัว ลด leakage จากการ encode                  |
| Logistic Regression | หา decision boundary เป็นเส้นตรง ตีความ coefficient ได้                                  |
| Decision Tree       | แบ่งข้อมูลตาม condition ทีละชั้น เห็นเหตุผลชัดเจน                         |
| SVM (SGD)           | หา hyperplane ที่แบ่ง class ฝึกด้วย SGD ให้เร็วกับข้อมูลใหญ่               |
| kNN                 | ทำนายจาก k ตัวอย่างที่ใกล้ที่สุด ไม่มีการฝึกล่วงหน้า           |
| Naive Bayes         | คำนวณ P(class)                                                                                            |

### Regression

| โมเดล        | วิธีทำงาน                                                                 |
| ----------------- | ---------------------------------------------------------------------------------- |
| Random Forest     | เฉลี่ยผลจากหลาย Decision Tree                                       |
| Gradient Boosting | แก้ error ทีละขั้น (HistGradient)                                       |
| XGBoost           | Gradient Boosting + regularization                                                 |
| LightGBM          | Histogram-based boosting                                                           |
| CatBoost          | Boosting + native categorical support                                              |
| Linear Regression | หาสมการ y = w₁x₁ + w₂x₂ + ... + b ด้วย least squares                |
| Decision Tree     | แบ่งข้อมูลตาม condition แล้วเฉลี่ยค่าในแต่ละ leaf |
| kNN               | เฉลี่ยค่าจาก k ตัวอย่างที่ใกล้ที่สุด              |

---

## โครงสร้างโปรเจค

```
Senior_Project_Explainable_Pipeline/
├── app.py                          # Entry point, routing, session recovery
│
├── data_prepare/features/
│   ├── loading_data.py             # อ่านไฟล์ CSV/Excel/JSON, แปลง encoding, cache ลง disk
│   ├── data_type_detection.py      # ตรวจจับ dtype → actual type (int/float/string/bool/datetime)
│   ├── data_distribute.py          # นับ outlier ด้วย IQR (รองรับ fixed bounds)
│   ├── statistics.py               # คำนวณ IQR bounds, skewness
│   ├── cleaning_logic.py           # Fill missing + clip/drop outlier
│   └── target_col.py              # Scoring heuristic สำหรับแนะนำ target
│
├── ml_process/features/
│   ├── config.py                   # Model registry, hyperparameter search spaces
│   ├── preprocessing.py            # Encode → Scale → Train/Test split (leak-safe)
│   ├── runner.py                   # ฝึกโมเดล, CV scoring, class weight
│   ├── evaluation.py               # คำนวณ Accuracy/F1/R²/MAE/RMSE
│   ├── scaling.py                  # Standard/MinMax/Robust/Log scaler
│   ├── encoding.py                 # One-Hot/Label/Ordinal encoder
│   ├── logic.py                    # Data leakage detection, Feature Importance
│   ├── export.py                   # สร้าง HTML report + CSV/ZIP export
│   └── views.py                    # UI components สำหรับหน้า ML
│
├── explainable/
│   ├── features/
│   │   ├── trace_log.py            # บันทึกทุกการตัดสินใจ + สร้างคำอธิบาย "ทำไม"
│   │   └── explainer.py            # Permutation Importance
│   ├── domains/                    # คำแนะนำเฉพาะด้าน (cleaning/transformation/modeling)
│   └── knowledge_base/             # Rule-based knowledge จาก Topic 1-12
│
├── interface/
│   ├── upload.py                   # หน้า Upload
│   ├── cleaning.py                 # หน้า Cleaning
│   ├── eda.py                      # หน้า EDA
│   ├── data_transformation.py      # หน้า Transformation
│   ├── model_process.py            # หน้า ML Process
│   ├── explainable.py              # หน้า Explainable
│   ├── cleaning_components/        # UI components สำหรับ missing/outlier
│   └── styles/app.css              # Dark theme + responsive CSS
│
├── classroom_knowledge/            # PDF เนื้อหาวิชา Topic 1-12
├── .streamlit/config.toml          # Theme + server config
├── pyproject.toml                  # Dependencies (uv)
└── requirements.txt                # Dependencies (pip)
```

---

## เทคโนโลยีที่ใช้

| หมวด        | เทคโนโลยี                                        |
| --------------- | --------------------------------------------------------- |
| Web Framework   | Streamlit 1.55                                            |
| ML              | scikit-learn 1.8, XGBoost 3.2, LightGBM 4.6, CatBoost 1.2 |
| Data            | pandas 2.3, NumPy 2.4, SciPy 1.17                         |
| Visualization   | Plotly 6.6, Matplotlib 3.10                               |
| Language        | Python 3.13                                               |
| Package Manager | uv                                                        |

---

โปรเจคนี้จัดทำเพื่อการศึกษาเท่านั้น (Education Only)
