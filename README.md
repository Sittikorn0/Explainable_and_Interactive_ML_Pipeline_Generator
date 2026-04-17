# Explainable & Interactive ML Pipeline Generator

ระบบ AutoML แบบ Interactive ที่อธิบายทุกขั้นตอนได้ สร้างด้วย Streamlit รองรับ workflow ตั้งแต่ Upload ข้อมูลไปจนถึง Train โมเดล พร้อมคำอธิบายแต่ละกระบวนการ

---

## หน้า Upload Dataset

### ทำอะไรได้บ้าง

**1. รองรับไฟล์หลายรูปแบบ**
- รับไฟล์ CSV, Excel (.xlsx / .xls), และ JSON ขนาดสูงสุด 200 MB
- CSV: ลอง encoding อัตโนมัติ 5 แบบตามลำดับ (`utf-8` → `utf-8-sig` → `cp874` → `cp1252` → `latin1`) เพื่อรองรับภาษาไทยและไฟล์จากหลาย locale โดยที่ผู้ใช้ไม่ต้องรู้ว่าไฟล์ใช้ encoding อะไร
- Excel: อ่านด้วย `pandas.read_excel()` โดยตรง
- JSON: รองรับ 3 รูปแบบโครงสร้าง:
  - **Array of objects**: `[{...}, {...}]`
  - **Single object**: `{...}` (wrap เป็น list อัตโนมัติ)
  - **JSONL**: 1 object ต่อบรรทัด `{...}\n{...}`
  - **Nested objects**: flatten ด้วย `json_normalize` สูงสุด 5 ระดับ (เช่น `address.city.zipcode`)
  - **Array columns**: join เป็น string ด้วย `, ` อัตโนมัติ
  - แจ้งเตือนรายชื่อ columns ที่ถูกแปลงโดยแยกประเภท Array/Object

**2. แสดง Data Preview**
- แสดงจำนวน Rows / Columns
- แสดงข้อมูล 10 แถวแรก

**3. แนะนำ Target Column อัตโนมัติ**
- คำนวณ score ให้กับทุก column โดยใช้ heuristic ดังนี้:

| เงื่อนไข | คะแนน |
|----------|-------|
| เป็น column สุดท้าย (ML convention) | +1.0 |
| Binary (unique = 2) | +3.0 |
| Low cardinality categorical (unique ≤ 5% ของ rows และ ≤ 20) | +2.0 |
| ไม่มี missing values | +0.5 |
| Unique สูงมาก (>90% ของ rows — น่าจะเป็น ID) | −3.0 |
| Missing >10% | −1.5 |
| เป็น Datetime | −5.0 |

- แสดงเหตุผลที่ระบบแนะนำ column นั้น
- ผู้ใช้เปลี่ยน target ได้เอง พร้อม analysis ของ column ที่เลือกและปุ่มกลับไปใช้ที่ระบบแนะนำ

**4. อนุมาน ML Task อัตโนมัติ**
- **Binary Classification**: target มี unique = 2
- **Classification**: target เป็น string หรือ int ที่มี unique ≤ 20
- **Regression**: target เป็น int หรือ float ที่มี unique > 20

**5. Data Persistence**
- บันทึก DataFrame เป็น Parquet ลง `temp_cache/` เพื่อไม่ให้ข้อมูลหายเมื่อ refresh
- บันทึก target column และชื่อไฟล์แยกกัน
- รองรับหลาย session พร้อมกัน (สูงสุด 5 sessions), ลบไฟล์เก่าเกิน 3 ชั่วโมงอัตโนมัติ

### มีประโยชน์ต่องานอย่างไร
ผู้ใช้ไม่ต้องแปลงหรือ preprocess ไฟล์ก่อน upload เอง ระบบจัดการ encoding, nested structure, และ format ต่างๆ ให้อัตโนมัติ การแนะนำ target column ช่วยนักศึกษาที่ยังไม่คุ้นเคยกับ convention ของ ML datasets ให้เริ่มต้นได้อย่างถูกต้อง

### ทำไมต้องทำขั้นตอนนี้
ถ้าไม่โหลดข้อมูลให้ถูกต้องตั้งแต่ต้น ขั้นตอน Cleaning และ EDA จะทำงานกับข้อมูลที่ผิดรูปแบบ target ที่เลือกไม่เหมาะสมทำให้โมเดลที่ train มาไม่มีความหมาย

---

## หน้า Data Cleaning

### ทำอะไรได้บ้าง

**1. Dataset Overview**
แสดง 5 metrics สำหรับ working dataset ปัจจุบัน:
- จำนวน Rows / Columns
- Missing Values (จำนวน + %)
- Duplicate Rows (จำนวน + %)
- Outliers (จำนวน + %) โดยใช้ Z-Score หรือ IQR ตามความเบ้ของข้อมูล

**2. Profile View (Tab 1)**
ตารางต่อ column แสดง Missing, Outlier count, Unique count พร้อม progress bar

**3. Cleaning Operations (Tab 2)**

**Drop Columns**
- เลือก drop columns ด้วยตนเองจาก multiselect (ไม่รวม target column)
- ปุ่ม "Drop แนะนำ" — suggest columns ที่ควรลบอัตโนมัติ:
  - Missing > 80% ของข้อมูล
  - มีค่า unique เพียง 1 ค่า (ไม่มี variance ใดๆ)
- Safety check: ห้ามเหลือน้อยกว่า 2 columns

**Remove Duplicates**
- ตรวจ complete row duplicates (ทุก column เหมือนกัน)
- ลบทั้งหมดด้วย click เดียว

**Handle Missing Values**

| Strategy | เหมาะกับ |
|----------|---------|
| `mean` | Float column ที่กระจายแบบ Normal |
| `median` | Float column ที่มี Skew หรือ Outlier |
| `median (rounded)` | Int column แบบ Discrete เช่น อายุ |
| `most frequent` | String/Categorical column |
| `drop rows` | ทุกประเภท (ลบแถวที่มี missing) |

- Dropdown ต่อ column แสดงเฉพาะ strategy ที่เหมาะกับ data type นั้นๆ
- Global action bar: Select All / Deselect All / Apply Selected / Apply All
- Apply All fallback อัตโนมัติไปยัง strategy ที่ compatible ถ้า global strategy ไม่เหมาะกับ column

**Handle Outliers**
- ตรวจจับอัตโนมัติโดยเลือก method ตาม skewness ของแต่ละ column:
  - `|skew| < 0.5` → **Z-Score** (ค่าที่ห่างจาก mean เกิน 3σ)
  - `|skew| ≥ 0.5` → **IQR** (ค่านอกช่วง Q1−1.5×IQR ถึง Q3+1.5×IQR)
- แสดง bounds [lower, upper] และเหตุผลที่เลือก method นั้นต่อ column
- 2 strategies:
  - `clip` — วนซ้ำสูงสุด 5 รอบด้วย bounds ที่ recalculate ใหม่ทุกรอบ จนกว่าจะไม่มี outlier เหลือ
  - `drop rows` — ลบแถวที่มีค่าเกิน bounds

**4. Before/After Comparison Table**

| Metric | สีที่แสดงเมื่อเปลี่ยนแปลง |
|--------|--------------------------|
| Rows | แดง = ลดลง (เสียข้อมูล), เทา = ไม่เปลี่ยน |
| Columns | เขียว = ลดลง (ดี), แดง = เพิ่มขึ้น |
| Missing Values | เขียว = ลดลง, แดง = เพิ่มขึ้น |
| Duplicates | เขียว = ลดลง, แดง = เพิ่มขึ้น |
| Outliers | เขียว = ลดลง, แดง = เพิ่มขึ้น |

**5. Confirm & Reset & Download**
- **Confirm & Save**: บันทึก cleaned data เป็น Parquet (ส่งต่อไป EDA) และ CSV (สำหรับ download)
- **Reset**: กลับไปยังข้อมูลดิบก่อนการ cleaning ทุกขั้นตอน (ใช้ `original_df` ไม่ใช่ `main_df` ที่อาจ overwrite แล้ว)
- **Download**: ปุ่ม download CSV ปรากฏหลัง Confirm เท่านั้น

### มีประโยชน์ต่องานอย่างไร
ข้อมูลที่ dirty ทำให้ algorithm ML เรียนรู้จาก noise แทนที่จะเรียนจาก pattern จริงๆ หน้านี้ช่วยให้นักศึกษาเข้าใจว่าข้อมูลมีปัญหาอะไร และตัดสินใจวิธีจัดการได้อย่าง informed แทนที่จะ apply เดาๆ เช่น การเลือก median สำหรับ skewed data แทน mean ทำให้ไม่ถูกดึงด้วย outlier

### ทำไมต้องทำขั้นตอนนี้
**GIGO (Garbage In, Garbage Out)** — โมเดล ML ที่ดีที่สุดไม่สามารถชดเชยข้อมูลที่สกปรกได้ missing values ทำให้ algorithm บางตัว error, outlier ทำให้ linear models เบ้ไปมาก, duplicates ทำให้ model overfit กับข้อมูลซ้ำ

---

## หน้า Exploratory Data Analysis (EDA)

### ทำอะไรได้บ้าง

**1. Dataset Overview**
- แสดง 5 metrics เดียวกับ Cleaning (Rows, Columns, Missing, Duplicates, Outliers) บน cleaned data
- แสดง Target column info: data type, unique count, ML task ที่อนุมาน
- **Proactive Class Imbalance Detection**: ถ้า target เป็น categorical และ max class > 3× min class → แสดง warning ทันทีในหน้า overview โดยไม่ต้องรอให้ user เลือก column เอง

**2. Profile View (Tab 1)**
ตารางต่อ column แสดง:

| Column | Data Types | ML Category | Missing | Outliers | Unique |
|--------|-----------|-------------|---------|----------|--------|
| ... | int/float/string/datetime | Numeric/Discrete, Numeric/Continuous, Categorical/Nominal, Datetime, (Target) | ... | ... | progress bar |

ML Category อ้างอิง Topic 2 — Attribute Types และมีคำอธิบายแบบขยายในหน้า

**3. Data Distributions (Tab 2)**

**Numeric columns**
- Histogram + Box plot marginal
- วิเคราะห์ Skewness อัตโนมัติ:
  - `|skew| < 0.5` → Normal (สมมาตร)
  - `|skew| < 1` → Moderately Skewed
  - `|skew| ≥ 1` → Highly Skewed พร้อมแนะนำ Transformation:
    - ถ้า min > 0: แนะนำ Log, Box-Cox, หรือ Yeo-Johnson
    - ถ้า min ≤ 0: แนะนำเฉพาะ **Yeo-Johnson** เท่านั้น (Log/Box-Cox ใช้กับค่า 0 หรือลบไม่ได้)

**Datetime columns**
- ตรวจจับทั้ง native datetime64 และ string column ที่มีรูปแบบวันที่ (สอดคล้องกับ Profile tab)
- เลือก Granularity: Year / Month / Day
- Line chart แสดง distribution ตามช่วงเวลา
- แสดง date range (min ถึง max)

**Categorical columns**
- Bar chart Top 20 values
- แสดง unique count, ค่าที่พบมากสุด และ %
- Class Imbalance detection ในกรณีที่เป็น target column (threshold: max > 3× min)

**4. Relationships & Redundancy (Tab 3)**

**Feature vs Target Visualization**
ระบบเลือก chart ที่เหมาะสมตาม combination ของ data type:

| Feature type | Target type | Chart |
|-------------|-------------|-------|
| Datetime | Numeric | Line chart (mean ต่อ period) |
| Datetime | Categorical | Multi-line chart (count ต่อ class ต่อ period) |
| Numeric | Numeric | Scatter + OLS trendline (เส้น regression) |
| Numeric | Categorical | Box plot (เรียงตาม median ของแต่ละ class) |
| Categorical | Numeric | Box plot Top 20 categories |
| Categorical | Categorical | Grouped bar chart Top 15 categories |

**Feature-Target Correlation**
- Numeric target: Pearson correlation bar chart ทุก numeric feature (สีเขียว = บวก, สีแดง = ลบ)
- Categorical target: Mean value ของแต่ละ feature แยกตาม target class เรียงตาม spread (feature ที่แยก class ได้ดีขึ้นก่อน)

**Correlation Heatmap**
- Pearson correlation matrix ของทุก numeric column
- Height dynamic ตามจำนวน columns: `max(400, min(800, 30 × n_columns))` px
- ซ่อนตัวเลขในเซลล์เมื่อมีมากกว่า 15 columns (hover เพื่ออ่านค่า)
- **Multicollinearity Detection**: แจ้งเตือน pairs ที่ `|r| > 0.8` พร้อมแนะนำให้พิจารณาลบออก 1 column ต่อ pair เพราะข้อมูลซ้ำซ้อนกัน

### มีประโยชน์ต่องานอย่างไร
EDA คือขั้นตอนที่ data scientist ใช้สร้าง intuition เกี่ยวกับข้อมูลก่อนตัดสินใจว่าจะใช้ feature ไหน, model ไหน, และต้อง transform อะไรบ้าง การดู distribution บอกว่าควร scale หรือ transform ก่อนไหม, correlation บอกว่า feature ไหน redundant, class imbalance บอกว่าต้องจัดการก่อน train หรือเปล่า ถ้าข้ามขั้นตอนนี้ไปสร้างโมเดลเลย มักได้โมเดลที่ performance แย่และไม่รู้ว่าทำไม

### ทำไมต้องทำขั้นตอนนี้
EDA ช่วยตอบคำถามสำคัญที่ algorithm ไม่สามารถตอบแทนได้เอง เช่น "ควร log transform ไหม?", "feature นี้กับ feature นั้น measure เรื่องเดียวกันไหม?", "ข้อมูลกลุ่มนี้มีน้อยเกินไปจนโมเดลไม่เคยเห็นหรือเปล่า?" การทำ EDA ก่อนทำให้ตัดสินใจได้ดีขึ้นและประหยัดเวลา trial-and-error ในขั้นตอน modeling

---

## หน้า Data Transformation

### ทำอะไรได้บ้าง

ระบบวิเคราะห์ข้อมูลอัตโนมัติแล้วแนะนำ transformation method ที่เหมาะสมพร้อมเหตุผล ผู้ใช้สามารถ override ได้ทุกรายการก่อน apply

**1. Encoding — แปลง Categorical เป็นตัวเลข**

ระบบวิเคราะห์ทุก categorical column แล้วแนะนำ method ตาม cardinality และ ratio:

| เงื่อนไข | Recommended |
|----------|-------------|
| Unique = 2 (binary) | Label Encoding (0/1) |
| Unique ≤ 10 (low cardinality) | One-hot Encoding |
| Unique ≤ 20 | One-hot Encoding (พร้อม warning ว่าจะเพิ่ม columns) |
| Unique > 50% ของ rows (ID/free-text) | Drop Column |
| Unique > 20 (high cardinality) | Label Encoding |

- แสดง sample values, cardinality badge, และ warning สำหรับแต่ละ column
- Radio button เลือกได้ 4 ตัวเลือก: `One-hot`, `Label`, `Ordinal`, `Drop`
- **Ordinal Encoding**: เรียง categories alphabetical แล้ว map เป็น integer (0, 1, 2, ...) พร้อมแสดง order ให้ user ตรวจสอบก่อนใช้
- ตารางเปรียบเทียบ method ทั้งหมดใน expander

**2. Scaling — ปรับ Scale ของ Numeric Features**

ระบบวิเคราะห์ numeric columns ด้วย IQR outlier detection และ skewness แล้วแนะนำ:

| เงื่อนไข | Recommended |
|----------|-------------|
| มี outlier (>5% ของข้อมูล) | Robust Scaler (ใช้ Median/IQR แทน Mean/Std) |
| Skewed รุนแรง (\|skew\| > 2) | Log Transform (log1p) ตามด้วย Standard Scaler |
| Skewed ปานกลาง (\|skew\| > 1) | MinMax Scaler |
| ไม่มี outlier, ไม่ skewed | Standard Scaler (Z-score) |

- แสดงตาราง statistics (min, max, mean, std, skew, outlier%) ของทุก numeric column
- Histogram + box plot ของ column ที่เลือก
- Radio button เลือกได้ 5 ตัวเลือก: `Log Transform`, `Standard`, `MinMax`, `Robust`, `ไม่ scale`
- **หมายเหตุ**: Scaling จะ**ไม่ถูก apply ที่นี่** — บันทึกแค่ method ไว้ให้ ML Process ทำหลัง train/test split เพื่อป้องกัน Data Leakage

**3. Feature Selection — ตัด Feature ที่ไม่จำเป็นออก**

- **High Correlation Pairs** (`|r| ≥ 0.85`): แสดง heatmap และรายการคู่ที่ correlation สูง พร้อม checkbox เลือกตัดทีละ column
- **Low Variance Features** (CV < 0.01): แสดง column ที่ค่าแทบไม่เปลี่ยนแปลง (std/mean < 1%) ซึ่ง model ไม่สามารถเรียนรู้ pattern ได้
- Safety: ตัด target column ออกได้ไม่

**4. Apply Transformation**
- ปุ่ม Apply Transformation: ทำ Feature Selection → Encoding ตามลำดับ
- แสดง summary: Original Columns, Dropped, After Encoding, Scaling method ที่เลือก
- แสดง Transformed Data 5 แถวแรกใน expander
- Validate ว่าต้องเหลือ feature อย่างน้อย 1 column หลัง feature selection (ป้องกัน crash)

### มีประโยชน์ต่องานอย่างไร
ML algorithm ส่วนใหญ่ทำงานกับตัวเลขเท่านั้น ไม่สามารถรับ string โดยตรงได้ การเลือก encoding ที่ผิดสร้าง ordinal relationship ปลอม (เช่น Label Encoding บน "Bangkok"=0, "Chiang Mai"=1 ทำให้ model คิดว่า Chiang Mai "มากกว่า" Bangkok) Feature ที่ redundant เพิ่ม computation โดยไม่เพิ่ม information และอาจทำให้ model overfit

### ทำไมต้องทำขั้นตอนนี้
ข้อมูลดิบที่มี categorical columns และ scale ต่างกันมาก (เช่น อายุ 20-80 กับรายได้ 10,000-100,000) ทำให้ algorithm ที่ใช้ distance หรือ gradient (SVM, KNN, Linear model) ให้ความสำคัญกับ feature ที่มี scale ใหญ่กว่าโดยไม่ตั้งใจ Transformation แก้ปัญหานี้ก่อนที่จะ train

---

## หน้า ML Process (Model Competition)

### ทำอะไรได้บ้าง

**1. Target Column Confirmation**
- แสดง target column ที่เลือกจากขั้นตอน Transformation พร้อม ML task ที่อนุมาน (Classification / Regression) และจำนวน unique values

**2. Data Splitting (80/20)**
- แสดง Total Rows, Train Set (80%), Test Set (20%)
- **Stratified Split**: สำหรับ Classification จะใช้ stratified sampling เพื่อให้สัดส่วน class ในทุก split เหมือนกัน (ถ้าทุก class มีอย่างน้อย 2 ตัวอย่าง)
- **Dataset Sampling**: ถ้า dataset ใหญ่กว่า 5,000 rows จะ sample ลงมาก่อน เพื่อให้ train ได้ในเวลาที่เหมาะสม

**3. Model Competition + Auto Hyperparameter Tuning**

ระบบ train ทุก model พร้อมกันและเปรียบเทียบ Cross-Validation score:

**Classification models:**
| Model | หมายเหตุ |
|-------|---------|
| Random Forest | — |
| Gradient Boosting (HistGBM) | — |
| Logistic Regression | — |
| Decision Tree | — |
| SVM (SGD / hinge loss) | ช้า — sample 2,000 rows |
| kNN | ช้า — sample 2,000 rows |
| Naive Bayes | — |
| XGBoost / LightGBM / CatBoost | ถ้า install แล้ว |

**Regression models:**
| Model | หมายเหตุ |
|-------|---------|
| Random Forest | — |
| Gradient Boosting (HistGBM) | — |
| Linear Regression | — |
| Decision Tree | — |
| kNN | ช้า — sample 2,000 rows |
| XGBoost / LightGBM / CatBoost | ถ้า install แล้ว |

- **Hyperparameter Tuning**: `RandomizedSearchCV` (3-fold CV, 10 combinations) สำหรับ model ที่มี parameter grid
- Progress bar แสดง training progress ทีละ model
- **Data Leakage Detection**: ตรวจหา columns ที่มี correlation > 0.99, bijective mapping, หรือชื่อคล้าย target ก่อนแสดงผล

**4. Leaderboard Tab**
- ตาราง ranking ทุก model ตาม CV score (สูงสุดขึ้นก่อน) พร้อม ±Std
- Model ที่ train ไม่ผ่านแสดง error ใน expander
- **Best Model Card**: อธิบายว่าทำไม model นี้ถึงชนะ — CV score สูงสุด, ห่างจากอันดับ 2 เท่าไร, stability (std เทียบกับ average), และลักษณะของ model นั้น
- แสดง best hyperparameters ที่ tuned ได้

**5. Evaluation Tab**
- ประเมิน best model บน Test set (20%) ที่ไม่เคยใช้ระหว่าง training

**Classification metrics:**
| Metric | ความหมาย |
|--------|---------|
| Accuracy | % ที่ทำนายถูกทั้งหมด |
| Precision (Macro) | ค่าเฉลี่ย Precision ทุก class เท่ากัน |
| Recall (Macro) | ค่าเฉลี่ย Recall ทุก class เท่ากัน |
| F1 (Macro) | ค่าเฉลี่ย F1 ทุก class เท่ากัน |

**Regression metrics:**
| Metric | ความหมาย |
|--------|---------|
| R² Score | อธิบาย variance ได้กี่ % (1.0 = perfect) |
| RMSE | error เฉลี่ยในหน่วยเดียวกับ target |
| MSE | RMSE ยกกำลัง 2 |

- Confusion Matrix (Classification) พร้อมอ่านค่า diagonal (ถูก) vs off-diagonal (ผิด)
- Actual vs Predicted Scatter Plot (Regression) พร้อม perfect line
- **Perfect Score Warning**: แจ้งเตือนถ้า metrics ทั้งหมด = 1.0 ซึ่งน่าสงสัยว่ามี Data Leakage

**6. Feature Importance Tab**
- คำนวณ feature importance สำหรับ best model:
  - Tree-based models: ใช้ `feature_importances_` (Gini impurity reduction)
  - Linear models: ใช้ `|coef_|` (mean absolute coefficient)
- Horizontal bar chart Top 20 features เรียงตาม importance ลดลง
- แสดง Top 3 features พร้อม rank, importance score, และ % ของ total importance
- ถ้า model ไม่รองรับ feature importance แจ้ง user แทนที่จะ crash

**7. Data Visualization Tab**
- Histogram + box plot ของ column ที่เลือก (numeric) หรือ bar chart (categorical)
- Color by categorical column
- Correlation Heatmap ของ numeric columns (ถ้ามี ≥ 2 columns)

### มีประโยชน์ต่องานอย่างไร
แทนที่จะเลือก algorithm เดาๆ หรือลองทีละตัว ระบบ train ทุก model พร้อมกันและเปรียบเทียบอย่างยุติธรรมบน test set ที่ไม่เคยเห็น Feature Importance ช่วยให้เข้าใจว่า model ตัดสินใจจาก feature ไหน ซึ่งเป็นหัวใจของ "Explainable ML" — ผู้ใช้ไม่ใช่แค่รู้ว่า model ได้ accuracy เท่าไร แต่รู้ว่า model ให้เหตุผลอย่างไร

### ทำไมต้องทำขั้นตอนนี้
การเลือก algorithm ที่เหมาะสมกับข้อมูลเป็นสิ่งที่ต้องทดลองจริง ไม่มี algorithm ไหนดีที่สุดทุกกรณี (No Free Lunch Theorem) การประเมินบน test set ที่แยกออกมาก่อน train ทำให้ metric ที่ได้สะท้อนความสามารถจริงในการทำนายข้อมูลใหม่ ไม่ใช่แค่ "จำ" training data

---

## โครงสร้างโปรเจค

```
Senior_Project_Explainable_Pipeline/
├── app.py                          # Entry point, navigation, session state management
├── interface/
│   ├── upload.py                   # หน้า Upload Dataset
│   ├── cleaning.py                 # หน้า Data Cleaning
│   ├── eda.py                      # หน้า EDA
│   ├── data_transformation.py      # หน้า Data Transformation
│   ├── model_process.py            # หน้า ML Process
│   └── ui_helpers.py               # Shared UI components (_badge, _rec_box)
├── data_prepare/
│   └── features/
│       ├── loading_data.py         # File I/O, CSV/Excel/JSON parsing, cache management (Parquet)
│       ├── cleaning_logic.py       # Missing value & outlier treatment logic
│       ├── data_distribute.py      # Outlier detection (Z-Score / IQR)
│       ├── data_type_detection.py  # Column type inference
│       └── target_col.py           # Target column suggestion & description
└── ml_process/
    └── features/
        ├── config.py               # Model registry, hyperparameter grids, limits
        ├── data_analyzer.py        # Encoding/Scaling/Feature Selection analysis
        ├── data_transformer.py     # Apply encoding & feature selection
        ├── preprocessing.py        # Train/test split, encode, scale (leakage-safe)
        ├── runner.py               # Model competition, RandomizedSearchCV
        ├── evaluation.py           # Metrics calculation & visualization
        ├── logic.py                # Data leakage detection, feature importance
        ├── encoding.py             # Encoding UI rendering
        ├── scaling.py              # Scaling UI rendering
        ├── feature_select.py       # Feature selection UI rendering
        └── views.py                # Model result visualization
```

## การรัน

```bash
pip install -r requirements.txt
streamlit run app.py
```
