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
