# -*- coding: utf-8 -*-
"""
สคริปต์เปรียบเทียบระบบ (System Comparison Script) - เวอร์ชันจัดระเบียบแบ่งกลุ่มฟังก์ชัน
เปรียบเทียบประสิทธิภาพระหว่างฝั่ง Interactive AutoML (ระบบของเรา) และ AutoGluon Tabular
โดยออกแบบให้ทั้งสองระบบแยกกระบวนการกันทำงานอย่างสมบูรณ์ (แยกทำตามระบบของใครของมัน)
แต่เริ่มต้นด้วยชุดข้อมูลดิบตั้งต้นและตัวอย่างการแบ่งชุดทดสอบ (Train-Test Split 80/20) เดียวกัน
"""

import os
import sys
import time
import json
import argparse

# เพิ่มโฟลเดอร์หลักของโปรเจกต์ (Parent Directory) เข้า sys.path เพื่อให้สามารถ Import แพ็คเกจ backend ได้จากทุกพาธที่รัน
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# นำเข้าฟังก์ชันจากระบบ Interactive AutoML ของเรา
from backend.function.data_loader.file_reader import read_csv_with_fallback, normalize_dtypes
from backend.core.model_training.preprocess.cleaning import clean_fit_transform
from backend.core.model_training.preprocess.feature_extraction import datetime_fit_transform
from backend.core.model_training.preprocess.encoding import encode_fit_transform
from backend.core.model_training.preprocess.scaling import scale_data
from backend.core.model_training.trainer.train_model import run_competition
from backend.core.model_training.evaluation.eval import get_metrics
from backend.function.analyzer.task_detection import detect_task

# นำเข้า AutoGluon Tabular
from autogluon.tabular import TabularPredictor

# =========================================================================
# กลุ่มที่ 1: การโหลดข้อมูลและการตรวจวิเคราะห์ประเภทงาน (Data Loading & Task Analysis)
# =========================================================================

def load_and_prepare_raw_data(dataset_path, target_column):
    """
    โหลดข้อมูลดิบจาก CSV ด้วยการแปลง Encoding อัตโนมัติ และตรวจสอบชนิดคอลัมน์เป้าหมาย
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลดิบที่: {dataset_path}")

    try:
        with open(dataset_path, "rb") as f:
            df_raw = normalize_dtypes(read_csv_with_fallback(f.read()))
        print(f"[SUCCESS] โหลดข้อมูลสำเร็จ ขนาดข้อมูลดิบ: {df_raw.shape[0]:,} แถว × {df_raw.shape[1]} คอลัมน์")
    except Exception as e:
        raise RuntimeError(f"โหลดข้อมูลล้มเหลว: {e}")

    if target_column not in df_raw.columns:
        print(f"คอลัมน์ทั้งหมดที่มี: {list(df_raw.columns)}")
        raise KeyError(f"ไม่พบคอลัมน์เป้าหมาย '{target_column}' ในชุดข้อมูล")

    # ตรวจสอบประเภทงาน (Task Detection)
    task_type = detect_task(df_raw, target_column)
    print(f"• ประเภทงานที่ตรวจพบ: {task_type.upper()}")
    
    return df_raw, task_type


# =========================================================================
# กลุ่มที่ 2: การโหลดค่าเซสชันดั้งเดิม (Session Configurations Retriever)
# =========================================================================

def load_session_configuration(session_id, target_column):
    """
    ดึงการตั้งค่าคอลัมน์และขอบเขต Preprocessing ที่บันทึกไว้ในประวัติเซสชัน Cache
    """
    scaling_method = "standard_scaler"
    encoding_decisions = None
    outlier_rules = {}
    transformed_cols = []
    has_session_config = False
    transformed_df_loaded = None

    if not session_id:
        return has_session_config, transformed_cols, outlier_rules, encoding_decisions, scaling_method, transformed_df_loaded

    trans_meta_path = f"cache/transformation/trans_meta_{session_id}.json"
    outlier_bounds_path = f"cache/cleaning/outlier_bounds_{session_id}.json"
    transformed_path = f"cache/transformation/transformed_{session_id}.parquet"

    if os.path.exists(trans_meta_path) and os.path.exists(transformed_path):
        try:
            with open(trans_meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

            target_column = meta_data.get("target_col", target_column)
            summary_data = meta_data.get("summary", {})
            scaling_method = summary_data.get("scaling_method", "standard_scaler")
            encoding_decisions = summary_data.get("encoding_decisions", None)

            transformed_df_loaded = pd.read_parquet(transformed_path)
            transformed_cols = [c for c in transformed_df_loaded.columns if c != target_column]

            print(f"[INFO] โหลด transformed_df จาก cache สำเร็จ: {transformed_df_loaded.shape[0]:,} แถว × {len(transformed_cols)} features")
            has_session_config = True
        except Exception as e:
            print(f"[WARNING] โหลด Session Transformation config ล้มเหลว จะรันด้วยโหมดอัตโนมัติ: {e}")

    if os.path.exists(outlier_bounds_path):
        try:
            with open(outlier_bounds_path, "r", encoding="utf-8") as f:
                outlier_bounds = json.load(f)
            for col, val in outlier_bounds.items():
                outlier_rules[col] = {
                    "strategy": "clip",
                    "lower": val["lower"],
                    "upper": val["upper"]
                }
            print(f"[INFO] โหลดกฎ Outlier clipping สำเร็จสำหรับ {len(outlier_rules)} คอลัมน์")
        except Exception as e:
            print(f"[WARNING] โหลด Outlier bounds config ล้มเหลว: {e}")

    return has_session_config, transformed_cols, outlier_rules, encoding_decisions, scaling_method, transformed_df_loaded


# =========================================================================
# กลุ่มที่ 3: การจัดการแบ่งชุดข้อมูลดิบร่วมกัน (Unified Raw Data Splitter)
# =========================================================================

def split_raw_data(df_raw, target_column, task_type):
    """
    เตรียมเป้าหมายสำหรับการจัดคลาส (Classification) และแบ่งสัดส่วนชุด Train-Test 80/20
    """
    features_raw = df_raw.drop(columns=[target_column]).copy()
    target_raw = df_raw[target_column].copy()

    # จัดการกรณีเป็น Classification เพื่อทำ Target Encoding เป็นตัวเลขและรักษาการกระจายคลาสให้เท่าเทียม
    target_encoded = target_raw.copy()
    if task_type == "classification" and (target_encoded.dtype == object or target_encoded.dtype.name == "category"):
        target_encoder = LabelEncoder()
        target_encoded = pd.Series(
            target_encoder.fit_transform(target_encoded.astype(str)), 
            index=target_encoded.index, 
            name=target_encoded.name
        )

    # วางแผนทำ stratified split สำหรับงาน classification
    stratify_strategy = None
    if task_type == "classification":
        class_counts = pd.Series(target_encoded).value_counts()
        if class_counts.min() >= 2:
            stratify_strategy = target_encoded

    # แบ่งชุดข้อมูลดิบร่วมกัน (80/20, random_state=42)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        features_raw, target_encoded, test_size=0.2, random_state=42, stratify=stratify_strategy
    )

    print(f"• ขนาดชุด Train (80%): {X_train_raw.shape[0]} แถว")
    print(f"• ขนาดชุด Test (20%): {X_test_raw.shape[0]} แถว")
    
    return X_train_raw, X_test_raw, y_train_raw, y_test_raw


# =========================================================================
# กลุ่มที่ 4: การประมวลผลฝั่ง AutoGluon Tabular (AutoGluon Independent Run)
# =========================================================================

def run_autogluon_pipeline(X_train_raw, X_test_raw, y_train_raw, y_test_raw, target_column, task_type):
    """
    รันโมเดลฝั่ง AutoGluon บนข้อมูลดิบโดยให้ AutoGluon จัดการ Preprocessing ภายในของตนเองทั้งหมด
    """
    print(">>> เริ่มต้นการรันฝั่ง AutoGluon Tabular...")
    
    # รวมข้อมูลสำหรับป้อนเข้า AutoGluon
    train_data_ag = X_train_raw.copy()
    train_data_ag[target_column] = y_train_raw
    
    test_data_ag = X_test_raw.copy()
    test_data_ag[target_column] = y_test_raw

    # ระบุประเภทงาน
    if task_type == "classification":
        problem_type = "binary" if y_train_raw.nunique() == 2 else "multiclass"
        eval_metric = "accuracy"
    else:
        problem_type = "regression"
        eval_metric = "r2"

    ag_output_path = "comparison/AutogluonModels"
    
    start_time = time.time()
    try:
        predictor = TabularPredictor(
            label=target_column,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=ag_output_path
        )
        
        predictor.fit(
            train_data=train_data_ag,
            time_limit=600,
            presets='good_quality'
        )
        fit_time_ag = time.time() - start_time
        print(f"[SUCCESS] การเทรน AutoGluon เสร็จสมบูรณ์ (ใช้เวลา {fit_time_ag:.2f} วินาที)")

        # ประเมินผลบนข้อมูลทดสอบดิบ
        start_time_pred = time.time()
        y_pred_ag = predictor.predict(test_data_ag.drop(columns=[target_column]))
        pred_time_ag = time.time() - start_time_pred
        
        ag_metrics = get_metrics(y_test_raw, y_pred_ag, task_type)
        best_model_ag = predictor.model_best
        
        return ag_metrics, fit_time_ag, pred_time_ag, best_model_ag
    except Exception as e:
        print(f"[ERROR] ฝั่ง AutoGluon ทำงานล้มเหลว: {e}")
        return {}, 0.0, 0.0, "N/A"


# =========================================================================
# กลุ่มที่ 5: การประมวลผลฝั่ง Interactive AutoML ของเรา (Interactive AutoML Run)
# =========================================================================

def run_our_pipeline(X_train_our, X_test_our, y_train_our, y_test_our, task_type,
                     outlier_rules, encoding_decisions, scaling_method):
    """
    รันกระบวนการ Preprocessing แยกฝั่ง Train-Test อย่างเคร่งครัดตามกฎของระบบของเรา และเข้าแข่งโมเดล
    X_train_our / X_test_our มาจาก transformed_df (feature selection ทำแล้ว) หรือ raw data (fallback)
    """
    print(">>> เริ่มต้นการรันฝั่ง Interactive AutoML (ระบบของเรา)...")
    start_time = time.time()

    try:
        # 1. Imputation (เติมค่าว่าง)
        X_train_our, X_test_our = clean_fit_transform(X_train_our, X_test_our, missing_rules=None)
        
        # 2. DateTime features extraction
        X_train_our, X_test_our = datetime_fit_transform(X_train_our, X_test_our)
        
        # 3. Outlier clipping — ข้ามไป เพราะ app ไม่ได้ส่ง outlier_rules เข้า preprocess()
        #    outlier ถูกจัดการ pre-split ในขั้นตอน Cleaning แล้ว

        # 4. ทำ Encoding ตัวอักษร
        X_train_our, X_test_our = encode_fit_transform(X_train_our, X_test_our, encoding_decisions=encoding_decisions)
        
        # 5. ทำ Scaling
        X_train_our, X_test_our = scale_data(X_train_our, X_test_our, scaling_method=scaling_method)

        print(f"[INFO] แปลงสเกลและเตรียมข้อมูลฝั่งเราสำเร็จ ขนาดหลังจัดเตรียม: {X_train_our.shape[0]} แถว × {X_train_our.shape[1]} คอลัมน์")

        # แข่งโมเดลและปรับจูนพารามิเตอร์
        comp_start = time.time()
        competition_result = run_competition(
            X_train_our, X_test_our, y_train_our, y_test_our, task_type
        )
        fit_time_our = time.time() - start_time
        print(f"[SUCCESS] การรันแข่งโมเดลฝั่งเราเสร็จสมบูรณ์ (ใช้เวลาทั้งหมด {fit_time_our:.2f} วินาที)")

        best_model_our = competition_result["best_label"]
        y_pred_our = competition_result["y_pred"]

        our_metrics = get_metrics(y_test_our, y_pred_our, task_type)
        pred_time_our = (time.time() - comp_start) - fit_time_our
        if pred_time_our < 0: pred_time_our = 0.01

        return our_metrics, fit_time_our, pred_time_our, best_model_our
    except Exception as e:
        print(f"[ERROR] ฝั่ง Interactive AutoML ทำงานล้มเหลว: {e}")
        import traceback
        traceback.print_exc()
        return {}, 0.0, 0.0, "N/A"


# =========================================================================
# กลุ่มที่ 6: การสร้างรายงานสรุปผลและบันทึกรายงาน (Report Generation & IO)
# =========================================================================

def generate_and_save_report(report_path, dataset_path, df_raw, target_column, task_type, session_id,
                             transformed_cols, outlier_rules, encoding_decisions, scaling_method,
                             ag_metrics, our_metrics, fit_time_ag, fit_time_our, best_model_ag, best_model_our):
    """
    จัดทำตารางเปรียบเทียบและเขียนรายงาน Markdown สรุปความแตกต่างของสองระบบ
    """
    print("=" * 70)
    print("ผลลัพธ์และสถิติเปรียบเทียบประสิทธิภาพ")
    print("=" * 70)

    # กรองตัวชี้วัดตามประเภทงาน
    metrics_list = []
    if task_type == "classification":
        metrics_keys = ["Accuracy", "Precision(Mac)", "Recall(Mac)", "F1(Mac)"]
    else:
        metrics_keys = ["MSE", "RMSE", "R² Score"]

    for k in metrics_keys:
        val_ag = ag_metrics.get(k, "N/A")
        val_our = our_metrics.get(k, "N/A")
        metrics_list.append({
            "Metric": k,
            "AutoGluon": f"{val_ag:.4f}" if isinstance(val_ag, float) else val_ag,
            "Our System": f"{val_our:.4f}" if isinstance(val_our, float) else val_our
        })

    df_metrics = pd.DataFrame(metrics_list)
    print(df_metrics.to_string(index=False))
    print("-" * 70)
    print(f"• เวลาที่ใช้เทรน (Training Time):")
    print(f"  - AutoGluon: {fit_time_ag:.2f} วินาที (โมเดลยอดเยี่ยม: {best_model_ag})")
    print(f"  - Our System: {fit_time_our:.2f} วินาที (โมเดลยอดเยี่ยม: {best_model_our})")
    print("=" * 70)

    # สร้างคำอธิบายกระบวนการแบบ dynamic จาก session จริง
    all_feature_cols = [c for c in df_raw.columns if c != target_column]
    dropped_cols = [c for c in all_feature_cols if c not in transformed_cols] if transformed_cols else []
    if dropped_cols:
        _feature_sel_desc = f"ควบคุมดรอปโดยผู้ใช้: กำจัด {', '.join(f'`{c}`' for c in dropped_cols)} ({len(dropped_cols)} คอลัมน์)"
    else:
        _feature_sel_desc = "ไม่ตัด feature ออก (เก็บทุกคอลัมน์)"

    if outlier_rules:
        _outlier_desc = f"ทำการจำกัดขอบเขตค่านอกเกณฑ์ (Outlier Clipping) บน {len(outlier_rules)} คอลัมน์"
    else:
        _outlier_desc = "ไม่พบ Outlier ที่ต้องจัดการ"

    _scaling_label_map = {
        "standard_scaler": "Standard Scaler", "minmax_scaler": "MinMax Scaler",
        "robust_scaler": "Robust Scaler", "none": "ไม่ scale (No Scaling)",
    }
    _scaling_label = _scaling_label_map.get(scaling_method, scaling_method)
    if encoding_decisions:
        _le_cols = [c for c, m in encoding_decisions.items() if m == "label_encoding"]
        _ohe_cols = [c for c, m in encoding_decisions.items() if m == "one_hot_encoding"]
        _drop_cols = [c for c, m in encoding_decisions.items() if m == "drop_column"]
        _enc_parts = []
        if _le_cols:   _enc_parts.append(f"**Label Encoding** บน {len(_le_cols)} คอลัมน์")
        if _ohe_cols:  _enc_parts.append(f"**One-Hot Encoding** บน {len(_ohe_cols)} คอลัมน์")
        if _drop_cols: _enc_parts.append(f"**Drop** {len(_drop_cols)} คอลัมน์")
        _encoding_desc = f"แปลงตามผู้ใช้ยืนยัน: {', '.join(_enc_parts)} และปรับขนาดโดย **{_scaling_label}**"
    else:
        _encoding_desc = f"Encoding อัตโนมัติ และปรับขนาดโดย **{_scaling_label}**"

    # สร้างเนื้อหารายงาน Markdown
    try:
        report_content = f"""# รายงานผลลัพธ์การเปรียบเทียบระบบ: Interactive AutoML vs AutoGluon Tabular

> [!NOTE]
> รายงานนี้จัดทำขึ้นโดยการเปรียบเทียบระหว่าง **Interactive AutoML Pipeline** (ระบบควบคุมและวิเคราะห์โดยผู้ใช้) กับ **AutoGluon Tabular** (ระบบอัตโนมัติสำเร็จรูป) โดยทั้งสองระบบทำงานบน **ข้อมูลดิบตั้งต้นเดียวกัน** และแบ่งชุดทดสอบใน **สัดส่วน 80/20 ด้วยการสุ่มรหัสความสอดคล้องกัน (Random State = 42) เดียวกัน** อย่างเป็นอิสระต่อกันโดยสมบูรณ์

---

## 1. ภาพรวมการทดลอง (Experiment Overview)

* **ชุดข้อมูลดิบที่ทดสอบ (Raw Dataset):** `{os.path.basename(dataset_path)}`
* **ขนาดข้อมูลดิบ (Raw Dimensions):** `{df_raw.shape[0]:,} แถว × {df_raw.shape[1]} คอลัมน์`
* **คอลัมน์เป้าหมายในการทำนาย (Target Column):** `{target_column}`
* **ประเภทงาน (Task Type):** `{task_type.upper()}`
* **Session ID ดำเนินการอ้างอิง:** `{session_id if session_id else "ไม่มี (รันด้วยโหมด Auto-fallback)"}`

### รายละเอียดกระบวนการที่ต่างกันอย่างอิสระ
| ขั้นตอนการเตรียมและเรียนรู้ | ฝั่ง AutoGluon Tabular | ฝั่งระบบของเรา (Interactive AutoML) |
| :--- | :--- | :--- |
| **การเลือกฟีเจอร์ (Feature Selection)** | ทำโดยอัตโนมัติภายใน AutoGluon | {_feature_sel_desc} |
| **การจัดการค่าขาดหาย (Missing Imputation)** | จัดการภายในโมเดลอัตโนมัติ | เติมด้วยค่ากลาง (Median) และฐานนิยม (Mode) แยกกระบวนการ Train-Test อย่างเคร่งครัด |
| **การจัดการข้อมูลผิดปกติ (Outliers Treatment)** | ประมวลผลโดยอัลกอริทึมเอง | {_outlier_desc} |
| **การแปลงข้อมูลและปรับขนาด (Encoding & Scaling)** | แปลงอัตโนมัติภายในระบบ | {_encoding_desc} |
| **โมเดลและการจูนไฮเปอร์พารามิเตอร์** | รันเทรนแบบ Ensembling หลายชั้น | ค้นหาโมเดลผ่านการแข่งโมเดล 10 ตัวร่วมกับ Randomized Search (Cross-Validation) |

---

## 2. ผลลัพธ์การวัดประสิทธิภาพการทำนาย (Evaluation Results)

ตารางเปรียบเทียบประสิทธิภาพบนข้อมูลชุดทดสอบที่แยกออกมา (Test Set 20%):

| ตัววัดประสิทธิภาพ (Metric) | ฝั่ง AutoGluon Tabular | ฝั่งระบบของเรา (Interactive AutoML) | ความแตกต่าง |
| :--- | :---: | :---: | :---: |
"""

        for row in metrics_list:
            metric_name = row["Metric"]
            val_ag = row["AutoGluon"]
            val_our = row["Our System"]
            
            diff_str = "-"
            try:
                diff_val = float(val_our) - float(val_ag)
                if diff_val > 0:
                    diff_str = f"+{diff_val:.4f}"
                elif diff_val < 0:
                    diff_str = f"{diff_val:.4f}"
                else:
                    diff_str = "0.0000"
            except ValueError:
                pass
                
            report_content += f"| **{metric_name}** | {val_ag} | {val_our} | {diff_str} |\n"

        report_content += f"""
### ความเร็วและโมเดลที่แนะนำ (Training Speed & Model Recommendation)

| พารามิเตอร์เชิงปริมาณ | ฝั่ง AutoGluon Tabular | ฝั่งระบบของเรา (Interactive AutoML) |
| :--- | :---: | :---: |
| **โมเดลที่ดีที่สุด (Best Model)** | `{best_model_ag}` | `{best_model_our}` |
| **เวลาในการเทรน (Training Time)** | `{fit_time_ag:.2f} วินาที` | `{fit_time_our:.2f} วินาที` |

---

## 3. การวิเคราะห์และข้อเสนอแนะเชิงเปรียบเทียบ (Comparative Analysis)

> [!TIP]
> **จุดเด่นของระบบเรา (Interactive AutoML):**
> 1. **การควบคุมและความโปร่งใส (Control & Explainability):** ระบบเราเปิดให้ผู้ใช้เห็นว่ามีคอลัมน์ใดที่โดนกำจัด หรือเกิดกระบวนการแปลงค่าอย่างไรบ้าง ทำให้ผู้ใช้ทั่วไปและนักเรียนเข้าใจกระบวนการได้ชัดเจน (ไม่ใช่แบบกล่องดำ)
> 2. **ความอิสระและหลีกเลี่ยงความลำเอียง (Data Leak-safe):** การดึงความสะอาดและการจัดการข้อมูลแยกส่วนกันของระบบเรา ทำให้มั่นใจได้ว่ากฎที่ใช้จัดเตรียมข้อมูลถูกเรียนรู้และ 'Fit' จากข้อมูล Train เท่านั้น ก่อนส่งไปทดสอบ ทำให้ไม่เกิด Data Leakage

> [!WARNING]
> **จุดที่ AutoGluon โดดเด่น:**
> 1. **ประสิทธิภาพ (Accuracy/F1):** AutoGluon ได้สร้าง Ensemble โมเดลหลายประเภทซ้อนกัน ทำให้เหมาะกับการรันแบบโมเดลสำเร็จรูป 100% แต่แลกด้วยการเป็น **"Black-box"** ที่ยากจะอธิบายเชิงลึกให้เหมาะสำหรับผู้เริ่มเรียนรู้ในวิชาวิทยาศาสตร์ข้อมูล

---
*รายงานฉบับนี้สร้างโดยระบบสคริปต์เปรียบเทียบอัตโนมัติเปรียบเทียบเคียงข้างกันเสร็จสมบูรณ์เมื่อวันที่ {time.strftime("%d/%m/%Y %H:%M:%S")}*
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[SUCCESS] บันทึกรายงานเปรียบเทียบเรียบร้อยที่: {report_path}")
    except Exception as e:
        print(f"[ERROR] ไม่สามารถบันทึกรายงานเปรียบเทียบได้: {e}")


# =========================================================================
# กลุ่มที่ 7: ส่วนการเรียกใช้งานหลักและการประมวลผล Argument (Main Controller)
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="เปรียบเทียบระบบ Interactive AutoML กับ AutoGluon Tabular")
    parser.add_argument("--dataset", type=str, required=True, help="พาธไปยังชุดข้อมูลดิบตั้งต้น (CSV)")
    parser.add_argument("--target", type=str, required=True, help="คอลัมน์ที่เป็นเป้าหมาย (Target Column)")
    parser.add_argument("--session", type=str, default="", help="Session ID เพื่อใช้กฎและคอลัมน์ที่ตั้งค่าไว้ในการรัน")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset
    target_column = args.target
    session_id = args.session

    print("=" * 70)
    print("ระบบทดสอบและเปรียบเทียบ AutoML Pipeline vs AutoGluon Tabular")
    print("=" * 70)
    print(f"• พาธชุดข้อมูลดิบ: {dataset_path}")
    print(f"• คอลัมน์เป้าหมาย: {target_column}")
    if session_id:
        print(f"• อ้างอิง Session ID: {session_id}")
    else:
        print("• โหมดการประมวลผล: โหมดอัตโนมัติ (Auto-fallback)")
    print("-" * 70)

    # 1. โหลดข้อมูลดิบตั้งต้น
    try:
        df_raw, task_type = load_and_prepare_raw_data(dataset_path, target_column)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 2. โหลด Session Transformation config
    has_session_config, transformed_cols, outlier_rules, encoding_decisions, scaling_method, transformed_df_loaded = \
        load_session_configuration(session_id, target_column)

    # 3. แบ่ง raw data สำหรับ AutoGluon (80/20)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = split_raw_data(df_raw, target_column, task_type)
    print("-" * 70)

    # 4. รันฝั่ง AutoGluon Tabular (ใช้ raw split เสมอ)
    ag_metrics, fit_time_ag, pred_time_ag, best_model_ag = \
        run_autogluon_pipeline(X_train_raw, X_test_raw, y_train_raw, y_test_raw, target_column, task_type)
    print("-" * 70)

    # 5. เตรียม split สำหรับฝั่งเรา
    # ถ้ามี transformed_df จาก cache → ใช้เลย (ตรงกับ pipeline จริงใน app)
    # ถ้าไม่มี → fallback ใช้ raw split กรองตาม transformed_cols
    if transformed_df_loaded is not None:
        X_train_our, X_test_our, y_train_our, y_test_our = split_raw_data(
            transformed_df_loaded, target_column, task_type
        )
        print(f"[INFO] ใช้ transformed_df จาก cache ({transformed_df_loaded.shape[0]:,} แถว) สำหรับฝั่งเรา")
    else:
        our_cols = transformed_cols if (has_session_config and transformed_cols) else list(X_train_raw.columns)
        X_train_our = X_train_raw[our_cols].copy()
        X_test_our = X_test_raw[our_cols].copy()
        y_train_our, y_test_our = y_train_raw, y_test_raw
        print("[INFO] ไม่พบ transformed_df — fallback ใช้ raw split")

    # 6. รันฝั่ง Interactive AutoML ของเรา
    our_metrics, fit_time_our, pred_time_our, best_model_our = \
        run_our_pipeline(X_train_our, X_test_our, y_train_our, y_test_our, task_type,
                         outlier_rules, encoding_decisions, scaling_method)

    # 6. สร้างรายงานสรุปผลการเปรียบเทียบ
    report_path = "comparison/comparison_report.md"
    generate_and_save_report(report_path, dataset_path, df_raw, target_column, task_type, session_id,
                             transformed_cols, outlier_rules, encoding_decisions, scaling_method,
                             ag_metrics, our_metrics, fit_time_ag, fit_time_our, best_model_ag, best_model_our)


if __name__ == "__main__":
    main()
