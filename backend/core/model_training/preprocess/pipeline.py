# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Logic Import
from backend.function.analyzer.task_detection import detect_task
from backend.core.model_training.config.config_ml import MAX_ROWS_TRAIN
from backend.core.model_training.preprocess.cleaning import *
from backend.core.model_training.preprocess.feature_extraction import *
from backend.core.model_training.preprocess.encoding import *
from backend.core.model_training.preprocess.scaling import *

# Function
def sample_data(features: pd.DataFrame, target: pd.Series, max_sample_rows: int,
                task_type: str = "regression") -> tuple:
    """
    ลดขนาดข้อมูลโดยการสุ่ม (Sampling) หากข้อมูลมีขนาดใหญ่เกินไป
    เพื่อลดเวลาในการ Train โมเดล
    """
    if len(features) <= max_sample_rows:
        return features, target

    if task_type == "classification":
        try:
            class_counts = pd.Series(target).value_counts()
            if class_counts.min() >= 2:
                sample_ratio = max_sample_rows / len(features)
                _, sampled_features, _, sampled_target = train_test_split(
                    features, target, test_size=sample_ratio, random_state=42, stratify=target
                )
                return sampled_features.reset_index(drop=True), sampled_target.reset_index(drop=True)
        except Exception:
            pass  # fallback to random sampling

    sampled_indices = np.random.RandomState(42).choice(len(features), max_sample_rows, replace=False)
    sampled_features = features.iloc[sampled_indices].reset_index(drop=True)
    sampled_target = target.iloc[sampled_indices].reset_index(drop=True)
    return sampled_features, sampled_target

def preprocess(dataset: pd.DataFrame, target_column: str, scaling_method: str = "standard_scaler",
               missing_rules: dict = None, outlier_rules: dict = None,
               encoding_decisions: dict | None = None) -> tuple:
    """
    ท่อส่งข้อมูลหลัก (Pipeline) สำหรับเตรียมข้อมูลก่อนเข้าโมเดล
    ออกแบบให้ปราศจาก Data Leakage อย่างสมบูรณ์

    encoding_decisions: dict จาก Transform step  ถ้าให้มา encode จะทำหลัง split (leak-safe)
                        ถ้าไม่ให้มา → fallback cardinality อัตโนมัติ
    """
    task_type = detect_task(dataset, target_column)
    
    features = dataset.drop(columns=[target_column]).copy()
    target = dataset[target_column].copy()

    # Validation เบื้องต้น (บน full data ก่อน sampling  ป้องกัน minority class ถูก sample ออก)
    num_unique_classes = target.nunique()
    if task_type == "classification" and num_unique_classes < 2:
        unique_values = target.dropna().unique().tolist()
        raise ValueError(
            f"Target column '{target_column}' มีค่าเพียง {num_unique_classes} คลาส ({unique_values}) "
            f" ต้องการอย่างน้อย 2 คลาส สำหรับกระบวนการ Classification "
            f"กรุณาตรวจสอบชุดข้อมูลหรือเลือก Target column ใหม่"
        )

    # Sampling (stratified สำหรับ classification เพื่อรักษา class distribution)
    features, target = sample_data(features, target, MAX_ROWS_TRAIN, task_type)

    # Validation หลัง sampling  ถ้า minority class น้อยมากจน stratified sampling พลาด
    if task_type == "classification" and target.nunique() < 2:
        raise ValueError(
            f"Target column '{target_column}' มีข้อมูลน้อยเกินไปหลัง Sampling "
            f" minority class อาจมีตัวอย่างน้อยกว่า 2 กรุณาตรวจสอบชุดข้อมูล"
        )

    # Train/Test Split
    stratify_strategy = None
    if task_type == "classification":
        class_counts = pd.Series(target).value_counts()
        if class_counts.min() >= 2:
            stratify_strategy = target

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=stratify_strategy
    )

    # 1. Cleaning  จัดการ Missing Values ก่อน เพื่อไม่ให้ NaN ถูก Encode เป็น "nan" string
    features_train, features_test = clean_fit_transform(features_train, features_test, missing_rules)

    # 2. Datetime Handling  แตก Feature วันเวลาออกเป็นตัวเลข (ข้อมูลสะอาดแล้ว ไม่มี NaT)
    features_train, features_test = datetime_fit_transform(features_train, features_test)

    # 3. Outlier Clipping  clip บน numeric ดั้งเดิมก่อน Encode
    features_train, features_test = outlier_fit_transform(features_train, features_test, outlier_rules)

    # 4. Encoding  fit บน train เท่านั้น ป้องกัน Data Leakage (categorical ไม่มี NaN แล้ว)
    features_train, features_test = encode_fit_transform(features_train, features_test, encoding_decisions)

    # 5. Scaling
    features_train, features_test = scale_data(features_train, features_test, scaling_method)

    return features_train, features_test, target_train, target_test, task_type