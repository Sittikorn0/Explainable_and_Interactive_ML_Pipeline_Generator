import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from ml_process.logic.config import MAX_ROWS_TRAIN
from ml_process.logic.data_analyzer import detect_task

# Data Sampling
def sample_data(features: pd.DataFrame, target: pd.Series, max_sample_rows: int) -> tuple:
    """
    ลดขนาดข้อมูลโดยการสุ่ม (Sampling) หากข้อมูลมีขนาดใหญ่เกินไป
    เพื่อลดเวลาในการ Train โมเดล
    """
    if len(features) <= max_sample_rows:
        return features, target
        
    sampled_indices = np.random.RandomState(42).choice(len(features), max_sample_rows, replace=False)
    
    sampled_features = features.iloc[sampled_indices].reset_index(drop=True)
    sampled_target = target.iloc[sampled_indices].reset_index(drop=True)
    
    return sampled_features, sampled_target

# Categorical Encoding
def encode_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame) -> tuple:
    """
    แปลงข้อมูลตัวอักษร (Categorical) เป็นตัวเลข:
    - ป้องกัน Data Leakage โดยการสร้างกฎ (Fit) จาก features_train เท่านั้น
    - นำกฎนั้นไปประยุกต์ใช้ (Transform) กับทั้ง Train และ Test
    """
    categorical_columns = features_train.select_dtypes(include=["object", "category"]).columns.tolist()

    for column_name in categorical_columns:
        num_unique_values = features_train[column_name].nunique()
        
        if num_unique_values <= 15:
            # ใช้ One-hot Encoding สำหรับข้อมูลที่มีความหลากหลายน้อย (Low Cardinality)
            one_hot_train = pd.get_dummies(features_train[column_name], prefix=column_name, drop_first=True, dtype=int)
            features_train = pd.concat([features_train.drop(columns=[column_name]), one_hot_train], axis=1)

            # แปลง features_test ด้วยโครงสร้าง (Schema) เดียวกับ features_train
            prefix = column_name + "_"
            for new_col_name in one_hot_train.columns:
                category_value = new_col_name[len(prefix):]
                features_test[new_col_name] = (features_test[column_name].astype(str) == category_value).astype(int)
                
            features_test = features_test.drop(columns=[column_name])
        else:
            # ใช้ Label Encoding สำหรับข้อมูลที่มีความหลากหลายมาก (High Cardinality)
            label_encoder = LabelEncoder()
            label_encoder.fit(features_train[column_name].astype(str))
            
            known_categories = set(label_encoder.classes_)
            features_train[column_name] = label_encoder.transform(features_train[column_name].astype(str))
            
            # ค่าสำรอง (Fallback) กรณีเจอ Category ใหม่ใน Test Set
            most_frequent_value = features_train[column_name].mode()
            fallback_value = int(most_frequent_value.iloc[0]) if len(most_frequent_value) > 0 else 0
            
            features_test[column_name] = features_test[column_name].astype(str).apply(
                lambda val: label_encoder.transform([val])[0] if val in known_categories else fallback_value
            )

    # จัดระเบียบคอลัมน์ของ Test Set ให้ตรงกับ Train Set หลังจาก Encode เสร็จ
    features_test = features_test.reindex(columns=features_train.columns, fill_value=0)
    
    return features_train, features_test

# Missing Value Handling (Cleaning)
def clean_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame) -> tuple:
    """
    เติมค่าว่าง (Missing Value) ป้องกัน Data Leakage โดยคำนวณค่าสถิติจาก features_train เท่านั้น
    """
    # จัดการข้อมูลประเภท Boolean และ Infinity
    for dataset_split in [features_train, features_test]:
        boolean_columns = dataset_split.select_dtypes(include="bool").columns
        for col in boolean_columns:
            dataset_split[col] = dataset_split[col].astype(int)
        dataset_split.replace([np.inf, -np.inf], 0, inplace=True)

    # คำนวณค่ามัธยฐาน (Median) จาก features_train 
    fill_values_dict = {}
    for column_name in features_train.columns:
        if features_train[column_name].isna().any() or features_test[column_name].isna().any():
            if pd.api.types.is_numeric_dtype(features_train[column_name]):
                fill_values_dict[column_name] = features_train[column_name].median()
            else:
                fill_values_dict[column_name] = 0
                
            if pd.isna(fill_values_dict[column_name]):
                fill_values_dict[column_name] = 0

    # เติมค่าว่างให้ทั้ง Train และ Test
    if fill_values_dict:
        features_train = features_train.fillna(fill_values_dict)
        features_test  = features_test.fillna(fill_values_dict)

    return features_train, features_test

# Scaling
def scale_data(features_train: pd.DataFrame, features_test: pd.DataFrame, scaling_method: str) -> tuple:
    """
    ปรับขนาดข้อมูล (Scaling) ป้องกัน Data Leakage โดยการ Fit บน features_train เท่านั้น
    """
    if scaling_method == "no_scaling":
        return features_train, features_test

    numeric_columns = features_train.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return features_train, features_test

    # กรณี Log Transform
    if scaling_method == "log_transform":
        features_train = features_train.copy()
        features_test  = features_test.copy()
        
        # ทำ log1p เฉพาะคอลัมน์ที่ค่าต่ำสุด >= 0
        valid_log_columns = [col for col in numeric_columns if features_train[col].min() >= 0]
        for col in valid_log_columns:
            features_train[col] = np.log1p(features_train[col])
            features_test[col]  = np.log1p(np.maximum(features_test[col], 0))  # ล็อกค่าให้เป็น 0 กรณีติดลบ
            
        # ตามด้วย Standard Scaler ทับอีกรอบ (Fit เฉพาะ Train)
        standard_scaler = StandardScaler()
        features_train[numeric_columns] = standard_scaler.fit_transform(features_train[numeric_columns])
        features_test[numeric_columns]  = standard_scaler.transform(features_test[numeric_columns])
        
        return features_train, features_test

    # กรณี Scaler อื่นๆ
    available_scalers = {
        "standard_scaler": StandardScaler(),
        "minmax_scaler":   MinMaxScaler(),
        "robust_scaler":   RobustScaler(),
    }
    
    selected_scaler = available_scalers.get(scaling_method)
    if not selected_scaler:
        return features_train, features_test

    features_train = features_train.copy()
    features_test  = features_test.copy()
    
    features_train[numeric_columns] = selected_scaler.fit_transform(features_train[numeric_columns])
    features_test[numeric_columns]  = selected_scaler.transform(features_test[numeric_columns])
    
    return features_train, features_test

# Main Pipeline
def preprocess(dataset: pd.DataFrame, target_column: str, scaling_method: str = "standard_scaler") -> tuple:
    """
    ท่อส่งข้อมูลหลัก (Pipeline) สำหรับเตรียมข้อมูลก่อนเข้าโมเดล
    ออกแบบให้ปราศจาก Data Leakage อย่างสมบูรณ์ 
    """
    task_type = detect_task(dataset, target_column)
    
    features = dataset.drop(columns=[target_column]).copy()
    target = dataset[target_column].copy()

    # Validation เบื้องต้น
    num_unique_classes = target.nunique()
    if task_type == "classification" and num_unique_classes < 2:
        unique_values = target.dropna().unique().tolist()
        raise ValueError(
            f"Target column '{target_column}' มีค่าเพียง {num_unique_classes} คลาส ({unique_values}) "
            f"— ต้องการอย่างน้อย 2 คลาส สำหรับกระบวนการ Classification "
            f"กรุณาตรวจสอบชุดข้อมูลหรือเลือก Target column ใหม่"
        )

    # Sampling
    features, target = sample_data(features, target, MAX_ROWS_TRAIN)

    # Train/Test Split
    stratify_strategy = None
    if task_type == "classification":
        class_counts = pd.Series(target).value_counts()
        if class_counts.min() >= 2:
            stratify_strategy = target

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=stratify_strategy
    )

    # Encoding
    features_train, features_test = encode_fit_transform(features_train, features_test)

    # Cleaning
    features_train, features_test = clean_fit_transform(features_train, features_test)

    # Scaling
    features_train, features_test = scale_data(features_train, features_test, scaling_method)

    return features_train, features_test, target_train, target_test, task_type