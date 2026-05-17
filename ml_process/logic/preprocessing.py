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
 
# Datetime Feature Extraction
def datetime_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame) -> tuple:
    """
    ตรวจจับและแตก Feature จากคอลัมน์ที่เป็น Datetime อัตโนมัติ (Year, Month, Day, DayOfWeek)
    """
    features_train = features_train.copy()
    features_test = features_test.copy()
    
    for col in features_train.columns:
        # 1. พยายามแปลง Object ที่มีรูปแบบวันที่ให้เป็น Datetime
        if features_train[col].dtype == 'object':
            try:
                non_nulls = features_train[col].dropna()
                if not non_nulls.empty:
                    sample = str(non_nulls.iloc[0])
                    # เช็คเบื้องต้นว่ามีสัญลักษณ์วันที่
                    if any(char in sample for char in ['-', '/', '.', ':']):
                        pd.to_datetime(sample) # ลองแปลงดู
                        features_train[col] = pd.to_datetime(features_train[col], errors='coerce')
                        features_test[col]  = pd.to_datetime(features_test[col], errors='coerce')
            except:
                continue

        # 2. ถ้าเป็น Datetime แล้ว ให้ทำการแตกข้อมูลเป็นตัวเลข
        if pd.api.types.is_datetime64_any_dtype(features_train[col]):
            for df in [features_train, features_test]:
                df[f"{col}_year"]      = df[col].dt.year
                df[f"{col}_month"]     = df[col].dt.month
                df[f"{col}_day"]       = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                # ดึงชั่วโมงมาด้วยถ้ามีข้อมูลเวลา
                if df[col].dt.hour.sum() > 0:
                    df[f"{col}_hour"] = df[col].dt.hour
            
            # ลบคอลัมน์ Datetime เดิมออกเพื่อให้โมเดลทำงานได้
            features_train = features_train.drop(columns=[col])
            features_test  = features_test.drop(columns=[col])
            
    return features_train, features_test

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
def clean_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame, missing_rules: dict = None) -> tuple:
    """
    เติมค่าว่าง (Missing Value) ป้องกัน Data Leakage โดยคำนวณค่าสถิติจาก features_train เท่านั้น
    และใช้กฎจากหน้า UI Data Cleaning (missing_rules)
    """
    if missing_rules is None:
        missing_rules = {}
        
    # จัดการข้อมูลประเภท Boolean และ Infinity
    for dataset_split in [features_train, features_test]:
        boolean_columns = dataset_split.select_dtypes(include="bool").columns
        for col in boolean_columns:
            dataset_split[col] = dataset_split[col].astype(int)
        dataset_split.replace([np.inf, -np.inf], 0, inplace=True)

    # คำนวณค่าจาก features_train ตาม rules
    fill_values_dict = {}
    for column_name in features_train.columns:
        if features_train[column_name].isna().any() or features_test[column_name].isna().any():
            strategy = missing_rules.get(column_name, "median" if pd.api.types.is_numeric_dtype(features_train[column_name]) else "most frequent")
            
            if strategy == "mean":
                fill_values_dict[column_name] = features_train[column_name].mean()
            elif strategy == "median":
                fill_values_dict[column_name] = features_train[column_name].median()
            elif strategy == "median (rounded)":
                fill_values_dict[column_name] = round(features_train[column_name].median())
            elif strategy == "most frequent":
                modes = features_train[column_name].mode()
                fill_values_dict[column_name] = modes.iloc[0] if not modes.empty else 0
            elif strategy == "forward fill":
                features_train[column_name] = features_train[column_name].ffill()
                features_test[column_name] = features_test[column_name].ffill()
                continue
            elif strategy == "backward fill":
                features_train[column_name] = features_train[column_name].bfill()
                features_test[column_name] = features_test[column_name].bfill()
                continue
            elif strategy == "drop rows":
                # Drop rows happens before split. If it reaches here, fallback to median/mode
                if pd.api.types.is_numeric_dtype(features_train[column_name]):
                    fill_values_dict[column_name] = features_train[column_name].median()
                else:
                    modes = features_train[column_name].mode()
                    fill_values_dict[column_name] = modes.iloc[0] if not modes.empty else 0
                
            if pd.isna(fill_values_dict.get(column_name, 0)):
                fill_values_dict[column_name] = 0

    # เติมค่าว่างให้ทั้ง Train และ Test
    if fill_values_dict:
        features_train = features_train.fillna(fill_values_dict)
        features_test  = features_test.fillna(fill_values_dict)

    return features_train, features_test

# Outlier Handling (Clipping)
def outlier_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame, outlier_rules: dict = None) -> tuple:
    """
    คลิป Outliers โดยใช้ขอบเขตที่ได้จากกฎหน้า UI (ซึ่งตั้งอยู่บนโครงสร้างข้อมูลดั้งเดิม แต่ตัดที่ Train/Test แบบแยกกัน)
    """
    if outlier_rules is None:
        outlier_rules = {}
        
    for column_name, rule in outlier_rules.items():
        if column_name in features_train.columns:
            strategy = rule.get("strategy")
            lower = rule.get("lower")
            upper = rule.get("upper")
            
            if strategy == "clip":
                features_train[column_name] = features_train[column_name].clip(lower=lower, upper=upper)
                features_test[column_name] = features_test[column_name].clip(lower=lower, upper=upper)
                
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
def preprocess(dataset: pd.DataFrame, target_column: str, scaling_method: str = "standard_scaler", missing_rules: dict = None, outlier_rules: dict = None) -> tuple:
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

    # Datetime Handling (แตก Feature วันเวลาออกเป็นตัวเลข)
    features_train, features_test = datetime_fit_transform(features_train, features_test)

    # Encoding
    features_train, features_test = encode_fit_transform(features_train, features_test)

    # Outlier Clipping
    features_train, features_test = outlier_fit_transform(features_train, features_test, outlier_rules)

    # Cleaning
    features_train, features_test = clean_fit_transform(features_train, features_test, missing_rules)

    # Scaling
    features_train, features_test = scale_data(features_train, features_test, scaling_method)

    return features_train, features_test, target_train, target_test, task_type