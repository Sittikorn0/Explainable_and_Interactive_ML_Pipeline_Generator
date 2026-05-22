# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ปรับขนาดข้อมูล fit บน train เท่านั้น (ป้องกัน leakage) รองรับ standard/minmax/robust/log+standard/no_scaling ใช้ใน preprocess
def scale_data(features_train: pd.DataFrame, features_test: pd.DataFrame, scaling_method: str) -> tuple:
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