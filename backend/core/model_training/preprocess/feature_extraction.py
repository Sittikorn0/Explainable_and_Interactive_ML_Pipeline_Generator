# Libraries
import pandas as pd


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
            include_hour = features_train[col].dt.hour.sum() > 0
            for df in [features_train, features_test]:
                df[f"{col}_year"]      = df[col].dt.year
                df[f"{col}_month"]     = df[col].dt.month
                df[f"{col}_day"]       = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                # ดึงชั่วโมงมาด้วยถ้า train มีข้อมูลเวลา (ตัดสินใจจาก train เท่านั้น — leak-safe)
                if include_hour:
                    df[f"{col}_hour"] = df[col].dt.hour
            
            # ลบคอลัมน์ Datetime เดิมออกเพื่อให้โมเดลทำงานได้
            features_train = features_train.drop(columns=[col])
            features_test  = features_test.drop(columns=[col])
            
    return features_train, features_test