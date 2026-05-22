# Libraries
import pandas as pd


# ตรวจหาและแตก datetime columns เป็น year/month/day/dayofweek/(hour) fit จาก train ใช้ใน preprocess
def datetime_fit_transform(features_train: pd.DataFrame, features_test: pd.DataFrame) -> tuple:
    features_train = features_train.copy()
    features_test = features_test.copy()

    for col in features_train.columns:
        if features_train[col].dtype == 'object':
            try:
                non_nulls = features_train[col].dropna()
                if not non_nulls.empty:
                    sample = str(non_nulls.iloc[0])
                    if any(char in sample for char in ['-', '/', '.', ':']):
                        pd.to_datetime(sample)
                        features_train[col] = pd.to_datetime(features_train[col], errors='coerce')
                        features_test[col]  = pd.to_datetime(features_test[col], errors='coerce')
            except:
                continue

        if pd.api.types.is_datetime64_any_dtype(features_train[col]):
            # decide from train only to avoid leakage
            include_hour = features_train[col].dt.hour.sum() > 0
            for df in [features_train, features_test]:
                df[f"{col}_year"]      = df[col].dt.year
                df[f"{col}_month"]     = df[col].dt.month
                df[f"{col}_day"]       = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                if include_hour:
                    df[f"{col}_hour"] = df[col].dt.hour

            features_train = features_train.drop(columns=[col])
            features_test  = features_test.drop(columns=[col])
            
    return features_train, features_test