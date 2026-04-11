"""ml_process/preprocess.py — preprocessing + split (ป้องกัน Data Leakage)"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from ml_process.config import MAX_ROWS_TRAIN


def detect_task(df: pd.DataFrame, target_col: str) -> str:
    """
    ตรวจสอบว่าเป็น classification หรือ regression

    เกณฑ์:
    - string/category → classification เสมอ
    - numeric unique ≤ 15 → classification (discrete class)
    - numeric unique > 100 → regression เสมอ (continuous)
    - numeric 16-100 unique → ดู ratio:
        ratio ≥ 1% → regression (เช่น 30 unique / 250 rows = 12%)
        ratio < 1% → classification (เช่น 30 unique / 30,000 rows = 0.1%)
    """
    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 15:
        return "classification"
    if n_unique > 100:
        return "regression"
    # 16-100 unique: ดู ratio
    ratio = n_unique / len(y)
    return "regression" if ratio >= 0.01 else "classification"


def _sample(X, y, max_rows):
    if len(X) <= max_rows:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _encode_fit_transform(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          force_label_cols: list | None = None) -> tuple:
    """
    Encode categorical columns:
    - fit LabelEncoder / get_dummies บน X_train เท่านั้น
    - transform X_test ด้วย schema เดียวกัน (ป้องกัน Leakage)

    force_label_cols: columns ที่ user เลือก label_encoding ใน Transformation page
                      (ยังเป็น categorical อยู่เพราะ transformer.py ไม่ fit ให้แล้ว)
    """
    force_label_cols = set(force_label_cols or [])
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    # รวม force_label_cols ที่ user เลือกไว้ (อาจยังไม่ถูก detect เป็น cat)
    extra = [c for c in force_label_cols if c in X_train.columns and c not in cat_cols]
    cat_cols = cat_cols + extra

    for col in cat_cols:
        # ถ้า user เลือก label_encoding ให้ใช้ LabelEncoder เสมอ (ไม่ว่า cardinality จะเป็นเท่าไร)
        use_label = col in force_label_cols
        if not use_label and X_train[col].nunique() <= 15:
            # One-hot: fit schema จาก X_train
            dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=True, dtype=int)
            X_train = pd.concat([X_train.drop(columns=[col]), dummies_train], axis=1)

            # transform X_test ด้วย column schema ของ X_train
            dummies_test = pd.get_dummies(X_test[col], prefix=col, drop_first=True, dtype=int)
            X_test = pd.concat([X_test.drop(columns=[col]), dummies_test], axis=1)

            # align columns — เติม 0 ถ้า X_test ขาด column ที่มีใน X_train
            for c in dummies_train.columns:
                if c not in X_test.columns:
                    X_test[c] = 0
            # ตัด column ที่มีใน X_test แต่ไม่มีใน X_train ออก
            X_test = X_test[[c for c in X_train.columns if c in X_test.columns]]
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        else:
            # LabelEncoder: fit บน X_train เท่านั้น
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            known = set(le.classes_)
            X_train[col] = le.transform(X_train[col].astype(str))
            # X_test: ค่าที่ไม่รู้จัก → แทนด้วย 0
            X_test[col] = X_test[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in known else 0
            )

    return X_train, X_test


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive cleanup — ไม่มี fit จึงปลอดภัยทั้ง X_train และ X_test
    หมายเหตุ: ไม่ encode categorical ที่นี่ เพราะจะได้ LabelEncoder คนละตัว
              ต่อ split ทำให้ mapping ไม่ตรงกัน — ใช้ _encode_fit_transform() แทน
    """
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    X = X.replace([np.inf, -np.inf], 0)
    for col in X.columns:
        if X[col].isna().any():
            fill = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0
            X[col] = X[col].fillna(fill if not pd.isna(fill) else 0)
    return X


def _scale(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str) -> tuple:
    """
    Fit scaler บน X_train เท่านั้น แล้ว transform ทั้งคู่ (ป้องกัน Leakage)
    """
    if method == "no_scaling":
        return X_train, X_test

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return X_train, X_test

    scaler_map = {
        "standard_scaler": StandardScaler(),
        "minmax_scaler":   MinMaxScaler(),
        "robust_scaler":   RobustScaler(),
    }
    scaler = scaler_map.get(method)
    if not scaler:
        return X_train, X_test

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    return X_train, X_test


def preprocess(df: pd.DataFrame, target_col: str,
               scaling_method: str = "standard_scaler",
               label_encoding_cols: list | None = None) -> tuple:
    """
    Pipeline ที่ปลอดจาก Data Leakage:
      1. แยก X / y
      2. sample (ถ้า dataset ใหญ่)
      3. split 80/20
      4. encode  ← fit บน X_train เท่านั้น
      5. clean   ← ไม่มี fit
      6. scale   ← fit บน X_train เท่านั้น
    """
    task_type = detect_task(df, target_col)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # sample ก่อน split เพื่อลดขนาดข้อมูล
    X, y = _sample(X, y, MAX_ROWS_TRAIN)

    # split ก่อน encode/scale
    stratify = None
    if task_type == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # encode หลัง split — fit บน X_train เท่านั้น
    # รวม label_encoding_cols จาก transformer.py (user เลือกไว้แต่ยังไม่ได้ encode)
    X_train, X_test = _encode_fit_transform(X_train, X_test,
                                            force_label_cols=label_encoding_cols or [])

    # clean (ไม่มี fit — ทำได้หลัง split ได้เลย)
    X_train = _clean(X_train)
    X_test  = _clean(X_test)

    # scale หลัง split — fit บน X_train เท่านั้น
    X_train, X_test = _scale(X_train, X_test, scaling_method)

    return X_train, X_test, y_train, y_test, task_type