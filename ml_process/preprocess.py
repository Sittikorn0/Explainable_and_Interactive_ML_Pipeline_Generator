"""
ml_process/preprocess.py
Data Splitting + Auto Preprocessing
ไม่มี Streamlit ในไฟล์นี้ → test ได้อิสระ
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_process.config import MAX_ROWS_TRAIN

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Task Detection ────────────────────────────────────────────
def detect_task(df: pd.DataFrame, target_col: str) -> str:
    """
    ตรวจสอบว่าเป็น classification หรือ regression
    เกณฑ์:
      - target เป็น string/category → classification
      - target เป็น numeric แต่ unique น้อย (≤10) → classification
      - อื่นๆ → regression
    """
    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique() <= 10:
        return "classification"
    if y.nunique() / len(y) < 0.05:
        return "classification"
    return "regression"


# ── Helpers ───────────────────────────────────────────────────
def _sample(X: pd.DataFrame, y: pd.Series, max_rows: int):
    """สุ่ม sample ถ้า rows เกิน cap"""
    if len(X) <= max_rows:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _clean_X(X: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive cleanup ก่อนส่ง model:
    - bool → int
    - object ที่หลงเหลือ → label encode
    - inf → 0
    - missing → median หรือ 0
    """
    X = X.copy()

    # bool → int
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)

    # object/category ที่ยังหลงเหลือ
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # inf → 0
    X = X.replace([np.inf, -np.inf], 0)

    # missing → median / 0
    for col in X.columns:
        if X[col].isna().any():
            fill = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0
            X[col] = X[col].fillna(fill if not pd.isna(fill) else 0)

    return X


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns:
    - cardinality ≤ 15 → one-hot encoding
    - cardinality > 15 → label encoding (ป้องกัน column explosion)
    """
    for col in list(X.select_dtypes(include=["object", "category"]).columns):
        if X[col].nunique() <= 15:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


# ── Main Preprocess ───────────────────────────────────────────
def preprocess(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Auto preprocess สำหรับ ML:
      1. แยก X, y
      2. Encode categorical features
      3. Clean X
      4. Sample ถ้าเกิน MAX_ROWS_TRAIN
      5. Train/Test split 80/20 พร้อม stratify (classification)
      6. Standard scale numeric columns

    Returns:
        X_train, X_test, y_train, y_test, task_type
    """
    task_type = detect_task(df, target_col)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # encode
    X = _encode(X)

    # clean
    X = _clean_X(X)

    # cap rows
    X, y = _sample(X, y, MAX_ROWS_TRAIN)

    # stratify เฉพาะ classification ที่ทุก class มี ≥ 2 samples
    stratify = None
    if task_type == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # scale numeric
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    if num_cols:
        scaler = StandardScaler()
        X_train = X_train.copy()
        X_test  = X_test.copy()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, task_type