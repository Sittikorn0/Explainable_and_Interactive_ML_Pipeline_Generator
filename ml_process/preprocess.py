"""ml_process/preprocess.py — preprocessing + split"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from ml_process.config import MAX_ROWS_TRAIN


def detect_task(df: pd.DataFrame, target_col: str) -> str:
    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if y.nunique() <= 10 or y.nunique() / len(y) < 0.05:
        return "classification"
    return "regression"


def _sample(X, y, max_rows):
    if len(X) <= max_rows:
        return X, y
    idx = np.random.RandomState(42).choice(len(X), max_rows, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    for col in list(X.select_dtypes(include=["object", "category"]).columns):
        if X[col].nunique() <= 15:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.replace([np.inf, -np.inf], 0)
    for col in X.columns:
        if X[col].isna().any():
            fill = X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0
            X[col] = X[col].fillna(fill if not pd.isna(fill) else 0)
    return X


def preprocess(df: pd.DataFrame, target_col: str,
               already_scaled: bool = False) -> tuple:
    """
    แยก X/y → encode → clean → split 80/20 → scale (ถ้ายังไม่ scale)
    already_scaled=True เมื่อ Data Transformation ทำ scaling ไปแล้ว
    """
    task_type = detect_task(df, target_col)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    X = _encode(X)
    X = _clean(X)
    X, y = _sample(X, y, MAX_ROWS_TRAIN)

    stratify = None
    if task_type == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    if not already_scaled:
        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        if num_cols:
            scaler = StandardScaler()
            X_train = X_train.copy()
            X_test  = X_test.copy()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, task_type