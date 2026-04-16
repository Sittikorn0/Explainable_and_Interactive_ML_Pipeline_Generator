"""ml_process/logic.py — Pure logic (no Streamlit/Plotly imports)"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def detect_leakage(df: pd.DataFrame, target_col: str) -> list[str]:
    """ตรวจหา column ที่อาจทำให้เกิด Data Leakage กับ target"""
    warnings = []
    y = df[target_col]

    for col in df.columns:
        if col == target_col:
            continue

        # 1. Correlation สูงมากกับ target (> 0.99)
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(y):
            try:
                if abs(df[col].corr(y)) > 0.99:
                    warnings.append(
                        f"**{col}** มี correlation กับ target สูงมาก "
                        f"(r={abs(df[col].corr(y)):.4f}) — อาจเป็น column ที่คำนวณมาจาก target โดยตรง"
                    )
            except Exception:
                pass

        # 2. จำนวน unique values เท่ากัน + correlation หลัง encode สูงมาก
        if df[col].nunique() == y.nunique() and df[col].nunique() <= 20:
            try:
                le = LabelEncoder()
                col_enc = le.fit_transform(df[col].astype(str))
                y_enc   = le.fit_transform(y.astype(str))
                if abs(pd.Series(col_enc).corr(pd.Series(y_enc))) > 0.99:
                    warnings.append(
                        f"**{col}** มี pattern เหมือน target มาก "
                        f"— อาจเป็น duplicate หรือ derived column"
                    )
            except Exception:
                pass

        # 3. ชื่อคล้าย target (substring match)
        col_lower    = col.lower().replace("_", "").replace("-", "")
        target_lower = target_col.lower().replace("_", "").replace("-", "")
        if (col_lower in target_lower or target_lower in col_lower) and col_lower != target_lower:
            warnings.append(
                f"**{col}** มีชื่อคล้าย target '{target_col}' "
                f"— ตรวจสอบว่าเป็น derived column หรือไม่"
            )

    return warnings


def compute_fi(df: pd.DataFrame, target_col: str,
               best_key: str, best_params: dict,
               trans_summary: dict) -> tuple[pd.DataFrame | None, str | None]:
    """
    คำนวณ Feature Importance สำหรับ best model
    คืน (fi_df, error_msg) — fi_df=None ถ้า model ไม่รองรับหรือเกิด error
    """
    from ml_process.preprocess import preprocess
    from ml_process.runner import get_model_map

    try:
        scaling_method      = trans_summary.get("scaling_method", "no_scaling")
        label_encoding_cols = trans_summary.get("label_encoding_cols", [])
        X_tr, _, y_tr, _, _ = preprocess(
            df, target_col,
            scaling_method=scaling_method,
            label_encoding_cols=label_encoding_cols,
        )
        m = get_model_map()[best_key]()
        if best_params:
            try:
                m.set_params(**best_params)
            except Exception:
                pass
        m.fit(X_tr, y_tr)

        importances = None
        if hasattr(m, "feature_importances_"):
            importances = m.feature_importances_
        elif hasattr(m, "coef_"):
            coef = m.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return None, None

        fi_df = (pd.DataFrame({"Feature": X_tr.columns, "Importance": importances})
                   .sort_values("Importance", ascending=False)
                   .reset_index(drop=True))
        return fi_df, None

    except Exception as e:
        return None, str(e)


